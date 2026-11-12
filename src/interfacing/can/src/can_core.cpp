#include "can_core.hpp"
#include <cerrno>
#include <cstdio>  // std::snprintf
#include <cstdlib> // std::system, to run the slcand setup script
#include <cstring>
#include <linux/can.h>     // can_frame, CAN_* flags and masks
#include <linux/can/raw.h> // CAN_RAW
#include <net/if.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/socket.h> // socket(), bind()
#include <sys/stat.h>   // stat(), to check the setup script exists
#include <sys/wait.h>   // WIFEXITED / WEXITSTATUS for std::system's status
#include <unistd.h>

#include "ament_index_cpp/get_package_share_directory.hpp" // to find the setup script

namespace {

std::runtime_error errnoError(const std::string& what) {
  return std::runtime_error(what + ": " + std::strerror(errno));
}

std::string hex(uint32_t value) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%X", value);
  return buf;
}

} // namespace

CanCore::CanCore(const rclcpp::Logger& logger) : logger_(logger) {}

CanCore::~CanCore() {
  if (socket_fd_ >= 0) {
    close(socket_fd_);
  }
}

void CanCore::initialize(const CanConfig& config) {
  if (socket_fd_ >= 0) {
    throw std::logic_error("CanCore::initialize called twice");
  }
  RCLCPP_INFO(logger_, "Initializing CAN interface: %s", config.interface_name.c_str());
  config_ = config;

  if (config.bustype == "socketcan") {
    setupSocketCan();
  } else if (config.bustype == "slcan") {
    setupSlcan();
  } else {
    throw std::runtime_error("Unsupported CAN bus type '" + config.bustype +
                             "' (expected 'socketcan' or 'slcan')");
  }
}

bool CanCore::sendMessage(const CanMessage& message) {
  if (socket_fd_ < 0) {
    throw std::logic_error("CanCore::sendMessage called before initialize()");
  }
  const uint32_t id_mask = message.is_extended_id ? CAN_EFF_MASK : CAN_SFF_MASK;
  if ((message.id & ~id_mask) != 0) {
    throw std::invalid_argument("CAN id " + hex(message.id) + " does not fit a " +
                                (message.is_extended_id ? "29-bit extended" : "11-bit standard") +
                                " frame");
  }
  if (message.data.size() > CAN_MAX_DLEN) {
    throw std::invalid_argument("CAN payload of " + std::to_string(message.data.size()) +
                                " bytes exceeds the classic CAN maximum of 8");
  }

  struct can_frame frame;
  std::memset(&frame, 0, sizeof(frame));
  frame.can_id = message.id;
  if (message.is_extended_id) {
    frame.can_id |= CAN_EFF_FLAG;
  }
  if (message.is_remote_frame) {
    frame.can_id |= CAN_RTR_FLAG;
  }
  frame.can_dlc = static_cast<__u8>(message.data.size());
  if (!message.is_remote_frame) {
    std::memcpy(frame.data, message.data.data(), frame.can_dlc);
  }

  const ssize_t bytes_written = write(socket_fd_, &frame, sizeof(frame));
  if (bytes_written < 0) {
    RCLCPP_ERROR(logger_, "Failed to write CAN frame 0x%X: %s", message.id, std::strerror(errno));
    return false;
  }
  if (static_cast<size_t>(bytes_written) != sizeof(frame)) {
    // A raw CAN socket writes a frame whole or not at all.
    throw std::runtime_error("Short write on CAN socket: " + std::to_string(bytes_written) +
                             " of " + std::to_string(sizeof(frame)) + " bytes");
  }
  return true;
}

bool CanCore::receiveMessage(CanMessage& message) {
  if (socket_fd_ < 0) {
    throw std::logic_error("CanCore::receiveMessage called before initialize()");
  }

  struct can_frame frame;
  const ssize_t bytes_read = read(socket_fd_, &frame, sizeof(frame));
  if (bytes_read < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false; // no frame waiting (non-blocking socket)
    }
    throw errnoError("Failed to read from CAN interface " + config_.interface_name);
  }
  if (static_cast<size_t>(bytes_read) != sizeof(frame)) {
    // A raw CAN socket without CAN_RAW_FD_FRAMES only ever returns whole classic frames.
    throw std::runtime_error("Unexpected CAN read of " + std::to_string(bytes_read) +
                             " bytes (expected " + std::to_string(sizeof(frame)) + ")");
  }
  if (frame.can_dlc > CAN_MAX_DLEN) {
    throw std::runtime_error("Received CAN frame with invalid DLC " +
                             std::to_string(frame.can_dlc));
  }

  message.is_extended_id = (frame.can_id & CAN_EFF_FLAG) != 0;
  message.is_remote_frame = (frame.can_id & CAN_RTR_FLAG) != 0;
  message.id = frame.can_id & (message.is_extended_id ? CAN_EFF_MASK : CAN_SFF_MASK);
  if (message.is_remote_frame) {
    message.data.clear(); // an RTR frame carries no payload
  } else {
    message.data.assign(frame.data, frame.data + frame.can_dlc);
  }

  RCLCPP_DEBUG(logger_, "CAN frame received: ID=0x%X, Extended=%d, RTR=%d", message.id,
               message.is_extended_id, message.is_remote_frame);
  return true;
}

void CanCore::setupSocketCan() {
  RCLCPP_INFO(logger_, "Setting up SocketCAN interface: %s", config_.interface_name.c_str());

  if (config_.interface_name.empty() || config_.interface_name.size() >= IFNAMSIZ) {
    throw std::runtime_error("Invalid CAN interface name '" + config_.interface_name + "'");
  }

  socket_fd_ = socket(PF_CAN, SOCK_RAW | SOCK_NONBLOCK, CAN_RAW);
  if (socket_fd_ < 0) {
    throw errnoError("Failed to create SocketCAN socket");
  }
  // On any failure below the destructor closes socket_fd_.

  struct ifreq ifr;
  std::memset(&ifr, 0, sizeof(ifr));
  std::strncpy(ifr.ifr_name, config_.interface_name.c_str(), IFNAMSIZ - 1);
  if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
    throw errnoError("Failed to get interface index for " + config_.interface_name);
  }

  struct sockaddr_can addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;
  if (bind(socket_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    throw errnoError("Failed to bind SocketCAN socket to " + config_.interface_name);
  }

  RCLCPP_INFO(logger_, "SocketCAN interface %s ready (classic CAN, non-blocking).",
              config_.interface_name.c_str());
}

void CanCore::setupSlcan() {
  RCLCPP_INFO(logger_, "Setting up SLCAN interface '%s' on %s at %u bps via setup_can.sh",
              config_.interface_name.c_str(), config_.device_path.c_str(), config_.bitrate);

  // Throws PackageNotFoundError if the can package is not installed.
  const std::string script_path =
      ament_index_cpp::get_package_share_directory("can") + "/scripts/setup_can.sh";
  struct stat st;
  if (stat(script_path.c_str(), &st) != 0) {
    throw errnoError("CAN setup script " + script_path + " not found");
  }

  // slcan bitrate codes (slcand -sN).
  std::string bitrate_code;
  switch (config_.bitrate) {
  case 10000:
    bitrate_code = "-s0";
    break;
  case 20000:
    bitrate_code = "-s1";
    break;
  case 50000:
    bitrate_code = "-s2";
    break;
  case 100000:
    bitrate_code = "-s3";
    break;
  case 125000:
    bitrate_code = "-s4";
    break;
  case 250000:
    bitrate_code = "-s5";
    break;
  case 500000:
    bitrate_code = "-s6";
    break;
  case 750000:
    bitrate_code = "-s7";
    break;
  case 1000000:
    bitrate_code = "-s8";
    break;
  default:
    throw std::runtime_error("Unsupported SLCAN bitrate " + std::to_string(config_.bitrate));
  }

  const std::string command = "sudo " + script_path + " " + config_.device_path + " " +
                              config_.interface_name + " " + bitrate_code;
  RCLCPP_INFO(logger_, "Executing CAN setup script: %s", command.c_str());
  const int status = std::system(command.c_str());
  if (status == -1) {
    throw errnoError("Failed to run CAN setup script");
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("CAN setup script failed (" +
                             (WIFEXITED(status) ? "exit code " + std::to_string(WEXITSTATUS(status))
                                                : std::string("killed by a signal")) +
                             "): " + command);
  }

  // setup_can.sh has created interface_name; open it like any SocketCAN interface.
  setupSocketCan();
}
