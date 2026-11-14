#pragma once

#include "rclcpp/rclcpp.hpp"
#include <cstdint>
#include <string>
#include <vector>

struct CanMessage {
  CanMessage() {}
  CanMessage(uint32_t id, size_t size) : id(id), data(size, 0) {}

  uint32_t id = 0;           // CAN id without flag bits: <= 0x7FF standard, <= 0x1FFFFFFF extended
  std::vector<uint8_t> data; // Payload, at most 8 bytes; its size is the DLC
  bool is_extended_id = false;  // Extended frame format flag
  bool is_remote_frame = false; // Remote transmission request flag
};

struct CanConfig {
  std::string interface_name; // CAN interface name (e.g., "can0")
  std::string device_path;    // Device path for SLCAN (e.g., "/dev/canable")
  std::string bustype;        // Bus type: "socketcan" or "slcan"
  uint32_t bitrate; // Bitrate in bps (SLCAN only; a socketcan interface is set up outside)
};

// Classic CAN (no CAN-FD) over a non-blocking SocketCAN raw socket.
class CanCore {
public:
  explicit CanCore(const rclcpp::Logger& logger);
  ~CanCore();
  CanCore(const CanCore&) = delete;
  CanCore& operator=(const CanCore&) = delete;

  // Opens the interface. Throws std::runtime_error on any failure.
  void initialize(const CanConfig& config);

  // Throws std::invalid_argument for a malformed message and std::logic_error before initialize().
  // Returns false if the kernel refused the frame (e.g. ENOBUFS: tx queue full, no ACK on the bus).
  bool sendMessage(const CanMessage& message);

  // Returns false only when no frame is waiting. Throws std::runtime_error on a socket error.
  bool receiveMessage(CanMessage& message);

private:
  rclcpp::Logger logger_;
  int socket_fd_ = -1;
  CanConfig config_;

  void setupSocketCan();
  void setupSlcan();
};
