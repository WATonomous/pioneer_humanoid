#pragma once

#include "rclcpp/rclcpp.hpp"
#include <cstdint>
#include <string>
#include <vector>

struct CanMessage {
  CanMessage() {}
  CanMessage(int size) : data(size, 0), dlc(size) {}
  CanMessage(int id, int size) : id(id), data(size, 0), dlc(size) {}

  uint32_t id;                  // CAN message ID
  std::vector<uint8_t> data;    // Message data 8 bytes
  uint8_t dlc;                  // Data Length Code
  bool is_extended_id = false;  // Extended frame format flag
  bool is_remote_frame = false; // Remote transmission request flag
  uint64_t timestamp_us;        // Timestamp in microseconds
};

struct CanConfig {
  std::string interface_name;  // CAN interface name (e.g., "can0")
  std::string device_path;     // Device path for SLCAN (e.g., "/dev/canable")
  std::string bustype;         // Bus type: "socketcan" or "slcan"
  uint32_t bitrate;            // Bitrate in bps for arbitration phase
  uint32_t data_bitrate;       // Data bitrate in bps for CAN-FD data phase
  uint32_t receive_timeout_ms; // Receive timeout in milliseconds
};

class CanCore {
public:
  CanCore(const rclcpp::Logger& logger);

  // CAN Interface Management
  bool initialize(const CanConfig& config);
  bool shutdown();
  bool isInitialized() const;

  // Message Transmission
  bool sendMessage(const CanMessage& message);

  // Message Reception
  bool receiveMessage(CanMessage& message);

private:
  rclcpp::Logger logger_;

  // Internal state
  int socket_fd_;
  bool initialized_;
  bool connected_;
  CanConfig config_;

  // Internal helper methods
  bool setupSocketCan();
  bool setupSlcan();
};
