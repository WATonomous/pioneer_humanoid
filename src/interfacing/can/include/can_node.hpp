#pragma once

#include "can_core.hpp"
#include "mit_protocol.hpp"

#include <dbcppp/Network.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "common_msgs/msg/motor_cmd.hpp"
#include "common_msgs/msg/motor_feedback.hpp"

class CanNode : public rclcpp::Node {
public:
  CanNode();

private:
  CanCore can_core;

  // A DBC message and the signals this node reads or writes, resolved once at startup so a
  // missing message or signal stops the node instead of failing per frame.
  struct DbcFrame {
    const dbcppp::IMessage* msg = nullptr;
    std::vector<const dbcppp::ISignal*> signals; // in the order the code passes values
  };

  std::unique_ptr<dbcppp::INetwork> dbc_net;
  DbcFrame duty_cycle_cmd_, current_loop_cmd_, current_brake_cmd_, velocity_loop_cmd_,
      position_loop_cmd_, set_origin_cmd_, position_velocity_cmd_, mit_control_cmd_, disable_cmd_;
  // FbkPosition, FbkSpeed, FbkCurrent, FbkTemperature, FbkErrorCode.
  DbcFrame servo_feedback_;
  // Extended-frame base ids (id & 0xFFFFFF00) of every DBC message, so a known but unhandled
  // frame is not reported as unknown.
  std::unordered_set<uint32_t> known_extended_base_ids_;

  void loadDbc();
  DbcFrame resolveDbc(const std::string& message, const std::vector<std::string>& signals) const;
  static double decodeSignalPhysical(const dbcppp::ISignal& signal, const uint8_t* data);
  CanMessage dbcCanMessage(const dbcppp::IMessage* msg, int device_id) const;
  // Encodes values into frame's signals and sends it. Refuses (logs, sends nothing) a value that is
  // not finite or outside the signal's DBC [min|max]: encoded, it would overflow and wrap.
  void sendDbcCommand(const DbcFrame& frame, int motor_id, const std::vector<double>& values);
  void publishCanMessage(const CanMessage& can_msg);

  void receiveCanMessages();
  void handleExtendedFrame(const CanMessage& message);
  void handleServoFeedback(const CanMessage& message);

  // MIT (Force Control) support: per-motor scaling constants + the manual's float_to_uint
  // packing formula (mit_protocol.hpp). See config/mit_profiles.yaml and can/README.md / MIT
  // protocol section of the CubeMars AK-series manual.
  std::unordered_map<int, MitProfile> mit_profiles_;
  // GL II motors by the low nibble of their id: their feedback carries only that nibble.
  std::unordered_map<uint8_t, int> gl2_by_nibble_;
  void loadMitProfiles();

  // GL II reply id ("master id", default 0x000).
  int mit_master_id_{0};
  bool handleMitFeedback(const CanMessage& message);
  void publishAkFeedback(int motor_id, const MitAkFeedback& fb);
  void sendMitSpecialFrame(int motor_id, uint8_t code, const char* what);

  void motorCMDCallback(const common_msgs::msg::MotorCmd::SharedPtr msg);

  rclcpp::Subscription<common_msgs::msg::MotorCmd>::SharedPtr motor_cmd_sub_;
  rclcpp::Publisher<common_msgs::msg::MotorFeedback>::SharedPtr feedback_pub_;
  rclcpp::TimerBase::SharedPtr receive_timer_; // polls the CAN socket
};
