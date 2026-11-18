#include "can_node.hpp"

#include <algorithm> // std::copy
#include <chrono>
#include <cmath> // M_PI, std::isfinite
#include <fstream>
#include <stdexcept>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

namespace {

// DBC ids carry bit 31 for an extended frame.
constexpr uint64_t kDbcExtendedFlag = 0x80000000;
// Servo-mode ids: command/feedback type in bits 28..8, drive id in bits 7..0.
constexpr uint32_t kServoBaseMask = 0xFFFFFF00;

} // namespace

CanNode::CanNode() : Node("can_node"), can_core(this->get_logger()) {
  // Startup fails loudly: a node with a broken DBC or MIT profile must not accept commands.
  loadDbc();
  loadMitProfiles();

  // No defaults: run with config/params.yaml (can.launch.py loads it), or set each one.
  const std::string can_interface = this->declare_parameter<std::string>("can_interface");
  const std::string device_path = this->declare_parameter<std::string>("device_path");
  const std::string bustype = this->declare_parameter<std::string>("bustype");
  const int64_t bitrate = this->declare_parameter<int64_t>("bitrate");
  const int64_t receive_poll_interval_ms =
      this->declare_parameter<int64_t>("receive_poll_interval_ms");
  const int64_t mit_master_id = this->declare_parameter<int64_t>("mit_master_id");
  if (bitrate <= 0) {
    throw std::runtime_error("bitrate must be > 0 (got " + std::to_string(bitrate) + ")");
  }
  if (receive_poll_interval_ms <= 0) {
    throw std::runtime_error("receive_poll_interval_ms must be > 0 (got " +
                             std::to_string(receive_poll_interval_ms) + ")");
  }
  if (mit_master_id < 0 || mit_master_id > 0x7FF) {
    throw std::runtime_error("mit_master_id must be a standard CAN id 0..0x7FF (got " +
                             std::to_string(mit_master_id) + ")");
  }
  mit_master_id_ = static_cast<int>(mit_master_id);

  RCLCPP_INFO(this->get_logger(),
              "Loaded parameters: interface=%s, bustype=%s, bitrate=%ld, poll_interval_ms=%ld",
              can_interface.c_str(), bustype.c_str(), static_cast<long>(bitrate),
              static_cast<long>(receive_poll_interval_ms));

  CanConfig config;
  config.interface_name = can_interface;
  config.device_path = device_path;
  config.bustype = bustype;
  config.bitrate = static_cast<uint32_t>(bitrate);
  // Throws on failure: a CAN node without a bus must not keep running and accept commands.
  can_core.initialize(config);
  RCLCPP_INFO(this->get_logger(), "CAN Core interface initialized successfully");

  // Depth: joint_command publishes every arm's commands in one burst per tick (and MIT_EXIT 3x
  // per joint on a fault); a shallow queue drops some before they are read.
  motor_cmd_sub_ = this->create_subscription<common_msgs::msg::MotorCmd>(
      "/interfacing/motorCMD", rclcpp::QoS(100),
      [this](const common_msgs::msg::MotorCmd::SharedPtr msg) { motorCMDCallback(msg); });
  feedback_pub_ =
      this->create_publisher<common_msgs::msg::MotorFeedback>("/interfacing/motorFeedback", 10);
  receive_timer_ = this->create_wall_timer(std::chrono::milliseconds(receive_poll_interval_ms),
                                           [this]() { receiveCanMessages(); });
}

void CanNode::loadDbc() {
  const std::string path =
      ament_index_cpp::get_package_share_directory("can") + "/config/humanoid.dbc";
  std::ifstream idbc(path);
  if (!idbc) {
    throw std::runtime_error("Cannot open DBC file " + path);
  }
  dbc_net = dbcppp::INetwork::LoadDBCFromIs(idbc);
  if (!dbc_net) {
    throw std::runtime_error("Failed to parse DBC file " + path);
  }
  for (const auto& msg : dbc_net->Messages()) {
    RCLCPP_INFO(this->get_logger(), "Loaded DBC message: %s (ID 0x%lX)", msg.Name().c_str(),
                msg.Id());
    if (msg.Id() & kDbcExtendedFlag) {
      known_extended_base_ids_.insert(static_cast<uint32_t>(msg.Id() & ~kDbcExtendedFlag) &
                                      kServoBaseMask);
    }
  }

  duty_cycle_cmd_ = resolveDbc("DutyCycleCmd", {"DutyCycle"});
  current_loop_cmd_ = resolveDbc("CurrentLoopCmd", {"IqCurrent"});
  current_brake_cmd_ = resolveDbc("CurrentBrakeCmd", {"BrakeCurrent"});
  velocity_loop_cmd_ = resolveDbc("VelocityLoopCmd", {"VelocityERPM"});
  position_loop_cmd_ = resolveDbc("PositionLoopCmd", {"PositionDeg"});
  set_origin_cmd_ = resolveDbc("SetOriginCmd", {"OriginMode"});
  position_velocity_cmd_ =
      resolveDbc("PositionVelocityCmd", {"PosVelPosition", "PosVelSpeed", "PosVelAccel"});
  mit_control_cmd_ = resolveDbc("MITControlCmd",
                                {"MIT_KP", "MIT_KD", "MIT_Position", "MIT_Velocity", "MIT_Torque"});
  disable_cmd_ = resolveDbc("MotorDisableCmd", {});
  servo_feedback_ = resolveDbc("ServoStatusFeedback", {"FbkPosition", "FbkSpeed", "FbkCurrent",
                                                       "FbkTemperature", "FbkErrorCode"});
  if (!(servo_feedback_.msg->Id() & kDbcExtendedFlag)) {
    throw std::runtime_error("DBC ServoStatusFeedback must be an extended frame");
  }
}

CanNode::DbcFrame CanNode::resolveDbc(const std::string& message,
                                      const std::vector<std::string>& signals) const {
  DbcFrame frame;
  for (const auto& msg : dbc_net->Messages()) {
    if (msg.Name() == message) {
      frame.msg = &msg;
      break;
    }
  }
  if (frame.msg == nullptr) {
    throw std::runtime_error("DBC has no message '" + message + "'");
  }
  for (const auto& name : signals) {
    const dbcppp::ISignal* found = nullptr;
    for (const auto& signal : frame.msg->Signals()) {
      if (signal.Name() == name) {
        found = &signal;
        break;
      }
    }
    if (found == nullptr) {
      throw std::runtime_error("DBC message '" + message + "' has no signal '" + name + "'");
    }
    if (found->BitSize() == 0 || found->BitSize() > 64) {
      throw std::runtime_error("DBC signal '" + name + "' has bit size " +
                               std::to_string(found->BitSize()));
    }
    frame.signals.push_back(found);
  }
  return frame;
}

double CanNode::decodeSignalPhysical(const dbcppp::ISignal& signal, const uint8_t* data) {
  // Decode() returns a uint64 bit pattern. For signed signals dbcppp may leave the
  // value sign-extended in that uint64; assigning it (or RawToPhys without a proper
  // signed cast) yields ~2^64. Mask + sign-extend, then apply DBC scale ourselves.
  uint64_t raw = signal.Decode(data);
  const uint64_t bit_size = signal.BitSize(); // 1..64, checked in resolveDbc
  const bool is_signed = signal.ValueType() == dbcppp::ISignal::EValueType::Signed;
  if (bit_size < 64) {
    const uint64_t mask = (1ULL << bit_size) - 1ULL;
    raw &= mask;
    if (is_signed && (raw & (1ULL << (bit_size - 1ULL)))) {
      raw |= ~mask;
    }
  }
  const double numeric =
      is_signed ? static_cast<double>(static_cast<int64_t>(raw)) : static_cast<double>(raw);
  return numeric * signal.Factor() + signal.Offset();
}

CanMessage CanNode::dbcCanMessage(const dbcppp::IMessage* msg, int device_id) const {
  // DBC ids carry bit 31 for an extended frame; CanMessage takes the bare id plus a flag.
  CanMessage can_msg(
      static_cast<uint32_t>((msg->Id() & ~kDbcExtendedFlag & kServoBaseMask) | (device_id & 0xFF)),
      msg->MessageSize());
  can_msg.is_extended_id = (msg->Id() & kDbcExtendedFlag) != 0;
  return can_msg;
}

void CanNode::sendDbcCommand(const DbcFrame& frame, int motor_id,
                             const std::vector<double>& values) {
  if (values.size() != frame.signals.size()) {
    throw std::logic_error("sendDbcCommand: " + std::to_string(values.size()) + " values for " +
                           std::to_string(frame.signals.size()) + " signals of " +
                           frame.msg->Name());
  }
  for (size_t i = 0; i < values.size(); ++i) {
    const dbcppp::ISignal& s = *frame.signals[i];
    if (!(values[i] >= s.Minimum() && values[i] <= s.Maximum())) { // NaN fails too
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "%s for motor %d: %s = %f is non-finite or outside its DBC range "
                            "[%g, %g] -- not sent",
                            frame.msg->Name().c_str(), motor_id, s.Name().c_str(), values[i],
                            s.Minimum(), s.Maximum());
      return;
    }
  }
  CanMessage can_msg = dbcCanMessage(frame.msg, motor_id);
  for (size_t i = 0; i < values.size(); ++i) {
    frame.signals[i]->Encode(frame.signals[i]->PhysToRaw(values[i]), can_msg.data.data());
  }
  publishCanMessage(can_msg);
}

void CanNode::motorCMDCallback(const common_msgs::msg::MotorCmd::SharedPtr msg) {
  const int motor_id = static_cast<int>(msg->motor_id);
  RCLCPP_DEBUG(this->get_logger(), "Received MotorCMD motor=%d control_type=%d", motor_id,
               static_cast<int>(msg->control_type));

  switch (msg->control_type) {
  case common_msgs::msg::MotorCmd::DUTY_CYCLE:
    sendDbcCommand(duty_cycle_cmd_, motor_id, {static_cast<double>(msg->duty_cycle)});
    break;

  case common_msgs::msg::MotorCmd::CURRENT_LOOP:
    sendDbcCommand(current_loop_cmd_, motor_id, {static_cast<double>(msg->current)});
    break;

  case common_msgs::msg::MotorCmd::CURRENT_BRAKE:
    sendDbcCommand(current_brake_cmd_, motor_id, {static_cast<double>(msg->current)});
    break;

  case common_msgs::msg::MotorCmd::VELOCITY_LOOP:
    sendDbcCommand(velocity_loop_cmd_, motor_id, {static_cast<double>(msg->velocity)});
    break;

  case common_msgs::msg::MotorCmd::POSITION_LOOP:
    sendDbcCommand(position_loop_cmd_, motor_id, {static_cast<double>(msg->position)});
    break;

  case common_msgs::msg::MotorCmd::SET_ORIGIN:
    // 0 = Temporary Origin, 1 = Permanent Origin (from DBC VAL_)
    sendDbcCommand(set_origin_cmd_, motor_id, {msg->temporary ? 0.0 : 1.0});
    break;

  case common_msgs::msg::MotorCmd::POSITION_VELOCITY:
    sendDbcCommand(position_velocity_cmd_, motor_id,
                   {static_cast<double>(msg->position), static_cast<double>(msg->velocity),
                    static_cast<double>(msg->acceleration)});
    break;

  case common_msgs::msg::MotorCmd::MIT_CONTROL: {
    // MIT_KP/KD/Position/Velocity/Torque are DBC signals with factor=1, offset=0 -- i.e.
    // their "physical value" IS the raw fixed-point code the manual's packing formula
    // produces, not real units (rad, rad/s, Nm). msg->kp/kd/position/velocity/torque are
    // real physical units, so they must be converted via packMitValue() + this motor's
    // MitProfile BEFORE encoding. Passing physical values straight through would
    // silently command the wrong stiffness/torque/position.
    const auto it = mit_profiles_.find(motor_id);
    if (it == mit_profiles_.end()) {
      RCLCPP_ERROR(this->get_logger(),
                   "MIT_CONTROL requested for motor %d with no MIT profile loaded "
                   "(see config/mit_profiles.yaml) -- refusing to send an unscaled command.",
                   motor_id);
      return;
    }
    const MitProfile& p = it->second;
    if (const char* field =
            mitCommandOutOfRange(msg->position, msg->velocity, msg->kp, msg->kd, msg->torque, p)) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "MIT command for motor %d (%s): %s is non-finite or outside the "
                            "drive's range (p=%f v=%f kp=%f kd=%f t=%f) -- not sent",
                            motor_id, p.model.c_str(), field, msg->position, msg->velocity, msg->kp,
                            msg->kd, msg->torque);
      return;
    }
    if (p.family == MitFamily::Ak) {
      // AK V3: extended id 0x800|id, KP-first payload.
      CanMessage ak_msg(akMitCanId(motor_id), 8);
      ak_msg.is_extended_id = true;
      const auto payload =
          packAkMitCommand(msg->position, msg->velocity, msg->kp, msg->kd, msg->torque, p);
      std::copy(payload.begin(), payload.end(), ak_msg.data.begin());
      publishCanMessage(ak_msg);
      break;
    }
    // GL II: standard frame on the motor id, laid out by the DBC. Gains use the nearest code:
    // truncation would apply kp 0.61 as 0.488.
    sendDbcCommand(mit_control_cmd_, motor_id,
                   {static_cast<double>(packMitGain(msg->kp, p.kp_max, 12)),
                    static_cast<double>(packMitGain(msg->kd, p.kd_max, 12)),
                    static_cast<double>(packMitValue(msg->position, p.p_min, p.p_max, 16)),
                    static_cast<double>(packMitValue(msg->velocity, p.v_min, p.v_max, 12)),
                    static_cast<double>(packMitValue(msg->torque, p.t_min, p.t_max, 12))});
    break;
  }

  // MIT special frames (FF..FF <code>): ENTER arms a GL II, EXIT frees it. Needs a profile.
  case common_msgs::msg::MotorCmd::MIT_ENTER:
    sendMitSpecialFrame(motor_id, MIT_SPECIAL_ENTER, "enter motor mode");
    break;

  case common_msgs::msg::MotorCmd::MIT_EXIT:
    sendMitSpecialFrame(motor_id, MIT_SPECIAL_EXIT, "exit motor mode");
    break;

  case common_msgs::msg::MotorCmd::MIT_SET_ZERO:
    // The current shaft position becomes the drive's zero: stored calibration is now wrong.
    RCLCPP_WARN(this->get_logger(),
                "MIT_SET_ZERO for motor %d: the drive's zero is now wherever the shaft is "
                "standing. Re-run calibration for this joint.",
                motor_id);
    sendMitSpecialFrame(motor_id, MIT_SPECIAL_SET_ZERO, "set zero");
    break;

  case common_msgs::msg::MotorCmd::MIT_CLEAR_ERRORS:
    sendMitSpecialFrame(motor_id, MIT_SPECIAL_CLEAR_ERR, "clear errors");
    break;

  case common_msgs::msg::MotorCmd::DISABLE:
    // MotorDisableCmd has no signals, just the CAN id.
    sendDbcCommand(disable_cmd_, motor_id, {});
    break;

  default:
    RCLCPP_ERROR(this->get_logger(), "Unknown control_type=%d for motor %d -- not sent",
                 static_cast<int>(msg->control_type), motor_id);
    return;
  }
}

void CanNode::loadMitProfiles() {
  const std::string path =
      ament_index_cpp::get_package_share_directory("can") + "/config/mit_profiles.yaml";
  const YAML::Node root = YAML::LoadFile(path); // throws if missing or malformed
  const YAML::Node motors = root["motors"];
  if (!motors || !motors.IsMap()) {
    throw std::runtime_error("MIT profiles file " + path + " has no 'motors' map");
  }
  for (const auto& kv : motors) {
    const int motor_id = std::stoi(kv.first.as<std::string>());
    const YAML::Node n = kv.second;
    const std::string where = path + " motor " + std::to_string(motor_id);
    const auto need = [&](const char* key) {
      if (!n[key]) {
        throw std::runtime_error(where + ": missing '" + key + "'");
      }
      return n[key];
    };
    MitProfile p;
    p.p_min = need("p_min").as<double>();
    p.p_max = need("p_max").as<double>();
    p.v_min = need("v_min").as<double>();
    p.v_max = need("v_max").as<double>();
    p.t_min = need("t_min").as<double>();
    p.t_max = need("t_max").as<double>();
    p.kp_min = need("kp_min").as<double>();
    p.kp_max = need("kp_max").as<double>();
    p.kd_min = need("kd_min").as<double>();
    p.kd_max = need("kd_max").as<double>();
    p.model = need("model").as<std::string>();
    const std::string family = need("family").as<std::string>();
    if (!mitFamilyFromString(family, p.family)) {
      throw std::runtime_error(where + ": unknown family '" + family + "' (expected ak|gl2)");
    }
    if (!(p.p_min < p.p_max && p.v_min < p.v_max && p.t_min < p.t_max)) {
      throw std::runtime_error(where + ": every *_min must be below its *_max");
    }
    // packMitGain encodes gains over [0, max].
    if (p.kp_min != 0.0 || p.kd_min != 0.0 || !(p.kp_max > 0.0) || !(p.kd_max > 0.0)) {
      throw std::runtime_error(where + ": kp/kd ranges must be [0, max > 0]");
    }
    if (p.family == MitFamily::Ak) {
      // Without kt the servo feedback torque stays 0 and joint_command's mit_max_torque can never
      // trip for this joint.
      p.kt = need("kt").as<double>();
      if (!(p.kt > 0.0)) {
        throw std::runtime_error(where + ": AK motor needs kt > 0");
      }
      if (motor_id < 0 || motor_id > 0xFF) {
        throw std::runtime_error(where + ": AK drive id must fit 8 bits");
      }
    } else if (motor_id < 0 || motor_id > 0x7FF) {
      throw std::runtime_error(where + ": GL II id must be a standard CAN id");
    }
    mit_profiles_[motor_id] = p;
    RCLCPP_INFO(this->get_logger(), "Loaded MIT profile for motor %d (%s, family %s)", motor_id,
                p.model.c_str(), family.c_str());
  }

  // GL II feedback carries only the id's low nibble, and AK feedback shares the master id: two
  // drives on one nibble make feedback ambiguous.
  for (const auto& [id, p] : mit_profiles_) {
    if (p.family != MitFamily::Gl2) {
      continue;
    }
    const uint8_t nibble = static_cast<uint8_t>(id & 0xF);
    const auto [it, inserted] = gl2_by_nibble_.emplace(nibble, id);
    if (!inserted) {
      throw std::runtime_error("GL II motors " + std::to_string(it->second) + " and " +
                               std::to_string(id) + " share CAN id nibble " +
                               std::to_string(nibble) +
                               ": their MIT feedback is indistinguishable. Renumber one drive.");
    }
  }
  for (const auto& [id, p] : mit_profiles_) {
    if (p.family == MitFamily::Ak && gl2_by_nibble_.count(static_cast<uint8_t>(id & 0xF))) {
      throw std::runtime_error("AK motor " + std::to_string(id) + " and GL II motor " +
                               std::to_string(gl2_by_nibble_.at(static_cast<uint8_t>(id & 0xF))) +
                               " share CAN id nibble " + std::to_string(id & 0xF) +
                               ": their MIT feedback on the master id is ambiguous. Renumber one "
                               "drive.");
    }
  }
}

void CanNode::sendMitSpecialFrame(int motor_id, uint8_t code, const char* what) {
  const auto it = mit_profiles_.find(motor_id);
  if (it == mit_profiles_.end()) {
    RCLCPP_ERROR(this->get_logger(),
                 "MIT special frame (%s) requested for motor %d with no MIT profile loaded "
                 "(see config/mit_profiles.yaml) -- refusing.",
                 what, motor_id);
    return;
  }
  if (it->second.family == MitFamily::Ak) {
    // AK V3 has no special frames.
    if (code == MIT_SPECIAL_SET_ZERO) {
      RCLCPP_ERROR(this->get_logger(),
                   "MIT set zero refused for AK motor %d: V3 firmware has no MIT zero frame. "
                   "Use SET_ORIGIN (servo mode 5) with the joint in its known zero pose.",
                   motor_id);
    } else {
      RCLCPP_INFO_ONCE(this->get_logger(),
                       "MIT %s not sent to AK motor %d: V3 firmware accepts MIT frames directly "
                       "(logged once)",
                       what, motor_id);
    }
    return;
  }
  // Standard frame on the node id.
  CanMessage can_msg(static_cast<uint32_t>(motor_id), 8);
  const auto payload = mitSpecialFrame(code);
  std::copy(payload.begin(), payload.end(), can_msg.data.begin());
  RCLCPP_INFO(this->get_logger(), "MIT %s -> motor %d", what, motor_id);
  publishCanMessage(can_msg);
}

// MIT feedback: a standard frame on the master id, outside the DBC. Byte 0 is the full motor id
// (ak) or status << 4 | id nibble (gl2). Returns true if the frame was on the master id.
bool CanNode::handleMitFeedback(const CanMessage& message) {
  if (static_cast<int>(message.id) != mit_master_id_) {
    return false;
  }
  if (message.data.size() != 8) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "MIT feedback on master id 0x%X has %zu bytes (expected 8) -- dropped",
                          mit_master_id_, message.data.size());
    return true;
  }

  const auto ak_it = mit_profiles_.find(static_cast<int>(message.data[0]));
  if (ak_it != mit_profiles_.end() && ak_it->second.family == MitFamily::Ak) {
    publishAkFeedback(ak_it->first, decodeAkFeedback(message.data.data(), ak_it->second));
    return true;
  }

  const auto gl_it = gl2_by_nibble_.find(static_cast<uint8_t>(message.data[0] & 0xF));
  if (gl_it == gl2_by_nibble_.end()) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "MIT feedback on master id 0x%X for unknown motor (byte 0 = 0x%02X) -- "
                          "no ak id or gl2 nibble matches (see config/mit_profiles.yaml)",
                          mit_master_id_, message.data[0]);
    return true;
  }
  const int motor_id = gl_it->second;
  const MitFeedback fb = decodeGl2Feedback(message.data.data(), mit_profiles_.at(motor_id));

  common_msgs::msg::MotorFeedback feedback_msg;
  feedback_msg.motor_id = static_cast<int8_t>(motor_id);
  // Degrees, like servo feedback, so every consumer shares one unit.
  feedback_msg.position = static_cast<float>(fb.position * 180.0 / M_PI);
  feedback_msg.velocity = static_cast<float>(fb.velocity);
  feedback_msg.current = 0.0f; // MIT feedback reports torque, not current
  feedback_msg.torque = static_cast<float>(fb.torque);
  feedback_msg.temperature = static_cast<int8_t>(fb.drive_temp);
  feedback_msg.error_code = static_cast<int8_t>(fb.status);

  if (!mitStatusIsOk(fb.status, MitFamily::Gl2)) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "Motor %d reports MIT status 0x%X (%s)", motor_id, fb.status,
                          mitStatusName(fb.status));
  }
  RCLCPP_DEBUG(this->get_logger(), "MIT feedback motor %d: pos=%.3f deg tau=%.3f Nm %dC (%s)",
               motor_id, feedback_msg.position, fb.torque, fb.drive_temp, mitStatusName(fb.status));
  feedback_pub_->publish(feedback_msg);
  return true;
}

void CanNode::publishAkFeedback(int motor_id, const MitAkFeedback& fb) {
  common_msgs::msg::MotorFeedback feedback_msg;
  feedback_msg.motor_id = static_cast<int8_t>(motor_id);
  feedback_msg.position = static_cast<float>(fb.position * 180.0 / M_PI); // degrees, as above
  feedback_msg.velocity = static_cast<float>(fb.velocity);                // rad/s
  feedback_msg.current = 0.0f;
  feedback_msg.torque = static_cast<float>(fb.torque);
  feedback_msg.temperature = static_cast<int8_t>(fb.motor_temp);
  feedback_msg.error_code = static_cast<int8_t>(fb.error);

  if (!mitStatusIsOk(fb.error, MitFamily::Ak)) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "Motor %d reports AK MIT error %d (%s)", motor_id, fb.error,
                          mitAkErrorName(fb.error));
  }
  RCLCPP_DEBUG(this->get_logger(), "AK MIT feedback motor %d: pos=%.3f deg tau=%.3f Nm %dC (%s)",
               motor_id, feedback_msg.position, fb.torque, fb.motor_temp, mitAkErrorName(fb.error));
  feedback_pub_->publish(feedback_msg);
}

void CanNode::receiveCanMessages() {
  CanMessage message;
  while (can_core.receiveMessage(message)) {
    if (message.is_extended_id) {
      handleExtendedFrame(message);
    } else if (!handleMitFeedback(message)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Received standard CAN frame with unknown ID 0x%X", message.id);
    }
  }
}

void CanNode::handleExtendedFrame(const CanMessage& message) {
  const uint32_t base_id = message.id & kServoBaseMask;
  const uint32_t servo_feedback_base =
      static_cast<uint32_t>(servo_feedback_.msg->Id() & ~kDbcExtendedFlag) & kServoBaseMask;
  if (base_id == servo_feedback_base) {
    handleServoFeedback(message);
  } else if (!known_extended_base_ids_.count(base_id)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Received extended CAN frame with unknown ID 0x%X", message.id);
  }
}

void CanNode::handleServoFeedback(const CanMessage& message) {
  const int device_id = static_cast<int>(message.id & 0xFF);
  if (message.data.size() != servo_feedback_.msg->MessageSize()) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "Servo feedback from motor %d has %zu bytes (expected %lu) -- dropped",
                          device_id, message.data.size(),
                          static_cast<unsigned long>(servo_feedback_.msg->MessageSize()));
    return;
  }
  const uint8_t* data = message.data.data();
  const auto& sig = servo_feedback_.signals; // see loadDbc for the order

  common_msgs::msg::MotorFeedback feedback_msg;
  feedback_msg.motor_id = static_cast<int8_t>(device_id);
  feedback_msg.position = static_cast<float>(decodeSignalPhysical(*sig[0], data));
  feedback_msg.velocity = static_cast<float>(decodeSignalPhysical(*sig[1], data));
  feedback_msg.current = static_cast<float>(decodeSignalPhysical(*sig[2], data));
  // Servo frames carry current: derive torque for AK drives (read by mit_max_torque). kt is
  // required for every AK profile (loadMitProfiles).
  feedback_msg.torque = 0.0f;
  const auto prof = mit_profiles_.find(device_id);
  if (prof != mit_profiles_.end() && prof->second.family == MitFamily::Ak) {
    feedback_msg.torque = static_cast<float>(feedback_msg.current * prof->second.kt);
  }
  feedback_msg.temperature = static_cast<int8_t>(decodeSignalPhysical(*sig[3], data));
  feedback_msg.error_code = static_cast<int8_t>(decodeSignalPhysical(*sig[4], data));
  RCLCPP_DEBUG(this->get_logger(),
               "Received feedback for motor %d: pos=%.2f vel=%.2f current=%.2f temp=%d error=%d",
               feedback_msg.motor_id, feedback_msg.position, feedback_msg.velocity,
               feedback_msg.current, feedback_msg.temperature, feedback_msg.error_code);
  feedback_pub_->publish(feedback_msg);
}

void CanNode::publishCanMessage(const CanMessage& can_msg) {
  if (can_core.sendMessage(can_msg)) {
    RCLCPP_DEBUG(this->get_logger(), "Sent CAN message: ID=0x%X", can_msg.id);
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to send CAN message: ID=0x%X", can_msg.id);
  }
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CanNode>());
  rclcpp::shutdown();
  return 0;
}
