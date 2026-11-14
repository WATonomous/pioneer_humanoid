#include "can_node.hpp"
#include <algorithm> // std::copy
#include <chrono>
#include <cmath>   // M_PI
#include <cstring> // Required for std::memcpy
#include <functional>
#include <rclcpp/serialization.hpp>
#include <thread>

CanNode::CanNode() : Node("can_node"), can_core(this->get_logger()) {
  RCLCPP_INFO(this->get_logger(), "CAN Node has been initialized");

  // Load DBC file
  {
    std::ifstream idbc(ament_index_cpp::get_package_share_directory("can") +
                       "/config/humanoid.dbc");
    dbc_net = dbcppp::INetwork::LoadDBCFromIs(idbc);
    if (!dbc_net) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load DBC file");
    } else {
      RCLCPP_INFO(this->get_logger(), "DBC file loaded successfully");
      // Build the message ID to IMessage* map for quick lookup during decoding
      for (const auto& msg : dbc_net->Messages()) {
        RCLCPP_INFO(this->get_logger(), "Loaded DBC message: %s (ID 0x%lX)", msg.Name().c_str(),
                    msg.Id());
        can_messages.insert(std::make_pair(msg.Name(), &msg));
        // Subtraction due to extended can frame
        can_id_map.insert(std::make_pair(msg.Id() - 0x80000000, &msg));
      }
    }
  }

  loadMitProfiles();

  // Load parameters from config/params.yaml
  this->declare_parameter("can_interface", "can0");
  this->declare_parameter("device_path", "/dev/canable");
  this->declare_parameter("bustype", "slcan");
  this->declare_parameter("bitrate", 500000);
  this->declare_parameter("receive_poll_interval_ms", 10);
  this->declare_parameter("receive_timeout_ms", 10000);
  this->declare_parameter("mit_master_id", 0);

  // Get parameter values
  std::string can_interface = this->get_parameter("can_interface").as_string();
  std::string device_path = this->get_parameter("device_path").as_string();
  std::string bustype = this->get_parameter("bustype").as_string();
  int bitrate = this->get_parameter("bitrate").as_int();

  int receive_poll_interval_ms = this->get_parameter("receive_poll_interval_ms").as_int();
  mit_master_id_ = static_cast<int>(this->get_parameter("mit_master_id").as_int());

  RCLCPP_INFO(this->get_logger(),
              "Loaded parameters: interface=%s, bustype=%s, bitrate=%d, "
              "poll_interval_ms=%d",
              can_interface.c_str(), bustype.c_str(), bitrate, receive_poll_interval_ms);

  // Configure CanCore
  CanConfig config;
  config.interface_name = can_interface;
  config.device_path = device_path;
  config.bustype = bustype;
  config.bitrate = bitrate;
  config.receive_timeout_ms = 10000;

  // Initialize the CAN interface
  if (can_core.initialize(config)) {
    RCLCPP_INFO(this->get_logger(), "CAN Core interface initialized successfully");

    // Setup a timer to periodically call receiveCanMessages
    receive_timer_ = this->create_wall_timer(std::chrono::milliseconds(receive_poll_interval_ms),
                                             [this]() { receiveCanMessages(); });

  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize CAN Core");
  }

  // Load topic configurations and create subscribers
  createSubscribersPublishers();
}

void CanNode::createSubscribersPublishers() {
  _subscribers.clear();
  _publishers.clear();

  // Create subscribers
  _subscribers["/interfacing/motorCMD"] = this->create_subscription<common_msgs::msg::MotorCmd>(
      "/interfacing/motorCMD", rclcpp::QoS(10),
      std::bind(&CanNode::motorCMDCallback, this, std::placeholders::_1));

  // Create publishers
  _publishers["/interfacing/motorFeedback"] =
      this->create_publisher<common_msgs::msg::MotorFeedback>("/interfacing/motorFeedback", 10);
  RCLCPP_INFO(this->get_logger(), "Subscribers and publishers created successfully");
}

// Subscriber callbacks
const dbcppp::ISignal* CanNode::findSignalByName(const dbcppp::IMessage* msg,
                                                 const std::string& signal_name) {
  for (const auto& signal : msg->Signals()) {
    if (signal.Name() == signal_name) {
      return const_cast<dbcppp::ISignal*>(&signal);
    }
  }
  RCLCPP_ERROR(rclcpp::get_logger("CanNode"), "Signal '%s' not found in message '%s'",
               signal_name.c_str(), msg->Name().c_str());
  return nullptr;
}

double CanNode::decodeSignalPhysical(const dbcppp::ISignal* signal, const uint8_t* data) {
  if (!signal || !data) {
    return 0.0;
  }

  // Decode() returns a uint64 bit pattern. For signed signals dbcppp may leave the
  // value sign-extended in that uint64; assigning it (or RawToPhys without a proper
  // signed cast) yields ~2^64. Mask + sign-extend, then apply DBC scale ourselves.
  uint64_t raw = signal->Decode(data);
  const uint64_t bit_size = signal->BitSize();
  if (bit_size == 0 || bit_size > 64) {
    return 0.0;
  }

  if (bit_size < 64) {
    const uint64_t mask = (1ULL << bit_size) - 1ULL;
    raw &= mask;
    if (signal->ValueType() == dbcppp::ISignal::EValueType::Signed) {
      const uint64_t sign_bit = 1ULL << (bit_size - 1ULL);
      if (raw & sign_bit) {
        raw |= ~mask;
      }
    }
  }

  const double numeric = (signal->ValueType() == dbcppp::ISignal::EValueType::Signed)
                             ? static_cast<double>(static_cast<int64_t>(raw))
                             : static_cast<double>(raw);
  return numeric * signal->Factor() + signal->Offset();
}

void CanNode::motorCMDCallback(const common_msgs::msg::MotorCmd::SharedPtr msg) {
  RCLCPP_DEBUG(this->get_logger(), "Received MotorCMD motor=%d control_type=%d",
               static_cast<int>(msg->motor_id), static_cast<int>(msg->control_type));

  const dbcppp::IMessage* dbc_msg = nullptr;

  try {
    switch (msg->control_type) {

    case common_msgs::msg::MotorCmd::DUTY_CYCLE: {
      dbc_msg = can_messages["DutyCycleCmd"];
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      encodeSignal(findSignalByName(dbc_msg, "DutyCycle"), static_cast<double>(msg->duty_cycle),
                   can_msg);
      publishCanMessage(can_msg);
      break;
    }

    case common_msgs::msg::MotorCmd::CURRENT_LOOP: {
      dbc_msg = can_messages["CurrentLoopCmd"];
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      encodeSignal(findSignalByName(dbc_msg, "IqCurrent"), static_cast<double>(msg->current),
                   can_msg);
      publishCanMessage(can_msg);
      break;
    }

    case common_msgs::msg::MotorCmd::CURRENT_BRAKE: {
      dbc_msg = can_messages["CurrentBrakeCmd"];
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      encodeSignal(findSignalByName(dbc_msg, "BrakeCurrent"), static_cast<double>(msg->current),
                   can_msg);
      publishCanMessage(can_msg);
      break;
    }

    case common_msgs::msg::MotorCmd::VELOCITY_LOOP: {
      dbc_msg = can_messages["VelocityLoopCmd"];
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      encodeSignal(findSignalByName(dbc_msg, "VelocityERPM"), static_cast<double>(msg->velocity),
                   can_msg);
      publishCanMessage(can_msg);
      break;
    }

    case common_msgs::msg::MotorCmd::POSITION_LOOP: {
      dbc_msg = can_messages["PositionLoopCmd"];
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      encodeSignal(findSignalByName(dbc_msg, "PositionDeg"), static_cast<double>(msg->position),
                   can_msg);
      publishCanMessage(can_msg);
      break;
    }

    case common_msgs::msg::MotorCmd::SET_ORIGIN: {
      dbc_msg = can_messages["SetOriginCmd"];
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      // 0 = Temporary Origin, 1 = Permanent Origin (from DBC VAL_)
      double origin_mode = msg->temporary ? 0.0 : 1.0;
      encodeSignal(findSignalByName(dbc_msg, "OriginMode"), origin_mode, can_msg);
      publishCanMessage(can_msg);
      break;
    }

    case common_msgs::msg::MotorCmd::POSITION_VELOCITY: {
      dbc_msg = can_messages["PositionVelocityCmd"];
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      encodeSignal(findSignalByName(dbc_msg, "PosVelPosition"), static_cast<double>(msg->position),
                   can_msg);
      encodeSignal(findSignalByName(dbc_msg, "PosVelSpeed"), static_cast<double>(msg->velocity),
                   can_msg);
      encodeSignal(findSignalByName(dbc_msg, "PosVelAccel"), static_cast<double>(msg->acceleration),
                   can_msg);
      publishCanMessage(can_msg);
      break;
    }

    case common_msgs::msg::MotorCmd::MIT_CONTROL: {
      // MIT_KP/KD/Position/Velocity/Torque are DBC signals with factor=1, offset=0 -- i.e.
      // their "physical value" IS the raw fixed-point code the manual's packing formula
      // produces, not real units (rad, rad/s, Nm). msg->kp/kd/position/velocity/torque are
      // real physical units, so they must be converted via packMitValue() + this motor's
      // MitProfile BEFORE calling encodeSignal, using the int64_t (raw) overload. Passing
      // physical values straight through here (as a prior version of this code did) would
      // silently command the wrong stiffness/torque/position.
      const auto it = mit_profiles_.find(static_cast<int>(msg->motor_id));
      if (it == mit_profiles_.end()) {
        RCLCPP_ERROR(this->get_logger(),
                     "MIT_CONTROL requested for motor %d with no MIT profile loaded "
                     "(see config/mit_profiles.yaml) -- refusing to send an unscaled command.",
                     static_cast<int>(msg->motor_id));
        return;
      }
      if (!std::isfinite(msg->position) || !std::isfinite(msg->velocity) ||
          !std::isfinite(msg->kp) || !std::isfinite(msg->kd) || !std::isfinite(msg->torque)) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                              "MIT command for motor %d has a non-finite field -- not sent",
                              static_cast<int>(msg->motor_id));
        return;
      }
      const MitProfile& p = it->second;
      if (p.family == MitFamily::Ak) {
        // AK V3: extended id 0x800|id, KP-first payload.
        CanMessage ak_msg(static_cast<int>(akMitCanId(msg->motor_id)), 8);
        ak_msg.is_extended_id = true;
        const auto payload =
            packAkMitCommand(msg->position, msg->velocity, msg->kp, msg->kd, msg->torque, p);
        std::copy(payload.begin(), payload.end(), ak_msg.data.begin());
        publishCanMessage(ak_msg);
        break;
      }
      dbc_msg = can_messages["MITControlCmd"];
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      // Nearest code for gains: truncation would apply kp 0.61 as 0.488.
      encodeSignal(findSignalByName(dbc_msg, "MIT_KP"),
                   static_cast<int64_t>(packMitGain(msg->kp, p.kp_max, 12)), can_msg);
      encodeSignal(findSignalByName(dbc_msg, "MIT_KD"),
                   static_cast<int64_t>(packMitGain(msg->kd, p.kd_max, 12)), can_msg);
      encodeSignal(findSignalByName(dbc_msg, "MIT_Position"),
                   static_cast<int64_t>(packMitValue(msg->position, p.p_min, p.p_max, 16)),
                   can_msg);
      encodeSignal(findSignalByName(dbc_msg, "MIT_Velocity"),
                   static_cast<int64_t>(packMitValue(msg->velocity, p.v_min, p.v_max, 12)),
                   can_msg);
      encodeSignal(findSignalByName(dbc_msg, "MIT_Torque"),
                   static_cast<int64_t>(packMitValue(msg->torque, p.t_min, p.t_max, 12)), can_msg);
      publishCanMessage(can_msg);
      break;
    }

    // MIT special frames (FF..FF <code>): ENTER arms a GL II, EXIT frees it. Needs a profile.
    case common_msgs::msg::MotorCmd::MIT_ENTER: {
      sendMitSpecialFrame(msg->motor_id, MIT_SPECIAL_ENTER, "enter motor mode");
      break;
    }

    case common_msgs::msg::MotorCmd::MIT_EXIT: {
      sendMitSpecialFrame(msg->motor_id, MIT_SPECIAL_EXIT, "exit motor mode");
      break;
    }

    case common_msgs::msg::MotorCmd::MIT_SET_ZERO: {
      // The current shaft position becomes the drive's zero: stored calibration is now wrong.
      RCLCPP_WARN(this->get_logger(),
                  "MIT_SET_ZERO for motor %d: the drive's zero is now wherever the shaft is "
                  "standing. Re-run calibration for this joint.",
                  static_cast<int>(msg->motor_id));
      sendMitSpecialFrame(msg->motor_id, MIT_SPECIAL_SET_ZERO, "set zero");
      break;
    }

    case common_msgs::msg::MotorCmd::MIT_CLEAR_ERRORS: {
      sendMitSpecialFrame(msg->motor_id, MIT_SPECIAL_CLEAR_ERR, "clear errors");
      break;
    }

    case common_msgs::msg::MotorCmd::DISABLE: {
      dbc_msg = can_messages["MotorDisableCmd"];
      // MotorDisableCmd has no signals, just send the CAN ID
      CanMessage can_msg(getMessageId(dbc_msg, msg->motor_id), dbc_msg->MessageSize());
      publishCanMessage(can_msg);
      break;
    }

    default:
      RCLCPP_WARN(this->get_logger(), "Unknown control_type=%d, ignoring",
                  static_cast<int>(msg->control_type));
      return;
    }

  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to encode CAN message: %s", e.what());
    return;
  }

  RCLCPP_DEBUG(this->get_logger(), "CAN message sent for control_type=%d motor=%d",
               static_cast<int>(msg->control_type), static_cast<int>(msg->motor_id));
}

void CanNode::loadMitProfiles() {
  const std::string path =
      ament_index_cpp::get_package_share_directory("can") + "/config/mit_profiles.yaml";
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to load MIT profiles from %s: %s. MIT_CONTROL "
                 "commands will be refused for all motors.",
                 path.c_str(), e.what());
    return;
  }
  const YAML::Node motors = root["motors"];
  if (!motors) {
    RCLCPP_ERROR(this->get_logger(), "MIT profiles file %s has no 'motors' key", path.c_str());
    return;
  }
  for (const auto& kv : motors) {
    const int motor_id = std::stoi(kv.first.as<std::string>());
    const YAML::Node n = kv.second;
    MitProfile p;
    p.p_min = n["p_min"].as<double>();
    p.p_max = n["p_max"].as<double>();
    p.v_min = n["v_min"].as<double>();
    p.v_max = n["v_max"].as<double>();
    p.t_min = n["t_min"].as<double>();
    p.t_max = n["t_max"].as<double>();
    p.kp_min = n["kp_min"].as<double>();
    p.kp_max = n["kp_max"].as<double>();
    p.kd_min = n["kd_min"].as<double>();
    p.kd_max = n["kd_max"].as<double>();
    p.model = n["model"] ? n["model"].as<std::string>() : std::string("?");
    p.kt = n["kt"] ? n["kt"].as<double>() : 0.0;
    const std::string family = n["family"] ? n["family"].as<std::string>() : std::string("ak");
    if (!mitFamilyFromString(family, p.family)) {
      RCLCPP_ERROR(this->get_logger(),
                   "MIT profile for motor %d has unknown family '%s' (expected ak|gl2) -- "
                   "skipping this motor; MIT_CONTROL for it will be refused.",
                   motor_id, family.c_str());
      continue;
    }
    if (p.family == MitFamily::Ak && p.kt <= 0.0) {
      RCLCPP_WARN(this->get_logger(),
                  "AK motor %d has no 'kt' in mit_profiles.yaml: its feedback torque stays 0, so "
                  "joint_command's mit_max_torque can never trip for it in MIT mode.",
                  motor_id);
    }
    mit_profiles_[motor_id] = p;
    RCLCPP_INFO(this->get_logger(), "Loaded MIT profile for motor %d (%s, family %s)", motor_id,
                p.model.c_str(), family.c_str());
  }

  // Both families reply on the master id: drop an AK profile whose id shares a GL II's low
  // nibble, rather than misread its feedback.
  std::vector<int> ambiguous;
  for (const auto& [ak_id, ak] : mit_profiles_) {
    if (ak.family != MitFamily::Ak) {
      continue;
    }
    for (const auto& [gl_id, gl] : mit_profiles_) {
      if (gl.family == MitFamily::Gl2 && ak_id >= 0 && ak_id <= 0xFF &&
          (ak_id & 0xF) == (gl_id & 0xF)) {
        RCLCPP_ERROR(this->get_logger(),
                     "AK motor %d and GL II motor %d share id nibble 0x%X: their MIT feedback "
                     "on the master id is ambiguous. Dropping motor %d's MIT profile -- "
                     "renumber one drive.",
                     ak_id, gl_id, ak_id & 0xF, ak_id);
        ambiguous.push_back(ak_id);
        break;
      }
    }
  }
  for (const int id : ambiguous) {
    mit_profiles_.erase(id);
  }
}

void CanNode::encodeSignal(const dbcppp::ISignal* signal, int64_t phys_value, CanMessage& can_msg) {
  auto raw = signal->PhysToRaw(phys_value);
  signal->Encode(raw, can_msg.data.data());
}

void CanNode::encodeSignal(const dbcppp::ISignal* signal, double phys_value, CanMessage& can_msg) {
  auto raw = signal->PhysToRaw(phys_value);
  signal->Encode(raw, can_msg.data.data());
}

int32_t CanNode::getMessageId(const dbcppp::IMessage* msg, int device_id) const {
  // CAN ID format: base_id | device_id (lower 2 bits)
  return (msg->Id() & 0xFFFFFF00) | (device_id & 0xFF);
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
  CanMessage can_msg(motor_id, 8);
  const auto payload = mitSpecialFrame(code);
  std::copy(payload.begin(), payload.end(), can_msg.data.begin());
  RCLCPP_INFO(this->get_logger(), "MIT %s -> motor %d", what, motor_id);
  publishCanMessage(can_msg);
}

// MIT feedback: a standard frame on the master id, outside the DBC. Byte 0 is the full motor id
// (ak) or status << 4 | id nibble (gl2). Returns true if consumed.
bool CanNode::handleMitFeedback(const CanMessage& message) {
  if (message.is_extended_id || static_cast<int>(message.id) != mit_master_id_ ||
      message.data.size() < 8) {
    return false;
  }

  const auto ak_it = mit_profiles_.find(static_cast<int>(message.data[0]));
  if (ak_it != mit_profiles_.end() && ak_it->second.family == MitFamily::Ak) {
    publishAkFeedback(ak_it->first, decodeAkFeedback(message.data.data(), ak_it->second));
    return true;
  }

  const uint8_t id_nibble = message.data[0] & 0xF;
  const MitProfile* profile = nullptr;
  int motor_id = -1;
  for (const auto& [id, p] : mit_profiles_) {
    if (p.family == MitFamily::Gl2 && (id & 0xF) == id_nibble) {
      if (profile != nullptr) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                              "Two gl2 motors (%d and %d) share CAN id nibble 0x%X -- their MIT "
                              "feedback is indistinguishable. Renumber one drive.",
                              motor_id, id, id_nibble);
        return true;
      }
      profile = &p;
      motor_id = id;
    }
  }
  if (profile == nullptr) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "MIT feedback on master id 0x%X for unknown motor (byte 0 = 0x%02X) -- "
                         "no ak id or gl2 nibble matches (see config/mit_profiles.yaml)",
                         mit_master_id_, message.data[0]);
    return true;
  }

  const MitFeedback fb = decodeGl2Feedback(message.data.data(), *profile);
  auto feedback_msg = common_msgs::msg::MotorFeedback();
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
  publishFeedback(feedback_msg);
  return true;
}

void CanNode::publishAkFeedback(int motor_id, const MitAkFeedback& fb) {
  auto feedback_msg = common_msgs::msg::MotorFeedback();
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
  publishFeedback(feedback_msg);
}

void CanNode::publishFeedback(const common_msgs::msg::MotorFeedback& feedback_msg) {
  auto pub = std::dynamic_pointer_cast<rclcpp::Publisher<common_msgs::msg::MotorFeedback>>(
      _publishers["/interfacing/motorFeedback"]);
  if (pub) {
    pub->publish(feedback_msg);
  } else {
    RCLCPP_ERROR(this->get_logger(), "Publisher for /interfacing/motorFeedback not found");
  }
}

void CanNode::receiveCanMessages() {
  std::vector<CanMessage> messages;
  CanMessage msg;
  while (can_core.receiveMessage(msg)) {
    messages.push_back(msg);
  }

  for (const auto& message : messages) {
    // MIT feedback first: master id 0x000 collides with DutyCycleCmd's DBC base id.
    if (handleMitFeedback(message)) {
      continue;
    }

    // all messages are extended frame CAN ids
    int device_id = message.id & 0xFF;
    int base_id = message.id & 0xFFFFFF00;

    if (can_id_map.find(base_id) != can_id_map.end()) {
      // Handling each feedback message on can bus
      std::string msg_name = can_id_map[base_id]->Name();
      if (msg_name == "ServoStatusFeedback") {
        const dbcppp::IMessage* dbc_msg = can_id_map[base_id];

        // Publish to ROS topic
        auto feedback_msg = common_msgs::msg::MotorFeedback();
        feedback_msg.motor_id = device_id;
        feedback_msg.position = static_cast<float>(
            decodeSignalPhysical(findSignalByName(dbc_msg, "FbkPosition"), message.data.data()));
        feedback_msg.velocity = static_cast<float>(
            decodeSignalPhysical(findSignalByName(dbc_msg, "FbkSpeed"), message.data.data()));
        feedback_msg.current = static_cast<float>(
            decodeSignalPhysical(findSignalByName(dbc_msg, "FbkCurrent"), message.data.data()));
        // Servo frames carry current: derive torque for AK drives (read by mit_max_torque).
        feedback_msg.torque = 0.0f;
        const auto prof = mit_profiles_.find(device_id);
        if (prof != mit_profiles_.end() && prof->second.family == MitFamily::Ak &&
            prof->second.kt > 0.0) {
          feedback_msg.torque = static_cast<float>(feedback_msg.current * prof->second.kt);
        }
        feedback_msg.temperature = static_cast<int8_t>(
            decodeSignalPhysical(findSignalByName(dbc_msg, "FbkTemperature"), message.data.data()));
        feedback_msg.error_code = static_cast<int8_t>(
            decodeSignalPhysical(findSignalByName(dbc_msg, "FbkErrorCode"), message.data.data()));
        RCLCPP_DEBUG(this->get_logger(),
                     "Received feedback for motor %d: pos=%.2f vel=%.2f "
                     "current=%.2f temp=%d error=%d",
                     feedback_msg.motor_id, feedback_msg.position, feedback_msg.velocity,
                     feedback_msg.current, feedback_msg.temperature, feedback_msg.error_code);
        auto pub = std::dynamic_pointer_cast<rclcpp::Publisher<common_msgs::msg::MotorFeedback>>(
            _publishers["/interfacing/motorFeedback"]);

        if (pub) {
          pub->publish(feedback_msg);
        } else {
          RCLCPP_ERROR(this->get_logger(), "Publisher for /interfacing/motorFeedback not found");
        }
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "Received CAN message with unknown ID: 0x%X", base_id);
    }
  }
}

void CanNode::publishCanMessage(CanMessage& can_msg) {
  // Send the CAN message
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