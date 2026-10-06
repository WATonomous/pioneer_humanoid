#include "joint_command_core.hpp"

#include "gravity_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
// 12-bit gain fields (every CubeMars MIT drive).
constexpr double kMitGainCodes = 4096.0;
constexpr double kGravityFfRampSec = 1.0;

// ArmPose joint order, shared by both config files.
const std::vector<std::pair<std::string, std::string>>& jointPaths() {
  static const std::vector<std::pair<std::string, std::string>> paths = {
      {"shoulder", "pitch"}, {"shoulder", "roll"}, {"shoulder", "yaw"},
      {"elbow", "pitch"},    {"elbow", "roll"},    {"wrist", "pitch"},
  };
  return paths;
}

} // namespace

std::string SeedReport::describe() const {
  std::ostringstream os;
  os << matched << " joint(s) seeded from real feedback";
  if (!unmatched.empty()) {
    os << "; " << unmatched.size() << " without feedback (left at 0)";
  }
  if (!out_of_range.empty()) {
    os << "; " << out_of_range.size() << " OUTSIDE their configured limits";
  }
  return os.str();
}

JointConfig JointCommandCore::loadJointConfig(const YAML::Node& joint_node) {
  JointConfig joint;
  joint.motor_id = static_cast<int8_t>(joint_node["can_id"].as<int>());
  joint.lower_limit = joint_node["lower_limit"].as<double>();
  joint.upper_limit = joint_node["upper_limit"].as<double>();
  joint.direction = joint_node["direction"].as<int>();
  joint.zero_offset = joint_node["zero_offset"].as<double>();
  joint.limit_range = joint_node["limit_range"].as<bool>();
  return joint;
}

bool JointCommandCore::loadFromYaml(const YAML::Node& config, const std::string& arm_side) {
  joints_.clear();
  arm_side_ = arm_side;

  if (!config[arm_side]) {
    return false;
  }

  const YAML::Node arm = config[arm_side];
  for (const auto& [group, joint_name] : jointPaths()) {
    const YAML::Node joint_node = arm[group][joint_name];
    if (!joint_node) {
      return false;
    }
    joints_.push_back(loadJointConfig(joint_node));
  }

  const bool ok = joints_.size() == 6;
  if (!ok) {
    return false;
  }

  safety_.assign(joints_.size(), JointSafetyConfig{});
  // Seed at 0 (the assumed safe starting pose an operator positions the arm at before
  // startup) and mark ready immediately, so the very first ArmPose message received is
  // ALSO velocity/delta rate-limited relative to that pose, not just position-clamped.
  // Previously this started false, letting the first command bypass all rate limiting
  // and jump straight to its target -- visible as a sudden snap before smooth tracking.
  prev_targets_.assign(joints_.size(), 0.0);
  last_motor_cmd_deg_.assign(joints_.size(), 0.0);
  blocked_.assign(joints_.size(), false);
  unpowered_.assign(joints_.size(), false);
  have_prev_targets_ = true;
  return true;
}

bool JointCommandCore::isActive(size_t joint) const {
  return joint >= safety_.size() || safety_[joint].active;
}

SeedReport
JointCommandCore::seedPrevTargetsFromFeedback(const std::map<int, double>& motor_positions) {
  SeedReport report;
  std::vector<double> seeded(joints_.size(), 0.0);
  unpowered_.assign(joints_.size(), false);
  if (last_motor_cmd_deg_.size() != joints_.size()) {
    last_motor_cmd_deg_.assign(joints_.size(), 0.0);
  }
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (!isActive(i)) {
      continue; // not part of this run: never seeded, reported or commanded
    }
    const auto it = motor_positions.find(static_cast<int>(joints_[i].motor_id));
    if (it == motor_positions.end()) {
      // Not reporting: exclude it rather than ramp it from an assumed 0.
      report.unmatched.push_back(i);
      unpowered_[i] = true;
      continue;
    }
    // Inverse of applyCalibration (motor = direction * (cmd - zero_offset)):
    //   cmd = zero_offset + motor / direction
    const double dir =
        (joints_[i].direction == 0) ? 1.0 : static_cast<double>(joints_[i].direction);
    seeded[i] = joints_[i].zero_offset + it->second / dir;
    // Start the watchdog's tracking error at zero, not at the distance from an unsent 0.
    last_motor_cmd_deg_[i] = it->second;
    ++report.matched;

    // Outside its own limits (stale calibration): the caller excludes it rather than let the
    // clamp walk it to the limit.
    if (joints_[i].limit_range && safety_[i].enable_position_clamp &&
        (seeded[i] < joints_[i].lower_limit || seeded[i] > joints_[i].upper_limit)) {
      report.out_of_range.push_back(i);
    }
  }
  prev_targets_ = std::move(seeded);
  have_prev_targets_ = true;
  gravity_ff_ramp_ = 0.0;
  return report;
}

void JointCommandCore::blockJoints(const std::vector<size_t>& indices) {
  if (blocked_.size() != joints_.size()) {
    blocked_.assign(joints_.size(), false);
  }
  for (const size_t i : indices) {
    if (i < blocked_.size()) {
      blocked_[i] = true;
    }
  }
}

bool JointCommandCore::isUnpowered(size_t joint) const {
  return (joint < unpowered_.size() && unpowered_[joint]) || !isActive(joint);
}

bool JointCommandCore::isBlocked(size_t joint) const {
  return (joint < blocked_.size() && blocked_[joint]) || isUnpowered(joint);
}

double JointCommandCore::clampAngle(double angle, const JointConfig& joint) {
  if (!joint.limit_range) {
    return angle;
  }
  return std::clamp(angle, joint.lower_limit, joint.upper_limit);
}

double JointCommandCore::applyCalibration(double angle, const JointConfig& joint) {
  return static_cast<double>(joint.direction) * (angle - joint.zero_offset);
}

double JointCommandCore::quantiseKp(double kp, double kp_max) {
  const double step = kp_max / kMitGainCodes;
  return std::round(kp / step) * step;
}

double JointCommandCore::quantiseKd(double kd, double kd_max) {
  const double step = kd_max / kMitGainCodes;
  return std::round(kd / step) * step;
}

JointSafetyConfig JointCommandCore::loadJointSafetyConfig(const YAML::Node& joint_node,
                                                          const JointSafetyConfig& base) {
  JointSafetyConfig cfg = base;
  if (!joint_node) {
    return cfg;
  }

  if (joint_node["active"]) {
    cfg.active = joint_node["active"].as<bool>();
  }
  if (joint_node["enable_position_clamp"]) {
    cfg.enable_position_clamp = joint_node["enable_position_clamp"].as<bool>();
  }
  if (joint_node["enable_velocity_limit"]) {
    cfg.enable_velocity_limit = joint_node["enable_velocity_limit"].as<bool>();
  }
  if (joint_node["enable_delta_limit"]) {
    cfg.enable_delta_limit = joint_node["enable_delta_limit"].as<bool>();
  }
  if (joint_node["enable_low_pass"]) {
    cfg.enable_low_pass = joint_node["enable_low_pass"].as<bool>();
  }
  if (joint_node["velocity_max"]) {
    cfg.velocity_max = joint_node["velocity_max"].as<double>();
  }
  if (joint_node["delta_max"]) {
    cfg.delta_max = joint_node["delta_max"].as<double>();
  }
  if (joint_node["low_pass_alpha"]) {
    cfg.low_pass_alpha = joint_node["low_pass_alpha"].as<double>();
  }
  cfg.low_pass_alpha = std::clamp(cfg.low_pass_alpha, 0.0, 1.0);
  if (joint_node["control_type"]) {
    cfg.control_type = joint_node["control_type"].as<int>();
  }
  if (joint_node["mit_kp"]) {
    cfg.mit_kp = joint_node["mit_kp"].as<double>();
  }
  if (joint_node["mit_kd"]) {
    cfg.mit_kd = joint_node["mit_kd"].as<double>();
  }
  if (joint_node["mit_max_torque"]) {
    cfg.mit_max_torque = joint_node["mit_max_torque"].as<double>();
  }
  if (joint_node["mit_max_track_err"]) {
    cfg.mit_max_track_err = joint_node["mit_max_track_err"].as<double>();
  }
  if (joint_node["mit_feedback_timeout"]) {
    cfg.mit_feedback_timeout = joint_node["mit_feedback_timeout"].as<double>();
  }
  if (joint_node["mit_family"]) {
    const std::string family = joint_node["mit_family"].as<std::string>();
    if (family == "gl2") {
      cfg.mit_family = MitDriveFamily::Gl2;
    } else if (family == "ak") {
      cfg.mit_family = MitDriveFamily::Ak;
    } else {
      throw std::runtime_error("mit_family must be 'ak' or 'gl2', got '" + family + "'");
    }
  }
  if (joint_node["mit_fault_action"]) {
    const std::string action = joint_node["mit_fault_action"].as<std::string>();
    if (action == "limp") {
      cfg.mit_fault_action = MitFaultAction::Limp;
    } else if (action == "damp") {
      cfg.mit_fault_action = MitFaultAction::Damp;
    } else {
      throw std::runtime_error("mit_fault_action must be 'limp' or 'damp', got '" + action + "'");
    }
    cfg.mit_fault_action_explicit = true;
  }
  if (!cfg.mit_fault_action_explicit) {
    cfg.mit_fault_action =
        cfg.mit_family == MitDriveFamily::Ak ? MitFaultAction::Damp : MitFaultAction::Limp;
  }
  if (joint_node["mit_fault_kd"]) {
    cfg.mit_fault_kd = joint_node["mit_fault_kd"].as<double>();
  }
  if (joint_node["gravity_ff_scale"]) {
    cfg.gravity_ff_scale = joint_node["gravity_ff_scale"].as<double>();
  }
  if (joint_node["gravity_ff_max_torque"]) {
    cfg.gravity_ff_max_torque = joint_node["gravity_ff_max_torque"].as<double>();
  }
  // The command frame IS the URDF frame (arm_calibration.yaml's zero and direction match the
  // URDF). Refuse the old cmd -> URDF keys rather than silently ignoring a -1 left in a config.
  for (const char* removed : {"urdf_direction", "urdf_offset_deg"}) {
    if (joint_node[removed]) {
      throw std::runtime_error(std::string(removed) +
                               " was removed: the command frame is the URDF frame. Fold it into "
                               "arm_calibration.yaml's direction / zero_offset / limits instead");
    }
  }
  if (joint_node["gravity_assume_deg"]) {
    cfg.gravity_assume_deg = joint_node["gravity_assume_deg"].as<double>();
  }
  return cfg;
}

bool JointCommandCore::validateMitGains() {
  std::ostringstream errors;
  for (size_t i = 0; i < joints_.size(); ++i) {
    const JointSafetyConfig& s = safety_[i];
    const std::string name = jointName(i);
    if (s.gravity_assume_deg.has_value() && !std::isfinite(*s.gravity_assume_deg)) {
      errors << "\n  " << name << ": gravity_assume_deg must be finite";
    }
    if (s.gravity_ff_scale < 0.0 || s.gravity_ff_scale > 2.0) {
      errors << "\n  " << name << ": gravity_ff_scale must be in [0, 2] (got " << s.gravity_ff_scale
             << ")";
    }
    if (s.gravity_ff_scale > 0.0) {
      if (!isMitJoint(i)) {
        errors << "\n  " << name << ": gravity_ff_scale > 0 needs control_type 0 (MIT) -- "
               << "POSITION_LOOP has no torque field";
      }
      if (arm_side_ != "left") {
        errors << "\n  " << name << ": the gravity model is the LEFT arm (joint1L..joint6l); "
               << "arm_side is '" << arm_side_ << "'";
      }
      if (s.gravity_ff_max_torque <= 0.0) {
        errors << "\n  " << name << ": gravity_ff_scale > 0 needs gravity_ff_max_torque > 0";
      }
    }
  }
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (!isMitJoint(i)) {
      continue;
    }
    const JointSafetyConfig& s = safety_[i];
    const std::string name = jointName(i);
    if (s.mit_kp <= 0.0) {
      errors << "\n  " << name << ": MIT joint needs mit_kp > 0 (got " << s.mit_kp
             << ") -- a zero-stiffness MIT joint is free, use POSITION_LOOP if that is intended";
    }
    if (s.mit_kd <= 0.0) {
      errors << "\n  " << name << ": MIT joint needs mit_kd > 0 (got " << s.mit_kd
             << ") -- the GL II manual warns kd = 0 in position control makes the motor "
                "oscillate / run away";
    }
    // Worst case is a stalled joint: the tracking fault caps PD torque at kp * mit_max_track_err.
    // Uses the quantised kp the drive applies; feed-forward adds on top.
    const double kp_q = quantiseKp(s.mit_kp);
    const double ff_max = s.gravity_ff_scale > 0.0 ? s.gravity_ff_max_torque : 0.0;
    const double worst = kp_q * s.mit_max_track_err * kDegToRad + ff_max;
    if (worst > s.mit_max_torque) {
      errors << "\n  " << name << ": mit_kp " << kp_q << " N.m/rad (quantised) x "
             << s.mit_max_track_err << " deg + gravity_ff_max_torque " << ff_max
             << " N.m = " << worst << " N.m exceeds mit_max_torque " << s.mit_max_torque
             << " N.m -- lower mit_kp, mit_max_track_err or gravity_ff_max_torque";
    }
    // Damp must damp, within the 12-bit kd field's 5.0 ceiling.
    if (s.mit_fault_action == MitFaultAction::Damp &&
        (s.mit_fault_kd <= 0.0 || s.mit_fault_kd > 5.0)) {
      errors << "\n  " << name << ": mit_fault_action is damp, so mit_fault_kd must be in "
             << "(0, 5] N.m.s/rad (got " << s.mit_fault_kd << ")";
    }
  }
  const std::string text = errors.str();
  if (!text.empty()) {
    last_error_ = "MIT gain configuration is unsafe:" + text;
    return false;
  }
  return true;
}

bool JointCommandCore::loadSafetyFromYaml(const YAML::Node& safety_cfg, double control_rate_hz) {
  last_error_.clear();
  if (joints_.empty()) {
    last_error_ = "no joints loaded (call loadFromYaml first)";
    return false;
  }
  if (control_rate_hz <= 0.0) {
    last_error_ = "control_rate_hz must be > 0";
    return false;
  }

  control_rate_hz_ = control_rate_hz;
  try {
    JointSafetyConfig defaults;
    if (safety_cfg["global"]) {
      defaults = loadJointSafetyConfig(safety_cfg["global"], defaults);
    }
    // Joints inherit mit_fault_action_explicit, so setting only mit_family: ak re-derives Damp.

    safety_.assign(joints_.size(), defaults);
    for (size_t i = 0; i < jointPaths().size(); ++i) {
      const auto& [group, joint_name] = jointPaths()[i];
      const YAML::Node joint_node = safety_cfg["joints"][group][joint_name];
      safety_[i] = loadJointSafetyConfig(joint_node, defaults);
    }
  } catch (const std::exception& e) {
    last_error_ = e.what();
    return false;
  }
  if (std::none_of(safety_.begin(), safety_.end(), [](const auto& s) { return s.active; })) {
    last_error_ = "no joint is active: set active: true on at least one joint";
    return false;
  }
  return validateMitGains();
}

double JointCommandCore::clampStep(double target, double previous, double delta_max) {
  return previous + std::clamp(target - previous, -delta_max, delta_max);
}

double JointCommandCore::applyLowPass(double target, double previous, double alpha) {
  return alpha * previous + (1.0 - alpha) * target;
}

bool JointCommandCore::isMitJoint(size_t joint) const {
  return joint < safety_.size() &&
         safety_[joint].control_type == common_msgs::msg::MotorCmd::MIT_CONTROL;
}

std::vector<int> JointCommandCore::mitMotorIds() const {
  std::vector<int> ids;
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (isMitJoint(i) && isActive(i)) {
      ids.push_back(static_cast<int>(joints_[i].motor_id));
    }
  }
  return ids;
}

int8_t JointCommandCore::motorId(size_t joint) const {
  return joints_.at(joint).motor_id;
}

std::string JointCommandCore::jointName(size_t joint) const {
  if (joint >= jointPaths().size()) {
    return "joint" + std::to_string(joint);
  }
  return jointPaths()[joint].first + "." + jointPaths()[joint].second;
}

common_msgs::msg::MotorCmd JointCommandCore::mitSafeCommand(size_t joint) const {
  common_msgs::msg::MotorCmd cmd;
  cmd.motor_id = joints_[joint].motor_id;
  cmd.control_type = common_msgs::msg::MotorCmd::MIT_CONTROL;
  // kp = t_ff = 0, so the position field pulls nowhere. Limp: kd = 0. Damp: kd = mit_fault_kd.
  cmd.position = 0.0f;
  cmd.velocity = 0.0f;
  cmd.torque = 0.0f;
  cmd.kp = 0.0f;
  cmd.kd = safety_[joint].mit_fault_action == MitFaultAction::Damp
               ? static_cast<float>(safety_[joint].mit_fault_kd)
               : 0.0f;
  return cmd;
}

std::vector<double>
JointCommandCore::gravityTorqueMotor(const std::vector<double>& cmd_targets_deg) const {
  std::vector<double> out(joints_.size(), 0.0);
  // One unknown angle makes every load unknown. Unpowered joints may use gravity_assume_deg;
  // out-of-range ones never do.
  std::array<double, 6> q_urdf{};
  for (size_t i = 0; i < q_urdf.size(); ++i) {
    const JointSafetyConfig& s = safety_[i];
    double q_cmd = cmd_targets_deg[i];
    if (isUnpowered(i) && s.gravity_assume_deg.has_value() &&
        !(i < blocked_.size() && blocked_[i])) {
      q_cmd = *s.gravity_assume_deg;
    } else if (isBlocked(i)) {
      return out;
    }
    q_urdf[i] = q_cmd * kDegToRad; // the command frame is the URDF frame
  }
  const std::array<double, 6> tau = leftArmGravityHoldTorque(q_urdf);
  for (size_t i = 0; i < q_urdf.size(); ++i) {
    // Torque follows the angle's sign mapping: urdf (= cmd) -> motor.
    const double dir = joints_[i].direction == 0 ? 1.0 : joints_[i].direction;
    out[i] = dir * tau[i];
  }
  return out;
}

std::vector<common_msgs::msg::MotorCmd> JointCommandCore::mitSafeCommands(bool damped_only) const {
  std::vector<common_msgs::msg::MotorCmd> cmds;
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (!isMitJoint(i) || !isActive(i) ||
        (damped_only && safety_[i].mit_fault_action != MitFaultAction::Damp)) {
      continue;
    }
    cmds.push_back(mitSafeCommand(i));
  }
  return cmds;
}

std::vector<common_msgs::msg::MotorCmd> JointCommandCore::mitModeCommands(int8_t control_type,
                                                                          bool limp_only) const {
  std::vector<common_msgs::msg::MotorCmd> cmds;
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (!isMitJoint(i) || !isActive(i) ||
        (limp_only && safety_[i].mit_fault_action != MitFaultAction::Limp)) {
      continue;
    }
    common_msgs::msg::MotorCmd cmd;
    cmd.motor_id = joints_[i].motor_id;
    cmd.control_type = control_type;
    cmds.push_back(cmd);
  }
  return cmds;
}

bool JointCommandCore::hasDampedMitJoints() const {
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (isMitJoint(i) && isActive(i) && safety_[i].mit_fault_action == MitFaultAction::Damp) {
      return true;
    }
  }
  return false;
}

std::optional<std::string>
JointCommandCore::checkMitFaults(const std::map<int, MotorFeedbackSample>& feedback) const {
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (!isMitJoint(i) || isBlocked(i)) {
      continue;
    }
    const JointSafetyConfig& s = safety_[i];
    const int id = static_cast<int>(joints_[i].motor_id);
    const std::string name = jointName(i) + " (motor " + std::to_string(id) + ")";

    const auto it = feedback.find(id);
    if (it == feedback.end()) {
      return name + ": no MIT feedback received at all";
    }
    const MotorFeedbackSample& fb = it->second;
    if (fb.age_s > s.mit_feedback_timeout) {
      std::ostringstream os;
      os << name << ": MIT feedback is " << fb.age_s << " s old (limit " << s.mit_feedback_timeout
         << " s)";
      return os.str();
    }
    // GL II: 0 Disable / 1 Enable / else fault. AK: 0 ok, 1 = over-temperature.
    const bool status_ok =
        s.mit_family == MitDriveFamily::Ak ? fb.status == 0 : (fb.status == 0 || fb.status == 1);
    if (!status_ok) {
      return name + ": drive reports fault status " + std::to_string(fb.status) +
             (s.mit_family == MitDriveFamily::Ak ? " (AK error code)" : " (GL II status)");
    }
    if (std::abs(fb.torque_nm) > s.mit_max_torque) {
      std::ostringstream os;
      os << name << ": torque " << fb.torque_nm << " N.m exceeds mit_max_torque "
         << s.mit_max_torque << " N.m";
      return os.str();
    }
    const double err = std::abs(fb.position_deg - last_motor_cmd_deg_[i]);
    if (err > s.mit_max_track_err) {
      std::ostringstream os;
      os << name << ": tracking error " << err << " deg exceeds mit_max_track_err "
         << s.mit_max_track_err << " deg (commanded " << last_motor_cmd_deg_[i] << ", at "
         << fb.position_deg << ")";
      return os.str();
    }
  }
  return std::nullopt;
}

std::vector<common_msgs::msg::MotorCmd>
JointCommandCore::armPoseToMotorCmds(const common_msgs::msg::ArmPose& pose,
                                     int8_t default_control_type) {
  if (joints_.size() != 6) {
    throw std::runtime_error("JointCommandCore is not configured for 6 joints");
  }

  if (pose.shoulder.position.size() < 3 || pose.elbow.position.size() < 2 ||
      pose.wrist.position.size() < 1) {
    throw std::runtime_error("ArmPose must contain 3 shoulder, 2 elbow, and 1 wrist positions");
  }

  const std::vector<double> source_angles = {
      pose.shoulder.position[0], pose.shoulder.position[1], pose.shoulder.position[2],
      pose.elbow.position[0],    pose.elbow.position[1],    pose.wrist.position[0],
  };

  std::vector<common_msgs::msg::MotorCmd> commands;
  commands.reserve(joints_.size());
  std::vector<std::pair<size_t, size_t>> mit_cmds; // (index in commands, joint)
  std::vector<double> next_targets = prev_targets_;

  if (safety_.size() != joints_.size()) {
    safety_.assign(joints_.size(), JointSafetyConfig{});
  }
  if (prev_targets_.size() != joints_.size()) {
    prev_targets_.assign(joints_.size(), 0.0);
    next_targets = prev_targets_;
    have_prev_targets_ = true;
  }
  if (last_motor_cmd_deg_.size() != joints_.size()) {
    last_motor_cmd_deg_.assign(joints_.size(), 0.0);
  }

  for (size_t i = 0; i < joints_.size(); ++i) {
    const JointSafetyConfig& safety = safety_[i];

    if (!isActive(i)) {
      continue; // not part of this run: no command at all
    }
    if (isBlocked(i)) {
      // MIT joints get their fault action; servo joints get no command (drive holds).
      if (isMitJoint(i)) {
        commands.push_back(mitSafeCommand(i));
      }
      continue;
    }

    double target = source_angles[i];
    if (safety.enable_position_clamp) {
      target = clampAngle(target, joints_[i]);
    }

    if (have_prev_targets_) {
      // Smooth before rate-limiting (the reverse capped speed at (1 - alpha) * velocity_max).
      if (safety.enable_low_pass) {
        target = applyLowPass(target, prev_targets_[i], safety.low_pass_alpha);
      }
      if (safety.enable_velocity_limit && control_rate_hz_ > 0.0) {
        const double velocity_step = std::abs(safety.velocity_max) / control_rate_hz_;
        target = clampStep(target, prev_targets_[i], velocity_step);
      }
      if (safety.enable_delta_limit) {
        target = clampStep(target, prev_targets_[i], std::abs(safety.delta_max));
      }
    }

    if (safety.enable_position_clamp) {
      // If the clamp bites (joint outside its limits), rate-limit the move back into range
      // instead of snapping to the limit in one tick.
      const double clamped = clampAngle(target, joints_[i]);
      if (clamped != target && have_prev_targets_) {
        double max_step = std::numeric_limits<double>::infinity();
        if (safety.enable_velocity_limit && control_rate_hz_ > 0.0) {
          max_step = std::abs(safety.velocity_max) / control_rate_hz_;
        }
        if (safety.enable_delta_limit) {
          max_step = std::min(max_step, std::abs(safety.delta_max));
        }
        target = std::isinf(max_step) ? clamped : clampStep(clamped, prev_targets_[i], max_step);
      } else {
        target = clamped;
      }
    }
    next_targets[i] = target;

    const double calibrated_deg = applyCalibration(target, joints_[i]);
    last_motor_cmd_deg_[i] = calibrated_deg;

    const int8_t control_type =
        safety.control_type >= 0 ? static_cast<int8_t>(safety.control_type) : default_control_type;

    common_msgs::msg::MotorCmd cmd;
    cmd.motor_id = joints_[i].motor_id;
    cmd.control_type = control_type;
    if (control_type == common_msgs::msg::MotorCmd::MIT_CONTROL) {
      // can_node's MIT path expects position in RADIANS (CubeMars manual MIT protocol),
      // unlike POSITION_LOOP's PositionDeg which is degrees -- see can_node.cpp packMitValue.
      // velocity feed-forward left at 0; torque gets gravity feed-forward below.
      cmd.position = static_cast<float>(calibrated_deg * kDegToRad);
      cmd.velocity = 0.0f;
      cmd.torque = 0.0f;
      cmd.kp = static_cast<float>(safety.mit_kp);
      cmd.kd = static_cast<float>(safety.mit_kd);
      mit_cmds.emplace_back(commands.size(), i);
    } else {
      cmd.position = static_cast<float>(calibrated_deg);
    }
    commands.push_back(cmd);
  }

  // After the loop: each load depends on every joint. Uses commanded targets, not feedback.
  last_gravity_torque_motor_ = gravityTorqueMotor(next_targets);
  gravity_ff_ramp_ = std::min(1.0, gravity_ff_ramp_ + 1.0 / (control_rate_hz_ * kGravityFfRampSec));
  for (const auto& [k, i] : mit_cmds) {
    const JointSafetyConfig& s = safety_[i];
    const double ff = s.gravity_ff_scale * gravity_ff_ramp_ * last_gravity_torque_motor_[i];
    commands[k].torque =
        static_cast<float>(std::clamp(ff, -s.gravity_ff_max_torque, s.gravity_ff_max_torque));
  }

  prev_targets_ = std::move(next_targets);
  have_prev_targets_ = true;
  return commands;
}
