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
constexpr double kMaxGravityFfScale = 5.0;

// Runaway watch: a drive whose encoder calibration is wrong pushes its joint AWAY from the target,
// harder the further it goes, and its damping pushes the wrong way too. A healthy loop never
// accelerates a joint away from its target: a lagging joint moves toward it, and an overshoot
// slows down. The left knee's runaway (AKH70-48, 2026-11-18, fixed by recalibrating the encoder in
// the CubeMars upper computer) moved away at 5, 25, 100 deg/s on successive samples.
constexpr double kRunawaySpeedDps = 20.0; // away speed that counts
constexpr double kRunawayMinErrDeg = 1.0; // ignore jitter around the target
constexpr int kRunawaySamples = 2;        // consecutive samples, each faster than the last
constexpr double kRunawayMaxGapSec = 0.2; // older previous sample: restart, no velocity from it

// ArmPose joint order, shared by both config files.
const std::vector<std::pair<std::string, std::string>>& armJointPaths() {
  static const std::vector<std::pair<std::string, std::string>> paths = {
      {"shoulder", "pitch"}, {"shoulder", "roll"}, {"shoulder", "yaw"},
      {"elbow", "pitch"},    {"elbow", "yaw"},     {"wrist", "pitch"},
  };
  return paths;
}

// LegPose joint order.
const std::vector<std::pair<std::string, std::string>>& legJointPaths() {
  static const std::vector<std::pair<std::string, std::string>> paths = {
      {"hip", "pitch"},  {"hip", "roll"},    {"hip", "yaw"},
      {"knee", "pitch"}, {"ankle", "pitch"}, {"ankle", "roll"},
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

JointConfig JointCommandCore::loadJointConfig(const YAML::Node& joint_node,
                                              bool allow_uncalibrated) {
  JointConfig joint;
  joint.motor_id = static_cast<int8_t>(joint_node["can_id"].as<int>());
  joint.lower_limit = joint_node["lower_limit"].as<double>();
  joint.upper_limit = joint_node["upper_limit"].as<double>();
  joint.limit_range = joint_node["limit_range"].as<bool>();
  // An explicit null (not a missing key) marks a joint not calibrated yet. Its direction and
  // zero_offset keep their defaults; loadSafetyFromYaml refuses the joint if it is active.
  if (allow_uncalibrated &&
      (joint_node["direction"].IsNull() || joint_node["zero_offset"].IsNull())) {
    joint.calibrated = false;
    return joint;
  }
  joint.direction = joint_node["direction"].as<int>();
  joint.zero_offset = joint_node["zero_offset"].as<double>();
  return joint;
}

bool JointCommandCore::loadFromYaml(const YAML::Node& config, const std::string& side,
                                    bool is_leg) {
  is_leg_ = is_leg;
  side_ = side;
  joints_.clear();
  gripper_mapping_.reset();

  if (!config[side]) {
    return false;
  }

  if (is_leg_) {
    const YAML::Node leg = config[side];
    for (const auto& [group, joint_name] : legJointPaths()) {
      const YAML::Node joint_node = leg[group][joint_name];
      if (!joint_node) {
        return false;
      }
      joints_.push_back(loadJointConfig(joint_node, /*allow_uncalibrated=*/true));
    }
  } else {
    const YAML::Node arm = config[side];
    for (const auto& [group, joint_name] : armJointPaths()) {
      const YAML::Node joint_node = arm[group][joint_name];
      if (!joint_node) {
        return false;
      }
      joints_.push_back(loadJointConfig(joint_node));
    }
    if (arm["gripper"] && arm["gripper"]["open_close"]) {
      gripper_mapping_ = loadJointConfig(arm["gripper"]["open_close"]);
    }
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
  runaway_track_.assign(joints_.size(), RunawayTrack{});
  runaway_.assign(joints_.size(), false);
  have_prev_targets_ = true;
  return true;
}

bool JointCommandCore::isActive(size_t joint) const {
  return joint >= safety_.size() || safety_[joint].active;
}

bool JointCommandCore::anyActive() const {
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (isActive(i)) {
      return true;
    }
  }
  return false;
}

std::optional<std::string>
findDuplicateMotorIds(const std::vector<const JointCommandCore*>& limbs) {
  std::map<int, std::string> owner; // motor id -> "side joint"
  for (const JointCommandCore* limb : limbs) {
    for (size_t i = 0; i < limb->jointCount(); ++i) {
      const int id = static_cast<int>(limb->motorId(i));
      const std::string name = limb->side() + " " + limb->jointName(i);
      const auto [it, inserted] = owner.emplace(id, name);
      if (!inserted) {
        return "motor " + std::to_string(id) + " is " + it->second + " and " + name;
      }
    }
  }
  return std::nullopt;
}

void JointCommandCore::resizeJointState() {
  prev_targets_.resize(joints_.size(), 0.0);
  last_motor_cmd_deg_.resize(joints_.size(), 0.0);
  blocked_.resize(joints_.size(), false);
  unpowered_.resize(joints_.size(), false);
  runaway_track_.resize(joints_.size(), RunawayTrack{});
  runaway_.resize(joints_.size(), false);
}

SeedReport
JointCommandCore::seedPrevTargetsFromFeedback(const std::map<int, double>& motor_positions) {
  SeedReport report;
  std::vector<double> seeded(joints_.size(), 0.0);
  unpowered_.assign(joints_.size(), false);
  if (last_motor_cmd_deg_.size() != joints_.size()) {
    last_motor_cmd_deg_.assign(joints_.size(), 0.0);
  }
  // New targets: runaway history from before the seed means nothing.
  runaway_track_.assign(joints_.size(), RunawayTrack{});
  runaway_.resize(joints_.size(), false);
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
    runaway_track_[i].armed = true;
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
  if (joint_node["mit_peak_torque"]) {
    cfg.mit_peak_torque = joint_node["mit_peak_torque"].as<double>();
  }
  if (joint_node["mit_max_temperature"]) {
    cfg.mit_max_temperature = joint_node["mit_max_temperature"].as<double>();
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
    // Up to 5x the model: the torque itself is still capped by gravity_ff_max_torque and the
    // stall rule below (kp x track_err + gravity_ff_max_torque <= cap).
    if (s.gravity_ff_scale < 0.0 || s.gravity_ff_scale > kMaxGravityFfScale) {
      errors << "\n  " << name << ": gravity_ff_scale must be in [0, " << kMaxGravityFfScale
             << "] (got " << s.gravity_ff_scale << ")";
    }
    if (s.gravity_ff_scale > 0.0) {
      if (!isMitJoint(i)) {
        errors << "\n  " << name << ": gravity_ff_scale > 0 needs control_type 0 (MIT) -- "
               << "POSITION_LOOP has no torque field";
      }
      if (!hasGravityModel()) {
        errors << "\n  " << name << ": gravity_model.hpp has only the left and right arms; "
               << "arm_side is '" << side_ << "'";
      }
      if (s.gravity_ff_max_torque <= 0.0) {
        errors << "\n  " << name << ": gravity_ff_scale > 0 needs gravity_ff_max_torque > 0";
      }
    }
  }
  for (size_t i = 0; i < joints_.size(); ++i) {
    // An inactive leg joint is never entered, commanded or watched; it is checked when activated
    // (active is read at startup, so that is a restart through this check).
    if (!isMitJoint(i) || (is_leg_ && !isActive(i))) {
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
    // Above rated is allowed only while the temperature is watched.
    if (s.mit_peak_torque.has_value()) {
      if (!(*s.mit_peak_torque >= s.mit_max_torque)) {
        errors << "\n  " << name << ": mit_peak_torque " << *s.mit_peak_torque
               << " N.m must be >= mit_max_torque " << s.mit_max_torque << " N.m";
      }
      if (!s.mit_max_temperature.has_value()) {
        errors << "\n  " << name << ": mit_peak_torque needs mit_max_temperature -- torque above "
               << "mit_max_torque is only allowed while the temperature is below it";
      }
    }
    if (s.mit_max_temperature.has_value() &&
        !(*s.mit_max_temperature > 0.0 && *s.mit_max_temperature <= 127.0)) {
      errors << "\n  " << name << ": mit_max_temperature must be in (0, 127] deg C (got "
             << *s.mit_max_temperature << "; MotorFeedback.temperature is int8)";
    }
    // Worst case is a stalled joint: the tracking fault caps PD torque at kp * mit_max_track_err.
    // Uses the quantised kp the drive applies; feed-forward adds on top. Bounded by the peak cap
    // when one is set: a stalled joint above rated is then stopped only by the temperature check.
    const double kp_q = quantiseKp(s.mit_kp);
    const double ff_max = s.gravity_ff_scale > 0.0 ? s.gravity_ff_max_torque : 0.0;
    const double worst = kp_q * s.mit_max_track_err * kDegToRad + ff_max;
    const double cap = s.mit_peak_torque.value_or(s.mit_max_torque);
    const char* cap_name = s.mit_peak_torque.has_value() ? "mit_peak_torque" : "mit_max_torque";
    if (worst > cap) {
      errors << "\n  " << name << ": mit_kp " << kp_q << " N.m/rad (quantised) x "
             << s.mit_max_track_err << " deg + gravity_ff_max_torque " << ff_max
             << " N.m = " << worst << " N.m exceeds " << cap_name << " " << cap
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

bool JointCommandCore::validateGripper() {
  if (!hasGripper()) {
    return true;
  }
  const JointConfig& g = joints_[kGripperJoint];
  const JointSafetyConfig& s = safety_[kGripperJoint];
  std::ostringstream errors;
  if (!isMitJoint(kGripperJoint)) {
    errors
        << "\n  gripper: needs control_type 0 (MIT) -- POSITION_LOOP drives it at full stiffness";
  }
  if (!g.limit_range || g.lower_limit > 0.0 || g.upper_limit <= 0.0) {
    errors << "\n  gripper: limits must be [<= 0, > 0]: open is its calibrated zero, closed is "
           << "upper_limit (got [" << g.lower_limit << ", " << g.upper_limit
           << "]). Calibrate it open, with direction so that closing is positive";
  }
  // A grasp leaves the gripper short of its target: that must not trip the tracking fault (which
  // halts the whole arm). With the startup rule this also caps the squeeze at mit_max_torque.
  if (s.mit_max_track_err < g.upper_limit - g.lower_limit) {
    errors << "\n  gripper: mit_max_track_err " << s.mit_max_track_err
           << " deg must cover its full travel (" << g.upper_limit - g.lower_limit
           << " deg) so a grasp never faults";
  }
  if (s.gravity_ff_scale != 0.0) {
    errors << "\n  gripper: has no gravity model, gravity_ff_scale must be 0";
  }
  // The startup rule would then cap the squeeze at the peak instead of mit_max_torque.
  if (s.mit_peak_torque.has_value()) {
    errors << "\n  gripper: mit_peak_torque would raise the squeeze limit; set mit_max_torque";
  }
  const std::string text = errors.str();
  if (!text.empty()) {
    last_error_ = "Gripper configuration is unsafe:" + text;
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
  if (!safety_cfg[side_]) {
    last_error_ = "safety config has no '" + side_ + "' block for this limb";
    return false;
  }
  if (is_leg_) {
    joints_.resize(legJointPaths().size());
  } else {
    joints_.resize(armJointPaths().size());
    const YAML::Node arm = safety_cfg[side_];
    // The gripper is driven only if this config has a block for it.
    // One level at a time: indexing through a missing key throws in yaml-cpp.
    const bool drives_gripper = arm["gripper"] && arm["gripper"]["open_close"];
    if (drives_gripper) {
      if (!gripper_mapping_) {
        last_error_ = "safety config has a gripper block but arm_calibration.yaml has no "
                      "gripper.open_close for this arm";
        return false;
      }
      joints_.push_back(*gripper_mapping_);
    }
  }
  resizeJointState();
  try {
    JointSafetyConfig defaults;
    if (safety_cfg["global"]) {
      defaults = loadJointSafetyConfig(safety_cfg["global"], defaults);
    }
    // Joints inherit mit_fault_action_explicit, so setting only mit_family: ak re-derives Damp.

    safety_.assign(joints_.size(), defaults);
    if (is_leg_) {
      const YAML::Node leg = safety_cfg[side_];
      for (size_t i = 0; i < legJointPaths().size(); ++i) {
        const auto& [group, joint_name] = legJointPaths()[i];
        safety_[i] = loadJointSafetyConfig(leg[group][joint_name], defaults);
      }
    } else {
      const YAML::Node arm = safety_cfg[side_];
      for (size_t i = 0; i < armJointPaths().size(); ++i) {
        const auto& [group, joint_name] = armJointPaths()[i];
        safety_[i] = loadJointSafetyConfig(arm[group][joint_name], defaults);
      }
      if (hasGripper()) {
        safety_[kGripperJoint] = loadJointSafetyConfig(arm["gripper"]["open_close"], defaults);
      }
    }
  } catch (const std::exception& e) {
    last_error_ = e.what();
    return false;
  }
  if (is_leg_) {
    std::ostringstream errors;
    for (size_t i = 0; i < joints_.size(); ++i) {
      if (isActive(i) && !joints_[i].calibrated) {
        errors << "\n  " << jointName(i) << " (motor " << static_cast<int>(joints_[i].motor_id)
               << "): direction / zero_offset are null in leg_calibration.yaml -- calibrate it "
               << "(calibrate_actuators.py) or set active: false";
      }
    }
    if (!errors.str().empty()) {
      last_error_ = "Active leg joint is not calibrated:" + errors.str();
      return false;
    }
  }
  // An arm with no active joint still loads (and is validated): the node skips it this run.
  return validateMitGains() && validateGripper();
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
  if (is_leg_) {
    if (joint < legJointPaths().size()) {
      return legJointPaths()[joint].first + "." + legJointPaths()[joint].second;
    }
  } else {
    if (joint == kGripperJoint) {
      return "gripper.open_close";
    }
    if (joint < armJointPaths().size()) {
      return armJointPaths()[joint].first + "." + armJointPaths()[joint].second;
    }
  }
  return "joint" + std::to_string(joint);
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
  // A runaway joint's drive pushes the wrong way, so its damping would too: always limp.
  cmd.kd = safety_[joint].mit_fault_action == MitFaultAction::Damp && !isRunaway(joint)
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
  const std::array<double, 6> tau =
      side_ == "right" ? rightArmGravityHoldTorque(q_urdf) : leftArmGravityHoldTorque(q_urdf);
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
    // Runaway joints are included too: an AK drive keeps running its last frame, and AK V3
    // ignores MIT exit, so only an explicit limp frame stops it.
    if (!isMitJoint(i) || !isActive(i) ||
        (damped_only && safety_[i].mit_fault_action != MitFaultAction::Damp && !isRunaway(i))) {
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
    // Torque: hard cap at the peak; above rated only while a real reading is below the maximum.
    const double torque = std::abs(fb.torque_nm);
    if (torque > s.mit_peak_torque.value_or(s.mit_max_torque)) {
      std::ostringstream os;
      os << name << ": torque " << fb.torque_nm << " N.m exceeds "
         << (s.mit_peak_torque.has_value() ? "mit_peak_torque " : "mit_max_torque ")
         << s.mit_peak_torque.value_or(s.mit_max_torque) << " N.m (at " << fb.temperature_c
         << " C)";
      return os.str();
    }
    if (s.mit_max_temperature.has_value() && fb.temperature_c >= *s.mit_max_temperature) {
      std::ostringstream os;
      os << name << ": temperature " << fb.temperature_c << " C reached mit_max_temperature "
         << *s.mit_max_temperature << " C (torque " << fb.torque_nm << " N.m)";
      return os.str();
    }
    if (torque > s.mit_max_torque &&
        !(s.mit_max_temperature.has_value() && fb.temperature_c > 0.0)) {
      std::ostringstream os;
      os << name << ": torque " << fb.torque_nm << " N.m exceeds mit_max_torque "
         << s.mit_max_torque << " N.m (rated) with no temperature reading to allow it";
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

void JointCommandCore::recordFeedback(int motor_id, double position_deg, double t_s) {
  for (size_t i = 0; i < joints_.size() && i < runaway_track_.size(); ++i) {
    if (static_cast<int>(joints_[i].motor_id) != motor_id) {
      continue;
    }
    RunawayTrack& tr = runaway_track_[i];
    if (!tr.armed || !isMitJoint(i) || !isActive(i) || isBlocked(i)) {
      return;
    }
    if (tr.have_sample && t_s <= tr.t_s) {
      return; // not a new sample
    }
    const double err = last_motor_cmd_deg_[i] - position_deg; // > 0: target is above
    if (!tr.have_sample || t_s - tr.t_s > kRunawayMaxGapSec) {
      tr = RunawayTrack{true, true, position_deg, t_s, 0.0, err, 0};
      return;
    }
    const double vel = (position_deg - tr.position_deg) / (t_s - tr.t_s);
    const double away = err > 0.0 ? -vel : (err < 0.0 ? vel : 0.0);
    const bool counts =
        std::abs(err) >= kRunawayMinErrDeg && away >= kRunawaySpeedDps && away > tr.away_dps;
    tr.count = counts ? tr.count + 1 : 0;
    tr.position_deg = position_deg;
    tr.t_s = t_s;
    tr.away_dps = away;
    tr.err_deg = err;
    return;
  }
}

std::optional<std::string> JointCommandCore::checkMitRunaway() {
  for (size_t i = 0; i < joints_.size() && i < runaway_track_.size(); ++i) {
    if (isRunaway(i) || runaway_track_[i].count < kRunawaySamples) {
      continue;
    }
    runaway_[i] = true;
    const RunawayTrack& tr = runaway_track_[i];
    std::ostringstream os;
    os << jointName(i) << " (motor " << static_cast<int>(joints_[i].motor_id)
       << "): RUNAWAY: moving away from its target at " << tr.away_dps
       << " deg/s and speeding up (error " << std::abs(tr.err_deg) << " deg, kp "
       << safety_[i].mit_kp << "). The drive is pushing the joint away from where it is told to "
       << "go (positive feedback). This joint is now held LIMP (kp 0, kd 0): damping on such a "
       << "drive also pushes the wrong way. Likely cause: the drive's encoder calibration. "
       << "Recalibrate the motor encoder in the CubeMars upper computer before running it again";
    return os.str();
  }
  return std::nullopt;
}

bool JointCommandCore::isRunaway(size_t joint) const {
  return joint < runaway_.size() && runaway_[joint];
}

std::vector<common_msgs::msg::MotorCmd>
JointCommandCore::armPoseToMotorCmds(const common_msgs::msg::ArmPose& pose,
                                     int8_t default_control_type) {
  if (joints_.size() != 6 && joints_.size() != 7) {
    throw std::runtime_error("JointCommandCore is not configured for 6 joints (+ gripper)");
  }

  if (pose.shoulder.position.size() < 3 || pose.elbow.position.size() < 2 ||
      pose.wrist.position.size() < 1) {
    throw std::runtime_error("ArmPose must contain 3 shoulder, 2 elbow, and 1 wrist positions");
  }

  std::vector<double> source_angles = {
      pose.shoulder.position[0], pose.shoulder.position[1], pose.shoulder.position[2],
      pose.elbow.position[0],    pose.elbow.position[1],    pose.wrist.position[0],
  };
  if (hasGripper()) {
    // Not in this pose (or not finite): hold where it is rather than open and drop the object.
    const bool driven = pose.include_gripper && std::isfinite(pose.gripper_position);
    source_angles.push_back(driven ? std::clamp(pose.gripper_position, 0.0, 1.0) *
                                         joints_[kGripperJoint].upper_limit
                                   : prev_targets_[kGripperJoint]);
  }

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
std::vector<common_msgs::msg::MotorCmd>
JointCommandCore::legPoseToMotorCmds(const common_msgs::msg::LegPose& pose,
                                     int8_t default_control_type) {
  if (pose.hip.position.size() < 3 || pose.knee.position.size() < 1 ||
      pose.ankle.position.size() < 2) {
    throw std::runtime_error("LegPose must contain 3 hip, 1 knee, and 2 ankle positions");
  }
  // legJointPaths() order.
  const std::vector<double> source_angles = {
      pose.hip.position[0],  pose.hip.position[1],   pose.hip.position[2],
      pose.knee.position[0], pose.ankle.position[0], pose.ankle.position[1],
  };
  if (!is_leg_ || joints_.size() != legJointPaths().size()) {
    throw std::runtime_error("JointCommandCore is not configured for a leg");
  }
  for (const double angle : source_angles) {
    if (!std::isfinite(angle)) {
      throw std::runtime_error("LegPose contains a non-finite angle");
    }
  }
  std::vector<common_msgs::msg::MotorCmd> commands;
  commands.reserve(joints_.size());
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
      continue;
    }
    if (isBlocked(i)) {
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
      cmd.position = static_cast<float>(calibrated_deg * kDegToRad);
      cmd.velocity = 0.0f;
      cmd.torque = 0.0f;
      cmd.kp = static_cast<float>(safety.mit_kp);
      cmd.kd = static_cast<float>(safety.mit_kd);
    } else {
      cmd.position = static_cast<float>(calibrated_deg);
    }
    commands.push_back(cmd);
  }

  prev_targets_ = std::move(next_targets);
  have_prev_targets_ = true;
  return commands;
}
