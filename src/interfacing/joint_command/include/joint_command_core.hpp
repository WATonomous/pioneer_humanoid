#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "common_msgs/msg/arm_pose.hpp"
#include "common_msgs/msg/motor_cmd.hpp"
#include "yaml-cpp/yaml.h"

struct JointConfig {
  int8_t motor_id{0};
  double lower_limit{-180.0};
  double upper_limit{180.0};
  int direction{1};
  double zero_offset{0.0};
  bool limit_range{false};
};

// CubeMars MIT dialect. GL II and AK disagree on the status byte (see mit_protocol.hpp).
enum class MitDriveFamily { Gl2, Ak };

// MIT joint behaviour on fault / stale stream / seeding / shutdown. Limp: kp = kd = 0.
// Damp: kp = 0, kd = mit_fault_kd -- the AK default, so a loaded arm sinks instead of dropping.
enum class MitFaultAction { Limp, Damp };

struct JointSafetyConfig {
  // false: this run leaves the motor alone (no command, no MIT enter, no watchdog).
  bool active{true};
  bool enable_position_clamp{true};
  bool enable_velocity_limit{true};
  bool enable_delta_limit{true};
  bool enable_low_pass{true};
  double velocity_max{30.0};   // degrees / second
  double delta_max{2.0};       // degrees / control step
  double low_pass_alpha{0.85}; // q_out = alpha * q_prev + (1-alpha) * q_cmd

  // MotorCmd::control_type for this joint; -1 = the node default (joint_command.yaml).
  int control_type{-1};

  // MIT_CONTROL (compliant holding) gains. Only used on joints whose control_type is
  // MIT_CONTROL; ignored for POSITION_LOOP etc. Default 0/0 is deliberately a
  // safe no-op (zero stiffness/damping = motor free) -- a joint must be explicitly configured
  // with nonzero mit_kp/mit_kd to actually hold under MIT. Units match the CubeMars AK-series
  // manual's MIT protocol range for this motor (see can/config/mit_profiles.yaml): kp in
  // [0,500], kd in [0,5] -- can_node clamps to the exact per-motor range before sending.
  double mit_kp{0.0};
  double mit_kd{0.0};

  // MIT watchdog limits (a PD drive has no internal limit checking).
  double mit_max_torque{0.3};       // N.m -- fault above this
  double mit_max_track_err{12.0};   // deg -- fault if the joint lags its setpoint by more
  double mit_feedback_timeout{0.2}; // s without feedback before faulting

  MitDriveFamily mit_family{MitDriveFamily::Gl2};
  // Motor model (AK10-9 / AK80-9 / GL40): MIT joints must name it; its testing ceiling caps
  // mit_max_torque.
  std::string motor;
  // Defaults from mit_family (ak -> Damp, gl2 -> Limp) unless set in the YAML.
  MitFaultAction mit_fault_action{MitFaultAction::Limp};
  bool mit_fault_action_explicit{false};
  double mit_fault_kd{0.0}; // N.m.s/rad, used by Damp (must be > 0)

  // Gravity feed-forward (MIT joints only), sent as MotorCmd.torque. scale 0 = off.
  double gravity_ff_scale{0.0};
  double gravity_ff_max_torque{0.0}; // N.m, |feed-forward| is clamped to this
  // Cmd-frame angle assumed while this joint is unpowered. Unset = feed-forward off.
  std::optional<double> gravity_assume_deg;
};

// One motor's latest feedback for the MIT watchdog (ROS-free so it is unit-testable).
struct MotorFeedbackSample {
  double position_deg{0.0}; // MOTOR frame, as published on /interfacing/motorFeedback
  double torque_nm{0.0};
  int status{0}; // GL II: 0 Disable / 1 Enable / else fault. AK: 0 ok / else fault.
  double age_s{0.0};
};

// Outcome of seeding from real feedback.
struct SeedReport {
  size_t matched{0};                // joints seeded from real feedback
  std::vector<size_t> unmatched;    // joint indices with no feedback (e.g. unwired)
  std::vector<size_t> out_of_range; // joints physically outside their configured limits
  std::string describe() const;
};

class JointCommandCore {
public:
  bool loadFromYaml(const YAML::Node& config, const std::string& arm_side);
  bool loadSafetyFromYaml(const YAML::Node& safety_cfg, double control_rate_hz);

  // Why loadSafetyFromYaml refused; empty on success.
  const std::string& lastError() const {
    return last_error_;
  }

  // Advance ONE control tick toward `pose`. Call at control_rate_hz: velocity_max is per tick.
  std::vector<common_msgs::msg::MotorCmd> armPoseToMotorCmds(const common_msgs::msg::ArmPose& pose,
                                                             int8_t default_control_type);

  // Fault-action frame per MIT joint (kp = 0; kd = 0 Limp / mit_fault_kd Damp). Used to poke
  // drives before seeding and to hold joints after a fault. damped_only skips Limp joints.
  std::vector<common_msgs::msg::MotorCmd> mitSafeCommands(bool damped_only = false) const;

  // MIT_ENTER / MIT_EXIT per MIT joint. limp_only: Damp joints are never exited.
  std::vector<common_msgs::msg::MotorCmd> mitModeCommands(int8_t control_type,
                                                          bool limp_only = false) const;
  bool hasDampedMitJoints() const;

  // Seed the rate-limiter's "previous target" from measured motor angles so the first
  // streamed ArmPose is velocity/delta-limited relative to the arm's ACTUAL pose, not an
  // assumed 0. Without this, an arm not physically at 0 gets a large first command (the
  // limiter ramps from 0), i.e. a slam. motor_positions: motor_id -> measured angle (deg).
  // Joints whose motor is ABSENT from the map are reported as unmatched and excluded from
  // commands until the next seed.
  SeedReport seedPrevTargetsFromFeedback(const std::map<int, double>& motor_positions);

  // arm_actuators.yaml `active` (default true). An inactive joint gets no command at all (no MIT
  // enter, no watchdog); to the gravity model its angle is unknown, as if unpowered.
  bool isActive(size_t joint) const;

  // Exclude joints seeded outside their limits (calibration mismatch) until the next seed.
  void blockJoints(const std::vector<size_t>& indices);
  bool isBlocked(size_t joint) const;
  bool isUnpowered(size_t joint) const;

  // MIT watchdog: a fault description, or nullopt when every MIT joint is healthy.
  std::optional<std::string>
  checkMitFaults(const std::map<int, MotorFeedbackSample>& feedback) const;

  bool isMitJoint(size_t joint) const;
  std::vector<int> mitMotorIds() const;
  int8_t motorId(size_t joint) const;
  std::string jointName(size_t joint) const;

  const std::vector<double>& prevTargets() const {
    return prev_targets_;
  }

  // Last command per motor, MOTOR frame (deg) -- the same frame as feedback.
  const std::vector<double>& lastMotorCmdDeg() const {
    return last_motor_cmd_deg_;
  }

  // Unscaled gravity-model torque for the last command, MOTOR frame (N.m).
  const std::vector<double>& lastGravityTorqueMotor() const {
    return last_gravity_torque_motor_;
  }

  size_t jointCount() const {
    return joints_.size();
  }

  // The gripper (GL40, gripper.open_close) is joint kGripperJoint, present only when
  // arm_actuators.yaml has a block for it. gripper_position 0 (open) -> cmd 0, 1 -> upper_limit.
  static constexpr size_t kGripperJoint = 6;
  bool hasGripper() const {
    return joints_.size() > kGripperJoint;
  }

  const JointSafetyConfig& safety(size_t joint) const {
    return safety_.at(joint);
  }

  const JointConfig& joint(size_t joint) const {
    return joints_.at(joint);
  }

  // Gain as the drive applies it after 12-bit quantisation (nearest code).
  static double quantiseKp(double kp, double kp_max = 500.0);
  static double quantiseKd(double kd, double kd_max = 5.0);

private:
  static JointConfig loadJointConfig(const YAML::Node& joint_node);
  static JointSafetyConfig loadJointSafetyConfig(const YAML::Node& joint_node,
                                                 const JointSafetyConfig& base);
  static double clampAngle(double angle, const JointConfig& joint);
  static double applyCalibration(double angle, const JointConfig& joint);
  static double clampStep(double target, double previous, double delta_max);
  static double applyLowPass(double target, double previous, double alpha);
  bool validateMitGains();
  bool validateGripper();
  void resizeJointState();
  common_msgs::msg::MotorCmd mitSafeCommand(size_t joint) const;
  std::vector<double> gravityTorqueMotor(const std::vector<double>& cmd_targets_deg) const;

  std::vector<JointConfig> joints_;
  // arm_calibration.yaml's gripper, appended to joints_ only if arm_actuators.yaml drives it.
  std::optional<JointConfig> gripper_mapping_;
  std::vector<JointSafetyConfig> safety_;
  std::vector<double> prev_targets_;
  std::vector<double> last_motor_cmd_deg_;
  std::vector<bool> blocked_;
  std::vector<bool> unpowered_; // no feedback at the last seed; recomputed every seed
  std::vector<double> last_gravity_torque_motor_;
  // 0 -> 1 after every seed, so feed-forward never steps on.
  double gravity_ff_ramp_{0.0};
  std::string arm_side_;
  bool have_prev_targets_{false};
  double control_rate_hz_{50.0};
  std::string last_error_;
};
