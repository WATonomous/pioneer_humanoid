#include "joint_command_node.hpp"

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>

JointCommandNode::JointCommandNode() : Node("joint_command_node") {
  this->declare_parameter("arm_side", "left");
  this->declare_parameter("control_rate_hz", 50.0);
  this->declare_parameter("input_topic", "/arm/joint_targets");
  this->declare_parameter("motor_cmd_topic", "/interfacing/motorCMD");
  this->declare_parameter("control_type", common_msgs::msg::MotorCmd::POSITION_LOOP);
  this->declare_parameter("feedback_topic", "/interfacing/motorFeedback");
  this->declare_parameter("command_timeout_sec", 10.0);
  this->declare_parameter("mit_shutdown_damp_sec", 2.0);

  const std::string arm_side = this->get_parameter("arm_side").as_string();
  control_rate_hz_ = this->get_parameter("control_rate_hz").as_double();
  const std::string input_topic = this->get_parameter("input_topic").as_string();
  const std::string motor_cmd_topic = this->get_parameter("motor_cmd_topic").as_string();
  control_type_ = static_cast<int8_t>(this->get_parameter("control_type").as_int());
  const std::string feedback_topic = this->get_parameter("feedback_topic").as_string();
  command_timeout_sec_ = this->get_parameter("command_timeout_sec").as_double();
  mit_shutdown_damp_sec_ = this->get_parameter("mit_shutdown_damp_sec").as_double();

  const YAML::Node hardware_config =
      YAML::LoadFile(ament_index_cpp::get_package_share_directory("joint_command") +
                     "/config/arm_calibration.yaml");

  if (!core_.loadFromYaml(hardware_config, arm_side)) {
    throw std::runtime_error("Failed to load 6-joint mapping for arm side '" + arm_side + "'");
  }

  const std::string safety_config_path =
      ament_index_cpp::get_package_share_directory("joint_command") + "/config/arm_actuators.yaml";
  const YAML::Node safety_config = YAML::LoadFile(safety_config_path);
  const YAML::Node safety_root = safety_config["safety"];
  if (!core_.loadSafetyFromYaml(safety_root, control_rate_hz_)) {
    // Refusing to start is deliberate: an unsafe gain here reaches a real motor.
    throw std::runtime_error("Failed to load safety config from '" + safety_config_path +
                             "': " + core_.lastError());
  }

  motor_cmd_pub_ =
      this->create_publisher<common_msgs::msg::MotorCmd>(motor_cmd_topic, rclcpp::QoS(10));

  arm_pose_sub_ = this->create_subscription<common_msgs::msg::ArmPose>(
      input_topic, rclcpp::QoS(10),
      std::bind(&JointCommandNode::armPoseCallback, this, std::placeholders::_1));

  feedback_sub_ = this->create_subscription<common_msgs::msg::MotorFeedback>(
      feedback_topic, rclcpp::QoS(20),
      std::bind(&JointCommandNode::motorFeedbackCallback, this, std::placeholders::_1));

  const auto timer_period = std::chrono::duration<double>(1.0 / control_rate_hz_);
  control_timer_ =
      this->create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(timer_period),
                              std::bind(&JointCommandNode::controlTimerCallback, this));

  const std::vector<int> mit_ids = core_.mitMotorIds();
  if (!mit_ids.empty()) {
    std::string ids;
    for (const int id : mit_ids) {
      ids += (ids.empty() ? "" : ", ") + std::to_string(id);
    }
    // A GL II ignores MIT frames until entered. Gains apply only after seeding.
    for (const auto& cmd : core_.mitModeCommands(common_msgs::msg::MotorCmd::MIT_ENTER)) {
      motor_cmd_pub_->publish(cmd);
    }
    RCLCPP_INFO(this->get_logger(), "MIT joints: motor ids [%s] -- sent enter-motor-mode",
                ids.c_str());
  }

  RCLCPP_INFO(this->get_logger(), "Joint command node ready: arm=%s, joints=%zu, rate=%.1f Hz",
              arm_side.c_str(), core_.jointCount(), control_rate_hz_);
}

void JointCommandNode::armPoseCallback(const common_msgs::msg::ArmPose::SharedPtr msg) {
  // Only cache the pose here. Moderation and publishing happen exclusively on control_timer_'s
  // own steady 50Hz clock (below) -- previously this also
  // called publishMotorCommands() directly, so with ArmPose already arriving at ~50Hz from
  // task_space_real.py, motor commands were sent from two independent, phase-unrelated 50Hz
  // sources at once (up to ~100Hz combined, jittery), not the intended steady 50Hz.
  for (const auto* joints : {&msg->shoulder, &msg->elbow, &msg->wrist}) {
    for (const double a : joints->position) {
      if (!std::isfinite(a)) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                              "ArmPose has a non-finite angle -- ignored, holding the last pose");
        return;
      }
    }
  }
  latest_pose_ = *msg;
  have_latest_pose_ = true;
  last_armpose_time_ = this->get_clock()->now();
  have_armpose_time_ = true;
}

void JointCommandNode::motorFeedbackCallback(const common_msgs::msg::MotorFeedback::SharedPtr msg) {
  // Record only; seeding happens on the first ArmPose using this latest snapshot.
  const int id = static_cast<int>(msg->motor_id);
  latest_feedback_[id] = static_cast<double>(msg->position);
  MotorFeedbackSample sample;
  sample.position_deg = static_cast<double>(msg->position);
  sample.torque_nm = static_cast<double>(msg->torque);
  sample.status = static_cast<int>(msg->error_code);
  latest_feedback_full_[id] = sample;
  latest_feedback_time_[id] = this->get_clock()->now();
}

std::map<int, MotorFeedbackSample> JointCommandNode::mitFeedbackSamples() {
  std::map<int, MotorFeedbackSample> out;
  const rclcpp::Time now = this->get_clock()->now();
  for (const auto& [id, sample] : latest_feedback_full_) {
    MotorFeedbackSample with_age = sample;
    const auto it = latest_feedback_time_.find(id);
    with_age.age_s = (it == latest_feedback_time_.end()) ? 1e9 : (now - it->second).seconds();
    out[id] = with_age;
  }
  return out;
}

bool JointCommandNode::trySeedFromFeedback() {
  // Only motors reporting now: latest_feedback_ keeps a powered-off motor's last angle.
  constexpr double kSeedFeedbackMaxAgeS = 0.5;
  const rclcpp::Time now = this->get_clock()->now();
  std::map<int, double> fresh;
  for (const auto& [id, pos] : latest_feedback_) {
    const auto it = latest_feedback_time_.find(id);
    if (it != latest_feedback_time_.end() && (now - it->second).seconds() < kSeedFeedbackMaxAgeS) {
      fresh[id] = pos;
    }
  }
  if (fresh.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Waiting for motor feedback before accepting ArmPose: the "
                         "rate-limiter must be seeded from the real pose first. No command "
                         "sent. (Is can_node running and are the motors powered?)");
    return false;
  }

  const SeedReport report = core_.seedPrevTargetsFromFeedback(fresh);
  seeded_from_feedback_ = true;

  const auto& seeded = core_.prevTargets();
  std::string s;
  for (size_t i = 0; i < seeded.size(); ++i) {
    s += (i ? ", " : "") + std::to_string(seeded[i]);
  }
  RCLCPP_INFO(this->get_logger(),
              "Rate-limiter seeded from feedback: %s. prev_targets(cmd-frame deg)=[%s]. Now "
              "accepting ArmPose; motion ramps from here.",
              report.describe().c_str(), s.c_str());

  for (const size_t i : report.unmatched) {
    RCLCPP_WARN(this->get_logger(),
                "EXCLUDING joint %s (motor %d): no feedback -- not commanded%s until the next "
                "seed (after the ArmPose stream goes stale).",
                core_.jointName(i).c_str(), static_cast<int>(core_.motorId(i)),
                core_.isMitJoint(i) ? " (MIT: held at kp=0)" : "");
    const auto& assume = core_.safety(i).gravity_assume_deg;
    if (assume.has_value()) {
      RCLCPP_WARN(this->get_logger(),
                  "Gravity model ASSUMES %s is at %.1f deg (cmd frame, gravity_assume_deg). "
                  "Only true while that limp joint is strapped there.",
                  core_.jointName(i).c_str(), *assume);
    }
  }

  if (!report.out_of_range.empty()) {
    core_.blockJoints(report.out_of_range);
    for (const size_t i : report.out_of_range) {
      RCLCPP_ERROR(this->get_logger(),
                   "EXCLUDING joint %s (motor %d): it is at %.1f deg, outside its configured "
                   "limits [%.1f, %.1f] in arm_calibration.yaml. Its calibration is stale or "
                   "the limits are placeholders. This joint will NOT be commanded (MIT joints "
                   "are held limp) until the mapping is corrected -- re-run calibrate_arm.py.",
                   core_.jointName(i).c_str(), static_cast<int>(core_.motorId(i)),
                   core_.prevTargets()[i], core_.joint(i).lower_limit, core_.joint(i).upper_limit);
    }
  }
  return true;
}

void JointCommandNode::mitFault(const std::string& reason) {
  if (mit_faulted_) {
    return;
  }
  mit_faulted_ = true;
  RCLCPP_ERROR(this->get_logger(),
               "MIT FAULT: %s. Limp MIT joints are freed, damped ones are held at kp=0 "
               "(sinking slowly) while this node runs; commands are halted. Restart the node "
               "after checking the hardware.",
               reason.c_str());
  exitMitJoints(/*limp_only=*/true);
  for (const auto& cmd : core_.mitSafeCommands(/*damped_only=*/true)) {
    motor_cmd_pub_->publish(cmd);
  }
}

void JointCommandNode::exitMitJoints(bool limp_only) {
  const auto cmds = core_.mitModeCommands(common_msgs::msg::MotorCmd::MIT_EXIT, limp_only);
  if (cmds.empty() || !motor_cmd_pub_) {
    return;
  }
  // Sent repeatedly: a dropped frame here leaves a stiff motor holding position.
  for (int attempt = 0; attempt < 3; ++attempt) {
    for (const auto& cmd : cmds) {
      motor_cmd_pub_->publish(cmd);
    }
  }
}

void JointCommandNode::shutdownMitJoints() {
  if (core_.mitMotorIds().empty() || !motor_cmd_pub_) {
    return;
  }
  // Damp joints sink under damping first so the final EXIT doesn't drop the arm. Needs can_node
  // alive: stop joint_command before can_node, arm supported.
  if (core_.hasDampedMitJoints() && mit_shutdown_damp_sec_ > 0.0) {
    RCLCPP_WARN(this->get_logger(), "Shutdown: damping MIT joints for %.1f s before exiting",
                mit_shutdown_damp_sec_);
    const auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                         std::chrono::duration<double>(mit_shutdown_damp_sec_));
    const auto damp = core_.mitSafeCommands(/*damped_only=*/true);
    while (std::chrono::steady_clock::now() < end) {
      for (const auto& cmd : damp) {
        motor_cmd_pub_->publish(cmd);
      }
      std::this_thread::sleep_for(period);
    }
  }
  exitMitJoints(/*limp_only=*/false);
}

void JointCommandNode::controlTimerCallback() {
  if (mit_faulted_) {
    // Keep damping Damp joints: a silent AK trips its CAN timeout and drops the arm.
    for (const auto& cmd : core_.mitSafeCommands(/*damped_only=*/true)) {
      motor_cmd_pub_->publish(cmd);
    }
    return;
  }

  // Poke MIT drives before seeding: they only answer when spoken to. kp = 0 pulls nowhere.
  if (!seeded_from_feedback_) {
    // Re-send MIT_ENTER (the constructor's can be lost before DDS matches can_node): every
    // 0.5 s for the first 5 s, then every 5 s.
    const int startup_ticks = static_cast<int>(control_rate_hz_ * 5.0);
    const int enter_every = std::max(
        1, static_cast<int>(control_rate_hz_ * (unseeded_ticks_ < startup_ticks ? 0.5 : 5.0)));
    if (unseeded_ticks_++ % enter_every == 0) {
      for (const auto& cmd : core_.mitModeCommands(common_msgs::msg::MotorCmd::MIT_ENTER)) {
        motor_cmd_pub_->publish(cmd);
      }
    }
    for (const auto& cmd : core_.mitSafeCommands()) {
      motor_cmd_pub_->publish(cmd);
    }
  } else {
    unseeded_ticks_ = 0;
  }

  if (!have_latest_pose_) {
    return;
  }

  if (have_armpose_time_ &&
      (this->get_clock()->now() - last_armpose_time_).seconds() > command_timeout_sec_) {
    // No fresh ArmPose within the timeout window: stop republishing the stale cached
    // target and force a fresh seed-from-feedback + ramp on the next real command,
    // instead of instantly resuming motion toward a target that could be arbitrarily old.
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "ArmPose stream stale (no message in %.1fs): halting motor commands "
                         "until a fresh ArmPose re-seeds the rate-limiter. MIT joints are held "
                         "at kp=0 (limp or damped, per mit_fault_action).",
                         command_timeout_sec_);
    have_latest_pose_ = false;
    seeded_from_feedback_ = false;
    for (const auto& cmd : core_.mitSafeCommands()) {
      motor_cmd_pub_->publish(cmd);
    }
    return;
  }

  // Refuse to act until the rate-limiter has been seeded from the arm's real pose. We seed
  // lazily on the first ArmPose (feedback has been streaming by the time an operator starts
  // commanding), using the freshest feedback right before motion begins.
  if (!seeded_from_feedback_ && !trySeedFromFeedback()) {
    return;
  }

  // Watchdog runs before the next command is generated, so a fault stops the arm this tick.
  if (!core_.mitMotorIds().empty()) {
    const auto fault = core_.checkMitFaults(mitFeedbackSamples());
    if (fault.has_value()) {
      mitFault(*fault);
      return;
    }
  }

  try {
    publishMotorCommands(core_.armPoseToMotorCmds(latest_pose_, control_type_));
    logGravityModel();
  } catch (const std::exception& e) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                          "Failed to process ArmPose: %s", e.what());
  }
}

void JointCommandNode::logGravityModel() {
  // Bring-up check: with gravity_ff_scale = 0, pred and meas should agree (sign included).
  const auto& pred = core_.lastGravityTorqueMotor();
  std::string s;
  for (size_t i = 0; i < pred.size(); ++i) {
    const auto it = latest_feedback_full_.find(static_cast<int>(core_.motorId(i)));
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s%s pred %+.2f meas %+.2f", i ? " | " : "",
                  core_.jointName(i).c_str(), pred[i],
                  it == latest_feedback_full_.end() ? 0.0 : it->second.torque_nm);
    s += buf;
  }
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                       "Gravity model, motor frame N.m: %s", s.c_str());
}

void JointCommandNode::publishMotorCommands(const std::vector<common_msgs::msg::MotorCmd>& cmds) {
  // One-shot proof of how many MotorCmds this node actually emits per cycle (and for which
  // motor ids), logged from INSIDE the node so it's independent of any subscriber-side
  // (ros2 echo/hz) message-drop artifact.
  if (!logged_publish_count_) {
    std::string ids;
    for (const auto& cmd : cmds) {
      ids += (ids.empty() ? "" : ", ") + std::to_string(static_cast<int>(cmd.motor_id)) + "(ct" +
             std::to_string(static_cast<int>(cmd.control_type)) + ")";
    }
    RCLCPP_INFO(this->get_logger(), "Publishing %zu MotorCmd per cycle; motor_ids=[%s]",
                cmds.size(), ids.c_str());
    logged_publish_count_ = true;
  }

  for (const auto& cmd : cmds) {
    motor_cmd_pub_->publish(cmd);
  }
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<JointCommandNode>();
  // Ctrl-C: leave MIT joints safe. Must run pre-shutdown: after spin returns the context is
  // invalid and publish() silently drops every message.
  std::weak_ptr<JointCommandNode> weak = node;
  rclcpp::contexts::get_global_default_context()->add_pre_shutdown_callback([weak]() {
    if (auto n = weak.lock()) {
      n->shutdownMitJoints();
    }
  });
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
