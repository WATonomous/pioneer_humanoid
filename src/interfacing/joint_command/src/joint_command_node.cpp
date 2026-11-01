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
  this->declare_parameter("arms", std::vector<std::string>{"left"});
  this->declare_parameter("control_rate_hz", 50.0);
  this->declare_parameter("motor_cmd_topic", "/interfacing/motorCMD");
  this->declare_parameter("control_type", common_msgs::msg::MotorCmd::POSITION_LOOP);
  this->declare_parameter("feedback_topic", "/interfacing/motorFeedback");
  this->declare_parameter("command_timeout_sec", 10.0);
  this->declare_parameter("mit_shutdown_damp_sec", 2.0);

  const std::vector<std::string> arms = this->get_parameter("arms").as_string_array();
  control_rate_hz_ = this->get_parameter("control_rate_hz").as_double();
  const std::string motor_cmd_topic = this->get_parameter("motor_cmd_topic").as_string();
  control_type_ = static_cast<int8_t>(this->get_parameter("control_type").as_int());
  const std::string feedback_topic = this->get_parameter("feedback_topic").as_string();
  command_timeout_sec_ = this->get_parameter("command_timeout_sec").as_double();
  mit_shutdown_damp_sec_ = this->get_parameter("mit_shutdown_damp_sec").as_double();

  const YAML::Node hardware_config =
      YAML::LoadFile(ament_index_cpp::get_package_share_directory("joint_command") +
                     "/config/arm_calibration.yaml");
  const std::string safety_config_path =
      ament_index_cpp::get_package_share_directory("joint_command") + "/config/arm_actuators.yaml";
  const YAML::Node safety_root = YAML::LoadFile(safety_config_path)["safety"];

  // Every listed arm is loaded and validated, active or not.
  std::vector<Limb> loaded(arms.size());
  std::vector<const JointCommandCore*> cores;
  for (size_t k = 0; k < arms.size(); ++k) {
    Limb& limb = loaded[k];
    limb.side = arms[k];
    if (!limb.core.loadFromYaml(hardware_config, limb.side)) {
      throw std::runtime_error("Failed to load 6-joint mapping for arm side '" + limb.side + "'");
    }
    if (!limb.core.loadSafetyFromYaml(safety_root, control_rate_hz_)) {
      // Refusing to start is deliberate: an unsafe gain here reaches a real motor.
      throw std::runtime_error("Failed to load safety config for arm '" + limb.side + "' from '" +
                               safety_config_path + "': " + limb.core.lastError());
    }
    cores.push_back(&limb.core);
  }
  if (const auto clash = findDuplicateMotorIds(cores)) {
    throw std::runtime_error("Motor ids must be unique across arms (feedback is matched by id "
                             "alone): " +
                             *clash);
  }

  for (Limb& limb : loaded) {
    if (!limb.core.anyActive()) {
      RCLCPP_INFO(this->get_logger(), "[%s] no active joint: this arm is not driven this run",
                  limb.side.c_str());
      continue;
    }
    std::string active;
    for (size_t i = 0; i < limb.core.jointCount(); ++i) {
      active += std::string(active.empty() ? "" : ", ") + limb.core.jointName(i) +
                (limb.core.isActive(i) ? "" : " (INACTIVE)");
    }
    RCLCPP_INFO(this->get_logger(), "[%s] joints this run: %s", limb.side.c_str(), active.c_str());
    limbs_.push_back(std::move(limb));
  }
  if (limbs_.empty()) {
    throw std::runtime_error("No arm in `arms` has an active joint: set active: true on at least "
                             "one joint in arm_actuators.yaml");
  }

  // Depth: one tick publishes every arm's commands in a burst, and a fault sends MIT_EXIT 3x per
  // joint; a shallow queue would overwrite some before can_node reads them.
  motor_cmd_pub_ =
      this->create_publisher<common_msgs::msg::MotorCmd>(motor_cmd_topic, rclcpp::QoS(100));

  // limbs_ is final here, so the index each callback captures stays valid.
  for (size_t i = 0; i < limbs_.size(); ++i) {
    const std::string topic = "/arm/" + limbs_[i].side + "/joint_targets";
    limbs_[i].arm_pose_sub = this->create_subscription<common_msgs::msg::ArmPose>(
        topic, rclcpp::QoS(10),
        [this, i](const common_msgs::msg::ArmPose::SharedPtr msg) { armPoseCallback(i, msg); });
    RCLCPP_INFO(this->get_logger(), "[%s] listening on %s", limbs_[i].side.c_str(), topic.c_str());
  }

  feedback_sub_ = this->create_subscription<common_msgs::msg::MotorFeedback>(
      feedback_topic, rclcpp::QoS(20),
      std::bind(&JointCommandNode::motorFeedbackCallback, this, std::placeholders::_1));

  const auto timer_period = std::chrono::duration<double>(1.0 / control_rate_hz_);
  control_timer_ =
      this->create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(timer_period),
                              std::bind(&JointCommandNode::controlTimerCallback, this));

  for (const Limb& limb : limbs_) {
    const std::vector<int> mit_ids = limb.core.mitMotorIds();
    if (mit_ids.empty()) {
      continue;
    }
    std::string ids;
    for (const int id : mit_ids) {
      ids += (ids.empty() ? "" : ", ") + std::to_string(id);
    }
    // A GL II ignores MIT frames until entered. Gains apply only after seeding.
    publish(limb.core.mitModeCommands(common_msgs::msg::MotorCmd::MIT_ENTER));
    RCLCPP_INFO(this->get_logger(), "[%s] MIT joints: motor ids [%s] -- sent enter-motor-mode",
                limb.side.c_str(), ids.c_str());
  }

  std::string sides;
  for (const Limb& limb : limbs_) {
    sides += (sides.empty() ? "" : ", ") + limb.side;
  }
  RCLCPP_INFO(this->get_logger(), "Joint command node ready: arms=[%s], rate=%.1f Hz",
              sides.c_str(), control_rate_hz_);
}

void JointCommandNode::armPoseCallback(size_t limb_index,
                                       const common_msgs::msg::ArmPose::SharedPtr msg) {
  Limb& limb = limbs_[limb_index];
  // Each arm has its own topic; a pose meant for the other arm moves nothing.
  if (msg->is_left != (limb.side == "left")) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "[%s] ArmPose with is_left=%s on this arm's topic -- ignored, holding "
                          "the last pose",
                          limb.side.c_str(), msg->is_left ? "true" : "false");
    return;
  }
  // Only cache the pose here. Moderation and publishing happen exclusively on control_timer_'s
  // own steady 50Hz clock (below) -- previously this also
  // called publishMotorCommands() directly, so with ArmPose already arriving at ~50Hz from
  // task_space_real.py, motor commands were sent from two independent, phase-unrelated 50Hz
  // sources at once (up to ~100Hz combined, jittery), not the intended steady 50Hz.
  for (const auto* joints : {&msg->shoulder, &msg->elbow, &msg->wrist}) {
    for (const double a : joints->position) {
      if (!std::isfinite(a)) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                              "[%s] ArmPose has a non-finite angle -- ignored, holding the last "
                              "pose",
                              limb.side.c_str());
        return;
      }
    }
  }
  if (msg->include_gripper && !std::isfinite(msg->gripper_position)) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "[%s] ArmPose has a non-finite gripper_position -- ignored, holding the "
                          "last pose",
                          limb.side.c_str());
    return;
  }
  limb.latest_pose = *msg;
  limb.have_latest_pose = true;
  limb.last_armpose_time = this->get_clock()->now();
  limb.have_armpose_time = true;
}

void JointCommandNode::motorFeedbackCallback(const common_msgs::msg::MotorFeedback::SharedPtr msg) {
  // Record only; seeding happens on the first ArmPose using this latest snapshot.
  const int id = static_cast<int>(msg->motor_id);
  latest_feedback_[id] = static_cast<double>(msg->position);
  MotorFeedbackSample sample;
  sample.position_deg = static_cast<double>(msg->position);
  sample.torque_nm = static_cast<double>(msg->torque);
  sample.temperature_c = static_cast<double>(msg->temperature);
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

bool JointCommandNode::trySeedFromFeedback(Limb& limb) {
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
                         "[%s] Waiting for motor feedback before accepting ArmPose: the "
                         "rate-limiter must be seeded from the real pose first. No command "
                         "sent. (Is can_node running and are the motors powered?)",
                         limb.side.c_str());
    return false;
  }

  JointCommandCore& core = limb.core;
  const SeedReport report = core.seedPrevTargetsFromFeedback(fresh);
  limb.seeded_from_feedback = true;

  const auto& seeded = core.prevTargets();
  std::string s;
  for (size_t i = 0; i < seeded.size(); ++i) {
    s += (i ? ", " : "") + std::to_string(seeded[i]);
  }
  RCLCPP_INFO(this->get_logger(),
              "[%s] Rate-limiter seeded from feedback: %s. prev_targets(cmd-frame deg)=[%s]. Now "
              "accepting ArmPose; motion ramps from here.",
              limb.side.c_str(), report.describe().c_str(), s.c_str());

  for (const size_t i : report.unmatched) {
    RCLCPP_WARN(this->get_logger(),
                "[%s] EXCLUDING joint %s (motor %d): no feedback -- not commanded%s until the next "
                "seed (after the ArmPose stream goes stale).",
                limb.side.c_str(), core.jointName(i).c_str(), static_cast<int>(core.motorId(i)),
                core.isMitJoint(i) ? " (MIT: held at kp=0)" : "");
    const auto& assume = core.safety(i).gravity_assume_deg;
    if (assume.has_value()) {
      RCLCPP_WARN(this->get_logger(),
                  "[%s] Gravity model ASSUMES %s is at %.1f deg (cmd frame, gravity_assume_deg). "
                  "Only true while that limp joint is strapped there.",
                  limb.side.c_str(), core.jointName(i).c_str(), *assume);
    }
  }

  if (!report.out_of_range.empty()) {
    core.blockJoints(report.out_of_range);
    for (const size_t i : report.out_of_range) {
      RCLCPP_ERROR(this->get_logger(),
                   "[%s] EXCLUDING joint %s (motor %d): it is at %.1f deg, outside its configured "
                   "limits [%.1f, %.1f] in arm_calibration.yaml. Its calibration is stale or "
                   "the limits are placeholders. This joint will NOT be commanded (MIT joints "
                   "are held limp) until the mapping is corrected -- re-run calibrate_arm.py.",
                   limb.side.c_str(), core.jointName(i).c_str(), static_cast<int>(core.motorId(i)),
                   core.prevTargets()[i], core.joint(i).lower_limit, core.joint(i).upper_limit);
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
               "MIT FAULT: %s. EVERY arm is halted: limp MIT joints are freed, damped ones are "
               "held at kp=0 (sinking slowly) while this node runs; commands are halted. Restart "
               "the node after checking the hardware.",
               reason.c_str());
  exitMitJoints(/*limp_only=*/true);
  for (const Limb& limb : limbs_) {
    publish(limb.core.mitSafeCommands(/*damped_only=*/true));
  }
}

void JointCommandNode::exitMitJoints(bool limp_only) {
  std::vector<common_msgs::msg::MotorCmd> cmds;
  for (const Limb& limb : limbs_) {
    const auto exits = limb.core.mitModeCommands(common_msgs::msg::MotorCmd::MIT_EXIT, limp_only);
    cmds.insert(cmds.end(), exits.begin(), exits.end());
  }
  if (cmds.empty() || !motor_cmd_pub_) {
    return;
  }
  // Sent repeatedly: a dropped frame here leaves a stiff motor holding position.
  for (int attempt = 0; attempt < 3; ++attempt) {
    publish(cmds);
  }
}

void JointCommandNode::shutdownMitJoints() {
  // Stop the control loop first: it would keep sending stiff frames toward a still-streaming
  // ArmPose (teleop left running) in between the damping frames below.
  if (control_timer_) {
    control_timer_->cancel();
  }
  std::vector<common_msgs::msg::MotorCmd> damp;
  bool any_mit = false;
  for (const Limb& limb : limbs_) {
    any_mit = any_mit || !limb.core.mitMotorIds().empty();
    const auto cmds = limb.core.mitSafeCommands(/*damped_only=*/true);
    damp.insert(damp.end(), cmds.begin(), cmds.end());
  }
  if (!any_mit || !motor_cmd_pub_) {
    return;
  }
  // Damp joints sink under damping first so the final EXIT doesn't drop the arm. Needs can_node
  // alive: stop joint_command before can_node, arm supported.
  if (!damp.empty() && mit_shutdown_damp_sec_ > 0.0) {
    RCLCPP_WARN(this->get_logger(), "Shutdown: damping MIT joints for %.1f s before exiting",
                mit_shutdown_damp_sec_);
    const auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                         std::chrono::duration<double>(mit_shutdown_damp_sec_));
    while (std::chrono::steady_clock::now() < end) {
      publish(damp);
      std::this_thread::sleep_for(period);
    }
  }
  exitMitJoints(/*limp_only=*/false);
}

bool JointCommandNode::readyToCommand(Limb& limb) {
  // Poke MIT drives before seeding: they only answer when spoken to. kp = 0 pulls nowhere.
  if (!limb.seeded_from_feedback) {
    // Re-send MIT_ENTER (the constructor's can be lost before DDS matches can_node): every
    // 0.5 s for the first 5 s, then every 5 s.
    const int startup_ticks = static_cast<int>(control_rate_hz_ * 5.0);
    const int enter_every = std::max(
        1, static_cast<int>(control_rate_hz_ * (limb.unseeded_ticks < startup_ticks ? 0.5 : 5.0)));
    if (limb.unseeded_ticks++ % enter_every == 0) {
      publish(limb.core.mitModeCommands(common_msgs::msg::MotorCmd::MIT_ENTER));
    }
    publish(limb.core.mitSafeCommands());
  } else {
    limb.unseeded_ticks = 0;
  }

  if (!limb.have_latest_pose) {
    return false;
  }

  if (limb.have_armpose_time &&
      (this->get_clock()->now() - limb.last_armpose_time).seconds() > command_timeout_sec_) {
    // No fresh ArmPose within the timeout window: stop republishing the stale cached
    // target and force a fresh seed-from-feedback + ramp on the next real command,
    // instead of instantly resuming motion toward a target that could be arbitrarily old.
    // Not throttled: it fires once per stale stream, and a throttle would hide the other arm's.
    RCLCPP_WARN(this->get_logger(),
                "[%s] ArmPose stream stale (no message in %.1fs): halting motor commands "
                "until a fresh ArmPose re-seeds the rate-limiter. MIT joints are held "
                "at kp=0 (limp or damped, per mit_fault_action).",
                limb.side.c_str(), command_timeout_sec_);
    limb.have_latest_pose = false;
    limb.seeded_from_feedback = false;
    publish(limb.core.mitSafeCommands());
    return false;
  }

  // Refuse to act until the rate-limiter has been seeded from the arm's real pose. We seed
  // lazily on the first ArmPose (feedback has been streaming by the time an operator starts
  // commanding), using the freshest feedback right before motion begins.
  return limb.seeded_from_feedback || trySeedFromFeedback(limb);
}

void JointCommandNode::controlTimerCallback() {
  if (mit_faulted_) {
    // Keep damping Damp joints: a silent AK trips its CAN timeout and drops the arm.
    for (const Limb& limb : limbs_) {
      publish(limb.core.mitSafeCommands(/*damped_only=*/true));
    }
    return;
  }

  std::vector<Limb*> ready;
  for (Limb& limb : limbs_) {
    if (readyToCommand(limb)) {
      ready.push_back(&limb);
    }
  }
  if (ready.empty()) {
    return;
  }

  // Watchdog runs before the next command is generated, so a fault stops every arm this tick.
  const auto feedback = mitFeedbackSamples();
  for (const Limb* limb : ready) {
    if (limb->core.mitMotorIds().empty()) {
      continue;
    }
    const auto fault = limb->core.checkMitFaults(feedback);
    if (fault.has_value()) {
      mitFault("[" + limb->side + "] " + *fault);
      return;
    }
  }

  for (Limb* limb : ready) {
    try {
      publishMotorCommands(*limb, limb->core.armPoseToMotorCmds(limb->latest_pose, control_type_));
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "[%s] Failed to process ArmPose: %s", limb->side.c_str(), e.what());
    }
  }
  logGravityModel(ready);
}

void JointCommandNode::logGravityModel(const std::vector<Limb*>& limbs) {
  // Bring-up check: with gravity_ff_scale = 0, pred and meas should agree (sign included).
  // One line for every arm: the throttle is per call site, so a line per arm would hide all but
  // one. An arm without a gravity model (gravity_model.hpp is the left arm) shows meas only.
  std::string s;
  for (const Limb* limb : limbs) {
    const JointCommandCore& core = limb->core;
    const auto& pred = core.lastGravityTorqueMotor();
    s += (s.empty() ? "[" : " || [") + limb->side + "] ";
    for (size_t i = 0; i < pred.size(); ++i) {
      const auto it = latest_feedback_full_.find(static_cast<int>(core.motorId(i)));
      const double meas = it == latest_feedback_full_.end() ? 0.0 : it->second.torque_nm;
      char buf[96];
      if (core.hasGravityModel()) {
        std::snprintf(buf, sizeof(buf), "%s%s pred %+.2f meas %+.2f", i ? " | " : "",
                      core.jointName(i).c_str(), pred[i], meas);
      } else {
        std::snprintf(buf, sizeof(buf), "%s%s meas %+.2f", i ? " | " : "",
                      core.jointName(i).c_str(), meas);
      }
      s += buf;
    }
  }
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                       "Gravity model, motor frame N.m: %s", s.c_str());
}

void JointCommandNode::publishMotorCommands(Limb& limb,
                                            const std::vector<common_msgs::msg::MotorCmd>& cmds) {
  // One-shot proof of how many MotorCmds this node actually emits per cycle (and for which
  // motor ids), logged from INSIDE the node so it's independent of any subscriber-side
  // (ros2 echo/hz) message-drop artifact.
  if (!limb.logged_publish_count) {
    std::string ids;
    for (const auto& cmd : cmds) {
      ids += (ids.empty() ? "" : ", ") + std::to_string(static_cast<int>(cmd.motor_id)) + "(ct" +
             std::to_string(static_cast<int>(cmd.control_type)) + ")";
    }
    RCLCPP_INFO(this->get_logger(), "[%s] Publishing %zu MotorCmd per cycle; motor_ids=[%s]",
                limb.side.c_str(), cmds.size(), ids.c_str());
    limb.logged_publish_count = true;
  }
  publish(cmds);
}

void JointCommandNode::publish(const std::vector<common_msgs::msg::MotorCmd>& cmds) {
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
