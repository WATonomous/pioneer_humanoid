#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_msgs/msg/arm_pose.hpp"
#include "common_msgs/msg/motor_cmd.hpp"
#include "common_msgs/msg/motor_feedback.hpp"
#include "joint_command_core.hpp"
#include "rclcpp/rclcpp.hpp"

class JointCommandNode : public rclcpp::Node {
public:
  JointCommandNode();

  // Pre-shutdown: damp Damp joints for mit_shutdown_damp_sec, then MIT_EXIT every MIT joint.
  void shutdownMitJoints();

private:
  void armPoseCallback(const common_msgs::msg::ArmPose::SharedPtr msg);
  void motorFeedbackCallback(const common_msgs::msg::MotorFeedback::SharedPtr msg);
  void controlTimerCallback();
  void publishMotorCommands(const std::vector<common_msgs::msg::MotorCmd>& cmds);
  void logGravityModel();
  bool trySeedFromFeedback();
  void mitFault(const std::string& reason);
  // MIT_EXIT (sent 3x) to every MIT joint, or only the Limp ones.
  void exitMitJoints(bool limp_only);
  std::map<int, MotorFeedbackSample> mitFeedbackSamples();

  JointCommandCore core_;
  rclcpp::Subscription<common_msgs::msg::ArmPose>::SharedPtr arm_pose_sub_;
  rclcpp::Subscription<common_msgs::msg::MotorFeedback>::SharedPtr feedback_sub_;
  rclcpp::Publisher<common_msgs::msg::MotorCmd>::SharedPtr motor_cmd_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  int8_t control_type_{common_msgs::msg::MotorCmd::POSITION_LOOP};
  double control_rate_hz_{50.0};
  // Latest ArmPose; moderation runs on the control tick, so velocity_max is per second.
  common_msgs::msg::ArmPose latest_pose_;
  bool have_latest_pose_{false};

  // Until every joint's real position has been seen and used to seed the rate-limiter,
  // ArmPose messages are ignored -- otherwise the limiter would ramp from an assumed 0
  // and command a large first step (slam) on any arm not physically at 0.
  std::map<int, double> latest_feedback_;
  std::map<int, MotorFeedbackSample> latest_feedback_full_;
  std::map<int, rclcpp::Time> latest_feedback_time_;
  bool seeded_from_feedback_{false};
  bool logged_publish_count_{false};
  int unseeded_ticks_{0}; // paces the MIT_ENTER re-send while unseeded
  // Latched until restart: drives are freed and nothing is published.
  bool mit_faulted_{false};

  // Command staleness watchdog: without this, have_latest_pose_/seeded_from_feedback_
  // stay true for the node's entire lifetime once set, so control_timer_ keeps
  // republishing whatever ArmPose target it last received FOREVER -- including across a
  // can_node/interfacing restart, causing an instant snap to a stale target the moment
  // CAN communication resumes (observed directly: arm jumped to a target from a much
  // earlier run the instant the interfacing container came back up). If no fresh ArmPose
  // arrives within command_timeout_sec_, stop publishing AND clear seeded_from_feedback_,
  // so the next real command must re-seed from fresh feedback and ramp safely again,
  // exactly like a first-ever command after node startup.
  double command_timeout_sec_{10.0};
  double mit_shutdown_damp_sec_{2.0};
  rclcpp::Time last_armpose_time_;
  bool have_armpose_time_{false};
};
