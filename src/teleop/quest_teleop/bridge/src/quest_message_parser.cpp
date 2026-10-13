#include "quest_teleop/quest_message_parser.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

void fill_pose(const json& pose_json, geometry_msgs::msg::Pose& pose) {
  if (pose_json.contains("position")) {
    const auto& p = pose_json.at("position");
    pose.position.x = p.value("x", 0.0);
    pose.position.y = p.value("y", 0.0);
    pose.position.z = p.value("z", 0.0);
  }

  if (pose_json.contains("orientation")) {
    const auto& q = pose_json.at("orientation");
    pose.orientation.x = q.value("x", 0.0);
    pose.orientation.y = q.value("y", 0.0);
    pose.orientation.z = q.value("z", 0.0);
    pose.orientation.w = q.value("w", 1.0);
  }
}

void fill_float_array(const json& source, std::vector<float>& target) {
  target.clear();
  target.reserve(source.size());

  for (const auto& value : source) {
    target.push_back(value.get<float>());
  }
}

} // namespace

common_msgs::msg::QuestHandPose QuestMessageParser::make_empty_message() {
  common_msgs::msg::QuestHandPose msg;

  msg.left_wrist.orientation.w = 1.0;
  msg.right_wrist.orientation.w = 1.0;
  msg.head_pose.orientation.w = 1.0;

  msg.left_hand_joints.resize(HAND_ARRAY_SIZE, 0.0f);
  msg.right_hand_joints.resize(HAND_ARRAY_SIZE, 0.0f);

  return msg;
}

common_msgs::msg::QuestHandPose QuestMessageParser::parse(const std::string& json_text) {
  auto msg = make_empty_message();

  const auto data = json::parse(json_text);

  if (data.contains("left_wrist")) {
    fill_pose(data.at("left_wrist"), msg.left_wrist);
  }

  if (data.contains("right_wrist")) {
    fill_pose(data.at("right_wrist"), msg.right_wrist);
  }

  if (data.contains("head_pose")) {
    fill_pose(data.at("head_pose"), msg.head_pose);
  }

  if (data.contains("left_hand_joints")) {
    fill_float_array(data.at("left_hand_joints"), msg.left_hand_joints);
  }

  if (data.contains("right_hand_joints")) {
    fill_float_array(data.at("right_hand_joints"), msg.right_hand_joints);
  }

  // Controller triggers (0..1); absent under hand tracking, so they stay 0.
  msg.left_trigger = data.value("left_trigger", 0.0f);
  msg.right_trigger = data.value("right_trigger", 0.0f);
  msg.reset_button = data.value("reset_button", false);
  msg.save_button = data.value("save_button", false);

  return msg;
}