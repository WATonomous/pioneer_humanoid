#pragma once

// Startup checks on params.yaml's upper/lower motor_ids. Header-only and ROS-free so it can be
// unit tested without a bus.

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

// MotorCmd.motor_id and MotorFeedback.motor_id are int8: a larger id wraps negative and can never
// be matched to its bus or its feedback.
constexpr int64_t kMaxRoutableMotorId = 127;

struct BusRoutingReport {
  std::vector<int64_t> invalid_ids;   // outside [0, kMaxRoutableMotorId]
  std::vector<int64_t> duplicate_ids; // listed on both buses (or twice on one)
  std::vector<int> unrouted_mit_ids;  // have an MIT profile but are on neither bus

  // invalid_ids and duplicate_ids are config errors: the node must not start with them.
  // unrouted_mit_ids is reported but allowed (a profile for a drive not fitted yet).
  bool hasErrors() const { return !invalid_ids.empty() || !duplicate_ids.empty(); }
};

inline BusRoutingReport checkBusRouting(const std::vector<int64_t>& upper_ids,
                                        const std::vector<int64_t>& lower_ids,
                                        const std::vector<int>& mit_profile_ids) {
  BusRoutingReport report;
  std::set<int64_t> seen;
  std::set<int64_t> duplicates;
  for (const auto* ids : {&upper_ids, &lower_ids}) {
    for (const int64_t id : *ids) {
      if (id < 0 || id > kMaxRoutableMotorId) {
        report.invalid_ids.push_back(id);
        continue;
      }
      if (!seen.insert(id).second) {
        duplicates.insert(id);
      }
    }
  }
  report.duplicate_ids.assign(duplicates.begin(), duplicates.end());
  for (const int id : mit_profile_ids) {
    if (seen.count(id) == 0) {
      report.unrouted_mit_ids.push_back(id);
    }
  }
  std::sort(report.unrouted_mit_ids.begin(), report.unrouted_mit_ids.end());
  return report;
}
