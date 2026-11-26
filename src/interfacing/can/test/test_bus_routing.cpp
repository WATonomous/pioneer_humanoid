#include "bus_routing.hpp"

#include <gtest/gtest.h>

TEST(BusRouting, CleanConfigHasNoFindings) {
  const auto r = checkBusRouting({30, 31, 32}, {40, 41}, {30, 31, 32, 40});
  EXPECT_FALSE(r.hasErrors());
  EXPECT_TRUE(r.invalid_ids.empty());
  EXPECT_TRUE(r.duplicate_ids.empty());
  EXPECT_TRUE(r.unrouted_mit_ids.empty());
}

TEST(BusRouting, IdOnBothBusesIsAnError) {
  // Without this check the lower bus silently wins and the upper drive never hears its commands.
  const auto r = checkBusRouting({30, 42}, {40, 42}, {});
  EXPECT_TRUE(r.hasErrors());
  ASSERT_EQ(r.duplicate_ids.size(), 1u);
  EXPECT_EQ(r.duplicate_ids[0], 42);
}

TEST(BusRouting, IdListedTwiceOnOneBusIsReportedOnce) {
  const auto r = checkBusRouting({30, 30, 30}, {}, {});
  ASSERT_EQ(r.duplicate_ids.size(), 1u);
  EXPECT_EQ(r.duplicate_ids[0], 30);
}

TEST(BusRouting, IdThatDoesNotFitInt8IsAnError) {
  // MotorCmd.motor_id is int8: 200 arrives as -56 and never matches its bus entry.
  const auto r = checkBusRouting({200}, {-1}, {});
  EXPECT_TRUE(r.hasErrors());
  ASSERT_EQ(r.invalid_ids.size(), 2u);
  EXPECT_EQ(r.invalid_ids[0], 200);
  EXPECT_EQ(r.invalid_ids[1], -1);
  EXPECT_TRUE(r.duplicate_ids.empty());
}

TEST(BusRouting, BoundaryIdsAreValid) {
  const auto r = checkBusRouting({0}, {kMaxRoutableMotorId}, {});
  EXPECT_FALSE(r.hasErrors());
}

TEST(BusRouting, MitProfileOnNeitherBusIsReportedButNotAnError) {
  // The case that hit the left wrist: MIT profile "1" exists, but 1 is on no bus, so every
  // command to it was dropped at runtime with only a throttled log line.
  const auto r = checkBusRouting({30, 31}, {40}, {40, 1, 31, 35});
  EXPECT_FALSE(r.hasErrors());
  ASSERT_EQ(r.unrouted_mit_ids.size(), 2u);
  EXPECT_EQ(r.unrouted_mit_ids[0], 1);
  EXPECT_EQ(r.unrouted_mit_ids[1], 35);
}
