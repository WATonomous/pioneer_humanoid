#pragma once

#include <array>

// Static gravity hold torque (N.m) of one arm (ArmPose order) from the URDF's CAD masses, base
// upright. q_urdf_rad is the URDF's convention, which the cmd frame matches (in radians).

// LEFT arm, joint1L..joint6l.
std::array<double, 6> leftArmGravityHoldTorque(const std::array<double, 6>& q_urdf_rad);

// RIGHT arm, joint1..joint6.
std::array<double, 6> rightArmGravityHoldTorque(const std::array<double, 6>& q_urdf_rad);
