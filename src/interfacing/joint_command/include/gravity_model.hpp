#pragma once

#include <array>

// Static gravity hold torque (N.m) of the LEFT arm (joint1L..joint6l, ArmPose order) from the
// URDF's CAD masses, base upright. q_urdf_rad is the URDF's convention, which the cmd frame
// matches (in radians).
std::array<double, 6> leftArmGravityHoldTorque(const std::array<double, 6>& q_urdf_rad);
