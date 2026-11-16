#pragma once

// CubeMars MIT ("Force Control") protocol helpers. Pure functions, unit-tested without a bus.
//   gl2 (GL40, GL II manual V1.0 s5): standard frame on the motor id; needs ENTER (FF..FC) first.
//   ak  (AK V3.0, manual V3.0.1 s4.2): extended frame akMitCanId(), KP-first byte order, no
//       special frames. Feedback is the servo status frame via the DBC.
//
// GL II command payload, 8 bytes:
//   byte 0: p[15:8]   byte 1: p[7:0]
//   byte 2: v[11:4]   byte 3: v[3:0]<<4 | kp[11:8]
//   byte 4: kp[7:0]   byte 5: kd[11:4]
//   byte 6: kd[3:0]<<4 | t[11:8]        byte 7: t[7:0]
// The drive then applies  torque = kp*(p_des - p) + kd*(v_des - v) + t_ff.
//
// GL II feedback payload, on the drive's master id (default 0x000):
//   byte 0: status[7:4] | (motor_id & 0xF)
//   byte 1-2: position(16)   byte 3 + byte4[7:4]: velocity(12)
//   byte4[3:0] + byte 5: torque(12)
//   byte 6: drive temp (degC, signed)   byte 7: motor temp (degC, signed)
//
// Classic-MIT AK feedback payload, on the master id (pre-V3 only; V3 sends the servo frame):
//   byte 0: driver id (FULL 8 bits -- unlike GL II, no status nibble)
//   byte 1-2: position(16)   byte 3 + byte4[7:4]: velocity(12)
//   byte4[3:0] + byte 5: torque(12)
//   byte 6: motor temp (degC, signed)   byte 7: error code (0 = no fault, see mitAkErrorName)
// From the manual only -- not yet checked against a bench capture.

#include <array>
#include <cstdint>
#include <string>

// Special frames: FF FF FF FF FF FF FF <code>.
inline constexpr uint8_t MIT_SPECIAL_ENTER = 0xFC;     // enter motor mode (enable)
inline constexpr uint8_t MIT_SPECIAL_EXIT = 0xFD;      // exit motor mode (motor goes limp)
inline constexpr uint8_t MIT_SPECIAL_SET_ZERO = 0xFE;  // set current position as zero
inline constexpr uint8_t MIT_SPECIAL_CLEAR_ERR = 0xFB; // clear errors

enum class MitFamily { Ak, Gl2 };

// Per-motor MIT (Force Control) protocol scaling constants -- see config/mit_profiles.yaml.
// Physical position/velocity/torque/kp/kd MIN..MAX for THIS motor model, used to pack
// floats into the MIT CAN frame's raw fixed-point fields. Differs per AK motor model
// (e.g. AK10-9 vs AK80-9 have different velocity/torque ranges).
struct MitProfile {
  double p_min{-12.5}, p_max{12.5};
  double v_min{-200.0}, v_max{200.0};
  double t_min{-10.0}, t_max{10.0};
  double kp_min{0.0}, kp_max{500.0};
  double kd_min{0.0}, kd_max{5.0};
  MitFamily family{MitFamily::Gl2};
  std::string model{};
  double kt{0.0}; // N.m/A at the output (ak only); 0 = unknown, torque not reported
};

// Decoded GL II feedback frame, in physical units.
struct MitFeedback {
  uint8_t id_nibble{0}; // low 4 bits of the motor's CAN id -- the frame carries no more
  uint8_t status{0};    // 0=Disable 1=Enable 8=over-voltage ... see mitStatusName()
  double position{0.0}; // rad
  double velocity{0.0}; // drive units (GL II: "r/s")
  double torque{0.0};   // N.m
  int drive_temp{0};    // degC
  int motor_temp{0};    // degC
};

// Decoded AK feedback frame, in physical units.
struct MitAkFeedback {
  uint8_t motor_id{0};  // full CAN id of the driver
  uint8_t error{0};     // 0 = no fault ... see mitAkErrorName()
  double position{0.0}; // rad
  double velocity{0.0}; // rad/s
  double torque{0.0};   // N.m
  int motor_temp{0};    // degC
};

// CubeMars AK-series manual (V3.2.0) section 4.2 float_to_uint:
// raw = (phys - min) * (2^bits / (max - min)). NOT (phys-min)/(max-min)*(2^bits - 1) --
// matches the manual's example code exactly (verified against its C reference impl).
// Unlike the manual, phys outside [min, max] (or NaN) throws std::out_of_range instead of being
// clamped: a clamped command moves the motor somewhere it was not told to go. At phys == max the
// formula gives 2^bits, which is clamped to 2^bits - 1 (it would otherwise wrap to the minimum).
uint32_t packMitValue(double phys, double min, double max, unsigned bits);

// Gains use the NEAREST code: truncation would send kp 0.61 as 0.488. Range [0, max]; outside
// (or NaN) throws std::out_of_range.
uint32_t packMitGain(double phys, double max, unsigned bits);

// Name of the first command field outside the profile's range (NaN counts), or nullptr if all
// fit. Lets can_node refuse a command with a reason instead of throwing.
const char* mitCommandOutOfRange(double p, double v, double kp, double kd, double t,
                                 const MitProfile& profile);

// Manual uint_to_float: code * span / (2^bits - 1) + min.
double unpackMitValue(uint32_t code, double min, double max, unsigned bits);

// Payload for a GL II MIT command frame, in the byte order documented above.
std::array<uint8_t, 8> packMitCommand(double p, double v, double kp, double kd, double t,
                                      const MitProfile& profile);

// AK V3 MIT frame id (extended): control mode 8 in bits 28..8, driver id in bits 7..0.
uint32_t akMitCanId(int drive_id);

// AK V3 MIT payload (manual V3.0.1 p.38), same field widths as GL II but KP-first:
//   byte 0: kp[11:4]   byte 1: kp[3:0]<<4 | kd[11:8]   byte 2: kd[7:0]
//   byte 3: p[15:8]    byte 4: p[7:0]
//   byte 5: v[11:4]    byte 6: v[3:0]<<4 | t[11:8]     byte 7: t[7:0]
std::array<uint8_t, 8> packAkMitCommand(double p, double v, double kp, double kd, double t,
                                        const MitProfile& profile);

// GL II feedback -> physical units. `data` must hold at least 8 bytes (null throws).
MitFeedback decodeGl2Feedback(const uint8_t* data, const MitProfile& profile);

// AK feedback -> physical units. `data` must hold at least 8 bytes (null throws).
MitAkFeedback decodeAkFeedback(const uint8_t* data, const MitProfile& profile);

std::array<uint8_t, 8> mitSpecialFrame(uint8_t code);

const char* mitStatusName(uint8_t status);

// AK error byte: 0 = no fault, 1 = motor over-temp, 2 = over-current, 3 = over-voltage,
// 4 = under-voltage, 5 = encoder fault, 6 = MOSFET over-temp, 7 = motor lock-up.
const char* mitAkErrorName(uint8_t error);

// Healthy status codes. Families disagree: GL II 1 = Enable, AK 1 = over-temperature.
bool mitStatusIsOk(uint8_t status, MitFamily family);

// "ak" / "gl2" (case-sensitive, as written in mit_profiles.yaml). Unknown -> false.
bool mitFamilyFromString(const std::string& name, MitFamily& out);
