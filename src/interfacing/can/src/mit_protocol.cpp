#include "mit_protocol.hpp"

#include <algorithm>
#include <cmath>

namespace {
constexpr uint32_t maxCode(unsigned bits) {
  return (1u << bits) - 1u;
}
} // namespace

uint32_t packMitValue(double phys, double min, double max, unsigned bits) {
  phys = std::clamp(phys, min, max);
  const double span = max - min;
  if (span <= 0.0) {
    return 0;
  }
  const double scale = static_cast<double>(1u << bits) / span;
  const double raw = (phys - min) * scale;
  // Truncate (manual parity), then clamp: phys == max yields exactly 2^bits, which does not
  // fit the field and would wrap to 0.
  return std::min(static_cast<uint32_t>(raw), maxCode(bits));
}

uint32_t packMitGain(double phys, double max, unsigned bits) {
  if (max <= 0.0) {
    return 0;
  }
  phys = std::clamp(phys, 0.0, max);
  const double scale = static_cast<double>(1u << bits) / max;
  const double raw = std::lround(phys * scale);
  return static_cast<uint32_t>(std::min(std::max(raw, 0.0), static_cast<double>(maxCode(bits))));
}

double unpackMitValue(uint32_t code, double min, double max, unsigned bits) {
  const double span = max - min;
  return static_cast<double>(code) * span / static_cast<double>(maxCode(bits)) + min;
}

std::array<uint8_t, 8> packMitCommand(double p, double v, double kp, double kd, double t,
                                      const MitProfile& profile) {
  const uint32_t p_i = packMitValue(p, profile.p_min, profile.p_max, 16);
  const uint32_t v_i = packMitValue(v, profile.v_min, profile.v_max, 12);
  const uint32_t kp_i = packMitGain(kp, profile.kp_max, 12);
  const uint32_t kd_i = packMitGain(kd, profile.kd_max, 12);
  const uint32_t t_i = packMitValue(t, profile.t_min, profile.t_max, 12);
  return {{
      static_cast<uint8_t>((p_i >> 8) & 0xFF),
      static_cast<uint8_t>(p_i & 0xFF),
      static_cast<uint8_t>((v_i >> 4) & 0xFF),
      static_cast<uint8_t>(((v_i & 0xF) << 4) | ((kp_i >> 8) & 0xF)),
      static_cast<uint8_t>(kp_i & 0xFF),
      static_cast<uint8_t>((kd_i >> 4) & 0xFF),
      static_cast<uint8_t>(((kd_i & 0xF) << 4) | ((t_i >> 8) & 0xF)),
      static_cast<uint8_t>(t_i & 0xFF),
  }};
}

uint32_t akMitCanId(int drive_id) {
  return (8u << 8) | (static_cast<uint32_t>(drive_id) & 0xFFu);
}

std::array<uint8_t, 8> packAkMitCommand(double p, double v, double kp, double kd, double t,
                                        const MitProfile& profile) {
  const uint32_t p_i = packMitValue(p, profile.p_min, profile.p_max, 16);
  const uint32_t v_i = packMitValue(v, profile.v_min, profile.v_max, 12);
  const uint32_t kp_i = packMitGain(kp, profile.kp_max, 12);
  const uint32_t kd_i = packMitGain(kd, profile.kd_max, 12);
  const uint32_t t_i = packMitValue(t, profile.t_min, profile.t_max, 12);
  return {{
      static_cast<uint8_t>((kp_i >> 4) & 0xFF),
      static_cast<uint8_t>(((kp_i & 0xF) << 4) | ((kd_i >> 8) & 0xF)),
      static_cast<uint8_t>(kd_i & 0xFF),
      static_cast<uint8_t>((p_i >> 8) & 0xFF),
      static_cast<uint8_t>(p_i & 0xFF),
      static_cast<uint8_t>((v_i >> 4) & 0xFF),
      static_cast<uint8_t>(((v_i & 0xF) << 4) | ((t_i >> 8) & 0xF)),
      static_cast<uint8_t>(t_i & 0xFF),
  }};
}

MitFeedback decodeGl2Feedback(const uint8_t* data, const MitProfile& profile) {
  MitFeedback fb;
  if (data == nullptr) {
    return fb;
  }
  fb.status = static_cast<uint8_t>(data[0] >> 4);
  fb.id_nibble = static_cast<uint8_t>(data[0] & 0xF);
  const uint32_t pos_i = (static_cast<uint32_t>(data[1]) << 8) | data[2];
  const uint32_t vel_i = (static_cast<uint32_t>(data[3]) << 4) | (data[4] >> 4);
  const uint32_t t_i = (static_cast<uint32_t>(data[4] & 0xF) << 8) | data[5];
  fb.position = unpackMitValue(pos_i, profile.p_min, profile.p_max, 16);
  fb.velocity = unpackMitValue(vel_i, profile.v_min, profile.v_max, 12);
  fb.torque = unpackMitValue(t_i, profile.t_min, profile.t_max, 12);
  fb.drive_temp = static_cast<int8_t>(data[6]);
  fb.motor_temp = static_cast<int8_t>(data[7]);
  return fb;
}

int gl2ReportedTemperature(const MitFeedback& fb) {
  return std::max(fb.drive_temp, fb.motor_temp);
}

MitAkFeedback decodeAkFeedback(const uint8_t* data, const MitProfile& profile) {
  MitAkFeedback fb;
  if (data == nullptr) {
    return fb;
  }
  fb.motor_id = data[0];
  const uint32_t pos_i = (static_cast<uint32_t>(data[1]) << 8) | data[2];
  const uint32_t vel_i = (static_cast<uint32_t>(data[3]) << 4) | (data[4] >> 4);
  const uint32_t t_i = (static_cast<uint32_t>(data[4] & 0xF) << 8) | data[5];
  fb.position = unpackMitValue(pos_i, profile.p_min, profile.p_max, 16);
  fb.velocity = unpackMitValue(vel_i, profile.v_min, profile.v_max, 12);
  fb.torque = unpackMitValue(t_i, profile.t_min, profile.t_max, 12);
  fb.motor_temp = static_cast<int8_t>(data[6]);
  fb.error = data[7];
  return fb;
}

std::array<uint8_t, 8> mitSpecialFrame(uint8_t code) {
  return {{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, code}};
}

const char* mitStatusName(uint8_t status) {
  switch (status) {
  case 0x0:
    return "Disable";
  case 0x1:
    return "Enable";
  case 0x8:
    return "Over-voltage";
  case 0x9:
    return "Under-voltage";
  case 0xA:
    return "Over-current";
  case 0xB:
    return "MOSFET over-temperature";
  case 0xC:
    return "Winding over-temperature";
  case 0xD:
    return "Communication loss";
  case 0xE:
    return "Overload";
  default:
    return "Unknown";
  }
}

const char* mitAkErrorName(uint8_t error) {
  switch (error) {
  case 0:
    return "No fault";
  case 1:
    return "Motor over-temperature";
  case 2:
    return "Over-current";
  case 3:
    return "Over-voltage";
  case 4:
    return "Under-voltage";
  case 5:
    return "Encoder fault";
  case 6:
    return "MOSFET over-temperature";
  case 7:
    return "Motor lock-up";
  default:
    return "Unknown";
  }
}

bool mitStatusIsOk(uint8_t status, MitFamily family) {
  if (family == MitFamily::Ak) {
    return status == 0;
  }
  return status == 0x0 || status == 0x1;
}

bool mitFamilyFromString(const std::string& name, MitFamily& out) {
  if (name == "ak") {
    out = MitFamily::Ak;
    return true;
  }
  if (name == "gl2") {
    out = MitFamily::Gl2;
    return true;
  }
  return false;
}
