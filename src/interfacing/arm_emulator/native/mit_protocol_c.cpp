// C entry points over the real src/interfacing/can/src/mit_protocol.cpp, so the emulator's drives use
// the arm's own packing code through ctypes instead of a second copy of it. Build: arm_emulator/mit_native.py.
#include <algorithm>

#include "mit_protocol.hpp"

extern "C" {
unsigned mit_pack_value(double phys, double min, double max, unsigned bits) {
  return packMitValue(phys, min, max, bits);
}

double mit_unpack_value(unsigned code, double min, double max, unsigned bits) {
  return unpackMitValue(code, min, max, bits);
}

void mit_pack_command(int is_ak, double p, double v, double kp, double kd, double t, double p_lim,
                      double v_lim, double t_lim, double kp_max, double kd_max, unsigned char* out) {
  MitProfile profile;
  profile.p_min = -p_lim;
  profile.p_max = p_lim;
  profile.v_min = -v_lim;
  profile.v_max = v_lim;
  profile.t_min = -t_lim;
  profile.t_max = t_lim;
  profile.kp_max = kp_max;
  profile.kd_max = kd_max;
  const auto payload = is_ak ? packAkMitCommand(p, v, kp, kd, t, profile)
                             : packMitCommand(p, v, kp, kd, t, profile);
  std::copy(payload.begin(), payload.end(), out);
}
}
