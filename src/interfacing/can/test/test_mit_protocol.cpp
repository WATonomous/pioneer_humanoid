// MIT packing/decoding vs the manuals' worked examples and a bench capture, plus the DBC layout
// (a reordered MIT field in humanoid.dbc fails here, not on the motor).
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

#include <dbcppp/Network.h>

#include "mit_protocol.hpp"

namespace {

// GL40 KV70 on a GL II drive, stock parameter page (config/mit_profiles.yaml id 22).
MitProfile gl40Profile() {
  MitProfile p;
  p.p_min = -12.5;
  p.p_max = 12.5;
  p.v_min = -200.0;
  p.v_max = 200.0;
  p.t_min = -10.0;
  p.t_max = 10.0;
  p.kp_min = 0.0;
  p.kp_max = 500.0;
  p.kd_min = 0.0;
  p.kd_max = 5.0;
  p.family = MitFamily::Gl2;
  p.model = "GL40-KV70";
  return p;
}

// AK80-9 as configured in config/mit_profiles.yaml (ids 11, 13, 104).
MitProfile ak809Profile() {
  MitProfile p;
  p.p_min = -12.56;
  p.p_max = 12.56;
  p.v_min = -65.0;
  p.v_max = 65.0;
  p.t_min = -18.0;
  p.t_max = 18.0;
  p.family = MitFamily::Ak;
  p.model = "AK80-9";
  return p;
}

} // namespace

// GL II manual section 5.5 "MIT position" worked example: p_des 2 rad, kp 0.123, kd 0.005
// packs to 94 7A 7F F0 01 00 47 FF. The upper computer emits 0x7FF for the zero velocity /
// torque codes where the manual's own float_to_uint gives 0x800; both decode to ~0, so those
// two fields are compared loosely and the rest exactly.
TEST(MitPacking, MatchesGlManualExample) {
  const auto p = gl40Profile();
  const auto got = packMitCommand(2.0, 0.0, 0.123, 0.005, 0.0, p);

  EXPECT_EQ(got[0], 0x94) << "position high byte";
  EXPECT_EQ(got[1], 0x7A) << "position low byte";
  EXPECT_EQ(got[3] & 0xF, 0x0) << "kp high nibble";
  EXPECT_EQ(got[4], 0x01) << "kp low byte";
  EXPECT_EQ(got[5], 0x00) << "kd high byte";
  EXPECT_EQ(got[6] >> 4, 0x4) << "kd low nibble";

  const uint32_t v_code = (static_cast<uint32_t>(got[2]) << 4) | (got[3] >> 4);
  const uint32_t t_code = (static_cast<uint32_t>(got[6] & 0xF) << 8) | got[7];
  EXPECT_TRUE(v_code == 0x7FF || v_code == 0x800) << "zero velocity code, got " << v_code;
  EXPECT_TRUE(t_code == 0x7FF || t_code == 0x800) << "zero torque code, got " << t_code;
}

// phys == max produces exactly 2^bits, which does not fit the field: unclamped it wraps to 0,
// i.e. a "go to maximum" command silently becomes "go to minimum".
TEST(MitPacking, ClampsAtTopOfRangeInsteadOfWrapping) {
  const auto p = gl40Profile();
  EXPECT_EQ(packMitValue(p.p_max, p.p_min, p.p_max, 16), 65535u);
  EXPECT_EQ(packMitValue(p.p_max * 10.0, p.p_min, p.p_max, 16), 65535u); // and beyond
  EXPECT_EQ(packMitValue(p.p_min, p.p_min, p.p_max, 16), 0u);
  EXPECT_EQ(packMitValue(p.t_max, p.t_min, p.t_max, 12), 4095u);
  EXPECT_EQ(packMitGain(p.kp_max, p.kp_max, 12), 4095u);
  EXPECT_EQ(packMitGain(p.kd_max, p.kd_max, 12), 4095u);
}

// Gains snap to the nearest code (truncation applied kp 0.61 as 0.488).
TEST(MitPacking, GainsSnapToNearestCode) {
  const auto p = gl40Profile();
  EXPECT_EQ(packMitGain(0.366, p.kp_max, 12), 3u);
  EXPECT_EQ(packMitGain(0.61, p.kp_max, 12), 5u);
  EXPECT_EQ(packMitGain(1.22, p.kp_max, 12), 10u);
  EXPECT_EQ(packMitGain(0.0098, p.kd_max, 12), 8u);
  EXPECT_EQ(packMitGain(0.0, p.kp_max, 12), 0u) << "zero gain must stay zero (limp)";
  EXPECT_EQ(packMitGain(-5.0, p.kp_max, 12), 0u) << "negative gain clamps to zero";
}

// Captured from the real GL40 at CAN id 22 on 2026-09-19.
TEST(MitDecode, DecodesBenchFeedbackCapture) {
  const uint8_t data[8] = {0x16, 0x99, 0x21, 0x7F, 0xE7, 0xFF, 0x28, 0x00};
  const auto fb = decodeGl2Feedback(data, gl40Profile());

  EXPECT_EQ(fb.id_nibble, 0x6);
  EXPECT_EQ(fb.status, 0x1);
  EXPECT_TRUE(mitStatusIsOk(fb.status, MitFamily::Gl2));
  EXPECT_STREQ(mitStatusName(fb.status), "Enable");
  EXPECT_NEAR(fb.position, 2.454, 0.002);
  EXPECT_NEAR(fb.velocity, 0.0, 0.2);
  EXPECT_NEAR(fb.torque, 0.0, 0.01);
  EXPECT_EQ(fb.drive_temp, 40);
  EXPECT_EQ(fb.motor_temp, 0);
}

TEST(MitDecode, ReportsFaultStatuses) {
  uint8_t data[8] = {0xA6, 0x99, 0x21, 0x7F, 0xE7, 0xFF, 0x28, 0x00}; // 0xA = over-current
  const auto fb = decodeGl2Feedback(data, gl40Profile());
  EXPECT_EQ(fb.status, 0xA);
  EXPECT_FALSE(mitStatusIsOk(fb.status, MitFamily::Gl2));
  EXPECT_STREQ(mitStatusName(fb.status), "Over-current");
}

// SYNTHETIC frame from the AK manual's layout, not a bench capture.
TEST(MitDecode, DecodesAkLayoutFromTheManual) {
  const auto p = ak809Profile();
  const uint32_t pos_i = packMitValue(1.25, p.p_min, p.p_max, 16);
  const uint32_t vel_i = packMitValue(-0.5, p.v_min, p.v_max, 12);
  const uint32_t t_i = packMitValue(3.0, p.t_min, p.t_max, 12);
  const uint8_t data[8] = {
      13,
      static_cast<uint8_t>(pos_i >> 8),
      static_cast<uint8_t>(pos_i & 0xFF),
      static_cast<uint8_t>(vel_i >> 4),
      static_cast<uint8_t>(((vel_i & 0xF) << 4) | (t_i >> 8)),
      static_cast<uint8_t>(t_i & 0xFF),
      35,
      0,
  };
  const auto fb = decodeAkFeedback(data, p);
  EXPECT_EQ(fb.motor_id, 13);
  EXPECT_NEAR(fb.position, 1.25, 25.12 / 65535.0);
  EXPECT_NEAR(fb.velocity, -0.5, 130.0 / 4095.0);
  EXPECT_NEAR(fb.torque, 3.0, 36.0 / 4095.0);
  EXPECT_EQ(fb.motor_temp, 35);
  EXPECT_EQ(fb.error, 0);
  EXPECT_TRUE(mitStatusIsOk(fb.error, MitFamily::Ak));
}

// Code 1: GL II "Enable" (healthy) vs AK "motor over-temp".
TEST(MitDecode, StatusOneIsAFaultOnAkButHealthyOnGl2) {
  EXPECT_TRUE(mitStatusIsOk(1, MitFamily::Gl2));
  EXPECT_FALSE(mitStatusIsOk(1, MitFamily::Ak));
  EXPECT_STREQ(mitAkErrorName(1), "Motor over-temperature");
  for (uint8_t e = 1; e <= 7; ++e) {
    EXPECT_FALSE(mitStatusIsOk(e, MitFamily::Ak)) << "AK error " << static_cast<int>(e);
  }
}

TEST(MitPacking, RoundTripsWithinOneCount) {
  const auto p = gl40Profile();
  for (const double rad : {-12.0, -1.5, 0.0, 0.7, 2.454, 11.9}) {
    const uint32_t code = packMitValue(rad, p.p_min, p.p_max, 16);
    const double back = unpackMitValue(code, p.p_min, p.p_max, 16);
    EXPECT_NEAR(back, rad, 25.0 / 65535.0) << "rad " << rad;
  }
}

TEST(MitSpecialFrames, MatchTheManualsMagicBytes) {
  EXPECT_EQ(mitSpecialFrame(MIT_SPECIAL_ENTER),
            (std::array<uint8_t, 8>{{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC}}));
  EXPECT_EQ(mitSpecialFrame(MIT_SPECIAL_EXIT)[7], 0xFD);
  EXPECT_EQ(mitSpecialFrame(MIT_SPECIAL_SET_ZERO)[7], 0xFE);
  EXPECT_EQ(mitSpecialFrame(MIT_SPECIAL_CLEAR_ERR)[7], 0xFB);
}

namespace {

// The drive in the AK manual V3.0.1 CAN examples (id 0x68): its frames only decode to the
// labelled values with p +-12.5 rad and v +-50 rad/s.
MitProfile akManualExampleProfile() {
  MitProfile p = ak809Profile();
  p.p_min = -12.5;
  p.p_max = 12.5;
  p.v_min = -50.0;
  p.v_max = 50.0;
  p.kp_max = 500.0;
  p.kd_max = 5.0;
  return p;
}

struct AkFields {
  uint32_t kp, kd, p, v, t;
};

AkFields akFields(const std::array<uint8_t, 8>& b) {
  return {(static_cast<uint32_t>(b[0]) << 4) | (b[1] >> 4),
          (static_cast<uint32_t>(b[1] & 0xF) << 8) | b[2],
          (static_cast<uint32_t>(b[3]) << 8) | b[4],
          (static_cast<uint32_t>(b[5]) << 4) | (b[6] >> 4),
          (static_cast<uint32_t>(b[6] & 0xF) << 8) | b[7]};
}

// The manual centres zero at 0x7FF/0x7FFF, float_to_uint at 0x800/0x8000 -- both ~0.
void expectWithinOneCount(uint32_t got, uint32_t want, const char* field) {
  EXPECT_LE(got > want ? got - want : want - got, 1u)
      << field << " got " << got << " want " << want;
}

} // namespace

TEST(AkMit, CanIdIsExtendedControlMode8) {
  EXPECT_EQ(akMitCanId(0x68), 0x868u) << "manual example id 00 00 08 68";
  EXPECT_EQ(akMitCanId(14), 0x80Eu) << "shoulder pitch";
  EXPECT_EQ(akMitCanId(0x1FF), 0x8FFu) << "drive id is only 8 bits";
}

// AK manual V3.0.1 p.53 "MIT Position Loop": Kp 2, Kd 2, rotate to 6 rad.
TEST(AkMit, PackingMatchesManualPositionExample) {
  const std::array<uint8_t, 8> manual{{0x01, 0x06, 0x66, 0xBD, 0x70, 0x7F, 0xF7, 0xFF}};
  const auto got = packAkMitCommand(6.0, 0.0, 2.0, 2.0, 0.0, akManualExampleProfile());

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(got[i], manual[i]) << "byte " << i << " (kp / kd / position)";
  }
  const AkFields g = akFields(got), m = akFields(manual);
  expectWithinOneCount(g.v, m.v, "velocity");
  expectWithinOneCount(g.t, m.t, "torque");
}

// AK manual V3.0.1 p.52 "MIT Velocity Loop": Kd 2, speed 6 rad/s.
TEST(AkMit, PackingMatchesManualVelocityExample) {
  const std::array<uint8_t, 8> manual{{0x00, 0x06, 0x66, 0x7F, 0xFF, 0x8F, 0x57, 0xFF}};
  const AkFields g = akFields(packAkMitCommand(0.0, 6.0, 0.0, 2.0, 0.0, akManualExampleProfile()));
  const AkFields m = akFields(manual);

  EXPECT_EQ(g.kp, m.kp);
  EXPECT_EQ(g.kd, m.kd);
  EXPECT_EQ(g.v, m.v) << "6 rad/s";
  expectWithinOneCount(g.p, m.p, "position");
  expectWithinOneCount(g.t, m.t, "torque");
}

// Same fields, different order: collapsing the two packers would send AK position bits as
// kp/kd (or GL II gains as position).
TEST(AkMit, LayoutDiffersFromGl2) {
  const auto p = akManualExampleProfile();
  const auto ak = packAkMitCommand(6.0, 0.0, 2.0, 2.0, 0.0, p);
  const auto gl = packMitCommand(6.0, 0.0, 2.0, 2.0, 0.0, p);
  EXPECT_NE(ak, gl);
  EXPECT_EQ(ak[3], gl[0]) << "AK position high byte sits where GL II puts it first";
  EXPECT_EQ(ak[4], gl[1]);
}

TEST(MitProfileParsing, KnowsBothFamilies) {
  MitFamily f{};
  EXPECT_TRUE(mitFamilyFromString("ak", f));
  EXPECT_EQ(f, MitFamily::Ak);
  EXPECT_TRUE(mitFamilyFromString("gl2", f));
  EXPECT_EQ(f, MitFamily::Gl2);
  EXPECT_FALSE(mitFamilyFromString("gl40", f));
  EXPECT_FALSE(mitFamilyFromString("", f));
}

// ---------------------------------------------------------------------------
// DBC layout: the frame can_node actually puts on the wire.
// ---------------------------------------------------------------------------

class DbcMitTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::ifstream idbc(DBC_PATH);
    ASSERT_TRUE(idbc.good()) << "cannot open " << DBC_PATH;
    net_ = dbcppp::INetwork::LoadDBCFromIs(idbc);
    ASSERT_NE(net_, nullptr) << "failed to parse " << DBC_PATH;
    for (const auto& msg : net_->Messages()) {
      if (msg.Name() == "MITControlCmd") {
        mit_ = &msg;
      }
    }
    ASSERT_NE(mit_, nullptr) << "MITControlCmd missing from the DBC";
  }

  const dbcppp::ISignal* signal(const std::string& name) const {
    for (const auto& sig : mit_->Signals()) {
      if (sig.Name() == name) {
        return &sig;
      }
    }
    return nullptr;
  }

  std::unique_ptr<dbcppp::INetwork> net_;
  const dbcppp::IMessage* mit_{nullptr};
};

// A GL II only listens to standard frames: MITControlCmd must not carry the extended flag the
// rest of this DBC uses.
TEST_F(DbcMitTest, MitCommandIsAStandardFrame) {
  EXPECT_LT(mit_->Id(), 0x80000000u) << "MITControlCmd must be a standard-id message";
  EXPECT_EQ(mit_->Id() & 0xFFu, 0u) << "low byte is the motor id, must be 0 in the DBC";
  EXPECT_EQ(mit_->MessageSize(), 8u);
}

// The DBC once laid the frame out kp, kd, pos, vel, t. Encoding through it must match the
// protocol helper: pos(16) vel(12) kp(12) kd(12) t(12).
TEST_F(DbcMitTest, SignalLayoutMatchesTheMitProtocol) {
  const auto p = gl40Profile();
  const auto expected = packMitCommand(2.0, 0.0, 0.123, 0.005, 0.0, p);

  std::array<uint8_t, 8> encoded{};
  encoded.fill(0);
  const struct {
    const char* name;
    uint32_t code;
  } fields[] = {
      {"MIT_Position", packMitValue(2.0, p.p_min, p.p_max, 16)},
      {"MIT_Velocity", packMitValue(0.0, p.v_min, p.v_max, 12)},
      {"MIT_KP", packMitGain(0.123, p.kp_max, 12)},
      {"MIT_KD", packMitGain(0.005, p.kd_max, 12)},
      {"MIT_Torque", packMitValue(0.0, p.t_min, p.t_max, 12)},
  };
  for (const auto& f : fields) {
    const dbcppp::ISignal* sig = signal(f.name);
    ASSERT_NE(sig, nullptr) << f.name << " missing from MITControlCmd";
    sig->Encode(sig->PhysToRaw(static_cast<int64_t>(f.code)), encoded.data());
  }

  EXPECT_EQ(encoded, expected) << "DBC signal placement disagrees with the MIT protocol byte order";
}

TEST_F(DbcMitTest, SignalsDecodeBackToTheCodesTheyWereGiven) {
  const auto p = gl40Profile();
  const auto frame = packMitCommand(2.0, 0.0, 1.22, 0.0098, 0.0, p);
  EXPECT_EQ(signal("MIT_Position")->Decode(frame.data()), packMitValue(2.0, p.p_min, p.p_max, 16));
  EXPECT_EQ(signal("MIT_KP")->Decode(frame.data()), 10u);
  EXPECT_EQ(signal("MIT_KD")->Decode(frame.data()), 8u);
}
