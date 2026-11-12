"""Seeded reset parameters; baseline preserves the original RNG sequence."""
import colorsys
import numpy as np

PROFILES = ("baseline", "varied", "wide", "tray")
# Offsets from the original (0.50, 0.14) pickup anchor. Asymmetric bounds keep
# the approach/grasp/lift inside the left arm's joint-limited workspace.
WIDE_XY_LOW = (-.025, -.025)
WIDE_XY_HIGH = (.025, .01)
WIDE_SUPPORT_CENTER = (.50, .13)
WIDE_SUPPORT_SIZE = (.50, .30)
# Reachable side/diagonal placements; keep the near rim clear of the fixed bag.
TRAY_XY_LOW = (.25, .49)
TRAY_XY_HIGH = (.45, .53)
TRAY_EXPERT_YAW = float(np.deg2rad(25.))


def tray_expert_yaw(target_xy):
    """Keep the original grasp orientation nearby; turn only for farther goals."""
    x, y = target_xy
    return 0. if x <= .34 and y <= .50 else TRAY_EXPERT_YAW


def sample_reset(seed, profile="baseline"):
    if profile not in PROFILES:
        raise ValueError(f"Unknown cargo randomization profile: {profile}")
    rng = np.random.default_rng(seed)
    position_range = .005 if profile == "baseline" else .015
    yaw_range = .025 if profile == "baseline" else .04
    xy = (rng.uniform(WIDE_XY_LOW, WIDE_XY_HIGH) if profile == "wide"
          else rng.uniform(-position_range, position_range, 2))
    yaw = rng.uniform(-yaw_range, yaw_range)
    if profile == "tray":
        xy = np.zeros(2)
        yaw = 0.0
    joints = np.zeros(6)
    color = np.ones(3)
    if profile in ("varied", "wide", "tray"):
        joints = rng.uniform(-.06, .06, 6)
        # Keep the red handle visually distinct while varying body hue and shade.
        if rng.random() >= .15:
            color = np.array(colorsys.hsv_to_rgb(rng.uniform(.12, .78),
                                                rng.uniform(.25, .65), rng.uniform(.65, 1.)))
    params = {"profile": profile, "seed": int(seed), "pickup_offset_xy_m": xy.tolist(),
              "yaw_rad": float(yaw), "arm_joint_offset_rad": joints.tolist(),
              "body_color_rgb": color.tolist()}
    if profile == "wide":
        params.update(pickup_support_center_xy_m=list(WIDE_SUPPORT_CENTER),
                      pickup_support_size_xy_m=list(WIDE_SUPPORT_SIZE))
    if profile == "tray":
        params["target_xy_m"] = rng.uniform(TRAY_XY_LOW, TRAY_XY_HIGH).tolist()
        params["expert_place_yaw_rad"] = tray_expert_yaw(params["target_xy_m"])
    return rng, params
