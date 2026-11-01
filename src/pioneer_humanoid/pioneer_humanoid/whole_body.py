"""Isaac Lab configs of the Pioneer whole-body humanoid (assets/pioneer_whole_body_humanoid), for learning to walk."""
import os

from isaaclab.actuators import ImplicitActuatorCfg

from .leg_only import _OUT_DIR, LEG_ONLY_HUMANOID_CFG, _isaac_urdf

# a closed fist: knuckles, middle and tip joints curled, the thumb folded across (all curl positive)
FIST = {
    f"{side}_{finger}{joint}": angle
    for side in ("l", "r")
    for finger in ("", "i_", "r_", "l_")
    for joint, angle in (("mcp", 1.4), ("pip", 1.5), ("dip", 1.2))
} | {f"{side}_th_{joint}": angle for side in ("l", "r") for joint, angle in (("c", 0.6), ("f", 0.9), ("mcp", 0.8), ("ip", 0.9))}


# The whole-body humanoid (arms and hands, 44.7 kg) walking on the same 12 leg joints, everything above the hips held rigid.
WHOLE_BODY_WALKER_CFG = LEG_ONLY_HUMANOID_CFG.replace(
    spawn=LEG_ONLY_HUMANOID_CFG.spawn.replace(
        asset_path=_isaac_urdf("pioneer_whole_body_humanoid", "pioneer_whole_body_humanoid.urdf", legs_only=True),
        usd_dir=os.path.join(_OUT_DIR, "..", "pioneer_whole_body_humanoid", "usd"),
    )
)

# The same with shoulder pitch and elbows free, for a scripted arm swing (humanoid_rl_tasks ... pioneer_leg_only/arm_swing.py).
ARM_SWING_JOINTS = ("l_shoulder_pitch", "r_shoulder_pitch", "l_elbow", "r_elbow")
WHOLE_BODY_ARM_SWING_CFG = WHOLE_BODY_WALKER_CFG.replace(
    spawn=WHOLE_BODY_WALKER_CFG.spawn.replace(
        asset_path=_isaac_urdf("pioneer_whole_body_humanoid", "pioneer_whole_body_humanoid.urdf", legs_only=True, keep=ARM_SWING_JOINTS,
                               name="pioneer_whole_body_humanoid_arm_swing", pose=FIST),
        usd_dir=os.path.join(_OUT_DIR, "..", "pioneer_whole_body_humanoid", "usd_arm_swing"),
    ),
    actuators={
        **WHOLE_BODY_WALKER_CFG.actuators,
        "arms": ImplicitActuatorCfg(
            joint_names_expr=list(ARM_SWING_JOINTS),
            effort_limit_sim={".*_shoulder_pitch": 53.0, ".*_elbow": 22.0},
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)
