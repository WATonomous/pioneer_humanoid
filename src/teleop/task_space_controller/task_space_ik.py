"""
Task-space IK for the Pioneer bimanual arm 8dof (left arm only), sim only.

A cube in the scene is the absolute tool-point (TCP) pose target -- drag/rotate it in the
viewport and the left arm (7 revolute joints) follows via the Quest teleop's mink IK
(quest_controller_teleop/mink_control.py). The right arm and both grippers are held at their default poses.
"""

import argparse
import math
import os
import sys
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pioneer bimanual-arm task-space IK (cube target, left arm only)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pioneer_humanoid package (canonical arm config). Editable-installed in the image; this fallback
# keeps a bare bind-mounted checkout working.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../pioneer_humanoid")
))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../quest_controller_teleop")))   # mink_control

from mink_control import PioneerMinkIK  # noqa: E402
from pioneer_humanoid.arm_8dof_params import (  # noqa: E402
    LEFT_ARM_JOINTS,
    LEFT_EE_BODY,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
    RIGHT_ARM_JOINTS,
)
from pioneer_humanoid.bimanual_arm_8dof import BIMANUAL_ARM_8DOF_CFG, tcp_pose_w  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
import torch  # noqa: E402


def mink_target(robot, pivot_body_id, ee_body_id, tcp_pos_b, tcp_quat_b):
    """PioneerMinkIK's target for a base-frame TCP pose: it tracks the wrist pivot (link6's origin) and the gripper's
    rotation, so the pivot goes where it puts the TCP on target (the measured pivot -> TCP offset, gripper frame)."""
    tcp_w, quat_w = tcp_pose_w(robot, ee_body_id)
    offset = quat_apply(quat_inv(quat_w), tcp_w - robot.data.body_pos_w[:, pivot_body_id])
    pivot_b = tcp_pos_b - quat_apply(tcp_quat_b, offset)
    return SimpleNamespace(active=True, saturation=0.0, position=pivot_b[0].double().cpu().numpy(),
                           rotation=matrix_from_quat(tcp_quat_b)[0].double().cpu().numpy())


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    cube = AssetBaseCfg(
        prim_path="/World/cube",
        spawn=sim_utils.CuboidCfg(size=[0.1, 0.1, 0.1]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.26, -0.23, 0.15)),
    )

    robot = BIMANUAL_ARM_8DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    scene.update(sim_dt)
    print(f"[INFO] Robot joints: {robot.data.joint_names}")
    print(f"[INFO] Left arm IK joints: {LEFT_ARM_JOINTS}")
    print(f"[INFO] IK tracks the tool point (arm_8dof_params.TCP_LOCAL on {LEFT_EE_BODY})")

    # mink IK as the Quest teleop (scene_driver.py): its speed caps and the elbow >= 10 deg rule.
    ik = PioneerMinkIK(LEFT_ARM_JOINTS, velocity=0.9, wrist_velocity=3.6)
    limits = ik.physical_limits.copy()
    limits[3, 0] = max(limits[3, 0], math.radians(10.0))
    ik.set_teleop_limits(limits)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    ee_body_id = robot.find_bodies(LEFT_EE_BODY)[0][0]
    pivot_body_id = robot.find_bodies("left_link6")[0][0]
    left_arm_ids = robot.find_joints(LEFT_ARM_JOINTS, preserve_order=True)[0]
    left_gripper_ids = robot.find_joints(LEFT_GRIPPER_JOINTS, preserve_order=True)[0]
    right_arm_ids = robot.find_joints(RIGHT_ARM_JOINTS, preserve_order=True)[0]
    right_default_pos = robot.data.default_joint_pos[:, right_arm_ids].clone()

    gripper_open_targets = torch.tensor(
        [[LEFT_GRIPPER_OPEN[name] for name in LEFT_GRIPPER_JOINTS]],
        device=sim.device,
    )

    joint_position = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_position, joint_vel)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    # Read the cube from its USD transform, not the physics tensor -- an AssetBaseCfg prim's
    # tensor pose doesn't update on a viewport drag.
    _cube_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/cube")
    _cube_xf = UsdGeom.Xformable(_cube_prim)

    def _cube_pose_w():
        m = _cube_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        im = q.GetImaginary()
        pos = torch.tensor([[t[0], t[1], t[2]]], dtype=torch.float32, device=sim.device)
        quat = torch.tensor([[q.GetReal(), im[0], im[1], im[2]]], dtype=torch.float32, device=sim.device)
        return pos, quat

    # Co-locate the cube with the gripper at launch (position only; orientation stays identity).
    _tp_w, _ = tcp_pose_w(robot, ee_body_id)
    _p = _tp_w[0].tolist()
    _cube_xf.ClearXformOpOrder()
    _cube_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(_p[0], _p[1], _p[2]))
    sim.step()
    scene.update(sim_dt)

    # link7's frame isn't identity, so the IK target is (cube rotation since launch) applied
    # to the gripper's rest orientation -- cube at rest -> gripper at rest, and a world-axis
    # cube rotation is a world-axis gripper rotation.
    _root0 = robot.data.root_state_w[:, 0:7]
    _cp0_w, _cq0_w = _cube_pose_w()
    _, _cube_q0_b = subtract_frame_transforms(_root0[:, 0:3], _root0[:, 3:7], _cp0_w, _cq0_w)
    _, _wrist_q0_b = subtract_frame_transforms(_root0[:, 0:3], _root0[:, 3:7], *tcp_pose_w(robot, ee_body_id))
    _cube_q0_b_inv = quat_inv(_cube_q0_b)

    while simulation_app.is_running():
        cube_pos_w, cube_quat_w = _cube_pose_w()
        root_pose_w = robot.data.root_state_w[:, 0:7]
        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], cube_pos_w, cube_quat_w
        )
        joint_pos = robot.data.joint_pos[:, left_arm_ids]

        d_cube_b = quat_mul(cube_quat_b, _cube_q0_b_inv)
        target_quat_b = quat_mul(d_cube_b, _wrist_q0_b)
        q = ik.solve(joint_pos[0].double().cpu().numpy(),
                     mink_target(robot, pivot_body_id, ee_body_id, cube_pos_b, target_quat_b), sim_dt)
        joint_pos_des = torch.tensor(q, dtype=torch.float32, device=sim.device).unsqueeze(0)
        robot.set_joint_position_target(joint_pos_des, joint_ids=left_arm_ids)

        # Hold left gripper open; right arm + gripper at default pose
        robot.set_joint_position_target(gripper_open_targets, joint_ids=left_gripper_ids)
        robot.set_joint_position_target(right_default_pos, joint_ids=right_arm_ids)
        robot.set_joint_velocity_target(
            torch.zeros(1, len(left_gripper_ids), device=sim.device),
            joint_ids=left_gripper_ids,
        )

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        tip_pos_w, tip_quat_w = tcp_pose_w(robot, ee_body_id)
        ee_marker.visualize(tip_pos_w, tip_quat_w)
        goal_marker.visualize(cube_pos_w, cube_quat_w)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])

    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
