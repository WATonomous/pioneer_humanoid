"""Shared Isaac cargo environment for generation, evaluation and human correction.

Import after AppLauncher. Cargo root state is written only by reset(). All
episode motion is produced by the real robot's joint/finger actuators.
"""
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from humanoid_isaac_scenes import make_scene_cfg
from humanoid_isaac_scenes.nasa_cargo.scene import TARGET_XY, CARGO_POS, CARGO_ROT
from humanoid_isaac_scenes.nasa_cargo.task import CargoTask
from pioneer_humanoid.bimanual_arm import (
    BIMANUAL_ARM_CFG, LEFT_ARM_JOINTS, LEFT_GRIPPER_JOINTS, LEFT_GRIPPER_OPEN,
    LEFT_GRIPPER_CLOSED, LEFT_FINGER_TIP_BODIES, LEFT_FINGER_DISTAL_TIP_LOCAL,
    compute_gripper_tip_pose_w, resolve_joint_name, apply_joint_limits, make_camera_cfg,
)

FPS = 25
TASK = "pick up the cargo transfer bag by its red handle and place it in the green tray"
CAMERAS = ("ego", "wrist_left")
IMAGE_HEIGHT, IMAGE_WIDTH = 240, 320


class CargoEnv:
    def __init__(self, device="cuda", cameras=True, randomization="baseline"):
        from humanoid_robot_learning.sim.cargo_randomization import PROFILES
        if randomization not in PROFILES:
            raise ValueError(f"Unknown cargo randomization: {randomization}")
        self.randomization = randomization
        self.reset_metadata = None
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01, device=device))
        self.device = self.sim.device
        cfg = make_scene_cfg("nasa_cargo", BIMANUAL_ARM_CFG)
        if randomization == "wide":
            from humanoid_robot_learning.sim.cargo_randomization import WIDE_SUPPORT_CENTER, WIDE_SUPPORT_SIZE
            # Keep its upper edge below the tray rim; extend toward the robot.
            # This is a reset-domain variation, recorded alongside each sample.
            height = cfg.pickup_station.spawn.size[2]
            cfg.pickup_station = cfg.pickup_station.replace(
                init_state=cfg.pickup_station.init_state.replace(pos=(*WIDE_SUPPORT_CENTER, height/2)),
                spawn=cfg.pickup_station.spawn.replace(size=(*WIDE_SUPPORT_SIZE, height)))
        self.target_parts = ("target_station", "target_front", "target_back", "target_left", "target_right")
        self.target_xy = np.array(TARGET_XY)
        if randomization == "tray":
            from isaaclab.assets import RigidObjectCfg
            # Kinematic colliders move with the visual tray on every reset.
            for name in self.target_parts:
                part = getattr(cfg, name)
                setattr(cfg, name, RigidObjectCfg(
                    prim_path=part.prim_path,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=part.init_state.pos, rot=part.init_state.rot),
                    spawn=part.spawn.replace(
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.))))
        if cameras:
            cfg.ego = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Ego", width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18., horizontal_aperture=24., clipping_range=(.03, 10.)),
            )
            cfg.wrist_left = make_camera_cfg("wrist_left", IMAGE_HEIGHT, IMAGE_WIDTH)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.scene.update(.01)
        self.robot, self.cargo = self.scene["robot"], self.scene["cargo"]
        apply_joint_limits(self.robot)
        self.arm_ids = [self.robot.joint_names.index(resolve_joint_name(self.robot, j)) for j in LEFT_ARM_JOINTS]
        self.grip_ids = [self.robot.joint_names.index(resolve_joint_name(self.robot, j)) for j in LEFT_GRIPPER_JOINTS]
        self.wrist_id = self.robot.body_names.index("link6l")
        self.finger_ids = [self.robot.body_names.index(j) for j in LEFT_FINGER_TIP_BODIES]
        self.g_open = torch.tensor([[LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS]], device=self.device)
        self.g_closed = torch.tensor([[LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS]], device=self.device)
        self.limits = self.robot.data.joint_pos_limits[0, self.arm_ids].cpu().numpy()
        self.home = self.robot.data.default_joint_pos[0, self.arm_ids].cpu().numpy().copy()
        self.task = CargoTask(self.scene, moving_target=randomization == "tray")
        self.cameras = bool(cameras)
        if cameras:
            self.scene["ego"].set_world_poses_from_view(
                torch.tensor([[.90, 1.20, 1.70]], device=self.device),
                torch.tensor([[.38, .30, .87]], device=self.device))
        self.sim.set_camera_view([1.4, 1.4, 1.8], [.4, .3, .9])
        self.last = np.r_[self.home, 0.].astype(np.float32)
        self.status = None

    def contract(self):
        import hashlib
        from humanoid_isaac_scenes.nasa_cargo.scene import CARGO_USD, PICKUP_XY, TARGET_HALF_XY, TABLE_TOP_Z
        return {
            "version": 1, "scene": "nasa_cargo", "layout": "left_arm_v2",
            "fps": FPS, "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT,
            "cameras": list(CAMERAS), "ego_eye": [.90, 1.20, 1.70], "ego_target": [.38, .30, .87],
            "action_names": [*LEFT_ARM_JOINTS, "left_gripper_closure"],
            "action_space": "absolute_joint_radians_and_gripper_0_1",
            "pickup_xy": list(PICKUP_XY), "target_xy": list(TARGET_XY),
            "target_half_xy": list(TARGET_HALF_XY), "table_top_z": TABLE_TOP_Z,
            "cargo_sha256": hashlib.sha256(CARGO_USD.read_bytes()).hexdigest(),
            "home": [float(q) for q in self.home], "task": TASK,
        }

    def reset(self, seed):
        from humanoid_robot_learning.sim.cargo_randomization import sample_reset
        rng, self.reset_metadata = sample_reset(seed, self.randomization)
        self.target_xy = np.array(self.reset_metadata.get("target_xy_m", TARGET_XY))
        if self.randomization == "tray":
            for name in self.target_parts:
                part = self.scene[name]
                tray_state = part.data.default_root_state.clone()
                tray_state[:, :2] += tray_state.new_tensor(self.target_xy - TARGET_XY)
                tray_state[:, 7:] = 0
                part.write_root_state_to_sim(tray_state)
                part.reset()
        joint_pos = self.robot.data.default_joint_pos.clone()
        start = np.clip(self.home + self.reset_metadata["arm_joint_offset_rad"], self.limits[:, 0], self.limits[:, 1])
        joint_pos[:, self.arm_ids] = torch.as_tensor(start, dtype=joint_pos.dtype, device=self.device)
        self.reset_metadata["initial_arm_joint_targets"] = start.tolist()
        self.robot.write_joint_state_to_sim(joint_pos, self.robot.data.default_joint_vel.clone())
        self.robot.reset()
        state = self.cargo.data.default_root_state.clone()
        state[:, :2] += state.new_tensor(self.reset_metadata["pickup_offset_xy_m"])
        yaw = self.reset_metadata["yaw_rad"]
        state[:, 3:7] = state.new_tensor([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        self.cargo.write_root_state_to_sim(state)
        self.cargo.reset()
        if self.randomization in ("varied", "wide", "tray"):
            from isaaclab.sim.utils import get_current_stage
            from pxr import Gf
            stage = get_current_stage()
            prim = stage.GetPrimAtPath("/World/envs/env_0/Cargo/Materials/body/Shader")
            if not prim.IsValid():
                raise RuntimeError("Cargo body shader unavailable; refusing silent colour-randomization failure")
            attribute = prim.GetAttribute("inputs:diffuseColor")
            if not attribute.Set(Gf.Vec3f(*self.reset_metadata["body_color_rgb"])):
                raise RuntimeError("Failed to apply cargo body colour")
        self.task.reset()
        self.last = np.r_[start, 0.].astype(np.float32)
        for _ in range(15):
            self.command(self.last)
            self.advance()
        self.task.reset()
        self.reset_metadata["settled_spawn_xyz_m"] = self.cargo.data.root_pos_w[0].cpu().tolist()
        if self.randomization == "wide":
            xyz = self.reset_metadata["settled_spawn_xyz_m"]
            print(f"[Cargo reset] wide seed={seed} settled bag xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m", flush=True)
        if self.randomization == "tray":
            actual = (self.scene["target_station"].data.root_pos_w[0, :2] - self.scene.env_origins[0, :2]).cpu().tolist()
            self.reset_metadata["settled_target_xy_m"] = actual
            print(f"[Cargo reset] tray seed={seed} target xy={actual}; fixed pickup xy={self.reset_metadata['settled_spawn_xyz_m'][:2]}", flush=True)
        return rng

    def tip_pose(self):
        p, q = compute_gripper_tip_pose_w(self.robot, self.wrist_id, self.finger_ids)
        return p[0].cpu().numpy(), q[0].cpu().numpy()

    def observe(self):
        if self.cameras:
            self.sim.render()
        closure = ((self.robot.data.joint_pos[:, self.grip_ids] - self.g_open)
                   / (self.g_closed - self.g_open)).mean().clamp(0, 1).item()
        state = np.r_[self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy(), closure].astype(np.float32)
        obs = {"observation.state": state}
        if self.cameras:
            for key in CAMERAS:
                self.scene[key].update(.04, force_recompute=True)
                obs[f"observation.images.{key}"] = self.scene[key].data.output["rgb"][0, ..., :3].cpu().numpy().copy()
        return obs

    def command(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite absolute arm/gripper commands")
        action = action.copy()
        action[:6] = np.clip(action[:6], self.limits[:, 0], self.limits[:, 1])
        action[:6] = self.last[:6] + np.clip(action[:6] - self.last[:6], -.06, .06)
        action[6] = np.clip(action[6], 0, 1)
        targets = self.robot.data.default_joint_pos.clone()
        targets[:, self.arm_ids] = torch.as_tensor(action[:6], device=self.device)
        targets[:, self.grip_ids] = self.g_open + float(action[6]) * (self.g_closed - self.g_open)
        self.robot.set_joint_position_target(targets)
        self.robot.set_joint_velocity_target(torch.zeros_like(targets))
        self.last = action
        return action.copy()

    def advance(self):
        for _ in range(4):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(.01)
            self.status = self.task.update(.01)
        return bool(self.status["success"][0].item())
