"""Success rate of a LeRobot policy checkpoint on a MuJoCo scene (humanoid_mujoco_scenes), left arm.

    python -m humanoid_robot_learning.sim.evaluate --scene peg_insert --policy <run>/checkpoints/last/pretrained_model

Works for any scene that registers ``success`` or a multi-step ``progress``. Episode ``i`` lays the scene out with
``reset`` seeded by ``--seed + i``, as the synthetic demos are, so pick seeds the training data did not use. The arm is
driven like the recorded teleop: state is the arm joints plus gripper closure, the action the joint targets plus the
gripper command, held for one frame (1 / ``--fps``); cameras are rendered at the size the policy was trained on.

Robot (``--robot``, default from the policy's action size): ``8dof`` = pioneer_bimanual_arm_8dof's left arm (7 joints +
gripper, pioneer_8dof_dataset_schema.yaml), ``armv2`` = the old 6-DoF left arm (6 + gripper).

RECAP (humanoid_robot_learning/recap): ``--record DIR`` saves every rollout as a LeRobot dataset in the teleop's
single-arm format plus sidecar files (each episode's outcome and every frame's sim state, to resume corrections from);
``--advantage positive|negative`` appends " Advantage: ..." to the prompt of an advantage-conditioned policy (the
recorded task stays the plain instruction).

Policies trained with LeRobot's ``--rename_map`` (pi0.5: ``ego`` -> ``base_0_rgb``, ``wrist_left`` -> ``left_wrist_0_rgb``)
see our cameras under their names: the map is read from the checkpoint's train_config.json (or ``--rename_map``), our
cameras are rendered and fed under the policy's keys, and policy cameras with no sim camera are left out (pi0.5 pads
them). Recorded rollouts keep our camera names, as the teleop data. A camera renders at the policy's image size when
that has the camera's sensor aspect, else at the size the robot's dataset schema records (pi0.5 declares 224x224 and
resizes internally: its 4:3 cameras render at 640x480).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

# leader_mapping.py: the teleop's control period (scene step hooks run at this rate) and the wrist damping it adds.
CONTROL_DT = 0.01
WRIST_DAMPING = 2.5
ASPECT_TOLERANCE = 0.02   # relative; pioneer_humanoid.arm_params.vertical_aperture's sensor-aspect check
HOLD_S = 0.4    # success must last this long (a peg that drops in and bounces out is not inserted)
# Left-arm action size (arm joints + gripper) per robot: how --robot is picked from the policy.
ACTION_DIM = {"armv2": 7, "8dof": 8}


def robot_params(robot: str):
    """(params module, set_home, wrist actuator to give the teleop's damping or None) of a sim robot."""
    if robot == "8dof":
        from pioneer_humanoid import arm_8dof_params as params
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home
        return params, set_home, None   # AK45-10 wrist PD group: no extra damping
    if robot == "armv2":
        from pioneer_humanoid import arm_params as params
        from pioneer_humanoid.mujoco_bimanual_arm import set_home
        return params, set_home, "joint6l"
    raise ValueError(f"unknown robot {robot!r}; known: {list(ACTION_DIM)}")


def robot_for_action_dim(dim: int) -> str:
    for robot, n in ACTION_DIM.items():
        if n == dim:
            return robot
    raise ValueError(f"policy action size {dim} matches no robot ({ACTION_DIM}); pass --robot")


class SceneEnv:
    def __init__(self, scene: str, cameras: dict[str, tuple[int, int]], robot: str = "8dof",
                 record_cameras: dict[str, tuple[int, int]] | None = None):
        """`cameras`: the policy's inputs, {name: (height, width)}; `record_cameras`: also mounted, rendered by render()."""
        from humanoid_mujoco_scenes import make_model, scene_progress, scene_reset, scene_step, scene_success

        params, self.set_home, wrist = robot_params(robot)
        LEFT_ARM_JOINTS, LEFT_GRIPPER_JOINTS = params.LEFT_ARM_JOINTS, params.LEFT_GRIPPER_JOINTS
        LEFT_GRIPPER_OPEN, LEFT_GRIPPER_CLOSED = params.LEFT_GRIPPER_OPEN, params.LEFT_GRIPPER_CLOSED
        self.robot, self.n_arm = robot, len(LEFT_ARM_JOINTS)

        self.success = scene_success(scene)
        if self.success is None:
            raise SystemExit(f"scene {scene!r} registers neither success nor progress: nothing to evaluate")
        self.randomise, self.hook, self.progress = scene_reset(scene), scene_step(scene), scene_progress(scene)
        self.model = model = make_model(scene, cameras={**(record_cameras or {}), **cameras}, robot=robot)
        self.data = mujoco.MjData(model)
        if wrist is not None:
            model.actuator_biasprm[model.actuator(wrist).id, 2] = -WRIST_DAMPING
        self.arm_acts = [model.actuator(j).id for j in LEFT_ARM_JOINTS]
        self.arm_qpos = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
        self.arm_lo, self.arm_hi = model.jnt_range[[model.joint(j).id for j in LEFT_ARM_JOINTS]].T
        self.grip_acts = [model.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
        self.grip_qpos = [model.joint(j).qposadr[0] for j in LEFT_GRIPPER_JOINTS]
        self.grip_open = np.array([LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS])
        self.grip_closed = np.array([LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS])
        self.substeps = max(1, round(CONTROL_DT / model.opt.timestep))
        self.cameras = cameras
        self.renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def reset(self, seed: int) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.set_home(self.model, self.data)
        if self.randomise is not None:
            self.randomise(self.model, self.data, np.random.default_rng([seed, 0]))
        mujoco.mj_forward(self.model, self.data)

    def observe(self) -> dict[str, np.ndarray]:
        closure = np.mean((self.data.qpos[self.grip_qpos] - self.grip_open) / (self.grip_closed - self.grip_open))
        obs = {"observation.state": np.append(self.data.qpos[self.arm_qpos], np.clip(closure, 0.0, 1.0)).astype(np.float32)}
        for name, size in self.cameras.items():
            obs[f"observation.images.{name}"] = self.render(name, size)
        return obs

    def render(self, camera: str, size: tuple[int, int]) -> np.ndarray:
        if size not in self.renderers:
            self.renderers[size] = mujoco.Renderer(self.model, *size)
        self.renderers[size].update_scene(self.data, camera)
        return self.renderers[size].render()

    def clip(self, action: np.ndarray) -> np.ndarray:
        """The action as applied: joint targets inside the joint limits, gripper command in 0..1."""
        if len(action) != self.n_arm + 1:
            raise ValueError(f"{self.robot}: action has {len(action)} values, expected {self.n_arm} joints + gripper")
        return np.append(np.clip(action[:self.n_arm], self.arm_lo, self.arm_hi), np.clip(action[self.n_arm], 0.0, 1.0))

    def step(self, action: np.ndarray, control_steps: int) -> None:
        action, n = self.clip(action), self.n_arm
        self.data.ctrl[self.arm_acts] = action[:n]
        self.data.ctrl[self.grip_acts] = self.grip_open + action[n] * (self.grip_closed - self.grip_open)
        for _ in range(control_steps):
            if self.hook is not None:
                self.hook(self.model, self.data)
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    def instruction(self) -> str | None:
        return self.progress(self.model, self.data)[2] if self.progress is not None else None

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()


def load_policy(path: str, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from humanoid_robot_learning.policies import register

    register()
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def training_task(path: str) -> str | None:
    """The task text of the policy's training dataset, if it has exactly one."""
    import pandas as pd

    config = Path(path) / "train_config.json"
    root = json.loads(config.read_text())["dataset"].get("root") if config.exists() else None
    tasks = Path(root or "") / "meta" / "tasks.parquet"
    if not root or not tasks.exists():
        return None
    names = list(pd.read_parquet(tasks).index)
    return names[0] if len(names) == 1 else None


def checkpoint_rename_map(path: str) -> dict[str, str]:
    """The `rename_map` ({dataset key: policy key}) the checkpoint was trained with, or {}."""
    config = Path(path) / "train_config.json"
    return (json.loads(config.read_text()).get("rename_map") or {}) if config.exists() else {}


def recorded_sizes(robot: str) -> dict[str, tuple[int, int]]:
    """{camera: (height, width)} the robot's dataset schema records (what teleop and rollout datasets hold)."""
    from humanoid_robot_learning.recording.schema import load_yaml
    from humanoid_robot_learning.sim.sim_teleop_record import DEFAULT_SCHEMAS

    images = load_yaml(DEFAULT_SCHEMAS[robot]).get("images") or {}
    return {name: (int(spec["height"]), int(spec["width"])) for name, spec in images.items()}


def render_size(mount, policy_size: tuple[int, int], recorded: tuple[int, int] | None) -> tuple[int, int]:
    """The policy's (height, width) if it has the sensor's aspect (`mount.aspect`, w/h; None: any), else `recorded`:
    a policy that resizes internally (pi0.5: 224x224) gets the camera as recorded and its preprocessor resizes it."""
    height, width = policy_size
    if mount.aspect is None or abs(width / height - mount.aspect) <= ASPECT_TOLERANCE * mount.aspect or recorded is None:
        return policy_size
    return recorded


def policy_cameras(policy, rename_map: dict[str, str], sim_cameras,
                   recorded: dict[str, tuple[int, int]]) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    """({sim camera: (height, width)} to render, {our observation key: the policy's key}).

    `rename_map`: {dataset key: policy key}, as LeRobot training's --rename_map; `sim_cameras`: the robot's
    {camera: CameraMount}; `recorded`: recorded_sizes(). A policy camera with no sim camera behind it is skipped
    (pi0.5 pads views it does not get)."""
    source = {policy_key: ours for ours, policy_key in rename_map.items()}
    cameras, to_policy = {}, {}
    for key, feature in policy.config.image_features.items():
        ours = source.get(key, key)
        name = ours.removeprefix("observation.images.")
        if name not in sim_cameras:
            print(f"[EVAL] policy camera {key}: no sim camera, not fed (the policy pads it)", flush=True)
            continue
        cameras[name] = render_size(sim_cameras[name], tuple(feature.shape[1:]), recorded.get(name))
        to_policy[ours] = key
        if cameras[name] != tuple(feature.shape[1:]):
            print(f"[EVAL] {name}: policy input {feature.shape[1]}x{feature.shape[2]} is not the sensor's aspect; "
                  f"rendering {cameras[name][0]}x{cameras[name][1]} for the policy to resize", flush=True)
    if not cameras:
        raise ValueError(f"none of the policy's cameras {list(policy.config.image_features)} maps to a sim camera "
                         f"{sorted(sim_cameras)}; pass --rename_map")
    return cameras, to_policy


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval of a success rate k / n."""
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_args():
    import torch

    from humanoid_robot_learning.recap.rollouts import ADVANTAGES

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", required=True)
    p.add_argument("--policy", required=True, help="pretrained_model directory of a LeRobot checkpoint")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=1000, help="layout seed of the first episode")
    p.add_argument("--max_s", type=float, default=40.0, help="time limit per episode (sim seconds)")
    p.add_argument("--fps", type=int, default=25, help="the training dataset's frame rate")
    p.add_argument("--task", default=None, help="task text (default: the scene's step instruction, else the training dataset's task)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--video_dir", default=None, help="write an mp4 per episode (cameras side by side)")
    p.add_argument("--robot", choices=tuple(ACTION_DIM), default=None,
                   help="sim arm (default: from the policy's action size: 8 = 8dof, 7 = armv2)")
    p.add_argument("--rename_map", type=json.loads, default=None,
                   help="JSON {dataset key: policy key} the policy was trained with (default: the checkpoint's "
                        "train_config.json rename_map)")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="RECAP: save every rollout to this LeRobot dataset (appends) plus its recap/ sidecar")
    p.add_argument("--round", type=int, default=0, help="RECAP iteration, stored with --record's episodes")
    p.add_argument("--advantage", choices=tuple(ADVANTAGES), default="none",
                   help="append ' Advantage: positive|negative' to the prompt (advantage-conditioned policies)")
    args = p.parse_args()
    control_steps = round(1 / (args.fps * CONTROL_DT))
    if abs(control_steps * args.fps * CONTROL_DT - 1) > 1e-9:
        p.error(f"--fps {args.fps} must divide {1 / CONTROL_DT:g} Hz")
    return p, args, control_steps


def pick_robot(p, args, policy) -> str:
    action_dim = int(policy.config.output_features["action"].shape[0])
    try:
        robot = args.robot or robot_for_action_dim(action_dim)
    except ValueError as exc:
        p.error(str(exc))
    if action_dim != ACTION_DIM[robot]:
        p.error(f"the policy outputs {action_dim} actions; --robot {robot} takes {ACTION_DIM[robot]}")
    print(f"[EVAL] robot {robot} ({action_dim} actions)", flush=True)
    return robot


def rollout(env: SceneEnv, act, args, control_steps: int, task: str | None, recorder=None):
    """One episode from the current state: (success, video frames). `act(obs, prompt) -> action`;
    `recorder`: a RolloutRecorder that gets every frame, or None."""
    from humanoid_robot_learning.recap.rollouts import advantage_prompt

    frames, held = [], 0.0
    for _ in range(int(args.max_s * args.fps)):
        obs = env.observe()
        if args.video_dir:
            frames.append(np.concatenate([obs[f"observation.images.{c}"] for c in env.cameras], axis=1))
        instruction = env.instruction() or task
        action = env.clip(act(obs, advantage_prompt(instruction, args.advantage)))
        if recorder is not None:
            recorder.add(env, obs, action, instruction)
        env.step(action, control_steps)
        held = held + 1 / args.fps if env.success(env.model, env.data) else 0.0
        if held >= HOLD_S:
            return True, frames
    return False, frames


def main() -> None:
    import torch
    try:
        from lerobot.common.control_utils import predict_action   # lerobot >= 0.7
    except ImportError:
        from lerobot.utils.control_utils import predict_action

    p, args, control_steps = parse_args()
    policy, pre, post = load_policy(args.policy, args.device)
    robot = pick_robot(p, args, policy)
    rename_map = checkpoint_rename_map(args.policy) if args.rename_map is None else args.rename_map
    try:
        cameras, to_policy = policy_cameras(policy, rename_map, robot_params(robot)[0].CAMERAS, recorded_sizes(robot))
    except ValueError as exc:
        p.error(str(exc))
    recorder = None
    if args.record:
        from humanoid_robot_learning.recap.rollouts import RolloutRecorder
        from humanoid_mujoco_scenes import scene_progress

        recorder = RolloutRecorder(args.record, robot, args.fps, args.max_s, steps=scene_progress(args.scene) is not None)
    env = SceneEnv(args.scene, cameras, robot, record_cameras=recorder.cameras if recorder else None)
    task = args.task or training_task(args.policy)
    if task is None and env.progress is None:
        p.error("pass --task: the scene has no step instructions and the training dataset's task was not found")

    def act(obs, prompt):
        obs = {to_policy.get(key, key): value for key, value in obs.items()}
        return predict_action(obs, policy, torch.device(args.device), pre, post, use_amp=False, task=prompt).reshape(-1).cpu().numpy()

    successes = 0
    for i in range(args.episodes):
        seed = args.seed + i
        env.reset(seed)
        policy.reset()
        ok, frames = rollout(env, act, args, control_steps, task, recorder)
        successes += ok
        print(f"[EVAL] seed {seed}: {'success' if ok else 'fail'} at {env.data.time:.1f} s ({successes}/{i + 1})", flush=True)
        if recorder is not None:
            recorder.finish(round=args.round, scene=args.scene, policy=str(args.policy), advantage=args.advantage,
                            seed=seed, success=ok)
        if args.video_dir:
            import imageio

            Path(args.video_dir).mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(Path(args.video_dir) / f"{args.scene}_seed{seed}_{'success' if ok else 'fail'}.mp4", frames, fps=args.fps)
    env.close()
    if recorder is not None:
        recorder.close()
        print(f"[EVAL] rollouts recorded in {args.record} (sidecar: {Path(args.record) / 'recap'})", flush=True)
    lo, hi = wilson(successes, args.episodes)
    print(f"[EVAL] {args.scene}: success {successes}/{args.episodes} = {successes / args.episodes:.0%} (95% CI {lo:.0%}-{hi:.0%})")


if __name__ == "__main__":
    main()
