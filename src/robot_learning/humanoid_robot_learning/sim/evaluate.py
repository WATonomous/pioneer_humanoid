"""Success rate of a LeRobot policy checkpoint on a MuJoCo scene (humanoid_mujoco_scenes), left arm.

    python -m humanoid_robot_learning.sim.evaluate --scene peg_insert --policy <run>/checkpoints/last/pretrained_model

Works for any scene that registers ``success`` or a multi-step ``progress``. Episode ``i`` lays the scene out with
``reset`` seeded by ``--seed + i``, as the synthetic demos are, so pick seeds the training data did not use. The arm is
driven like the recorded teleop: state is the 6 arm joints plus gripper closure, the action 6 joint targets plus the
gripper command, held for one frame (1 / ``--fps``); cameras are rendered at the size the policy was trained on.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from humanoid_robot_learning.utils.checkpoint import load_policy, training_task

# leader_mapping.py: the teleop's control period (scene step hooks run at this rate) and the wrist damping it adds.
CONTROL_DT = 0.01
WRIST_DAMPING = 2.5
HOLD_S = 0.4    # success must last this long (a peg that drops in and bounces out is not inserted)


class SceneEnv:
    def __init__(self, scene: str, cameras: dict[str, tuple[int, int]]):
        from humanoid_mujoco_scenes import make_model, scene_progress, scene_reset, scene_step, scene_success
        from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, LEFT_GRIPPER_CLOSED, LEFT_GRIPPER_JOINTS, LEFT_GRIPPER_OPEN

        self.success = scene_success(scene)
        if self.success is None:
            raise SystemExit(f"scene {scene!r} registers neither success nor progress: nothing to evaluate")
        self.randomise, self.hook, self.progress = scene_reset(scene), scene_step(scene), scene_progress(scene)
        self.model = model = make_model(scene, cameras=cameras)
        self.data = mujoco.MjData(model)
        model.actuator_biasprm[model.actuator("joint6l").id, 2] = -WRIST_DAMPING
        self.arm_acts = [model.actuator(j).id for j in LEFT_ARM_JOINTS]
        self.arm_qpos = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
        self.arm_lo, self.arm_hi = model.jnt_range[[model.joint(j).id for j in LEFT_ARM_JOINTS]].T
        self.grip_acts = [model.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
        self.grip_qpos = [model.joint(j).qposadr[0] for j in LEFT_GRIPPER_JOINTS]
        self.grip_open = np.array([LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS])
        self.grip_closed = np.array([LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS])
        self.substeps = max(1, round(CONTROL_DT / model.opt.timestep))
        self.cameras = cameras
        self.renderers = {size: mujoco.Renderer(model, *size) for size in set(cameras.values())}

    def reset(self, seed: int) -> None:
        from pioneer_humanoid.mujoco_bimanual_arm import set_home

        mujoco.mj_resetData(self.model, self.data)
        set_home(self.model, self.data)
        if self.randomise is not None:
            self.randomise(self.model, self.data, np.random.default_rng([seed, 0]))
        mujoco.mj_forward(self.model, self.data)

    def observe(self) -> dict[str, np.ndarray]:
        closure = np.mean((self.data.qpos[self.grip_qpos] - self.grip_open) / (self.grip_closed - self.grip_open))
        obs = {"observation.state": np.append(self.data.qpos[self.arm_qpos], np.clip(closure, 0.0, 1.0)).astype(np.float32)}
        for name, size in self.cameras.items():
            self.renderers[size].update_scene(self.data, name)
            obs[f"observation.images.{name}"] = self.renderers[size].render()
        return obs

    def step(self, action: np.ndarray, control_steps: int) -> None:
        self.data.ctrl[self.arm_acts] = np.clip(action[:6], self.arm_lo, self.arm_hi)
        self.data.ctrl[self.grip_acts] = self.grip_open + np.clip(action[6], 0.0, 1.0) * (self.grip_closed - self.grip_open)
        for _ in range(control_steps):
            if self.hook is not None:
                self.hook(self.model, self.data)
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    def instruction(self) -> str | None:
        return self.progress(self.model, self.data)[2] if self.progress is not None else None

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval of a success rate k / n."""
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    import torch
    from lerobot.utils.control_utils import predict_action

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
    args = p.parse_args()

    control_steps = round(1 / (args.fps * CONTROL_DT))
    if abs(control_steps * args.fps * CONTROL_DT - 1) > 1e-9:
        p.error(f"--fps {args.fps} must divide {1 / CONTROL_DT:g} Hz")
    policy, pre, post = load_policy(args.policy, args.device)
    cameras = {key.removeprefix("observation.images."): tuple(ft.shape[1:]) for key, ft in policy.config.image_features.items()}
    env = SceneEnv(args.scene, cameras)
    task = args.task or training_task(args.policy)
    if task is None and env.progress is None:
        p.error("pass --task: the scene has no step instructions and the training dataset's task was not found")

    successes = 0
    for i in range(args.episodes):
        seed = args.seed + i
        env.reset(seed)
        policy.reset()
        frames, held, ok = [], 0.0, False
        for _ in range(int(args.max_s * args.fps)):
            obs = env.observe()
            if args.video_dir:
                frames.append(np.concatenate([obs[f"observation.images.{c}"] for c in cameras], axis=1))
            action = predict_action(obs, policy, torch.device(args.device), pre, post, use_amp=False, task=env.instruction() or task)
            env.step(action.reshape(-1).cpu().numpy(), control_steps)
            held = held + 1 / args.fps if env.success(env.model, env.data) else 0.0
            if held >= HOLD_S:
                ok = True
                break
        successes += ok
        print(f"[EVAL] seed {seed}: {'success' if ok else 'fail'} at {env.data.time:.1f} s ({successes}/{i + 1})", flush=True)
        if args.video_dir:
            import imageio

            Path(args.video_dir).mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(Path(args.video_dir) / f"{args.scene}_seed{seed}_{'success' if ok else 'fail'}.mp4", frames, fps=args.fps)
    env.close()
    lo, hi = wilson(successes, args.episodes)
    print(f"[EVAL] {args.scene}: success {successes}/{args.episodes} = {successes / args.episodes:.0%} (95% CI {lo:.0%}-{hi:.0%})")


if __name__ == "__main__":
    main()
