"""Train bounded PPO corrections to a frozen ACT cube policy in MuJoCo."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from humanoid_mujoco_scenes.rubiks_cube.scene import LIFT_MIN, cube_height
from humanoid_robot_learning.sim.evaluate import CONTROL_DT, HOLD_S, SceneEnv, load_policy, wilson
from lerobot.utils.control_utils import predict_action
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from eval_rubiks_poses import FPS, MAX_SECONDS, perturb_start


RESIDUAL_LIMIT = np.array([0.14] * 6 + [0.30], dtype=np.float32)
TASK = "sim leader teleop demonstration"
ROLLOUT_STEPS = 1024


class RubiksResidual(gym.Env):
    """ACT supplies absolute targets. PPO sees the simulator pose and adds a small offset."""

    def __init__(self, policy_path: str, device: str):
        self.device = torch.device(device)
        self.act, self.pre, self.post = load_policy(policy_path, str(self.device))
        cfg = self.act.config
        if cfg.type != "act" or cfg.temporal_ensemble_coeff is not None or cfg.n_action_steps != 100:
            raise ValueError("This pilot requires the validated ACT settings (100 actions, no ensembling)")
        self.act.requires_grad_(False)
        cameras = {key.removeprefix("observation.images."): tuple(f.shape[1:]) for key, f in cfg.image_features.items()}
        self.scene = SceneEnv("rubiks_cube", cameras)
        model = self.scene.model
        self.cube = model.body("rubiks_cube").id
        self.cube_geom = model.geom("rubiks_cube").id
        cube_joint = model.joint("rubiks_cube").id
        self.cube_vel = slice(model.jnt_dofadr[cube_joint], model.jnt_dofadr[cube_joint] + 6)
        self.arm_vel = model.jnt_dofadr[model.actuator_trnid[self.scene.arm_acts, 0]]
        self.fingers = {model.body(name).id for name in ("link7l", "link8l")}
        self.finger_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in self.fingers and model.geom_contype[i]]
        self.ctrl_steps = round(1 / (FPS * CONTROL_DT))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (36,), np.float32)

    @torch.inference_mode()
    def _act_action(self) -> np.ndarray:
        # The pinned ACT implementation consumes queued actions without looking at a new frame.
        # Apply the same postprocessor while avoiding render and image preprocessing until replanning.
        if self.act._action_queue:
            action = self.post(self.act._action_queue.popleft())
        else:
            action = predict_action(self.scene.observe(), self.act, self.device, self.pre, self.post, False, task=TASK)
        return action.reshape(-1).cpu().numpy()

    def _relative_cube(self) -> np.ndarray:
        data = self.scene.data
        return data.xpos[self.cube] - data.geom_xpos[self.finger_geoms].mean(axis=0)

    def _two_finger_contact(self) -> bool:
        model, data = self.scene.model, self.scene.data
        touched = set()
        for contact in data.contact[: data.ncon]:
            if self.cube_geom in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1
                touched.add(model.geom_bodyid[other])
        return self.fingers.issubset(touched)

    def _observation(self) -> np.ndarray:
        scene, data = self.scene, self.scene.data
        closure = np.mean((data.qpos[scene.grip_qpos] - scene.grip_open) / (scene.grip_closed - scene.grip_open))
        return np.concatenate((
            data.qpos[scene.arm_qpos], 0.2 * data.qvel[self.arm_vel], [np.clip(closure, 0, 1)], self.base_action,
            10 * self._relative_cube(), data.xquat[self.cube], 0.2 * data.qvel[self.cube_vel],
            [self.steps / (FPS * MAX_SECONDS), np.clip(cube_height(scene.model, data) / LIFT_MIN, -1, 2),
             min(data.userdata[0] / LIFT_MIN, 2)],
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.layout_seed = options.get("layout_seed") if "layout_seed" in options else int(self.np_random.integers(100_000, 2**31))
        self.scene.reset(0 if self.layout_seed is None else self.layout_seed)
        self.offsets = {} if self.layout_seed is None else perturb_start(self.scene, self.layout_seed, "both")
        self.steps, self.held = 0, 0.0
        self.best_lift = 0.0
        self.best_approach = 0.0
        self.grasp_seen = False
        self.act.reset()
        self.base_action = self._act_action()
        return self._observation(), {}

    def step(self, action):
        correction = np.clip(action, -1, 1) * RESIDUAL_LIMIT
        self.scene.step(self.base_action + correction, self.ctrl_steps)
        self.steps += 1
        data = self.scene.data
        self.held = self.held + 1 / FPS if self.scene.success(self.scene.model, data) else 0.0
        success = self.held >= HOLD_S
        truncated = not success and self.steps >= FPS * MAX_SECONDS

        approach = max(self.best_approach, 1 - np.linalg.norm(self._relative_cube()) / 0.18)
        lift = max(self.best_lift, min(max(data.userdata[0], 0.0) / LIFT_MIN, 1.0))
        first_grasp = not self.grasp_seen and self._two_finger_contact()
        # Pay these bonuses once as the episode reaches a new best, never for repeated contact.
        reward = (0.4 * (approach - self.best_approach) + 1.5 * (lift - self.best_lift)
                  + 0.35 * first_grasp + 6 * success - 0.002 - 0.01 * np.mean(np.square(action)))
        self.best_approach, self.best_lift = approach, lift
        self.grasp_seen |= first_grasp
        if not (success or truncated):
            self.base_action = self._act_action()
        info = {"is_success": success, "peak_lift_cm": round(float(data.userdata[0]) * 100, 2),
                "elapsed_s": round(float(data.time), 2)}
        return self._observation(), float(reward), success, truncated, info

    def close(self):
        self.scene.close()


def trial(env, model, seed):
    obs, _ = env.reset(options={"layout_seed": seed})
    while True:
        action = np.zeros(7, dtype=np.float32) if model is None else model.predict(obs, deterministic=True)[0]
        obs, _, done, truncated, info = env.step(action)
        if done or truncated:
            return {"seed": seed, "success": info.pop("is_success"), **info, **env.offsets}


def score(env, model, seeds, label):
    rows = []
    for seed in seeds:
        row = trial(env, model, seed)
        rows.append(row)
        print(json.dumps({"suite": label, **row}), flush=True)
    successes = sum(row["success"] for row in rows)
    print(f"{label}: {successes}/{len(rows)}", flush=True)
    return {"successes": successes, "trial_count": len(rows), "wilson_95": wilson(successes, len(rows)), "trials": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Device for frozen ACT inference")
    parser.add_argument("--steps", type=int, default=24576)
    parser.add_argument("--save-every", type=int, default=8192)
    parser.add_argument("--smoke", action="store_true", help="Run environment checks and one PPO update")
    args = parser.parse_args()
    if min(args.steps, args.save_every) < ROLLOUT_STEPS or args.steps % ROLLOUT_STEPS or args.save_every % ROLLOUT_STEPS:
        parser.error(f"step counts must be positive multiples of {ROLLOUT_STEPS}")

    args.output.mkdir(parents=True, exist_ok=True)
    env = RubiksResidual(args.policy, args.device)
    try:
        reference = trial(env, None, None)
        print(f"Zero-correction reference: {reference}", flush=True)
        if not reference["success"]:
            raise RuntimeError("Zero correction failed to reproduce the ACT reference success")
        if args.smoke:
            check_env(env, warn=True, skip_render_check=True)
        model = PPO("MlpPolicy", env, n_steps=ROLLOUT_STEPS, batch_size=128, n_epochs=5,
                    learning_rate=3e-4, gamma=0.99, seed=90210, device="cpu", verbose=1,
                    policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.25})
        # Begin at zero deterministic correction while keeping stochastic exploration available.
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
        horizon = ROLLOUT_STEPS if args.smoke else args.steps
        checkpoints = []
        for start in range(0, horizon, args.save_every):
            end = min(horizon, start + args.save_every)
            model.learn(end - start, reset_num_timesteps=False)
            path = args.output / f"residual-{end}.zip"
            model.save(path)
            checkpoints.append(path)
            print(f"Saved {path}", flush=True)
        if args.smoke:
            reload = PPO.load(checkpoints[0], device="cpu")
            print(f"Reloaded trained rollout: {trial(env, reload, 4000)}", flush=True)
            return

        # Select on development seeds only; report the original benchmark and a new matched test.
        development = {}
        for path in checkpoints:
            candidate = PPO.load(path, device="cpu")
            development[path.name] = score(env, candidate, range(4000, 4012), f"development/{path.stem}")
        best = max(checkpoints, key=lambda path: development[path.name]["successes"])
        chosen = PPO.load(best, device="cpu")
        report = {"act_checkpoint": args.policy, "selected_residual": best.name, "reference": reference,
                  "corrected_reference": trial(env, chosen, None), "development": development, "evaluation": {}}
        for name, seeds in (("benchmark", range(2026, 2066)), ("fresh", range(8000, 8040))):
            report["evaluation"][name] = {
                "act": score(env, None, seeds, f"{name}/act"),
                "corrected": score(env, chosen, seeds, f"{name}/corrected"),
            }
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
