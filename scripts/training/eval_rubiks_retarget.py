"""Retarget the ACT cube trajectory using the recording-era MuJoCo cube pose."""

import argparse
import json
from collections.abc import Iterable, Iterator
from importlib.metadata import version
from itertools import chain, repeat
from pathlib import Path

import mujoco
import numpy as np
import torch
from humanoid_mujoco_scenes.rubiks_cube.scene import LIFT_MIN
from humanoid_robot_learning.sim.evaluate import CONTROL_DT, HOLD_S, SceneEnv, load_policy, wilson
from lerobot.policies.utils import prepare_observation_for_inference

from eval_rubiks_poses import FPS, MAX_SECONDS, perturb_start


CUBE = "rubiks_cube"
FINGERS = ("link7l", "link8l")
TASK = "sim leader teleop demonstration"
CONTROL_STEPS = round(1 / (FPS * CONTROL_DT))
ARM_BLEND_STEPS = 8
ORIENTATION_WEIGHT_M = 0.16
IK_DAMPING = 0.002
IK_ITERATIONS = 12
IK_MAX_UPDATE_RAD = 0.12


def cube_yaw(data: mujoco.MjData, cube: int) -> float:
    rotation = data.xmat[cube].reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


class RetargetedACT:
    """Replay ACT's nominal grasp in the frame of the cube at the start of the trial."""

    def __init__(self, env: SceneEnv, reference_actions: np.ndarray):
        self.env = env
        self.model = env.model
        self.work = mujoco.MjData(self.model)
        self.work.qpos[:] = env.data.qpos
        self.wrist = self.model.body("link6l").id
        self.cube = self.model.body(CUBE).id
        self.arm_dofs = self.model.jnt_dofadr[self.model.actuator_trnid[env.arm_acts, 0]]
        self.home = env.data.qpos[env.arm_qpos].copy()
        self.reference_cube = env.data.xpos[self.cube].copy()
        self.reference_yaw = cube_yaw(env.data, self.cube)
        self.actions = reference_actions.copy()
        self.actions[:, :-1] = np.clip(self.actions[:, :-1], env.arm_lo, env.arm_hi)
        self.reference_poses = [self._wrist_pose(action[:-1]) for action in self.actions]

        self.jac_position = np.zeros((3, self.model.nv))
        self.jac_rotation = np.zeros_like(self.jac_position)
        self.damping = IK_DAMPING**2 * np.eye(len(self.arm_dofs))

    def _wrist_pose(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.work.qpos[self.env.arm_qpos] = joints
        mujoco.mj_kinematics(self.model, self.work)
        return self.work.xpos[self.wrist].copy(), self.work.xmat[self.wrist].reshape(3, 3).copy()

    def _solve(self, joints: np.ndarray, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        quat_error = np.empty(4)
        rotation_error = np.empty(3)
        for _ in range(IK_ITERATIONS):
            wrist_position, wrist_rotation = self._wrist_pose(joints)
            mujoco.mju_mat2Quat(quat_error, (rotation @ wrist_rotation.T).ravel())
            mujoco.mju_quat2Vel(rotation_error, quat_error, 1.0)
            error = np.r_[position - wrist_position, ORIENTATION_WEIGHT_M * rotation_error]
            if np.linalg.norm(error) < 1e-5:
                break
            mujoco.mj_comPos(self.model, self.work)
            mujoco.mj_jacBody(self.model, self.work, self.jac_position, self.jac_rotation, self.wrist)
            jacobian = np.vstack(
                (self.jac_position[:, self.arm_dofs], ORIENTATION_WEIGHT_M * self.jac_rotation[:, self.arm_dofs])
            )
            update = np.linalg.solve(jacobian.T @ jacobian + self.damping, jacobian.T @ error)
            update = np.clip(update, -IK_MAX_UPDATE_RAD, IK_MAX_UPDATE_RAD)
            joints = np.clip(joints + update, self.env.arm_lo, self.env.arm_hi)
        return joints

    def actions_for_start(self) -> Iterator[np.ndarray]:
        data = self.env.data
        return self.actions_for_pose(
            data.xpos[self.cube], cube_yaw(data, self.cube), data.qpos[self.env.arm_qpos]
        )

    def actions_for_pose(
        self, cube_position: np.ndarray, yaw: float, arm_joints: np.ndarray
    ) -> Iterator[np.ndarray]:
        yaw_delta = yaw - self.reference_yaw
        cosine, sine = np.cos(yaw_delta), np.sin(yaw_delta)
        rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        arm_delta = arm_joints - self.home
        actions = self.actions.copy()
        for step, (position, orientation) in enumerate(self.reference_poses):
            target_position = cube_position + rotation @ (position - self.reference_cube)
            joints = self._solve(actions[step, :-1], target_position, rotation @ orientation)
            # Ease out of the measured arm offset before the gripper reaches the cube.
            blend = max(0.0, 1.0 - step / ARM_BLEND_STEPS) ** 2
            actions[step, :-1] = np.clip(joints + blend * arm_delta, self.env.arm_lo, self.env.arm_hi)
        return chain(actions, repeat(actions[-1]))


def rollout(env: SceneEnv, actions: Iterable[np.ndarray]) -> dict:
    model, data = env.model, env.data
    cube = model.geom(CUBE).id
    fingers = {model.body(name).id for name in FINGERS}
    pads = [geom for geom in range(model.ngeom) if model.geom_bodyid[geom] in fingers and model.geom_contype[geom]]
    held = 0.0
    two_finger_contact = False
    closest_distance = float("inf")
    for _, action in zip(range(FPS * MAX_SECONDS), actions):
        env.step(action, CONTROL_STEPS)
        if not two_finger_contact:
            midpoint = data.geom_xpos[pads].mean(axis=0)
            closest_distance = min(closest_distance, float(np.linalg.norm(midpoint - data.geom_xpos[cube])))
            touching = set()
            for contact in data.contact[: data.ncon]:
                if contact.efc_address >= 0 and cube in (contact.geom1, contact.geom2):
                    other = contact.geom1 if contact.geom2 == cube else contact.geom2
                    touching.add(model.geom_bodyid[other])
            two_finger_contact = fingers <= touching
        held = held + 1 / FPS if env.success(model, data) else 0.0
        if held >= HOLD_S:
            break
    return {
        "success": held >= HOLD_S,
        "elapsed_s": round(float(data.time), 2),
        "reached_lift": bool(data.userdata[0] >= LIFT_MIN),
        "peak_lift_cm": round(float(data.userdata[0]) * 100, 2),
        "two_finger_contact": two_finger_contact,
        "closest_gripper_center_cm": round(closest_distance * 100, 2),
    }


def summarize(trials: list[dict], reference: dict) -> dict:
    successes = sum(trial["success"] for trial in trials)
    return {
        "reference": reference,
        "successes": successes,
        "trial_count": len(trials),
        "wilson_95": wilson(successes, len(trials)),
        "reached_lift": sum(trial["reached_lift"] for trial in trials),
        "two_finger_contact": sum(trial["two_finger_contact"] for trial in trials),
        "trials": trials,
    }


@torch.inference_mode()
def predict_act_chunk(env: SceneEnv, policy, pre, post, device: torch.device) -> np.ndarray:
    observation = pre(prepare_observation_for_inference(env.observe(), device, TASK))
    actions = policy.predict_action_chunk(observation)[:, : policy.config.n_action_steps]
    return np.stack([post(action).reshape(-1).cpu().numpy() for action in actions.transpose(0, 1)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compare-act", action="store_true", help="Also evaluate unchanged ACT on the same seeds")
    args = parser.parse_args()
    if args.seed < 0 or args.episodes < 1:
        parser.error("seed must be nonnegative and episodes must be positive")

    device = torch.device(args.device)
    policy, pre, post = load_policy(args.policy, args.device)
    config = policy.config
    if config.type != "act" or config.temporal_ensemble_coeff is not None or config.n_action_steps != 100:
        parser.error("the controller requires the validated ACT settings (100 actions, no temporal ensembling)")
    cameras = {
        key.removeprefix("observation.images."): tuple(feature.shape[1:])
        for key, feature in config.image_features.items()
    }
    env = SceneEnv(CUBE, cameras)

    def act_actions() -> Iterator[np.ndarray]:
        while True:
            yield from predict_act_chunk(env, policy, pre, post, device)

    try:
        env.reset(0)
        retargeted = RetargetedACT(env, predict_act_chunk(env, policy, pre, post, device))
        controllers = {"retargeted": retargeted.actions_for_start}
        if args.compare_act:
            controllers["act"] = act_actions
        reports = {}
        for name, actions in controllers.items():
            env.reset(0)
            reference = rollout(env, actions())
            if not reference["success"]:
                raise RuntimeError(f"{name} failed the fixed-pose reference")
            trials = []
            for seed in range(args.seed, args.seed + args.episodes):
                env.reset(seed)
                offsets = perturb_start(env, seed, "both")
                trial = {"seed": seed, **offsets, **rollout(env, actions())}
                trials.append(trial)
                print(json.dumps({"controller": name, **trial}), flush=True)
            reports[name] = summarize(trials, reference)
            print(f"{name}: {reports[name]['successes']}/{args.episodes}", flush=True)
    finally:
        env.close()

    report = {
        "policy": args.policy,
        "device": args.device,
        "versions": {"mujoco": mujoco.__version__, "lerobot": version("lerobot"), "torch": torch.__version__},
        "evaluation": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
