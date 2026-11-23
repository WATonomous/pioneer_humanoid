"""Fit an ego-camera cube pose estimator on synthetic frames and score camera-fed ACT."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from humanoid_mujoco_scenes.rubiks_cube.scene import REST_Z

from eval_rubiks_retarget import CUBE, RetargetedACT, SceneEnv, cube_yaw, load_policy, predict_act_chunk, rollout, summarize
from eval_rubiks_poses import perturb_start


EGO_SIZE = (480, 640)
TRAIN_SEEDS = range(51000, 51800)
DEV_SEEDS = range(53000, 53100)
TEST_SUITES = {"original_range": (range(55000, 55100), 1), "wide_range": (range(56000, 56100), 3)}


def set_start(env: SceneEnv, seed: int, scale: int) -> dict:
    env.reset(seed)
    offsets = perturb_start(env, seed, "both")
    if scale != 1:
        start = env.model.joint(CUBE).qposadr[0]
        xy = np.asarray(offsets["cube_xy_m"])
        env.data.qpos[start:start + 2] += (scale - 1) * xy
        offsets["cube_xy_m"] = (scale * xy).tolist()
        offsets["cube_yaw_deg"] *= scale
        half_yaw = np.deg2rad(offsets["cube_yaw_deg"]) / 2
        env.data.qpos[start + 3:start + 7] = [np.cos(half_yaw), 0, 0, np.sin(half_yaw)]
        mujoco.mj_forward(env.model, env.data)
    return offsets


def ego_frame(env: SceneEnv) -> np.ndarray:
    renderer = env.renderers[EGO_SIZE]
    renderer.update_scene(env.data, "ego")
    return renderer.render()


def image_features(frame: np.ndarray) -> np.ndarray:
    # This fixed-camera crop covers the cube throughout the two benchmark ranges.
    red, green, blue = frame[215:398, 85:280].astype(np.int16).transpose(2, 0, 1)
    orange = (red - green >= 8) & (red - blue >= 12)
    green_face = (green - red >= 8) & (green - blue >= 6)
    values = []
    for mask in (orange, green_face):
        rows, cols = np.nonzero(mask)
        if rows.size < 4:
            values.extend(np.zeros(13))
            continue
        values.extend([
            np.log1p(rows.size),
            *np.percentile(rows, [5, 25, 50, 75, 95]),
            *np.percentile(cols, [5, 25, 50, 75, 95]),
            np.cov(rows, cols, bias=True)[0, 1],
            np.corrcoef(rows, cols)[0, 1] if rows.std() and cols.std() else 0.0,
        ])
    return np.asarray(values)


def design_matrix(values: np.ndarray, quadratic: bool) -> np.ndarray:
    matrix = np.column_stack((np.ones(len(values)), values))
    if quadratic:
        row, col = np.triu_indices(values.shape[1])
        matrix = np.column_stack((matrix, values[:, row] * values[:, col]))
    return matrix


def cube_pose(env: SceneEnv) -> np.ndarray:
    cube = env.model.body(CUBE).id
    return np.r_[env.data.xpos[cube, :2], cube_yaw(env.data, cube)]


def pose_errors(predicted: np.ndarray, actual: np.ndarray) -> dict:
    delta = predicted - actual
    xy_mm = np.linalg.norm(delta[:, :2], axis=1) * 1000
    yaw_deg = np.rad2deg(np.abs(delta[:, 2]))
    return {
        "median_xy_mm": float(np.median(xy_mm)),
        "p90_xy_mm": float(np.percentile(xy_mm, 90)),
        "median_yaw_deg": float(np.median(yaw_deg)),
        "p90_yaw_deg": float(np.percentile(yaw_deg, 90)),
    }


def collect(env: SceneEnv, specs) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for seed, scale in specs:
        set_start(env, seed, scale)
        features.append(image_features(ego_frame(env)))
        labels.append(cube_pose(env))
    return np.asarray(features), np.asarray(labels)


def fit(output: Path) -> None:
    env = SceneEnv(CUBE, {"ego": EGO_SIZE})
    try:
        train, targets = collect(env, ((seed, 1 if seed % 2 == 0 else 3) for seed in TRAIN_SEEDS))
        print(f"Rendered {len(train)} training frames", flush=True)
        development, truth = collect(env, ((seed, 1 if seed % 2 == 0 else 3) for seed in DEV_SEEDS))
    finally:
        env.close()

    center = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-6)
    train, development = (train - center) / scale, (development - center) / scale
    candidates = []
    for quadratic in (False, True):
        matrix = design_matrix(train, quadratic)
        validation = design_matrix(development, quadratic)
        penalty = np.eye(matrix.shape[1])
        penalty[0, 0] = 0
        for alpha in (0.1, 1.0, 10.0):
            weights = np.linalg.solve(matrix.T @ matrix + alpha * penalty, matrix.T @ targets)
            predicted = validation @ weights
            # Convert yaw error to displacement at the edge of the cube (about 3 cm from its center).
            error = np.linalg.norm((predicted - truth) * [1, 1, 0.03], axis=1)
            candidates.append((float(np.percentile(error, 90)), quadratic, alpha, weights, pose_errors(predicted, truth)))
    _, quadratic, alpha, weights, errors = min(candidates, key=lambda candidate: candidate[0])
    np.savez(output / "ego-pose.npz", center=center, scale=scale, quadratic=quadratic, weights=weights)
    report = {"training_seeds": list(TRAIN_SEEDS), "development_seeds": list(DEV_SEEDS), "quadratic": quadratic,
              "ridge_alpha": alpha, "development": errors}
    (output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["development"]), flush=True)


def evaluate(output: Path, policy_path: str) -> None:
    with np.load(output / "ego-pose.npz") as model:
        center, scale, weights = model["center"], model["scale"], model["weights"]
        quadratic = bool(model["quadratic"].item())
    report = {"policy": policy_path}
    weights_path = Path(policy_path) / "model.safetensors"
    if weights_path.is_file():
        with weights_path.open("rb") as weights_file:
            report["checkpoint_sha256"] = hashlib.file_digest(weights_file, "sha256").hexdigest()
    device = torch.device("cpu")
    policy, pre, post = load_policy(policy_path, str(device))
    cameras = {
        key.removeprefix("observation.images."): tuple(feature.shape[1:])
        for key, feature in policy.config.image_features.items()
    }
    env = SceneEnv(CUBE, cameras)

    def act_actions():
        while True:
            yield from predict_act_chunk(env, policy, pre, post, device)

    def camera_actions():
        features = (image_features(ego_frame(env)) - center) / scale
        estimate = (design_matrix(features[None, :], quadratic) @ weights)[0]
        position = np.r_[estimate[:2], REST_Z]
        return retargeted.actions_for_pose(position, estimate[2], env.data.qpos[env.arm_qpos]), estimate

    try:
        env.reset(0)
        retargeted = RetargetedACT(env, predict_act_chunk(env, policy, pre, post, device))
        reports = {}
        for name, (seeds, pose_scale) in TEST_SUITES.items():
            reports[name] = {}
            for controller in ("camera", "act", "exact"):
                env.reset(0)
                reference_actions = {
                    "camera": lambda: camera_actions()[0],
                    "act": act_actions,
                    "exact": retargeted.actions_for_start,
                }[controller]
                reference = rollout(env, reference_actions())
                trials = []
                estimates, truth = [], []
                for seed in seeds:
                    offsets = set_start(env, seed, pose_scale)
                    trial = {"seed": seed, **offsets}
                    if controller == "camera":
                        actions, estimate = camera_actions()
                        estimates.append(estimate)
                        truth.append(cube_pose(env))
                        trial["estimated_xy_m"] = estimate[:2].tolist()
                        trial["estimated_yaw_deg"] = float(np.rad2deg(estimate[2]))
                    else:
                        actions = reference_actions()
                    trials.append({**trial, **rollout(env, actions)})
                controller_report = summarize(trials, reference)
                if estimates:
                    controller_report["pose_errors"] = pose_errors(np.asarray(estimates), np.asarray(truth))
                reports[name][controller] = controller_report
                print(f"{name} {controller}: {controller_report['successes']}/{len(trials)}", flush=True)
    finally:
        env.close()
    report["evaluation"] = reports
    (output / "test.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", help="ACT pretrained_model directory for evaluate")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "fit":
        fit(args.output)
    elif not args.policy:
        parser.error("evaluate requires --policy")
    else:
        evaluate(args.output, args.policy)


if __name__ == "__main__":
    main()
