"""Read-only full-frame cargo dataset audit. Run with IsaacLab's Python."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/pioneer_humanoid"))
from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/nasa_cargo/synthetic_50")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/nasa_cargo/audit")
    args = parser.parse_args()
    root, output = args.dataset_root, args.output
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((root / "cargo_contract.json").read_text())
    manifest = [json.loads(line) for line in (root / "collection_manifest.jsonl").read_text().splitlines()]
    table = pq.read_table(sorted((root / "data").glob("chunk-*/file-*.parquet")))
    episodes = np.array(table["episode_index"])
    action = np.stack(table["action"].to_pylist())
    state = np.stack(table["observation.state"].to_pylist())
    timestamp = np.array(table["timestamp"])
    frame_index = np.array(table["frame_index"])
    limits = np.array([JOINT_POS_LIMITS[j] for j in contract["action_names"][:6]])
    assert np.isfinite(action).all() and np.isfinite(state).all()
    assert (action[:, :6] >= limits[:, 0] - 1e-6).all()
    assert (action[:, :6] <= limits[:, 1] + 1e-6).all()
    assert ((action[:, 6] >= 0) & (action[:, 6] <= 1)).all()
    checks = []
    thumbs = {}
    phase_names = ["approach", "close", "lift", "transfer", "release", "settle"]
    for row in manifest:
        episode = row["episode"]
        selected = episodes == episode
        a, s = action[selected], state[selected]
        phases = np.array(row["phases"])
        assert len(a) == row["frames"] == len(phases)
        assert np.array_equal(frame_index[selected], np.arange(len(a)))
        assert np.allclose(timestamp[selected], np.arange(len(a)) / 25, atol=1e-5)
        assert np.max(np.abs(np.diff(a[:, :6], axis=0))) <= .060001
        assert a[0, 6] < .05 and a[-1, 6] < .05
        assert np.all(a[np.isin(phases, ["lift", "transfer", "lower"]), 6] > .99)
        assert row["success"] and row["released"] and row["lifted"] and not row["failure"]
        chosen = [int(np.flatnonzero(phases == phase)[-1]) for phase in phase_names]
        chosen[0] = 0
        result = {
            "episode": episode, "seed": row["seed"], "frames": len(a),
            "max_command_step_rad": float(np.abs(np.diff(a[:, :6], axis=0)).max()),
            "arm_tracking_abs_p99_rad": float(np.quantile(np.abs(a[:, :6] - s[:, :6]), .99)),
            "final_measured_gripper_closure": float(s[-1, 6]),
            "max_ik_error_m": row["max_ik_error"],
            "videos": {},
        }
        for camera in contract["cameras"]:
            # Original writer finalizes each episode into its own numbered video.
            path = root / f"videos/observation.images.{camera}/chunk-000/file-{episode:03d}.mp4"
            last = None
            motion = []
            digest = hashlib.sha256()
            count = 0
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                assert float(stream.average_rate) == 25
                for index, frame in enumerate(container.decode(video=0)):
                    rgb = frame.to_ndarray(format="rgb24")
                    assert rgb.shape == (240, 320, 3)
                    assert int(rgb.max()) - int(rgb.min()) > 5, "Blank camera frame"
                    assert np.isclose(float(frame.pts * frame.time_base), index / 25, atol=.001)
                    small = rgb[::8, ::8].astype(np.float32)
                    if last is not None:
                        motion.append(float(np.abs(small - last).mean()))
                    last = small
                    digest.update(rgb.tobytes())
                    if index in chosen:
                        thumbs[(episode, camera, chosen.index(index))] = Image.fromarray(rgb)
                    count += 1
            assert count == len(a), (episode, camera, count, len(a))
            active = np.isin(phases[1:], ["approach", "descend", "lift", "transfer", "retreat"])
            active_motion = np.array(motion)[active]
            result["videos"][camera] = {
                "decoded_frames": count, "sha256_decoded": digest.hexdigest(),
                "active_frame_difference_mean_0_255": float(active_motion.mean()),
                "active_frozen_fraction": float((active_motion < .001).mean()),
            }
        checks.append(result)
        print(f"AUDIT episode={episode} frames={len(a)} tracking_p99={result['arm_tracking_abs_p99_rad']:.4f}", flush=True)
    for camera in contract["cameras"]:
        for start in range(0, len(manifest), 10):
            sheet = Image.new("RGB", (1280, 1640), "#202020")
            draw = ImageDraw.Draw(sheet)
            for col, label in enumerate(["start", "grasp", "lifted", "over tray", "released", "settled"]):
                draw.text((80 + col * 200, 10), label, fill="white")
            for row_index, episode in enumerate(range(start, min(start + 10, len(manifest)))):
                y = 35 + row_index * 160
                draw.text((5, y + 60), f"ep {episode}\nseed {manifest[episode]['seed']}", fill="white")
                for col in range(6):
                    sheet.paste(thumbs[(episode, camera, col)].resize((200, 150)), (80 + col * 200, y))
            sheet.save(output / f"{camera}_{start:02d}_{min(start+9,len(manifest)-1):02d}.jpg", quality=90)
    phase_counts = {phase: sum(row["phases"].count(phase) for row in manifest)
                    for phase in dict.fromkeys(manifest[0]["phases"])}
    # Align demonstrations by phase to distinguish timing variation from new paths.
    aligned = []
    for row in manifest:
        a = action[episodes == row["episode"]]
        phases = np.array(row["phases"])
        aligned.append(np.concatenate([
            np.stack([np.interp(np.linspace(0, 1, 20), np.linspace(0, 1, len(a[phases == phase])),
                                a[phases == phase, joint]) for joint in range(6)], axis=1)
            for phase in phase_counts
        ]))
    report = {"episodes": len(checks), "frames": len(action), "decoded_images": len(action)*2,
              "unique_action_trajectories": len({hashlib.sha256(action[episodes == i].tobytes()).hexdigest() for i in range(len(checks))}),
              "phase_frames": phase_counts,
              "phase_aligned_joint_std_mean_rad": np.std(aligned, axis=0).mean(axis=0).tolist(),
              "joint_tracking_abs_p99_rad": np.quantile(np.abs(action[:, :6] - state[:, :6]), .99, axis=0).tolist(),
              "duration_s": len(action)/25, "per_episode": checks}
    (output / "dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"CARGO_FULL_AUDIT_PASS episodes={len(checks)} decoded_images={len(action)*2}", flush=True)


if __name__ == "__main__":
    main()
