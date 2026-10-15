"""Roll out a policy trained with the Patch Policy authors' trainer (configs/train_pioneer_peg.yaml) in our MuJoCo scene.

    MUJOCO_GL=egl python rollout.py --snapshot <run dir>/snapshot.pt --episodes 200 --seed 1000 --out result.json

Run with the repo's own .venv (our scenes need its MuJoCo); the authors' model classes are imported from their
repository, and the control loop follows their train_policy.py eval_on_env for action_window_size 1: a window of
the last ``window`` encoded observations, the action read from the last step, one action executed per 1/25 s frame.
Scoring is sim_policy_eval.py's: the scene's success check must hold for 0.4 s within --max_s.
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "patch_policy"))
sys.path.insert(0, str(REPO / "src/teleop/pioneer_leader_arm_teleop"))

import cv2  # noqa: E402
import einops  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

VIEWS = ("ego", "wrist_left")


def frame_to_views(images, size):
    """As the training data was made: 640x480 render -> 320x240 (area) -> size x size (area), (V, C, H, W) in 0..1."""
    out = []
    for name in VIEWS:
        small = cv2.resize(images[name], (320, 240), interpolation=cv2.INTER_AREA)
        out.append(cv2.resize(small, (size, size), interpolation=cv2.INTER_AREA))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--scene", default="peg_insert")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max_s", type=float, default=40.0)
    p.add_argument("--hold_s", type=float, default=0.4)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--action_window", type=int, default=1, help="the checkpoint's action_window_size (chunk length)")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--out", default=None)
    p.add_argument("--videos", type=int, default=0, help="save the first N episodes (both cameras) as mp4 next to --out")
    a = p.parse_args()

    import sim_policy_eval as E
    from models.encoder.dino import DinoV2Encoder
    from synthetic_tasks import TASKS

    saved = torch.load(a.snapshot, map_location="cuda", weights_only=False)
    model = saved["model"] if isinstance(saved, dict) else saved
    model.cuda()
    model.eval()            # (their eval() does not return the module)
    epoch = saved.get("epoch") if isinstance(saved, dict) else None
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens").cuda().eval()
    task, fps, results, t0 = TASKS[a.scene], 25, [], time.time()
    for i in range(a.episodes):
        seed = a.seed + i
        sim = E.TeleopSim(a.scene, seed, {n: E.RENDER_SIZE for n in VIEWS})
        m, d = sim.model, sim.data
        lo, hi = m.jnt_range[[m.joint(j).id for j in sim.joints]].T
        stack, held, ok = deque(maxlen=a.window), 0.0, False
        chunks = []                                   # their eval_on_env for action_window_size > 1
        video = None
        if i < a.videos:
            import imageio
            stem = Path(a.out) if a.out else Path(a.snapshot)
            video = imageio.get_writer(str(stem.with_name(f"{stem.stem}_seed{seed}.mp4")), fps=fps, macro_block_size=1)
        for _ in range(int(a.max_s * fps)):
            _, images = E.observe(sim)
            if video is not None:
                video.append_data(np.concatenate([images[n] for n in VIEWS], axis=1))
            with torch.no_grad():
                obs = frame_to_views(images, a.size).cuda()[None]                        # N V C H W
                stack.append(einops.rearrange(encoder(obs), "N V P E -> N (V P) E"))
                window = einops.rearrange(torch.stack(tuple(stack)), "T N P E -> N T P E")
                goal = torch.ones(1, window.shape[1], 1, 1, 1, device="cuda")
                pred = model(window, goal, None)[0]
            if a.action_window == 1:
                action = pred[:, -1, 0, :][0].float().cpu().numpy()
            else:                                     # average the overlapping chunks of the last action_window queries,
                chunks.append(pred[:, -1].float().cpu().numpy())          # each shifted by one step per frame (theirs)
                chunks = chunks[-a.action_window:]
                action = np.array(chunks).mean(axis=0)[:, 0][0]
                chunks = [np.concatenate((c[:, 1:], np.zeros((c.shape[0], 1, c.shape[-1]))), axis=1) for c in chunks]
            g = float(np.clip(action[6], 0.0, 1.0))
            d.ctrl[sim.arm_acts] = np.clip(action[:6], lo, hi)
            d.ctrl[sim.grip_acts] = [o + g * (c - o) for o, c in zip(sim.grip_open, sim.grip_closed)]
            for _ in range(round(1.0 / fps / sim.control_dt)):
                if sim.hook is not None:
                    sim.hook(m, d)
                sim.mujoco.mj_step(m, d, nstep=sim.substeps)
            held = held + 1.0 / fps if task.success(m, d) else 0.0
            if held >= a.hold_s:
                ok = True
                break
        if video is not None:
            video.close()
        results.append({"seed": seed, "success": ok, "sim_s": round(float(d.time), 2)})
        sim.close()
        print(f"[ROLLOUT] seed {seed}: {'ok  ' if ok else 'FAIL'} {d.time:5.1f} s   running "
              f"{sum(r['success'] for r in results)}/{len(results)} ({time.time() - t0:.0f} s wall)", flush=True)
    n = sum(r["success"] for r in results)
    out = Path(a.out) if a.out else Path(a.snapshot).with_name(f"rollout_{a.scene}.json")
    out.write_text(json.dumps({"scene": a.scene, "snapshot": str(a.snapshot), "epoch": epoch, "episodes": len(results),
                               "success": n, "results": results}, indent=1))
    print(f"[ROLLOUT] {a.scene}: {n}/{len(results)} (snapshot epoch {epoch}) -> {out}")


if __name__ == "__main__":
    main()
