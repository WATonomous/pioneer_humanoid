"""Run the Q-Planning authors' code (github.com/varungiridhar/qplanning-code + their LeRobot fork) on a Pioneer MuJoCo scene.

Needs the ``qplan_official`` environment (their fork ``lerobot`` 0.4.4 and ``qplanning`` installed). Their planner
(``QWeightedPlanner``: top-K softmax-Q-weighted mean of N draws), their Q-function (``QFunctionPolicy``, trained with their
``lerobot-train --policy.type=q_function``), their online buffer and their fine-tune step run unmodified. This file only
connects them to our setup:
  - ``DPSampler``: a LeRobot Diffusion Policy as their ``BCChunkSampler`` (N draws from one observation encoding);
  - the rollout loop for ``TeleopSim`` in place of their LIBERO / RoboTwin gym loop;
  - the self-improvement driver around their ``save_iteration`` / ``finetune``.

    python qplan_official_pioneer.py eval --policy <dp>/pretrained_model [--q <q ckpt>/pretrained_model] --episodes 100
    python qplan_official_pioneer.py improve --policy <dp>/pretrained_model --q <q0>/pretrained_model --demos <dataset root> --out <dir>

Not the paper: the Q-function is built from their code at a reduced size (the model the checkpoint was trained with).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from synthetic_teleop import TeleopSim  # noqa: E402

FPS = 25
ROBOT = "pioneer_v1_left_arm"
REPO_ID = "humanoid/pioneer_v1_left_arm"


class DPSampler:
    """A frozen LeRobot Diffusion Policy as the planner's proposal distribution (qplanning.policies.base.BCChunkSampler)."""

    def __init__(self, path, device="cuda"):
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.policy = DiffusionPolicy.from_pretrained(str(path)).to(device).eval()
        self.cfg = self.policy.config
        self.pre, self.post = make_pre_post_processors(self.cfg, pretrained_path=str(path),
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.device = torch.device(device)
        self.chunk_size, self.action_dim = int(self.cfg.n_action_steps), 7
        self.cameras = {k.removeprefix("observation.images."): tuple(f.shape[-2:]) for k, f in self.cfg.input_features.items()
                        if k.startswith("observation.images.")}
        zero = torch.zeros(1, 1, 7, device=self.device)
        with torch.inference_mode():
            self.a_mean = self.post(zero.clone()).reshape(7).to(self.device).float()
            self.a_std = self.post(zero.clone() + 1).reshape(7).to(self.device).float() - self.a_mean
        self.history = deque(maxlen=self.cfg.n_obs_steps)

    def reset(self) -> None:
        self.history.clear()

    def observe(self, state, images, instruction) -> None:
        """Every control step: add this frame to the two-frame history the policy conditions on."""
        from lerobot.policies.utils import prepare_observation_for_inference
        obs = {"observation.state": state, **{f"observation.images.{n}": img for n, img in images.items()}}
        with torch.inference_mode():
            batch = self.pre(prepare_observation_for_inference(obs, self.device, instruction, ROBOT))
        while True:
            self.history.append(batch)
            if len(self.history) == self.cfg.n_obs_steps:
                break

    @torch.no_grad()
    def sample_chunks(self, batch, n_samples, *, denoise_steps=None, generator=None):
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
        stacked = {OBS_STATE: torch.stack([b[OBS_STATE] for b in self.history], dim=1),
                   OBS_IMAGES: torch.stack([torch.stack([b[k] for k in self.cfg.image_features], dim=1) for b in self.history], dim=1)}
        model = self.policy.diffusion
        cond = model._prepare_global_conditioning(stacked)            # encoded once, shared by all N draws
        noise = torch.randn(n_samples, self.cfg.horizon, self.action_dim, generator=generator,
                            device=generator.device if generator is not None else self.device).to(self.device)
        keep = model.num_inference_steps
        if denoise_steps:
            model.num_inference_steps = int(denoise_steps)
        try:
            actions = model.conditional_sample(n_samples, global_cond=cond.expand(n_samples, -1), noise=noise)
        finally:
            model.num_inference_steps = keep
        start = self.cfg.n_obs_steps - 1
        return actions[:, start:start + self.chunk_size].float()

    def unnormalize_actions(self, actions):
        return actions.to(self.device) * self.a_std + self.a_mean

    def to(self, device):
        return self

    def parameters(self):
        return self.policy.parameters()


class MTDiTSampler(DPSampler):
    """A frozen Multi-Task DiT policy (mtdit/, flow matching or diffusion) as the planner's proposal distribution."""

    def __init__(self, path, device="cuda"):
        import mtdit
        mtdit.install()
        from lerobot.policies.factory import make_pre_post_processors

        from mtdit.modeling_multi_task_dit import MultiTaskDiTPolicy
        self.policy = MultiTaskDiTPolicy.from_pretrained(str(path)).to(device).eval()
        self.cfg = self.policy.config
        self.pre, self.post = make_pre_post_processors(self.cfg, pretrained_path=str(path),
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.device = torch.device(device)
        self.chunk_size, self.action_dim = int(self.cfg.n_action_steps), 7
        self.cameras = {k.removeprefix("observation.images."): tuple(f.shape[-2:]) for k, f in self.cfg.input_features.items()
                        if k.startswith("observation.images.")}
        zero = torch.zeros(1, 1, 7, device=self.device)
        with torch.inference_mode():
            self.a_mean = self.post(zero.clone()).reshape(7).to(self.device).float()
            self.a_std = self.post(zero.clone() + 1).reshape(7).to(self.device).float() - self.a_mean
        self.history = deque(maxlen=self.cfg.n_obs_steps)

    @torch.no_grad()
    def sample_chunks(self, batch, n_samples, *, denoise_steps=None, generator=None):
        from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
        last = self.history[-1]
        stacked = {OBS_STATE: torch.stack([b[OBS_STATE] for b in self.history], dim=1),
                   OBS_IMAGES: torch.stack([torch.stack([b[k] for k in self.cfg.image_features], dim=-4) for b in self.history], dim=1),
                   OBS_LANGUAGE_TOKENS: last[OBS_LANGUAGE_TOKENS], OBS_LANGUAGE_ATTENTION_MASK: last[OBS_LANGUAGE_ATTENTION_MASK]}
        cond = self.policy.observation_encoder.encode(stacked)          # encoded once, shared by all N draws
        objective = self.policy.objective
        # Flow matching reads its step count from the config at each call; the diffusion objective keeps its own copy.
        owner, attr = (objective.config, "num_integration_steps") if self.cfg.is_flow_matching else (objective, "num_inference_steps")
        keep = getattr(owner, attr)
        if denoise_steps:
            setattr(owner, attr, int(denoise_steps))
        try:
            actions = objective.conditional_sample(self.policy.noise_predictor, n_samples, cond.expand(n_samples, -1))
        finally:
            setattr(owner, attr, keep)
        start = self.cfg.n_obs_steps - 1
        return actions[:, start:start + self.chunk_size].float()


def make_sampler(path, device="cuda"):
    """The proposal sampler for a checkpoint: Diffusion Policy or Multi-Task DiT, by the checkpoint's policy type."""
    kind = json.loads((Path(path) / "config.json").read_text()).get("type")
    return MTDiTSampler(path, device) if kind == "multi_task_dit" else DPSampler(path, device)


class ChunkPolicy:
    """The policy alone without the qplanning package installed: the same protocol as their QPlanningPolicy + NoPlanner
    (one draw with the full step budget, execute its first n_action_steps, replan when the queue runs dry)."""

    def __init__(self, sampler, n_action_steps, denoise_steps):
        self.sampler, self.n_action_steps, self.denoise_steps = sampler, n_action_steps, denoise_steps
        self.last_plan, self._queue = None, deque()

    def reset(self):
        self._queue.clear()
        self.sampler.reset()

    def select_action(self, batch):
        if not self._queue:
            chunk = self.sampler.sample_chunks(batch, 1, denoise_steps=self.denoise_steps)[0, : self.n_action_steps]
            self._queue.extend(chunk[i : i + 1] for i in range(chunk.shape[0]))
        return self._queue.popleft()


def build_policy(args, sampler, q_checkpoint):
    """Their QPlanningPolicy: the unguided baseline (NoPlanner) or their QWeightedPlanner on a Q checkpoint."""
    try:
        import qplanning  # noqa: F401
    except ImportError:
        if q_checkpoint:
            raise
        return ChunkPolicy(sampler, args.n_action_steps, args.baseline_denoise_steps)
    from qplanning.planner import NoPlanner, QWeightedPlanner
    from qplanning.policies.base import BCBundle
    from qplanning.policy import QPlanningPolicy
    bundle = BCBundle(sampler=sampler, preprocessor=None, postprocessor=None, policy_config=sampler.cfg, n_action_steps=args.n_action_steps)
    if not q_checkpoint:
        return QPlanningPolicy(bundle, NoPlanner(denoise_steps=args.baseline_denoise_steps), args.n_action_steps)
    cfg = SimpleNamespace(type="q_weighted", n_samples=args.n_samples, n_elites=args.n_elites, temperature=args.temperature,
                          denoise_steps=args.denoise_steps, seed=None)
    planner = QWeightedPlanner.build(cfg, q_checkpoint, bundle, device="cuda", seed=args.seed)
    return QPlanningPolicy(bundle, planner, args.n_action_steps)


def rollout(scene, seed, policy, sampler, max_s=40.0, hold_s=0.4, keep_frames=False):
    """One episode. Returns the dict their online buffer stores (action, images, success, task) plus frames run."""
    from synthetic_tasks import TASKS
    task = TASKS[scene]
    # Rendered at the policy's own size, or (RENDER=480x640) at the recording size and area-downscaled, for a policy trained on
    # downscale_dataset.py copies.
    size = tuple(int(v) for v in os.environ["RENDER"].split("x")) if os.environ.get("RENDER") else next(iter(sampler.cameras.values()))
    sim = TeleopSim(scene, seed, {c: size for c in sampler.cameras})
    model, data = sim.model, sim.data
    lo, hi = model.jnt_range[[model.joint(j).id for j in sim.joints]].T
    ticks = round(1.0 / FPS / sim.control_dt)
    policy.reset()
    actions, frames, held, ok = [], {f"observation.images.{c}": [] for c in sampler.cameras}, 0.0, False
    q_spread = []
    instruction = task.instruction
    # duplo: which brick the policy disturbs first (15 mm) vs the announced first brick (chance 1/3), as in order_probe.py.
    probe = None
    if scene == "duplo":
        import importlib
        S = importlib.import_module("humanoid_mujoco_scenes.duplo.scene")
        probe = {"order": list(S._order(data)), "p0": {c: data.body(f"brick_{c}").xpos.copy() for c in S.COLOURS}, "moved": None}
    for _ in range(int(max_s * FPS)):
        # Cameras are only needed when the planner is about to replan, and one step earlier (the policy conditions on
        # two frames); when frames are kept for the online buffer every step is rendered.
        if keep_frames or len(policy._queue) <= 1:
            closure = sum((data.qpos[q] - o) / (c - o) for q, o, c in zip(sim.grip_qpos, sim.grip_open, sim.grip_closed)) / len(sim.grip_qpos)
            state = np.append(data.qpos[sim.arm_qpos], min(max(closure, 0.0), 1.0)).astype(np.float32)
            images = sim.render_at(data.qpos.copy())
            if any(img.shape[:2] != sampler.cameras[c] for c, img in images.items()):
                import cv2
                images = {c: cv2.resize(img, sampler.cameras[c][::-1], interpolation=cv2.INTER_AREA) for c, img in images.items()}
            if sim.progress is not None:              # multi-step scenes: the current step's instruction, as recorded
                instruction = sim.progress(model, data)[2]
            sampler.observe(state, images, instruction)
            batch = {"task": [instruction]}
            for c, img in images.items():
                batch[f"observation.images.{c}"] = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float().div(255).to(sampler.device)
        action = sampler.unnormalize_actions(policy.select_action(batch))[0].cpu().numpy().astype(float)
        if policy.last_plan is not None and policy.last_plan.spread is not None and len(policy._queue) == policy.n_action_steps - 1:
            q_spread.append(policy.last_plan.spread.std)
        g = min(max(action[6], 0.0), 1.0)
        data.ctrl[sim.arm_acts] = np.clip(action[:6], lo, hi)
        data.ctrl[sim.grip_acts] = [o + g * (c - o) for o, c in zip(sim.grip_open, sim.grip_closed)]
        for _ in range(ticks):
            if sim.hook is not None:
                sim.hook(model, data)
            sim.mujoco.mj_step(model, data, nstep=sim.substeps)
        actions.append(torch.tensor(action, dtype=torch.float32))
        if keep_frames:                      # kept as uint8 (a float copy of a 40 s episode is ~1.8 GB)
            for c, img in images.items():
                frames[f"observation.images.{c}"].append(torch.from_numpy(img.copy()).permute(2, 0, 1))
        if probe is not None and probe["moved"] is None:
            for c, p0 in probe["p0"].items():
                if np.linalg.norm(data.body(f"brick_{c}").xpos - p0) > 0.015:
                    probe["moved"] = c
                    break
        held = held + 1.0 / FPS if task.success(model, data) else 0.0
        if held >= hold_s:
            ok = True
            break
    sim.close()
    episode = {"action": torch.stack(actions), "success": bool(ok), "task": task.instruction, "frames": len(actions),
               "q_std": float(np.mean(q_spread)) if q_spread else None,
               "first_moved_ok": None if probe is None or probe["moved"] is None else probe["moved"] == probe["order"][0]}
    if keep_frames:
        for k, v in frames.items():
            # Their save_episodes_lerobot flips H and W (a LIBERO convention). Our demonstrations are stored unflipped,
            # so flip here first: the two cancel and online frames match the demonstrations.
            episode[k] = torch.flip(torch.stack(v), dims=[2, 3])         # uint8; to_buffer_format() makes it float
    return episode


def evaluate(args, policy, sampler, seeds, label, keep_frames=False):
    episodes, t0 = [], time.time()
    video_dir = os.environ.get("QOFF_VIDEO_DIR")    # an mp4 per successful episode: ego | wrist, played at half speed
    for i, seed in enumerate(seeds):
        ep = rollout(args.scene, seed, policy, sampler, args.max_s, keep_frames=keep_frames or bool(video_dir))
        if video_dir and ep["success"]:
            import imageio
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            cams = sorted(k for k in ep if k.startswith("observation.images."))
            with imageio.get_writer(Path(video_dir) / f"{args.scene}_seed{seed}.mp4", fps=FPS / 2, macro_block_size=1) as video:
                for t in range(ep["frames"]):
                    frame = np.concatenate([torch.flip(ep[c][t], dims=[1, 2]).permute(1, 2, 0).numpy() for c in cams], axis=1)
                    video.append_data(frame.repeat(2, 0).repeat(2, 1))
        if video_dir and not keep_frames:
            for k in [k for k in ep if k.startswith("observation.images.")]:
                del ep[k]
        episodes.append(ep)
        if (i + 1) % 10 == 0 or i + 1 == len(seeds):
            print(f"[QOFF] {label}: {sum(e['success'] for e in episodes)}/{len(episodes)} ({time.time() - t0:.0f} s)", flush=True)
    ok = sum(e["success"] for e in episodes)
    qs = [e["q_std"] for e in episodes if e["q_std"] is not None]
    choice = [e["first_moved_ok"] for e in episodes if e.get("first_moved_ok") is not None]
    if choice:
        print(f"[QOFF] {label}: first brick moved = the announced one: {sum(choice)}/{len(choice)} (chance 1/3)", flush=True)
    print(f"[QOFF] {label}: success {ok / len(episodes):.1%} ({ok}/{len(episodes)}), mean episode {np.mean([e['frames'] for e in episodes]) / FPS:.1f} s"
          + (f", mean Q std across candidates {np.mean(qs):.4f}" if qs else ""), flush=True)
    return episodes, ok


def to_buffer_format(episode, cams):
    """The episode dict their save_episodes_lerobot expects: images as float [0, 1]."""
    out = {k: v for k, v in episode.items() if k in ("action", "success", "task")}
    for k in cams:
        out[k] = episode[k].float().div_(255)
    return out


def improve(args, sampler):
    """Their self-improvement loop (Alg. 1): planner rollouts -> their online buffer -> their Q-only fine-tune."""
    from qplanning.data.online_buffer import EPISODES_DIRNAME, iteration_dir, save_iteration
    from qplanning.modeling_qfunction import load_q_function
    from qplanning.self_improve import finetune
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = out / "status.txt"

    def note(line):
        print(line, flush=True)
        with status.open("a") as f:
            f.write(time.strftime("%H:%M:%S ") + line + "\n")

    heldout = list(range(args.seed, args.seed + args.heldout))
    info = json.loads((Path(args.demos) / "meta/info.json").read_text())
    image_shape = tuple(next(v["shape"] for k, v in info["features"].items() if k.startswith("observation.images.")))
    si = SimpleNamespace(finetune_steps=args.finetune_steps, batch_size=args.batch_size, online_fraction=0.5, finetune_lr=args.finetune_lr,
                         grad_clip_norm=10.0, offline=SimpleNamespace(repo_ids=[REPO_ID], root=str(args.demos)),
                         resolved_min_shards=lambda: 1)
    cfg = SimpleNamespace(self_improve=si)
    q_ckpt, step = str(args.q), 0
    if args.start == 0 and not args.skip_initial:
        _, ok = evaluate(args, build_policy(args, sampler, None), sampler, heldout, "policy alone")
        note(f"policy alone (their NoPlanner, {args.baseline_denoise_steps} denoise steps, {args.n_action_steps} of {sampler.chunk_size} executed): {ok}/{len(heldout)}")
        _, ok = evaluate(args, build_policy(args, sampler, q_ckpt), sampler, heldout, "offline Q")
        note(f"offline Q-Planning (Q trained on demonstrations only): {ok}/{len(heldout)}")
    if args.start > 0:
        q_ckpt = str(out / f"q_iter{args.start:02d}")
    for it in range(args.start + 1, args.iterations + 1):
        policy = build_policy(args, sampler, q_ckpt)
        seeds = [1_000_000 + 1000 * it + i for i in range(args.train_episodes)]
        cams = tuple(f"observation.images.{c}" for c in sampler.cameras)
        ok, t0 = 0, time.time()
        for i, seed in enumerate(seeds):     # one buffer shard per episode, so only one episode is ever held in memory
            ep = rollout(args.scene, seed, policy, sampler, args.max_s, keep_frames=True)
            ok += ep["success"]
            save_iteration([to_buffer_format(ep, cams)], iteration_dir(out, it, shard=i) / EPISODES_DIRNAME, camera_keys=cams,
                           image_shape=image_shape, action_dim=7,
                           manifest={"iteration": it, "q_checkpoint": q_ckpt, "seed": seed, "success": ep["success"]})
            del ep
        print(f"[QOFF] iteration {it} collection: {ok}/{len(seeds)} ({time.time() - t0:.0f} s)", flush=True)
        del policy
        torch.cuda.empty_cache()
        q_policy, _ = load_q_function(q_ckpt, device="cuda", eval_mode=False)
        new_ckpt = out / f"q_iter{it:02d}"
        loss, step = finetune(cfg, q_policy, out, new_ckpt, q_ckpt, global_step=step, iteration=None)
        del q_policy
        torch.cuda.empty_cache()
        q_ckpt = str(new_ckpt)
        stale = out / f"q_iter{it - 2:02d}"                 # keep the last two Q checkpoints (each is the whole model)
        if stale.exists():
            import shutil
            shutil.rmtree(stale)
        line = f"iteration {it}: planner collected {ok}/{args.train_episodes} on fresh layouts; fine-tune td_ce {loss:.4f}"
        if it % args.eval_every == 0 or it == args.iterations:
            _, ok = evaluate(args, build_policy(args, sampler, q_ckpt), sampler, heldout, f"iteration {it} held-out")
            line += f"; held-out {ok}/{len(heldout)}"
        note(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["eval", "improve"])
    p.add_argument("--scene", default="peg_insert")
    p.add_argument("--policy", required=True, help="Diffusion Policy pretrained_model directory")
    p.add_argument("--q", default=None, help="Q-function pretrained_model directory (their q_function policy)")
    p.add_argument("--demos", default=None, help="demonstration dataset root (improve)")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=1000, help="first held-out layout")
    p.add_argument("--episodes", type=int, default=100, help="eval: number of layouts")
    p.add_argument("--heldout", type=int, default=100, help="improve: held-out layouts evaluated")
    p.add_argument("--max_s", type=float, default=40.0)
    # The authors' defaults (configs/base.yaml): N = 64 draws, top K = 16, temperature 1, 3 denoise steps, execute 10 of H = 32.
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--n_elites", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--denoise_steps", type=int, default=3)
    p.add_argument("--baseline_denoise_steps", type=int, default=10)
    p.add_argument("--n_action_steps", type=int, default=10)
    # Self-improvement: 10 iterations, 200 Q-only steps at lr 1e-5, half of each batch online (theirs: batch 48, 100 episodes per iteration over 10 tasks).
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--start", type=int, default=0, help="resume after this iteration")
    p.add_argument("--train_episodes", type=int, default=30)
    p.add_argument("--finetune_steps", type=int, default=200)
    p.add_argument("--finetune_lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--skip_initial", action="store_true", help="skip the policy-alone and offline-Q evaluations")
    args = p.parse_args()

    sampler = make_sampler(args.policy)
    if args.cmd == "eval":
        policy = build_policy(args, sampler, args.q)
        evaluate(args, policy, sampler, range(args.seed, args.seed + args.episodes), "Q-Planning" if args.q else "policy alone")
    else:
        improve(args, sampler)


if __name__ == "__main__":
    main()
