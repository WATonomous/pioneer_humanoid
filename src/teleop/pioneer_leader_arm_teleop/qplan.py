"""Q-Planning on a frozen ACT policy (after Giridhar et al., arXiv 2608.21204), for the Pioneer MuJoCo scenes.

The ACT policy stays frozen. At every planning step (once per action chunk) it proposes N candidate chunks
(its CVAE latent sampled from the prior instead of pinned to zero, optionally plus temporally smooth noise);
a small Q-function over (observation, chunk) scores them and the softmax-Q-weighted average is executed.
The Q-function is trained on demonstrations and on the planner's own rollouts, failures included.

This module: candidate proposals, chunk execution in the sim, the Q-network, and the rollout loop.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import nn

from sim_policy_eval import RENDER_SIZE, load_policy, observe  # noqa: E402
from synthetic_teleop import TeleopSim  # noqa: E402

FPS = int(os.environ.get("QPLAN_FPS", 25))      # the policy's control rate (QPLAN_FPS=50 for a 50 Hz dataset)
ROBOT = "pioneer_v1_left_arm"


class Proposer:
    """The frozen ACT policy as a proposal distribution over action chunks."""

    def __init__(self, path, device="cuda"):
        names = Path(path) / "subtask_state.json"
        spec = json.loads(names.read_text()) if names.exists() else None
        if isinstance(spec, dict) and spec.get("film"):       # checkpoint trained with train_film_act.py
            os.environ["ACT_FILM"] = "1"
            import film_act
            film_act._install()
        self.policy, self.pre, self.post, self.cfg, self.cameras = load_policy(path, device)
        self.device = torch.device(device)
        self.model = self.policy.model
        self.H = self.cfg.chunk_size
        # Multi-step scenes: the current instruction as a one-hot appended to the state (add_subtask_state.py).
        names = Path(path) / "subtask_state.json"
        self.subtasks = json.loads(names.read_text()) if names.exists() else None
        self._z = None
        self._feats = []
        self.model.encoder_latent_input_proj.register_forward_pre_hook(self._latent_hook)
        self.model.backbone.register_forward_hook(lambda m, i, o: self._feats.append(o["feature_map"]))
        zero = torch.zeros(1, 1, 7, device=self.device)
        with torch.inference_mode():
            self.a_mean = self.post(zero.clone()).reshape(7).to(self.device).float()
            self.a_std = (self.post(zero.clone() + 1).reshape(7).to(self.device).float() - self.a_mean)

    def _latent_hook(self, module, args):
        if self._z is not None:
            return (self._z,)

    def unnormalise(self, chunks):
        return chunks * self.a_std + self.a_mean

    @torch.inference_mode()
    def encode(self, state, images, instruction):
        """Normalised observation batch (B=1) for the policy."""
        import cv2
        from lerobot.policies.utils import prepare_observation_for_inference
        seen = {n: img if img.shape[:2] == tuple(self.cameras[n]) else
                cv2.resize(img, tuple(self.cameras[n])[::-1], interpolation=cv2.INTER_AREA) for n, img in images.items()}
        if self.subtasks is not None:
            from add_subtask_state import encode
            state = np.concatenate([state, encode(instruction, self.subtasks)]).astype(np.float32)
        if os.environ.get("QPLAN_BLANK"):                  # diagnostic: show the policy a black image from these cameras
            seen = {n: (img * 0 if n in os.environ["QPLAN_BLANK"].split(",") else img) for n, img in seen.items()}
        obs = {"observation.state": state, **{f"observation.images.{n}": img for n, img in seen.items()}}
        return self.pre(prepare_observation_for_inference(obs, self.device, instruction, ROBOT))

    @torch.inference_mode()
    def propose(self, batch, n, z_std=1.0, noise_std=0.0, noise_len=12.0, rng=None, offset_std=0.0):
        """n candidate chunks (n, H, 7), normalised action units; candidate 0 is the policy's own (z = 0, no noise).
        Also returns the observation features for the Q-function: (wrist map, ego map, normalised state)."""
        from lerobot.utils.constants import OBS_IMAGES
        g = rng if rng is not None else torch.Generator(device="cpu")
        z = torch.randn(n, self.cfg.latent_dim, generator=g) * z_std
        z[0] = 0
        rep = {k: (v.expand(n, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in batch.items()}
        rep[OBS_IMAGES] = [rep[k] for k in self.cfg.image_features]
        self._z, self._feats = z.to(self.device), []
        chunks = self.model(rep)[0]
        self._z = None
        feats = dict(zip([k.removeprefix("observation.images.") for k in self.cfg.image_features], [f[0] for f in self._feats]))
        if noise_std > 0:
            eps = torch.randn(n, self.H + 4 * int(noise_len), 7, generator=g)
            k = torch.exp(-0.5 * (torch.arange(-2 * int(noise_len), 2 * int(noise_len) + 1) / noise_len * 2) ** 2)
            k = (k / k.pow(2).sum().sqrt()).view(1, 1, -1)
            smooth = torch.nn.functional.conv1d(eps.transpose(1, 2).reshape(n * 7, 1, -1), k).reshape(n, 7, -1)[:, :, :self.H]
            smooth = smooth.transpose(1, 2) * noise_std
            smooth[0] = 0
            smooth[:, :, 6] = 0                      # leave the gripper channel alone
            chunks = chunks + smooth.to(self.device)
        if offset_std > 0:                               # one joint-space offset per candidate, eased in over 0.4 s
            delta = torch.randn(n, 1, 7, generator=g) * offset_std
            delta[0] = 0
            delta[:, :, 6] = 0
            ramp = torch.clamp(torch.arange(self.H) / 10.0, max=1.0).view(1, -1, 1)
            chunks = chunks + (delta * ramp).to(self.device)
        obs_feat = {"wrist": feats["wrist_left"].half().cpu(), "ego": torch.nn.functional.avg_pool2d(feats["ego"][None], 2)[0].half().cpu(),
                    "state": batch["observation.state"][0, :7].float().cpu()}
        return chunks.float(), obs_feat


PROBE = None      # optional callback(sim) after every executed frame (measurements; see peg_tilt_probe.py)


def observe_state(sim):
    """The 7-value state without rendering."""
    data = sim.data
    closure = sum((data.qpos[q] - o) / (c - o) for q, o, c in zip(sim.grip_qpos, sim.grip_open, sim.grip_closed)) / len(sim.grip_qpos)
    return np.append(data.qpos[sim.arm_qpos], min(max(closure, 0.0), 1.0))


def execute(sim, task, actions, held, hold_s, on_frame=None):
    """Run a chunk (H, 7) of real-unit actions, one per 1/FPS frame. Returns (frames run, success, held seconds).
    ``on_frame(sim)`` is called after every other frame (video)."""
    model, data = sim.model, sim.data
    lo, hi = model.jnt_range[[model.joint(j).id for j in sim.joints]].T
    ticks = round(1.0 / FPS / sim.control_dt)
    for i, action in enumerate(actions):
        g = min(max(float(action[6]), 0.0), 1.0)
        data.ctrl[sim.arm_acts] = np.clip(action[:6], lo, hi)
        data.ctrl[sim.grip_acts] = [o + g * (c - o) for o, c in zip(sim.grip_open, sim.grip_closed)]
        for _ in range(ticks):
            if sim.hook is not None:
                sim.hook(model, data)
            sim.mujoco.mj_step(model, data, nstep=sim.substeps)
        if on_frame is not None and i % 2 == 0:
            on_frame(sim)
        if PROBE is not None:
            PROBE(sim)
        held = held + 1.0 / FPS if task.success(model, data) else 0.0
        if held >= hold_s:
            return i + 1, True, held
    return len(actions), False, held


def rollout(scene, seed, proposer, select, n=16, z_std=1.0, noise_std=0.0, max_s=40.0, hold_s=0.4, rng=None, n_exec=None,
            offset_std=0.0, oracle=False, retry_after_s=0.0, rewind_s=4.0, retry_offset_std=0.005):
    """One episode under ``select(chunks, obs_feat, info) -> (chunk (H,7) normalised, extras dict)``.
    Returns a dict: success, frames, and the chunk-level transitions (obs features, executed chunk, frames run)."""
    from synthetic_tasks import TASKS
    task = TASKS[scene]
    # QPLAN_RENDER=240x320: render at the policy's own size (for data recorded at that size, not downscaled from 480x640).
    size = tuple(int(v) for v in os.environ["QPLAN_RENDER"].split("x")) if os.environ.get("QPLAN_RENDER") else RENDER_SIZE
    sim = TeleopSim(scene, seed, {c: size for c in proposer.cameras})
    n_exec = n_exec or proposer.H
    steps, frames, held, ok = [], 0, 0.0, False
    instruction = task.instruction
    video = on_frame = None
    if os.environ.get("QPLAN_VIDEO_DIR"):            # an mp4 per episode: ego | wrist, half speed frames (12.5 fps)
        import imageio
        Path(os.environ["QPLAN_VIDEO_DIR"]).mkdir(parents=True, exist_ok=True)
        video = imageio.get_writer(Path(os.environ["QPLAN_VIDEO_DIR"]) / f"{scene}_seed{seed}.mp4", fps=FPS / 2, macro_block_size=1)

        def on_frame(sim):
            imgs = observe(sim)[1]
            frame = np.concatenate([imgs[c] for c in sorted(imgs)], axis=1)
            video.append_data(frame if frame.shape[0] >= 480 else frame.repeat(2, 0).repeat(2, 1))
    history, since, retries = [], 0, 0          # executed actions; frames since the start or the last retry
    while frames < int(max_s * FPS):
        if retry_after_s and since >= int(retry_after_s * FPS) and len(history) > 1:
            # Stalled: retrace the last rewind_s of motion (the executed targets, backwards), then let the policy try again,
            # nudged by a small constant offset so it does not repeat the same unlucky trajectory.
            hist = np.array(history)
            still = np.abs(hist[:, :6] - hist[-1, :6]).max(1) < 0.003       # the tail where the arm just sat there pushing
            end = len(hist) - int(np.argmin(still[::-1])) if not still.all() else len(hist)
            back = hist[max(0, end - int(rewind_s * FPS)):end][::-1][: int(max_s * FPS) - frames]
            del history[max(0, end - int(rewind_s * FPS)):]
            ran, ok, held = execute(sim, task, back, held, hold_s)
            frames, since, retries = frames + ran, 0, retries + 1
            if ok or frames >= int(max_s * FPS):
                break
        if sim.progress is not None:
            index, _, instruction = sim.progress(sim.model, sim.data)
            if isinstance(proposer.subtasks, dict) and proposer.subtasks.get("mode") == "order":
                import importlib
                mod = importlib.import_module(f"humanoid_mujoco_scenes.{scene}.scene")
                if hasattr(mod, "_step_text"):          # every step's object word, as in the dataset (drawer_stow: open, colours, close)
                    from add_subtask_state import split
                    order = [split(mod._step_text(sim.data, k))[0][0] for k in range(proposer.subtasks["steps"])]
                else:
                    order = mod._order(sim.data)
                instruction = ",".join(order) + f"|{int(index)}"
        state, images = observe(sim)
        batch = proposer.encode(state, images, instruction)
        chunks, obs_feat = proposer.propose(batch, n, z_std, noise_std, rng=rng, offset_std=offset_std)
        if retries:
            chunks = proposer.propose(batch, 2, 0.0, rng=rng, offset_std=retry_offset_std)[0][1:]
        if oracle:                                       # try every candidate from this exact state (simulator reset)
            returns = lookahead(sim, task, proposer, chunks, frames, held, max_s, hold_s, instruction)
            best = int(np.argmax(returns))
            chunk, extra = chunks[best], {"cands": chunks.float().cpu(), "returns": torch.tensor(returns), "picked": best}
        else:
            chunk, extra = select(chunks, obs_feat, {"frame": frames, "seed": seed})
        actions = proposer.unnormalise(chunk)[:n_exec].cpu().numpy().astype(float)
        actions = actions[: int(max_s * FPS) - frames]
        ran, ok, held = execute(sim, task, actions, held, hold_s, on_frame)
        steps.append({"obs": obs_feat, "chunk": chunk.float().cpu(), "base": chunks[0].float().cpu(), "ran": ran, "frame": frames, **extra})
        if os.environ.get("QPLAN_KEEP_IMAGES"):          # what the policy saw at each planning step, for a contact sheet
            steps[-1]["img"] = np.concatenate([images[c][::2, ::2] for c in sorted(images)], axis=1)
            steps[-1]["instruction"] = instruction
        history.extend(actions[:ran])
        frames, since = frames + ran, since + ran
        if ok:
            break
    if video is not None:
        for _ in range(12):                          # hold the last view for a second
            on_frame(sim)
        video.close()
    diag = None
    if scene == "peg_insert":                    # how it ended: peg tilt (deg), height of its centre over the table, xy miss, grip
        m, d = sim.model, sim.data
        peg, block = d.body("peg"), d.body("hole_block")
        up = peg.xmat.reshape(3, 3)[:, 2]
        diag = {"tilt": float(np.degrees(np.arccos(min(1.0, abs(up[2]))))), "z": float(peg.xpos[2] - 0.705),
                "miss": float(np.linalg.norm(peg.xpos[:2] - block.xpos[:2])), "grip": float(observe_state(sim)[6])}
    done = None
    if sim.progress is not None:
        index, total, _ = sim.progress(sim.model, sim.data)
        done = (int(index), int(total))
    sim.close()
    return {"seed": seed, "success": bool(ok), "frames": frames, "steps": steps, "progress": done, "retries": retries, "diag": diag}


def lookahead(sim, task, proposer, chunks, frames, held, max_s, hold_s, instruction, gamma=0.997):
    """Counterfactual return of each candidate chunk from the current sim state: execute it, then follow the frozen
    policy to the end. Return = gamma^(frames to success), 0 on failure. The sim state is restored afterwards."""
    mj, model, data = sim.mujoco, sim.model, sim.data
    spec = mj.mjtState.mjSTATE_INTEGRATION
    snap = np.empty(mj.mj_stateSize(model, spec))
    mj.mj_getState(model, data, snap, spec)
    limit, returns = int(max_s * FPS), []
    for chunk in chunks:
        mj.mj_setState(model, data, snap, spec)
        mj.mj_forward(model, data)
        f, h, ok, nxt = frames, held, False, chunk
        while f < limit:
            actions = proposer.unnormalise(nxt).cpu().numpy().astype(float)[: limit - f]
            ran, ok, h = execute(sim, task, actions, h, hold_s)
            f += ran
            if ok or f >= limit:
                break
            state, images = observe(sim)
            nxt = proposer.propose(proposer.encode(state, images, instruction), 1)[0][0]
        returns.append(gamma ** (f - frames) if ok else 0.0)
    mj.mj_setState(model, data, snap, spec)
    mj.mj_forward(model, data)
    return returns


# ---------------------------------------------------------------------------------------------------------------------
# Q-function


class HLGauss:
    """Scalar value <-> categorical over a fixed grid of bins (Farebrother et al. 2024)."""

    def __init__(self, vmin=0.0, vmax=1.0, bins=51, sigma_ratio=0.75, device="cuda"):
        self.edges = torch.linspace(vmin, vmax, bins + 1, device=device)
        self.centres = (self.edges[:-1] + self.edges[1:]) / 2
        self.sigma = sigma_ratio * (vmax - vmin) / bins

    def probs(self, target):
        target = target.clamp(self.edges[0], self.edges[-1])
        cdf = torch.special.erf((self.edges[None] - target[:, None]) / (math.sqrt(2) * self.sigma))
        z = cdf[:, -1:] - cdf[:, :1]
        return (cdf[:, 1:] - cdf[:, :-1]) / z

    def value(self, logits):
        return (logits.softmax(-1) * self.centres).sum(-1)


class QNet(nn.Module):
    """Q(observation features, action chunk): a small transformer over [cls, state, image tokens, chunk tokens]."""

    def __init__(self, feat_dim=512, n_wrist=80, n_ego=20, H=100, patch=5, dim=256, layers=4, heads=4, bins=51, dropout=0.1,
                 delta_scale=0.03):
        super().__init__()
        self.patch, self.delta_scale = patch, delta_scale
        self.img = nn.Linear(feat_dim, dim)
        self.state = nn.Linear(7, dim)
        self.act = nn.Linear(patch * 14, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, 2 + n_wrist + n_ego + H // patch, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, bins))

    def obs_tokens(self, wrist, ego, state):
        """wrist (B, C, h, w), ego (B, C, h', w'), state (B, 7) -> (B, 1 + tokens, dim)."""
        w = self.img(wrist.flatten(2).transpose(1, 2).float())
        e = self.img(ego.flatten(2).transpose(1, 2).float())
        return torch.cat([self.state(state.float())[:, None], w, e], 1)

    def forward(self, obs_tok, chunks, base):
        """obs_tok (B, T, dim); chunks, base (B, H, 7): the candidate and the policy's own chunk at this observation
        (the candidate's offset from it is fed too, scaled up: candidates differ by milliradians). Returns bin logits."""
        b = chunks.shape[0]
        a = torch.cat([chunks.float(), (chunks.float() - base.float()) / self.delta_scale], -1)
        a = self.act(a.reshape(b, -1, self.patch * 14))
        x = torch.cat([self.cls.expand(b, -1, -1), obs_tok, a], 1) + self.pos
        return self.head(self.body(x)[:, 0])


def q_scores(qnet, hl, obs_feat, chunks, device="cuda"):
    """Q-values (n,) for n candidate chunks at one observation."""
    with torch.inference_mode():
        tok = qnet.obs_tokens(obs_feat["wrist"][None].to(device), obs_feat["ego"][None].to(device), obs_feat["state"][None].to(device))
        return hl.value(qnet(tok.expand(chunks.shape[0], -1, -1), chunks.to(device), chunks[:1].to(device).expand_as(chunks)))


def q_weighted(qnet, hl, temperature=0.02, device="cuda"):
    """The paper's selector: softmax(Q / temperature)-weighted average of the candidates."""
    def select(chunks, obs_feat, info):
        q = q_scores(qnet, hl, obs_feat, chunks, device)
        w = torch.softmax(q / temperature, 0)
        return (w[:, None, None] * chunks).sum(0), {"q": q.cpu(), "q_exec": float((w * q).sum())}
    return select


def best_of_n(qnet, hl, device="cuda"):
    def select(chunks, obs_feat, info):
        q = q_scores(qnet, hl, obs_feat, chunks, device)
        return chunks[int(q.argmax())], {"q": q.cpu(), "q_exec": float(q.max())}
    return select


def pick_first(chunks, obs_feat, info):
    """The frozen policy itself (candidate 0: latent at its mean, no noise)."""
    return chunks[0], {}


def pick_random(rng):
    """Behaviour policy for exploration data: a random candidate each planning step."""
    def select(chunks, obs_feat, info):
        return chunks[int(rng.integers(len(chunks)))], {}
    return select
