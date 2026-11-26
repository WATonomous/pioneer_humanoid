"""Closed-loop behaviour cloning benchmark: an ACT-style policy on top of each visual encoder, trained on the
synthetic peg_insert demos, then rolled out in MuJoCo on unseen layouts.

Policy (same for every encoder; only the vision tokens change):
  vision tokens (per (encoder, camera) stream: LayerNorm + Linear -> 256, + learned 2D position embedding)
  + proprio token (7: 6 joint angles + gripper closure)
  -> 4-layer transformer encoder -> 4-layer decoder with K=20 action queries (0.8 s at 25 Hz) -> L1 loss.
  (ACT without the CVAE: the synthetic operator is close to unimodal.)
At test time the policy is queried at 12.5 Hz (every new camera frame) and overlapping chunks are
temporally ensembled (ACT, w_i = exp(-m i)).

Encoders are frozen (features cached once) or fine-tuned end to end (``--finetune``, backbone lr 1e-5 as in ACT).

    .venv/bin/python bc.py --streams r18_imnet@ego+wrist_left --finetune --tag act_r18_ft
    .venv/bin/python bc.py --streams defm_s14@ego+wrist_left --tag defm_s14
    .venv/bin/python bc.py --streams dinov2_s14@ego+wrist_left,defm_s14@ego+wrist_left --tag fuse_dino_defm
"""
from __future__ import annotations

import argparse
import os
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
import dataset as D
import encoders as E
from probe import parse_config

K = 20              # action chunk (25 Hz steps)
TICKS_PER_ACT = C.RECORD_EVERY   # 100 Hz control ticks per 25 Hz action
ACTS_PER_IMG = 2    # 25 Hz actions per 12.5 Hz image


# --------------------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------------------
class ACTLite(nn.Module):
    def __init__(self, stream_shapes, d=256, layers=4, heads=8, dim=7, cvae=False, z_dim=32, state_extra=0, dec_layers=None):
        """dim: state / action size (C.ArmIO: 7 for the left 6-DoF arm, 16 for both 8dof arms).
        cvae: ACT's conditional VAE. In training a transformer encodes (state, true action chunk) into a style latent z
        (the operator's speed / timing for this take) that conditions the decoder; at test time z = 0 (the prior
        mean). Without it, L1 regression averages differently-timed takes into one blurred action."""
        super().__init__()
        # Dense streams (> 400 tokens, e.g. PioneerNet v3's stride-8 32x40 grid) are merged 2x2 by a learned
        # strided conv first: attention cost grows with tokens^2, and ACT itself reads a stride-32 map.
        self.merge = nn.ModuleList([nn.Conv2d(c, c, 2, 2) if h * w > 400 else nn.Identity() for c, h, w in stream_shapes])
        sz = [(h // 2, w // 2) if h * w > 400 else (h, w) for c, h, w in stream_shapes]
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(c), nn.Linear(c, d)) for c, h, w in stream_shapes])
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(1, h * w, d) * 0.02) for h, w in sz])
        self.state = nn.Linear(dim + state_extra, d)   # state_extra: one-hot instruction (--goal)
        self.state_pos = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.queries = nn.Parameter(torch.randn(1, K, d) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True),
                                         dec_layers or layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dim))
        self.cvae, self.z_dim, self.kl, self.kl_raw, self.kl_free = cvae, z_dim, None, None, 0.0
        if cvae:
            self.z_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
            self.z_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.z_state = nn.Linear(dim + state_extra, d)
            self.z_act = nn.Linear(dim, d)
            self.z_pos = nn.Parameter(torch.randn(1, K + 2, d) * 0.02)
            self.z_out = nn.Linear(d, 2 * z_dim)
            self.z_in = nn.Linear(z_dim, d)
            self.z_tok_pos = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, toks, state, actions=None):
        """actions: the true (normalised) chunk, training only (CVAE posterior); sets self.kl."""
        mem = [p(m(t).flatten(2).transpose(1, 2)) + pe for t, m, p, pe in zip(toks, self.merge, self.proj, self.pos)]
        extra = []
        if self.cvae:
            if actions is not None:
                h = torch.cat([self.z_cls.expand(len(state), -1, -1), self.z_state(state).unsqueeze(1), self.z_act(actions)], 1)
                mu, logvar = self.z_out(self.z_enc(h + self.z_pos)[:, 0]).float().chunk(2, -1)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                kl_dim = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).mean(0)          # per latent dim, batch mean
                self.kl_raw = kl_dim.sum()
                # free bits: each dim may carry up to kl_free nats at no cost, so z cannot collapse to the prior
                self.kl = torch.clamp(kl_dim, min=self.kl_free).sum() if self.kl_free > 0 else self.kl_raw
            else:
                z = torch.zeros(len(state), self.z_dim, device=state.device)
            extra = [self.z_in(z.to(self.z_in.weight.dtype)).unsqueeze(1) + self.z_tok_pos]
        mem = torch.cat(mem + extra + [self.state(state).unsqueeze(1) + self.state_pos], 1)
        mem = self.enc(mem)
        x = self.dec(self.queries.expand(len(state), -1, -1), mem)
        return self.out(x)


class Vision(nn.Module):
    """The (encoder, camera) streams. Frozen: tokens come from the cache. Fine-tuned: run the encoders."""

    def __init__(self, streams, finetune, pool=1):
        super().__init__()
        self.streams = streams
        names = sorted({e for e, _ in streams})
        self.encs = nn.ModuleDict({n: E.load(n) for n in names})
        self.pool = pool   # average-pool each token grid by this factor (--pool: e.g. PioneerNet's stride 8 -> 32)
        self.finetune = finetune
        if finetune:
            for e in self.encs.values():
                e.requires_grad_(True)   # BatchNorm stays in eval mode (frozen stats, as ACT's FrozenBatchNorm)

    def shapes(self):
        out = []
        dummy_rgb = torch.zeros(1, C.H, C.W, 3, dtype=torch.uint8, device="cuda")
        dummy_d = torch.ones(1, C.H, C.W, device="cuda")
        for e, c in self.streams:
            with torch.no_grad():
                t = self.run(e, dummy_rgb, dummy_d)
            out.append(tuple(t.shape[1:]))
        return out

    def mods(self):
        return set().union(*[E.needs(enc) for enc in self.encs.values()]) if self.encs else set()

    def forward(self, raw):
        """raw: {(modality, cam): tensor on cuda} (rgb uint8 (B,H,W,3), depth float metres) -> token grids."""
        return [self.run(e, raw.get(("rgb", c)), raw.get(("depth", c))) for e, c in self.streams]

    def run(self, e, rgb, depth):
        t = E.run(self.encs[e], rgb, depth)[1]
        return F.avg_pool2d(t, self.pool, ceil_mode=True) if self.pool > 1 else t


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
def build_samples(d):
    """For every image: proprio state at its 25 Hz step and the next K actions (padded with the last)."""
    states, chunks, masks = [], [], []
    for ep_i, s in enumerate(d["streams"]):
        a, st, idx = s["action"], s["state"], s["img_idx"]
        T = len(a)
        sel = d["t"][d["ep"] == ep_i]
        for t in idx[sel]:
            j = np.minimum(np.arange(t, t + K), T - 1)
            chunks.append(a[j])
            masks.append(np.arange(t, t + K) < T)
            states.append(st[t])
    return np.stack(states).astype(np.float32), np.stack(chunks).astype(np.float32), np.stack(masks)


def make_policy(shapes, norm, dim, cvae=False):
    """ACTLite for this run's inputs: with --history every stream appears twice (now, ~1 s ago; separate learned
    position embeddings tag the time) and the past joint state joins the state input; --goal adds the instruction."""
    hist = bool(norm.get("hist_rows"))
    size = norm.get("policy_size", {})   # --d_model / --enc_layers / --dec_layers (default ACTLite: 256, 4, 4)
    return ACTLite(list(shapes) * (2 if hist else 1), dim=dim, cvae=cvae,
                   state_extra=len(norm.get("goal_vocab", [])) + (dim if hist else 0),
                   d=size.get("d", 256), layers=size.get("layers", 4), dec_layers=size.get("dec_layers", 4))


def history_rows(d, gap):
    """For every loaded image: the row of the image `gap` rows earlier in the same episode (clamped at its start)."""
    ep = d["ep"]
    starts = np.r_[0, np.flatnonzero(np.diff(ep)) + 1]
    rows = np.arange(len(ep))
    return np.maximum(rows - gap, starts[ep])


def goal_onehot(texts, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    out = np.zeros((len(texts), len(vocab)), np.float32)
    for r, t in enumerate(texts):
        if t in idx:
            out[r, idx[t]] = 1.0
    return out


def goal_texts(d):
    """The instruction active at each training image, rebuilt from the episode's seed (which fixes the colour order
    in the scene's reset) and the recorded step index."""
    from synthetic_teleop import TeleopSim
    texts = []
    for ep_i, s in enumerate(d["streams"]):
        seed = int(s["file"].split("_")[-1][:-4])
        sim = TeleopSim(C.SCENE, seed, cameras=None, robot=C.ROBOT)
        texts += [C.instruction(sim.model, sim.data, k) for k in d["phase"][d["ep"] == ep_i]]
        sim.close()
    return texts


@torch.no_grad()
def cache_tokens(vision, d, noise=False):
    """Run each frozen (encoder, camera) stream once over all training images. Image arrays are dropped from
    ``d`` as soon as no later stream needs them, so peak RAM ~ images + one stream's tokens."""
    toks = []
    n = len(d["t"])
    for si, (e, c) in enumerate(vision.streams):
        enc = vision.encs[e]
        out = None
        gen = torch.Generator(device="cuda").manual_seed(11)
        for i in range(0, n, 128):
            raw = raw_batch(d, np.arange(i, min(n, i + 128)), vision, gen if noise else None, only=(e, c))
            t = vision.run(e, raw.get(("rgb", c)), raw.get(("depth", c)))
            if out is None:
                out = E.Q8(n, tuple(t.shape[1:]))
            out.put(i, t)
        toks.append(out)
        later = {(m, c2) for e2, c2 in vision.streams[si + 1:] for m in E.needs(vision.encs[e2])}
        for m, c2 in {(m, c) for m in E.needs(enc)} - later:
            d.pop(f"{c2}_{m}", None)
    return toks


def raw_batch(d, idx, vision, noise_gen=None, only=None):
    raw = {}
    for e, c in ([only] if only else vision.streams):
        for mod in E.needs(vision.encs[e]):
            if (mod, c) in raw:
                continue
            if mod == "rgb":
                raw[(mod, c)] = torch.from_numpy(d[f"{c}_rgb"][idx]).cuda(non_blocking=True)
            else:
                x = torch.from_numpy(d[f"{c}_depth"][idx].astype(np.float32)).cuda() / 1000
                if noise_gen is not None:
                    x = E.NOISE[c](x, noise_gen)
                raw[(mod, c)] = x
    return raw


def train(args, vision, d, norm):
    st, ch, mk = build_samples(d)
    if norm.get("delta"):
        # relative actions: joint targets as offsets from the current joint state (grippers stay absolute 0..1).
        # On the 7-DoF arm the operator's IK reaches one gripper pose with different elbow angles; averaging absolute
        # joint targets across such demos gives a pose none of them had, averaging offsets does not.
        jm = norm["joint_mask"]
        ch = ch - st[:, None, :] * jm
        norm["a_mu"], norm["a_sd"] = ch.reshape(-1, ch.shape[-1]).mean(0), ch.reshape(-1, ch.shape[-1]).std(0) + 1e-3
    st_n = (st - norm["s_mu"]) / norm["s_sd"]
    hist = history_rows(d, norm["hist_rows"]) if norm.get("hist_rows") else None
    if hist is not None:   # the joint state ~1 s ago, after the current one
        st_n = np.concatenate([st_n, st_n[hist]], 1)
    if norm.get("goal_vocab"):   # instruction one-hot after the normalised joint state
        st_n = np.concatenate([st_n, goal_onehot(d["goal_text"], norm["goal_vocab"])], 1)
    st_n = torch.from_numpy(st_n.astype(np.float32))
    ch_n = torch.from_numpy((ch - norm["a_mu"]) / norm["a_sd"])
    mk = torch.from_numpy(mk)
    shapes = vision.shapes()
    policy = make_policy(shapes, norm, st.shape[1], args.cvae).cuda()
    hist_t = torch.from_numpy(hist) if hist is not None else None
    policy.kl_free = args.kl_free
    params = [{"params": policy.parameters(), "lr": args.lr}]
    if args.finetune:
        params.append({"params": [p for p in vision.parameters() if p.requires_grad], "lr": args.lr_backbone})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    toks = None if args.finetune else cache_tokens(vision, d, noise=args.train_noise)
    if toks is not None and sum(t.nbytes() for t in toks) < 9e9 and not args.tokens_cpu:
        toks = [t.cuda() for t in toks]
    gen = torch.Generator(device="cuda").manual_seed(5)
    n = len(st)
    t0 = time.time()
    policy.train()
    # Resumable (2026-11-14: the GPU is shared and a job can be killed): the training state goes to <tag>.resume every
    # --ckpt_every steps and once training ends; rerunning the same command continues from it (or skips straight to
    # evaluation). main() removes it when the run completes.
    resume = C.OUT / "bc" / f"{args.tag}.resume"
    resume.parent.mkdir(exist_ok=True)
    step0 = 0
    if resume.exists():
        ck = torch.load(resume, map_location="cuda", weights_only=False)
        policy.load_state_dict(ck["policy"])
        if args.finetune:
            vision.load_state_dict(ck["vision"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu())
        torch.set_rng_state(ck["torch"].cpu())
        step0 = ck["step"]
        print(f"  resumed from {resume.name} at step {step0}", flush=True)

    def save_resume(step):
        torch.save(dict(policy=policy.state_dict(), vision=vision.state_dict() if args.finetune else None,
                        opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(), torch=torch.get_rng_state(),
                        step=step), str(resume) + ".tmp")
        os.replace(str(resume) + ".tmp", resume)
    for step in range(step0, args.steps):
        if args.ckpt_every and step > step0 and step % args.ckpt_every == 0:
            save_resume(step)
        idx = torch.randint(0, n, (args.bs,))
        if toks is None:
            raw = raw_batch(d, idx.numpy(), vision, gen if args.train_noise else None)
            tk = vision(raw)
            if hist_t is not None:
                tk = tk + vision(raw_batch(d, hist_t[idx].numpy(), vision, gen if args.train_noise else None))
        else:
            tk = [t.get(idx) for t in toks]
            if hist_t is not None:
                tk = tk + [t.get(hist_t[idx]) for t in toks]
            tk = [x.cuda(non_blocking=True) for x in tk]   # no-op when the cache is already on the GPU
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = policy(tk, st_n[idx].cuda(), ch_n[idx].cuda() if args.cvae else None)
        err = (pred.float() - ch_n[idx].cuda()).abs().mean(-1)
        m = mk[idx].cuda().float()
        loss = (err * m).sum() / m.sum()
        if args.cvae:
            loss = loss + args.kl_w * policy.kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == args.steps - 1:
            kl = f" kl {policy.kl_raw.item():.3f}" if args.cvae else ""
            print(f"  step {step:6d} loss {loss.item():.4f}{kl} [{time.time()-t0:.0f}s]", flush=True)
    if step0 < args.steps:
        save_resume(args.steps)   # training done: a rerun skips straight to evaluation
    policy.eval()
    del toks
    return policy


# --------------------------------------------------------------------------------------------------
# Closed-loop evaluation in MuJoCo
# --------------------------------------------------------------------------------------------------
class SharedRenderer:
    """One GL context for all eval envs (each mujoco.Renderer holds ~1 GB of GPU memory with the lightbox CAD).
    peg_insert layouts differ only in body_pos (the block) and qpos, so we copy those in and render."""

    def __init__(self, variant="clean"):
        import mujoco
        from synthetic_teleop import TeleopSim
        from gen_data import apply_variant
        self.mj = mujoco
        self.sim = TeleopSim(C.SCENE, 0, cameras={c: (C.H, C.W) for c in C.CAMS}, robot=C.ROBOT)
        self.m, self.d = self.sim.model, self.sim.data
        self.variant, self.apply_variant = variant, apply_variant
        self.base = {k: getattr(self.m, k).copy() for k in self.VAR_FIELDS}
        self.base_head = (self.m.vis.headlight.ambient.copy(), self.m.vis.headlight.diffuse.copy())
        self.looks = {}   # episode seed -> its shifted model fields
        self.r = mujoco.Renderer(self.m, C.H, C.W)
        import scenarios
        self.S = scenarios
        self.scn = variant in scenarios.SCENARIOS   # weather / environment scenario (scenarios.py)
        if self.scn:
            self.scn_base = {k: getattr(self.m, k).copy() for k in scenarios.LIGHT_FIELDS}
            self.scn_looks = {}

    # model fields apply_variant may change; restored and re-shifted per episode
    VAR_FIELDS = ("light_diffuse", "light_ambient", "light_pos", "geom_rgba", "mat_rgba", "cam_pos", "cam_quat")

    def _look(self, seed):
        """The shift for one episode, drawn exactly as gen_data.py draws it for test episodes (rng [seed, 7]),
        so every eval episode gets its own lighting / colours / camera pose, as in the probe's test sets."""
        if seed not in self.looks:
            m = self.m
            for k, v in self.base.items():
                getattr(m, k)[:] = v
            m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = self.base_head
            self.apply_variant(m, self.variant, np.random.default_rng([seed, 7]))
            self.looks[seed] = ({k: getattr(m, k).copy() for k in self.VAR_FIELDS},
                                (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy()))
        return self.looks[seed]

    def _render_scn(self, env, mods):
        """Scenario shifts: drawn once per episode on its own layout (rng [seed, 7]); distractors added to the
        rendered scene only; image-space effects (fog, glare, lens) applied per frame, to colour and depth alike."""
        m, S = self.m, self.S
        for k, v in self.scn_base.items():
            getattr(m, k)[:] = v
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = self.base_head
        m.body_pos[:] = env.m.body_pos
        self.d.qpos[:] = env.d.qpos
        self.mj.mj_forward(m, self.d)
        if env.seed not in self.scn_looks:
            self.scn_looks[env.seed] = S.make(self.variant, m, self.d, np.random.default_rng([env.seed, 7]))
        look = self.scn_looks[env.seed]
        S.apply_model(m, look)
        frng = np.random.default_rng([env.seed, 13, int(round(env.d.time * 1000))])
        need_d = "depth" in mods or self.variant in S.NEEDS_DEPTH
        out = {}
        for c in C.CAMS:
            self.r.update_scene(self.d, c)
            S.add_geoms(self.r.scene, look)
            rgb = self.r.render().copy()
            dep = None
            if need_d:
                self.r.enable_depth_rendering()
                self.r.update_scene(self.d, c)
                S.add_geoms(self.r.scene, look)
                dep = self.r.render().copy()
                self.r.disable_depth_rendering()
            rgb, dep = S.post(look, c, rgb, dep, frng)
            if "rgb" in mods:
                out[("rgb", c)] = rgb
            if "depth" in mods:
                out[("depth", c)] = dep.astype(np.float32)
        for k, v in self.scn_base.items():   # leave the model clean for render_demo
            getattr(m, k)[:] = v
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = self.base_head
        return out

    def render(self, env, mods):
        if self.scn:
            return self._render_scn(env, mods)
        if self.variant != "clean":
            fields, head = self._look(env.seed)
            for k, v in fields.items():
                getattr(self.m, k)[:] = v
            self.m.vis.headlight.ambient[:], self.m.vis.headlight.diffuse[:] = head
        self.m.body_pos[:] = env.m.body_pos
        self.d.qpos[:] = env.d.qpos
        self.mj.mj_forward(self.m, self.d)
        out = {}
        for c in C.CAMS:
            if "rgb" in mods:
                self.r.update_scene(self.d, c)
                out[("rgb", c)] = self.r.render().copy()
            if "depth" in mods:
                self.r.enable_depth_rendering()
                self.r.update_scene(self.d, c)
                out[("depth", c)] = self.r.render().copy()
                self.r.disable_depth_rendering()
        return out

    DEMO_HW = (480, 640)   # third-person demo view (video only; the policy never sees it)

    def render_demo(self, env):
        """Close third-person view of the peg and hole block, from the scene's own viewer angle."""
        if not hasattr(self, "r_demo"):
            self.r_demo = self.mj.Renderer(self.m, *self.DEMO_HW)
            self.cam = self.mj.MjvCamera()
            from humanoid_mujoco_scenes import scene_camera
            self.view = scene_camera(C.SCENE) or dict(lookat=[0.38, 0.29, 1.0], distance=0.6, azimuth=200, elevation=-35)
            self.hole_id = self.m.body("hole_block").id if C.SCENE == "peg_insert" else None
        self.m.body_pos[:] = env.m.body_pos
        self.d.qpos[:] = env.d.qpos
        self.mj.mj_forward(self.m, self.d)
        if self.hole_id is not None:   # peg_insert: close on the hole block
            self.cam.lookat[:] = self.d.xpos[self.hole_id] + np.array([0.0, 0.0, 0.03])
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.36, 200.0, -28.0
        else:                          # other tasks: the scene's own viewer camera
            v = self.view
            self.cam.lookat[:] = v["lookat"]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = v["distance"], v["azimuth"], v["elevation"]
        look = self.scn_looks.get(env.seed) if self.scn else None
        if look is not None:
            self.S.apply_model(self.m, look)
        self.r_demo.update_scene(self.d, self.cam)
        if look is not None:
            self.S.add_geoms(self.r_demo.scene, look)
        img = self.r_demo.render().copy()
        if look is not None:
            for k, v in self.scn_base.items():
                getattr(self.m, k)[:] = v
            self.m.vis.headlight.ambient[:], self.m.vis.headlight.diffuse[:] = self.base_head
        return img

    def close(self):
        self.r.close()
        if hasattr(self, "r_demo"):
            self.r_demo.close()


class EvalEnv:
    def __init__(self, seed, mods, renderer):
        import mujoco
        from humanoid_mujoco_scenes import scene_progress, scene_step, scene_success
        import humanoid_mujoco_scenes.peg_insert.scene as S
        from synthetic_teleop import TeleopSim
        self.mj, self.S = mujoco, S
        self.seed = seed
        self.peg_task = C.SCENE == "peg_insert"
        self.is_success = scene_success(C.SCENE)
        self.progress_fn = scene_progress(C.SCENE)   # multi-step tasks: (steps done, total, text); latched
        self.steps_done, self.steps_total = 0, 1
        self.sim = TeleopSim(C.SCENE, seed, cameras=None, robot=C.ROBOT)
        self.m, self.d = self.sim.model, self.sim.data
        self.io = C.ArmIO(self.m)
        self.renderer = renderer
        self.mods = mods
        self.hook = scene_step(C.SCENE)
        self.peg = self.m.body("peg").id if self.peg_task else None
        self.hole = self.m.body("hole_block").id if self.peg_task else None
        self.done, self.success, self.grasped, self.aligned = False, False, False, False
        self.min_align = 1.0

    def obs(self):
        out = self.renderer.render(self, self.mods) if self.renderer is not None else {}
        return out, self.io.state(self.d.qpos)

    def act(self, a0, a1=None):
        """One 25 Hz action = 4 control ticks, linearly interpolated toward the next action (the recorded
        operator targets change every 10 ms tick; a zero-order hold drops open-loop replay from 90% to 55%)."""
        sim, d = self.sim, self.d
        a1 = a0 if a1 is None else a1
        for k in range(TICKS_PER_ACT):
            self.io.apply(d, a0 + (a1 - a0) * (k / TICKS_PER_ACT))
            if self.hook is not None:
                self.hook(self.m, d)
            self.mj.mj_step(self.m, d, nstep=sim.substeps)
        S = self.S
        if not self.peg_task:   # other tasks: the scene's own success test (grasp / align are peg metrics)
            if self.progress_fn is not None:
                k, n, _ = self.progress_fn(self.m, d)
                self.steps_done, self.steps_total = max(self.steps_done, int(k)), int(n)
            if self.is_success(self.m, d):
                self.success = self.done = True
            return
        peg, hole = d.xpos[self.peg], d.xpos[self.hole]
        lifted = peg[2] - S.PEG_HEIGHT / 2 - S.TABLE_TOP_Z
        if lifted > 0.02:
            self.grasped = True
            off = float(np.linalg.norm(peg[:2] - hole[:2]))
            self.min_align = min(self.min_align, off)
            if off < 0.005:
                self.aligned = True
        if S.is_inserted(self.m, d):
            self.success = self.done = True

    def close(self):
        self.sim.close()


@torch.no_grad()
def evaluate(args, vision, policy, norm, seeds, max_s=25.0, test_noise=False, video=None, variant="clean", dark=False,
             video_n=4):
    mods = vision.mods()
    results = []
    max_s = norm.get("max_s", max_s)   # per task: 1.5x the longest demo (bc.py main)
    hist_gap = norm.get("hist_steps", 0)   # --history: image steps between now and the past frame
    exec_mode, grip_snap = getattr(args, "exec_mode", "ensemble"), getattr(args, "grip_snap", None)
    gen = torch.Generator(device="cuda").manual_seed(123)
    a_mu, a_sd = torch.tensor(norm["a_mu"]).cuda(), torch.tensor(norm["a_sd"]).cuda()
    renderer = SharedRenderer(variant)
    for b0 in range(0, len(seeds), args.eval_batch):
        envs = [EvalEnv(s, mods, renderer) for s in seeds[b0:b0 + args.eval_batch]]
        if hist_gap:
            import collections
            for e in envs:
                e.buf = collections.deque(maxlen=hist_gap + 1)   # buf[0]: hist_gap image steps ago (or the first)
        n_img = int(max_s * 12.5)
        hist = [[] for _ in envs]   # per env: list of (start_action_step, chunk)
        for it in range(n_img):
            live = [i for i, e in enumerate(envs) if not e.done]
            if not live:
                break
            obs = [envs[i].obs() for i in live]

            def to_raw(ims):
                raw = {}
                for key in ims[0]:
                    x = torch.from_numpy(np.stack([o[key] for o in ims])).cuda()
                    if key[0] == "depth" and test_noise:
                        x = E.NOISE[key[1]](x, gen)
                    if key[0] == "rgb" and dark:          # colour camera dead
                        x = torch.zeros_like(x)
                    raw[key] = x
                return raw
            raw = to_raw([o[0] for o in obs])
            if hist_gap:   # each env's observation hist_gap image steps ago (its first one early in the episode)
                for j, i in enumerate(live):
                    envs[i].buf.append(obs[j])
                past = [envs[i].buf[0] for i in live]
            st_raw = torch.from_numpy(np.stack([o[1] for o in obs])).cuda()
            st = (st_raw - torch.tensor(norm["s_mu"]).cuda()) / torch.tensor(norm["s_sd"]).cuda()
            if hist_gap:
                st_p = torch.from_numpy(np.stack([o[1] for o in past])).cuda()
                st = torch.cat([st, (st_p - torch.tensor(norm["s_mu"]).cuda()) / torch.tensor(norm["s_sd"]).cuda()], 1)
            if norm.get("goal_vocab"):   # each env's current instruction, from its own progress counter
                txt = [C.instruction(envs[i].m, envs[i].d,
                                     envs[i].progress_fn(envs[i].m, envs[i].d)[0] if envs[i].progress_fn else 0) for i in live]
                st = torch.cat([st, torch.from_numpy(goal_onehot(txt, norm["goal_vocab"])).cuda()], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = vision(raw) + (vision(to_raw([o[0] for o in past])) if hist_gap else [])
                pred = policy(toks, st).float() * a_sd + a_mu
            if norm.get("delta"):   # offsets -> absolute joint targets (the data contract's action)
                pred = pred + st_raw[:, None, :] * torch.tensor(norm["joint_mask"], device=pred.device)
            pred = pred.cpu().numpy()
            n_vid = min(video_n, len(envs)) if video is not None and b0 == 0 else 0
            def status(e): return dict(grasped=e.grasped, aligned=e.aligned, success=e.success, t=e.d.time)
            for vi in range(n_vid):
                if vi in live:
                    e = envs[vi]
                    video.append((vi, it, dict(obs[live.index(vi)][0], demo=renderer.render_demo(e)), status(e)))
            for j, i in enumerate(live):
                t_now = it * ACTS_PER_IMG
                hist[i].append((t_now, pred[j]))
                hist[i] = [(t0, c) for t0, c in hist[i] if t0 + K > t_now]

                def ens(t):
                    cand = [c[t - t0] for t0, c in hist[i] if 0 <= t - t0 < K]   # oldest first
                    if exec_mode == "newest":   # receding horizon: follow only the latest chunk
                        a = cand[-1].copy()
                    else:                       # ACT temporal ensembling, w_i = exp(-m i), oldest first
                        w = np.exp(-args.ens_m * np.arange(len(cand)))
                        a = (np.stack(cand) * w[:, None]).sum(0) / w.sum()
                    if grip_snap is not None:   # gripper fully open or fully closed, no partial loosening
                        a[envs[i].io.grip_idx] = (a[envs[i].io.grip_idx] > grip_snap).astype(a.dtype)
                    return a
                for k in range(ACTS_PER_IMG):
                    t = t_now + k
                    envs[i].act(ens(t), ens(t + 1))
                    if envs[i].done:
                        if i < n_vid:   # final frame at the moment of insertion
                            e = envs[i]
                            video.append((i, it + 1, dict(renderer.render(e, mods), demo=renderer.render_demo(e)), status(e)))
                        break
        for e in envs:
            results.append(dict(success=e.success, grasped=e.grasped, aligned=e.aligned, min_align_mm=e.min_align * 1000,
                                time=e.d.time, progress=(1.0 if e.success else e.steps_done / max(1, e.steps_total))))
            e.close()
        sr = np.mean([r["success"] for r in results])
        pr = np.mean([r["progress"] for r in results])
        print(f"  eval {len(results)}/{len(seeds)}: success {sr:.3f}  steps done {pr:.2f}", flush=True)
    renderer.close()
    return results


def summarize(res):
    n = len(res)
    s = np.array([r["success"] for r in res], float)
    p = s.mean()
    # Wilson 95% interval
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(n=n, success=p, ci_lo=c - h, ci_hi=c + h,
                grasped=float(np.mean([r["grasped"] for r in res])),
                aligned=float(np.mean([r["aligned"] for r in res])),
                time_success=float(np.mean([r["time"] for r in res if r["success"]])) if s.any() else None,
                progress=float(np.mean([r.get("progress", r["success"]) for r in res])))   # fraction of task steps done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--train_noise", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--ens_m", type=float, default=0.6,
                    help="temporal-ensembling weight w_i = exp(-m i), oldest first. 0.6 = best on validation layouts 8000+ "
                         "for both rigs (2026-11-12, m 0.1/0.3/0.6/1.0: armv2 v3 79/82/83/66%%, 8dof v4 12/36/46/35%%); "
                         "ACT's 0.1 was used for the Nov 11 runs")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eval_batch", type=int, default=25)
    ap.add_argument("--max_eps", type=int, default=None, help="limit the number of training demos")
    ap.add_argument("--noise_eval", action="store_true", help="also evaluate with sensor-noised depth")
    ap.add_argument("--shift_eval", action="store_true", help="also evaluate RGB policies under light / colour shift")
    ap.add_argument("--shift_episodes", type=int, default=50)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--cvae", action="store_true", help="ACT's conditional VAE (style latent; z = 0 at test time)")
    ap.add_argument("--kl_w", type=float, default=10.0, help="KL weight for --cvae (ACT uses 10)")
    ap.add_argument("--kl_free", type=float, default=0.0,
                    help="free bits (nats per latent dim) for --cvae. With ACT's KL weight 10 and our ~0.03 L1, the latent "
                         "collapsed (KL 0.000 on 2026-11-12); 0.5 keeps it in use")
    ap.add_argument("--train_split", default="train", help="demo split to train on (e.g. train_dr)")
    ap.add_argument("--train_variant", default="clean", help="image variant in that split (dr = domain-randomised)")
    ap.add_argument("--ckpt_every", type=int, default=2000, help="save resumable training state every N steps (0 = off)")
    ap.add_argument("--tokens_cpu", action="store_true",
                    help="keep the frozen-token cache in system RAM (one batch at a time on the GPU): fits when another job "
                         "holds most of the GPU; same training otherwise")
    ap.add_argument("--d_model", type=int, default=256, help="policy width (ACT: 512)")
    ap.add_argument("--enc_layers", type=int, default=4, help="policy encoder layers (ACT: 4)")
    ap.add_argument("--dec_layers", type=int, default=4, help="policy decoder layers (ACT: 7)")
    ap.add_argument("--pool", type=int, default=1,
                    help="average-pool every encoder's token grid by this factor before the policy (fewer, coarser tokens)")
    ap.add_argument("--history", type=float, default=0.0,
                    help="seconds of memory: also feed both cameras' images and the joint state from this long ago "
                         "(same episode, clamped at its start). 0 = single frame")
    ap.add_argument("--goal", action="store_true",
                    help="condition on the task's current instruction (one-hot over the instructions seen in training): "
                         "multi-step scenes shuffle the colour order per episode, so the next target is otherwise ambiguous")
    ap.add_argument("--delta", action="store_true",
                    help="relative actions: predict joint targets as offsets from the current joint state (converted "
                         "back to absolute targets before they are sent; grippers stay absolute)")
    ap.add_argument("--extra_split", default=None,
                    help="more demo episodes (e.g. train_more), used together with the standard train split")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every n-th image of each demo (2 with --extra_split keeps RAM at the 147-demo size)")
    args = ap.parse_args()
    C.contract_check()

    t0 = time.time()
    streams = parse_config(args.streams)
    vision = Vision(streams, args.finetune, args.pool)
    mods = vision.mods()
    sources = [(args.train_split, args.train_variant)] + ([(args.extra_split, "clean")] if args.extra_split else [])
    d = D.load_split(sources, every=args.every, depth="depth" in mods, rgb="rgb" in mods, max_eps=args.max_eps)
    st = np.concatenate([s["state"] for s in d["streams"]])
    ac = np.concatenate([s["action"] for s in d["streams"]])
    norm = dict(s_mu=st.mean(0), s_sd=st.std(0) + 1e-3, a_mu=ac.mean(0), a_sd=ac.std(0) + 1e-3,
                # eval time limit (s): peg_insert keeps its 25 s protocol; other tasks 1.5x the 90th-percentile demo
                max_s=25.0 if C.SCENE == "peg_insert" else
                max(25.0, 1.5 * float(np.percentile([len(s["action"]) for s in d["streams"]], 90)) / 25.0))
    if (args.d_model, args.enc_layers, args.dec_layers) != (256, 4, 4):
        norm["policy_size"] = dict(d=args.d_model, layers=args.enc_layers, dec_layers=args.dec_layers)
    if args.history > 0:   # in loaded rows (every args.every images at 12.5 Hz); eval uses the same time gap
        norm["hist_rows"] = max(1, round(args.history * 12.5 / args.every))
        norm["hist_steps"] = norm["hist_rows"] * args.every
    if args.goal:
        d["goal_text"] = goal_texts(d)
        norm["goal_vocab"] = sorted(set(d["goal_text"]))
        print(f"[{args.tag}] instruction vocabulary ({len(norm['goal_vocab'])}): {norm['goal_vocab']}", flush=True)
    if args.delta:
        jm = np.ones(st.shape[1], np.float32)
        jm[C.grip_indices()] = 0.0
        norm |= dict(delta=True, joint_mask=jm)
    print(f"[{args.tag}] {len(d['streams'])} demos, {len(d['t'])} images, streams {streams}, finetune={args.finetune} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    n_demos = len(d["streams"])
    policy = train(args, vision, d, norm)
    t_train = time.time() - t0
    del d
    seeds = list(range(7000, 7000 + args.episodes))
    rec = dict(tag=args.tag, streams=args.streams, finetune=args.finetune, train_noise=args.train_noise,
               steps=args.steps, train_secs=t_train, n_demos=n_demos, robot=C.ROBOT, cvae=args.cvae, ens_m=args.ens_m, delta=args.delta, goal=args.goal,
               history=args.history, pool=args.pool, policy_size=[args.d_model, args.enc_layers, args.dec_layers],
               train_split=args.train_split, train_variant=args.train_variant,
               kl_w=args.kl_w if args.cvae else None, kl_free=args.kl_free if args.cvae else None, every=args.every, extra_split=args.extra_split)
    video = [] if args.video else None
    res = evaluate(args, vision, policy, norm, seeds, video=video)
    rec["clean"] = summarize(res)
    rec["clean_raw"] = res
    print(f"[{args.tag}] CLEAN success {rec['clean']['success']:.3f} "
          f"[{rec['clean']['ci_lo']:.2f}, {rec['clean']['ci_hi']:.2f}] grasp {rec['clean']['grasped']:.2f} "
          f"align {rec['clean']['aligned']:.2f}", flush=True)
    if video:   # written now, so the demo video exists before the (long) shift evaluations
        import video_util
        video_util.rollout_video(video, C.FIG / f"rollout_{args.tag}.mp4", args.tag)
        video = None
    if args.shift_eval and "rgb" in mods:
        for v in ("light", "color"):
            res = evaluate(args, vision, policy, norm, seeds[:args.shift_episodes], variant=v)
            rec[v] = summarize(res)
            rec[v + "_raw"] = res
            print(f"[{args.tag}] {v.upper()} success {rec[v]['success']:.3f} [{rec[v]['ci_lo']:.2f}, {rec[v]['ci_hi']:.2f}]", flush=True)
    # held-out shift for every policy: both cameras re-mounted 1.5-2.5 cm / 2-4 deg off (RGB and depth change),
    # a different pose per episode
    res = evaluate(args, vision, policy, norm, seeds[:args.shift_episodes], variant="camera")
    rec["camera"], rec["camera_raw"] = summarize(res), res
    print(f"[{args.tag}] CAMERA success {rec['camera']['success']:.3f} [{rec['camera']['ci_lo']:.2f}, {rec['camera']['ci_hi']:.2f}]", flush=True)
    if {"rgb", "depth"} <= mods:
        res = evaluate(args, vision, policy, norm, seeds[:args.shift_episodes], dark=True)
        rec["dark"], rec["dark_raw"] = summarize(res), res
        print(f"[{args.tag}] DARK success {rec['dark']['success']:.3f} [{rec['dark']['ci_lo']:.2f}, {rec['dark']['ci_hi']:.2f}]", flush=True)
    if args.noise_eval and "depth" in mods:
        res = evaluate(args, vision, policy, norm, seeds[:args.shift_episodes], test_noise=True)
        rec["noise"] = summarize(res)
        rec["noise_raw"] = res
        print(f"[{args.tag}] NOISY-DEPTH success {rec['noise']['success']:.3f} "
              f"[{rec['noise']['ci_lo']:.2f}, {rec['noise']['ci_hi']:.2f}]", flush=True)
    rec["total_secs"] = time.time() - t0
    (C.OUT / "bc").mkdir(exist_ok=True)
    with open(C.OUT / "bc" / f"{args.tag}.json", "w") as f:
        json.dump(rec, f, indent=1, default=float)

    def half(sd): return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}   # disk is nearly full
    torch.save(dict(policy=half(policy.state_dict()), norm=norm, streams=args.streams, dim=len(norm["a_mu"]), robot=C.ROBOT, cvae=args.cvae,
                    pool=args.pool,
                    vision={k: half(v.state_dict()) for k, v in vision.encs.items()} if args.finetune else None),
               C.OUT / "bc" / f"{args.tag}.pt")
    resume = C.OUT / "bc" / f"{args.tag}.resume"
    if resume.exists():
        resume.unlink()
    print(f"[{args.tag}] done in {rec['total_secs']:.0f}s  (peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")


if __name__ == "__main__":
    main()
