"""PioneerNet: our own encoder. A single ResNet-18 that sees RGB + depth (6 input channels) and is distilled from
two frozen foundation teachers at once on the robot's own camera images:

    RGB teacher  : DINOv2 (semantic / appearance features)       <- student RGB head
    depth teacher: DeFM   (metric geometry features)             <- student depth head

"Denoising distillation": the teachers see the CLEAN image / CLEAN sim depth, the student sees a CORRUPTED
version (colour/brightness jitter and random relighting-like gains for RGB; D455 / wrist-stereo noise, holes and
min-range dropout for depth). The student therefore learns features that (a) carry both teachers' information,
(b) are invariant to exactly the shifts that break single-modality encoders, (c) come from one ACT-sized CNN
(drop-in for ACT's ResNet-18, ~2x its FLOPs on 6 channels... in practice the same trunk).

Output tokens: a stride-16 grid (layer3 + upsampled layer4, 14x18 at 224x288) with 512 channels -> the same
Encoder interface as encoders.py, registered as "pioneernet".

    .venv/bin/python distill.py --rgb_teacher dinov2_s14 --depth_teacher defm_s14 --steps 30000
"""
from __future__ import annotations

import argparse
import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
import dataset as D
import encoders as E

CKPT = C.ENC_DIR / "pioneernet.pt"   # per robot, shared by every task scene


class PioneerNet(nn.Module):
    """6-channel ResNet-18 trunk -> stride-16 fused token grid (+ global vector) and two distillation heads."""

    modality = "rgbd"

    def __init__(self, rgb_dim=384, depth_dim=384, d=512, init=True, stride=16, hw=(224, 288)):
        """stride 16 (v1/v2): layer3 + layer4 fused, 14x18 tokens at 224x288. stride 8 (v3): layer2 joins too,
        so each token covers 8x8 input pixels (32x40 tokens at 256x320, the full 320x240 frame)."""
        super().__init__()
        import torchvision
        rn = torchvision.models.resnet18(weights="IMAGENET1K_V1" if init else None)
        w_rgb = rn.conv1.weight.data.clone()
        conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        if init:
            from defm.model_factory import create_defm_model
            dm = create_defm_model("defm_resnet18", pretrained=True)
            w_d = [m for m in dm.modules() if isinstance(m, nn.Conv2d)][0].weight.data.clone()
            conv1.weight.data = torch.cat([w_rgb, w_d], 1) * 0.5
        rn.conv1 = conv1
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.l1, self.l2, self.l3, self.l4 = rn.layer1, rn.layer2, rn.layer3, rn.layer4
        self.stride = stride
        cin = 256 + 512 + (128 if stride == 8 else 0)
        self.fuse = nn.Sequential(nn.Conv2d(cin, d, 1), nn.GroupNorm(32, d), nn.GELU(), nn.Conv2d(d, d, 3, 1, 1))
        self.head_rgb = nn.Conv2d(d, rgb_dim, 1)
        self.head_depth = nn.Conv2d(d, depth_dim, 1)
        self.g_rgb = nn.Linear(d, rgb_dim)
        self.g_depth = nn.Linear(d, depth_dim)
        self.dim = self.gdim = d
        self.hw = tuple(hw)
        self.register_buffer("mean_rgb", torch.tensor(E.IMNET[0]).view(1, 3, 1, 1))
        self.register_buffer("std_rgb", torch.tensor(E.IMNET[1]).view(1, 3, 1, 1))
        from defm.utils.utils import DEFM_MEAN, DEFM_STD
        self.register_buffer("mean_d", torch.tensor(DEFM_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std_d", torch.tensor(DEFM_STD).view(1, 3, 1, 1))

    def prep(self, rgb_u8, depth_m):
        x = E._resize(rgb_u8.permute(0, 3, 1, 2).float() / 255.0, self.hw)
        x = (x - self.mean_rgb) / self.std_rgb
        z = (E._resize(E.defm_depth3(depth_m.float()), self.hw) - self.mean_d) / self.std_d
        return torch.cat([x, z], 1)

    def trunk(self, x):
        x = self.stem(x)
        f2 = self.l2(self.l1(x))
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        ref = f2 if self.stride == 8 else f3
        def up(f): return F.interpolate(f, size=ref.shape[2:], mode="bilinear", align_corners=False)
        parts = [f2, up(f3), up(f4)] if self.stride == 8 else [f3, up(f4)]
        return self.fuse(torch.cat(parts, 1))

    def forward(self, x):
        f = self.trunk(x)
        return f.mean((2, 3)), f

    def heads(self, f):
        g = f.mean((2, 3))
        return self.head_rgb(f), self.head_depth(f), self.g_rgb(g), self.g_depth(g)


# ---------------------------------------------------------------------------------------------------------
# Corruptions the student sees (the teachers see the clean inputs)
# ---------------------------------------------------------------------------------------------------------
def corrupt_rgb(x_u8: torch.Tensor, gen: torch.Generator, hue_mix: float = 0.35) -> torch.Tensor:
    """Photometric jitter that mimics relighting / colour shifts: per-image gain (with a colour cast), gamma,
    contrast, additive offset, per-channel hue-ish mixing, sensor noise. Input/Output uint8 (B,H,W,3)."""
    B = x_u8.shape[0]
    dev = x_u8.device
    x = x_u8.float() / 255.0

    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(B, *shape, generator=gen, device=dev)

    gain = U(0.35, 1.25, 1, 1, 1) * U(0.7, 1.15, 1, 1, 3)
    x = x * gain
    x = x.clamp(0, 1) ** U(0.7, 1.5, 1, 1, 1)
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * U(0.6, 1.3, 1, 1, 1) + mean + U(-0.08, 0.08, 1, 1, 3)
    # random channel mixing (colour change of whole objects is approximated by strong hue shifts). v1-v4: +-0.35.
    # It also erases which object is which colour, which colour-instructed tasks need ("put the red block in"):
    # v5 turns it off (hue_mix 0) and keeps only the per-channel colour cast of lighting.
    if hue_mix > 0:
        mix = torch.eye(3, device=dev).expand(B, 3, 3) + U(-hue_mix, hue_mix, 3, 3) * (torch.rand(B, 1, 1, generator=gen, device=dev) < 0.5)
        x = torch.einsum("bhwc,bdc->bhwd", x, mix)
    x = x + torch.randn(x.shape, generator=gen, device=dev) * U(0, 0.03, 1, 1, 1)
    return (x.clamp(0, 1) * 255).to(torch.uint8)


def _textures(B, H, W, gen, dev):
    """(B,H,W,3) random surface textures in [0,1]: two-colour gradient x low-frequency blotches x fine noise."""
    def R(*s):
        return torch.rand(*s, generator=gen, device=dev)
    c1, c2 = R(B, 1, 1, 3), R(B, 1, 1, 3)
    ang = R(B, 1, 1) * 2 * math.pi
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H, device=dev), torch.linspace(0, 1, W, device=dev), indexing="ij")
    ramp = ((xx * torch.cos(ang) + yy * torch.sin(ang)) % 1.0).unsqueeze(-1) * R(B, 1, 1, 1)
    blot = F.interpolate(R(B, 1, int(4 + 8 * R(1).item()), int(5 + 10 * R(1).item())), size=(H, W), mode="bilinear",
                         align_corners=False).permute(0, 2, 3, 1)
    tex = (c1 * (1 - ramp) + c2 * ramp) * (0.65 + 0.7 * blot * R(B, 1, 1, 1))
    return (tex + 0.04 * torch.randn(tex.shape, generator=gen, device=dev)).clamp(0, 1)


def _rects(B, K, H, W, gen, dev, smin, smax, border=False):
    """(B,K,H,W) bool masks of K random axis-aligned boxes (sides smin..smax of the image side); with border=True
    each box touches an image edge (panels / furniture behind and beside the workspace)."""
    def R(*s):
        return torch.rand(*s, generator=gen, device=dev)
    h, w = (smin + (smax - smin) * R(B, K)) * H, (smin + (smax - smin) * R(B, K)) * W
    y0, x0 = R(B, K) * (H - h), R(B, K) * (W - w)
    if border:   # snap one side to an edge
        side = (R(B, K) * 4).long()
        y0 = torch.where(side == 0, torch.zeros_like(y0), torch.where(side == 1, H - h, y0))
        x0 = torch.where(side == 2, torch.zeros_like(x0), torch.where(side == 3, W - w, x0))
    yy = torch.arange(H, device=dev).view(1, 1, H, 1)
    xx = torch.arange(W, device=dev).view(1, 1, 1, W)
    return ((yy >= y0[..., None, None]) & (yy < (y0 + h)[..., None, None]) &
            (xx >= x0[..., None, None]) & (xx < (x0 + w)[..., None, None]))


def corrupt_scene(rgb_u8: torch.Tensor, d: torch.Tensor, gen: torch.Generator, p: float = 0.5, box_max: float = 0.14):
    """v6: CONTENT changes, which the photometric corruptions above never produce. Per sample with prob p:
      background panels: 1-3 large textured boxes touching an image edge, drawn only where the scene is far
                         (depth above a random per-image quantile), placed in front of it in depth;
      distractors:       0-6 small textured boxes anywhere, a few cm in front of whatever they cover.
    The teachers still see the clean frame, so the student learns to report the task scene, not the new content.
    Generic 2-D pastes, deliberately NOT the simulator geoms the clutter/background test scenarios use.
    rgb uint8 (B,H,W,3), depth metres (B,H,W) -> both corrupted."""
    B, H, W = d.shape
    dev = d.device

    def R(*s):
        return torch.rand(*s, generator=gen, device=dev)
    x = rgb_u8.float() / 255.0
    d = d.clone()
    on = R(B, 1, 1) < p
    valid = (d > 0.05) & (d < 10.0)            # 65.535 m = no reading
    # background panels, behind the task: only on pixels farther than the image's q-quantile of valid depth
    q = 0.45 + 0.35 * R(B)
    dd = torch.where(valid, d, torch.full_like(d, float("nan"))).flatten(1)
    thr = torch.nanquantile(dd, q.clamp(0, 1), dim=1).diagonal().view(B, 1, 1).nan_to_num(10.0)
    far = valid & (d > thr)
    K = 3
    panels = _rects(B, K, H, W, gen, dev, 0.25, 0.7, border=True) & (R(B, K, 1, 1) < torch.tensor([1.0, 0.5, 0.3], device=dev).view(1, K, 1, 1))
    pm = panels.any(1) & far & on
    tex = _textures(B, H, W, gen, dev)
    x = torch.where(pm.unsqueeze(-1), tex, x)
    d = torch.where(pm, thr * (0.75 + 0.25 * R(B, 1, 1)), d)        # a flat panel in front of the old background
    # distractors: small boxes anywhere, a few cm proud of the surface under them
    K = 6
    n_k = (R(B, 1, 1, 1) * (K + 1)).floor()
    boxes = _rects(B, K, H, W, gen, dev, 0.04, box_max) & (torch.arange(K, device=dev).view(1, K, 1, 1) < n_k)
    tex2 = _textures(B, H, W, gen, dev)
    for k in range(K):   # later boxes on top
        m = boxes[:, k] & on
        x = torch.where(m.unsqueeze(-1), tex2.roll(shifts=k * 37, dims=2), x)
        lift = 0.01 + 0.06 * R(B, 1, 1)
        loc = torch.where(m & valid, d, torch.zeros_like(d)).flatten(1).amax(1).view(B, 1, 1)   # farthest covered point
        d = torch.where(m, (loc - lift).clamp(min=0.06), d)
    return (x * 255).to(torch.uint8), d


def corrupt_depth(d: torch.Tensor, cam: str, gen: torch.Generator) -> torch.Tensor:
    """Half the time the full sensor model, otherwise clean (the student must work on both)."""
    noisy = E.NOISE[cam](d, gen)
    keep = (torch.rand(len(d), 1, 1, generator=gen, device=d.device) < 0.5)
    return torch.where(keep, d, noisy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_teacher", default="dinov2_s14")
    ap.add_argument("--depth_teacher", default="defm_s14")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clean_frac", type=float, default=0.25, help="fraction of batches the student sees clean RGB")
    ap.add_argument("--out", default=str(CKPT))
    ap.add_argument("--max_eps", type=int, default=None)
    ap.add_argument("--stride", type=int, default=16, choices=(8, 16))
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--hw", type=int, nargs=2, default=(224, 288), help="student input size")
    ap.add_argument("--teacher_w", type=float, default=1.0,
                    help="weight of the teacher (distillation) losses; 0 = no-teacher ablation: the same RGB-D ResNet-18, "
                         "corruptions and modality dropout, trained on the geometry head alone (needs --geo > 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra_scenes", default=None,
                    help="other task scenes to distil on too, scene:every_nth_image, e.g. zip_tie:2,drawer_stow:4 "
                         "(teacher losses only; the geometry head keeps to peg_insert frames). Leave a task out to test "
                         "transfer to it")
    ap.add_argument("--ckpt_every", type=int, default=5000, help="save resumable training state every N steps (0 = off)")
    ap.add_argument("--hue_mix", type=float, default=0.35,
                    help="channel-mixing (hue rotation) strength of the RGB corruption; 0 keeps colour identity (v5)")
    ap.add_argument("--no_corrupt", action="store_true",
                    help="ablation: the student sees the same clean RGB and depth as the teachers (plain distillation)")
    ap.add_argument("--scene_aug", type=float, default=0.0,
                    help="v6: probability that the student's frame also gets content changes (background panels + "
                         "distractor boxes, corrupt_scene); v5 had photometric corruptions only and failed new backgrounds")
    ap.add_argument("--scene_keep_clean", action="store_true",
                    help="v7: the clean_frac batches get no scene corruption either (v6 applied it before the clean draw)")
    ap.add_argument("--scene_box_max", type=float, default=0.14,
                    help="largest distractor box side as a fraction of the image (v6: 0.14; big boxes can hide the peg)")
    ap.add_argument("--cam_aug", default=None,
                    help="split rendered with the camjit variant (gen_data.py): train on half fixed-camera, half "
                         "randomly-moved-camera frames (every 2nd image of each, so RAM stays at the v3 size)")
    ap.add_argument("--geo", type=float, default=0.0,
                    help="weight of a privileged geometry head: regress the simulator's TCP / peg / hole positions and "
                         "task phase from the student's tokens (train episodes only)")
    ap.add_argument("--moddrop", type=float, default=0.0,
                    help="per sample, blank the student's RGB (prob moddrop/2) or depth (prob moddrop/2) input. v1 without "
                         "it learned to predict DeFM's depth features from sim RGB alone and ignored its depth input")
    args = ap.parse_args()

    t0 = time.time()
    extra = []   # other task scenes' clean demos: teacher distillation only (they have no peg / hole labels)
    for item in filter(None, (args.extra_scenes or "").split(",")):
        sc, ev = item.split(":")
        extra.append((f"../{C.DATA.name}_{sc}/train", "clean", int(ev)))
    if args.cam_aug:
        d = D.load_split([("train", "clean", 2), (args.cam_aug, "camjit", 2)] + extra, max_eps=args.max_eps)
    elif extra:
        d = D.load_split([("train", "clean", 1)] + extra, max_eps=args.max_eps)
    else:
        d = D.load_split("train", every=1, max_eps=args.max_eps)
    imgs = {c: (d[f"{c}_rgb"], d[f"{c}_depth"]) for c in C.CAMS}
    n = len(d["t"])
    print(f"distill data: {n} frames x {len(C.CAMS)} cams [{time.time()-t0:.0f}s]", flush=True)
    t_rgb, t_d = E.load(args.rgb_teacher), E.load(args.depth_teacher)
    arch = dict(d=args.dim, stride=args.stride, hw=tuple(args.hw))
    student = PioneerNet(t_rgb.dim, t_d.dim, **arch).cuda()
    params = list(student.parameters())
    geo = None
    if args.geo > 0:
        import probe as P
        with torch.no_grad():
            _, f0 = student(student.prep(torch.zeros(1, C.H, C.W, 3, dtype=torch.uint8, device="cuda"),
                                         torch.ones(1, C.H, C.W, device="cuda")))
        ys = P.stack_targets(d)
        has_geo = np.isfinite(ys).all(1)          # peg_insert frames; the extra scenes have no peg / hole
        y_mu, y_sd = ys[has_geo].mean(0), ys[has_geo].std(0) + 1e-6
        ys = torch.from_numpy(np.nan_to_num((ys - y_mu) / y_sd).astype(np.float32))
        ph = torch.from_numpy(np.where(has_geo, d["phase"], 0)).long()
        has_geo = torch.from_numpy(has_geo)
        print(f"geometry labels on {int(has_geo.sum())} of {len(has_geo)} frames", flush=True)
        geo = P.AttentiveProbe([tuple(f0.shape[1:])], ys.shape[1], 5).cuda()
        params += list(geo.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    student.train()
    # Resumable: every --ckpt_every steps the full training state goes to <out>.resume (written atomically); a rerun
    # of the same command continues from it. A shared machine can stall or kill a 2.5 h run (2026-11-13).
    resume = args.out + ".resume"
    step0 = 0
    if os.path.exists(resume):
        ck = torch.load(resume, map_location="cuda", weights_only=False)
        student.load_state_dict(ck["student"])
        if geo is not None:
            geo.load_state_dict(ck["geo"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        np.random.set_state(ck["np"])
        torch.set_rng_state(ck["torch"].cpu())
        step0 = ck["step"]
        print(f"resumed from {resume} at step {step0}", flush=True)
    for step in range(step0, args.steps):
        if args.ckpt_every and step > step0 and step % args.ckpt_every == 0:
            torch.save(dict(student=student.state_dict(), geo=geo.state_dict() if geo is not None else None,
                            opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(), np=np.random.get_state(),
                            torch=torch.get_rng_state(), step=step), resume + ".tmp")
            os.replace(resume + ".tmp", resume)
        cam = C.CAMS[step % 2]
        idx = np.sort(np.random.randint(0, n, args.bs))
        rgb = torch.from_numpy(imgs[cam][0][idx]).cuda()
        dep = torch.from_numpy(imgs[cam][1][idx].astype(np.float32)).cuda() / 1000
        # random-resized-crop-ish shift (same crop for teacher and student): small translations
        if args.teacher_w > 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gr, tr = t_rgb(t_rgb.prep(rgb))
                gd, td = t_d(t_d.prep(dep))
        if args.no_corrupt:
            s_rgb, s_dep = rgb, dep
        else:
            if args.scene_keep_clean:   # v7: clean batches stay fully clean (v6 scene-corrupted them too: 12% clean)
                if np.random.rand() < args.clean_frac:
                    s_rgb, s_dep = rgb, dep
                else:
                    s_rgb, s_dep = (corrupt_scene(rgb, dep, gen, args.scene_aug, args.scene_box_max)
                                    if args.scene_aug > 0 else (rgb, dep))
                    s_rgb = corrupt_rgb(s_rgb, gen, args.hue_mix)
                s_dep = corrupt_depth(s_dep, cam, gen)
            else:
                s_rgb, s_dep = (corrupt_scene(rgb, dep, gen, args.scene_aug, args.scene_box_max)
                                if args.scene_aug > 0 else (rgb, dep))
                s_rgb = s_rgb if np.random.rand() < args.clean_frac else corrupt_rgb(s_rgb, gen, args.hue_mix)
                s_dep = corrupt_depth(s_dep, cam, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = student.prep(s_rgb, s_dep)
            if args.moddrop > 0:   # blanked input = 0 after normalisation (the dataset mean)
                u = torch.rand(len(x), 1, 1, 1, device="cuda")
                x = torch.cat([x[:, :3] * (u >= args.moddrop / 2), x[:, 3:] * ((u < args.moddrop / 2) | (u >= args.moddrop))], 1)
            f = student.trunk(x)
            pr, pd, pgr, pgd = student.heads(f)

        def tok_loss(p, t):
            p = F.interpolate(p.float(), size=t.shape[2:], mode="bilinear", align_corners=False)
            p, t = p.flatten(2), t.float().flatten(2)
            t = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-4)   # per-token standardise
            return (1 - F.cosine_similarity(p, t, dim=1)).mean() + 0.5 * F.smooth_l1_loss(p, t)

        if geo is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                py, pc = geo([f])
            ti = torch.from_numpy(idx)
            m = has_geo[ti].cuda()
            if m.any():
                gl = F.smooth_l1_loss(py.float()[m], ys[ti].cuda()[m], beta=0.05) + \
                    0.2 * F.cross_entropy(pc.float()[m], ph[ti].cuda()[m])
            else:
                gl = torch.zeros((), device="cuda")
        else:
            gl = torch.zeros((), device="cuda")
        loss = args.geo * gl
        if args.teacher_w > 0:
            loss = loss + args.teacher_w * (
                tok_loss(pr, tr) + tok_loss(pd, td) +
                0.5 * (1 - F.cosine_similarity(pgr.float(), gr.float(), dim=1)).mean() +
                0.5 * (1 - F.cosine_similarity(pgd.float(), gd.float(), dim=1)).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps - 1:
            print(f"  step {step:6d} loss {loss.item():.4f} geo {gl.item():.4f} [{time.time()-t0:.0f}s]", flush=True)
    torch.save(dict(state=student.state_dict(), rgb_teacher=args.rgb_teacher, depth_teacher=args.depth_teacher,
                    rgb_dim=t_rgb.dim, depth_dim=t_d.dim, arch=arch, geo=args.geo, moddrop=args.moddrop, cam_aug=args.cam_aug,
                    teacher_w=args.teacher_w, no_corrupt=args.no_corrupt, hue_mix=args.hue_mix, extra_scenes=args.extra_scenes, steps=args.steps, seed=args.seed, scene_aug=args.scene_aug,
                    scene_keep_clean=args.scene_keep_clean, scene_box_max=args.scene_box_max), args.out)
    print("saved", args.out, f"[{time.time()-t0:.0f}s]")
    if os.path.exists(resume):
        os.remove(resume)


if __name__ == "__main__":
    main()
