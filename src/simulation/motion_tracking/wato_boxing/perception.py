"""Opponent perception for the fight policy: simulated RGB camera + ultrasonic
range finder, fused by an extended Kalman filter (EKF), batched on the GPU.

  RGB camera (D455 site on the shoulder-frame bar)
     per target: pixel position (u, v) and apparent radius r_px
     -> bearing, and a rough range from the target's known size
     (optional: the D455's depth, valid 0.4-6 m)
  Ultrasonic (chest site)
     distance to the nearest thing inside its cone (the robot's own guard
     included) -> assigned to the tracked target that best explains it, or
     ignored when none does
  EKF, one per target (opponent head, left glove, right glove, torso):
     state = world position + velocity, constant-velocity model;
     predict every step, update with whatever measurement arrived.
  Policy observation, per target, in the observer's heading frame:
     position, velocity, position uncertainty (m), time since last seen (s).

The sensors are simulated from geometry rather than rendered images (rendering
for thousands of envs is too slow): targets are projected through a pinhole
model with the D455's field of view; a target is hidden when its line of
sight passes through a glove or arm of either fighter (own guard included);
detections drop out at random, arrive late (latency) and at the sensor's rate.
The camera's own pose is taken as known (the real robot gets it from its IMU
and joint encoders).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from wato_boxing.fighters import CAMERA_SITE, GLOVE_RADIUS, HEAD_RADIUS, ULTRASONIC_SITE

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

TARGETS = ("head", "glove_l", "glove_r", "torso")
TARGET_RADIUS = {"head": HEAD_RADIUS, "glove_l": GLOVE_RADIUS, "glove_r": GLOVE_RADIUS, "torso": 0.15}
# capsules (from body, to body) along the arms, for occlusion
ARM_SEGMENTS = [
  ("Mirrorlink2__1__1", "Mirrorlink3__1__1"),
  ("Mirrorlink3__1__1", "Mirrorlink4__1__1"),
  ("Mirrorlink4__1__1", "Mirrorlink5__1__1"),
  ("Mirrorlink5__1__1", "Mirrorlink6__1__1"),
  ("link2__1__1", "link3__1__1"),
  ("link3__1__1", "link4__1__1"),
  ("link4__1__1", "link5__1__1"),
  ("link5__1__1", "link6__1__1"),
]
ARM_RADIUS = 0.045


@dataclass
class CameraCfg:
  width: int = 640
  height: int = 400  # 16:10
  hfov_deg: float = 86.0  # vertical follows from the 16:10 image (60.5 deg)
  period_steps: int = 2  # 25 Hz at 50 Hz env steps
  latency_steps: int = 2  # 40 ms capture + detector
  pixel_noise: float = 2.0  # px, detection centre
  radius_noise: float = 1.5  # px, apparent radius
  dropout: float = 0.05  # missed detection per target per frame
  use_depth: bool = False  # the D455's depth stream
  depth_range: tuple[float, float] = (0.4, 6.0)
  depth_noise: tuple[float, float] = (0.005, 0.005)  # sigma = a + b * z^2  [m]


def vfov_deg(cfg: CameraCfg) -> float:
  return math.degrees(2 * math.atan(math.tan(math.radians(cfg.hfov_deg) / 2) * cfg.height / cfg.width))


@dataclass
class UltrasonicCfg:
  enabled: bool = True
  cone_half_angle_deg: float = 15.0
  range: tuple[float, float] = (0.02, 4.0)
  period_steps: int = 2
  latency_steps: int = 1
  noise: tuple[float, float] = (0.01, 0.01)  # sigma = a + b * range  [m]
  dropout: float = 0.05
  gate_sigma: float = 3.0  # accept for a track only within this many sigmas


@dataclass
class EkfCfg:
  accel_std: dict[str, float] = field(default_factory=lambda: {"head": 3.0, "glove_l": 15.0, "glove_r": 15.0, "torso": 2.0})
  init_range: float = 1.2  # m, prior range when a target is first seen
  init_pos_std: float = 0.5
  init_vel_std: float = 1.0
  max_pos_std: float = 2.0  # reported for targets with no track
  coast_s: float = 0.2  # unseen this long: velocity decays (stop extrapolating)
  vel_decay: float = 0.85  # per step while coasting
  lost_s: float = 1.0  # unseen this long: drop the track, restart on the next sighting


@dataclass
class PerceptionCfg:
  camera: CameraCfg = field(default_factory=CameraCfg)
  ultrasonic: UltrasonicCfg = field(default_factory=UltrasonicCfg)
  ekf: EkfCfg = field(default_factory=EkfCfg)


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def _point_seg_dist(p, a, b) -> torch.Tensor:
  ab = b - a
  t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)
  return (a + ab * t.clamp(0, 1)[..., None] - p).norm(dim=-1)


def _seg_seg_dist(p1, q1, p2, q2) -> torch.Tensor:
  d1, d2, r = q1 - p1, q2 - p2, p1 - p2
  a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
  b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
  s = ((b * f - c * e) / (a * e - b * b).clamp_min(1e-9)).clamp(0, 1)
  t = ((b * s + f) / e.clamp_min(1e-9)).clamp(0, 1)
  s = ((b * t - c) / a.clamp_min(1e-9)).clamp(0, 1)
  return ((p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None])).norm(dim=-1)


# --------------------------------------------------------------------------- #
# world state the sensors look at
# --------------------------------------------------------------------------- #


class _Scene:
  """Ids and per-step poses of what the sensors need."""

  def __init__(self, env: ManagerBasedRlEnv, observer: str):
    self.env, self.observer = env, observer
    self.opp = "blue" if observer == "red" else "red"
    m = env.sim.mj_model
    self.cam_site = m.site(f"{observer}/{CAMERA_SITE}").id
    self.us_site = m.site(f"{observer}/{ULTRASONIC_SITE}").id
    self.target_geoms = [m.geom(f"{self.opp}/{t}").id for t in TARGETS]
    self.glove_geoms = {team: [m.geom(f"{team}/glove_{s}").id for s in "lr"] for team in (observer, self.opp)}
    self.arm_bodies = {
      team: [(env.scene[team].find_bodies(a)[0][0], env.scene[team].find_bodies(b)[0][0]) for a, b in ARM_SEGMENTS]
      for team in (observer, self.opp)
    }
    ent = env.scene[self.opp]
    names = [m.body(m.geom_bodyid[g]).name.split("/", 1)[1] for g in self.target_geoms]
    self.target_bodies = [ent.find_bodies(nm)[0][0] for nm in names]
    # a glove is not hidden by its own forearm (capsule index in occluders())
    n_arm = len(ARM_SEGMENTS)
    self.own_forearm = {TARGETS.index("glove_l"): n_arm + 3, TARGETS.index("glove_r"): n_arm + 7}

  def site_pose(self, sid):
    d = self.env.sim.data
    rot = d.site_xmat[:, sid].reshape(-1, 3, 3)
    return d.site_xpos[:, sid], rot  # frame: -z forward, +y up, +x right

  def targets(self):
    """[N, T, 3] true positions and [N, T, 3] velocities of the opponent's targets."""
    pos = self.env.sim.data.geom_xpos[:, self.target_geoms]
    vel = self.env.scene[self.opp].data.body_link_lin_vel_w[:, self.target_bodies]
    return pos, vel

  def occluders(self):
    """Spheres [N, S, 4] (centre, radius) and capsules [N, C, 7] (a, b, radius)."""
    d = self.env.sim.data
    spheres = []
    for team, gids in self.glove_geoms.items():
      for g in gids:
        spheres.append(torch.cat([d.geom_xpos[:, g], torch.full_like(d.geom_xpos[:, g, :1], GLOVE_RADIUS)], -1))
    caps = []
    for team, pairs in self.arm_bodies.items():
      pos = self.env.scene[team].data.body_link_pos_w
      for a, b in pairs:
        caps.append(torch.cat([pos[:, a], pos[:, b], torch.full_like(pos[:, a, :1], ARM_RADIUS)], -1))
    return torch.stack(spheres, 1), torch.stack(caps, 1), [g for gids in self.glove_geoms.values() for g in gids]


# --------------------------------------------------------------------------- #
# sensors
# --------------------------------------------------------------------------- #


def camera_project(p_world, cam_pos, cam_rot, f):
  """Pinhole projection. Returns camera-frame point (x right, y down, z forward)
  and pixel offsets from the image centre (u, v)."""
  rel = p_world - cam_pos[:, None]
  right, up, fwd = cam_rot[..., 0], cam_rot[..., 1], -cam_rot[..., 2]
  pc = torch.stack([(rel * right[:, None]).sum(-1), -(rel * up[:, None]).sum(-1), (rel * fwd[:, None]).sum(-1)], -1)
  z = pc[..., 2].clamp_min(1e-3)
  uv = f * pc[..., :2] / z[..., None]
  return pc, uv


def simulate_camera(scene: _Scene, cfg: CameraCfg, gen: torch.Generator | None):
  """One camera frame. Returns measurement [N, T, 4] = (u, v, r_px, depth) and
  validity masks [N, T] for (u, v, r_px) and for depth."""
  cam_pos, cam_rot = scene.site_pose(scene.cam_site)
  targets, _ = scene.targets()
  n, t = targets.shape[:2]
  dev = targets.device
  f = (cfg.width / 2) / math.tan(math.radians(cfg.hfov_deg) / 2)
  pc, uv = camera_project(targets, cam_pos, cam_rot, f)
  z, dist = pc[..., 2], pc.norm(dim=-1)
  in_fov = (
    (z > 0.05)
    & (pc[..., 0].abs() < z * math.tan(math.radians(cfg.hfov_deg) / 2))
    & (pc[..., 1].abs() < z * math.tan(math.radians(vfov_deg(cfg)) / 2))
  )

  # occlusion: line of sight passes through a glove or an arm (either fighter)
  spheres, caps, sphere_geoms = scene.occluders()
  radii = torch.tensor([TARGET_RADIUS[k] for k in TARGETS], device=dev)
  # stop the sight line at the target's front surface
  dirn = (targets - cam_pos[:, None]) / dist[..., None].clamp_min(1e-6)
  end = targets - dirn * radii[None, :, None]
  start = cam_pos[:, None].expand_as(end)
  hidden = torch.zeros(n, t, dtype=torch.bool, device=dev)
  for k in range(spheres.shape[1]):
    c, r = spheres[:, k, None, :3], spheres[:, k, None, 3]
    blocks = _point_seg_dist(c.expand_as(end), start, end) < r
    for ti, g in enumerate(scene.target_geoms):  # a glove does not hide itself
      if g == sphere_geoms[k]:
        blocks[:, ti] = False
    hidden |= blocks
  for k in range(caps.shape[1]):
    a, b, r = caps[:, k, None, :3], caps[:, k, None, 3:6], caps[:, k, None, 6]
    blocks = _seg_seg_dist(start, end, a.expand_as(end), b.expand_as(end)) < r
    for ti, ck in scene.own_forearm.items():
      if ck == k:
        blocks[:, ti] = False
    hidden |= blocks

  drop = torch.rand(n, t, device=dev, generator=gen) < cfg.dropout
  valid = in_fov & ~hidden & ~drop
  r_px = f * radii[None] / dist.clamp_min(1e-3)
  noise = torch.randn(n, t, 4, device=dev, generator=gen)
  meas = torch.stack(
    [
      uv[..., 0] + cfg.pixel_noise * noise[..., 0],
      uv[..., 1] + cfg.pixel_noise * noise[..., 1],
      r_px + cfg.radius_noise * noise[..., 2],
      z + (cfg.depth_noise[0] + cfg.depth_noise[1] * z**2) * noise[..., 3],
    ],
    -1,
  )
  depth_ok = valid & (z > cfg.depth_range[0]) & (z < cfg.depth_range[1]) & cfg.use_depth
  return meas, valid, depth_ok, dict(in_fov=in_fov, hidden=hidden)


def simulate_ultrasonic(scene: _Scene, cfg: UltrasonicCfg, gen: torch.Generator | None):
  """Distance from the sensor to the nearest thing inside its cone: the
  opponent's target spheres and arms, and the robot's own gloves and arms (a
  chest sensor looks through its own guard). Returns noisy range [N], valid
  [N], true range [N] and whether the nearest thing was the robot's own arm [N]."""
  pos, rot = scene.site_pose(scene.us_site)
  fwd = -rot[..., 2]
  targets, _ = scene.targets()
  dev = targets.device
  radii = torch.tensor([TARGET_RADIUS[k] for k in TARGETS], device=dev)
  pts, rads, own = [targets], [radii.expand(targets.shape[:2])], [torch.zeros(targets.shape[:2], dtype=torch.bool, device=dev)]
  d = scene.env.sim.data
  for team in (scene.opp, scene.observer):
    is_own = team == scene.observer
    arm = scene.env.scene[team].data.body_link_pos_w
    for a, b in scene.arm_bodies[team]:
      for s in (0.0, 0.5, 1.0):
        pts.append((arm[:, a] + s * (arm[:, b] - arm[:, a]))[:, None])
        rads.append(torch.full_like(pts[-1][..., 0], ARM_RADIUS))
        own.append(torch.full_like(rads[-1], is_own, dtype=torch.bool))
    if is_own:
      for g in scene.glove_geoms[team]:
        pts.append(d.geom_xpos[:, g][:, None])
        rads.append(torch.full_like(pts[-1][..., 0], GLOVE_RADIUS))
        own.append(torch.ones_like(rads[-1], dtype=torch.bool))
  pts, rads, own = torch.cat(pts, 1), torch.cat(rads, 1), torch.cat(own, 1)
  rel = pts - pos[:, None]
  dist = rel.norm(dim=-1)
  # an object is in the cone if any of it is within the half-angle
  cos_axis = (rel * fwd[:, None]).sum(-1) / dist.clamp_min(1e-6)
  angle = torch.acos(cos_axis.clamp(-1, 1)) - torch.asin((rads / dist.clamp_min(1e-6)).clamp(max=1.0))
  in_cone = (angle < math.radians(cfg.cone_half_angle_deg)) & (cos_axis > 0)
  surface = torch.where(in_cone, (dist - rads).clamp_min(0), torch.full_like(dist, float("inf")))
  rng, idx = surface.min(1)
  hit_own = own.gather(1, idx[:, None])[:, 0] & torch.isfinite(rng)
  n = rng.shape[0]
  valid = (rng > cfg.range[0]) & (rng < cfg.range[1]) & (torch.rand(n, device=dev, generator=gen) >= cfg.dropout)
  sigma = cfg.noise[0] + cfg.noise[1] * rng.clamp(max=cfg.range[1])
  noisy = rng + sigma * torch.randn(n, device=dev, generator=gen)
  return torch.where(valid, noisy, torch.zeros_like(noisy)), valid, rng, hit_own


# --------------------------------------------------------------------------- #
# EKF
# --------------------------------------------------------------------------- #


def _ekf_update(x, P, z, h, H, R, mask):
  """x [B,6], P [B,6,6], z,h [B,m], H [B,m,6], R [B,m,m]; update where mask [B]."""
  S = H @ P @ H.transpose(-1, -2) + R
  K = P @ H.transpose(-1, -2) @ torch.linalg.inv(S)
  x_new = x + (K @ (z - h)[..., None])[..., 0]
  I = torch.eye(6, device=x.device).expand_as(P)
  KH = K @ H
  P_new = (I - KH) @ P @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)  # Joseph form
  m = mask[:, None]
  return torch.where(m, x_new, x), torch.where(m[..., None], P_new, P)


class OpponentTracker:
  """Per-env EKFs for the opponent's targets, as seen by `observer`."""

  def __init__(self, env: ManagerBasedRlEnv, observer: str, cfg: PerceptionCfg, seed: int = 0):
    self.env, self.cfg, self.scene = env, cfg, _Scene(env, observer)
    n, t, dev = env.num_envs, len(TARGETS), env.device
    self.dt = env.step_dt
    self.x = torch.zeros(n, t, 6, device=dev)
    self.P = torch.zeros(n, t, 6, 6, device=dev)
    self.init = torch.zeros(n, t, dtype=torch.bool, device=dev)
    self.since_seen = torch.full((n, t), 99.0, device=dev)
    self.gen = torch.Generator(device=dev)
    self.gen.manual_seed(seed)
    self.pending: list = []  # (ready_step, kind, data...) in arrival order
    self.step = 0
    self.f = (cfg.camera.width / 2) / math.tan(math.radians(cfg.camera.hfov_deg) / 2)
    q = torch.tensor([cfg.ekf.accel_std[k] ** 2 for k in TARGETS], device=dev)
    dt = self.dt
    # white-noise acceleration model
    Q1 = torch.tensor([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]], device=dev)
    self.Q = torch.zeros(t, 6, 6, device=dev)
    for a in range(3):
      idx = [a, a + 3]
      self.Q[:, idx[0], idx[0]] = q * Q1[0, 0]
      self.Q[:, idx[0], idx[1]] = q * Q1[0, 1]
      self.Q[:, idx[1], idx[0]] = q * Q1[1, 0]
      self.Q[:, idx[1], idx[1]] = q * Q1[1, 1]
    self.F = torch.eye(6, device=dev)
    self.F[:3, 3:] = torch.eye(3, device=dev) * dt
    self.last_us = torch.zeros(n, device=dev)
    self.last_us_valid = torch.zeros(n, dtype=torch.bool, device=dev)
    self.last_cam_info: dict = {}

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.x[ids] = 0
    self.P[ids] = 0
    self.init[ids] = False
    self.since_seen[ids] = 99.0
    # readings taken before the reset must not reach the new round
    for item in self.pending:
      for v in item[2:]:
        if isinstance(v, torch.Tensor) and v.dtype == torch.bool:
          v[ids] = False

  # -- one 50 Hz step ---------------------------------------------------------
  def update(self):
    cfg, sc = self.cfg, self.scene
    n, t = self.x.shape[:2]
    # predict
    self.x = self.x @ self.F.T
    self.P = self.F @ self.P @ self.F.T + self.Q
    self.since_seen += self.dt
    # track management: a hidden target coasts briefly, then stops moving in the
    # estimate; after lost_s the track is dropped and restarts on the next sighting
    coasting = (self.since_seen > cfg.ekf.coast_s)[..., None]
    self.x[..., 3:] = torch.where(coasting, self.x[..., 3:] * cfg.ekf.vel_decay, self.x[..., 3:])
    self.init &= self.since_seen <= cfg.ekf.lost_s

    # sensors fire at their rate; their readings arrive `latency` steps later
    if self.step % cfg.camera.period_steps == 0:
      meas, valid, depth_ok, info = simulate_camera(sc, cfg.camera, self.gen)
      self.last_cam_info = info
      pose = [v.clone() for v in sc.site_pose(sc.cam_site)]
      self.pending.append((self.step + cfg.camera.latency_steps, "cam", meas, valid, depth_ok, *pose))
    if cfg.ultrasonic.enabled and self.step % cfg.ultrasonic.period_steps == 1 % cfg.ultrasonic.period_steps:
      rng, ok, _, _ = simulate_ultrasonic(sc, cfg.ultrasonic, self.gen)
      pose = [v.clone() for v in sc.site_pose(sc.us_site)]
      self.pending.append((self.step + cfg.ultrasonic.latency_steps, "us", rng, ok, *pose))

    # readings whose latency has passed update the filter, oldest first
    ready = [p for p in self.pending if p[0] <= self.step]
    self.pending = [p for p in self.pending if p[0] > self.step]
    for item in ready:
      (self._camera_update if item[1] == "cam" else self._ultrasonic_update)(*item[2:])
    self.step += 1

  def _camera_update(self, meas, valid, depth_ok, cam_pos, cam_rot):
    cfg = self.cfg
    n, t = valid.shape
    f = self.f
    radii = torch.tensor([TARGET_RADIUS[k] for k in TARGETS], device=meas.device)
    right, up, fwd = cam_rot[..., 0], cam_rot[..., 1], -cam_rot[..., 2]
    A = torch.stack([right, -up, fwd], 1)  # world -> camera frame rows [N,3,3]

    # first sighting: place the track on the ray, at the size-based range
    new = valid & ~self.init
    if new.any():
      u, v, r = meas[..., 0], meas[..., 1], meas[..., 2].clamp_min(1.0)
      rng = (f * radii[None] / r).clamp(0.2, 4.0)
      ray_c = torch.stack([u / f, v / f, torch.ones_like(u)], -1)
      ray_c = ray_c / ray_c.norm(dim=-1, keepdim=True)
      ray_w = torch.einsum("nji,ntj->nti", A, ray_c)
      p0 = cam_pos[:, None] + ray_w * rng[..., None]
      P0 = torch.zeros(n, t, 6, 6, device=meas.device)
      P0[..., :3, :3] = torch.eye(3, device=meas.device) * cfg.ekf.init_pos_std**2
      P0[..., 3:, 3:] = torch.eye(3, device=meas.device) * cfg.ekf.init_vel_std**2
      self.x = torch.where(new[..., None], torch.cat([p0, torch.zeros_like(p0)], -1), self.x)
      self.P = torch.where(new[..., None, None], P0, self.P)
      self.init |= new

    upd = valid & self.init
    pc = torch.einsum("nij,ntj->nti", A, self.x[..., :3] - cam_pos[:, None])
    z = pc[..., 2].clamp_min(0.05)
    dist = pc.norm(dim=-1).clamp_min(0.05)
    # h = (u, v, r_px [, depth]) and its Jacobian w.r.t. the world position
    h = [f * pc[..., 0] / z, f * pc[..., 1] / z, f * radii[None] / dist]
    J_c = torch.zeros(n, t, 3, 3, device=meas.device)
    J_c[..., 0, 0] = f / z
    J_c[..., 0, 2] = -f * pc[..., 0] / z**2
    J_c[..., 1, 1] = f / z
    J_c[..., 1, 2] = -f * pc[..., 1] / z**2
    J_c[..., 2, :] = -f * radii[None, :, None] * pc / dist[..., None] ** 3
    rows = [cfg.camera.pixel_noise**2, cfg.camera.pixel_noise**2, cfg.camera.radius_noise**2]
    use_depth = cfg.camera.use_depth
    if use_depth:
      h.append(pc[..., 2])
      J_c = torch.cat([J_c, torch.zeros(n, t, 1, 3, device=meas.device)], -2)
      J_c[..., 3, 2] = 1.0
    H_pos = J_c @ A[:, None]  # [N,T,m,3]
    m = H_pos.shape[-2]
    H = torch.cat([H_pos, torch.zeros_like(H_pos)], -1)
    hv = torch.stack(h, -1)
    zv = meas[..., :m]
    R = torch.zeros(n, t, m, m, device=meas.device)
    for i, r in enumerate(rows):
      R[..., i, i] = r
    if use_depth:
      zd = meas[..., 3]
      R[..., 3, 3] = torch.where(depth_ok, (cfg.camera.depth_noise[0] + cfg.camera.depth_noise[1] * zd**2) ** 2, torch.full_like(zd, 1e6))
    x, P = _ekf_update(
      self.x.reshape(-1, 6), self.P.reshape(-1, 6, 6), zv.reshape(-1, m), hv.reshape(-1, m),
      H.reshape(-1, m, 6), R.reshape(-1, m, m), upd.reshape(-1),
    )
    self.x, self.P = x.reshape(n, t, 6), P.reshape(n, t, 6, 6)
    self.since_seen = torch.where(upd, torch.zeros_like(self.since_seen), self.since_seen)

  def _ultrasonic_update(self, rng, ok, pos, rot):
    cfg = self.cfg.ultrasonic
    radii = torch.tensor([TARGET_RADIUS[k] for k in TARGETS], device=rng.device)
    fwd = -rot[..., 2]
    rel = self.x[..., :3] - pos[:, None]
    dist = rel.norm(dim=-1).clamp_min(1e-3)
    pred = dist - radii[None]
    cos_axis = (rel * fwd[:, None]).sum(-1) / dist
    in_cone = torch.acos(cos_axis.clamp(-1, 1)) < math.radians(cfg.cone_half_angle_deg) + 0.1
    H_pos = rel / dist[..., None]  # d(range)/d(position)
    sigma = cfg.noise[0] + cfg.noise[1] * rng
    S = (H_pos[..., None, :] @ self.P[..., :3, :3] @ H_pos[..., :, None])[..., 0, 0] + sigma[:, None] ** 2
    nis = (rng[:, None] - pred) ** 2 / S
    # nearest-neighbour association: the in-cone, gated track that explains it best
    score = torch.where(self.init & in_cone & (nis < cfg.gate_sigma**2), nis, torch.full_like(nis, float("inf")))
    best = score.argmin(1)
    has = ok & torch.isfinite(score.min(1).values)
    mask = torch.zeros_like(self.init)
    mask[torch.arange(len(best)), best] = True
    mask &= has[:, None]
    n, t = mask.shape
    H = torch.cat([H_pos, torch.zeros_like(H_pos)], -1)[..., None, :]
    x, P = _ekf_update(
      self.x.reshape(-1, 6), self.P.reshape(-1, 6, 6), rng[:, None].expand(n, t).reshape(-1, 1),
      pred.reshape(-1, 1), H.reshape(-1, 1, 6), (sigma[:, None] ** 2).expand(n, t).reshape(-1, 1, 1), mask.reshape(-1),
    )
    self.x, self.P = x.reshape(n, t, 6), P.reshape(n, t, 6, 6)
    self.last_us, self.last_us_valid = rng, has

  # -- outputs ----------------------------------------------------------------
  def heading_frame(self):
    """Observer pelvis position and heading rotation (world -> heading frame)."""
    ent = self.env.scene[self.scene.observer]
    q = ent.data.root_link_quat_w
    w, x, y, z = q.unbind(-1)
    # robot forward is base +y; heading = yaw of that axis
    fx, fy = 2 * (x * y - w * z), 1 - 2 * (x * x + z * z)
    yaw = torch.atan2(fy, fx)
    c, s = torch.cos(yaw), torch.sin(yaw)
    R = torch.stack([torch.stack([c, s, torch.zeros_like(c)], -1), torch.stack([-s, c, torch.zeros_like(c)], -1),
                     torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], -1)], 1)
    return ent.data.root_link_pos_w, R

  def observation(self) -> torch.Tensor:
    """[N, T*8]: per target position (3), velocity (3), position std (1), time unseen (1)."""
    origin, R = self.heading_frame()
    pos = torch.einsum("nij,ntj->nti", R, self.x[..., :3] - origin[:, None])
    vel = torch.einsum("nij,ntj->nti", R, self.x[..., 3:])
    std = torch.diagonal(self.P[..., :3, :3], dim1=-2, dim2=-1).sum(-1).clamp_min(0).sqrt()
    unseen = torch.where(self.init, self.since_seen, torch.full_like(self.since_seen, 99.0)).clamp(max=5.0)
    pos = torch.where(self.init[..., None], pos, torch.zeros_like(pos))
    vel = torch.where(self.init[..., None], vel, torch.zeros_like(vel))
    std = torch.where(self.init, std, torch.full_like(std, self.cfg.ekf.max_pos_std))
    return torch.cat([pos, vel, std[..., None], unseen[..., None]], -1).reshape(pos.shape[0], -1)

  def truth(self) -> torch.Tensor:
    """[N, T*6]: exact position and velocity in the same frame (privileged)."""
    origin, R = self.heading_frame()
    p, v = self.scene.targets()
    pos = torch.einsum("nij,ntj->nti", R, p - origin[:, None])
    vel = torch.einsum("nij,ntj->nti", R, v)
    return torch.cat([pos, vel], -1).reshape(pos.shape[0], -1)


# --------------------------------------------------------------------------- #
# mjlab observation / event terms
# --------------------------------------------------------------------------- #


def tracker(env: ManagerBasedRlEnv, team: str, cfg: PerceptionCfg | None = None) -> OpponentTracker:
  trackers = getattr(env, "_opponent_trackers", None)
  if trackers is None:
    trackers = env._opponent_trackers = {}
  if team not in trackers:
    trackers[team] = OpponentTracker(env, team, cfg or PerceptionCfg())
    trackers[team]._last_step = -1
  return trackers[team]


def _stepped(env, team, cfg):
  tr = tracker(env, team, cfg)
  if tr._last_step != env.common_step_counter:
    tr.update()
    tr._last_step = env.common_step_counter
  return tr


def opponent_perceived(env: ManagerBasedRlEnv, team: str, cfg: PerceptionCfg | None = None) -> torch.Tensor:
  """Actor observation: EKF estimate of the opponent from camera + ultrasonic."""
  return _stepped(env, team, cfg).observation()


def opponent_true(env: ManagerBasedRlEnv, team: str, cfg: PerceptionCfg | None = None) -> torch.Tensor:
  """Critic / teacher observation: the exact opponent state."""
  return tracker(env, team, cfg).truth()


def reset_trackers(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  for tr in getattr(env, "_opponent_trackers", {}).values():
    tr.reset(env_ids)
