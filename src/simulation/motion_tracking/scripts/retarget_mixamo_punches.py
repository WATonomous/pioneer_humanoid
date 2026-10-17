"""Retarget punches to Wato from the Mixamo fight takes (FBX) and the ACCAD
martial-arts punch clips (AMASS SMPL-X), one clip per punch, each starting and
ending in the generate_moves.py guard so they chain with the rest of the move
library (scripts/build_move_library.py runs this).

From the source punch (in its stance frame: x towards the opponent, y left,
z up; southpaw takes mirrored to orthodox, the library's stance):
  * glove and elbow paths, scaled by Wato's / the source's arm length and
    added to Wato's guard glove and elbow, either as they move in the stance
    frame ("world": the boxer's torso turns about as far as Wato's hips can)
    or relative to the turning chest, carried with Wato's own (clipped) hip
    turn and up to MAX_SWING more made up by swinging the arm about the
    shoulder ("chest": a martial artist's hook turns the torso ~110 deg);
    the glove is kept within reach of the shoulder; damped least-squares IK
    on the punching arm (ArmIK)
  * hip turn, hip drop and weight shift: from the source pelvis, scaled to
    Wato's legs and clipped, driven through the whole-body leg IK of
    generate_moves.py with both feet planted
Each channel is detrended so it returns exactly to the guard at the end. Where
the punch takes any arm or leg joint past --max-cap-use of its motor speed
cap, that stretch of it is slowed down and solved again (the real Wato arm is
slower than a boxer); the rest keeps the boxer's timing.

Needs: the Mixamo takes in --fbx-dir (read through `assimp`), the ACCAD clips
in --amass-dir and the SMPL-X body models in --smplx-dir.

  uv run scripts/retarget_mixamo_punches.py --output-dir data/moves/csv
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_moves as gm  # noqa: E402
from humanoid_tracking.robot import CSV_JOINT_NAMES, SPEED_CAPS  # noqa: E402

SRC_FPS = 30  # source motion is resampled to this rate

# move -> (take, start s, end s, source hand, mirror, mapping). A Mixamo take is
# <take>_mixamo.fbx in --fbx-dir; "ACCAD/<clip>" is an AMASS SMPL-X clip from
# Male2MartialArtsPunches. Picked for a punch that starts from the guard;
# southpaw ones are mirrored (their lead hand becomes Wato's left). The ACCAD
# performer is southpaw in the hook and uppercut clips, so its "hook left" is
# a rear hook, and so on. Windows: the punching hand's speed (smoothed over
# 5 frames) above 0.3 m/s around its peak.
# mapping: "world" copies the glove's path in the stance frame (the boxer's
# torso turns about as far as Wato's hips can); "chest" copies it relative to
# the turning chest (chest_targets), for the ACCAD punches, whose torso turns
# up to ~110 deg; "hook" keeps the source's legs and timing but gives the arm
# a hook's shape (hook_paths): the boxer's own rear hook folds the elbow to
# 30-45 deg and travels too little. Not used: ACCAD E5 (a second rear hook)
# and E7 (rear uppercut), which looked wrong on Wato.
SOURCES = {
  "jab": ("Boxing", 9.30, 10.10, "Left", False, "world"),
  "cross": ("Boxing", 5.47, 6.07, "Left", True, "world"),
  "rear_hook": ("Boxing", 9.67, 10.20, "Right", False, "hook"),
  "lead_hook": ("ACCAD/E6_-__hook_right", 0.67, 1.87, "Right", True, "chest"),
  "lead_uppercut": ("ACCAD/E8_-_uppercut_right", 0.30, 1.73, "Right", True, "chest"),
}
# move -> (cut s, recovery s): after `cut` (source time from the window start)
# the source's own recovery is replaced by a smooth return to the guard over
# `recovery` (the ACCAD lead hook unwinds its torso slowly for ~0.65 s)
RECOVER = {"lead_hook": (0.55, 0.40)}
# SMPL-X body joint index for each Mixamo joint name used here
SMPLX_JOINTS = {
  "Hips": 0, "LeftUpLeg": 1, "RightUpLeg": 2, "LeftFoot": 7, "RightFoot": 8, "Spine2": 9,
  "LeftToeBase": 10, "RightToeBase": 11, "Head": 15, "LeftArm": 16, "RightArm": 17,
  "LeftForeArm": 18, "RightForeArm": 19, "LeftHand": 20, "RightHand": 21,
}  # fmt: skip
MAX_HEADING = 0.6  # rad, hip turn
MAX_SWING = 0.6  # rad, body turn beyond MAX_HEADING made up by the arm
DROP_RANGE = (-0.02, 0.08)  # m, hip drop
MAX_WEIGHT = 0.35  # fraction of the way from the neutral CoM to a sole
# "hook" mapping: elbow angle, glove arc about the shoulder (deg from straight
# ahead, outside -> across) and glove height above the shoulder [m]
HOOK_ELBOW = 120.0
HOOK_ARC = (60.0, 20.0)
HOOK_HEIGHT = 0.02
SETTLE = 8  # frames at the end of a punch blended into the guard arm angles
MAX_SLOW = 4.0  # give up slowing a punch down past this factor
KEY_DT = 0.1  # s of source time between body waypoints
REACH = 0.92  # furthest glove from the shoulder, as a fraction of the arm length


# ---------------------------------------------------------------- source motion


HEIGHT = 1.75  # source skeleton is scaled to this standing height [m]
BONES = [
  ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"), ("Neck", "Head"),
  ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
  ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
  ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
  ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
]  # fmt: skip


def fbx_to_glb(fbx: Path, cache: Path) -> Path:
  glb = cache / f"{fbx.stem}.glb"
  if not glb.exists() or glb.stat().st_mtime < fbx.stat().st_mtime:
    cache.mkdir(parents=True, exist_ok=True)
    assimp = shutil.which("assimp")
    if assimp is None:
      raise SystemExit("assimp not found on PATH (needed to read FBX)")
    subprocess.run([assimp, "export", str(fbx), str(glb)], check=True, capture_output=True)
  return glb


def load_take(glb: Path) -> dict[str, np.ndarray]:
  """Joint world positions (T, 3), z-up, metres, feet on z=0, sampled at SRC_FPS."""
  b = glb.read_bytes()
  jlen = struct.unpack_from("<I", b, 12)[0]
  g = json.loads(b[20 : 20 + jlen])
  blob = b[20 + jlen + 8 :]

  def acc(i):
    a = g["accessors"][i]
    bv = g["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(blob, np.float32, a["count"] * n, off).reshape(-1, n).astype(np.float64)

  nodes = g["nodes"]
  parent = [-1] * len(nodes)
  for i, n in enumerate(nodes):
    for c in n.get("children", []):
      parent[c] = i
  anim = g["animations"][0]
  tracks = {}
  for ch in anim["channels"]:
    s = anim["samplers"][ch["sampler"]]
    tracks[(ch["target"]["node"], ch["target"]["path"])] = (acc(s["input"])[:, 0], acc(s["output"]))
  ts = np.arange(0, max(t[-1] for t, _ in tracks.values()) + 1e-6, 1 / SRC_FPS)
  T = len(ts)

  def local(i):
    n = nodes[i]
    if "matrix" in n:
      return np.broadcast_to(np.array(n["matrix"]).reshape(4, 4).T, (T, 4, 4))
    tr = np.tile(n.get("translation", [0.0, 0, 0]), (T, 1))
    q = np.tile(n.get("rotation", [0.0, 0, 0, 1]), (T, 1))
    sc = np.tile(n.get("scale", [1.0, 1, 1]), (T, 1))
    for path in ("translation", "scale"):
      if (i, path) in tracks:
        t, v = tracks[(i, path)]
        x = np.stack([np.interp(ts, t, v[:, k]) for k in range(3)], 1)
        tr, sc = (x, sc) if path == "translation" else (tr, x)
    if (i, "rotation") in tracks:
      t, v = tracks[(i, "rotation")]
      q = np.tile(v[0], (T, 1)) if len(t) == 1 else Slerp(t, Rotation.from_quat(v))(np.clip(ts, t[0], t[-1])).as_quat()
    M = np.zeros((T, 4, 4))
    M[:, :3, :3] = Rotation.from_quat(q).as_matrix() * sc[:, None, :]
    M[:, :3, 3] = tr
    M[:, 3, 3] = 1
    return M

  G: dict[int, np.ndarray] = {}

  def world(i):
    if i not in G:
      G[i] = local(i) if parent[i] < 0 else world(parent[i]) @ local(i)
    return G[i]

  P = {}
  for i, n in enumerate(nodes):
    name = n.get("name", "").split(":")[-1]
    if "$" not in name and name not in P and name in {b for e in BONES for b in e}:
      P[name] = world(i)[:, :3, 3][:, [0, 2, 1]] * np.array([1, -1, 1])  # y-up -> z-up
  floor = min(P["LeftToeBase"][:, 2].min(), P["RightToeBase"][:, 2].min())
  scale = HEIGHT / (P["Head"][0, 2] - floor)
  P = {k: (v - [0, 0, floor]) * scale for k, v in P.items()}
  P["t"] = ts
  return P


def unit(v):
  return v / np.linalg.norm(v, axis=-1, keepdims=True)


def target_dir(P) -> tuple[np.ndarray, np.ndarray]:
  """Per-frame horizontal direction to the (imaginary) opponent, and the lead
  foot (+1 left foot forward / orthodox, -1 right / southpaw, 0 square).

  The chest turns ~35 deg away from the opponent in a bladed stance and the
  rear->lead foot line turns about as far the other way; their mean is steady
  across stance switches."""
  side = P["LeftArm"] - P["RightArm"] + P["LeftUpLeg"] - P["RightUpLeg"]
  chest = unit(np.stack([-side[:, 1], side[:, 0], 0 * side[:, 0]], 1))
  toe = P["LeftToeBase"] + P["RightToeBase"] - P["LeftFoot"] - P["RightFoot"]
  if np.mean(np.sum(toe * chest, 1)) < 0:
    chest = -chest
  feet = (P["LeftFoot"] - P["RightFoot"]) * [1, 1, 0]
  ahead = np.sum(feet * chest, 1)
  lead = uniform_filter1d(np.where(ahead > 0.08, 1.0, np.where(ahead < -0.08, -1.0, 0.0)), 9)
  lead = np.where(lead > 0.5, 1, np.where(lead < -0.5, -1, 0))
  feet_dir = unit(feet * np.where(lead == 0, np.sign(ahead), lead)[:, None] + 1e-9)
  fwd = uniform_filter1d(unit(chest + feet_dir), 2 * SRC_FPS + 1, axis=0, mode="nearest")
  return unit(fwd * [1, 1, 0]), lead


class SmplX:
  def __init__(self, model_dir: str, gender: str):
    # official SMPL-X release; a few metadata fields are pickled objects
    m = np.load(Path(model_dir) / f"SMPLX_{gender.upper()}.npz", allow_pickle=True)
    self.v_template = m["v_template"]
    self.shapedirs = m["shapedirs"]
    self.J_regressor = m["J_regressor"]
    self.weights = m["weights"]
    self.faces = m["f"].astype(np.int32)
    self.parents = m["kintree_table"][0].astype(np.int64)
    self.parents[0] = -1

  def pose(self, betas: np.ndarray, poses: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 165) axis-angle poses -> vertices (T, V, 3), joints (T, 55, 3)."""
    v = self.v_template + self.shapedirs[:, :, : len(betas)] @ betas
    J = self.J_regressor @ v
    T, nj = len(poses), len(self.parents)
    R = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(T, nj, 3, 3)
    G = np.zeros((T, nj, 4, 4))
    G[:, :, 3, 3] = 1
    for j in range(nj):
      p = self.parents[j]
      local = np.zeros((T, 4, 4))
      local[:, :3, :3] = R[:, j]
      local[:, :3, 3] = J[j] - (J[p] if p >= 0 else 0)
      local[:, 3, 3] = 1
      G[:, j] = local if p < 0 else G[:, p] @ local
    joints = G[:, :, :3, 3] + trans[:, None]
    # skinning: move each joint's frame from its rest position
    A = G.copy()
    A[:, :, :3, 3] -= np.einsum("tjab,jb->tja", G[:, :, :3, :3], J)
    W = np.einsum("vj,tjab->tvab", self.weights, A[:, :, :3])  # (T, V, 3, 4)
    verts = np.einsum("tvab,vb->tva", W[..., :3], v) + W[..., 3] + trans[:, None]
    return verts.astype(np.float32), joints.astype(np.float32)


def load_clip(path: Path):
  d = np.load(path, allow_pickle=False)
  return {
    "gender": str(d["gender"]),
    "fps": float(d["mocap_frame_rate"]),
    "poses": d["poses"],
    "trans": d["trans"],
    "betas": d["betas"][: int(d["num_betas"])],
  }


ARM_JOINTS = ("shoulder_ak10_1_pitch", "shoulder_ak10_2_roll", "elbow_ak80_1_yaw", "elbow_ak80_2_bend", "elbow_ak80_3_forearm")
GLOVE = {"left": "Mirrorlink6__1__1", "right": "link6__1__1"}  # wrist link
ELBOW = {"left": "Mirrorlink4__1__1", "right": "link4__1__1"}  # elbow bend link


class ArmIK:
  def __init__(self, robot: gm.Wato, side: str):
    m = robot.m
    self.robot, self.side = robot, side
    joints = [m.joint(f"{side}_{j}").id for j in ARM_JOINTS]
    self.dofs = np.array([m.jnt_dofadr[j] for j in joints])
    self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
    self.range = np.array([m.jnt_range[j] for j in joints])
    self.glove = m.body(GLOVE[side]).id
    self.elbow = m.body(ELBOW[side]).id

  # pull towards the guard angles; the upper-arm twist (elbow yaw) most, so the
  # IK swings the arm instead of spinning it
  REST_W = np.array([0.05, 0.05, 0.3, 0.05, 0.05])

  def solve(self, glove_pos, elbow_pos, elbow_w, q_rest, q_prev, prev_w=0.5, iters=60) -> float:
    """Glove on target first, then the elbow target, the least motion from the
    previous frame (no flips between IK solutions) and the guard angles."""
    m, d, nv = self.robot.m, self.robot.d, self.robot.m.nv
    jp = np.zeros((3, nv))
    err = np.inf
    for _ in range(iters):
      mujoco.mj_jacBody(m, d, jp, None, self.glove)
      jg = jp[:, self.dofs].copy()
      mujoco.mj_jacBody(m, d, jp, None, self.elbow)
      je = jp[:, self.dofs].copy()
      eg = glove_pos - d.xpos[self.glove]
      ee = elbow_pos - d.xpos[self.elbow]
      q = d.qpos[self.qadr]
      err = float(np.linalg.norm(eg))
      eye = np.eye(len(self.dofs))
      J = np.vstack([10.0 * jg, elbow_w * je, prev_w * eye, self.REST_W[:, None] * eye])
      e = np.concatenate([10.0 * eg, elbow_w * ee, prev_w * (q_prev - q), self.REST_W * (q_rest - q)])
      dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(len(self.dofs)), J.T @ e)
      dq *= min(1.0, 0.1 / max(np.abs(dq).max(), 1e-12))
      d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.range[:, 0], self.range[:, 1])
      self.robot.forward()
      if err < 1e-4:
        break
    return err


def load_amass(path: Path, model_dir: str) -> dict:
  """AMASS SMPL-X clip -> joint positions (T, 3) by Mixamo name, z-up, feet
  on z=0, sampled at the Mixamo takes' rate."""
  c = load_clip(path)
  step = max(1, int(round(c["fps"] / SRC_FPS)))
  _, J = SmplX(model_dir, c["gender"]).pose(c["betas"], c["poses"][::step], c["trans"][::step])
  J = J.astype(np.float64)
  floor = min(J[:, 10, 2].min(), J[:, 11, 2].min())
  P = {k: J[:, i] - [0.0, 0.0, floor] for k, i in SMPLX_JOINTS.items()}
  P["t"] = np.arange(len(J)) * step / c["fps"]
  return P


def source_channels(P: dict, a: float, b: float, hand: str, mirror: bool) -> dict:
  """Punch channels on the source's own time base, detrended to start and end at 0."""
  t = P["t"]
  i = np.where((t >= a) & (t <= b))[0]
  fwd, _ = target_dir(P)
  f0 = fwd[i].mean(0)
  f0 /= np.linalg.norm(f0)
  l0 = np.array([-f0[1], f0[0], 0.0])
  sy = -1.0 if mirror else 1.0

  def stance(v):  # world -> stance frame (x fwd, y left, z up), mirrored
    return np.stack([v[..., :2] @ f0[:2], sy * (v[..., :2] @ l0[:2]), v[..., 2]], -1)

  side = stance(P["LeftArm"] - P["RightArm"] + P["LeftUpLeg"] - P["RightUpLeg"])
  yaw = np.unwrap(np.arctan2(sy * side[i, 1], sy * side[i, 0]))  # chest yaw (mirrored: negated)
  yaw -= yaw[0]
  c, s_ = np.cos(-yaw), np.sin(-yaw)

  def chest_frame(v):  # stance-frame vectors -> the chest's turned frame
    return np.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1], v[:, 2]], 1)

  chest = stance(P["Spine2"][i])
  ch = {
    "glove": chest_frame(stance(P[hand + "Hand"][i]) - chest),
    "elbow": chest_frame(stance(P[hand + "ForeArm"][i]) - chest),
    "glove_world": stance(P[hand + "Hand"][i]),
    "elbow_world": stance(P[hand + "ForeArm"][i]),
    "heading": yaw[:, None],
    "drop": -P["Hips"][i, 2:3],
    "hips": stance(P["Hips"][i])[:, :2],
  }
  s = np.linspace(0.0, 1.0, len(i))[:, None]
  for k, v in ch.items():
    ch[k] = v - v[0] - s * (v[-1] - v[0])
  arm = np.linalg.norm(P[hand + "Arm"][i[0]] - P[hand + "ForeArm"][i[0]]) + np.linalg.norm(
    P[hand + "ForeArm"][i[0]] - P[hand + "Hand"][i[0]]
  )
  ch["t"] = t[i] - t[i[0]]
  ch["arm_len"] = arm
  ch["hip_z"] = P["Hips"][i[0], 2]
  return ch


def recover(ch: dict, cut: float, recovery: float) -> dict:
  """Channels up to `cut`, then a minimum-jerk return to 0 over `recovery`."""
  t = ch["t"]
  keep = t <= cut
  t_rec = cut + np.arange(1, int(np.ceil(recovery * SRC_FPS)) + 1) / SRC_FPS
  w = gm.min_jerk((t_rec - cut) / (t_rec[-1] - cut))[:, None]
  out = dict(ch, t=np.concatenate([t[keep], t_rec]))
  for k, v in ch.items():
    if isinstance(v, np.ndarray) and v.ndim == 2:
      at_cut = np.array([np.interp(cut, t, v[:, j]) for j in range(v.shape[1])])
      out[k] = np.concatenate([v[keep], (1 - w) * at_cut])
  return out


def rot_z(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def base_yaw(robot) -> float:
  R = robot.d.xmat[robot.base].reshape(3, 3)
  return float(np.arctan2(R[1, 0], R[0, 0]))


def sample(ch: dict, key: str, tau: float) -> np.ndarray:
  v = ch[key]
  return np.array([np.interp(tau, ch["t"], v[:, j]) for j in range(v.shape[1])])


class Warp:
  """Source time -> output time, stretched locally by `stretch` (>= 1) on a
  grid over the punch."""

  def __init__(self, dur: float, n: int = 120):
    self.tau = np.linspace(0.0, dur, n)
    self.stretch = np.ones(n)

  def out_time(self, tau):
    cum = np.concatenate([[0.0], np.cumsum((self.stretch[1:] + self.stretch[:-1]) / 2 * np.diff(self.tau))])
    return np.interp(tau, self.tau, cum)

  def src_time(self, t_out):
    return np.interp(t_out, self.out_time(self.tau), self.tau)

  def slow(self, tau_frames: np.ndarray, use: np.ndarray, target: float) -> bool:
    """Stretch where a frame went over target cap use; False if none did."""
    need = np.maximum(1.0, use / target * 1.05)
    if need.max() <= 1.0:
      return False
    grid = np.interp(self.tau, tau_frames, need)
    k = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)  # smooth the warp (~0.1 s)
    grid = np.maximum(grid, np.convolve(np.pad(grid, 15, mode="edge"), k / k.sum(), "valid"))
    self.stretch *= grid
    return True


def chest_targets(robot, ch, tau, yaw0, g_rel, e_rel, sh_rel, scale):
  """CHEST mapping: the source's arm motion relative to its chest, on Wato's
  turned torso. The part of the source's body turn the legs could not make
  (planted feet, MAX_HEADING) is made up, up to MAX_SWING, by swinging the arm
  about the shoulder."""
  yaw = base_yaw(robot)
  rest = float(np.clip(sample(ch, "heading", tau)[0] - (yaw - yaw0), -MAX_SWING, MAX_SWING))
  R, swing = rot_z(yaw), rot_z(rest)
  base = robot.d.xpos[robot.base]
  glove = base + R @ (sh_rel + swing @ (g_rel + scale * sample(ch, "glove", tau) - sh_rel))
  elbow = base + R @ (sh_rel + swing @ (e_rel + scale * sample(ch, "elbow", tau) - sh_rel))
  return glove, elbow


def hook_paths(robot, side, ch, frame_tau, start, end, n, g0, e0, sh0):
  """HOOK mapping: glove (wrist) and elbow paths relative to the shoulder
  (the shoulder roll joint, the arm's pivot) in the hips' turned frame. Guard -> chamber (glove HOOK_ARC[0] outside) -> a
  horizontal arc at HOOK_HEIGHT with the elbow held at HOOK_ELBOW and up at
  shoulder height -> impact (HOOK_ARC[1] across) -> guard. The chamber and
  impact fall where the source glove is furthest back and furthest forward,
  so the punch keeps the source's timing."""
  fwd = ch["glove_world"][:, 0]
  i_imp = int(np.argmax(fwd))
  i_ch = int(np.argmin(fwd[: i_imp + 1]))
  k_ch = start + int(np.searchsorted(frame_tau, ch["t"][i_ch]))
  k_imp = start + int(np.searchsorted(frame_tau, ch["t"][i_imp]))
  k_ch = min(max(k_ch, start + 2), end - 6)
  k_imp = min(max(k_imp, k_ch + 4), end - 2)

  upper = np.linalg.norm(e0 - sh0)
  fore = np.linalg.norm(g0 - e0)
  bend = np.radians(HOOK_ELBOW)
  reach = np.sqrt(upper**2 + fore**2 - 2 * upper * fore * np.cos(bend))  # shoulder -> glove
  alpha = np.arccos((upper**2 + reach**2 - fore**2) / (2 * upper * reach))  # upper arm vs that line
  out = -1.0 if side == "right" else 1.0  # sign of "outside" azimuth

  def glove_at(az):
    return np.array([reach * np.cos(az), reach * np.sin(az), HOOK_HEIGHT])

  def elbow_at(az):
    a = az + out * alpha
    return np.array([upper * np.cos(a), upper * np.sin(a), HOOK_HEIGHT])

  a0, a1 = np.radians(HOOK_ARC[0]) * out, -np.radians(HOOK_ARC[1]) * out
  arc = [(k_ch + (k_imp - k_ch) * j / 4, a0 + (a1 - a0) * j / 4) for j in range(5)]
  vg, ve = g0 - sh0, e0 - sh0
  gw = [(0, vg, True), (start, vg, True)] + [(int(k), glove_at(a), j in (0, 4)) for j, (k, a) in enumerate(arc)]
  ew = [(0, ve, True), (start, ve, True)] + [(int(k), elbow_at(a), j in (0, 4)) for j, (k, a) in enumerate(arc)]
  gw += [(end, vg, True), (n, vg, True)]
  ew += [(end, ve, True), (n, ve, True)]
  return gm.smooth_through(gw, n), gm.smooth_through(ew, n)


def build(robot, stance, guard_row, ch, side: str, mapping: str, warp: Warp, fps: float, hold: float):
  """Returns (CSV rows, worst glove IK error [m], punch frame span, source time per frame)."""
  dur = ch["t"][-1]
  tl = gm.Timeline(stance, fps)
  tl.stance_body(stop=True)
  tl.frames(hold)
  tl.stance_body(stop=True)
  start = len(tl.feet_frames)

  leg_scale = stance.base_z / ch["hip_z"]
  lead, rear = tl.sole(gm.LEAD), tl.sole(gm.REAR)
  axis = (lead - rear) / np.linalg.norm(lead - rear)
  half = np.linalg.norm(lead - rear) / 2
  n_keys = max(2, int(round(dur / KEY_DT)))
  for k in range(1, n_keys + 1):
    tau = dur * k / n_keys
    tl.frames(warp.out_time(tau) - warp.out_time(dur * (k - 1) / n_keys))
    w = np.clip(leg_scale * sample(ch, "hips", tau) @ axis / half, -MAX_WEIGHT, MAX_WEIGHT)
    com = tl.towards(gm.LEAD, w) if w >= 0 else tl.towards(gm.REAR, -w)
    drop = float(np.clip(leg_scale * sample(ch, "drop", tau)[0], *DROP_RANGE))
    heading = float(np.clip(sample(ch, "heading", tau)[0], -MAX_HEADING, MAX_HEADING))
    tl.body(com, drop=drop, heading=heading, stop=k == n_keys)
  end = len(tl.feet_frames)
  tl.frames(hold)
  tl.stance_body(stop=True)

  robot.set_csv_row(guard_row)
  legs, _ = gm.solve_timeline(robot, stance, tl)

  robot.set_csv_row(guard_row)
  arm = ArmIK(robot, side)
  g0, e0 = robot.d.xpos[arm.glove].copy(), robot.d.xpos[arm.elbow].copy()
  # guard glove and elbow in the torso's frame (Wato's torso is rigid on the base)
  yaw0 = base_yaw(robot)
  g_rel = rot_z(-yaw0) @ (g0 - robot.d.xpos[robot.base])
  e_rel = rot_z(-yaw0) @ (e0 - robot.d.xpos[robot.base])
  q_rest = robot.d.qpos[arm.qadr].copy()
  # Wato arm length (shoulder pitch joint -> elbow -> wrist link) at the guard
  sh_body = int(robot.m.joint(f"{side}_shoulder_ak10_1_pitch").bodyid[0])
  sh_rel = rot_z(-yaw0) @ (robot.d.xpos[sh_body] - robot.d.xpos[robot.base])
  sh = robot.d.xpos[sh_body]
  arm_len = np.linalg.norm(e0 - sh) + np.linalg.norm(g0 - e0)
  scale = arm_len / ch["arm_len"]
  reach = REACH * arm_len

  out, worst = [], 0.0
  roll = robot.m.joint(f"{side}_shoulder_ak10_2_roll").id
  if mapping == "hook":
    frame_tau = warp.src_time((np.arange(start, end) + 1 - start) / (end - start) * warp.out_time(dur))
    hook_g, hook_e = hook_paths(robot, side, ch, frame_tau, start, end, len(legs), g0, e0, robot.d.xanchor[roll].copy())

  arm_q = q_rest.copy()
  taus = np.zeros(len(legs))
  for k, row in enumerate(legs):
    robot.set_csv_row(row)
    robot.d.qpos[arm.qadr] = arm_q  # warm start from the previous frame
    robot.forward()
    if start <= k < end:
      tau = float(warp.src_time((k + 1 - start) / (end - start) * warp.out_time(dur)))
      taus[k] = tau
      env = gm.min_jerk(min(1.0, 4 * min(tau, dur - tau) / dur))  # elbow target fades in/out
      elbow_w = 0.5 * env
      if mapping == "hook":  # carried with the hips' turn
        R = rot_z(base_yaw(robot) - yaw0)
        glove = robot.d.xanchor[roll] + R @ hook_g[k]
        elbow = robot.d.xanchor[roll] + R @ hook_e[k]
        elbow_w = 1.0
      elif mapping == "world":  # the glove's own path, Wato's hips turn as far as they can
        glove = g0 + scale * sample(ch, "glove_world", tau)
        elbow = e0 + scale * sample(ch, "elbow_world", tau)
      else:
        glove, elbow = chest_targets(robot, ch, tau, yaw0, g_rel, e_rel, sh_rel, scale)
      # Wato's guard already holds the glove well out, so the boxer's full
      # extension can overshoot: stop at the arm's reach from its shoulder
      to = glove - robot.d.xpos[sh_body]
      glove = robot.d.xpos[sh_body] + to * min(1.0, reach / np.linalg.norm(to))
      worst = max(worst, arm.solve(glove, elbow, elbow_w, q_rest, arm_q))
      # ease into the exact guard angles (the IK can land on another arm
      # configuration for the same glove, which would jump at the end)
      w = gm.min_jerk(np.clip((k + 1 - (end - SETTLE)) / SETTLE, 0.0, 1.0))
      if w > 0:
        robot.d.qpos[arm.qadr] = (1 - w) * robot.d.qpos[arm.qadr] + w * q_rest
        robot.forward()
    else:
      robot.d.qpos[arm.qadr] = q_rest
      robot.forward()
    arm_q = robot.d.qpos[arm.qadr].copy()
    out.append(robot.csv_row())
  taus[end:] = dur
  return np.array(out), worst, (start, end), taus


def cap_use(rows: np.ndarray, fps: float) -> np.ndarray:
  """(frames, joints): joint speed / its motor's speed cap (claws and wrists left out)."""
  caps = np.array([next(v for p, v in SPEED_CAPS.items() if re.fullmatch(p, n)) for n in CSV_JOINT_NAMES])
  use = np.abs(np.gradient(rows[:, 7:], 1.0 / fps, axis=0)) / caps
  use[:, [i for i, n in enumerate(CSV_JOINT_NAMES) if "claw" in n or "wrist" in n]] = 0
  return use


def main(
  moves: list[str] = list(SOURCES),  # noqa: B006
  fbx_dir: str = "../../../../motion_data/Combat",
  amass_dir: str = "data/amass/ACCAD/Male2MartialArtsPunches_c3d",
  smplx_dir: str = "../../../../body_models/smplx",
  output_dir: str = "data/moves/csv",
  stance_file: str = "data/motions/boxing.csv",
  stance_time: float = 8.2,
  stance_fps: float = 120.0,
  stance_scale: float = 0.75,
  crouch: float = 0.06,
  hold: float = 0.15,
  fps: float = 50.0,
  max_cap_use: float = 0.8,
):
  """Args:
    moves: which punches to retarget (names in SOURCES).
    fbx_dir: folder with the Mixamo takes (<take>_mixamo.fbx).
    amass_dir: folder with the ACCAD clips (<clip>_stageii.npz).
    smplx_dir: folder with SMPLX_MALE/FEMALE/NEUTRAL.npz.
    output_dir: one <move>.csv per punch.
    stance_file / stance_time / stance_fps / stance_scale / crouch: the stance,
      same defaults as generate_moves.py so the guard matches its clips.
    hold: stance pause before and after the punch [s].
    fps: output frame rate.
    max_cap_use: slow a punch down until no joint exceeds this fraction of its cap.
  """
  robot = gm.Wato()
  stance = gm.make_stance(robot, stance_file, stance_time, stance_fps, stance_scale, crouch)
  guard_row = robot.csv_row()
  os.makedirs(output_dir, exist_ok=True)
  takes: dict[str, dict] = {}

  print(f"{'punch':14s} {'source':38s} {'time':>6s} {'slowed':>7s} {'peak cap use':>13s} {'joint':24s} {'glove err':>9s}")
  for name in moves:
    take, a, b, hand, mirror, mapping = SOURCES[name]
    if take not in takes:
      if take.startswith("ACCAD/"):
        takes[take] = load_amass(Path(amass_dir) / f"{take[6:]}_stageii.npz", smplx_dir)
      else:
        takes[take] = load_take(fbx_to_glb(Path(fbx_dir) / f"{take}_mixamo.fbx", Path("data/mixamo/glb")))
    ch = source_channels(takes[take], a, b, hand, mirror)
    if name in RECOVER:
      ch = recover(ch, *RECOVER[name])
    side = hand.lower() if not mirror else ("right" if hand == "Left" else "left")
    # slow down where needed; an IK jump is not fixed by slowing, so stop at
    # MAX_SLOW and keep the best attempt
    warp = Warp(ch["t"][-1])
    best = None
    for _ in range(20):
      rows, worst, (s, e), taus = build(robot, stance, guard_row, ch, side, mapping, warp, fps, hold)
      per_joint = cap_use(rows, fps)
      use, joint = float(per_joint.max()), CSV_JOINT_NAMES[int(np.argmax(per_joint.max(0)))]
      if best is None or use < best[1]:
        best = (rows, use, joint, worst, (s, e), warp.out_time(ch["t"][-1]))
      if use <= max_cap_use or warp.out_time(ch["t"][-1]) > MAX_SLOW * ch["t"][-1]:
        break
      if not warp.slow(taus[s:e], per_joint.max(1)[s:e], max_cap_use):
        break
    rows, use, joint, worst, (s, e), out_dur = best
    tempo = out_dur / ch["t"][-1]
    np.savetxt(os.path.join(output_dir, f"{name}.csv"), rows, delimiter=",")
    src = f"{take} {a:.2f}-{b:.2f}{' mirrored' if mirror else ''}"
    print(f"{name:14s} {src:38s} {(e - s) / fps:5.2f}s {tempo:6.2f}x {use:12.0%}  {joint:24s} {100 * worst:7.1f} cm")
    if worst > 0.03:
      print(f"  WARNING: {name}: glove misses its path by up to {100 * worst:.1f} cm (out of reach)")
    if use > max_cap_use:
      print(f"  WARNING: {name}: still above {max_cap_use:.0%} of a speed cap")
  print(f"(source frame rate {SRC_FPS} fps)")


if __name__ == "__main__":
  tyro.cli(main)
