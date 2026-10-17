"""Build all the boxing motion data: the move library and the two training combos.

1. Move library (data/moves/): one clip per move, each starting and ending in
   the same guard stance, as CSV and NPZ, listed in library.yaml.
   Footwork and defence come from generate_moves.py, one move per run. The
   punches are retargeted by retarget_mixamo_punches.py (the source's glove
   path, elbow, hip turn and timing, slowed only where a motor would pass
   --max-cap-use of its speed cap): jab, cross and rear hook from the Mixamo
   boxing takes, lead hook and lead uppercut from the ACCAD martial-arts clips.
   Every punch is whole body: the hips turn.
2. Impact labels: for each punch, which glove, which way it drives (straight:
   forward, hook: across, uppercut: up) and the impact frame, mid-strike where
   the glove has covered IMPACT_FRACTION of its travel. The power task rewards
   force there (humanoid_tracking/power.py).
3. Training combos (data/motions/): moves chained end to end into one long
   clip, each turned and moved to start where the previous one ended, for the
   tracking skills: punch_training (every punch several times, mixed with
   footwork and defence) and footwork_training (no punches). Steps and pivots
   cancel so each ends near where it started. Each comes with a .yaml of its
   segments and punch impacts.

  uv run scripts/build_move_library.py                  # all of it
  uv run scripts/build_move_library.py --combos-only    # rebuild only the combos
"""

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import tyro
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# move id -> generate_moves.py code
GENERATED = {
  "step_forward": "F",
  "step_back": "B",
  "step_left": "L",
  "step_right": "R",
  "pivot_left": "PL",
  "pivot_right": "PR",
  "slip_left": "SL",
  "slip_right": "SR",
  "weave_left": "WL",
  "weave_right": "WR",
  "duck": "D",
  "deep_duck": "DD",
}

# retarget_mixamo_punches.py, move -> (glove, kind); orthodox: the left hand leads
PUNCHES = {
  "jab": ("left", "straight"),
  "cross": ("right", "straight"),
  "rear_hook": ("right", "hook"),
  "lead_hook": ("left", "hook"),
  "lead_uppercut": ("left", "uppercut"),
}
IMPACT_FRACTION = 0.5
STRAIGHT_PULL = 0.5  # a straight's target is pulled this far towards the centreline...
STRAIGHT_PULL_MAX = 0.04  # ...at most this far [m]

PUNCH_COMBO = (
  "guard", "jab", "cross", "step_forward", "lead_hook", "rear_hook", "step_back",
  "lead_uppercut", "slip_left", "cross", "slip_right", "jab", "duck", "rear_hook",
  "step_left", "jab", "step_right", "weave_left", "lead_hook", "weave_right", "cross",
  "pivot_left", "jab", "cross", "pivot_right", "deep_duck", "lead_uppercut", "guard",
)  # fmt: skip
FOOTWORK_COMBO = (
  "guard", "step_forward", "step_forward", "slip_left", "step_back", "step_back", "step_left",
  "duck", "step_right", "weave_left", "step_forward", "pivot_left", "step_back", "slip_right",
  "step_right", "step_right", "weave_right", "step_left", "step_left", "pivot_right", "deep_duck",
  "step_forward", "step_left", "step_back", "step_right", "slip_left", "step_forward", "step_back",
  "duck", "step_right", "step_left", "guard",
)  # fmt: skip


def run(cmd: list[str]) -> None:
  print("$", " ".join(cmd))
  subprocess.run(cmd, check=True, cwd=HERE.parent)


# ---------------------------------------------------------------- impact labels


class GloveFK:
  """Glove sphere and head hitbox centres from CSV rows (root pos, root quat xyzw, 28 joints)."""

  def __init__(self):
    from humanoid_boxing.fighters import add_hitboxes
    from humanoid_tracking.robot import CSV_JOINT_NAMES, get_spec

    spec = get_spec()
    add_hitboxes(spec, "red")
    self.m = spec.compile()
    self.d = mujoco.MjData(self.m)
    self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in CSV_JOINT_NAMES])
    self.geom = {"left": self.m.geom("glove_l").id, "right": self.m.geom("glove_r").id, "head": self.m.geom("head").id}

  def __call__(self, rows: np.ndarray, part: str) -> np.ndarray:
    out = np.empty((len(rows), 3))
    for k, row in enumerate(rows):
      self.d.qpos[:3] = row[:3]
      self.d.qpos[3:7] = row[[6, 3, 4, 5]]
      self.d.qpos[self.qadr] = row[7:]
      mujoco.mj_kinematics(self.m, self.d)
      out[k] = self.d.geom_xpos[self.geom[part]]
    return out


def impact_label(rows: np.ndarray, move: str, fps: float, fk: GloveFK) -> dict:
  """Impact of one punch clip, in the clip's frame (opponent along +x at the start).
  `aim_pos` is where a target is centred: the glove, except for straights, pulled
  towards the centreline so the target sits in front of the middle of the body."""
  from humanoid_boxing.fighters import GLOVE_RADIUS

  hand, kind = PUNCHES[move]
  d = {"straight": [1.0, 0.0, 0.0], "hook": [0.0, -1.0 if hand == "left" else 1.0, 0.0], "uppercut": [0.0, 0.0, 1.0]}
  d = np.array(d[kind])
  pos = fk(rows, hand)
  along = (pos - pos[0]) @ d
  top = int(np.argmax(along))
  low = int(np.argmin(along[: top + 1]))
  k = low + int(np.argmax(along[low : top + 1] >= along[low] + IMPACT_FRACTION * (along[top] - along[low])))
  aim = pos[k].copy()
  if kind == "straight":
    head_y = fk(rows[k : k + 1], "head")[0, 1]
    aim[1] += np.clip(STRAIGHT_PULL * (head_y - aim[1]), -STRAIGHT_PULL_MAX, STRAIGHT_PULL_MAX)
  speed = np.gradient(pos, 1.0 / fps, axis=0) @ d
  return {
    "hand": hand,
    "kind": kind,
    "frame": k,
    "time_s": round(k / fps, 3),
    "direction": [float(x) for x in d],
    "glove_pos": [round(float(x), 4) for x in pos[k]],
    "aim_pos": [round(float(x), 4) for x in aim],
    "glove_speed": round(float(speed[k]), 3),
    "glove_radius": GLOVE_RADIUS,
  }


# ---------------------------------------------------------------- training combos


def yaw_of(q: np.ndarray) -> np.ndarray:
  """Yaw of xyzw quaternions."""
  x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot2(a: float) -> np.ndarray:
  return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def place(rows: np.ndarray, xy: np.ndarray, yaw: float) -> tuple[np.ndarray, float]:
  """Clip turned and moved so its first root pose is at `xy`, heading `yaw`.
  Returns (rows, the turn applied)."""
  d = yaw - float(yaw_of(rows[0, 3:7]))
  c, s = np.cos(d / 2), np.sin(d / 2)
  x, y, z, w = rows[:, 3], rows[:, 4], rows[:, 5], rows[:, 6]
  out = rows.copy()
  out[:, :2] = (rows[:, :2] - rows[0, :2]) @ rot2(d).T + xy
  out[:, 3:7] = np.stack([c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z], 1)
  return out, d


def build_combo(sequence, library_dir: Path, output: Path, device: str) -> None:
  """Chain library clips into one training clip: <output>.csv / .npz / .yaml."""
  lib = yaml.safe_load((library_dir / "library.yaml").read_text())
  fps = float(lib["fps"])
  out_rows, segments, n = [], [], 0
  for i, move in enumerate(sequence):
    src = np.loadtxt(library_dir / "csv" / f"{move}.csv", delimiter=",", ndmin=2)
    rows, turn, skipped = src, 0.0, 0
    if out_rows:
      prev = out_rows[-1][-1]
      rows, turn = place(src, prev[:2], float(yaw_of(prev[3:7])))
      gap = np.abs(rows[0, 2:] - prev[2:]).max()
      if gap > 1e-3:
        raise SystemExit(f"{sequence[i - 1]} -> {move}: clips do not join ({gap:.3g})")
      rows, skipped = rows[1:], 1  # the first frame repeats the previous clip's last
    seg = {"move": move, "start": n, "end": n + len(rows) - 1}
    imp = lib["moves"][move].get("impact")
    if imp:  # the impact, carried into the combo's frame
      first, R = (out_rows[-1][-1] if out_rows else rows[0]), rot2(turn)
      glove, aim, d = np.array(imp["glove_pos"]), np.array(imp["aim_pos"]), np.array(imp["direction"])
      glove[:2] = R @ (glove[:2] - src[0, :2]) + first[:2]
      aim[:2] = R @ (aim[:2] - src[0, :2]) + first[:2]
      d[:2] = R @ d[:2]
      seg["impact"] = {
        "hand": imp["hand"],
        "kind": imp["kind"],
        "frame": n + imp["frame"] - skipped,
        "direction": [round(float(x), 4) for x in d],
        "glove_pos": [round(float(x), 4) for x in glove],
        "aim_pos": [round(float(x), 4) for x in aim],
      }
    segments.append(seg)
    out_rows.append(rows)
    n += len(rows)

  rows = np.concatenate(out_rows)
  output.parent.mkdir(parents=True, exist_ok=True)
  csv, npz = output.with_suffix(".csv"), output.with_suffix(".npz")
  np.savetxt(csv, rows, delimiter=",")
  run([sys.executable, "scripts/csv_to_npz.py", "--input-file", str(csv), "--input-fps", str(fps),
       "--output-file", str(npz), "--device", device])  # fmt: skip
  # frame numbers are the CSV's; the NPZ can come out a frame shorter (resampling)
  n_npz = len(np.load(npz)["joint_pos"])
  meta = {"fps": fps, "frames": n_npz, "duration_s": round(n_npz / fps, 2), "segments": segments}
  output.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
  drift = np.linalg.norm(rows[-1, :2] - rows[0, :2])
  turn = np.degrees(float(yaw_of(rows[-1, 3:7]) - yaw_of(rows[0, 3:7])))
  punches = sum("impact" in s for s in segments)
  print(f"{output.name}: {len(sequence)} moves, {punches} punches, {len(rows) / fps:.1f} s; "
        f"ends {drift:.2f} m / {turn:+.0f} deg from the start")  # fmt: skip


# ---------------------------------------------------------------- main


def main(
  out_dir: str = "data/moves",
  combos_dir: str = "data/motions",
  combos_only: bool = False,
  hold: float = 0.15,
  guard_hold: float = 0.5,
  max_cap_use: float = 0.8,
  device: str = "cuda:0",
):
  """Args:
    out_dir: the move library: csv/, npz/ and library.yaml.
    combos_dir: where punch_training.* and footwork_training.* go.
    combos_only: keep the existing library, rebuild only the training combos.
    hold: stance pause before and after each move [s] (short, so moves chain quickly).
    guard_hold: length of the idle guard clip is 3x this [s].
    max_cap_use: punches are slowed until every joint is under this fraction of its speed cap.
    device: device for the CSV -> NPZ conversion.
  """
  out = Path(out_dir)
  if not combos_only:
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "npz").mkdir(parents=True, exist_ok=True)
    py, fps = sys.executable, 50.0
    library: dict[str, dict] = {}

    # guard: the idle loop and the base for every other move
    run([py, "scripts/generate_moves.py", "--sequence", "H", "--hold", str(guard_hold),
         "--output-file", str(out / "csv" / "guard.csv")])  # fmt: skip
    library["guard"] = {"type": "idle", "source": "generate_moves.py H"}
    for move, code in GENERATED.items():
      run([py, "scripts/generate_moves.py", "--sequence", code, "--hold", str(hold),
           "--output-file", str(out / "csv" / f"{move}.csv")])  # fmt: skip
      kind = "footwork" if code in ("F", "B", "L", "R", "PL", "PR") else "defence"
      library[move] = {"type": kind, "source": f"generate_moves.py {code}"}

    run([py, "scripts/retarget_mixamo_punches.py", "--moves", *PUNCHES, "--hold", str(hold),
         "--max-cap-use", str(max_cap_use), "--output-dir", str(out / "csv")])  # fmt: skip
    from retarget_mixamo_punches import SOURCES

    fk = GloveFK()
    for move in PUNCHES:
      take, t0, t1, _, mirror, _ = SOURCES[move]
      rows = np.loadtxt(out / "csv" / f"{move}.csv", delimiter=",", ndmin=2)
      library[move] = {
        "type": "attack",
        "source": f"retarget_mixamo_punches.py: {take} {t0:.2f}-{t1:.2f} s" + (" mirrored" if mirror else ""),
        "impact": impact_label(rows, move, fps, fk),
      }

    for move, info in library.items():
      csv, npz = out / "csv" / f"{move}.csv", out / "npz" / f"{move}.npz"
      run([py, "scripts/csv_to_npz.py", "--input-file", str(csv), "--input-fps", str(fps),
           "--output-file", str(npz), "--device", device])  # fmt: skip
      n = len(np.load(npz)["joint_pos"])
      info.update(file=f"npz/{move}.npz", frames=n, duration_s=round(n / fps, 2))
    with open(out / "library.yaml", "w") as f:
      yaml.safe_dump({"fps": fps, "moves": library}, f, sort_keys=False)
    print(f"\nWrote {len(library)} moves to {out / 'library.yaml'}")

  for name, seq in (("punch_training", PUNCH_COMBO), ("footwork_training", FOOTWORK_COMBO)):
    build_combo(seq, out, Path(combos_dir) / name, device)


if __name__ == "__main__":
  tyro.cli(main)
