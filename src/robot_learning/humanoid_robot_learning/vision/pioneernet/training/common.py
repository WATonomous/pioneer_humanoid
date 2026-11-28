"""Shared plumbing for the encoder benchmark: repo paths, MuJoCo rendering, label extraction.

Every script in this folder imports this first. Run with the bench venv:
    scratch/encoder_bench/.venv/bin/python scratch/encoder_bench/<script>.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parents[5]   # <repo>/src/robot_learning/humanoid_robot_learning/vision/pioneernet/training
# generated data, checkpoints and figures (several GB) go here, not next to the code
WORK = Path(os.environ.get("PIONEERNET_WORK", Path.home() / "pioneernet_work"))   # outside the repo
# The robot / scene / teleop code the benchmark runs on is PINNED to its own git worktree (commit ca66cb70, defm-depth),
# not the shared checkout: on 2026-11-14 the shared checkout moved to a newer main mid-queue (#565 dropped TeleopSim's
# robot argument, #555 changed the gripper physics) and every job failed. BENCH_SRC overrides it.
# The v13 run used the simulator code at commit ca66cb70 (a git worktree of that commit); set BENCH_SRC to its src/.
PINNED_SRC = Path(os.environ.get("BENCH_SRC_PINNED", "/nonexistent"))
SRC = Path(os.environ.get("BENCH_SRC", PINNED_SRC if PINNED_SRC.exists() else REPO / "src"))
SYNTH = SRC / "teleop" / "pioneer_leader_arm_teleop" / "synthetic"
for p in (SRC / "pioneer_humanoid", SRC / "simulation" / "mujoco_scenes", SRC / "robot_learning",
          SYNTH.parent, SYNTH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Sim robot: "8dof" (default) = pioneer_bimanual_arm_8dof, the current rig (7-DoF arms + 1-DoF grippers), BOTH arms
# controlled (16-D state/action). "armv2" = the previous 6-DoF arms, left arm only (7-D): the Nov 11 results.
# Each robot has its own data / results / figures folders, so the two never mix.
ROBOT = os.environ.get("BENCH_ROBOT", "8dof")
BIMANUAL = ROBOT == "8dof"
# Task scene (a humanoid_mujoco_scenes scene with a synthetic operator: peg_insert, drawer_stow, zip_tie, duplo).
# peg_insert is the main benchmark (with its geometric probe labels); the others test transfer to new tasks.
SCENE = os.environ.get("BENCH_SCENE", "peg_insert")
_SUF = ("" if ROBOT == "armv2" else f"_{ROBOT}") + ("" if SCENE == "peg_insert" else f"_{SCENE}")
# Encoders (PioneerNet checkpoints) are per robot, not per task: v4 is distilled on peg_insert images and reused
# unchanged on the other tasks -- the transfer test.
ENC_DIR = WORK / ("results" if ROBOT == "armv2" else f"results_{ROBOT}")
DATA = WORK / f"data{_SUF}"
OUT = WORK / f"results{_SUF}"
FIG = WORK / f"figures{_SUF}"
for d in (DATA, OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)

# Render size: 4:3 like both real sensors (D455 colour crop and the wrist webcam pinhole).
H, W = 240, 320
CAMS = ("ego", "wrist_left")
# 100 Hz control ticks per recorded frame (synthetic_teleop record_every) -> 25 Hz state/action, the dataset fps.
RECORD_EVERY = 4
SCHEMA = SRC / "robot_learning" / "config" / ("pioneer_8dof_dataset_schema.yaml" if ROBOT == "8dof"
                                              else "pioneer_humanoid_dataset_schema.yaml")


def robot_params():
    """The robot's arm_params module (joint / gripper names, cameras)."""
    if ROBOT == "8dof":
        from pioneer_humanoid import arm_8dof_params as p
    elif ROBOT == "armv2":
        from pioneer_humanoid import arm_params as p
    else:
        raise ValueError(f"BENCH_ROBOT={ROBOT!r}: known 8dof, armv2")
    return p


def sides():
    return ("LEFT", "RIGHT") if BIMANUAL else ("LEFT",)


def instruction(model, data, k: int) -> str:
    """The task's current instruction (what a language prompt would say) at step k, e.g. "put the red block in
    the drawer". Multi-step scenes randomise the colour order per episode, so without it the next target is
    ambiguous. Scenes without steps return their one task."""
    import importlib
    from humanoid_mujoco_scenes import scene_progress
    mod = importlib.import_module(f"humanoid_mujoco_scenes.{SCENE}.scene")
    prog = scene_progress(SCENE)
    if prog is None or not hasattr(mod, "_step_text"):
        return SCENE
    total = prog(model, data)[1]
    return mod._step_text(data, min(int(k), total - 1))


def grip_indices():
    """Positions of the gripper entries in the state / action vector (ArmIO order), without building a model."""
    p, idx, i = robot_params(), [], 0
    for side in sides():
        i += len(getattr(p, f"{side}_ARM_JOINTS"))
        idx.append(i)
        i += 1
    return idx


class ArmIO:
    """The policy's state / action vector on a model: for each controlled arm (left, then right when BIMANUAL),
    the arm joints (rad) + gripper closure 0 (open) .. 1 (closed) -- the order of the dataset schema's joint_names.
    state() reads measured joints and the fingers' mean closure; apply() writes one action as actuator targets."""

    def __init__(self, model):
        import numpy as np
        p = robot_params()
        self.arms = []
        for side in sides():
            joints, grips = getattr(p, f"{side}_ARM_JOINTS"), getattr(p, f"{side}_GRIPPER_JOINTS")
            opn, cls = getattr(p, f"{side}_GRIPPER_OPEN"), getattr(p, f"{side}_GRIPPER_CLOSED")
            self.arms.append(dict(
                qpos=[model.joint(j).qposadr[0] for j in joints], acts=[model.actuator(j).id for j in joints],
                gq=[model.joint(j).qposadr[0] for j in grips], ga=[model.actuator(j).id for j in grips],
                o=[opn[j] for j in grips], c=[cls[j] for j in grips]))
        self.dim = sum(len(a["acts"]) + 1 for a in self.arms)
        ends = np.cumsum([len(a["acts"]) + 1 for a in self.arms])
        self.grip_idx = [int(e) - 1 for e in ends]   # positions of the gripper entries in the vector

    @staticmethod
    def _closure(vals, a):
        g = sum((v - o) / (c - o) for v, o, c in zip(vals, a["o"], a["c"])) / len(vals)
        return min(max(g, 0.0), 1.0)

    def state(self, qpos):
        import numpy as np
        return np.concatenate([np.append(qpos[a["qpos"]], self._closure(qpos[a["gq"]], a)) for a in self.arms]).astype(np.float32)

    def action_from_ctrl(self, ctrl):
        """The action that reproduces the current actuator targets (for arms the operator does not drive)."""
        import numpy as np
        return np.concatenate([np.append(ctrl[a["acts"]], self._closure(ctrl[a["ga"]], a)) for a in self.arms]).astype(np.float32)

    def apply(self, data, action):
        i = 0
        for a in self.arms:
            n = len(a["acts"])
            data.ctrl[a["acts"]] = action[i:i + n]
            g = float(min(max(action[i + n], 0.0), 1.0))
            data.ctrl[a["ga"]] = [o + g * (c - o) for o, c in zip(a["o"], a["c"])]
            i += n + 1


def contract_check() -> None:
    """Fail loudly if this benchmark's data no longer matches the repo's data contract for the robot
    (pioneer_8dof_dataset_schema.yaml / pioneer_humanoid_dataset_schema.yaml): frame rate, joint list and order,
    units, camera aspect. A policy trained here outputs the schema's action vector (ArmIO): per controlled arm the
    joint targets in rad + gripper closure 0 (open) .. 1 (closed), at 25 Hz."""
    import yaml
    from leader_mapping import CONTROL_DT
    p = robot_params()
    cfg = yaml.safe_load(open(SCHEMA))
    errs = []
    fps = round(1 / (CONTROL_DT * RECORD_EVERY))
    if cfg["fps"] != fps:
        errs.append(f"schema fps {cfg['fps']} != benchmark {fps} Hz (CONTROL_DT {CONTROL_DT} x {RECORD_EVERY})")
    if cfg.get("state_unit") != "rad":
        errs.append(f"schema state_unit {cfg.get('state_unit')!r}, benchmark records rad")
    for side in sides():
        sim_joints = getattr(p, f"{side}_ARM_JOINTS")
        arm = cfg["arms"][side.lower()]
        if len(arm) != len(sim_joints) + 1 or arm[-1] != f"{side.lower()}_gripper":
            errs.append(f"schema {side.lower()} arm {arm} is not {len(sim_joints)} joints + {side.lower()}_gripper")
        elif ROBOT == "8dof" and arm[:-1] != list(sim_joints):   # the 8dof schema names ARE the sim joints
            errs.append(f"schema {side.lower()} joints {arm[:-1]} != sim {list(sim_joints)}")
    for c in CAMS:
        spec = cfg["images"].get(c)
        if spec is None or abs(spec["width"] / spec["height"] - W / H) > 1e-6:
            errs.append(f"camera {c}: schema {spec} does not have the benchmark's {W}x{H} aspect")
    if errs:
        raise RuntimeError("benchmark / data-contract mismatch:\n  " + "\n  ".join(errs))


def ffmpeg_exe() -> str:
    """An ffmpeg with libx264 (imageio-ffmpeg's static build)."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()
