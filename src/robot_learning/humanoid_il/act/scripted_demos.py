"""Scripted tidy_table demonstrations, written in the leader teleop's LeRobot format. Sim only (MuJoCo, CPU).

A hand-written controller (gripper-down IK on the simulator's true object poses) tidies each episode in the
announced order and returns home; only episodes the scene scores a success (all binned, in order, nothing
dropped or toppled) are kept. Same cameras, state/action, task text, subtask_index and
observation.environment_state as `pioneer_leader_arm_teleop.py --record`, so the datasets train the same policy
and can be merged. leader_angles / leader_counts are NaN here: no leader arm was involved.

The true poses are used only to make the data; a policy trained on it sees just the images and joints.

    cd src/robot_learning/humanoid_il/act
    PYTHONPATH=../../:../../../pioneer_humanoid:../../../simulation/mujoco_scenes \\
        MUJOCO_GL=egl python scripted_demos.py --episodes 50 --dataset_root /data/tidy_scripted

Needs: mujoco, lerobot (>= 0.4), torch -- the repo's src/robot_learning[sim] extras.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))                                   # humanoid_robot_learning
sys.path.insert(0, str(_HERE.parents[2] / "pioneer_humanoid"))
sys.path.insert(0, str(_HERE.parents[2] / "simulation" / "mujoco_scenes"))

from humanoid_robot_learning.sim_teleop_record import add_record_args, load_record_schema, make_sim_recorder  # noqa: E402
from ik import FINGERS_BELOW_GRASP, LeftArmIK  # noqa: E402
from tidy_sim import CONTROL_DT, RECORD_EVERY, TidySim  # noqa: E402

LEADER_SERVOS = ["A", "B", "C", "D", "E", "F", "G"]   # pioneer_leader_arm_teleop/servo_leader.SERVO_IDS
OPEN_GAP = 0.0958        # jaw gap fully open (m); closes ~linearly to slightly negative at grip 1
SPEED = dict(travel=0.12, descend=0.05, lift=0.06, carry=0.10, lower=0.05)
READY_HEIGHT = 0.18      # finger tips this far above the table at the ready pose (the tallest object is 75 mm)


class Abort(Exception):
    """The script can't finish this episode (unreachable pose, object not binned): discard it."""


class Demo:
    def __init__(self, sim: TidySim, recorder):
        self.sim, self.rec = sim, recorder
        self.ik = LeftArmIK(sim.model)
        self.steps = 0
        self.nan_leader = np.full(len(LEADER_SERVOS), np.nan, dtype=np.float32)

    # -- one control step, recorded like the teleop: (state, action) before the physics step
    def tick(self, q, grip):
        sim = self.sim
        action = np.append(q, grip).astype(np.float32)
        if self.rec is not None and self.steps % RECORD_EVERY == 0:
            index, _, task = sim.step_info()
            extras = {"leader_angles": self.nan_leader, "leader_counts": self.nan_leader,
                      "subtask_index": np.array([index], dtype=np.float32)}
            env = sim.env_state()
            if env is not None:
                extras["observation.environment_state"] = env
            self.rec.push_frame_to_buffer(action, sim.state(), sim.images(), extras=extras, task=task)
        sim.act(action)
        self.steps += 1

    # -- motions
    def move(self, pos, yaw, speed):
        sim = self.sim
        q = sim.target.copy()
        p0, R0 = self.ik.fk(sim.data.qpos, q)
        y0 = self.ik.yaw_of(R0)
        dist = max(float(np.linalg.norm(np.asarray(pos) - p0)), abs(yaw - y0) * 0.08, 1e-3)
        n = max(2, int(dist / speed / CONTROL_DT))
        path = []
        for i in range(1, n + 1):           # straight line, each step's IK warm-started from the last
            s = i / n
            s = s * s * (3 - 2 * s)
            q, err = self.ik.solve(sim.data.qpos, q, p0 + s * (np.asarray(pos) - p0), y0 + s * (yaw - y0), iters=15)
            path.append(q)
        if err > 0.002:                     # stuck in a poor local solution: solve the goal afresh, go joint-space
            q_goal, err = self.ik.solve_any(sim.data.qpos, [q, sim.target, sim.home], pos, yaw)
            if err > 0.005:
                raise Abort(f"pose out of reach ({err * 1000:.0f} mm short)")
            q_start = sim.target.copy()
            path = [q_start + (s * s * (3 - 2 * s)) * (q_goal - q_start) for s in np.arange(1, n + 1) / n]
        for q in path:
            self.tick(q, sim.grip)

    def grip(self, target, seconds):
        g0, n = self.sim.grip, max(1, int(seconds / CONTROL_DT))
        for i in range(1, n + 1):
            self.tick(self.sim.target, g0 + (target - g0) * i / n)

    def joint_move(self, q1, seconds, grip=0.0):
        q0, n = self.sim.target.copy(), max(1, int(seconds / CONTROL_DT))
        for i in range(1, n + 1):
            s = i / n
            self.tick(q0 + (s * s * (3 - 2 * s)) * (np.asarray(q1) - q0), grip)

    def ready(self, S, seconds=1.5):
        """Joint-space to gripper-down high over the zone: home has the gripper pointing forward, and asking for
        "down" straight from there flips the wrist in one step and sweeps the fingers through the objects."""
        (x0, x1), (y0, y1) = S.ZONE
        pos = ((x0 + x1) / 2, (y0 + y1) / 2, S.T + 0.004 + FINGERS_BELOW_GRASP + READY_HEIGHT)
        q, err = self.ik.solve_any(self.sim.data.qpos, [self.sim.target, self.sim.home], pos, 0.0)
        if err > 0.005:
            raise Abort("ready pose out of reach")
        self.joint_move(q, seconds, self.sim.grip)

    def home(self, seconds=1.5, hold=0.5):
        self.joint_move(self.sim.home, seconds)
        for _ in range(int(hold / CONTROL_DT)):
            self.tick(self.sim.home, 0.0)


def grasp_plan(model, data, obj, S):
    """(gripper yaw, width across the jaws, object half-height) for a top-down grasp, or Abort."""
    b = model.body(obj["name"]).id
    R = data.xmat[b].reshape(3, 3)
    size = obj["size"]
    lo, hi = S.grasp_yaw_range(*data.xpos[b][:2])
    if obj["shape"] == "box":
        yaw_b = math.atan2(R[1, 0], R[0, 0])
        options = []
        for k in range(-2, 3):        # any of the box's four sides faces the jaws
            g = math.atan2(math.sin(yaw_b + k * math.pi / 2), math.cos(yaw_b + k * math.pi / 2))
            if lo - 0.02 <= g <= hi + 0.02:
                options.append((abs(g), g, 2 * (size[1] if k % 2 == 0 else size[0])))
        if not options:
            raise Abort("box turned where the gripper can't follow")
        _, yaw, width = min(options)
        return yaw, width, size[2]
    if obj["shape"] == "cylinder" and obj["lying"]:
        ax = R[:, 2]
        a = math.atan2(ax[1], ax[0])
        yaw = math.atan2(math.sin(a), math.cos(a))
        yaw = yaw - math.pi if yaw > math.pi / 2 else yaw + math.pi if yaw < -math.pi / 2 else yaw
        if not lo - 0.02 <= yaw <= hi + 0.02:
            raise Abort("lying cylinder's axis turned where the gripper can't follow")
        return yaw, 2 * size[0], size[0]
    if obj["shape"] == "cylinder":
        return 0.0, 2 * size[0], size[1]
    return 0.0, 2 * size[0], size[0]


def release_yaw(demo: Demo, yaw: float, off, bin_xy, release_z: float) -> float:
    """The grasp yaw if the arm reaches the release poses with it, else the nearest of a few that it does."""
    d = demo.sim.data
    for cand in sorted({yaw, 0.0, 0.25, -0.25, 0.45}, key=lambda c: abs(c - yaw)):
        c, s_ = math.cos(cand - yaw), math.sin(cand - yaw)
        o = (c * off[0] - s_ * off[1], s_ * off[0] + c * off[1])
        pos = (bin_xy[0] - o[0], bin_xy[1] - o[1])
        if all(demo.ik.solve_any(d.qpos, [demo.sim.target, demo.sim.home], (*pos, z), cand, restarts=2)[1] < 0.002
               for z in (release_z, release_z + 0.05)):
            return cand
    raise Abort("no reachable release pose over the bin")


def run_episode(demo: Demo, seed: int, S) -> dict:
    sim = demo.sim
    m, d = sim.model, sim.data
    sim.reset(seed)
    demo.steps = 0
    for _ in range(10):
        demo.tick(sim.target, 0.0)
    objs = S.episode_objects(m, d)
    demo.ready(S)
    in_bin = {}
    for k, obj in enumerate(objs):
        yaw, width, half_h = grasp_plan(m, d, obj, S)
        pos = d.xpos[m.body(obj["name"]).id].copy()
        grasp_z = S.T + 0.004 + FINGERS_BELOW_GRASP
        demo.move((pos[0], pos[1], grasp_z + 0.10), yaw, SPEED["travel"])
        demo.grip(float(np.clip((OPEN_GAP - width - 0.03) / 0.1, 0, 1)), 0.3)      # half-close: spare neighbours
        demo.move((pos[0], pos[1], grasp_z), yaw, SPEED["descend"])
        demo.grip(1.0, 0.6)
        demo.move((pos[0], pos[1], grasp_z + 0.12), yaw, SPEED["lift"])
        p_now, _ = demo.ik.fk(d.qpos, sim.target)
        o_now = d.xpos[m.body(obj["name"]).id]
        under = p_now[2] - (o_now[2] - half_h)                 # grasp point above the object's bottom
        off = o_now[:2] - p_now[:2]                            # object centre relative to the grasp point
        bx, by = obj["bin"]
        twin = sum(o["shape"] == obj["shape"] for o in objs) > 1
        if twin:
            by += 0.025 if in_bin.get(obj["shape"], 0) else -0.025
        in_bin[obj["shape"]] = in_bin.get(obj["shape"], 0) + 1
        release_z = S.T + S.BIN_T + S.BIN_WALL_H + 0.012 + under
        # How it lands in the bin doesn't matter: release at a yaw the arm reaches over this bin. The object
        # turns with the gripper, so its offset from the grasp point turns too.
        r_yaw = release_yaw(demo, yaw, off, (bx, by), release_z)
        c, s_ = math.cos(r_yaw - yaw), math.sin(r_yaw - yaw)
        off = np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]])
        demo.move((bx - off[0], by - off[1], release_z + 0.05), r_yaw, SPEED["carry"])
        demo.move((bx - off[0], by - off[1], release_z), r_yaw, SPEED["lower"])
        demo.grip(float(np.clip((OPEN_GAP - width - 0.02) / 0.1, 0, 1)), 0.4)
        demo.move((bx - off[0], by - off[1], release_z + 0.08), r_yaw, SPEED["travel"])
        for _ in range(150):                 # let it land and settle: the step latches once it rests in the bin
            if sim.step_info()[0] > k:
                break
            demo.tick(sim.target, sim.grip)
        if sim.step_info()[0] <= k:
            raise Abort(f"step {k + 1} ({obj['colour']} {obj['shape']}) not done")
    demo.ready(S, seconds=1.0)
    demo.home()
    return S.episode_status(m, d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=10, help="successful episodes to keep")
    parser.add_argument("--max_attempts", type=int, default=None, help="give up after this many (default 4 x episodes)")
    parser.add_argument("--seed", type=int, default=0, help="first scene seed; each attempt uses the next")
    add_record_args(parser, task_description="tidy_table scripted demonstration")
    parser.set_defaults(cameras="top,wrist_left")
    args = parser.parse_args()
    args.record = True
    schema = load_record_schema(parser, args)
    cameras = {n: (int(s["height"]), int(s["width"])) for n, s in schema.images.items()}
    sim = TidySim(cameras)
    from humanoid_mujoco_scenes.tidy_table import scene as S

    extra = {"leader_angles": LEADER_SERVOS, "leader_counts": LEADER_SERVOS, "subtask_index": ["subtask_index"]}
    if sim.condition_names is not None:
        extra["observation.environment_state"] = sim.condition_names
    args.num_episodes = None
    recorder, every = make_sim_recorder(args, schema, device="cpu", sim_dt=CONTROL_DT, extra_features=extra)
    if every != RECORD_EVERY:
        raise SystemExit(f"schema fps gives a frame every {every} control steps, tidy_sim expects {RECORD_EVERY}")
    demo = Demo(sim, recorder)
    kept, attempts = 0, 0
    max_attempts = args.max_attempts or 4 * args.episodes
    t0 = time.monotonic()
    try:
        while kept < args.episodes and attempts < max_attempts:
            seed = args.seed + attempts
            attempts += 1
            try:
                status = run_episode(demo, seed, S)
                ok, why = status["success"], ("" if status["success"] else
                                              f"dropped {status['dropped']} toppled {status['toppled']} early {status['early']}")
            except Abort as exc:
                ok, why = False, str(exc)
            if ok:
                recorder.save_episode()
                kept += 1
            else:
                recorder.cancel_recording()
            print(f"[DEMO] seed {seed}: {'kept' if ok else 'discarded'} {why} | kept {kept}/{args.episodes}, "
                  f"{attempts} attempts, {time.monotonic() - t0:.0f} s", flush=True)
    finally:
        recorder.finalize()
        sim.close()
    print(f"[DEMO] {kept} episodes in {recorder.dataset_root} ({kept}/{attempts} attempts succeeded)")


if __name__ == "__main__":
    main()
