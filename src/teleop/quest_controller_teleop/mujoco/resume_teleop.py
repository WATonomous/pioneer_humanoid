"""RECAP corrections: resume a recorded policy rollout from its snapshot and finish the task with the Quest controllers.

    python src/teleop/quest_controller_teleop/mujoco/resume_teleop.py --rollouts datasets/recap/drawer_stow --auto

`--rollouts`: a dataset written by `python -m humanoid_robot_learning.sim.evaluate --record` (LeRobot + recap/ sidecar).
`--episode N --t S` restores rollout N at sim time S. `--episode N` alone, or `--auto` (every failed rollout in turn),
resumes `--before_s` (1.5 s) before the scene's progress counter last went up: just before the policy stalled.

The arms re-anchor at the restored pose, so they never jump. Each gripper keeps its restored command only until its
controller's next tracked packet, after which it follows the trigger: hold the trigger at the closure printed on restore
(e.g. fully for a held block) before restoring. Right B starts and saves a
take. It goes into the rollouts dataset (or --dataset_root) as a correction episode, with a recap/episodes.parquet row:
source correction, parent episode, resume time, success from the scene. Left Y (or 6) restores the same snapshot again
and discards an unsaved take. With --auto, N in the window skips to the next failed rollout, and saving a take moves on
to it. Both arms are driven; the left arm is recorded, as the rollouts are (--record_arms left).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from session import MujocoSession, add_session_args, open_rig, serve   # first: puts the repo's packages on sys.path
from scene_driver import check_controller_args  # noqa: E402
from humanoid_robot_learning.recap import sidecar  # noqa: E402
from humanoid_robot_learning.sim.sim_teleop_record import add_record_args, load_record_schema  # noqa: E402

BEFORE_S = 1.5   # s, how far before the last progress the --auto snapshot is: room to see and redo the stall
SIDES = ("left",)   # recorded arm: the rollouts' (sim evaluation drives the left arm)


@dataclass
class Snapshot:
    """A rollout frame to resume from: its sim state and the rollout's episodes.parquet row."""

    episode: int
    frame: int
    time: float
    state: np.ndarray
    parent: dict


class Snapshots:
    """The snapshots to correct, in order: one (--episode) or every failed rollout (--auto)."""

    def __init__(self, root: Path, episodes: list[int], t: float | None, before_s: float):
        rows = sidecar.read_episodes(root).set_index("episode_index")
        self.root, self.rows, self.episodes, self.t, self.before_s = root, rows, episodes, t, before_s
        self.index, self._current = 0, None

    def current(self) -> Snapshot:
        if self._current is None or self._current.episode != self.episodes[self.index]:
            self._current = self._load(self.episodes[self.index])
        return self._current

    def advance(self) -> bool:
        """Move to the next episode; False when there is none."""
        if self.index + 1 >= len(self.episodes):
            return False
        self.index += 1
        return True

    def _load(self, episode: int) -> Snapshot:
        frames = sidecar.read_frames(self.root, episode)
        t = self.t if self.t is not None else sidecar.last_advance_time(frames) - self.before_s
        i = sidecar.frame_at(frames, t)
        state = np.asarray(frames["state"].iloc[i], np.float64)
        return Snapshot(episode, i, float(frames["time"].iloc[i]), state, self.rows.loc[episode].to_dict())


def failed_rollouts(root: Path) -> list[int]:
    rows = sidecar.read_episodes(root)
    return sorted(int(e) for e in rows[(rows["source"] == "rollout") & ~rows["success"].astype(bool)]["episode_index"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", required=True, help="rollouts dataset (evaluate.py --record DIR)")
    parser.add_argument("--episode", type=int, default=None, help="rollout episode to resume")
    parser.add_argument("--t", type=float, default=None, help="sim time (s) to resume it at (default: --before_s rule)")
    parser.add_argument("--auto", action="store_true", help="every failed rollout in turn; N in the window skips")
    parser.add_argument("--before_s", type=float, default=BEFORE_S,
                        help="resume this long before the progress counter last went up")
    add_session_args(parser, scene_default=None)
    add_record_args(parser, task_description="Quest VR correction")
    parser.set_defaults(record=True)   # always records: a take is a correction episode
    args = parser.parse_args()
    check_controller_args(parser, args)
    if args.auto == (args.episode is not None):
        parser.error("pass --episode N (optionally --t S) or --auto")
    if args.t is not None and args.auto:
        parser.error("--t needs --episode")
    args.rollouts = Path(args.rollouts)
    args.dataset_root = args.dataset_root or str(args.rollouts)
    if args.cameras is None:
        args.cameras = "ego,wrist_left"
    return parser, args, load_record_schema(parser, args, arms=list(SIDES), robot="8dof")


def pick_snapshots(parser, args) -> Snapshots:
    episodes = failed_rollouts(args.rollouts) if args.auto else [args.episode]
    if not episodes:
        parser.error(f"{args.rollouts}: no failed rollouts to correct")
    snapshots = Snapshots(args.rollouts, episodes, args.t, args.before_s)
    missing = [e for e in episodes if e not in snapshots.rows.index]
    if missing:
        parser.error(f"{args.rollouts}: no recap row for episode(s) {missing}")
    scene = snapshots.rows.loc[episodes[0], "scene"]
    if args.scene not in (None, scene):
        parser.error(f"--scene {args.scene}: the rollouts are of {scene}")
    args.scene = scene
    print(f"[RECAP] {len(episodes)} rollout(s) to correct: {episodes}", flush=True)
    return snapshots


def correction_row(args, snap: Snapshot, take: list) -> dict:
    """The sidecar row of a correction of `snap` recorded as `take` (its frames' FrameMeta)."""
    same = Path(args.dataset_root).resolve() == args.rollouts.resolve()
    return dict(source="correction", round=int(snap.parent["round"]), scene=args.scene, policy=snap.parent["policy"],
                seed=int(snap.parent["seed"]), success=take[-1].success, parent_episode=snap.episode,
                parent_root="" if same else str(args.rollouts), resume_s=snap.time)


def run() -> None:
    parser, args, record = parse_args()
    snapshots = pick_snapshots(parser, args)
    sim = MujocoSession(args.scene)
    sim.restore(snapshots.current().state)
    rig = open_rig(args, record, sim, SIDES)
    rig.driver.grippers.update(sim.gripper_commands())   # a fresh driver anchors the arms at the measured pose
    writer = sidecar.SidecarWriter(rig.recorder.dataset_root, rig.recorder.dataset.meta.total_episodes)

    def start_episode(driver) -> str:
        snap = snapshots.current()
        sim.restore(snap.state)
        driver.reset(sim.joint_positions())
        driver.grippers.update(sim.gripper_commands())   # after reset (which opens them): keep a held block held
        return f"Rollout {snap.episode} restored at {snap.time:.1f} s ({grip_text()}). Right B records the correction."

    def grip_text() -> str:
        return f"left gripper {sim.gripper_commands()['left']:.0%} closed"

    def on_saved(take) -> None:
        rig.recording.saved()
        if take:
            snap = snapshots.current()
            episode = writer.write(take, **correction_row(args, snap, take))
            print(f"[RECAP] correction episode {episode} of rollout {snap.episode}: "
                  f"{'success' if take[-1].success else 'not finished'}", flush=True)
        next_snapshot() if args.auto else rig.reset_key.set()

    def next_snapshot() -> None:
        if snapshots.advance():
            rig.reset_key.set()
        else:
            print("[RECAP] No more failed rollouts. Close the window to finish.", flush=True)

    rig.window.on_saved = on_saved
    if args.auto:
        rig.keys["n"] = next_snapshot
    snap = snapshots.current()
    print(f"[RECAP] Rollout {snap.episode} restored at {snap.time:.1f} s ({grip_text()}). Right B records the correction; left Y "
          f"restores again{'; N skips to the next' if args.auto else ''}.", flush=True)
    serve(args, rig, sim, start_episode)


if __name__ == "__main__":
    run()
