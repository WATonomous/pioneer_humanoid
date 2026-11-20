"""Collect demos with the handheld gripper: raw takes, host-timestamped, into datasets/umi_raw/<session>/.

    python -m umi_gripper.collect --task "put the cup on the plate"
    python -m umi_gripper.collect --task "..." --headset quest3 --width trigger --no-camera

On the mounted controller: primary (B / Y) starts a take and saves it, secondary (A / X) discards it. Ctrl+C ends
the session (an unsaved take is discarded). Each saved take gets a quick quality check here; the full one runs in
postprocess.to_lerobot.
"""
from __future__ import annotations

import argparse
import datetime as dt
import signal
import time
from pathlib import Path

from umi_gripper import config as umi_config
from umi_gripper import quest_tracker, width as umi_width
from umi_gripper.take import TakeWriter, next_take_path

TICK_S = 0.02
STALE_POSE_S = 0.2


class Session:
    """Wires the streams into takes. Streams are injected so tests drive it without hardware."""

    def __init__(self, cfg: dict, task: str, session_dir: Path, tracker, width, camera=None, log=print):
        self.cfg, self.task, self.session_dir = cfg, task, Path(session_dir)
        self.tracker, self.width, self.camera, self.log = tracker, width, camera, log
        self.take: TakeWriter | None = None
        self.saved: list[Path] = []

    def meta(self) -> dict:
        return {"task": self.task, "headset": self.cfg["headset"], "controller": self.cfg["controller"],
                "width_source": self.width.source, "camera": self.camera is not None, "config": self.cfg}

    def start_take(self) -> None:
        path = next_take_path(self.session_dir)
        self.take = TakeWriter(path, self.meta(), int(self.cfg["camera"].get("jpeg_quality", 92)))
        self.width.drain()   # only samples from now on
        if self.camera is not None:
            self.camera.set_recording(True)
        self.log(f"[UMI] recording {path.name}")

    def stop_take(self, save: bool) -> Path | None:
        take, self.take = self.take, None
        if self.camera is not None:
            self.camera.set_recording(False)
        if take is None:
            return None
        if not save:
            take.discard()
            self.log(f"[UMI] discarded {take.path.name}")
            return None
        path = take.save()
        self.saved.append(path)
        tracked = sum(p.tracked for p in take.poses)
        self.log(f"[UMI] saved {path.name}: {take.duration_s:.1f} s, {len(take.poses)} poses "
                 f"({tracked / max(1, len(take.poses)):.0%} tracked), {len(take.camera_s)} frames, "
                 f"{len(take.width)} width samples")
        return path

    def check_streams(self) -> None:
        for stream in (self.width, self.camera):
            if stream is not None and getattr(stream, "error", None) is not None:
                raise RuntimeError(f"{type(stream).__name__} failed: {stream.error}") from stream.error

    def tick(self, now: float) -> None:
        self.check_streams()
        poses = self.tracker.drain()
        if isinstance(self.width, umi_width.TriggerWidth):
            self.width.feed(poses)
        for event in self.tracker.poll_events():
            if event.name == "primary":
                if self.take is None:
                    self.start_take()
                else:
                    self.stop_take(save=True)
            elif event.name == "secondary" and self.take is not None:
                self.stop_take(save=False)
        widths = self.width.drain()
        if self.take is not None:
            for pose in poses:
                self.take.add_pose(pose)
            for stamp, closure, raw in widths:
                self.take.add_width(stamp, closure, raw)
            if self.camera is not None:
                for stamp, rgb in self.camera.drain():
                    self.take.add_frame(stamp, rgb)
        self.tracker.set_status(self.status(now))

    def status(self, now: float) -> str:
        latest = self.tracker.latest
        if latest is None or now - latest.host_s > STALE_POSE_S:
            tracking = "no poses: enter VR on the headset page"
        elif not latest.tracked:
            tracking = f"{self.cfg['controller']} controller NOT tracked"
        else:
            tracking = "tracking"
        width = "" if self.width.latest is None else f", jaw {self.width.latest:.2f}"
        if self.take is None:
            return f"Ready ({len(self.saved)} saved): primary button starts a take. {tracking}{width}"
        return f"REC {self.take.duration_s:5.1f} s  {tracking}{width}. Primary saves, secondary discards."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, help="task description, stored with every take")
    parser.add_argument("--config", type=Path, default=umi_config.DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=umi_config.DEFAULT_CALIBRATION)
    parser.add_argument("--headset", choices=umi_config.HEADSETS, default=None, help="override the config's headset")
    parser.add_argument("--controller", choices=umi_config.SIDES, default=None, help="mounted controller")
    parser.add_argument("--width", choices=("encoder", "trigger"), default=None, help="override width.source")
    parser.add_argument("--camera", default=None, help="override camera.device")
    parser.add_argument("--no-camera", action="store_true", help="poses and width only (tracking tests)")
    parser.add_argument("--session", default=None, help="session folder name (default: date_time)")
    parser.add_argument("--raw-root", type=Path, default=None, help="override record.raw_root")
    return parser


def overrides_from(args) -> dict:
    out: dict = {}
    if args.headset:
        out["headset"] = args.headset
    if args.controller:
        out["controller"] = args.controller
    if args.width:
        out["width"] = {"source": args.width}
    if args.camera is not None:
        out["camera"] = {"device": args.camera}
    return out


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _interrupt)   # start_umi.sh stops a background run with TERM (it ignores INT)
    cfg = umi_config.load_config(args.config, args.calibration, overrides_from(args))
    raw_root = args.raw_root or umi_config.resolve_repo_path(cfg["record"]["raw_root"])
    session_dir = Path(raw_root) / (args.session or dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[UMI] {cfg['headset']}, {cfg['controller']} controller, width from {cfg['width']['source']}; "
          f"takes -> {session_dir}")

    tracker = quest_tracker.from_config(cfg).start()
    width = umi_width.from_config(cfg).start()
    camera = None
    if not args.no_camera:
        from umi_gripper.camera import CameraStream
        camera = CameraStream(cfg["camera"]).start()
    session = Session(cfg, args.task, session_dir, tracker, width, camera)
    last_print = 0.0
    try:
        while True:
            now = time.monotonic()
            session.tick(now)
            if now - last_print > 1.0:
                print("\r" + session.status(now)[:110].ljust(110), end="", flush=True)
                last_print = now
            time.sleep(TICK_S)
    except KeyboardInterrupt:
        print()
    finally:
        session.stop_take(save=False)
        for stream in (camera, width):
            if stream is not None:
                stream.stop()
        tracker.stop()
    print(f"[UMI] {len(session.saved)} takes saved in {session_dir}")
    if session.saved:
        print(f"      convert: python -m umi_gripper.postprocess.to_lerobot {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
