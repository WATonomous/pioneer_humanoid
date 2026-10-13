"""RealSense D455 source: left IR + depth as Frames, live or from a recording.

Why left IR (not color): depth is computed in the left IR camera's frame, so depth
pixel (u, v) is IR pixel (u, v). No alignment step, no color/depth timing mismatch,
and the IR cameras are global shutter with a wider field of view.

Emitter modes (the IR dot projector):
  off     clean IR for tracking, but depth is sparse on blank surfaces
  on      dense depth, but the dots move with the camera and ORB tracks them,
          so VO thinks the camera is still. Only for depth-only use.
  on_off  the emitter alternates every frame. Emitter-off frames give the tracking
          image, the neighbouring emitter-on frame gives the depth. Run at 60 fps
          to get ~30 tracking frames a second. Each frame's metadata says whether
          the emitter was on, so pairing doesn't rely on frame order.

In on_off mode the depth is from the previous frame (~17 ms earlier at 60 fps), so
under fast motion it is slightly misregistered with the image. That's the trade for
dense depth plus a dot-free tracking image.

Recordings are ROS 2 bags (.db3): librealsense 2.58 can't write the old ROS 1 .bag.
IMU (gyro + accel) is recorded for later VIO, but not yielded yet.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from slam_core.types import Frame

EMITTER_MODES = ("off", "on", "on_off")


# Camera settings. Keep one copy in config/d455.yaml so live runs and recordings match.
@dataclass
class RealSenseConfig:
    width: int = 848
    height: int = 480
    fps: int = 60
    emitter: str = "on_off"             # off | on | on_off
    laser_power: Optional[int] = None   # emitter power in mW, None = the camera's default
    exposure_us: Optional[int] = None   # None = auto exposure. Short manual exposure = less motion blur
    gain: Optional[int] = None          # only used with exposure_us
    imu: bool = True                    # record gyro + accel when recording
    settle_frames: int = 30             # live only: frames to drop while auto exposure settles

    def __post_init__(self):
        if self.emitter not in EMITTER_MODES:
            raise ValueError(f"emitter must be one of {EMITTER_MODES}, got {self.emitter!r}")

    # Load from a yaml file whose keys are the fields above (missing keys keep defaults).
    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as fh:
            return cls(**(yaml.safe_load(fh) or {}))


class RealSense:
    # play: path to a recording (.db3) to read instead of the live camera.
    # record: path (.db3) to also record every raw stream (both emitter states, + IMU) to.
    # reset: hardware-reset the camera before opening it, to clear a stuck state.
    # Only safe here, before this process holds any handle to the camera: resetting
    # while it's open can deadlock the Linux uvcvideo driver (unkillable process
    # until the camera is unplugged).
    def __init__(self, config=RealSenseConfig(), play=None, record=None, reset=False):
        import pyrealsense2 as rs
        self.rs, self.config, self.live = rs, config, play is None
        c = config
        if reset and not play:
            self._hardware_reset()

        self.pipe = rs.pipeline()
        cfg = rs.config()
        if play:
            cfg.enable_device_from_file(play, repeat_playback=False)
        cfg.enable_stream(rs.stream.infrared, 1, c.width, c.height, rs.format.y8, c.fps)
        cfg.enable_stream(rs.stream.depth, c.width, c.height, rs.format.z16, c.fps)
        if record and c.imu:
            cfg.enable_stream(rs.stream.accel)
            cfg.enable_stream(rs.stream.gyro)
        if record:
            cfg.enable_record_to_file(record)
        if not play:
            # Apply settings before streaming starts. Changing them mid-stream fails
            # intermittently on the D455 (firmware "hwmon" errors, "permission denied").
            try:
                self._configure(cfg.resolve(rs.pipeline_wrapper(self.pipe)).get_device().first_depth_sensor())
            except RuntimeError as e:
                raise RuntimeError(f"camera rejected settings ({e}). It may be stuck: rerun with "
                                   f"--reset (or reset=True), or unplug and replug it") from e
        profile = self.pipe.start(cfg)

        dev = profile.get_device()
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()   # raw uint16 -> meters
        if play:
            # process every frame as fast as we can instead of in real time
            dev.as_playback().set_real_time(False)

        intr = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist = np.array(intr.coeffs[:5], dtype=np.float64)

        # stats, for checking the setup
        self.n_framesets = 0
        self.n_unpaired = 0     # emitter-off frames dropped because no recent emitter-on depth
        self.emitter_ok = None  # on_off: False if emitter-on frames aren't denser (laser not firing)

    # Power-cycle the camera over USB and wait for it to come back (a few seconds).
    def _hardware_reset(self, timeout=15.0):
        import time
        rs = self.rs
        devs = rs.context().query_devices()
        if len(devs) == 0:
            raise RuntimeError("no RealSense connected")
        devs[0].hardware_reset()
        time.sleep(2)   # let it drop off the bus first
        t0 = time.time()
        while time.time() - t0 < timeout:
            if len(rs.context().query_devices()) > 0:
                time.sleep(1)   # enumerated, give the firmware a moment
                return
            time.sleep(0.5)
        raise RuntimeError("camera didn't come back after a hardware reset")

    # Apply emitter / exposure settings to the live camera, and check they took.
    def _configure(self, sensor):
        rs, c = self.rs, self.config
        if c.emitter == "on_off" and not sensor.supports(rs.option.emitter_on_off):
            raise RuntimeError("this camera/firmware doesn't support emitter_on_off")
        # Order matters: the firmware rejects exposure/gain changes while on_off is
        # active, so turn on_off off first, set exposure, then set the emitter mode.
        want = []
        if sensor.supports(rs.option.emitter_on_off):
            want.append((rs.option.emitter_on_off, 0))
        if c.exposure_us:
            want += [(rs.option.enable_auto_exposure, 0), (rs.option.exposure, c.exposure_us)]
            if c.gain:   # gain is only settable with auto exposure off
                want.append((rs.option.gain, c.gain))
        else:
            want.append((rs.option.enable_auto_exposure, 1))
        want.append((rs.option.emitter_enabled, 0 if c.emitter == "off" else 1))
        if c.emitter != "off":
            # set power explicitly: the emitter can be "enabled" with its power left at 0
            power = c.laser_power if c.laser_power is not None else sensor.get_option_range(rs.option.laser_power).default
            want.append((rs.option.laser_power, power))
        if c.emitter == "on_off":
            want.append((rs.option.emitter_on_off, 1))
        for opt, val in want:
            self._set(sensor, opt, val)

    # Set one option, retrying a few times: right after another process releases the
    # camera, the firmware sometimes rejects the first commands.
    def _set(self, sensor, opt, val, tries=5):
        import time
        for i in range(tries):
            try:
                sensor.set_option(opt, val)
                if abs(sensor.get_option(opt) - val) < 1e-3:
                    return
            except RuntimeError:
                if i == tries - 1:
                    raise
            time.sleep(0.1)
        raise RuntimeError(f"camera didn't accept {opt} = {val} (reads back {sensor.get_option(opt)})")

    # True if the emitter was on for this frame, from its metadata. None if unknown.
    def _emitter_on(self, f):
        md = self.rs.frame_metadata_value
        for key in (md.frame_emitter_mode, md.frame_laser_power_mode):
            if f.supports_frame_metadata(key):
                return f.get_frame_metadata(key) > 0
        return None

    # on_off sanity check: with the laser really firing, emitter-on frames have clearly
    # more depth than emitter-off ones. Warn once if they don't (laser not firing, or
    # the scene is out of the projector's range).
    def _check_emitter(self, fill_on, fill_off):
        on, off = np.mean(fill_on), np.mean(fill_off)
        self.emitter_ok = on > off + 0.10
        if not self.emitter_ok:
            print(f"WARNING: emitter-on frames have {100 * on:.0f}% depth vs {100 * off:.0f}% for "
                  f"emitter-off: the laser doesn't seem to be firing (or nothing is within ~4 m). "
                  f"If the image is also black, the camera is stuck: rerun with --reset")

    # Yield Frames until the recording ends or the caller stops iterating.
    def frames(self):
        c = self.config
        on_off = c.emitter == "on_off"
        max_pair_dt = 1.5 / c.fps   # emitter-on depth must be from the frame just before
        last_on = None              # (stamp, depth) of the latest emitter-on frame
        fill_on, fill_off = [], []  # depth fill per emitter state, for _check_emitter
        seq = 0
        try:
            if self.live:
                for _ in range(c.settle_frames):
                    self.pipe.wait_for_frames()
            while True:
                try:
                    fs = self.pipe.wait_for_frames(5000)
                except RuntimeError:
                    return   # end of the recording, or the camera stopped delivering
                ir, d = fs.get_infrared_frame(1), fs.get_depth_frame()
                if not ir or not d:
                    continue   # IMU-only frameset
                self.n_framesets += 1
                stamp = ir.get_timestamp() / 1000.0
                depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale

                if on_off:
                    on = self._emitter_on(ir)
                    if on is None:
                        raise RuntimeError("no emitter metadata on frames, can't run on_off mode "
                                           "(needs a librealsense build with metadata support)")
                    if self.emitter_ok is None:
                        (fill_on if on else fill_off).append(np.mean(depth > 0))
                        if len(fill_on) >= 30 and len(fill_off) >= 30:
                            self._check_emitter(fill_on, fill_off)
                    if on:
                        last_on = (stamp, depth)
                        continue
                    if last_on is None or stamp - last_on[0] > max_pair_dt:
                        self.n_unpaired += 1
                        continue
                    depth = last_on[1]

                yield Frame(stamp=stamp, image=np.asanyarray(ir.get_data()).copy(), depth=depth,
                            K=self.K, dist=self.dist, seq=seq)
                seq += 1
        finally:
            self.pipe.stop()


# Quick check of the camera setup: python -m slam_core.io.realsense [--emitter on_off] [--seconds 5]
if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="yaml file (defaults: RealSenseConfig)")
    ap.add_argument("--emitter", choices=EMITTER_MODES)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--play", help="read a recording (.db3) instead of the live camera")
    ap.add_argument("--record", help="also record to this file (must end in .db3)")
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--view", action="store_true",
                    help="show each Frame (IR | depth) as VO gets it, until you press q")
    ap.add_argument("--exposure", type=int, help="manual exposure in microseconds, overrides the config")
    ap.add_argument("--reset", action="store_true", help="hardware-reset the camera first (if it's stuck)")
    args = ap.parse_args()

    cfg = RealSenseConfig.from_yaml(args.config) if args.config else RealSenseConfig()
    if args.emitter:
        cfg.emitter = args.emitter
    if args.fps:
        cfg.fps = args.fps
    if args.exposure:
        cfg.exposure_us = args.exposure
    if args.record:
        import os
        if not args.record.endswith(".db3"):
            raise SystemExit("--record path must end in .db3")
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    cam = RealSense(cfg, play=args.play, record=args.record, reset=args.reset)
    print(f"K fx={cam.K[0, 0]:.1f} fy={cam.K[1, 1]:.1f} cx={cam.K[0, 2]:.1f} cy={cam.K[1, 2]:.1f} "
          f"dist={np.round(cam.dist, 4)} depth_scale={cam.depth_scale}")

    # rates from camera timestamps, so they show what the camera delivered (not our loop)
    stamps, fill, bright = [], [], []
    for f in cam.frames():
        stamps.append(f.stamp)
        fill.append(np.mean(f.depth > 0))
        bright.append(f.image.mean())
        if args.view:
            import cv2
            # depth 0-4 m colorized, black = no depth
            dvis = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(f.depth, 0, 4), alpha=255 / 4), cv2.COLORMAP_TURBO)
            dvis[f.depth == 0] = 0
            vis = np.hstack([cv2.cvtColor(f.image, cv2.COLOR_GRAY2BGR), dvis])
            cv2.putText(vis, f"t={f.stamp - stamps[0]:5.1f}s  depth fill {100 * fill[-1]:.0f}%  emitter={cfg.emitter}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("realsense: IR | depth (0-4 m)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif f.stamp - stamps[0] > args.seconds:
            break
    span = stamps[-1] - stamps[0]
    print(f"emitter={cfg.emitter} {cfg.fps} fps: {len(stamps)} frames over {span:.1f} s "
          f"({(len(stamps) - 1) / span:.1f} tracking frames/s), {cam.n_unpaired} unpaired, "
          f"depth fill {100 * np.mean(fill):.0f}%, image brightness {np.mean(bright):.0f}/255")
