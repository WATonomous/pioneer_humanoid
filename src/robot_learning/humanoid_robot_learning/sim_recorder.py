"""GPU-buffered async LeRobot dataset recorder for Isaac Sim teleop."""
from __future__ import annotations

import queue
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from humanoid_robot_learning.episode_keys import EpisodeFlags, EpisodeKeyboard


def _encode_video_frames_subprocess(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    overwrite: bool = False,
    **_ignored: Any,
) -> None:
    """Drop-in replacement for lerobot's encode_video_frames using the ffmpeg CLI.

    lerobot encodes videos in-process through PyAV/SVT-AV1, which leaks
    ~0.6 GB of native memory per encoded episode when running inside Isaac
    Sim (measured; the leak eventually freezes the host). Encoding in a
    short-lived ffmpeg subprocess produces identical output (same codec and
    parameters) while the leaked memory dies with the child process.
    """
    video_path = Path(video_path)
    if video_path.exists() and not overwrite:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(Path(imgs_dir) / "frame-%06d.png"),
        "-c:v", vcodec,
        "-pix_fmt", pix_fmt,
    ]
    if vcodec == "libsvtav1":
        cmd += ["-preset", "12"]
    if g is not None:
        cmd += ["-g", str(g)]
    if crf is not None:
        cmd += ["-crf", str(crf)]
    cmd.append(str(video_path))
    subprocess.run(cmd, check=True, capture_output=True)


def _install_subprocess_video_encoder() -> None:
    """Route lerobot's video encoding through the ffmpeg CLI (idempotent).

    lerobot offers no encoder hook, so the module-level symbol is replaced.
    Covers both the sequential path and the per-camera worker processes
    (which resolve the same module attribute after fork).
    """
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    lerobot_dataset.encode_video_frames = _encode_video_frames_subprocess


class SimLeRobotRecorder:
    """Buffer frames in GPU tensors, flush to a LeRobot dataset asynchronously.

    Episodes travel through a fixed pool of pinned CPU slots (constant
    memory, at most _NUM_CPU_SLOTS episodes in flight).

    Designed for Isaac Sim where observations are already on-device — batching
    the PCIe transfer to one copy-per-episode avoids per-frame overhead.

    Usage::

        recorder = SimLeRobotRecorder(...)
        recorder.init_dataset()
        # inside sim loop:
        recorder.push_frame_to_buffer(action, obs, visual_buffers)
        # on episode end:
        recorder.save_episode()   # enqueues async save, clears buffers
        # on cancel:
        recorder.cancel_recording()
        # after all episodes:
        recorder.finalize()
    """

    def __init__(
        self,
        task_name: str,
        repo_id: str,
        dataset_root: str | Path,
        fps: int,
        device: str,
        joint_names: list[str],
        cameras: dict[str, dict[str, int]],
        save_mp4: bool = False,
        depth: bool = False,
        instance_id_seg: bool = False,
        num_episodes: int | None = None,
        buffer_capacity_s: float = 120.0,
        robot_type: str = "so101_follower",
        extra_features: dict[str, list[str]] | None = None,
        rate_limit: bool = True,
        wall_clock_resample: bool = False,
    ) -> None:
        self.fps = fps
        self.save_mp4 = save_mp4
        self.depth = depth
        self.instance_id_seg = instance_id_seg
        self.device = device
        self.joint_names = list(joint_names)
        self.cameras = cameras
        self.repo_id = repo_id
        self.dataset_root = Path(dataset_root)
        self.task_name = task_name
        self.num_episodes = num_episodes
        self.robot_type = robot_type
        # extra float32 vector features: {feature_name: [component names]}
        self.extra_features = dict(extra_features or {})
        self.num_recorded_episodes = 0

        self._capacity = int(buffer_capacity_s * fps)
        self._current_frame = 0

        self._action_buf: torch.Tensor | None = None
        self._obs_buf: torch.Tensor | None = None
        self._rgb_bufs: dict[str, torch.Tensor] = {}
        self._depth_bufs: dict[str, torch.Tensor] = {}
        self._seg_bufs: dict[str, torch.Tensor] = {}
        self._extra_bufs: dict[str, torch.Tensor] = {}
        self._frame_tasks: list[str | None] = []

        self._episode_queue: queue.Queue = queue.Queue()
        # reusable pinned CPU episode slots; see _allocate_cpu_slots
        self._free_slots: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._processor_thread = threading.Thread(
            target=self._async_processor, daemon=True
        )
        self._processor_thread.start()

        self._flags: EpisodeFlags | None = None
        self._keyboard: EpisodeKeyboard | None = None
        self._last_frame_t: float = 0.0
        self._wall_clock_resample = bool(wall_clock_resample)
        self._record_start_t: float | None = None
        self._next_frame_t: float | None = None
        self._repeated_frames = 0
        # rate_limit=False: caller owns the cadence, tick() pushes every call. Use it when the
        # images only change on some ticks (sim rendering every Nth step) -- a second decimator
        # here can't line up with the render gate, and the extra ticks duplicate images.
        self._frame_period: float = (1.0 / fps) if rate_limit else 0.0

    def start_keyboard(self) -> bool:
        """Create episode flags, attach keyboard listener, and return whether keyboard started."""
        self._flags = EpisodeFlags(start=False)
        self._keyboard = EpisodeKeyboard(self._flags)
        return self._keyboard.start()

    def tick(
        self,
        action: np.ndarray | torch.Tensor,
        state: np.ndarray | torch.Tensor,
        images: dict[str, np.ndarray | torch.Tensor] | Callable[[], dict[str, np.ndarray | torch.Tensor]],
        depth_buffers: dict[str, np.ndarray | torch.Tensor] | None = None,
        instance_id_seg_buffers: dict[str, np.ndarray | torch.Tensor] | None = None,
        task: str | None = None,
        extras: dict[str, np.ndarray | torch.Tensor] | None = None,
    ) -> bool:
        """Handle episode flags and push one frame if the rate allows.

        `task`: this frame's language instruction (e.g. the current step of a multi-step task);
        None records the session's task_name. `extras`: values for `extra_features`.

        `images` may be a plain dict or a zero-arg callable returning one -- pass a
        callable when capturing images is expensive (GPU->CPU reads); it's invoked
        only on frames actually pushed, not on no-op ticks.

        Returns True if an episode was just saved (so callers can trigger a scene reset).
        No-op if start_keyboard() was never called.
        """
        if self._flags is None:
            return False
        flags = self._flags
        if flags.remove:
            self.cancel_recording()
            flags.remove = False
        if flags.undo:
            self.discard_last_saved_episode()
            flags.undo = False
        if flags.start:
            now = time.monotonic()
            if self._wall_clock_resample:
                if self._record_start_t is None:
                    self._record_start_t = now
                    self._next_frame_t = now
                assert self._next_frame_t is not None
                if now + 1e-9 >= self._next_frame_t:
                    due = int((now - self._next_frame_t + 1e-9) / (1.0 / self.fps)) + 1
                    # If Isaac missed output ticks, hold the preceding synchronized sample for
                    # those timestamps, then append the newest sample.  The resulting fixed-rate
                    # trajectory has the same duration as wall time instead of playing fast.
                    repeated = 0
                    for _ in range(max(0, due - 1)):
                        if self._repeat_last_frame():
                            repeated += 1
                    self._repeated_frames += repeated
                    resolved_images = images() if callable(images) else images
                    self.push_frame_to_buffer(
                        action, state, resolved_images, depth_buffers, instance_id_seg_buffers,
                        extras=extras, task=task,
                    )
                    self._next_frame_t += due / self.fps
            elif now - self._last_frame_t >= self._frame_period:
                self._last_frame_t = now
                resolved_images = images() if callable(images) else images
                self.push_frame_to_buffer(
                    action, state, resolved_images, depth_buffers, instance_id_seg_buffers, extras=extras, task=task
                )
        if flags.success:
            # N arrives between loop ticks. Capture any wall-clock timestamp now due before
            # closing the take, rather than shortening it by the final render interval.
            saved = self.save_episode()
            flags.success = False
            flags.start = False
            return saved
        return False

    @property
    def num_buffered_frames(self) -> int:
        """Frames in the current (unsaved) episode."""
        return self._current_frame

    @property
    def is_recording(self) -> bool:
        """Whether S has armed frame capture for the current episode."""
        return bool(self._flags is not None and self._flags.start)

    @property
    def num_pending_episodes(self) -> int:
        """Episodes queued or currently being written by the background worker."""
        return int(self._episode_queue.unfinished_tasks)

    @property
    def capture_elapsed_s(self) -> float:
        """Wall-clock duration of the active take."""
        return 0.0 if self._record_start_t is None else max(0.0, time.monotonic() - self._record_start_t)

    @property
    def num_repeated_frames(self) -> int:
        """Timeline frames repeated because the simulator produced samples below dataset FPS."""
        return self._repeated_frames

    @property
    def is_complete(self) -> bool:
        return (
            self.num_episodes is not None
            and self.num_recorded_episodes >= self.num_episodes
        )

    def _build_features(self) -> dict[str, Any]:
        dim = len(self.joint_names)
        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "fps": self.fps,
                "shape": (dim,),
                "names": self.joint_names,
            },
            "action": {
                "dtype": "float32",
                "fps": self.fps,
                "shape": (dim,),
                "names": self.joint_names,
            },
        }
        for name, spec in self.cameras.items():
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": (spec["height"], spec["width"], 3),
                "names": ["height", "width", "channels"],
            }
        for name, comp_names in self.extra_features.items():
            features[name] = {
                "dtype": "float32",
                "fps": self.fps,
                "shape": (len(comp_names),),
                "names": list(comp_names),
            }
        return features

    _NUM_CPU_SLOTS = 2

    def init_dataset(self) -> None:
        """Create or re-open the LeRobot dataset on disk."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        _install_subprocess_video_encoder()
        if self._free_slots.empty():
            self._allocate_cpu_slots()
        root = self.dataset_root
        if root.exists():
            try:
                self.dataset = LeRobotDataset(self.repo_id, root=root)
                self.num_recorded_episodes = int(self.dataset.num_episodes)
                print(
                    f"[INFO]: Opened existing dataset at {root} "
                    f"({self.num_recorded_episodes} episode(s) already saved)"
                )
                return
            except Exception as exc:
                raise ValueError(
                    f"[ERROR]: Dataset folder exists but cannot be opened: {root}\n"
                    f"  Cause: {type(exc).__name__}: {exc}\n"
                    "  A run that ended before saving an episode leaves a folder LeRobotDataset "
                    "cannot re-open. Delete it (or pass --dataset_root elsewhere) and retry."
                ) from exc

        self.dataset = LeRobotDataset.create(
            self.repo_id,
            fps=self.fps,
            features=self._build_features(),
            root=root,
            robot_type=self.robot_type,
        )
        print(f"[INFO]: Created new dataset at {root}")

    def _allocate_buffers(self) -> None:
        dim = len(self.joint_names)
        cap, dev = self._capacity, self.device
        self._action_buf = torch.zeros((cap, dim), dtype=torch.float32, device=dev)
        self._obs_buf = torch.zeros((cap, dim), dtype=torch.float32, device=dev)
        for name, spec in self.cameras.items():
            h, w = spec["height"], spec["width"]
            self._rgb_bufs[name] = torch.zeros(
                (cap, h, w, 3), dtype=torch.uint8, device=dev
            )
            if self.depth:
                self._depth_bufs[name] = torch.zeros(
                    (cap, h, w, 1), dtype=torch.float32, device=dev
                )
            if self.instance_id_seg:
                self._seg_bufs[name] = torch.zeros(
                    (cap, h, w, 3), dtype=torch.uint8, device=dev
                )
        for name, comp_names in self.extra_features.items():
            self._extra_bufs[name] = torch.zeros(
                (cap, len(comp_names)), dtype=torch.float32, device=dev
            )

    @staticmethod
    def _as_tensor(
        x: np.ndarray | torch.Tensor, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(x)).to(
                device=device, dtype=dtype
            )
        return x.to(device=device, dtype=dtype)

    def push_frame_to_buffer(
        self,
        action: np.ndarray | torch.Tensor,
        observation: np.ndarray | torch.Tensor,
        visual_buffers: dict[str, np.ndarray | torch.Tensor],
        depth_buffers: dict[str, np.ndarray | torch.Tensor] | None = None,
        instance_id_seg_buffers: dict[str, np.ndarray | torch.Tensor] | None = None,
        extras: dict[str, np.ndarray | torch.Tensor] | None = None,
        task: str | None = None,
    ) -> None:
        """Push one timestep of data into the GPU buffers (`task`: this frame's instruction, or None)."""
        if self._current_frame >= self._capacity:
            print(
                f"[WARN]: Buffer full at frame {self._current_frame}, skipping"
            )
            return
        if self._action_buf is None:
            self._allocate_buffers()

        i = self._current_frame
        self._action_buf[i] = self._as_tensor(action, torch.float32, self.device)
        self._obs_buf[i] = self._as_tensor(observation, torch.float32, self.device)

        for name in self.cameras:
            self._rgb_bufs[name][i] = self._as_tensor(
                visual_buffers[name], torch.uint8, self.device
            )
            if self.depth and depth_buffers:
                self._depth_bufs[name][i] = self._as_tensor(
                    depth_buffers[name], torch.float32, self.device
                )
            if self.instance_id_seg and instance_id_seg_buffers:
                self._seg_bufs[name][i] = self._as_tensor(
                    instance_id_seg_buffers[name], torch.uint8, self.device
                )

        for name in self.extra_features:
            if extras is None or name not in extras:
                raise KeyError(f"extra feature '{name}' missing from extras")
            self._extra_bufs[name][i] = self._as_tensor(
                extras[name], torch.float32, self.device
            )

        self._frame_tasks.append(task)
        self._current_frame += 1

    def _allocate_cpu_slots(self) -> None:
        """Preallocate reusable pinned CPU episode slots (once per session).

        Copying episodes into fresh pageable CPU memory every save leaked
        ~0.6 GB of RSS per episode inside Isaac Sim (the freed pages were
        never returned to the OS), eventually freezing the host. Two pinned
        slots, allocated once and reused, keep memory constant and make the
        device-to-host DMA faster. Two slots also bound how many episodes
        can be in flight — save_episode() blocks when both are busy.
        Pinning needs CUDA; without a GPU (MuJoCo on CPU) the slots are plain memory.
        """
        pin = torch.cuda.is_available()
        dim = len(self.joint_names)
        cap = self._capacity
        for _ in range(self._NUM_CPU_SLOTS):
            slot: dict[str, Any] = {
                "total_frames": 0,
                "action": torch.empty((cap, dim), dtype=torch.float32, pin_memory=pin),
                "observation": torch.empty((cap, dim), dtype=torch.float32, pin_memory=pin),
                "rgb": {}, "depth": {}, "seg": {}, "extras": {},
            }
            for name, spec in self.cameras.items():
                h, w = spec["height"], spec["width"]
                slot["rgb"][name] = torch.empty((cap, h, w, 3), dtype=torch.uint8, pin_memory=pin)
                if self.depth:
                    slot["depth"][name] = torch.empty((cap, h, w, 1), dtype=torch.float32, pin_memory=pin)
                if self.instance_id_seg:
                    slot["seg"][name] = torch.empty((cap, h, w, 3), dtype=torch.uint8, pin_memory=pin)
            for name, comp_names in self.extra_features.items():
                slot["extras"][name] = torch.empty(
                    (cap, len(comp_names)), dtype=torch.float32, pin_memory=pin
                )
            self._free_slots.put(slot)

    def save_episode(self) -> bool:
        """Copy the episode into a reusable pinned CPU slot and enqueue it.

        Blocks while both CPU slots are in flight (writer backpressure).
        """
        if self._action_buf is None:
            print("[WARN]: save_episode called with no buffered frames, skipping")
            return False
        elapsed = self.capture_elapsed_s
        repeated = self._repeated_frames
        if self._free_slots.empty():
            print("[INFO]: Waiting for a free episode slot (writer catching up)...")
        slot = self._free_slots.get()

        n = self._current_frame
        slot["total_frames"] = n
        slot["action"][:n].copy_(self._action_buf[:n])
        slot["observation"][:n].copy_(self._obs_buf[:n])
        for name in self.cameras:
            slot["rgb"][name][:n].copy_(self._rgb_bufs[name][:n])
            if self.depth:
                slot["depth"][name][:n].copy_(self._depth_bufs[name][:n])
            if self.instance_id_seg:
                slot["seg"][name][:n].copy_(self._seg_bufs[name][:n])
        for name in self.extra_features:
            slot["extras"][name][:n].copy_(self._extra_bufs[name][:n])
        slot["tasks"] = list(self._frame_tasks[:n])

        self._episode_queue.put(slot)
        self._clear_buffers()
        print(
            f"[INFO]: Episode queued: {n / self.fps:.2f}s dataset timeline from "
            f"{elapsed:.2f}s wall time ({repeated} held frame(s))."
        )
        return True

    def cancel_recording(self) -> None:
        """Discard the current episode buffer without saving."""
        self._clear_buffers()
        print("[INFO]: Recording cancelled.")

    def _repeat_last_frame(self) -> bool:
        """Append a zero-order-held copy of the preceding synchronized sample."""
        if self._current_frame <= 0 or self._current_frame >= self._capacity:
            return False
        i = self._current_frame
        previous = i - 1
        self._action_buf[i].copy_(self._action_buf[previous])
        self._obs_buf[i].copy_(self._obs_buf[previous])
        for name in self.cameras:
            self._rgb_bufs[name][i].copy_(self._rgb_bufs[name][previous])
            if self.depth:
                self._depth_bufs[name][i].copy_(self._depth_bufs[name][previous])
            if self.instance_id_seg:
                self._seg_bufs[name][i].copy_(self._seg_bufs[name][previous])
        for name in self.extra_features:
            self._extra_bufs[name][i].copy_(self._extra_bufs[name][previous])
        self._frame_tasks.append(self._frame_tasks[-1])
        self._current_frame += 1
        return True

    def discard_last_saved_episode(self) -> bool:
        """Remove the most recently saved episode and reopen the dataset in place.

        LeRobot v3 stores multiple episodes in shared parquet/video chunks, so removing a saved
        episode must rebuild those chunks.  The rebuilt dataset is prepared beside the original,
        validated, then atomically swapped into place; the original remains available as a backup
        until the replacement has opened successfully.
        """
        self._episode_queue.join()
        total = int(self.dataset.num_episodes)
        if total <= 0:
            print("[WARN]: D pressed but there are no saved episodes to discard.")
            return False

        root = self.dataset_root
        token = uuid.uuid4().hex[:8]
        replacement = root.parent / f".{root.name}.discard-replacement-{token}"
        backup = root.parent / f".{root.name}.discard-backup-{token}"
        print(f"[INFO]: Removing saved episode {total}; rebuilding dataset metadata/videos...")

        try:
            if total == 1:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                new_dataset = LeRobotDataset.create(
                    self.repo_id,
                    fps=self.fps,
                    features=self._build_features(),
                    root=replacement,
                    robot_type=self.robot_type,
                )
            else:
                from lerobot.datasets.dataset_tools import delete_episodes

                new_dataset = delete_episodes(
                    self.dataset,
                    [total - 1],
                    output_dir=replacement,
                    repo_id=self.repo_id,
                )

            # Release the old reader before the directory swap.  os.replace keeps both rename
            # operations on the same filesystem, so there is no partially-copied live dataset.
            self.dataset = None
            os.replace(root, backup)
            os.replace(replacement, root)

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self.dataset = LeRobotDataset(self.repo_id, root=root) if total > 1 else new_dataset
            if total == 1:
                # LeRobot's empty freshly-created dataset is usable for appending but cannot be
                # reopened until its first replacement episode is saved.
                self.dataset.root = root
                self.dataset.meta.root = root
            self.num_recorded_episodes = total - 1
            shutil.rmtree(backup)
            print(f"[INFO]: Discarded saved episode {total}; {self.num_recorded_episodes} remain.")
            return True
        except Exception:
            if not root.exists() and backup.exists():
                os.replace(backup, root)
            if replacement.exists():
                shutil.rmtree(replacement)
            if self.dataset is None:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                self.dataset = LeRobotDataset(self.repo_id, root=root)
            raise

    def _clear_buffers(self) -> None:
        self._action_buf = None
        self._obs_buf = None
        self._rgb_bufs = {}
        self._depth_bufs = {}
        self._seg_bufs = {}
        self._extra_bufs = {}
        self._frame_tasks = []
        self._current_frame = 0
        self._record_start_t = None
        self._next_frame_t = None
        self._repeated_frames = 0

    def _async_processor(self) -> None:
        while not self._stop_event.is_set():
            try:
                episode = self._episode_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process_episode(episode)
                self.num_recorded_episodes += 1
                print(f"[INFO]: Episode {self.num_recorded_episodes} saved.")
            except Exception as exc:
                print(f"[ERROR]: Episode processing failed: {exc}")
            finally:
                self._free_slots.put(episode)  # recycle the pinned slot
                self._episode_queue.task_done()
                self._trim_native_heap()

    @staticmethod
    def _trim_native_heap() -> None:
        """Return freed glibc heap pages to the OS.

        Each episode moves ~0.7 GB of frames through this thread; inside
        Isaac Sim (many threads, many malloc arenas) glibc retains the freed
        pages indefinitely, growing RSS by ~0.6 GB per saved episode until
        the host runs out of memory. malloc_trim(0) after each episode
        returns them (measured: flat RSS with, linear growth without).
        """
        import ctypes

        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass  # non-glibc platform: nothing to trim

    def _process_episode(self, episode: dict[str, Any]) -> None:
        """Write one episode (a pinned CPU slot) to the LeRobot dataset.

        Tensor rows handed to add_frame are views into the slot; lerobot
        copies/encodes everything before this method returns, after which
        the slot is recycled by the caller.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        n = episode["total_frames"]
        tasks = episode.get("tasks") or [None] * n
        for i in tqdm(range(n), desc="Processing frames", unit="frame"):
            frame: dict[str, Any] = {
                "action": episode["action"][i],
                "observation.state": episode["observation"][i],
                "task": tasks[i] or self.task_name,
            }
            for name in self.cameras:
                frame[f"observation.images.{name}"] = episode["rgb"][name][i]
            for name in self.extra_features:
                frame[name] = episode["extras"][name][i]
            self.dataset.add_frame(frame)

        if self.save_mp4:
            for name in self.cameras:
                self._save_rgb_video(episode["rgb"][name][:n].numpy(), name)
                if self.depth and episode["depth"]:
                    self._save_depth_video(episode["depth"][name][:n].numpy(), name)
                if self.instance_id_seg and episode["seg"]:
                    self._save_seg_video(episode["seg"][name][:n].numpy(), name)

        self.dataset.save_episode()
        self.dataset.finalize()
        self.dataset = LeRobotDataset(self.repo_id, root=self.dataset_root)

    def finalize(self) -> None:
        """Block until all queued episodes are saved, then stop the worker thread."""
        if self._keyboard is not None:
            self._keyboard.stop()
            self._keyboard = None
        self._episode_queue.join()
        self._stop_event.set()
        self._processor_thread.join(timeout=5.0)

    # --- video helpers ---

    def _save_video(
        self, frames_rgb: np.ndarray, camera_name: str, data_type: str
    ) -> None:
        ep = self.num_recorded_episodes + 1
        out = (
            self.dataset_root
            / "mp4"
            / self.repo_id.split("/")[-1]
            / camera_name
            / f"{camera_name}_{data_type}_{ep:03d}.mp4"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        _, h, w, _ = frames_rgb.shape
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p", str(out),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = proc.communicate(
                input=b"".join(f.tobytes() for f in frames_rgb)
            )
            if proc.returncode != 0:
                print(f"[ERROR]: ffmpeg failed: {stderr.decode()}")
            else:
                print(f"[INFO]: Saved {data_type} video → {out}")
        except FileNotFoundError:
            print("[ERROR]: ffmpeg not found")

    def _save_depth_video(self, frames: np.ndarray, camera_name: str) -> None:
        if frames.ndim == 4 and frames.shape[-1] == 1:
            frames = frames.squeeze(-1)
        valid = np.isfinite(frames)
        lo = np.percentile(frames[valid], 1) if valid.any() else 0.0
        hi = np.percentile(frames[valid], 99) if valid.any() else 1.0
        if hi - lo < 1e-6:
            hi = lo + 1.0
        norm = np.clip((frames - lo) / (hi - lo), 0, 1)
        rgb = np.stack([(norm * 255).astype(np.uint8)] * 3, axis=-1)
        self._save_video(rgb, camera_name, "depth")

    def _save_rgb_video(self, frames: np.ndarray, camera_name: str) -> None:
        self._save_video(frames.astype(np.uint8), camera_name, "rgb")

    def _save_seg_video(self, frames: np.ndarray, camera_name: str) -> None:
        self._save_video(frames.astype(np.uint8), camera_name, "instance_id_segmentation")

    def __del__(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_processor_thread"):
            self._processor_thread.join(timeout=0.1)
