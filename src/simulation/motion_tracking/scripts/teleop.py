"""Keyboard teleop: press a key, that move comes out.

Keys are read in this terminal (keep it focused); with --viewer viser the
browser also gets one button per move. Moves queue (up to 4) and play back to
back; with nothing queued the robot holds the guard.

  i / k   step forward / back       f / g   slip left / right
  j / l   step left / right         r / t   bob-and-weave left / right
  u / o   pivot left / right        space   duck
  c       deep duck                 x   clear the queue      ?   help
  1 jab   2 cross   3 lead hook   4 rear hook   5 lead uppercut

None of these are viser camera keys (W A S D Q E and the arrows).

Each punch puts a 2 kg target ball where it lands (mid-punch); the ball only
moves along the punch's line (straight: forward, hook: across, uppercut: up)
and the momentum it got is printed after the punch. --checkpoint-file needs a
policy trained on the punches (e.g. on data/motions/punch_training.npz).

Two modes:
  reference  replays the clips exactly, no physics policy: checks the clips
             and the key -> move wiring
  policy     the trained tracking policy follows the clips under physics

  uv run scripts/build_move_library.py           # once, writes data/moves/
  uv run scripts/teleop.py --mode reference --viewer viser
  uv run scripts/teleop.py --mode policy --viewer viser \
      --checkpoint-file logs/rsl_rl/wato_tracking/<run>/model_<n>.pt
"""

import os
import sys
import threading
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("MUJOCO_GL") == "osmesa":
  import torch._dynamo  # noqa: F401  (see play.py)

import torch  # noqa: E402
import tyro  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

import humanoid_tracking  # noqa: E402,F401  (registers the task)
from mjlab.entity import EntityCfg  # noqa: E402

from humanoid_boxing.fighters import add_hitboxes  # noqa: E402
from humanoid_tracking.target_ball import GLOVE_HIT, free_ball_spec  # noqa: E402
from humanoid_tracking.teleop import TeleopMotionCommandCfg  # noqa: E402

TASK = "Mjlab-Tracking-Flat-Wato"
BEST = "logs/rsl_rl/wato_tracking/2026-10-08_07-05-43_exp6_angvel_std1.5/model_12995.pt"

# key -> (move, button label); kept off viser's camera keys (W A S D Q E, arrows)
KEYMAP = {
  "i": ("step_forward", "I  step forward"),
  "k": ("step_back", "K  step back"),
  "j": ("step_left", "J  step left"),
  "l": ("step_right", "L  step right"),
  "u": ("pivot_left", "U  pivot left"),
  "o": ("pivot_right", "O  pivot right"),
  "f": ("slip_left", "F  slip left"),
  "g": ("slip_right", "G  slip right"),
  "r": ("weave_left", "R  weave left"),
  "t": ("weave_right", "T  weave right"),
  " ": ("duck", "Space  duck"),
  "c": ("deep_duck", "C  deep duck"),
  "1": ("jab", "1  jab"),
  "2": ("cross", "2  cross"),
  "3": ("lead_hook", "3  lead hook"),
  "4": ("rear_hook", "4  rear hook"),
  "5": ("lead_uppercut", "5  lead uppercut"),
}


def print_help() -> None:
  print("\nkeys:  " + "   ".join(f"{label.split()[0].lower()}={m}" for m, label in KEYMAP.values()))
  print("       x=clear queue   ?=help   ctrl-c in the viewer / this terminal quits\n", flush=True)


def keyboard_thread(request) -> None:
  """Read single key presses from the terminal (no Enter needed)."""
  if not sys.stdin.isatty():
    print("[teleop] stdin is not a terminal: keyboard off (use the viser buttons)")
    return
  import atexit
  import termios
  import tty

  fd = sys.stdin.fileno()
  old = termios.tcgetattr(fd)
  atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)
  tty.setcbreak(fd)

  def loop() -> None:
    while True:
      ch = sys.stdin.read(1).lower()
      if ch in KEYMAP:
        request(KEYMAP[ch][0])
      elif ch == "x":
        request("__clear__")
        print("[teleop] queue cleared", flush=True)
      elif ch == "?":
        print_help()

  threading.Thread(target=loop, daemon=True).start()


# moves the footwork policy plays when --footwork-checkpoint-file is given
FOOTWORK_MOVES = (
  "step_forward", "step_back", "step_left", "step_right", "pivot_left", "pivot_right",
  "slip_left", "slip_right", "weave_left", "weave_right", "duck", "deep_duck",
)  # fmt: skip


def make_env(mode: str, checkpoint_file: str, library_dir: str, device: str, footwork_checkpoint_file: str | None = None):
  """Tracking env driven by the move scheduler, plus the policy to run on it."""
  if not (Path(library_dir) / "library.yaml").exists():
    sys.exit(f"{library_dir}/library.yaml not found: run scripts/build_move_library.py first")

  env_cfg = load_env_cfg(TASK, play=True)
  old = env_cfg.commands["motion"]
  env_cfg.commands["motion"] = TeleopMotionCommandCfg(
    **{f.name: getattr(old, f.name) for f in fields(old)},
    library_dir=library_dir,
    kinematic=mode == "reference",
    buttons=tuple((label, move) for move, label in KEYMAP.values()),
    on_start=lambda move, pending: print(
      f"[teleop] {move}" + (f"   (next: {', '.join(pending)})" if pending else ""), flush=True
    ),
  )
  env_cfg.commands["motion"].motion_file = str(Path(library_dir) / "npz" / "guard.npz")
  env_cfg.scene.num_envs = 1
  # gloves + a real target ball for the punches (they only touch each other)
  robot = env_cfg.scene.entities["robot"]
  base_spec_fn = robot.spec_fn

  def spec_with_gloves():
    spec = base_spec_fn()
    add_hitboxes(spec, "red")
    return spec

  robot.spec_fn = spec_with_gloves
  robot.collisions = (*robot.collisions, GLOVE_HIT)
  env_cfg.scene.entities["target"] = EntityCfg(spec_fn=free_ball_spec)
  if mode == "reference":
    env_cfg.terminations = {}
  else:
    # reset only on a fall, not when a hand or foot misses its target
    env_cfg.terminations.pop("ee_body_pos", None)

  agent_cfg = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)

  if mode == "policy":

    def load(path):
      runner = MotionTrackingOnPolicyRunner(env, asdict(agent_cfg), device=device)
      runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)
      return runner.get_inference_policy(device=device)

    punch_policy = load(checkpoint_file)
    print(f"[teleop] policy: {checkpoint_file}")
    if footwork_checkpoint_file:
      footwork_policy = load(footwork_checkpoint_file)
      scheduler = env.unwrapped.command_manager.get_term("motion").scheduler
      print(f"[teleop] footwork policy ({', '.join(FOOTWORK_MOVES)}): {footwork_checkpoint_file}")

      def policy(obs):
        # each move starts and ends in the same guard, so the hand-over is clean
        return footwork_policy(obs) if scheduler.move in FOOTWORK_MOVES else punch_policy(obs)
    else:
      policy = punch_policy
  else:
    shape = env.unwrapped.action_space.shape

    def policy(obs):
      return torch.zeros(shape, device=env.unwrapped.device)

  return env, policy


def main(
  mode: Literal["reference", "policy"] = "reference",
  checkpoint_file: str = BEST,
  footwork_checkpoint_file: str | None = None,
  library_dir: str = "data/moves",
  viewer: Literal["auto", "native", "viser"] = "auto",
  device: str | None = None,
):
  """Args:
    mode: reference = replay the clips (no policy); policy = run the checkpoint.
    checkpoint_file: tracking policy for --mode policy (punches and guard, and
      everything else when there is no footwork policy).
    footwork_checkpoint_file: optional separate policy for the footwork and
      defence moves (FOOTWORK_MOVES), e.g. one trained on
      Mjlab-Tracking-Flat-Wato-Footwork.
    library_dir: output of scripts/build_move_library.py.
    viewer: native needs a display; viser serves http://localhost:8080.
    device: torch / MuJoCo Warp device (default cuda:0 if available).
  """
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env, policy = make_env(mode, checkpoint_file, library_dir, device, footwork_checkpoint_file)
  scheduler = env.unwrapped.command_manager.get_term("motion").scheduler
  keyboard_thread(scheduler.request)
  print_help()

  if viewer == "auto":
    viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
  if viewer == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
