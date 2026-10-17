"""Fight teleop: you box as red against blue in the arena, with full physics.

Same keys as scripts/teleop.py (keep this terminal focused; with --viewer viser
the browser also gets one button per move). Moves queue (up to 4) and start
when red is back in guard; with nothing queued red holds the guard. Both
fighters carry out their moves with the frozen move skills the fight task
uses, so what you see is what the fight policy trains on.

  i / k   step forward / back       f / g   slip left / right
  j / l   step left / right         r / t   bob-and-weave left / right
  u / o   pivot left / right        space   duck
  c       deep duck                 x   clear the queue      ?   help
  1 jab   2 cross   3 lead hook   4 rear hook   5 lead uppercut

Blue (--opponent):
  hold      stands in guard (a still target)
  scripted  random footwork, defence and punches
  ai        a trained fight policy (--checkpoint-file, from Mjlab-Boxing-Fight-Wato)

A round ends when someone goes down (or after 20 s); the ring then resets.

  uv run scripts/fight_teleop.py --opponent hold --viewer viser
  uv run scripts/fight_teleop.py --opponent ai --viewer viser \
      --checkpoint-file logs/rsl_rl/wato_fight/<run>/model_<n>.pt
"""

import contextlib
import io
import os
import sys
import threading
from collections import deque
from dataclasses import asdict
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("MUJOCO_GL") == "osmesa":
  import torch._dynamo  # noqa: F401  (see play.py)

import torch  # noqa: E402
import tyro  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

import humanoid_boxing  # noqa: E402,F401  (registers the fight task)
import humanoid_tracking  # noqa: E402,F401
from humanoid_boxing import mdp as fight  # noqa: E402
from humanoid_boxing import skills  # noqa: E402
from humanoid_boxing.rl_cfg import FightRunner  # noqa: E402

TASK = "Mjlab-Boxing-Fight-Wato"
QUEUE = 4
OPPONENT_MODE = {"hold": skills.HOLD, "scripted": skills.SCRIPTED, "ai": skills.SELFPLAY}

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


class Controller:
  """Your move queue for red, and the round's running score."""

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.red = env.action_manager.get_term("red")
    self.queue: deque[str] = deque(maxlen=QUEUE)
    self.lock = threading.Lock()
    self.landed = {"red": 0, "blue": 0}
    self.score = {"red": 0, "blue": 0}

  def request(self, move: str) -> None:
    with self.lock:
      if move == "__clear__":
        self.queue.clear()
      else:
        self.queue.append(move)

  def __call__(self, obs) -> torch.Tensor:
    self._report()
    move = skills.GUARD
    if bool(self.red.player.ready[0]):
      with self.lock:
        if self.queue:
          name = self.queue.popleft()
          move = skills.MOVES.index(name)
          pending = list(self.queue)
          print(f"[fight] {name}" + (f"   (next: {', '.join(pending)})" if pending else ""), flush=True)
    action = torch.zeros(self.env.num_envs, self.red.action_dim, device=self.env.device)
    action[:, 0] = move  # technique 0: the punches as recorded
    return action

  def _report(self) -> None:
    st = fight.fight_state(self.env)
    if not st.teams:
      return
    for team in ("red", "blue"):
      n = int(st.teams[team].landed[0])
      if n > self.landed[team]:
        print(f"[fight] {team} lands a punch ({n} this round)", flush=True)
      self.landed[team] = n
    red_down, blue_down = bool(st.teams["red"].down[0]), bool(st.teams["blue"].down[0])
    if red_down or blue_down:
      if blue_down and not red_down:
        hit = float(st.teams["blue"].since_hit[0]) <= fight.HIT_WINDOW
        print(f"[fight] blue is down ({'KNOCKDOWN' if hit else 'slip'})", flush=True)
        self.score["red"] += 1
      elif red_down and not blue_down:
        print("[fight] you are down", flush=True)
        self.score["blue"] += 1
      else:
        print("[fight] both down", flush=True)
      print(f"[fight] rounds: you {self.score['red']} - blue {self.score['blue']}   (new round)", flush=True)
      self.landed = {"red": 0, "blue": 0}
      with self.lock:
        self.queue.clear()


def keyboard_thread(request) -> None:
  """Read single key presses from the terminal (no Enter needed)."""
  if not sys.stdin.isatty():
    print("[fight] stdin is not a terminal: keyboard off (use the viser buttons)")
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
        print("[fight] queue cleared", flush=True)
      elif ch == "?":
        print_help()

  threading.Thread(target=loop, daemon=True).start()


class FightViser(ViserPlayViewer):
  """The play viewer plus one button per move."""

  def __init__(self, env, policy, controller: Controller):
    self._controller = controller
    super().__init__(env, policy)

  def setup(self) -> None:
    super().setup()
    with self._server.gui.add_folder("Your moves"):
      for move, label in KEYMAP.values():
        self._server.gui.add_button(label).on_click(lambda _, m=move: self._controller.request(m))
      self._server.gui.add_button("Clear queue").on_click(lambda _: self._controller.request("__clear__"))


def main(
  opponent: Literal["hold", "scripted", "ai"] = "hold",
  checkpoint_file: str | None = None,
  viewer: Literal["auto", "native", "viser"] = "auto",
  device: str | None = None,
):
  """Args:
    opponent: blue's behaviour (hold = still target, scripted, ai = trained policy).
    checkpoint_file: a Mjlab-Boxing-Fight-Wato checkpoint, for --opponent ai.
    viewer: native needs a display; viser serves http://localhost:8080.
    device: torch / MuJoCo Warp device (default cuda:0 if available).
  """
  if opponent == "ai" and not checkpoint_file:
    sys.exit("--opponent ai needs --checkpoint-file logs/rsl_rl/wato_fight/<run>/model_<n>.pt")
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.curriculum = {}  # blue's mode is fixed by --opponent
  agent_cfg = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  u = env.unwrapped

  blue = u.action_manager.get_term("blue")
  probs = [0.0, 0.0, 0.0]
  probs[OPPONENT_MODE[opponent]] = 1.0
  blue.mode_probs = torch.tensor(probs, device=u.device)
  if opponent == "ai":
    with contextlib.redirect_stdout(io.StringIO()):
      runner = FightRunner(env, asdict(agent_cfg), device=device)
      runner.load(checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=device)
    blue.learner_actor = runner.alg.actor  # blue plays this policy from its own side
    print(f"[fight] blue: {checkpoint_file}")
  else:
    print(f"[fight] blue: {opponent}")
  env.reset()

  controller = Controller(u)
  keyboard_thread(controller.request)
  print_help()

  if viewer == "auto":
    viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
  if viewer == "native":
    NativeMujocoViewer(env, controller).run()
  else:
    FightViser(env, controller, controller).run()
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
