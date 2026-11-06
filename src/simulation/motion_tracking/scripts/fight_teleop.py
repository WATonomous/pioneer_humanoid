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
  n       new match
  1 jab   2 cross   3 lead hook   4 rear hook   5 lead uppercut   6 rear uppercut

Blue (--opponent):
  hold      stands in guard (a still target)
  scripted  random footwork, defence and punches
  ai        a trained fight policy (--checkpoint-file, from Mjlab-Boxing-Fight-Pioneer)

A MATCH lasts --match-minutes (5) of sim time (it pauses with the viewer). A
knockdown resets the ring; the score carries on. A live scoreboard sits at the
top of the viser panel (and above the ring); the terminal prints every punch.
n starts a new match.

Scoring, per clean punch (a punch move's glove landing on the head or torso at
>= 0.4 m/s, once per punch; humanoid_boxing/impact.py):
  points = where (head 10, torso 5) x punch (jab 1, cross 1.2, lead hook 1.3,
           rear hook 1.4, lead uppercut 1.4, rear uppercut 1.5)
           x power (peak glove force over the first 0.1 s / 100 N, 0.5-1.5)

A punch that only lands on the arms or gloves is blocked: 0 points.

  uv run scripts/fight_teleop.py --opponent hold --viewer viser
  uv run scripts/fight_teleop.py --opponent ai --viewer viser \
      --checkpoint-file logs/rsl_rl/pioneer_fight/<run>/model_<n>.pt
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
from humanoid_boxing import impact  # noqa: E402
from humanoid_boxing import mdp as fight  # noqa: E402
from humanoid_boxing import skills  # noqa: E402
from humanoid_boxing.rl_cfg import FightRunner  # noqa: E402

TASK = "Mjlab-Boxing-Fight-Pioneer"
QUEUE = 4
OPPONENT_MODE = {"hold": skills.HOLD, "scripted": skills.SCRIPTED, "ai": skills.SELFPLAY}

# match scoring
ZONE_POINTS = {"head": 10.0, "torso": 5.0}
PUNCH_MULT = {
  "jab": 1.0, "cross": 1.2, "lead_hook": 1.3, "rear_hook": 1.4, "lead_uppercut": 1.4, "rear_uppercut": 1.5,
}
POWER_REF = 100.0  # N of peak glove force for power x1 (a 30 s test vs the bot: clean punches peaked 62-515 N, median ~150)
POWER_RANGE = (0.5, 1.5)
NAMES = {"red": "YOU", "blue": "BOT"}

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
  "6": ("rear_uppercut", "6  rear uppercut"),
}


def print_help() -> None:
  print("\nkeys:  " + "   ".join(f"{label.split()[0].lower()}={m}" for m, label in KEYMAP.values()))
  print("       x=clear queue   ?=help   ctrl-c in the viewer / this terminal quits\n", flush=True)


class Controller:
  """Your move queue for red, the match clock and the live score."""

  def __init__(self, env: ManagerBasedRlEnv, match_s: float):
    self.env = env
    self.red = env.action_manager.get_term("red")
    self.queue: deque[str] = deque(maxlen=QUEUE)
    self.lock = threading.Lock()
    self.match_s = match_s
    self.on_update = None  # scoreboard callback: f(controller)
    self.new_match()

  def new_match(self) -> None:
    with self.lock:
      self.queue.clear()
    self.t = 0.0
    self.over = False
    self.points = {"red": 0.0, "blue": 0.0}
    self.landed = {"red": 0, "blue": 0}
    self.blocked = {"red": 0, "blue": 0}
    self.knockdowns = {"red": 0, "blue": 0}  # knockdowns scored BY this fighter
    self.last = "fight!"
    self._count = {"red": 0, "blue": 0}  # impact's clean-punch count this ring reset
    self._pending: dict[str, dict] = {}  # a landed punch whose power is still being measured
    self._dirty = True

  def request(self, move: str) -> None:
    if move == "__new__":
      self.new_match()
      print("[fight] new match", flush=True)
      return
    with self.lock:
      if move == "__clear__":
        self.queue.clear()
      else:
        self.queue.append(move)

  def __call__(self, obs) -> torch.Tensor:
    if not self.over:
      self._score()
      self.t += self.env.step_dt
      if self.t >= self.match_s:
        self._finish()
    if self._dirty and self.on_update:
      self.on_update(self)
      self._dirty = False
    move = skills.GUARD
    if bool(self.red.player.ready[0]) and not self.over:
      with self.lock:
        if self.queue:
          name = self.queue.popleft()
          move = skills.MOVES.index(name)
          pending = list(self.queue)
          print(f"[fight] {name}" + (f"   (next: {', '.join(pending)})" if pending else ""), flush=True)
    action = torch.zeros(self.env.num_envs, self.red.action_dim, device=self.env.device)
    action[:, 0] = move  # technique 0: the punches as recorded
    return action

  # -- scoring ---------------------------------------------------------------------
  def _score(self) -> None:
    im = impact.impacts(self.env)
    for team in ("red", "blue"):
      opp = fight.other(team)
      s = im.sides[team]
      n = int(s.count[0])
      if n < self._count[team]:  # the ring was reset (knockdown)
        self._count[team] = 0
        self._pending.pop(team, None)
      if bool(s.blocked[0]):
        self.blocked[team] += 1
        self._dirty = True
      force = float(impact._force(self.env, team, opp)[0])
      if n > self._count[team]:  # a clean punch landed in the last step
        self._finalise(team)  # (a previous one still measuring)
        zone = "head" if int(s.zone[0]) == 0 else "torso"
        punch = skills.PUNCHES[int(s.recent_type[0, -1])]
        self._pending[team] = {"zone": zone, "punch": punch, "peak": force, "steps": 0}
      elif team in self._pending:
        p = self._pending[team]
        p["peak"] = max(p["peak"], force)
        p["steps"] += 1
        if p["steps"] * self.env.step_dt >= impact.FORCE_WINDOW:
          self._finalise(team)
      self._count[team] = n
    st = fight.fight_state(self.env)
    if st.teams:
      for team in ("red", "blue"):
        if bool(st.teams[team].down[0]) and not bool(st.teams[fight.other(team)].down[0]):
          self.knockdowns[fight.other(team)] += 1
          self.last = f"{NAMES[team]} DOWN"
          print(f"[fight] {NAMES[team].lower()} down: the ring resets, the score carries on", flush=True)
          self._dirty = True
          with self.lock:
            self.queue.clear()

  def _finalise(self, team: str) -> None:
    p = self._pending.pop(team, None)
    if p is None:
      return
    power = min(max(p["peak"] / POWER_REF, POWER_RANGE[0]), POWER_RANGE[1])
    pts = ZONE_POINTS[p["zone"]] * PUNCH_MULT[p["punch"]] * power
    self.points[team] += pts
    self.landed[team] += 1
    self.last = (
      f"{NAMES[team]}: {p['punch'].replace('_', ' ')} to the {p['zone']}  "
      f"{ZONE_POINTS[p['zone']]:.0f} x{PUNCH_MULT[p['punch']]:.1f} x{power:.2f} ({p['peak']:.0f} N) = +{pts:.1f}"
    )
    print(f"[fight] {self.last}   |   {self.board()}", flush=True)
    self._dirty = True

  def _finish(self) -> None:
    for team in list(self._pending):
      self._finalise(team)
    self.over = True
    self.t = self.match_s
    r, b = self.points["red"], self.points["blue"]
    self.last = "DRAW" if abs(r - b) < 1e-6 else ("YOU WIN" if r > b else "BOT WINS")
    print(f"\n[fight] FULL TIME: {self.last}   {self.board()}   (n = new match)\n", flush=True)
    with self.lock:
      self.queue.clear()
    self._dirty = True

  # -- display ---------------------------------------------------------------------
  def clock(self) -> str:
    left = max(self.match_s - self.t, 0.0)
    return f"{int(left // 60)}:{int(left % 60):02d}"

  def board(self) -> str:
    return f"YOU {self.points['red']:.1f} - {self.points['blue']:.1f} BOT   {self.clock()}"

  def html(self) -> str:
    r, b = self.points["red"], self.points["blue"]
    lead = "#2e7d32" if r > b else ("#c62828" if b > r else "#555")
    state = f"<b>{self.last}</b>" if self.over else self.last
    row = lambda t: (  # noqa: E731
      f"{self.landed[t]} landed &middot; {self.blocked[t]} blocked &middot; {self.knockdowns[t]} KD"
    )
    return (
      f"<div style='font-family:sans-serif;text-align:center;padding:4px'>"
      f"<div style='font-size:12px;color:#888'>{'FULL TIME' if self.over else 'TIME LEFT'} {self.clock()}</div>"
      f"<div style='font-size:28px;font-weight:700;color:{lead}'>YOU {r:.1f} &ndash; {b:.1f} BOT</div>"
      f"<div style='font-size:11px;color:#888'>you: {row('red')}<br>bot: {row('blue')}</div>"
      f"<div style='font-size:12px;margin-top:4px'>{state}</div></div>"
    )


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
      elif ch == "n":
        request("__new__")
      elif ch == "?":
        print_help()

  threading.Thread(target=loop, daemon=True).start()


class FightViser(ViserPlayViewer):
  """The play viewer plus the live scoreboard (top of the panel, and above the
  ring) and one button per move."""

  def __init__(self, env, policy, controller: Controller):
    self._controller = controller
    super().__init__(env, policy)

  def setup(self) -> None:
    self._board = self._server.gui.add_html(self._controller.html())  # first: the top of the panel
    self._label = self._server.scene.add_label("/scoreboard", self._controller.board(), position=(0.0, 0.0, 2.2))

    def update(c: Controller) -> None:
      self._board.content = c.html()
      self._label.text = c.board() + (f"   {c.last}" if c.over else "")

    self._controller.on_update = update
    super().setup()
    self._server.gui.add_button("New match (n)").on_click(lambda _: self._controller.request("__new__"))
    with self._server.gui.add_folder("Your moves"):
      for move, label in KEYMAP.values():
        self._server.gui.add_button(label).on_click(lambda _, m=move: self._controller.request(m))
      self._server.gui.add_button("Clear queue").on_click(lambda _: self._controller.request("__clear__"))


def main(
  opponent: Literal["hold", "scripted", "ai"] = "hold",
  checkpoint_file: str | None = None,
  viewer: Literal["auto", "native", "viser"] = "auto",
  device: str | None = None,
  match_minutes: float = 5.0,
):
  """Args:
    opponent: blue's behaviour (hold = still target, scripted, ai = trained policy).
    checkpoint_file: a Mjlab-Boxing-Fight-Pioneer checkpoint, for --opponent ai.
    viewer: native needs a display; viser serves http://localhost:8080.
    device: torch / MuJoCo Warp device (default cuda:0 if available).
    match_minutes: match length (sim time); the higher score at full time wins.
  """
  if opponent == "ai" and not checkpoint_file:
    sys.exit("--opponent ai needs --checkpoint-file logs/rsl_rl/pioneer_fight/<run>/model_<n>.pt")
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.curriculum = {}  # blue's mode is fixed by --opponent
  env_cfg.episode_length_s = match_minutes * 60.0 * 4  # no 20 s round resets: the match clock decides
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

  controller = Controller(u, match_minutes * 60.0)
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
