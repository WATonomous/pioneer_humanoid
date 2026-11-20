"""Controller pose filter: a 1-euro filter on position, SLERP on rotation with its own adaptive cutoff, speed caps.

The scheme is OpenArm's VR teleop (enactic/dora-openarm-vr, smoothing.py; there min_cutoff 10 Hz, beta 0.8, caps
1 m/s and 6 rad/s, rotation sharing position's cutoff). Our defaults are re-tuned on the 7-DoF arm benchmark
(IK benchmark, PR #535): min_cutoff 3 Hz, beta 50 s/m, beta_rot 8 s/rad, caps 1 m/s and 10 rad/s.
Against no filter: still-hand wrist jitter 2.9 -> 1.1 mm, ~10 ms more lag. OpenArm's values: 1.6 mm jitter,
~20 ms more lag, and a pure wrist twist lagged 15 deg vs 5 (beta 0.8 hardly raises the cutoff at hand speeds,
rotation ignored its own speed, and 6 rad/s capped fast twists).

The 1-euro filter's cutoff rises with hand speed (cutoff = min_cutoff + beta * speed): a hand held still gets strong
smoothing (Quest jitter roughly halved), a moving hand barely any lag. The caps spread a one-frame tracking jump
over several frames instead of passing it to the IK at once.

Applied to the raw controller pose, before the target mapping (gain, precision), so the caps are in controller
space: with the default --gain 1.5 the robot's wrist target moves at most 1.5 m/s.
"""
from __future__ import annotations

import math

import numpy as np

from control import rotation_exp, rotation_vector


class OneEuroPose:
    def __init__(self, min_cutoff=3.0, beta=50.0, d_cutoff=1.0, max_speed=1.0, max_turn_rate=10.0, beta_rot=8.0):
        """beta_rot (s/rad): rotation's own speed term; 0 = OpenArm's scheme (rotation uses position's cutoff)."""
        if not all(math.isfinite(v) and v >= 0
                   for v in (min_cutoff, beta, d_cutoff, max_speed, max_turn_rate, beta_rot)):
            raise ValueError("Input filter parameters must be finite and non-negative")
        if min_cutoff == 0 or d_cutoff == 0:
            raise ValueError("Input filter cutoffs must be positive (disable the filter instead)")
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.max_speed, self.max_turn_rate, self.beta_rot = max_speed, max_turn_rate, beta_rot
        self.reset()

    def reset(self):
        """Forget the history: the next sample passes through unchanged (call on tracking loss, reset, calibration)."""
        self.position = self.rotation = self.time = None
        self.velocity = np.zeros(3)
        self.turn_speed = 0.0

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, position, rotation, t):
        """Filtered (position, rotation) for a sample at time t (seconds, monotonic)."""
        position, rotation = np.asarray(position, float), np.asarray(rotation, float)
        if self.position is None:
            self.position, self.rotation, self.time = position.copy(), rotation.copy(), t
            return position.copy(), rotation.copy()
        dt = t - self.time
        if not dt > 0:   # same packet again, or a clock step back: hold
            return self.position.copy(), self.rotation.copy()
        self.time = t
        a_d = self._alpha(dt, self.d_cutoff)
        self.velocity = a_d * (position - self.position) / dt + (1 - a_d) * self.velocity
        a = self._alpha(dt, self.min_cutoff + self.beta * float(np.linalg.norm(self.velocity)))
        step = a * (position - self.position)
        length = float(np.linalg.norm(step))
        if self.max_speed > 0 and length > self.max_speed * dt:
            step *= self.max_speed * dt / length
        self.position = self.position + step
        turn = rotation_vector(rotation @ self.rotation.T)
        angle = float(np.linalg.norm(turn))
        w = a
        if self.beta_rot > 0:   # rotation's own adaptive cutoff from its own (filtered) turn speed
            self.turn_speed = a_d * angle / dt + (1 - a_d) * self.turn_speed
            w = self._alpha(dt, self.min_cutoff + self.beta_rot * self.turn_speed)
        if self.max_turn_rate > 0 and angle * w > self.max_turn_rate * dt:
            w = self.max_turn_rate * dt / angle
        self.rotation = rotation_exp(turn * w) @ self.rotation
        return self.position.copy(), self.rotation.copy()


def add_args(parser):
    parser.add_argument("--input-filter-cutoff", type=float, default=3.0,
                        help="1-euro filter minimum cutoff (Hz) on the controller pose; 0 turns the filter off")
    parser.add_argument("--input-filter-beta", type=float, default=50.0,
                        help="1-euro filter position speed coefficient (s/m): higher = less lag when moving")
    parser.add_argument("--input-filter-beta-rot", type=float, default=8.0,
                        help="1-euro filter rotation speed coefficient (s/rad); 0 = rotation follows position's cutoff")
    parser.add_argument("--max-hand-speed", type=float, default=1.0,
                        help="Cap on the filtered controller speed (m/s, before --gain); 0 = no cap")
    parser.add_argument("--max-hand-turn-rate", type=float, default=10.0,
                        help="Cap on the filtered controller turn rate (rad/s); 0 = no cap")


def from_args(args):
    """A filter factory from the flags, or None when --input-filter-cutoff is 0."""
    if args.input_filter_cutoff == 0:
        return None
    params = (args.input_filter_cutoff, args.input_filter_beta, 1.0, args.max_hand_speed, args.max_hand_turn_rate,
              args.input_filter_beta_rot)
    OneEuroPose(*params)   # validate now, at startup
    return lambda: OneEuroPose(*params)
