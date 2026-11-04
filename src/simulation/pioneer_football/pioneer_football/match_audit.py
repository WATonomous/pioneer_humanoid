"""Audit both physical robots independently of score-event priority and kickoff resets."""

from dataclasses import dataclass, field
from math import isfinite


@dataclass
class MatchAudit:
    steps: int = 0
    outside_steps: list = field(default_factory=lambda: [0, 0])
    maximum_outside: list = field(default_factory=lambda: [0.0, 0.0])
    physical_falls: list = field(default_factory=lambda: [0, 0])
    was_fallen: list = field(default_factory=lambda: [False, False])

    # Count physical faults during live play and dead balls, including a fall simultaneous with a goal.
    def observe(self, states):
        if len(states) != 2:
            raise ValueError("A match audit requires both physical players")
        self.steps += 1
        new_falls = []
        for team, (x, y, z, gravity_z) in enumerate(states):
            if not all(isfinite(value) for value in (x, y, z, gravity_z)):
                raise ValueError("Nonfinite physical player state")
            self.outside_steps[team] += int(abs(x) > 6.3 or abs(y) > 4.3)
            self.maximum_outside[team] = max(
                self.maximum_outside[team], abs(x) - 6.0, abs(y) - 4.0
            )
            fallen = z < 0.5 or gravity_z > -0.5
            fresh = fallen and not self.was_fallen[team]
            self.physical_falls[team] += int(fresh)
            self.was_fallen[team] = fallen
            new_falls.append(fresh)
        return new_falls

    # A kickoff starts new physical bouts while preserving the match-wide audit.
    def kickoff(self):
        self.was_fallen[:] = [False, False]

    def report(self):
        return {
            "physical_falls": self.physical_falls,
            "in_pitch_fraction": [
                1 - count / max(1, self.steps) for count in self.outside_steps
            ],
            "maximum_outside_pitch_m": self.maximum_outside,
        }
