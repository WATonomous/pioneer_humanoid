"""Small-sided simulation rules with explicit dead balls and full-ball goal crossings."""

from dataclasses import dataclass, field
from math import isfinite


@dataclass
class MatchRules:
    duration: float = 180.0
    restart_delay: float = 2.0
    elapsed: float = 0.0
    score: list = field(default_factory=lambda: [0, 0])
    phase: str = "PLAY"
    restart_left: float = 0.0
    kickoff: int = 0
    events: list = field(default_factory=list)

    # Award a goal only after the entire ball crosses between the physical posts and below the bar.
    def observe(
        self,
        previous,
        ball,
        dt,
        player_fall=False,
        opponent_fall=False,
        attempt_end=False,
    ):
        if dt <= 0 or not all(isfinite(float(x)) for x in (*previous, *ball, dt)):
            raise ValueError(
                "Match observations must be finite with positive time step"
            )
        if self.phase == "FULL_TIME":
            return None
        self.elapsed += dt
        if self.elapsed + 1e-9 >= self.duration:
            self.elapsed = self.duration
            self.phase = "FULL_TIME"
            return self.event("full_time")
        if self.phase == "RESTART":
            self.restart_left -= dt
            if self.restart_left <= 0:
                self.phase = "PLAY"
                return self.event("kickoff")
            return None
        for team, sign in enumerate((1, -1)):
            plane = sign * 6.11
            if sign * previous[0] <= 6.11 < sign * ball[0]:
                fraction = (plane - previous[0]) / (ball[0] - previous[0])
                y = previous[1] + fraction * (ball[1] - previous[1])
                z = previous[2] + fraction * (ball[2] - previous[2])
                if abs(y) <= 0.85 and 0.085 <= z <= 0.85:
                    self.score[team] += 1
                    self.kickoff = 1 - team
                    return self.dead_ball("goal", team=team)
        if abs(ball[0]) > 6.11 or abs(ball[1]) > 4.11:
            self.kickoff = 1 - self.kickoff
            return self.dead_ball("out_of_play")
        if player_fall or opponent_fall:
            self.kickoff = 1 if player_fall else 0
            return self.dead_ball(
                "fall", player=bool(player_fall), opponent=bool(opponent_fall)
            )
        if attempt_end:
            self.kickoff = 1 - self.kickoff
            return self.dead_ball("turnover")
        return None

    # Physical resets are permitted only after these explicit dead-ball decisions.
    def dead_ball(self, reason, **details):
        self.phase = "RESTART"
        self.restart_left = self.restart_delay
        return self.event(reason, **details)

    def event(self, kind, **details):
        event = {
            "event": kind,
            "time_s": self.elapsed,
            "score": self.score.copy(),
            **details,
        }
        self.events.append(event)
        return event
