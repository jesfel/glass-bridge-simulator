"""Game and behavior parameters. Defaults follow the Article 1 rules spec."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum


class Outcome(IntEnum):
    SURVIVED = 0
    GLASS = 1     # chose the breaking pane
    PUSHED = 2    # pushed onto a breaking pane, or off the bridge
    DRAG = 3      # dragged down with (or by) the other player in a push
    BACKFIRE = 4  # pusher fell while pushing
    TIMEOUT = 5   # still on the bridge when the clock ran out


@dataclass(frozen=True)
class Params:
    # The world
    n_steps: int = 18
    n_players: int = 16
    clock: float = 960.0      # seconds; math.inf = no clock
    t_move: float = 3.0       # seconds per known step

    # Hesitation: decision delay at an unknown pane
    d0: float = 20.0          # median delay with no pressure (s)
    sigma: float = 1.0        # spread of log delay
    delay: str = "lognormal"  # or "gamma": same median and mean, robustness check
    beta: float = 0.0         # pressure: <0 rush, 0 none, >0 freeze

    # Pushing
    n_pushers: int = 0        # exact count, placed at random in the queue
    patience: float = 15.0    # a pusher pushes a leader stalled longer than this (s)
    t_start: float = 420.0    # nobody pushes before this time (s); the 7-minute rule
    t_push: float = 3.0       # duration of the struggle (s)
    p_land: float = 0.6       # struggle: victim goes over
    p_drag: float = 0.2       # struggle: both go over
    p_backfire: float = 0.2   # struggle: pusher goes over
    p_contact: float = 0.5    # whoever goes over touches the next pane (reveals it)

    def __post_init__(self) -> None:
        if self.n_steps < 0 or self.n_players < 1:
            raise ValueError("need n_steps >= 0 and n_players >= 1")
        if not 0 <= self.n_pushers <= self.n_players:
            raise ValueError("n_pushers must be between 0 and n_players")
        if not math.isclose(self.p_land + self.p_drag + self.p_backfire, 1.0):
            raise ValueError("p_land + p_drag + p_backfire must equal 1")
        for name in ("p_land", "p_drag", "p_backfire", "p_contact"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.delay not in ("lognormal", "gamma"):
            raise ValueError("delay must be 'lognormal' or 'gamma'")
        if self.clock <= 0 or self.d0 <= 0 or self.sigma < 0:
            raise ValueError("clock and d0 must be positive, sigma non-negative")

    def with_(self, **changes) -> Params:
        return replace(self, **changes)


BASELINE = Params(clock=math.inf, n_pushers=0)
"""Ideal play: no clock, no pushers. Must match the exact odds."""
