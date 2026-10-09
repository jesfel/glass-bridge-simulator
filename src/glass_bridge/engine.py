"""One Glass Bridge game, step by step.

Players cross in single file, following the leader. Only the leader faces an unknown
pane; everyone behind walks known panes. See README for the full rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

from .draws import GameDraws, draw_batch
from .params import Outcome, Params


class Event(NamedTuple):
    t: float
    kind: str      # hesitate, safe, fall, push, drag, backfire, reveal, exit, timeout
    player: int
    step: int
    detail: object = None


@dataclass
class GameResult:
    outcome: list[Outcome]        # per player, in queue order (0 = first to go)
    time: list[float]             # exit or elimination time
    pushers: frozenset[int]
    bridge: tuple[int, ...]       # safe side (0 = left, 1 = right) of each step
    events: list[Event] = field(default_factory=list)

    @property
    def survivors(self) -> int:
        return sum(o == Outcome.SURVIVED for o in self.outcome)


def play_game(p: Params, d: GameDraws, log: bool = True) -> GameResult:
    N, P, T = p.n_steps, p.n_players, p.clock
    bridge = tuple(d.bridge)
    if p.focal < 0:
        pushers = frozenset(d.order[: p.n_pushers])
    else:  # the other pushers are the same whether or not the focal player pushes
        others = [x for x in d.order if x != p.focal][: p.n_pushers]
        pushers = frozenset(others + [p.focal] * p.focal_pushes)
    n_hesitations = n_pushes = 0

    outcome: list[Outcome | None] = [None] * P
    when = [math.nan] * P
    events: list[Event] = []

    def ev(t, kind, player, step, detail=None):
        if log:
            events.append(Event(t, kind, player, step, detail))

    def eliminate(player, t, cause, step):
        outcome[player], when[player] = cause, t
        ev(t, cause.name.lower(), player, step)

    queue = list(range(P))  # alive players, leader first
    frontier = 0            # furthest step whose safe pane is known (0 = start platform)
    pos = 0                 # leader's position
    t = 0.0

    def jump(player, t, cause_if_wrong):
        """Leader (or pushed victim) lands on the next pane. Returns True if they survive."""
        nonlocal frontier, pos, queue
        step = frontier + 1
        if d.guess[step - 1] == bridge[step - 1]:
            frontier = pos = step
            ev(t, "safe", player, step)
            return True
        eliminate(player, t, cause_if_wrong, step)
        frontier = step                 # the broken pane reveals the safe one
        queue.pop(0)
        pos = max(step - 2, 0)          # next player stood right behind
        return False

    while queue:
        leader = queue[0]

        if frontier == N:  # path complete: walk off in single file
            first_exit = t + (N - pos + 1) * p.t_move
            for k, player in enumerate(queue):
                t_exit = first_exit + k * p.t_move
                if t_exit <= T:
                    outcome[player], when[player] = Outcome.SURVIVED, t_exit
                    ev(t_exit, "exit", player, N)
                else:
                    eliminate(player, T, Outcome.TIMEOUT, N)
            queue = []
            break

        t_walk = t + (frontier - pos) * p.t_move
        if t_walk > T:
            break
        t, pos = t_walk, frontier

        u = 0.0 if math.isinf(T) else min(t / T, 1.0)
        delay = p.d0 * math.exp(p.beta * u) * d.delay[n_hesitations]
        n_hesitations += 1
        ev(t, "hesitate", leader, pos + 1, delay)
        t_decide = t + delay

        if len(queue) > 1 and queue[1] in pushers:
            t_push = max(t + p.patience, p.t_start)
            if t_push < t_decide and t_push + p.t_push <= T:
                pusher = queue[1]
                t = t_push + p.t_push
                r, contact = d.struggle[n_pushes], d.contact[n_pushes] < p.p_contact
                n_pushes += 1
                if r < p.p_land:
                    ev(t, "push", pusher, pos + 1, leader)
                    if contact:
                        jump(leader, t, Outcome.PUSHED)
                    else:  # off the side: no pane revealed
                        eliminate(leader, t, Outcome.PUSHED, pos + 1)
                        queue.pop(0)
                        pos = max(pos - 1, 0)
                    continue
                if r < p.p_land + p.p_drag:
                    eliminate(leader, t, Outcome.DRAG, pos + 1)
                    eliminate(pusher, t, Outcome.DRAG, pos + 1)
                    if contact:
                        frontier += 1
                        ev(t, "reveal", leader, frontier)
                    queue = queue[2:]
                    pos = max(pos - 2, 0)
                    continue
                eliminate(pusher, t, Outcome.BACKFIRE, pos)
                queue.pop(1)
                t_decide = max(t_decide, t)  # leader is still hesitating

        if t_decide > T:
            break
        t = t_decide
        jump(leader, t, Outcome.GLASS)

    for player in queue:  # clock ran out
        eliminate(player, T, Outcome.TIMEOUT, pos)

    return GameResult(outcome, when, pushers, bridge, events)


def trace(p: Params, seed: int, game: int = 0) -> GameResult:
    """Game number `game` of run_batch(p, ..., seed), replayed with a full event log."""
    return play_game(p, draw_batch(p, game + 1, seed).game(game), log=True)
