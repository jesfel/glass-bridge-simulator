"""Deterministic timing checks: hand-built draws, so every time in the game is known exactly.

With beta = 0 and every delay multiplier 1, each decision takes exactly d0 seconds.
"""

import math

from glass_bridge import Outcome, Params, play_game
from glass_bridge.draws import GameDraws

N, P, D0, MOVE = 18, 16, 10.0, 3.0


def draws(wrong_steps=()):
    """Bridge all on side 0; the leader guesses side 0 except at the given (1-based) steps."""
    return GameDraws(bridge=[0] * N, guess=[1 if s in wrong_steps else 0 for s in range(1, N + 1)],
                     delay=[1.0] * (N + P), order=list(range(P)), struggle=[0.5] * (N + P),
                     contact=[0.5] * (N + P))


def game(clock, wrong_steps=()):
    p = Params(clock=clock, d0=D0, beta=0.0, t_move=MOVE, n_pushers=0)
    return play_game(p, draws(wrong_steps), log=True)


def test_perfect_crossing_times():
    """18 right guesses: the leader reaches step 18 at 180 s, steps off 3 s later, and each
    player behind exits 3 s after the one in front."""
    g = game(clock=math.inf)
    assert all(o == Outcome.SURVIVED for o in g.outcome)
    assert g.time == [18 * D0 + MOVE + k * MOVE for k in range(P)]


def test_group_exit_straddles_the_deadline():
    """Exits at 183, 186, ... s; with a 200 s clock the first six make it and the rest time out."""
    g = game(clock=200.0)
    assert g.outcome[:6] == [Outcome.SURVIVED] * 6 and g.time[5] == 198.0
    assert g.outcome[6:] == [Outcome.TIMEOUT] * 10 and all(t == 200.0 for t in g.time[6:])


def test_reaching_the_last_pane_before_the_deadline_is_not_enough():
    """The leader stands on step 18 at 180 s but can't step off before 183 s: everyone times out."""
    g = game(clock=182.0)
    assert all(o == Outcome.TIMEOUT for o in g.outcome)


def test_hesitation_that_crosses_the_deadline():
    """Decisions end at 10 and 20 s; the third would end at 30 s, after a 25 s clock."""
    g = game(clock=25.0)
    assert all(o == Outcome.TIMEOUT for o in g.outcome)
    assert [e.step for e in g.events if e.kind == "safe"] == [1, 2]


def test_next_leader_only_walks_the_gap_after_a_fall():
    """Player 1 falls at step 3 at 30 s. Player 2 was standing on step 1, so they walk 2 known
    steps (6 s), not the whole path, and start hesitating at step 4 at 36 s."""
    g = game(clock=math.inf, wrong_steps={3})
    assert g.outcome[0] == Outcome.GLASS and g.time[0] == 3 * D0
    first_hesitation_p2 = next(e for e in g.events if e.kind == "hesitate" and e.player == 1)
    assert (first_hesitation_p2.t, first_hesitation_p2.step) == (3 * D0 + 2 * MOVE, 4)


def test_every_wrong_guess_costs_one_player():
    """Wrong guesses at steps 1, 2 and 3 kill players 1-3; the other 13 cross."""
    g = game(clock=math.inf, wrong_steps={1, 2, 3})
    assert g.outcome[:3] == [Outcome.GLASS] * 3
    assert g.outcome[3:] == [Outcome.SURVIVED] * 13
