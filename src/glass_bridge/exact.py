"""Exact baseline odds (no clock, no pushers), used to validate the simulator.

Every step is revealed by exactly one guess, right or wrong, so the number of deaths
W ~ Binomial(n_steps, 1/2), capped at the number of players. Player k (1-based)
survives iff W <= k - 1. The Markov chain version follows Tijms and must agree.
"""

from __future__ import annotations

from math import comb

import numpy as np


def _wrong_guesses_pmf(n_steps: int) -> np.ndarray:
    return np.array([comb(n_steps, w) for w in range(n_steps + 1)], dtype=float) / 2.0**n_steps


def survival_by_position(n_steps: int = 18, n_players: int = 16) -> np.ndarray:
    cdf = np.cumsum(_wrong_guesses_pmf(n_steps))
    return np.array([cdf[min(k - 1, n_steps)] for k in range(1, n_players + 1)])


def survivors_distribution(n_steps: int = 18, n_players: int = 16) -> np.ndarray:
    """P(exactly s survivors), s = 0..n_players."""
    pmf = _wrong_guesses_pmf(n_steps)
    dist = np.zeros(n_players + 1)
    for w, pw in enumerate(pmf):
        dist[max(n_players - w, 0)] += pw
    return dist


def expected_survivors(n_steps: int = 18, n_players: int = 16) -> float:
    return float(survival_by_position(n_steps, n_players).sum())


def markov_survival_by_position(n_steps: int = 18, n_players: int = 16) -> np.ndarray:
    """Tijms' absorbing chain: state i = step of the last fall, state N+1 = crossed."""
    N = n_steps
    M = np.zeros((N + 2, N + 2))
    for i in range(N + 1):
        for j in range(i + 1, N + 1):
            M[i, j] = 0.5 ** (j - i)
        M[i, N + 1] = 0.5 ** (N - i)
    M[N + 1, N + 1] = 1.0
    out, Mk = [], np.eye(N + 2)
    for _ in range(n_players):
        Mk = Mk @ M
        out.append(Mk[0, N + 1])
    return np.array(out)
