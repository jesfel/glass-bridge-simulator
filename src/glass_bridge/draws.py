"""Random numbers for a batch of games, drawn up front from separate streams.

Common random numbers: game g under a given seed sees the same bridge, the same leader
guesses and the same delay draws in every scenario, so differences between scenarios come
from the rules, not from luck. Pushers are the first n_pushers of a random order, so the
pusher set grows by one when n_pushers does. Rows never depend on the batch size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaincinv, ndtr

from .params import Params

STREAMS = ("bridge", "guess", "delay", "order", "struggle", "contact")


@dataclass(frozen=True)
class GameDraws:
    """Everything random in one game. Lengths cover the worst case of each counter."""

    bridge: list[int]        # safe side of each step
    guess: list[int]         # side the leader picks at each step (first jump onto it)
    delay: list[float]       # delay multiplier per hesitation (median 1), N + P of them
    order: list[int]         # random queue order; pushers are its first n_pushers
    struggle: list[float]    # uniform per push, N + P of them (a push can remove nobody)
    contact: list[float]     # uniform per push


@dataclass(frozen=True)
class BatchDraws:
    bridge: np.ndarray
    guess: np.ndarray
    delay: np.ndarray
    order: np.ndarray
    struggle: np.ndarray
    contact: np.ndarray

    @property
    def n_games(self) -> int:
        return self.bridge.shape[0]

    def game(self, g: int) -> GameDraws:
        return GameDraws(*(getattr(self, name)[g].tolist() for name in STREAMS))


def draw_batch(p: Params, n_games: int, seed: int) -> BatchDraws:
    N, P = p.n_steps, p.n_players
    rng = {name: np.random.default_rng(s) for name, s in zip(STREAMS, np.random.SeedSequence(seed).spawn(len(STREAMS)))}
    z = rng["delay"].standard_normal((n_games, N + P))
    return BatchDraws(
        bridge=rng["bridge"].integers(0, 2, (n_games, N), dtype=np.int8),
        guess=rng["guess"].integers(0, 2, (n_games, N), dtype=np.int8),
        delay=delay_multiplier(z, p.sigma, p.delay),
        order=np.argsort(rng["order"].random((n_games, P)), axis=1),
        struggle=rng["struggle"].random((n_games, N + P)),
        contact=rng["contact"].random((n_games, N + P)),
    )


def delay_multiplier(z: np.ndarray, sigma: float, dist: str) -> np.ndarray:
    """Map standard normal draws to delay multipliers with median 1.

    lognormal: exp(sigma z). gamma: same median and same mean (exp(sigma^2 / 2)) as the
    lognormal, through the quantile of z, so both families share the same draws.
    """
    if dist == "lognormal" or sigma == 0:
        return np.exp(sigma * z)
    k = gamma_shape(sigma)
    u = ndtr(np.clip(z, -8.0, 8.0))
    return gammaincinv(k, u) / gammaincinv(k, 0.5)


@lru_cache
def gamma_shape(sigma: float) -> float:
    """Gamma shape whose mean / median ratio equals the lognormal's, exp(sigma^2 / 2)."""
    target = math.exp(sigma**2 / 2)
    return brentq(lambda k: k / gammaincinv(k, 0.5) - target, 1e-3, 1e4)
