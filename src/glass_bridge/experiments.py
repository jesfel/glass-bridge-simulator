"""Run many games and summarize them."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from .engine import play_game
from .params import Outcome, Params


@dataclass
class Batch:
    params: Params
    seed: int
    outcome: np.ndarray    # (games, players) Outcome codes, queue order
    is_pusher: np.ndarray  # (games, players) bool
    runtime_s: float

    @property
    def n_games(self) -> int:
        return self.outcome.shape[0]

    @property
    def survived(self) -> np.ndarray:
        return self.outcome == Outcome.SURVIVED

    def survival_by_position(self) -> np.ndarray:
        return self.survived.mean(axis=0)

    def survivors(self) -> np.ndarray:
        return self.survived.sum(axis=1)

    def survivors_distribution(self) -> np.ndarray:
        return np.bincount(self.survivors(), minlength=self.params.n_players + 1) / self.n_games

    def mean_survivors(self) -> tuple[float, float]:
        """Mean and its Monte Carlo standard error."""
        s = self.survivors()
        return float(s.mean()), float(s.std(ddof=1) / math.sqrt(len(s)))

    def cause_shares(self) -> dict[str, float]:
        counts = np.bincount(self.outcome.ravel(), minlength=len(Outcome)) / self.outcome.size
        return {o.name.lower(): float(counts[o]) for o in Outcome}

    def pusher_survival(self) -> tuple[float, float]:
        """Survival rate of pushers vs non-pushers (nan if a group is empty)."""
        s, m = self.survived, self.is_pusher
        rate = lambda mask: float(s[mask].mean()) if mask.any() else math.nan
        return rate(m), rate(~m)


def run_batch(p: Params, n_games: int, seed: int) -> Batch:
    rng = np.random.default_rng(seed)
    outcome = np.empty((n_games, p.n_players), dtype=np.int8)
    is_pusher = np.zeros((n_games, p.n_players), dtype=bool)
    start = time.perf_counter()
    for g in range(n_games):
        res = play_game(p, rng, log=False)
        outcome[g] = res.outcome
        is_pusher[g, list(res.pushers)] = True
    return Batch(p, seed, outcome, is_pusher, time.perf_counter() - start)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a batch of Glass Bridge games and print a summary.")
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1)
    for name, value in asdict(Params()).items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = vars(ap.parse_args())
    games, seed = args.pop("games"), args.pop("seed")
    b = run_batch(Params(**args), games, seed)
    mean, se = b.mean_survivors()
    print(f"{games} games, seed {seed}, {b.runtime_s:.1f} s")
    print(f"mean survivors {mean:.2f} ± {se:.2f}")
    print("survival by position", np.round(b.survival_by_position(), 3))
    print("causes", {k: round(v, 3) for k, v in b.cause_shares().items()})
    print("pusher / non-pusher survival", tuple(round(x, 3) for x in b.pusher_survival()))


if __name__ == "__main__":
    main()
