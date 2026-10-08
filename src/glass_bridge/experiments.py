"""Run many games and summarize them; run the experiment sweeps in configs/."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import subprocess
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from .draws import draw_batch
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
    draws = draw_batch(p, n_games, seed)
    outcome = np.empty((n_games, p.n_players), dtype=np.int8)
    is_pusher = np.zeros((n_games, p.n_players), dtype=bool)
    start = time.perf_counter()
    for g in range(n_games):
        res = play_game(p, draws.game(g), log=False)
        outcome[g] = res.outcome
        is_pusher[g, list(res.pushers)] = True
    return Batch(p, seed, outcome, is_pusher, time.perf_counter() - start)


def summarize(b: Batch) -> dict[str, float]:
    """One results row: mean survivors, their distribution, survival by position, causes, pusher gain."""
    mean, se = b.mean_survivors()
    row = {"mean_survivors": mean, "se": se}
    row |= {f"p_{k}_survivors": float(x) for k, x in enumerate(b.survivors_distribution())}
    row |= {f"survival_pos_{k + 1}": float(x) for k, x in enumerate(b.survival_by_position())}
    row |= {f"share_{name}": x for name, x in b.cause_shares().items()}
    row["pusher_survival"], row["nonpusher_survival"] = b.pusher_survival()
    return row


# ---- Sweeps -------------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def expand(cfg: dict) -> list[Params]:
    """Settings of one experiment: base, crossed with the grid, crossed with the default plus each [[vary]] value."""
    defaults = Params()
    cast = lambda key, value: type(getattr(defaults, key))(value)  # TOML 600 -> 600.0 for float fields
    variants = [{}]
    for block in cfg.get("vary", []):
        keys = list(block)
        variants += [dict(zip(keys, values)) for values in zip(*(block[k] for k in keys))]
    grid = cfg.get("grid", {})
    settings = []
    for combo in itertools.product(*grid.values()):
        for variant in variants:
            changes = cfg.get("base", {}) | dict(zip(grid, combo)) | variant
            settings.append(Params(**{k: cast(k, v) for k, v in changes.items()}))
    return settings


def _run_setting(job: tuple[Params, int, int]) -> dict[str, float]:
    p, games, seed = job
    return summarize(run_batch(p, games, seed))


def _git_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    git = lambda *args: subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def run_sweep(config: str | Path, out_dir: str | Path = "results", workers: int | None = None,
              games: int | None = None) -> Path:
    """Run every setting of one config with the same seed (same bridges) and save
    results/<name>.csv (one row per setting) and results/<name>.json (provenance)."""
    config = Path(config)
    cfg = load_config(config)
    games = games or cfg["games"]
    settings = expand(cfg)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_run_setting, [(p, games, cfg["seed"]) for p in settings]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{config.stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["setting", *asdict(settings[0]), "games", "seed", *rows[0]]
        w.writerow(header)
        for i, (p, row) in enumerate(zip(settings, rows)):
            w.writerow([i, *asdict(p).values(), games, cfg["seed"], *row.values()])
    meta = {
        "config": config.as_posix(), "settings": len(settings), "games": games, "seed": cfg["seed"],
        **_git_state(), "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.perf_counter() - start, 1), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
    }
    (out_dir / f"{config.stem}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a batch of Glass Bridge games and print a summary, "
                                             "or run sweeps: pass one or more configs/*.toml files.")
    ap.add_argument("configs", nargs="*", help="sweep configs to run (writes results/<name>.csv and .json)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=None, help="processes for sweeps (default: all cores)")
    ap.add_argument("--games", type=int, default=None, help="games per setting (default: 10,000, or the config's)")
    ap.add_argument("--seed", type=int, default=1)
    for name, value in asdict(Params()).items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = vars(ap.parse_args())
    configs, out, workers, games, seed = (args.pop(k) for k in ("configs", "out", "workers", "games", "seed"))
    if configs:
        for c in configs:
            print(f"{c} -> {run_sweep(c, out, workers, games)}")
        return
    games = games or 10_000
    b = run_batch(Params(**args), games, seed)
    mean, se = b.mean_survivors()
    print(f"{games} games, seed {seed}, {b.runtime_s:.1f} s")
    print(f"mean survivors {mean:.2f} ± {se:.2f}")
    print("survival by position", np.round(b.survival_by_position(), 3))
    print("causes", {k: round(v, 3) for k, v in b.cause_shares().items()})
    print("pusher / non-pusher survival", tuple(round(x, 3) for x in b.pusher_survival()))


if __name__ == "__main__":
    main()
