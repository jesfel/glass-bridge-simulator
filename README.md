# Glass Bridge Simulator

A Monte Carlo simulator of the Glass Stepping Stones game from *Squid Game*, built to answer a question:
**the math says ~7 of 16 players survive. In the show, 3 do. How much does being human cost?**

**Articles:** [The Deadliest Thing in Squid Game Wasn't the Glass. It Was Hesitation.](https://medium.com/@jesfel/the-deadliest-thing-in-squid-game-wasnt-the-glass-it-was-hesitation-bc64ab3df9f5) (Medium, October 2026)

**How it works, in full:** [METHODOLOGY.md](METHODOLOGY.md) (the model, hesitation and pushing rules, random
numbers, validation, every experiment and output column, limitations).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # macOS/Linux: .venv/bin/python
.venv/Scripts/python -m pytest
.venv/Scripts/python -m glass_bridge.experiments --games 10000 --n-pushers 5 --beta 1.5
```

```python
from glass_bridge import Params, run_batch, trace

batch = run_batch(Params(n_pushers=5), n_games=10_000, seed=1)
batch.mean_survivors()        # (mean, standard error)
batch.survival_by_position()

game = trace(Params(n_pushers=5), seed=3)   # one game, every event logged
```

## The rules

- **Bridge:** 18 steps; each has a safe and a breaking pane (50/50). 16 players in a fixed order.
- **Follow the leader:** players walk in single file behind the leader, so nobody needs to remember the path.
  Only the leader faces an unknown pane.
- **Clock:** 16 minutes. Anyone still on the bridge at 960 s is eliminated.
- **Hesitation:** at an unknown pane the leader takes a lognormal decision delay (median 20 s, σ = 1).
  `delay="gamma"` swaps in a gamma delay with the same median and mean, as a robustness check.
  Pressure scales the median by `exp(β × fraction of clock used)`: β < 0 players rush, β > 0 they freeze.
  Walking a known step takes 3 s.
- **Pushing:** a fixed number of players are pushers, placed at random. From minute 7 on, a pusher
  directly behind a leader who has stalled for more than 15 s pushes:
  - **lands** (60%): the victim goes over; **drag** (20%): both go over; **backfire** (20%): the pusher falls.
  - Whoever goes over touches the next pane with probability `p_contact` (default 50%), revealing it. A pushed
    victim who lands on the safe pane survives one step ahead. Without contact, nothing is revealed.
- **Ideal play** (`BASELINE`: no clock, no pushers) reproduces the exact odds; see `exact.py` and the tests.

Every parameter lives in `Params` (`src/glass_bridge/params.py`).

## Experiments

Each file in `configs/` is one experiment: a seed, games per setting, a `base`, a `grid` (every
combination is run) and optional `[[vary]]` blocks (one parameter at a time around the defaults).

```bash
.venv/Scripts/python -m glass_bridge.experiments configs/*.toml
```

This writes `results/<experiment>.csv` (one row per setting: parameters, mean survivors and standard
error, the survivor distribution, survival by position, causes of death, pusher vs non-pusher survival)
and `results/<experiment>.json` (seed, git commit, versions, run time). Re-running gives the same numbers.

**Same bridges across scenarios.** Every random number a game needs is drawn up front from its own
stream: bridge, leader's guesses, delays, pusher order, push struggle, pane contact. Game *g* under a
seed sees the same bridge, guesses and delays in every scenario, and adding a pusher keeps the
others in place, so differences between settings come from the rules, not from luck.

## Layout

```text
src/glass_bridge/
  params.py       parameters and outcome codes
  engine.py       one game, with an event log
  draws.py        random numbers, drawn up front per stream (same bridges across scenarios)
  exact.py        exact baseline odds (binomial and Markov chain)
  experiments.py  batch runner, summaries, sweeps, command line
  figures.py      article figures from results/ (needs the `figures` extra)
METHODOLOGY.md    the model and experiments in full
configs/          one TOML file per experiment
results/          sweep summaries (CSV) and provenance (JSON)
figures/          figures (PNG); figures/article/ holds the phone-readable versions used in the articles
tests/            exact-odds validation, rule checks, reproducibility
```

## Prior work

The ideal-play odds are well known. This project reproduces them as validation and builds on:

- Eric Pham-Hung, [Squid Game: a data scientist's take on surviving the glass bridge](https://medium.com/@ericphamhung_76823/squid-game-a-data-scientists-take-on-surviving-the-glass-bridge-f143f739ebb5) (2021)
- John Paul Helveston, [Simulating the Squid Game bridge scene in R](https://www.jhelvy.com/blog/2021-10-19-monte-carlo-bridge-game/) (2021; step delays and a clock)
- Aster Hu, [Squid Game: The probability of Glass Bridge](https://www.asterhu.com/post/2021-11-04-squid-game-probability/) (2021)
- Henk Tijms, [Markov chain treatment](https://kraket.nl/f/files/download/sector/2021-2022/tutorialmarkovsector.pdf)
