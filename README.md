# Glass Bridge Simulator

A Monte Carlo simulator of the Glass Stepping Stones game from *Squid Game*, built to answer a question:
**the math says ~7 of 16 players survive. In the show, 3 do. How much does being human cost?**

> Work in progress. Private until the first article is published.

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
  Pressure scales the median by `exp(β × fraction of clock used)`: β < 0 players rush, β > 0 they freeze.
  Walking a known step takes 3 s.
- **Pushing:** a fixed number of players are pushers, placed at random. From minute 7 on, a pusher
  directly behind a leader who has stalled for more than 15 s pushes:
  - **lands** (60%): the victim goes over; **drag** (20%): both go over; **backfire** (20%): the pusher falls.
  - Whoever goes over touches the next pane with probability `p_contact` (default 50%), revealing it. A pushed
    victim who lands on the safe pane survives one step ahead. Without contact, nothing is revealed.
- **Ideal play** (`BASELINE`: no clock, no pushers) reproduces the exact odds; see `exact.py` and the tests.

Every parameter lives in `Params` (`src/glass_bridge/params.py`).

## Layout

```text
src/glass_bridge/
  params.py       parameters and outcome codes
  engine.py       one game, seeded, with an event log
  exact.py        exact baseline odds (binomial and Markov chain)
  experiments.py  batch runner, summaries, command line
tests/            exact-odds validation and rule checks
```

## Prior work

The ideal-play odds are well known. This project reproduces them as validation and builds on:
Eric Pham-Hung (Medium, 2021), John Paul Helveston (2021, step delays and a clock), Aster Hu (2021),
Henk Tijms (Markov chain treatment) and others, credited in the articles.
