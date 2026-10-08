import math

import numpy as np
import pytest

from glass_bridge import Params, run_batch, trace
from glass_bridge.draws import STREAMS, delay_multiplier, draw_batch


def test_same_bridges_guesses_and_delays_across_scenarios():
    a = draw_batch(Params(), 500, seed=1)
    b = draw_batch(Params(n_pushers=5, beta=1.5, p_contact=1.0, clock=600.0), 500, seed=1)
    for name in STREAMS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_rows_do_not_depend_on_batch_size():
    small, big = draw_batch(Params(), 10, seed=2), draw_batch(Params(), 1_000, seed=2)
    for name in STREAMS:
        np.testing.assert_array_equal(getattr(small, name), getattr(big, name)[:10])


def test_trace_replays_a_batch_game():
    p = Params(n_pushers=5, beta=1.5)
    batch = run_batch(p, 50, seed=3)
    for g in (0, 17, 49):
        assert trace(p, 3, game=g).outcome == list(batch.outcome[g])


def test_pusher_sets_are_nested():
    for n in range(16):
        small = run_batch(Params(n_pushers=n), 200, seed=4).is_pusher
        big = run_batch(Params(n_pushers=n + 1), 200, seed=4).is_pusher
        assert (big | ~small).all() and (big.sum(axis=1) == n + 1).all()


@pytest.mark.parametrize("dist", ["lognormal", "gamma"])
def test_delay_has_median_1_and_lognormal_mean(dist):
    sigma = 1.0
    m = delay_multiplier(np.random.default_rng(5).standard_normal(400_000), sigma, dist)
    assert np.median(m) == pytest.approx(1.0, abs=0.01)
    assert m.mean() == pytest.approx(math.exp(sigma**2 / 2), rel=0.02)


def test_gamma_scenario_runs_and_differs():
    a = run_batch(Params(), 2_000, seed=6).outcome
    b = run_batch(Params(delay="gamma"), 2_000, seed=6).outcome
    assert not np.array_equal(a, b)
    with pytest.raises(ValueError):
        Params(delay="weibull")
