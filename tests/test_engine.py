import math

import numpy as np
import pytest

from glass_bridge import BASELINE, Outcome, Params, exact, run_batch, trace

N_GAMES = 40_000


def assert_matches_exact(batch, n_steps=18, n_players=16, z=4.0):
    """Simulated survival by position within z standard errors of the exact odds."""
    expected = exact.survival_by_position(n_steps, n_players)
    se = np.sqrt(np.maximum(expected * (1 - expected), 1e-12) / batch.n_games)
    observed = batch.survival_by_position()
    assert np.all(np.abs(observed - expected) <= z * se + 1e-9), (observed, expected)


@pytest.fixture(scope="module")
def baseline():
    return run_batch(BASELINE, N_GAMES, seed=1)


def test_baseline_survival_by_position_matches_exact(baseline):
    assert_matches_exact(baseline)


def test_baseline_mean_and_distribution_match_exact(baseline):
    mean, se = baseline.mean_survivors()
    assert abs(mean - exact.expected_survivors()) <= 4 * se
    dist, ref = baseline.survivors_distribution(), exact.survivors_distribution()
    se_bins = np.sqrt(np.maximum(ref * (1 - ref), 1e-12) / N_GAMES)
    assert np.all(np.abs(dist - ref) <= 4 * se_bins + 1e-9)


def test_baseline_only_glass_deaths(baseline):
    shares = baseline.cause_shares()
    assert shares["survived"] + shares["glass"] == pytest.approx(1.0)


def test_same_seed_same_result():
    p = Params(n_pushers=5)
    a, b = run_batch(p, 500, seed=7), run_batch(p, 500, seed=7)
    np.testing.assert_array_equal(a.outcome, b.outcome)
    assert not np.array_equal(a.outcome, run_batch(p, 500, seed=8).outcome)
    assert trace(p, 3).events == trace(p, 3).events


def test_pushes_that_always_land_on_a_pane_do_not_change_survival():
    """With no clock, a landed push with contact is just a forced 50/50 jump."""
    p = BASELINE.with_(n_pushers=16, t_start=0.0, p_land=1.0, p_drag=0.0, p_backfire=0.0, p_contact=1.0)
    batch = run_batch(p, N_GAMES, seed=2)
    assert batch.cause_shares()["pushed"] > 0
    assert_matches_exact(batch)


def test_tiny_clock_means_nobody_survives():
    batch = run_batch(Params(clock=1.0), 200, seed=3)
    assert (batch.outcome == Outcome.TIMEOUT).all()


def test_tiny_bridges():
    assert run_batch(BASELINE.with_(n_steps=0, n_players=3), 10, seed=4).survived.all()
    one = run_batch(BASELINE.with_(n_steps=1, n_players=1), 20_000, seed=5)
    assert one.survival_by_position()[0] == pytest.approx(0.5, abs=4 * math.sqrt(0.25 / 20_000))


def test_no_pushing_before_t_start():
    p = Params(n_pushers=16, t_start=10_000.0, clock=math.inf)
    shares = run_batch(p, 2_000, seed=6).cause_shares()
    assert shares["pushed"] == shares["drag"] == shares["backfire"] == 0.0


def test_every_player_ends_exactly_once():
    for seed in range(200):
        g = trace(Params(n_pushers=6, beta=1.5), seed)
        assert all(o is not None for o in g.outcome)
        terminal = [e for e in g.events if e.kind in {"exit", "glass", "pushed", "drag", "backfire", "timeout"}]
        assert sorted(e.player for e in terminal) == list(range(16))
        assert all(0 <= t <= 960.0 for t in g.time)


def test_push_outcomes_all_occur():
    shares = run_batch(Params(n_pushers=8, t_start=0.0, beta=1.5), 3_000, seed=9).cause_shares()
    for cause in ("pushed", "drag", "backfire", "timeout", "glass", "survived"):
        assert shares[cause] > 0, cause


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        Params(p_land=0.5, p_drag=0.2, p_backfire=0.2)
    with pytest.raises(ValueError):
        Params(n_pushers=17)
