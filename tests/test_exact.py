import numpy as np
import pytest

from glass_bridge import exact


def test_binomial_matches_markov_chain():
    for n_steps, n_players in [(18, 16), (5, 3), (1, 1), (10, 12)]:
        np.testing.assert_allclose(
            exact.survival_by_position(n_steps, n_players),
            exact.markov_survival_by_position(n_steps, n_players),
            atol=1e-12,
        )


def test_default_game_expects_about_seven_survivors():
    assert exact.expected_survivors() == pytest.approx(7.0, abs=0.05)


def test_distribution_sums_to_one_and_matches_mean():
    dist = exact.survivors_distribution()
    assert dist.sum() == pytest.approx(1.0)
    assert (np.arange(len(dist)) * dist).sum() == pytest.approx(exact.expected_survivors())


def test_tiny_bridges():
    assert exact.survival_by_position(1, 1)[0] == pytest.approx(0.5)
    assert exact.survival_by_position(2, 1)[0] == pytest.approx(0.25)
    assert exact.survival_by_position(0, 3).tolist() == [1.0, 1.0, 1.0]
    # With more players than steps, the last ones always survive
    assert exact.survival_by_position(2, 4)[-2:].tolist() == [1.0, 1.0]
