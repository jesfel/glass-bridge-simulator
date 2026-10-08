import csv
import json
from pathlib import Path

import pytest

from glass_bridge import Params
from glass_bridge.experiments import expand, load_config, run_sweep

CONFIGS = sorted(Path(__file__).parents[1].glob("configs/*.toml"))

TINY = """
seed = 11
games = 300

[base]
n_pushers = 4

[grid]
beta = [0.0, 1.5]

[[vary]]
d0 = [10, 30]

[[vary]]
p_land = [1.0]
p_drag = [0.0]
p_backfire = [0.0]
"""


@pytest.fixture
def tiny(tmp_path):
    path = tmp_path / "tiny.toml"
    path.write_text(TINY)
    return path


def test_expand_crosses_grid_with_default_and_each_varied_value(tiny):
    settings = expand(load_config(tiny))
    assert len(settings) == 2 * (1 + 2 + 1)
    assert Params(n_pushers=4, beta=1.5, d0=30.0) in settings
    assert Params(n_pushers=4, p_land=1.0, p_drag=0.0, p_backfire=0.0) in settings
    assert all(isinstance(p.d0, float) for p in settings)


def test_rerunning_a_sweep_reproduces_the_numbers(tiny, tmp_path):
    a = run_sweep(tiny, tmp_path / "a", workers=2).read_text()
    b = run_sweep(tiny, tmp_path / "b", workers=1).read_text()
    assert a == b
    rows = list(csv.DictReader(a.splitlines()))
    assert len(rows) == 8 and rows[0]["seed"] == "11" and rows[0]["games"] == "300"
    meta = json.loads((tmp_path / "a" / "tiny.json").read_text())
    assert meta["settings"] == 8 and "commit" in meta


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_config_expands_to_valid_settings(path):
    cfg = load_config(path)
    assert cfg["games"] == 100_000 and expand(cfg)
