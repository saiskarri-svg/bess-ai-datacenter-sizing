from dataclasses import replace

import numpy as np
import pytest

from bess_model import ModelInputs, evaluate, recommend_system, size_backup, size_grid_cap, size_smoothing
from bess_model.load import peak_it_mw
from bess_model.sizing import ramp_limited_grid


@pytest.fixture
def m():
    return ModelInputs.default()


def test_peak_it_counts_whole_racks(m):
    # 50,000 / 72 = 694.4 -> 695 racks * 132 kW * 1.10 for networking/storage
    assert peak_it_mw(m.cluster) == pytest.approx(695 * 0.132 * 1.10)


def test_backup_energy_covers_hours_plus_losses(m):
    r = size_backup(m)
    assert r.energy_mwh > r.power_mw * m.bess.backup_hours


def test_ramp_limited_grid_never_exceeds_ramp():
    load = np.array([0, 100, 100, 0, 0, 100], dtype=float)
    grid = ramp_limited_grid(load, 5.0)
    assert np.all(np.abs(np.diff(grid)) <= 5.0 + 1e-9)


def test_smoothing_battery_covers_gap(m):
    r, prof = size_smoothing(m)
    assert r.power_mw == pytest.approx(prof["bess_mw"].abs().max())
    assert np.allclose(prof["grid_mw"] + prof["bess_mw"], prof["facility_mw"])


def test_grid_cap_not_needed_when_limit_is_high(m):
    high = replace(m, grid=replace(m.grid, interconnection_limit_mw=1_000))
    r, _, feasible = size_grid_cap(high)
    assert r.power_mw == 0 and feasible


def test_grid_cap_infeasible_when_average_load_exceeds_limit(m):
    low = replace(m, grid=replace(m.grid, interconnection_limit_mw=50))
    _, _, feasible = size_grid_cap(low)
    assert not feasible


def test_backup_does_not_stack_with_daily_jobs(m):
    s = recommend_system(m)
    backup = size_backup(m)
    assert s.power_mw == pytest.approx(backup.power_mw)


def test_cheaper_batteries_raise_npv(m):
    base = evaluate(m, recommend_system(m)).npv
    cheap = replace(m, econ=replace(m.econ, energy_cost_per_kwh=150))
    assert evaluate(cheap, recommend_system(cheap)).npv > base


def test_arbitrage_does_not_use_reserved_backup(m):
    no_trade = replace(m, econ=replace(m.econ, backup_energy_tradable=0.0))
    e = evaluate(no_trade, recommend_system(no_trade))
    assert e.annual_arbitrage < evaluate(m, recommend_system(m)).annual_arbitrage


def test_c_rate_limit_sets_minimum_energy(m):
    smooth_only = replace(m, bess=replace(m.bess, use_backup=False, use_grid_cap=False))
    s = recommend_system(smooth_only)
    assert s.power_mw / s.energy_mwh <= smooth_only.bess.max_c_rate + 1e-9
