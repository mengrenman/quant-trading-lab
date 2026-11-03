"""Tests for estimating the Mike and Farmer (2008) model's parameters (lob/calibration.py)."""

import math

import numpy as np
import pandas as pd
import pytest

from lob.analysis.stylized_facts import compute_acf
from lob.calibration import (
    calibrate_mf2008, estimate_sign_hurst, farima_acf, fit_rounded_t, relative_offsets, sample_acf,
)
from lob.models.long_memory import LongMemorySigns
from lob.models.mike_farmer import MikeFarmer2008Model
from lob.order import reset_order_id_counter
from lob.simulation import Simulation, SimulationConfig


def test_farima_and_sample_autocorrelations():
    d = 0.25
    acf = farima_acf(d, 3)
    assert acf[0] == pytest.approx(d / (1 - d))
    assert acf[1] == pytest.approx(acf[0] * (1 + d) / (2 - d))
    x = np.random.default_rng(0).standard_normal(3_000).cumsum() % 5
    np.testing.assert_allclose(sample_acf(x, 30), compute_acf(x, 30)[1:])


def test_rounded_t_fit_recovers_its_parameters():
    rng = np.random.default_rng(1)
    offsets = np.round(0.8 + 2.0 * rng.standard_t(1.7, size=60_000)).astype(int)
    fit = fit_rounded_t(offsets)
    assert fit["nu"] == pytest.approx(1.7, abs=0.08)
    assert fit["loc"] == pytest.approx(0.8, abs=0.05)
    assert fit["scale"] == pytest.approx(2.0, abs=0.06)


def test_sign_hurst_is_recovered_from_the_sign_generator():
    rng = np.random.default_rng(2)
    generator = LongMemorySigns(hurst=0.8)
    signs = np.array([generator.next(rng) for _ in range(200_000)], dtype=float)
    assert estimate_sign_hurst(signs) == pytest.approx(0.8, abs=0.03)
    assert estimate_sign_hurst(rng.choice([-1.0, 1.0], size=200_000)) == pytest.approx(0.5, abs=0.02)


def test_relative_offsets_use_the_same_side_quote_before_the_order():
    nan = math.nan
    log = pd.DataFrame({
        "event_type": ["limit", "limit", "limit", "cancel", "limit"],
        "side": ["bid", "ask", "bid", "bid", "ask"],
        "price": [100, 104, 98, 0, 103],
        # quotes after each event
        "best_bid": [100, 100, 100, 100, 100],
        "best_ask": [nan, 104, 104, 104, 103],
    })
    # The first order has no quote before it; the bid at 98 is 2 below the bid of 100;
    # the ask at 103 is 1 inside the ask of 104 (offset -1). The ask at 104 came while no ask was quoted.
    np.testing.assert_array_equal(relative_offsets(log), [2, -1])


def test_calibration_recovers_the_parameters_of_a_simulated_run():
    true = dict(lam=1.0, cancel_rate=0.02, imbalance_offset=0.5, hurst=0.7, nu=2.0, loc=0.5, scale=2.5)
    reset_order_id_counter()
    log = Simulation(model=MikeFarmer2008Model(initial_mid=10000, **true),
                     config=SimulationConfig(n_events=150_000, warmup_events=0, seed=5)).run(progress=False)
    est = calibrate_mf2008(log, initial_mid=10000)
    assert est["lam"] == pytest.approx(true["lam"], rel=0.03)
    assert est["cancel_rate"] == pytest.approx(true["cancel_rate"], rel=0.08)
    assert est["imbalance_offset"] == pytest.approx(true["imbalance_offset"], abs=0.15)
    assert est["hurst"] == pytest.approx(true["hurst"], abs=0.03)
    assert est["nu"] == pytest.approx(true["nu"], abs=0.15)
    assert est["loc"] == pytest.approx(true["loc"], abs=0.1)
    assert est["scale"] == pytest.approx(true["scale"], abs=0.1)
    assert est["lot_size"] == 1
