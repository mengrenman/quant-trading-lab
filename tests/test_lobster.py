"""Tests for reading LOBSTER data and estimating the Mike and Farmer (2008) model from it (lob/lobster.py)."""

import math

import numpy as np
import pandas as pd
import pytest

from lob.lobster import (
    DAY_START, MESSAGE_COLUMNS, calibrate_lobster, fit_rounded_t_censored, lobster_orders, order_offsets,
    read_lobster, save_lobster, simulated_lobster,
)
from lob.models.mike_farmer import MikeFarmer2008Model
from lob.order import reset_order_id_counter

TRUE = dict(lam=1.0, cancel_rate=0.02, imbalance_offset=0.5, hurst=0.7, nu=2.0, loc=0.5, scale=2.5)


def make_model():
    return MikeFarmer2008Model(initial_mid=10000, **TRUE)


def quotes(*rows):
    """A two-level-free order-book file: (bid, ask) in cents after each message."""
    book = pd.DataFrame([[ask * 100, 10, bid * 100, 10] for bid, ask in rows],
                        columns=["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"])
    return book.astype(float)


def test_messages_become_orders():
    t = DAY_START
    messages = pd.DataFrame([
        (t + 0.5, 3, 1, 1, 9_800, 1),   # a bid placed before the file began leaves
        (t + 1, 1, 11, 5, 9_900, 1),    # a buy at 99 rests
        (t + 2, 4, 21, 3, 10_100, -1),  # a buy hits the asks at 101 and 102 and executes in full
        (t + 2, 4, 22, 2, 10_200, -1),
        (t + 3, 4, 11, 5, 9_900, 1),    # a sell takes the bid at 99, and 4 lots of it rest at 99
        (t + 3, 1, 31, 4, 9_900, -1),
        (t + 4, 5, 0, 7, 9_850, -1),    # a hidden sell order at the midpoint executed: a buy of 7
        (t + 5, 2, 31, 1, 9_900, -1),   # a partial, then a full, cancellation
        (t + 6, 3, 31, 3, 9_900, -1),
    ], columns=MESSAGE_COLUMNS)
    book = quotes((99, 101), (99, 101), (99, 102), (99, 102), (98, 102), (98, 99), (98, 99), (98, 99), (98, 102))
    orders = lobster_orders(messages, book)
    assert orders[["side", "size", "executed", "order_id"]].values.tolist() == [
        ["bid", 5, 0, 11], ["bid", 5, 5, -1], ["ask", 9, 5, 31], ["bid", 7, 7, -1]]
    assert orders["worst"].tolist()[1:3] == [102, 99]  # the worst price each aggressive order traded at
    assert math.isnan(orders["price"][1]) and orders["price"][2] == 99  # executed in full: no price
    exact, bounds = order_offsets(orders)
    # The resting buy is at the bid before it (offset 0); the sell rests at 99, 3 inside the ask of
    # 102 before it (-3). The buy that executed in full was at least 3 ticks more aggressive than
    # the bid of 99 (99 - 102); the buy of the hidden order at least 0.5 (98 - 98.5), so 1 in whole ticks.
    assert exact.tolist() == [0, -3]
    assert bounds.tolist() == [-3, -1]


def test_the_files_round_trip(tmp_path):
    reset_order_id_counter()
    messages, book = simulated_lobster(make_model, 2_000, seed=1, levels=3)
    message_file, orderbook_file = save_lobster(messages, book, tmp_path)
    assert message_file.name == "SIM_2012-06-21_34200000_57600000_message_3.csv"
    again, again_book = read_lobster(message_file, orderbook_file)
    pd.testing.assert_frame_equal(again, messages, check_dtype=False)
    pd.testing.assert_frame_equal(again_book, book, check_dtype=False)  # empty levels come back as NaN
    assert again_book.isna().any().any()  # the book starts empty, so some levels are


def test_the_censored_price_fit_recovers_its_parameters():
    rng = np.random.default_rng(4)
    draws = np.round(0.5 + 2.5 * rng.standard_t(2.0, size=80_000)).astype(int)
    # As in a book: how aggressive an order must be to execute in full depends on the book, not on
    # the order. Those that are show only that bound; the rest show their offset.
    limits = -2 - rng.integers(0, 6, size=draws.size)
    full = draws <= limits
    fit = fit_rounded_t_censored(draws[~full], limits[full])
    assert 0.1 < full.mean() < 0.3
    assert fit["nu"] == pytest.approx(2.0, abs=0.1)
    assert fit["loc"] == pytest.approx(0.5, abs=0.05)
    assert fit["scale"] == pytest.approx(2.5, abs=0.06)


def test_every_parameter_is_recovered_from_simulated_lobster_files():
    reset_order_id_counter()
    messages, book = simulated_lobster(make_model, 150_000, seed=5)
    est = calibrate_lobster(messages, book)
    assert est["lam"] == pytest.approx(TRUE["lam"], rel=0.03)  # here per unit of model time
    assert est["cancel_rate"] == pytest.approx(TRUE["cancel_rate"], rel=0.08)
    assert est["imbalance_offset"] == pytest.approx(TRUE["imbalance_offset"], abs=0.15)
    assert est["hurst"] == pytest.approx(TRUE["hurst"], abs=0.03)
    assert est["nu"] == pytest.approx(TRUE["nu"], abs=0.15)
    assert est["loc"] == pytest.approx(TRUE["loc"], abs=0.1)
    assert est["scale"] == pytest.approx(TRUE["scale"], abs=0.1)
    assert est["lot_size"] == 1
    assert 0.15 < est["n_executed_in_full"] / est["n_orders"] < 0.4  # a good share of prices hidden
    assert est["n_cancels_skipped"] == 0 and est["last_unknown"] is None


def test_a_file_that_starts_on_a_full_book_needs_its_burn_in():
    reset_order_id_counter()
    messages, book = simulated_lobster(make_model, 60_000, seed=6, skip_events=40_000)
    early = calibrate_lobster(messages, book, burn_in=0.0)
    assert early["n_cancels_skipped"] > 0  # orders placed before the file began
    late = calibrate_lobster(messages, book, burn_in=math.ceil(early["last_unknown"]) + 1)
    assert late["n_cancels_skipped"] == 0
    assert late["cancel_rate"] == pytest.approx(TRUE["cancel_rate"], rel=0.12)
    assert late["imbalance_offset"] == pytest.approx(TRUE["imbalance_offset"], abs=0.3)
