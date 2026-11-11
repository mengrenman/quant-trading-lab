"""Tests for the zero-intelligence model with informed traders (lob/models/informed.py).

Each statistical test compares the model with a computation made another way: theory (Gaussian and compound
Poisson moments, Poisson counts), hand-set quotes, or the event log the simulation harness keeps.
"""

import math
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from lob.book import OrderBook
from lob.matching import PriceTimeEngine
from lob.models import InformedZIModel as ExportedModel
from lob.models.informed import InformedZIModel
from lob.models.zero_intelligence import SmithFarmerModel
from lob.order import OrderType, Side, limit_order, reset_order_id_counter
from lob.simulation import Simulation, SimulationConfig

# The dashboard's zero-intelligence defaults and its mid-price.
ZI = dict(mu=6.0, lam=1.0, delta=0.03, price_range=15, lot_size=5, initial_mid=100)


@pytest.fixture(autouse=True)
def reset_ids():
    reset_order_id_counter()
    yield
    reset_order_id_counter()


def _seeded_book(mid=100, half_spread=1, levels=16, volume=25):
    """The book the dashboard starts from: deep symmetric levels around the mid."""
    book = OrderBook(tick_size=1, lot_size=1)
    for i in range(levels):
        book.add_order(limit_order(mid - half_spread - i, volume + 5 * i, 0.0, Side.BID))
        book.add_order(limit_order(mid + half_spread + i, volume + 5 * i, 0.0, Side.ASK))
    return book


def _frozen_book(bid, ask, n_each=5):
    """A book the test never trades against: only its quotes and its number of orders matter."""
    book = OrderBook()
    for _ in range(n_each):
        book.add_order(limit_order(bid, 5, 0.0, Side.BID))
        book.add_order(limit_order(ask, 5, 0.0, Side.ASK))
    return book


def _simulate(model, n_events, seed, warmup=0):
    reset_order_id_counter()
    sim = Simulation(model=model, book=_seeded_book(), config=SimulationConfig(
        n_events=n_events, warmup_events=warmup, seed=seed))
    return sim, sim.run(progress=False)


def _drive(model, n_events, seed, on_order=None, mid=100):
    """The simulation loop without its log: each order the model returns, then the matching engine."""
    reset_order_id_counter()
    book, engine, rng = _seeded_book(mid=mid), PriceTimeEngine(), np.random.default_rng(seed)
    t = 0.0
    for _ in range(n_events):
        order = model.generate_event(book, t, rng)
        if order is None:
            continue
        t = order.timestamp
        if on_order is not None:
            on_order(order, book)
        engine.process(book, order)
    return book


def test_the_model_is_exported_from_the_package():
    assert ExportedModel is InformedZIModel


# ---- The latent value ----

def test_value_diffusion_matches_theory_over_a_run():
    """Increments of V between the events of a run: mean 0, variance sigma_v^2 dt, no autocorrelation."""
    sigma = 0.5
    model = InformedZIModel(sigma_v=sigma, jump_rate=0.0, kappa=0.0, record_path=True, **ZI)
    _, df = _simulate(model, 100_000, seed=5)
    times = np.concatenate(([0.0], df["time"].to_numpy()))
    values = np.concatenate(([100.0], model.value_at(df["time"].to_numpy())))
    z = np.diff(values) / (sigma * np.sqrt(np.diff(times)))

    n = len(z)
    assert np.all(np.diff(times) > 0)
    assert abs(z.mean()) < 4 / math.sqrt(n)
    assert abs(z.var() - 1.0) < 4 * math.sqrt(2 / n)
    assert abs(np.corrcoef(z[:-1], z[1:])[0, 1]) < 4 / math.sqrt(n)
    assert model.value() == pytest.approx(values[-1])


def test_value_jumps_match_poisson_theory_over_a_run():
    """Jumps counted from the increments of V, and by the model, against rate times time; signs are fair."""
    rate, size = 0.2, 3.0
    model = InformedZIModel(sigma_v=0.3, jump_rate=rate, jump_size=size, kappa=0.0, record_path=True, **ZI)
    _, df = _simulate(model, 100_000, seed=6)
    values = np.concatenate(([100.0], model.value_at(df["time"].to_numpy())))
    steps = np.diff(values) / size  # a diffusion step is 0.05 here, a jump 3: they do not mix
    jumps_seen = np.rint(steps)
    n_seen = np.abs(jumps_seen).sum()

    expected = rate * df["time"].iloc[-1]
    sd = math.sqrt(expected)
    assert expected > 300
    assert abs(n_seen - expected) < 4 * sd
    assert abs(model.jumps - expected) < 4 * sd
    # Only a pair of opposite jumps inside one wait is invisible in the increments.
    assert 0 <= model.jumps - n_seen <= 10
    assert abs(jumps_seen.sum()) < 4 * math.sqrt(n_seen)
    assert np.abs(steps - jumps_seen).max() < 0.5  # what is left of a step after its jumps is diffusion


@pytest.mark.parametrize("steps", [1, 4, 32])
def test_value_advance_is_exact_for_any_split_of_the_time(steps):
    """V(T) - V(0) has mean 0 and variance (sigma^2 + rate size^2) T whether it is advanced in 1 step or many."""
    sigma, rate, size, horizon, reps = 0.4, 0.3, 2.0, 4.0, 10_000
    changes = np.empty(reps)
    for r in range(reps):
        model = InformedZIModel(sigma_v=sigma, jump_rate=rate, jump_size=size, value_seed=1000 + r, **ZI)
        for _ in range(steps):
            model._advance_value(horizon / steps)
        changes[r] = model.value() - model.initial_mid

    variance = (sigma**2 + rate * size**2) * horizon
    fourth_cumulant = rate * horizon * size**4  # the Gaussian's is 0
    se_mean = math.sqrt(variance / reps)
    se_var = math.sqrt((fourth_cumulant + 2 * variance**2) / reps)  # variance of a sample variance
    assert abs(changes.mean()) < 4 * se_mean
    assert abs(changes.var() - variance) < 4 * se_var


def test_value_starts_at_the_initial_mid_where_the_model_starts():
    """A model that takes over a book at a late clock does not diffuse across the time before it."""
    model = InformedZIModel(sigma_v=5.0, jump_rate=0.0, **ZI)
    assert model.value() == 100.0 and model.value_time is None
    rng = np.random.default_rng(1)
    model.generate_event(_seeded_book(), 1e6, rng)  # sd of a walk over 1e6 units: 5,000 ticks
    assert model.value_time > 1e6
    assert abs(model.value() - 100.0) < 25.0


def test_value_catches_up_with_a_clock_that_runs_ahead():
    model = InformedZIModel(sigma_v=1.0, jump_rate=0.0, value_seed=3, **ZI)
    book, rng = _seeded_book(), np.random.default_rng(2)
    model.generate_event(book, 0.0, rng)
    start = model.value_time
    model.generate_event(book, start + 400.0, rng)  # time passed while the model was not asked
    assert model.value_time > start + 400.0
    assert model.value_time == model._t


# ---- The floor under the value ----

@pytest.mark.parametrize("kwargs, floor", [
    (dict(), 20.0),                          # the dashboard's L of 15: above L + 1, with room
    (dict(price_range=40), 45.0),            # a wider placement window needs a higher floor
    (dict(price_range=200), 100.0),          # but never above where V starts
    (dict(initial_mid=10), 10.0),            # nor is V cut up to it at the first advance
])
def test_the_value_floor_is_above_the_placement_window_and_not_above_the_start(kwargs, floor):
    assert InformedZIModel(**{**ZI, **kwargs}).value_floor == floor
    assert InformedZIModel(**{**ZI, **kwargs}).value() >= floor


def test_the_floor_cuts_the_value_short_and_draws_no_random_number():
    """V is max(raw V, floor) after the same draws, so a run that never reaches the floor is unchanged."""
    cut = kept = 0
    zi = {**ZI, "initial_mid": 25}
    for seed in range(60):
        floored = InformedZIModel(sigma_v=5.0, jump_rate=0.5, jump_size=4.0, value_seed=seed, **zi)
        raw = InformedZIModel(sigma_v=5.0, jump_rate=0.5, jump_size=4.0, value_seed=seed, **zi)
        raw.value_floor = -math.inf
        floored._advance_value(4.0)
        raw._advance_value(4.0)
        assert floored.value() == max(raw.value(), 20.0)
        assert floored._vrng.bit_generator.state == raw._vrng.bit_generator.state
        assert floored.jumps == raw.jumps
        cut += raw.value() < 20.0
        kept += raw.value() >= 20.0
    assert cut > 5 and kept > 5  # both paths of the rule were taken


def test_a_run_that_never_reaches_the_floor_is_the_same_with_and_without_it():
    """At the dashboard's defaults, over a stretch where V stays far from 20: the floor changes nothing."""
    runs = []
    for floor in (None, -math.inf):
        model = InformedZIModel(record_path=True, **ZI)
        if floor is not None:
            model.value_floor = floor
        sim, df = _simulate(model, 40_000, seed=52)
        runs.append((model, sim, df))
    (m1, s1, d1), (m2, s2, d2) = runs
    assert m1.value_floor == 20.0
    assert np.min(np.frombuffer(m1._path[1])) > m1.value_floor + 10  # never touched
    pd.testing.assert_frame_equal(d1, d2)
    pd.testing.assert_frame_equal(s1.trades, s2.trades)
    assert np.array_equal(m1.informed_ids, m2.informed_ids) and len(m1.informed_ids) > 100
    assert np.array_equal(np.frombuffer(m1._path[1]), np.frombuffer(m2._path[1]))


def _run_near_the_floor(floor, sigma_v, kappa, seed, n_events=20_000):
    """A run from a mid of 25, five ticks above the floor, with the model's floor or (``-inf``) with none."""
    model = InformedZIModel(sigma_v=sigma_v, jump_rate=0.0, kappa=kappa, cap=10.0, record_path=True,
                            **{**ZI, "initial_mid": 25})
    if floor is not None:
        model.value_floor = floor
    stats = dict(informed_sells=0, sells_into_a_bid_at_the_floor=0, events_with_a_bid_at_the_floor=0,
                 events_with_v_at_the_floor=0)

    def watch(order, book):
        at_floor = book.best_bid is not None and book.best_bid <= 20.0
        stats["events_with_a_bid_at_the_floor"] += at_floor
        stats["events_with_v_at_the_floor"] += model.value() == 20.0
        if model.last_informed and order.side == Side.ASK:
            stats["informed_sells"] += 1
            stats["sells_into_a_bid_at_the_floor"] += at_floor

    _drive(model, n_events, seed=seed, mid=25, on_order=watch)
    stats["lowest_value"] = float(np.min(np.frombuffer(model._path[1])))
    return stats


def test_with_the_floor_the_value_never_goes_below_it_where_without_it_the_value_runs_off():
    floored = _run_near_the_floor(None, sigma_v=5.0, kappa=2.0, seed=61)
    free = _run_near_the_floor(-math.inf, sigma_v=5.0, kappa=2.0, seed=61)
    assert floored["lowest_value"] == 20.0 and floored["events_with_v_at_the_floor"] > 50  # it got there, and stayed
    assert free["lowest_value"] < 0  # the same draws: V crosses 20 and keeps going, through 0


def test_informed_sells_stop_when_the_bid_is_at_the_floor():
    """An informed sell needs a bid above V, and V is at or above 20: none is sent into a bid at or below 20. Without
    the floor V is below the book, and they go on selling into it."""
    floored = _run_near_the_floor(None, sigma_v=1.0, kappa=4.0, seed=66)
    free = _run_near_the_floor(-math.inf, sigma_v=1.0, kappa=4.0, seed=66)
    assert floored["events_with_a_bid_at_the_floor"] > 1_000 and floored["informed_sells"] > 100  # the book came down
    assert floored["sells_into_a_bid_at_the_floor"] == 0
    assert free["lowest_value"] < 10 and free["sells_into_a_bid_at_the_floor"] > 100


# ---- The informed rate ----

@pytest.mark.parametrize("bid, ask, value, kappa, cap, tick, buy, sell", [
    (99, 101, 100.0, 2.0, 30.0, 1, 0.0, 0.0),       # inside the touch: nothing
    (99, 101, 100.9, 2.0, 30.0, 1, 0.0, 0.0),       # still inside
    (99, 101, 101.0, 2.0, 30.0, 1, 0.0, 0.0),       # at the ask: a buy there makes nothing
    (99, 101, 99.0, 2.0, 30.0, 1, 0.0, 0.0),        # at the bid
    (99, 101, 102.5, 2.0, 30.0, 1, 3.0, 0.0),       # 1.5 ticks beyond the ask: kappa times 1.5
    (99, 101, 96.0, 0.5, 30.0, 1, 0.0, 1.5),        # 3 ticks below the bid
    (99, 101, 105.0, 2.0, 5.0, 1, 5.0, 0.0),        # 8 ticks of gap, capped
    (99, 101, 90.0, 2.0, 5.0, 1, 0.0, 5.0),         # the same on the sell side
    (99, 101, 100.0, 0.0, 30.0, 1, 0.0, 0.0),       # kappa 0: off
    (99, 101, 105.0, 2.0, 0.0, 1, 0.0, 0.0),        # cap 0: off
    (98, 102, 105.0, 1.0, 30.0, 2, 1.5, 0.0),       # a tick of 2: 3 of gap is 1.5 ticks
    (98, 102, 92.0, 1.0, 30.0, 2, 0.0, 3.0),        # and 6 of gap is 3 ticks
])
def test_informed_rate_rule_at_hand_set_quotes(bid, ask, value, kappa, cap, tick, buy, sell):
    book = OrderBook(tick_size=tick)
    book.add_order(limit_order(bid, 5, 0.0, Side.BID))
    book.add_order(limit_order(ask, 5, 0.0, Side.ASK))
    model = InformedZIModel(kappa=kappa, cap=cap, **ZI)
    assert model.informed_rates(book, value) == pytest.approx((buy, sell))


def test_informed_rate_is_zero_for_a_side_without_a_quote():
    model = InformedZIModel(kappa=2.0, cap=30.0, **ZI)
    only_bids, only_asks = OrderBook(), OrderBook()
    only_bids.add_order(limit_order(99, 5, 0.0, Side.BID))
    only_asks.add_order(limit_order(101, 5, 0.0, Side.ASK))
    assert model.informed_rates(only_bids, 120.0) == (0.0, 0.0)   # nothing to buy from
    assert model.informed_rates(only_bids, 90.0) == pytest.approx((0.0, 18.0))
    assert model.informed_rates(only_asks, 80.0) == (0.0, 0.0)    # nothing to sell to
    assert model.informed_rates(only_asks, 110.0) == pytest.approx((18.0, 0.0))
    assert model.informed_rates(OrderBook(), 120.0) == (0.0, 0.0)


@pytest.mark.parametrize("bid, ask, kappa, cap, buy, sell", [
    (99, 101, 2.0, 30.0, 0.0, 0.0),    # V = 100 inside the touch
    (100, 101, 2.0, 30.0, 0.0, 0.0),   # V on the bid
    (95, 98, 1.0, 10.0, 2.0, 0.0),     # V two ticks above the ask: 2 a unit
    (95, 98, 2.0, 3.0, 3.0, 0.0),      # kappa times the gap is 4: the cap
    (103, 105, 1.0, 10.0, 0.0, 3.0),   # V three ticks below the bid
])
def test_informed_rate_shows_in_the_events_of_a_frozen_book(bid, ask, kappa, cap, buy, sell):
    """With V fixed at 100 and the quotes fixed, informed orders are a Poisson stream at the rule's rate."""
    model = InformedZIModel(sigma_v=0.0, jump_rate=0.0, kappa=kappa, cap=cap, **ZI)
    book, rng = _frozen_book(bid, ask), np.random.default_rng(4)
    base = model._total_rate(book.num_orders())
    n, t, informed_sides = 60_000, 0.0, []
    for _ in range(n):
        order = model.generate_event(book, t, rng)
        t = order.timestamp
        if model.last_informed:
            informed_sides.append(order.side)

    total = buy + sell
    share = len(informed_sides) / n
    expected = total / (base + total)
    assert abs(share - expected) < 4 * math.sqrt(expected * (1 - expected) / n) + 1e-12
    assert t / n == pytest.approx(1 / (base + total), rel=4 / math.sqrt(n))  # the rate is added to the total
    assert set(informed_sides) <= ({Side.BID} if buy else {Side.ASK})
    assert model.value() == 100.0


def test_dropped_candidates_end_in_no_event_not_a_hang():
    """No other events, and V inside the touch: every candidate is dropped, and the call gives up."""
    model = InformedZIModel(sigma_v=0.0, jump_rate=0.0, kappa=2.0, cap=4.0, mu=0.0, lam=0.0, delta=0.0,
                            price_range=15, lot_size=5, initial_mid=100)
    model.MAX_DROPPED = 50
    book, rng = _frozen_book(99, 101), np.random.default_rng(0)
    assert model.generate_event(book, 0.0, rng) is None
    assert model._t == pytest.approx(50 / 4.0, rel=0.4)  # 50 waits at a rate of 4
    assert model.informed_lots == 0 and not model.last_informed
    # Beyond the quote every candidate is accepted.
    model = InformedZIModel(sigma_v=0.0, jump_rate=0.0, kappa=2.0, cap=4.0, mu=0.0, lam=0.0, delta=0.0,
                            price_range=15, lot_size=5, initial_mid=105)
    order = model.generate_event(book, 0.0, rng)
    assert order is not None and order.side == Side.BID and model.last_informed
    assert model.informed_ids.tolist() == [order.order_id]


def test_informed_orders_are_sent_only_when_the_value_is_beyond_the_quote_they_hit():
    """Glosten and Milgrom's condition at the time the order is sent, with jumps and a drifting value."""
    model = InformedZIModel(sigma_v=1.0, jump_rate=0.1, jump_size=3.0, kappa=2.0, cap=10.0, **ZI)
    seen = {"buy": 0, "sell": 0, "uninformed": 0}
    gaps = []

    def check(order, book):
        if order.order_type == OrderType.MARKET and not model.last_informed:
            seen["uninformed"] += 1
        if not model.last_informed:
            return
        assert order.order_type == OrderType.MARKET and order.size == ZI["lot_size"]
        v = model.value()
        assert model.value_time == order.timestamp  # V is read at the time of the order
        if order.side == Side.BID:
            assert v > book.best_ask
            gaps.append(v - book.best_ask)
            seen["buy"] += 1
        else:
            assert v < book.best_bid
            gaps.append(book.best_bid - v)
            seen["sell"] += 1

    _drive(model, 100_000, seed=7, on_order=check)
    assert seen["buy"] > 300 and seen["sell"] > 300 and seen["uninformed"] > 1_000
    assert min(gaps) > 0
    assert len(model.informed_ids) == seen["buy"] + seen["sell"] > 0


# ---- Ground truth recorded on the model ----

def test_ground_truth_counters_agree_with_the_event_log():
    model = InformedZIModel(kappa=2.0, cap=10.0, record_path=True, **ZI)
    sim, df = _simulate(model, 40_000, seed=8)  # no warm-up: the log holds every event the counters saw
    market = df[df["event_type"] == "market"]
    informed = market[market["order_id"].isin(model.informed_ids)]

    assert len(informed) == len(model.informed_ids) > 200
    assert model.taker_lots == market["size"].sum()
    assert model.informed_lots == informed["size"].sum()
    assert model.informed_share == pytest.approx(informed["size"].sum() / market["size"].sum())
    assert 0.05 < model.informed_share < 0.5
    # Every informed order is a market order; no limit order or cancel is one.
    assert set(df.loc[df["order_id"].isin(model.informed_ids), "event_type"]) == {"market"}
    # The fills mark their takers the same way.
    trades = sim.trades
    flagged = model.is_informed(trades["taker_order_id"].to_numpy())
    assert np.array_equal(flagged, trades["taker_order_id"].isin(model.informed_ids).to_numpy())
    assert 0 < flagged.sum() < len(trades)


def test_informed_share_is_none_before_any_market_order():
    model = InformedZIModel(**ZI)
    assert model.informed_share is None and model.taker_lots == 0 and model.informed_lots == 0


def test_last_informed_flags_the_order_just_returned_and_the_ids_are_those_flagged():
    model = InformedZIModel(kappa=2.0, cap=10.0, **ZI)
    flagged, other_market, sent = [], 0, []

    def watch(order, book):
        nonlocal other_market
        sent.append(order.order_id)
        if model.last_informed:
            flagged.append(order.order_id)
        else:
            other_market += order.order_type == OrderType.MARKET

    _drive(model, 30_000, seed=54, on_order=watch)
    ids = model.informed_ids
    assert ids.dtype == np.int64 and len(ids) == len(flagged) > 200 and other_market > 1_000
    assert np.array_equal(ids, flagged)                       # the order they were sent in
    assert np.array_equal(ids, np.sort(ids))                  # and ids rise
    assert np.array_equal(model.is_informed(sent), np.isin(sent, flagged))
    assert model.informed_lots == len(ids) * ZI["lot_size"]

    # No event, and an uninformed event after an informed one, both clear the flag.
    quiet = InformedZIModel(sigma_v=0.0, jump_rate=0.0, kappa=2.0, cap=4.0, mu=0.0, lam=0.0, delta=0.0,
                            price_range=15, lot_size=5, initial_mid=105)
    book, rng = _frozen_book(99, 101), np.random.default_rng(0)
    assert quiet.generate_event(book, 0.0, rng) is not None and quiet.last_informed
    quiet.initial_mid = 100
    quiet._value = 100.0  # V back inside the touch: every candidate is now dropped
    quiet.MAX_DROPPED = 20
    assert quiet.generate_event(book, 1.0, rng) is None and not quiet.last_informed


def test_the_ids_are_kept_compactly_and_the_model_can_go_on_after_they_are_read():
    model = InformedZIModel(sigma_v=0.0, jump_rate=0.0, kappa=2.0, cap=4.0, mu=0.0, lam=0.0, delta=0.0,
                            price_range=15, lot_size=5, initial_mid=105)
    book, rng = _frozen_book(99, 101), np.random.default_rng(0)
    first = [model.generate_event(book, 0.0, rng).order_id for _ in range(10)]
    kept = model.informed_ids
    assert model.is_informed([first[3], 10**9, first[0], -1]).tolist() == [True, False, True, False]
    for _ in range(10):  # an array a view still exports could not grow: reading must not stop the model
        model.generate_event(book, 0.0, rng)
    assert kept.tolist() == first and len(model.informed_ids) == 20  # what was read is a copy, and the model went on
    assert not np.shares_memory(kept, model.informed_ids)

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(50_000):
            model.generate_event(book, 0.0, rng)
        retained = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert len(model.informed_ids) == 50_020
    assert retained < 50_000 * 12  # 8 bytes an order and the array's slack; a set of ints takes 70 to 100


def test_value_at_reads_the_recorded_path():
    model = InformedZIModel(sigma_v=1.0, jump_rate=0.0, kappa=0.0, record_path=True, **ZI)
    _, df = _simulate(model, 2_000, seed=9)
    times = df["time"].to_numpy()
    at_events = model.value_at(times)
    assert at_events[-1] == model.value()
    # Between two events it is the value at the earlier one, and before the first it is the initial mid.
    midpoints = (times[1:] + times[:-1]) / 2
    assert np.array_equal(model.value_at(midpoints), at_events[:-1])
    assert model.value_at([-1.0, times[0] / 2]).tolist() == [100.0, 100.0]
    with pytest.raises(RuntimeError):
        InformedZIModel(**ZI).value_at([0.0])


def test_maker_loss_equals_the_value_read_when_the_order_was_sent():
    """The path gives the same loss as reading value() right after each order: V - price for a taker buy."""
    model = InformedZIModel(kappa=2.0, cap=10.0, record_path=True, **ZI)
    reset_order_id_counter()
    book, engine, rng = _seeded_book(), PriceTimeEngine(), np.random.default_rng(10)
    live, times, prices, buys = [], [], [], []
    t = 0.0
    for _ in range(20_000):
        order = model.generate_event(book, t, rng)
        if order is None:
            continue
        t = order.timestamp
        value = model.value()
        for fill in engine.process(book, order):
            buys.append(order.side == Side.BID)
            live.append(value - fill.price if order.side == Side.BID else fill.price - value)
            times.append(fill.timestamp)
            prices.append(fill.price)
    assert len(live) > 1_000
    assert np.allclose(model.maker_loss(times, prices, buys), live, rtol=0, atol=1e-12)


def test_maker_loss_is_positive_on_informed_fills_and_the_spread_on_uninformed_ones():
    """Against V, informed takers cost the makers; uninformed ones pay them about the half-spread."""
    model = InformedZIModel(kappa=2.0, cap=3.0, record_path=True, **ZI)  # the defaults
    sim, df = _simulate(model, 150_000, seed=21)
    trades = sim.trades
    loss = model.maker_loss(trades["time"].to_numpy(), trades["price"].to_numpy(),
                            (trades["aggressor"] == "bid").to_numpy())
    informed = model.is_informed(trades["taker_order_id"].to_numpy())

    # The mid just before each fill's event, from the log (its row's predecessor), and the taker's sign.
    market = df["event_type"] == "market"  # a taker is always one of the model's market orders here
    mid_before = pd.Series(df["mid_price"].shift(1)[market].to_numpy(), index=df.loc[market, "order_id"].to_numpy())
    mid_before = mid_before.reindex(trades["taker_order_id"].to_numpy()).to_numpy()
    sign = np.where(trades["aggressor"] == "bid", 1.0, -1.0)
    ok = ~np.isnan(mid_before)
    value_minus_mid = loss + sign * (trades["price"].to_numpy() - mid_before)  # = sign * (V - mid), whatever V was

    informed_loss, uninformed_loss = loss[ok & informed], loss[ok & ~informed]
    assert len(informed_loss) > 500 and len(uninformed_loss) > 5_000

    def t_stat(x):
        return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))

    assert t_stat(informed_loss) > 8
    # Uninformed takers' sides do not depend on V, so the value they trade against is the mid on average:
    # their fills cost the makers V - price = (V - mid) - (price - mid), and only the second part is not 0.
    assert abs(t_stat(value_minus_mid[ok & ~informed])) < 4
    assert value_minus_mid[ok & informed].mean() > 1.0
    effective_half_spread = (sign * (trades["price"].to_numpy() - mid_before))[ok & ~informed].mean()
    assert 0.9 < effective_half_spread < 1.6
    assert uninformed_loss.mean() == pytest.approx(-effective_half_spread, abs=4 * uninformed_loss.std() /
                                                   math.sqrt(len(uninformed_loss)))
    assert uninformed_loss.mean() < 0  # resting orders earn on the uninformed, and lose on the informed


def _informed_losses(model, n_events, seed):
    """(informed share, mean loss per informed fill against V) read live: V at the time of each order."""
    losses = []
    reset_order_id_counter()
    book, engine, rng = _seeded_book(), PriceTimeEngine(), np.random.default_rng(seed)
    t = 0.0
    for _ in range(n_events):
        order = model.generate_event(book, t, rng)
        if order is None:
            continue
        t = order.timestamp
        value, informed = model.value(), model.last_informed
        for fill in engine.process(book, order):
            if informed:
                losses.append(value - fill.price if order.side == Side.BID else fill.price - value)
    return model.informed_share, float(np.mean(losses)), len(losses)


def test_the_loss_on_an_informed_fill_is_positive_by_construction_and_the_share_does_not_fix_it():
    """The first fill of an informed order is at the quote it hit, and V is beyond it when it is sent: its loss is
    positive whatever the model does. What has content is its size, and that is not a function of the share:
    kappa raises the share but lowers the loss per informed fill (the book tracks V), sigma_v raises both."""
    sim, df = _simulate(InformedZIModel(record_path=True, **ZI), 60_000, seed=55)
    model = sim.model
    trades = sim.trades
    loss = model.maker_loss(trades["time"].to_numpy(), trades["price"].to_numpy(),
                            (trades["aggressor"] == "bid").to_numpy())
    informed = model.is_informed(trades["taker_order_id"].to_numpy())
    first = ~trades["taker_order_id"].duplicated().to_numpy()
    assert (informed & first).sum() > 300
    assert (loss[informed & first] > 0).all()

    def measure(**kwargs):
        runs = [_informed_losses(InformedZIModel(**kwargs, **ZI), 120_000, seed) for seed in (56, 57)]
        return np.mean([r[0] for r in runs]), np.mean([r[1] for r in runs])

    share_low, loss_low = measure(kappa=0.5, cap=3.0)
    share_high, loss_high = measure(kappa=4.0, cap=3.0)
    assert share_high > share_low + 0.02 and loss_high < loss_low - 0.5  # share up, loss per fill down
    share_calm, loss_calm = measure(sigma_v=0.3)
    share_wild, loss_wild = measure(sigma_v=0.8)
    assert share_wild > share_calm + 0.05 and loss_wild > loss_calm + 0.2  # share up, loss per fill up


def test_the_same_seed_does_not_give_the_same_value_path_once_kappa_changes():
    """V is advanced over the run's own event and candidate times, so a run with another kappa has other times: the
    same seed shares the standard normals V draws, not its realized path."""
    paths = []
    for kappa in (0.0, 2.0):
        model = InformedZIModel(kappa=kappa, cap=3.0, value_seed=7, record_path=True, **ZI)
        _simulate(model, 20_000, seed=58)
        paths.append(model.value_at(np.arange(1.0, 200.0, 10.0)))
    assert not np.allclose(paths[0], paths[1], atol=0.5)


# ---- Event for event, and reproducibility ----

@pytest.mark.parametrize("kwargs", [
    dict(kappa=0.0),
    dict(kappa=0.0, persistence=0.5),  # the fair coin, said out loud
    dict(kappa=0.0, sigma_v=2.0, jump_rate=1.0, jump_size=5.0),  # a value that moves a lot changes nothing
    dict(kappa=3.0, cap=0.0),
])
def test_without_informed_traders_the_run_is_smith_farmers_event_for_event(kwargs):
    smith_sim, smith = _simulate(SmithFarmerModel(**ZI), 30_000, seed=12)
    model = InformedZIModel(record_path=True, **kwargs, **ZI)
    sim, df = _simulate(model, 30_000, seed=12)

    pd.testing.assert_frame_equal(df, smith)
    pd.testing.assert_frame_equal(sim.trades, smith_sim.trades)
    # The same numbers were taken from the run's generator, and so the same ones are left.
    assert sim.rng.bit_generator.state == smith_sim.rng.bit_generator.state
    assert len(model.informed_ids) == 0 and model.informed_lots == 0 and model.taker_lots > 0
    assert not model.last_informed
    assert model.value() != 100.0  # the value did move


def test_the_same_seed_gives_the_same_run():
    runs = []
    for _ in range(2):
        model = InformedZIModel(kappa=2.0, cap=3.0, record_path=True, **ZI)
        sim, df = _simulate(model, 20_000, seed=13)
        runs.append((model, sim, df))
    (m1, s1, d1), (m2, s2, d2) = runs
    pd.testing.assert_frame_equal(d1, d2)
    pd.testing.assert_frame_equal(s1.trades, s2.trades)
    assert np.array_equal(m1.informed_ids, m2.informed_ids) and len(m1.informed_ids) > 100
    assert (m1.informed_lots, m1.taker_lots, m1.jumps) == (m2.informed_lots, m2.taker_lots, m2.jumps)
    assert np.array_equal(m1.value_at(d1["time"]), m2.value_at(d2["time"]))

    other = InformedZIModel(kappa=2.0, cap=3.0, record_path=True, **ZI)
    _, d3 = _simulate(other, 20_000, seed=14)
    assert not np.array_equal(other.value_at(d3["time"]), m1.value_at(d1["time"]))
    assert not d3["price"].equals(d1["price"])


def test_the_value_has_its_own_random_numbers_in_a_fixed_order():
    """With value_seed the value's draws do not depend on the run: the same standard normals in the same order."""
    paths = []
    for run_seed in (15, 16):
        model = InformedZIModel(sigma_v=0.7, jump_rate=0.0, kappa=0.0, value_seed=99, record_path=True, **ZI)
        _, df = _simulate(model, 3_000, seed=run_seed)
        times = np.concatenate(([0.0], df["time"].to_numpy()))
        values = np.concatenate(([100.0], model.value_at(df["time"].to_numpy())))
        paths.append(np.diff(values) / (0.7 * np.sqrt(np.diff(times))))
    assert np.allclose(paths[0], paths[1], rtol=0, atol=1e-9)
    assert np.allclose(paths[0], np.random.default_rng(99).standard_normal(len(paths[0])), rtol=0, atol=1e-9)


def test_value_generator_of_a_bit_generator_without_a_jump_is_still_seeded():
    values = []
    for _ in range(2):
        model = InformedZIModel(sigma_v=1.0, kappa=0.0, **ZI)
        rng = np.random.Generator(np.random.SFC64(5))
        book = _seeded_book()
        for _ in range(100):
            model.generate_event(book, 0.0, rng)
        values.append(model.value())
    assert values[0] == values[1] != 100.0


# ---- Persistence: one-sided but harmless ----

def _sides_of_the_market_orders(model, n_events, seed):
    """The sides (+1 a buy, -1 a sell) of what the model sent, in order: its uninformed and all its market orders,
    and its limit orders. Read from the orders the model returns, with ``last_informed`` as the model says it."""
    uninformed, market, limit = [], [], []

    def on_order(order, book):
        sign = 1.0 if order.side == Side.BID else -1.0
        if order.order_type == OrderType.MARKET:
            market.append(sign)
            if not model.last_informed:
                uninformed.append(sign)
        elif order.order_type == OrderType.LIMIT:
            limit.append(sign)

    _drive(model, n_events, seed, on_order)
    return np.array(uninformed), np.array(market), np.array(limit)


def _lag1(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1]


class _ScriptedRng:
    """A generator whose uniforms are the ones the test hands it."""

    def __init__(self, *draws):
        self.draws = list(draws)

    def random(self):
        return self.draws.pop(0)


def test_the_chain_of_sides_at_hand_set_draws():
    """A side is one uniform: bid if it is below the chance of a bid, which is 1/2 for the first order, then
    ``persistence`` after a bid and ``1 - persistence`` after an ask."""
    model = InformedZIModel(persistence=0.8, **ZI)
    draws = [
        (0.51, Side.ASK),   # the first order: a fair coin (an ask from 0.5)
        (0.21, Side.ASK),   # after an ask, an ask from 0.2: the same side
        (0.19, Side.BID),   # after an ask, a bid below 0.2: the other side
        (0.79, Side.BID),   # after a bid, a bid below 0.8: the same side
        (0.81, Side.ASK),   # after a bid, an ask from 0.8: the other side
        (0.0, Side.BID),    # after an ask, the other side
        (0.99, Side.ASK),   # after a bid, the other side
    ]
    rng = _ScriptedRng(*(u for u, _ in draws))
    assert [model._sample_market_side(rng) for _ in draws] == [side for _, side in draws]
    assert rng.draws == []  # one uniform a side
    assert InformedZIModel(persistence=0.8, **ZI)._sample_market_side(_ScriptedRng(0.49)) == Side.BID  # first, a coin

    # At 0.5 both chances are 1/2: the fair coin, whatever the side before.
    fair = InformedZIModel(persistence=0.5, **ZI)
    rng = _ScriptedRng(0.49, 0.51, 0.49, 0.5, 0.0)
    assert [fair._sample_market_side(rng) for _ in range(5)] == [Side.BID, Side.ASK, Side.BID, Side.ASK, Side.BID]


def test_a_side_drawn_for_an_order_that_finds_nothing_to_take_still_steps_the_chain():
    """The chain is on the sides drawn: with no ask to take, a buy is not sent, and the next side is still drawn
    from the buy before it (0.7 after a buy at persistence 0.8 is a buy again; from a fresh start, a sell)."""
    model = InformedZIModel(persistence=0.8, **ZI)
    book = OrderBook()
    book.add_order(limit_order(99, 5, 0.0, Side.BID))
    rng = _ScriptedRng(0.1, 0.7, 0.9)
    assert model._generate_market_order(book, rng) is None  # a buy: no ask to take
    assert model._generate_market_order(book, rng) is None  # after a buy, 0.7 is a buy again
    sell = model._generate_market_order(book, rng)  # after a buy, 0.9 is a sell
    assert sell is not None and sell.side == Side.ASK
    assert model.taker_lots == 5  # only what was sent is counted


@pytest.mark.parametrize("persistence", [0.5, 0.7, 0.85, 0.95])
def test_uninformed_sides_have_a_lag_one_autocorrelation_of_two_p_minus_one(persistence):
    """Another computation: the product of two successive signs is +1 with probability p, whatever came before,
    so the mean of the products over m pairs is 2p - 1 with a standard error of 2 sqrt(p (1 - p) / m)."""
    model = InformedZIModel(persistence=persistence, **ZI)  # the informed traders are on, at their defaults
    uninformed, market, limit = _sides_of_the_market_orders(model, 150_000, seed=7)
    pairs = len(uninformed) - 1
    expected, se = 2 * persistence - 1, 2 * math.sqrt(persistence * (1 - persistence) / (len(uninformed) - 1))
    assert pairs > 14_000 and len(market) - len(uninformed) == len(model.informed_ids) > 2_000
    assert abs(_lag1(uninformed) - expected) < 4 * se
    assert abs(np.mean(uninformed[:-1] * uninformed[1:]) - expected) < 4 * se
    # The chain's long-run share of buys is a half (its standard error is sqrt(p / (1 - p) / n) for the mean).
    assert abs(uninformed.mean()) < 4 * math.sqrt(persistence / (1 - persistence) / len(uninformed))
    # The informed orders are not in the chain: among all the market orders the sides are less persistent.
    if persistence >= 0.7:
        assert _lag1(market) < _lag1(uninformed) - 0.05
    # Only the market orders persist: the limit orders' sides are fair coins, as are Smith's.
    assert abs(_lag1(limit)) < 4 / math.sqrt(len(limit)) and abs(limit.mean()) < 4 / math.sqrt(len(limit))


def test_informed_orders_that_outnumber_the_uninformed_leave_the_chain_alone():
    """A frozen book with V five ticks above the ask: informed buys come at 10 a unit, the uninformed market orders
    at 6, and the sides of the uninformed ones are still the chain's, as a fair coin's chain would be."""
    persistence, n = 0.9, 60_000
    model = InformedZIModel(sigma_v=0.0, jump_rate=0.0, kappa=2.0, cap=30.0, persistence=persistence, **ZI)
    book, rng = _frozen_book(90, 95), np.random.default_rng(23)
    t, uninformed, informed = 0.0, [], 0
    for _ in range(n):
        order = model.generate_event(book, t, rng)
        t = order.timestamp
        if order.order_type == OrderType.MARKET:
            if model.last_informed:
                informed += 1
                assert order.side == Side.BID
            else:
                uninformed.append(1.0 if order.side == Side.BID else -1.0)
    uninformed = np.array(uninformed)
    se = 2 * math.sqrt(persistence * (1 - persistence) / len(uninformed))
    assert informed > 1.3 * len(uninformed) > 6_000  # 10 against 6 a unit
    assert abs(_lag1(uninformed) - (2 * persistence - 1)) < 4 * se
    assert abs(uninformed.mean()) < 4 * math.sqrt(persistence / (1 - persistence) / len(uninformed))


class _CoinFlipModel(InformedZIModel):
    """The model as it was before persistence: the market orders' sides come from ``_sample_side``, the fair coin."""

    def _sample_market_side(self, rng):
        return self._sample_side(rng)


@pytest.mark.parametrize("kwargs", [dict(), dict(kappa=6.0, cap=10.0, sigma_v=0.8), dict(jump_rate=1.0, jump_size=5.0)])
def test_persistence_one_half_is_the_fair_coin_event_for_event(kwargs):
    """With the informed traders on, the run at 0.5 (the default, or said out loud) is the fair coin's run: the same
    events, fills, informed orders, value path and generator state."""
    reference_model = _CoinFlipModel(record_path=True, **kwargs, **ZI)
    reference_sim, reference = _simulate(reference_model, 30_000, seed=17)
    for extra in (dict(), dict(persistence=0.5)):
        model = InformedZIModel(record_path=True, **kwargs, **extra, **ZI)
        sim, df = _simulate(model, 30_000, seed=17)
        pd.testing.assert_frame_equal(df, reference)
        pd.testing.assert_frame_equal(sim.trades, reference_sim.trades)
        assert sim.rng.bit_generator.state == reference_sim.rng.bit_generator.state
        assert np.array_equal(model.informed_ids, reference_model.informed_ids) and len(model.informed_ids) > 100
        assert np.array_equal(model.value_at(df["time"]), reference_model.value_at(reference["time"]))
        assert (model.informed_lots, model.taker_lots, model.jumps) == (
            reference_model.informed_lots, reference_model.taker_lots, reference_model.jumps)


def test_persistence_changes_the_run_and_the_same_seed_repeats_it():
    runs = []
    for _ in range(2):
        model = InformedZIModel(persistence=0.85, kappa=2.0, cap=3.0, record_path=True, **ZI)
        sim, df = _simulate(model, 20_000, seed=13)
        runs.append((model, sim, df))
    (m1, s1, d1), (m2, s2, d2) = runs
    pd.testing.assert_frame_equal(d1, d2)
    pd.testing.assert_frame_equal(s1.trades, s2.trades)
    assert np.array_equal(m1.informed_ids, m2.informed_ids) and len(m1.informed_ids) > 100
    assert (m1.informed_lots, m1.taker_lots, m1.jumps) == (m2.informed_lots, m2.taker_lots, m2.jumps)
    assert np.array_equal(m1.value_at(d1["time"]), m2.value_at(d2["time"]))
    assert s1.rng.bit_generator.state == s2.rng.bit_generator.state

    other_seed = InformedZIModel(persistence=0.85, kappa=2.0, cap=3.0, **ZI)
    _, d3 = _simulate(other_seed, 20_000, seed=14)
    assert not d3["price"].equals(d1["price"])
    fair = InformedZIModel(persistence=0.5, kappa=2.0, cap=3.0, **ZI)
    _, d4 = _simulate(fair, 20_000, seed=13)
    assert not d4["side"].equals(d1["side"])


@pytest.mark.parametrize("persistence", [0.5, 0.95])
def test_without_informed_traders_persistence_leaves_the_informed_share_zero_and_the_value_its_own(persistence):
    """At kappa 0 nothing is informed, whatever the persistence, and V's random numbers are its own: its standardized
    increments are the same standard normals in the same order for every run, persistent or not."""
    model = InformedZIModel(sigma_v=0.7, jump_rate=0.0, kappa=0.0, persistence=persistence, value_seed=99,
                            record_path=True, **ZI)
    _, df = _simulate(model, 6_000, seed=18)
    assert len(model.informed_ids) == 0 and model.informed_lots == 0 and model.taker_lots > 0
    assert model.informed_share == 0.0
    times = np.concatenate(([0.0], df["time"].to_numpy()))
    values = np.concatenate(([100.0], model.value_at(df["time"].to_numpy())))
    steps = np.diff(values) / (0.7 * np.sqrt(np.diff(times)))
    assert np.allclose(steps, np.random.default_rng(99).standard_normal(len(steps)), rtol=0, atol=1e-9)


@pytest.mark.parametrize("persistence, first_seed", [(0.5, 1000), (0.95, 2000)])
def test_the_value_has_the_same_distribution_whatever_the_persistence(persistence, first_seed):
    """V at the end of a run, over seeds, standardized by the theory's variance for the run's own length: mean 0 and
    variance 1 (the run's length depends on the book, which the persistence changes; V does not depend on the book
    when kappa is 0, so given the length V is Gaussian plus jumps, with variance (sigma^2 + rate size^2) t)."""
    sigma, rate, size, n_seeds = 0.3, 0.05, 2.0, 200
    z = []
    for seed in range(first_seed, first_seed + n_seeds):
        model = InformedZIModel(sigma_v=sigma, jump_rate=rate, jump_size=size, kappa=0.0, persistence=persistence,
                                **ZI)
        _drive(model, 1_500, seed)
        z.append((model.value() - 100.0) / math.sqrt((sigma ** 2 + rate * size ** 2) * model.value_time))
    z = np.array(z)
    assert abs(z.mean()) < 4 / math.sqrt(n_seeds)
    # The sample variance's standard error is sqrt((2 + excess kurtosis) / n), the excess being 0.3 here (the jumps).
    assert abs(z.var(ddof=1) - 1.0) < 4 * math.sqrt(2.5 / n_seeds)


# ---- The informed share ----

def test_the_informed_share_rises_with_kappa():
    shares = []
    for kappa in (0.0, 0.5, 2.0, 8.0):
        lots = []
        for seed in (31, 32):
            model = InformedZIModel(kappa=kappa, cap=10.0, **ZI)
            _drive(model, 80_000, seed)
            lots.append((model.informed_lots, model.taker_lots))
        shares.append(sum(i for i, _ in lots) / sum(t for _, t in lots))
    assert shares[0] == 0.0
    assert shares[0] < shares[1] < shares[2] < shares[3]
    assert shares[2] - shares[1] > 0.02 and shares[3] - shares[2] > 0.02


def test_defaults_give_an_informed_share_of_ten_to_twenty_five_percent():
    model = InformedZIModel(**ZI)
    _drive(model, 150_000, seed=41)
    assert model.informed_share is not None and 0.10 < model.informed_share < 0.25


@pytest.mark.parametrize("name, value", [
    ("sigma_v", -0.1), ("jump_rate", -1.0), ("jump_size", -2.0), ("kappa", -1.0), ("cap", -0.5),
    ("kappa", float("nan")), ("cap", float("inf")),
    ("persistence", -0.01), ("persistence", 1.01), ("persistence", float("nan")), ("persistence", float("inf")),
])
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        InformedZIModel(**{name: value}, **ZI)


def test_new_parameters_are_keyword_only():
    with pytest.raises(TypeError):
        InformedZIModel(6.0, 1.0, 0.03)  # the zero-intelligence ones are not positional here either
