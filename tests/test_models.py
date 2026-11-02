"""Tests for the simulation models and the simulation harness."""

import numpy as np
import pytest

from lob.book import OrderBook
from lob.matching import PriceTimeEngine
from lob.models.mike_farmer import MikeFarmerModel, _generate_fractional_noise
from lob.models.long_memory import LongMemorySigns, farima_weights
from lob.models.strategic import ParlourModel
from lob.models.zero_intelligence import SmithFarmerModel
from lob.order import OrderType, Side, limit_order, reset_order_id_counter
from lob.simulation import Simulation, SimulationConfig


@pytest.fixture(autouse=True)
def reset_ids():
    reset_order_id_counter()
    yield
    reset_order_id_counter()


def _book(bid, ask, depth=1):
    book = OrderBook()
    for _ in range(depth):
        book.add_order(limit_order(bid, 1, 0.0, Side.BID))
        book.add_order(limit_order(ask, 1, 0.0, Side.ASK))
    return book


def _fgn_autocorrelation(k, H):
    k = np.abs(np.asarray(k, dtype=float))
    return 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * k ** (2 * H) + (k + 1) ** (2 * H))


# ---- Zero-intelligence model ----

def test_zi_places_limit_orders_uniformly_next_to_opposite_quote():
    book = _book(100, 105)
    model = SmithFarmerModel(price_range=10)
    rng = np.random.default_rng(0)
    model.generate_event(book, 0.0, rng)  # picks up the current quotes
    bids, asks = [], []
    for _ in range(20_000):
        o = model._generate_limit_order(book, rng)
        (bids if o.side == Side.BID else asks).append(o.price)
    bids, asks = np.array(bids), np.array(asks)

    assert set(bids) == set(range(95, 105))   # a - L, ..., a - 1
    assert set(asks) == set(range(101, 111))  # b + 1, ..., b + L
    shares = np.bincount(bids - 95, minlength=10) / len(bids)
    assert np.all(np.abs(shares - 0.1) < 0.015)
    # A buy lands inside the spread with probability (s - 1) / L.
    assert np.mean(bids > 100) == pytest.approx(0.4, abs=0.02)


def test_zi_limit_orders_never_execute_on_arrival():
    model = SmithFarmerModel(mu=1.0, lam=0.5, delta=0.1, price_range=20)
    book, engine, rng = OrderBook(), PriceTimeEngine(), np.random.default_rng(1)
    for _ in range(5_000):
        o = model.generate_event(book, 0.0, rng)
        if o is None:
            continue
        fills = engine.process(book, o)
        if o.order_type == OrderType.LIMIT:
            assert fills == []


def test_zi_event_sequence_is_invariant_to_time_units():
    """Multiplying every rate by k must only rescale time."""
    runs = []
    for k in (1.0, 10.0):
        reset_order_id_counter()
        model = SmithFarmerModel(mu=1.0 * k, lam=0.5 * k, delta=0.1 * k, price_range=50)
        config = SimulationConfig(n_events=3_000, warmup_events=0, seed=5)
        runs.append(Simulation(model=model, config=config).run(progress=False))
    slow, fast = runs
    assert slow["event_type"].tolist() == fast["event_type"].tolist()
    assert slow["price"].tolist() == fast["price"].tolist()
    np.testing.assert_allclose(slow["time"].values, 10.0 * fast["time"].values)


def test_zi_side_hook_is_applied_before_pricing():
    class BuyOnly(SmithFarmerModel):
        def _sample_side(self, rng):
            return Side.BID

    book = _book(100, 105)
    model = BuyOnly(price_range=10)
    rng = np.random.default_rng(2)
    model.generate_event(book, 0.0, rng)
    for _ in range(1_000):
        o = model._generate_limit_order(book, rng)
        assert o.side == Side.BID and o.price < 105


@pytest.mark.parametrize("model_cls", [SmithFarmerModel, MikeFarmerModel, ParlourModel])
def test_models_continue_from_callers_clock(model_cls):
    model = model_cls(initial_mid=10000)
    book = _book(9999, 10001, depth=5)
    rng = np.random.default_rng(3)
    for _ in range(20):
        o = model.generate_event(book, 500.0, rng)
        if o is not None:
            assert o.timestamp > 500.0
    assert model._t > 500.0


# ---- Fractional Gaussian noise ----

@pytest.mark.parametrize("H", [0.7, 0.8])
def test_fractional_noise_has_fgn_autocorrelation(H):
    rng = np.random.default_rng(11)
    lags = np.array([1, 2, 10])
    reps, n = 60, 4096
    lagged = np.zeros(len(lags))
    power = 0.0
    for _ in range(reps):
        x = _generate_fractional_noise(n, H, rng)
        power += np.mean(x * x)
        lagged += [np.mean(x[:-k] * x[k:]) for k in lags]
    # Known zero mean, so no demeaning bias; previously every lag came out halved.
    np.testing.assert_allclose(lagged / power, _fgn_autocorrelation(lags, H), atol=0.03)
    assert power / reps == pytest.approx(1.0, abs=0.1)


# ---- Long-memory sign stream ----

def _sign_acf_theory(hurst, memory, lags):
    """Sign autocorrelation of truncated FARIMA(0, H - 1/2, 0) noise: (2/pi) arcsin(rho_k)."""
    psi = farima_weights(hurst - 0.5, memory)
    rho = np.array([psi[:-k] @ psi[k:] for k in lags]) / (psi @ psi)
    return 2 / np.pi * np.arcsin(rho)


def test_long_memory_signs_match_farima_theory():
    stream, rng = LongMemorySigns(0.75, memory=1024), np.random.default_rng(5)
    signs = np.array([stream.next(rng) for _ in range(300_000)], dtype=float)
    lags = [1, 10, 100]
    measured = np.array([np.mean(signs[:-k] * signs[k:]) for k in lags])
    error = np.abs(measured - _sign_acf_theory(0.75, 1024, lags))
    np.testing.assert_array_less(error, [0.01, 0.01, 0.008])


def test_long_memory_signs_stay_correlated_across_chunks():
    """Independent blocks would make signs on either side of a chunk boundary uncorrelated."""
    memory = 64
    stream, rng = LongMemorySigns(0.75, memory=memory), np.random.default_rng(6)
    signs = np.array([stream.next(rng) for _ in range(memory * 4000)], dtype=float)
    boundaries = np.arange(memory, len(signs), memory)
    expected = _sign_acf_theory(0.75, memory, [1])[0]
    assert np.mean(signs[boundaries - 1] * signs[boundaries]) == pytest.approx(expected, abs=0.06)


def test_long_memory_signs_without_memory_are_independent():
    stream, rng = LongMemorySigns(0.5, memory=256), np.random.default_rng(7)
    signs = np.array([stream.next(rng) for _ in range(100_000)], dtype=float)
    assert abs(np.mean(signs[:-1] * signs[1:])) < 0.01


# ---- Mike-Farmer model ----

@pytest.mark.parametrize("bid, ask", [(9999, 10000), (9999, 10001), (9999, 10002)])
@pytest.mark.parametrize("delta", [1, 2, 5])
def test_mike_farmer_rounding_is_mirror_symmetric(bid, ask, delta):
    """A bid and an ask with the same relative price sit equally far from the mid."""
    book = _book(bid, ask)
    model = MikeFarmerModel()
    rng = np.random.default_rng(0)
    model._sample_relative_price = lambda rng: delta
    model._next_sign = lambda rng: 1
    buy = model._generate_limit_order(book, rng)
    model._next_sign = lambda rng: -1
    sell = model._generate_limit_order(book, rng)

    mid = book.mid_price
    assert buy.side == Side.BID and sell.side == Side.ASK
    assert mid - buy.price == pytest.approx(sell.price - mid)
    assert buy.price < ask and sell.price > bid


# ---- Parlour model ----

def test_parlour_market_order_probability_follows_spread_and_depth():
    model = ParlourModel()

    def book_with(spread, depth):
        return _book(10000, 10000 + spread, depth=depth)

    assert model._prob_market_order(book_with(1, 20)) > model._prob_market_order(book_with(10, 20))
    assert model._prob_market_order(book_with(2, 20)) > model._prob_market_order(book_with(2, 1))


# ---- Simulation harness ----

def test_simulation_records_events_after_warmup():
    config = SimulationConfig(n_events=2_000, warmup_events=500, seed=1,
                              record_book_snapshots=True, snapshot_interval=100)
    sim = Simulation(model=SmithFarmerModel(), config=config)
    df = sim.run(progress=False)

    assert 0 < len(df) <= 2_000
    assert df["time"].is_monotonic_increasing
    assert set(df["event_type"]) <= {"limit", "market", "cancel"}
    assert 0 < len(sim.snapshots) <= 20
    assert {"best_bid", "best_ask", "bids", "asks", "event_index"} <= set(sim.snapshots[0])
