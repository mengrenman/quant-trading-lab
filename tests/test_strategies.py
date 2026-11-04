"""The strategies' decisions (lob/strategies.py) and their settings (app/strategy_specs.py)."""

import inspect
import math

import pytest

from app.strategy_specs import STRATEGY_SPECS
from lob.order import Side
from lob.strategies import MarketMaking, MeanReversion, Momentum, QueueImbalance, View


def view(bid=99, ask=101, bid_depth=10, ask_depth=10, position=0, time=0.0):
    return View(time=time, best_bid=bid, best_ask=ask, bid_depth=bid_depth, ask_depth=ask_depth, position=position)


def at_mid(mid, **kw):
    """A book whose mid is ``mid``: two ticks wide around a whole number, one around a half."""
    if float(mid).is_integer():
        return view(bid=int(mid) - 1, ask=int(mid) + 1, **kw)
    return view(bid=math.floor(mid), ask=math.ceil(mid), **kw)


# ---- Market making ----

def test_the_market_maker_quotes_around_the_mid_and_leans_against_its_inventory():
    mm = MarketMaking(half_spread=2, size=5, limit=50, lean=0.5)
    assert mm.decide(view(99, 101)).quotes == {Side.BID: (98, 5), Side.ASK: (102, 5)}
    # At its limit, long: no bid, and the ask moves down by a half spread (lean 0.5 of 2 ticks... = 1).
    assert mm.decide(view(99, 101, position=50)).quotes == {Side.ASK: (101, 5)}
    # Short 30: both quotes move up by 0.3 of a half spread (0.6 ticks), rounded away from the mid.
    assert mm.decide(view(99, 101, position=-30)).quotes == {Side.BID: (98, 5), Side.ASK: (103, 5)}
    # Near its limit it quotes only what it may still take on.
    assert mm.decide(view(99, 101, position=48)).quotes[Side.BID] == (98 - 1, 2)


def test_the_market_makers_quotes_never_cross_the_spread():
    mm = MarketMaking(half_spread=1, size=5, limit=5, lean=1.0)
    quotes = mm.decide(view(99, 101, position=-5)).quotes  # short at its limit: it leans up hard
    assert quotes == {Side.BID: (100, 5)}  # stops short of the best ask, and sells no more
    assert MarketMaking(half_spread=1).decide(view(1, 3)).quotes[Side.BID] == (1, 5)
    assert MarketMaking(half_spread=5).decide(view(1, 3)).quotes.get(Side.BID) is None  # no price below 1


def test_the_market_maker_does_not_quote_without_a_mid():
    assert MarketMaking().decide(view(bid=None)).quotes == {}


# ---- Momentum ----

def test_momentum_waits_for_its_slow_average_then_follows_the_trend():
    m = Momentum(fast=2, slow=4, threshold=0.5, size=5)  # averages with weights 2/3 and 2/5
    plans = [m.decide(at_mid(100)) for _ in range(3)]
    assert [p.target for p in plans] == [0, 0, 0] and m.warming_up() == (3, 4)
    assert m.decide(at_mid(100)).target == 0 and m.warming_up() is None  # warm, and no trend
    # Gaps: 0.8 (long), 0.75, 0.27 (within the threshold but above zero: it holds), -0.16 (crossed:
    # flat), -0.74 (short).
    plans = [m.decide(at_mid(mid)) for mid in (103, 103, 102, 101, 99)]
    assert [p.target for p in plans] == [5, 5, 5, 0, -5]
    assert [round(p.signal, 2) for p in plans] == [0.8, 0.75, 0.27, -0.16, -0.74]


def test_the_momentum_averages_are_exponential():
    m = Momentum(fast=3, slow=7)
    m.decide(at_mid(100))
    plan = m.decide(at_mid(104))
    assert plan.signal == pytest.approx(4 * 2 / 4 - 4 * 2 / 8)  # alphas 2/(n+1)


# ---- Mean reversion ----

def test_mean_reversion_fades_a_large_move_and_exits_near_the_average():
    mr = MeanReversion(window=5, entry=1.5, exit=0.5, size=5)
    for mid in (100, 101, 100, 101):
        assert mr.decide(at_mid(mid)).target == 0
    assert mr.warming_up() == (4, 5)
    plan = mr.decide(at_mid(100))
    assert plan.signal is not None and mr.warming_up() is None
    plan = mr.decide(at_mid(106))  # far above its average: short
    window = [101, 100, 101, 100, 106]
    mean = sum(window) / 5
    std = math.sqrt(sum((m - mean) ** 2 for m in window) / 5)
    assert plan.signal == pytest.approx((106 - mean) / std)
    assert plan.target == -5
    plan = mr.decide(at_mid(104))  # still above, but within entry: it holds
    assert 0.5 <= plan.signal <= 1.5 and plan.target == -5
    plan = mr.decide(at_mid(103))  # back near its average
    assert abs(plan.signal) < 0.5 and plan.target == 0


def test_mean_reversion_running_sums_match_a_direct_computation():
    import random

    rng = random.Random(3)
    mr = MeanReversion(window=7, entry=1.5, exit=0.3)
    mids = []
    for _ in range(300):
        bid = 100 + rng.randint(-6, 6)
        ask = bid + rng.randint(1, 3)
        plan = mr.decide(view(bid, ask))
        mids.append((bid + ask) / 2)
        if len(mids) >= 7:
            w = mids[-7:]
            mean = sum(w) / 7
            std = math.sqrt(sum((m - mean) ** 2 for m in w) / 7)
            assert plan.signal == pytest.approx((w[-1] - mean) / std if std else 0.0, abs=1e-12)


# ---- Queue imbalance ----

def test_queue_imbalance_buys_when_the_bid_queue_is_much_the_larger_and_exits_when_it_tips():
    qi = QueueImbalance(threshold=0.6, size=5)
    assert qi.decide(view(bid_depth=10, ask_depth=10)).target == 0
    plan = qi.decide(view(bid_depth=90, ask_depth=10))  # imbalance 0.8
    assert (plan.target, plan.signal) == (5, pytest.approx(0.8))
    assert qi.decide(view(bid_depth=60, ask_depth=40)).target == 5  # 0.2: within the threshold, it holds
    assert qi.decide(view(bid_depth=40, ask_depth=60)).target == 0  # tipped the other way: flat
    assert qi.decide(view(bid_depth=1, ask_depth=19)).target == -5
    assert qi.decide(view(bid=None)).target is None  # nothing to read: leave things as they are


# ---- Settings ----

@pytest.mark.parametrize("make, message", [
    (lambda: MarketMaking(size=10, limit=5), "below the quote size"),
    (lambda: Momentum(fast=20, slow=20), "shorter than the slow"),
    (lambda: MeanReversion(entry=1.0, exit=1.0), "below the entry"),
    (lambda: QueueImbalance(threshold=1.0), "below 1"),
    (lambda: Momentum(execution="sideways"), "execution"),
])
def test_contradictory_settings_are_refused(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_each_spec_matches_its_strategy():
    for spec in STRATEGY_SPECS.values():
        parameters = inspect.signature(spec.factory.__init__).parameters
        assert set(parameters) - {"self"} == {p.name for p in spec.params}, spec.key
        for p in spec.params:
            assert parameters[p.name].default == p.default, (spec.key, p.name)
        strategy = spec.build(spec.defaults())
        assert strategy.kind == spec.key and strategy.describe()


def test_a_spec_parses_a_form_and_says_what_is_wrong():
    spec = STRATEGY_SPECS["momentum"]
    values, errors = spec.parse({"fast": "3", "slow": "12", "threshold": "0.25", "execution": "passive"})
    assert errors == {} and values == {**spec.defaults(), "fast": 3, "slow": 12, "threshold": 0.25,
                                       "execution": "passive"}
    _, errors = spec.parse({"fast": "2.5", "size": "0", "execution": "maybe"})
    assert set(errors) == {"fast", "size", "execution"}
    assert errors["execution"] == "Trade by: choose crossing the spread or resting at the best quote"
