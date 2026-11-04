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


# ---- Avellaneda-Stoikov ----

def test_avellaneda_stoikov_waits_for_its_volatility_then_quotes_around_its_reservation_price():
    from lob.strategies import AvellanedaStoikov

    a = AvellanedaStoikov(gamma=0.1, k=1.5, horizon=10, window=4, size=5, limit=50, interval=1.0)
    for mid in (100, 101, 100, 101):
        assert a.decide(at_mid(mid)).quotes == {}  # three changes: still estimating
    assert a.warming_up() == (3, 4)
    plan = a.decide(at_mid(100))  # changes of 1 tick a unit: variance 1
    assert a.variance == pytest.approx(1.0) and plan.signal == 0
    spread = 0.1 * 1 * 10 + 2 / 0.1 * math.log(1 + 0.1 / 1.5)  # 2.29
    assert plan.quotes == {Side.BID: (math.floor(100 - spread / 2), 5), Side.ASK: (math.ceil(100 + spread / 2), 5)}
    long = AvellanedaStoikov(gamma=0.1, k=1.5, horizon=10, window=4, size=5, limit=50, interval=1.0)
    for mid in (100, 101, 100, 101, 100):
        plan = long.decide(at_mid(mid, position=2))
    assert plan.signal == pytest.approx(-2 * 0.1 * 1 * 10)  # long: the reservation price 2 ticks lower
    assert plan.quotes[Side.ASK][0] == math.ceil(98 + spread / 2)
    assert plan.quotes[Side.BID][0] == math.floor(98 - spread / 2)


def test_avellaneda_stoikov_never_crosses_and_stops_at_its_limit():
    from lob.strategies import AvellanedaStoikov

    a = AvellanedaStoikov(gamma=1.0, k=1.0, horizon=10, window=2, size=5, limit=5)
    for mid in (100, 103, 100):
        plan = a.decide(at_mid(mid, position=-5))  # short at its limit, in a volatile market: it leans up hard
    assert Side.ASK not in plan.quotes and plan.quotes[Side.BID][0] <= 100  # bids, but not across the ask at 101


# ---- Penny jumping ----

def test_penny_jumping_steps_ahead_when_the_spread_allows_and_joins_otherwise():
    from lob.strategies import PennyJumping

    p = PennyJumping(min_spread=3, size=5, limit=50)
    assert p.decide(view(97, 101)).quotes == {Side.BID: (98, 5), Side.ASK: (100, 5)}  # spread 4: one tick ahead
    assert p.decide(view(99, 101)).quotes == {Side.BID: (99, 5), Side.ASK: (101, 5)}  # spread 2: joins
    long = p.decide(view(97, 101, position=10)).quotes  # long: does not step ahead to buy more
    assert long == {Side.BID: (97, 5), Side.ASK: (100, 5)}
    assert p.decide(view(97, 101, position=50)).quotes == {Side.ASK: (100, 5)}  # at its limit: only sells
    assert p.decide(view(bid=None)).quotes == {}
    assert PennyJumping.keeps_queue


# ---- Breakout ----

def test_breakout_enters_beyond_its_range_and_exits_past_the_middle():
    from lob.strategies import Breakout

    b = Breakout(window=3, size=5)
    for mid in (100, 101, 102):
        assert b.decide(at_mid(mid)).target == 0
    assert b.warming_up() is None
    plan = b.decide(at_mid(103))  # above the range 100 to 102
    assert (plan.target, plan.signal) == (5, pytest.approx(2.0))
    assert b.decide(at_mid(102.5)).target == 5  # inside 101 to 103, above its middle: holds
    assert b.decide(at_mid(102)).target == 0  # inside 102 to 103, below its middle 102.5: flat
    assert b.decide(at_mid(101)).target == -5  # below the range 102 to 103


# ---- Order flow imbalance ----

def test_order_flow_imbalance_adds_up_the_flow_at_the_best_quotes():
    from lob.strategies import OrderFlowImbalance

    o = OrderFlowImbalance(window=2, threshold=1.0, size=5)
    assert o.decide(view(99, 101, bid_depth=10, ask_depth=10)).target == 0  # the first look: nothing to compare
    assert o.decide(view(99, 101, bid_depth=20, ask_depth=10)).target == 0  # +10 at the bid; one change so far
    plan = o.decide(view(99, 101, bid_depth=30, ask_depth=5))  # +10 at the bid, -5 at the ask: buying
    # flows +10 and +15 over average depths of 15 and 17.5
    assert plan.signal == pytest.approx(25 / 16.25) and plan.target == 5
    plan = o.decide(view(100, 101, bid_depth=4, ask_depth=5))  # the bid moved up: +4 more
    assert plan.signal == pytest.approx(19 / ((17.5 + 4.5) / 2)) and plan.target == 5
    plan = o.decide(view(99, 101, bid_depth=30, ask_depth=5))  # the bid fell back: -4 for the lots that left
    assert plan.signal == pytest.approx(0) and plan.target == 5  # not below zero yet: holds
    plan = o.decide(view(99, 100, bid_depth=30, ask_depth=40))  # the ask came down with 40 lots: selling
    assert plan.signal < -1 and plan.target == -5


# ---- Random ----

def test_the_random_trader_repeats_for_its_seed():
    from lob.strategies import RandomTrader

    def targets(**settings):
        r = RandomTrader(**settings)
        return [r.decide(view()).target for _ in range(200)]

    assert targets(seed=3) == targets(seed=3) != targets(seed=4)
    assert set(targets(switch=0.0)) == {0}
    always = targets(switch=1.0, size=2)
    assert set(always) == {-2, 0, 2} and 50 < always.count(0) < 85  # a third of the draws, about


# ---- What every strategy remembers ----

@pytest.mark.parametrize("kind", list(STRATEGY_SPECS))
def test_every_strategy_takes_up_its_state_exactly(kind):
    import json
    import random

    spec = STRATEGY_SPECS[kind]
    rng = random.Random(kind)

    def views(n):
        bid = 100
        for _ in range(n):
            bid += rng.choice((-1, 0, 0, 1))
            yield View(0.0, bid, bid + rng.choice((1, 1, 2, 3, 4)), rng.randint(1, 40), rng.randint(1, 40),
                       rng.choice((-10, -5, 0, 5, 10)))

    first = spec.build(spec.defaults())
    for v in views(60):
        first.decide(v)
    again = spec.build(spec.defaults())
    again.restore(json.loads(json.dumps(first.state())))
    for v in views(60):
        a, b = first.decide(v), again.decide(v)
        assert (a.quotes, a.target, a.signal) == (b.quotes, b.target, b.signal)
    if first.state():  # one it could not have had is refused
        with pytest.raises(ValueError):
            again.restore({key: "nonsense" for key in first.state()})
