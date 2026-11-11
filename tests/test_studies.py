"""Tests for lob.studies: every study against an independent computation, on built and on recorded event streams.

The references are plain loops, numpy or scipy written here, not calls into the code under test: VPIN's buckets
are made by listing every lot, OFI's increments come from ``lob.metrics.order_flow_imbalance``, the windows are
summed by looking at every event, and depletion is checked by a literal, unfiltered rule. The recorded streams
come from ``Simulation(observer=...)``, which also feeds a spy that keeps every call's arguments.
"""

import dataclasses
import json
import math
from collections import deque
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app import facts
from lob import studies as st
from lob.book import OrderBook
from lob.matching import PriceTimeEngine, ProRataEngine
from lob.metrics import compute_order_imbalance, order_flow_imbalance, trades_by_order
from lob.models.strategic import ParlourModel
from lob.models.zero_intelligence import SmithFarmerModel
from lob.order import Fill, Side, cancel_order, limit_order, market_order, reset_order_id_counter
from lob.simulation import Simulation, SimulationConfig

BID, ASK = Side.BID, Side.ASK


@pytest.fixture(autouse=True)
def reset_ids():
    reset_order_id_counter()
    yield
    reset_order_id_counter()


class Stub:
    """A book that has only a mid, for the studies that sample nothing else."""

    def __init__(self, mid):
        self.mid_price = mid


def q(bid=100, bid_lots=10, ask=102, ask_lots=10):
    return (bid, bid_lots, ask, ask_lots)


def quotes_at(mid, lots=5):
    """Quotes whose mid is exactly ``mid`` (a multiple of a half)."""
    twice = int(round(2 * mid))
    bid, ask = (twice // 2 - 1, twice // 2 + 1) if twice % 2 == 0 else ((twice - 1) // 2, (twice + 1) // 2)
    return (bid, lots, ask, lots)


def fills_of(*pairs):
    return [Fill(1, 2, price, size, 0.0) for price, size in pairs]


def see(study, t, before=None, after=None, kind=st.ARRIVAL_LIMIT, side=BID, price=100, size=1, fills=(),
        aggressor=0, book=None, own=False):
    before = q() if before is None else before
    study.observe(t, kind, side, price, size, fills, aggressor, before, before if after is None else after,
                  book, own)


def trade(study, t, fill_pairs, aggressor, before=None, after=None, own=False):
    see(study, t, before, after, fills=fills_of(*fill_pairs), aggressor=aggressor, own=own)


# ---- Settings, switches, resets ----

def test_settings_are_parsed_against_their_ranges():
    window = st.OFI.SETTINGS[0]
    assert window.parse("7.5") == 7.5
    with pytest.raises(dataclasses.FrozenInstanceError):    # the studies of a class share their settings
        window.high = 1000.0
    for bad in (0.1, 101, float("nan"), float("inf"), "x", None, True):
        with pytest.raises(ValueError):
            window.parse(bad)
    levels = st.BookImbalance.SETTINGS[0]
    assert levels.parse(5) == 5 and levels.parse("5") == 5
    for bad in (0, 21, 2.5, True):
        with pytest.raises(ValueError):
            levels.parse(bad)
    side = st.TradeDelta.SETTINGS[1]
    assert side.parse("tick") == "tick"
    with pytest.raises(ValueError, match="choose true or tick or quote"):
        side.parse("signed")
    with pytest.raises(ValueError, match="no setting"):
        st.OFI(span=3)


def test_configure_changes_atomically_and_starts_afresh():
    ofi = st.OFI(window=2.0)
    see(ofi, 3.0, after=q(100, 25))
    assert ofi.value(3.0) == 15  # the bid queue grew by 15: buying pressure
    assert ofi.configure(window=2.0) is False and ofi.value(3.0) == 15  # nothing changed: nothing cleared
    with pytest.raises(ValueError):
        ofi.configure(window=4.0, nonsense=1)
    assert ofi.window == 2.0 and ofi.value(3.0) == 15  # a bad value changes nothing
    assert ofi.configure(window=4.0) is True
    assert ofi.window == 4.0
    assert ofi.value(3.0) is None  # afresh from the last time seen (3.0): the window of 4 is not full
    assert ofi.value(7.0) == 0

    with pytest.raises(ValueError, match="above"):
        st.Resilience(depleted=0.6, recovered=0.5)
    res = st.Resilience()
    with pytest.raises(ValueError, match="above"):
        res.configure(depleted=0.95)  # recovered 0.9 would no longer be above it
    assert res.depleted == 0.5


def test_reset_keeps_settings_and_clears_data():
    study_set = st.StudySet()
    study_set["vpin"].configure(buckets=7, volume=20)
    study_set["otr"].configure(scope="everyone")
    trade(study_set["cvd"], 1.0, [(101, 3)], 1)
    assert study_set["cvd"].value() == 3
    study_set.reset(40, 40.5)
    assert study_set["cvd"].value() == 0
    assert study_set["vpin"].buckets == 7 and study_set["vpin"].volume_in_use == 20
    assert study_set["otr"].scope == "everyone"
    assert study_set.unit == 40 and study_set["ofi"].start == 40.5


def test_a_study_that_is_off_sees_nothing_and_starts_afresh_when_on():
    class Spy(st.CVD):
        calls = 0

        def observe(self, *args, **kwargs):
            Spy.calls += 1
            super().observe(*args, **kwargs)

    spy = Spy()
    study_set = st.StudySet([spy, st.OFI()], auto_sample=False)
    q1 = fills_of((101, 4))
    study_set.observe(1.0, st.ARRIVAL_MARKET, BID, 0, 4, q1, 1, q(), q(), None)
    assert Spy.calls == 1 and spy.value() == 4
    study_set.enable("cvd", False)
    study_set.observe(2.0, st.ARRIVAL_MARKET, BID, 0, 4, q1, 1, q(), q(), None)
    assert Spy.calls == 1 and "cvd" not in study_set.values(2.0)
    study_set.enable("cvd", True)
    assert spy.value() == 0  # the 4 lots counted before it was switched off are gone, not carried over
    study_set.observe(3.0, st.ARRIVAL_MARKET, BID, 0, 4, q1, 1, q(), q(), None)
    assert Spy.calls == 2 and spy.value() == 4


def test_studies_that_need_fills_are_skipped_for_events_that_traded_nothing():
    calls = []

    class Spy(st.CVD):
        def observe(self, *args, **kwargs):
            calls.append(args[1])

    study_set = st.StudySet([Spy()], auto_sample=False)
    study_set.observe(1.0, st.ARRIVAL_LIMIT, BID, 99, 4, (), 1, q(), q(), None)
    study_set.observe(2.0, st.CANCEL, BID, 99, 4, (), 0, q(), q(), None)
    assert calls == []
    study_set.observe(3.0, st.ARRIVAL_MARKET, ASK, 0, 4, fills_of((100, 4)), -1, q(), q(), None)
    assert calls == [st.ARRIVAL_MARKET]


def test_the_studies_that_read_the_book_when_sampled_are_given_no_events():
    seen = []

    class Quiet(st.Spread):
        key = "quiet"

        def observe(self, *args, **kwargs):
            seen.append(args)

    study_set = st.StudySet([Quiet(), st.BookImbalance()], auto_sample=False)
    study_set.observe(1.0, st.ARRIVAL_LIMIT, BID, 99, 5, (), 1, q(), q(), None)
    study_set.observe(2.0, st.ARRIVAL_MARKET, ASK, 0, 5, fills_of((100, 5)), -1, q(), q(), None)
    assert seen == [] and not st.BookImbalance.sees_events and not st.Spread.sees_events
    assert all(s.sees_events for s in (st.OFI(), st.CVD(), st.TradeDelta(), st.VPIN(), st.OrderToTrade(),
                                       st.Resilience()))


def test_two_studies_cannot_share_a_key():
    with pytest.raises(ValueError, match="same key"):
        st.StudySet([st.OFI(), st.OFI()])


def test_the_kinds_of_event_are_three_different_numbers():
    assert len({st.ARRIVAL_LIMIT, st.ARRIVAL_MARKET, st.CANCEL}) == 3


def test_the_ofi_increment_the_facts_recorder_uses_is_the_library_s():
    assert facts.ofi_increment is st.ofi_increment


def test_book_quotes_reads_the_touch_and_the_lots_there():
    book = OrderBook()
    assert st.book_quotes(book) == (None, 0, None, 0)
    engine = PriceTimeEngine()
    for order in (limit_order(100, 4, 0.0, BID), limit_order(100, 6, 0.0, BID), limit_order(99, 9, 0.0, BID),
                  limit_order(103, 7, 0.0, ASK)):
        engine.process(book, order)
    assert st.book_quotes(book) == (100, 10, 103, 7)


# ---- Order flow imbalance ----

def random_states(rng, n):
    """A random walk of quote states (best bid, lots, best ask, lots), a side sometimes empty."""
    states = []
    for _ in range(n):
        bid = None if rng.random() < 0.08 else int(rng.integers(98, 102))
        ask = None if rng.random() < 0.08 else int(rng.integers(102, 106))
        states.append((bid, int(rng.integers(1, 30)) if bid is not None else 0,
                       ask, int(rng.integers(1, 30)) if ask is not None else 0))
    return states


def numpy_increments(states):
    """e_n for consecutive states, from lob.metrics.order_flow_imbalance (a missing side as NaN)."""
    nan = float("nan")
    pb, qb, pa, qa = (np.array(column, dtype=float) for column in zip(*[
        (nan if s[0] is None else s[0], s[1], nan if s[2] is None else s[2], s[3]) for s in states], strict=True))
    return order_flow_imbalance(pb, qb, pa, qa)


def test_ofi_increment_is_the_metrics_function():
    rng = np.random.default_rng(1)
    states = random_states(rng, 3000)
    expected = numpy_increments(states)
    got = [st.ofi_increment(a, b) for a, b in zip(states[:-1], states[1:], strict=True)]
    np.testing.assert_array_equal(got, expected)
    assert (expected != 0).sum() > 1000 and (expected < 0).any() and (expected > 0).any()


def test_ofi_increment_by_hand():
    assert st.ofi_increment(q(100, 10, 102, 10), q(100, 25, 102, 10)) == 15   # lots added at the bid
    assert st.ofi_increment(q(100, 10, 102, 10), q(100, 10, 102, 4)) == 6     # lots taken from the ask: buying
    assert st.ofi_increment(q(100, 10, 102, 10), q(101, 3, 102, 10)) == 3     # the bid improves: its new queue
    assert st.ofi_increment(q(100, 10, 102, 10), q(99, 8, 102, 10)) == -10    # the bid queue is gone
    assert st.ofi_increment(q(100, 10, 102, 10), q(100, 10, 101, 5)) == -5    # the ask improves: selling pressure
    assert st.ofi_increment(q(100, 10, None, 0), q(100, 20, 102, 5)) == 0     # an empty side: nothing


@pytest.mark.parametrize("window", [0.5, 5.0, 13.0])
def test_ofi_window_sums_the_increments_of_exactly_the_last_w_units(window):
    rng = np.random.default_rng(int(window * 10))
    n = 4000
    states = random_states(rng, n + 1)
    e = numpy_increments(states)
    times = np.cumsum(rng.choice([0.0, 0.0, 0.05, 0.3, 1.2, 4.0], size=n))  # events sharing a clock reading too
    ofi = st.OFI(window=window)
    for i in range(n):
        ofi.observe(times[i], st.ARRIVAL_LIMIT, BID, 100, 1, (), 1, states[i], states[i + 1], None)
        if i % 7 == 0:
            now = times[i]
            expected = e[: i + 1][(times[: i + 1] > now - window)].sum()
            got = ofi.value(now)
            if now - 0.0 < window:
                assert got is None
            else:
                assert got == pytest.approx(expected, abs=1e-9)
    # idle time drains the window; an event exactly W old is out
    last = times[-1]
    assert ofi.value(last + 0.4 * window) == pytest.approx(e[times > last + 0.4 * window - window].sum())
    assert ofi.value(last + window) == 0


def test_ofi_boundary_and_warm_up():
    ofi = st.OFI(window=5.0)
    see(ofi, 1.0, after=q(100, 20))      # +10
    see(ofi, 6.0, before=q(100, 20), after=q(100, 20))  # the quotes did not change: 0
    assert ofi.value(4.9) is None        # started at 0: not full until t = 5
    assert ofi.value(5.0) == 10          # (0, 5] holds the event at t = 1
    assert ofi.value(5.99) == 10
    assert ofi.value(6.0) == 0           # an event exactly W old is outside (t - W, t]
    ofi.reset(100)
    assert ofi.value(104.0) is None and ofi.value(105.0) == 0


def test_ofi_does_nothing_when_the_quotes_object_is_the_same():
    ofi = st.OFI()
    same = q()
    ofi.observe(1.0, st.CANCEL, BID, 90, 3, (), 0, same, same, None)
    assert ofi._window.items == deque()


# ---- Cumulative volume delta ----

def test_cvd_counts_buyer_less_seller_initiated_lots():
    cvd = st.CVD()
    assert cvd.value() == 0
    trade(cvd, 1.0, [(101, 3), (102, 4)], 1)
    trade(cvd, 2.0, [(100, 10)], -1)
    trade(cvd, 3.0, [(101, 2)], 1)
    assert cvd.value() == 3 + 4 - 10 + 2
    cvd.clear()
    assert cvd.value() == 0
    trade(cvd, 4.0, [(100, 5)], -1)
    assert cvd.value() == -5
    cvd.reset(3)
    assert cvd.value() == 0


# ---- Delta of trades ----

def tick_signs(prices):
    """The tick rule written out: the sign of the last change of price, unknown before the first."""
    signs = []
    for i, price in enumerate(prices):
        if i == 0:
            signs.append(None)
        elif price != prices[i - 1]:
            signs.append(1 if price > prices[i - 1] else -1)
        else:
            signs.append(signs[-1])
    return signs


def test_delta_with_true_sides_sums_signed_lots_over_the_window():
    delta = st.TradeDelta(window=4.0, side="true")
    trade(delta, 1.0, [(101, 3), (102, 2)], 1)
    trade(delta, 2.0, [(100, 4)], -1)
    assert delta.value(3.0) is None                    # a window of 4 is not yet full
    assert delta.value(4.0) == 5 - 4                   # (0, 4]
    assert delta.value(5.9) == -4                      # (1.9, 5.9]: the buys at t = 1 have left
    assert delta.value(6.0) == 0                       # (2, 6]: an event exactly W old is out
    trade(delta, 6.0, [(101, 1)], 1)
    assert delta.value(6.0) == 1
    values = delta.values(6.0)
    assert values["coverage"] is None and values["accuracy"] is None   # nothing to score with the true side


def test_delta_with_true_sides_after_idle_time():
    delta = st.TradeDelta(window=4.0)
    trade(delta, 1.0, [(101, 3)], 1)
    assert delta.value(4.9) == 3 and delta.value(5.0) == 0


def test_delta_tick_rule_signs_each_fill_by_the_last_change_of_price():
    prices = [100, 100, 101, 101, 101, 100, 100, 102]
    signs = tick_signs(prices)
    assert signs == [None, None, 1, 1, 1, -1, -1, 1]
    sizes = [2, 3, 5, 7, 1, 4, 6, 8]
    aggressors = [1, -1, 1, -1, 1, 1, -1, 1]
    delta = st.TradeDelta(window=100.0, side="tick")
    for i, (p, n, a) in enumerate(zip(prices, sizes, aggressors, strict=True)):
        trade(delta, float(i + 1), [(p, n)], a)
        known = [(s, sizes[j], aggressors[j]) for j, s in enumerate(signs[: i + 1]) if s is not None]
        values = delta.values(float(i + 1))
        total = sum(sizes[: i + 1])
        assert values["coverage"] == pytest.approx(sum(n for _, n, _ in known) / total)
        if known:
            assert values["accuracy"] == pytest.approx(sum(n for s, n, a in known if s == a)
                                                       / sum(n for _, n, _ in known))
        else:
            assert values["accuracy"] is None
    assert delta.value(100.0) == sum(s * n for s, n, _ in known)  # 5 + 7 + 1 - 4 - 6 + 8


def test_delta_tick_rule_works_through_a_multi_fill_event_and_carries_its_state_between_events():
    delta = st.TradeDelta(window=100.0, side="tick")
    trade(delta, 1.0, [(100, 5)], 1)                          # unknown: the first fill
    trade(delta, 2.0, [(101, 2), (102, 3), (102, 4)], 1)      # up, up, same: three buys
    trade(delta, 3.0, [(102, 6)], -1)                         # the same price after a buy: a buy (wrong)
    trade(delta, 4.0, [(101, 1)], -1)                         # down: a sell
    assert delta.value(100.0) == 2 + 3 + 4 + 6 - 1
    values = delta.values(100.0)
    assert values["coverage"] == pytest.approx(16 / 21)
    assert values["accuracy"] == pytest.approx((2 + 3 + 4 + 1) / 16)


def test_delta_quote_rule_is_exact_with_lag_zero():
    delta = st.TradeDelta(window=100.0, side="quote", quote_lag=0)
    trade(delta, 1.0, [(102, 4)], 1, before=q(100, 10, 102, 10))                    # a fill at the ask: above the mid
    trade(delta, 2.0, [(100, 3), (99, 2)], -1, before=q(100, 10, 102, 10))          # fills at the bid: below it
    trade(delta, 3.0, [(105, 6)], 1, before=q(103, 5, 105, 9))
    values = delta.values(100.0)
    assert values["coverage"] == 1.0 and values["accuracy"] == 1.0
    assert delta.value(100.0) == 4 - 5 + 6
    trade(delta, 4.0, [(102, 3)], 1, before=q(None, 0, 102, 3))                     # one-sided: no mid, unknown
    assert delta.values(100.0)["coverage"] == pytest.approx(15 / 18)


def test_delta_quote_rule_reads_the_mid_sampled_lag_units_back_and_falls_back_on_the_tick_rule():
    # the mid of unit u is 99 + u: 100, 101, ...
    def last_fill(lag, t, price, prior=None):
        """The signed lots the fill at ``price`` adds at time ``t``, after a ``prior`` fill if given."""
        delta = st.TradeDelta(window=100.0, side="quote", quote_lag=lag)
        for u in range(1, int(t) + 1):
            delta.sample(u, Stub(99.0 + u))
        if prior is not None:
            trade(delta, t - 0.1, [(prior, 1)], 1)
        before = delta.value(100.0)
        trade(delta, t, [(price, 10)], 1)
        return delta.value(100.0) - before

    # a fill at t = 5.3: lag 1 reads the sample of unit 5 (mid 104), lag 2 unit 4 (103), lag 3 unit 3 (102)
    assert last_fill(1, 5.3, 105) == 10 and last_fill(1, 5.3, 103) == -10
    assert last_fill(2, 5.3, 104) == 10 and last_fill(2, 5.3, 102) == -10
    assert last_fill(3, 5.3, 103) == 10 and last_fill(3, 5.3, 101) == -10
    # at the mid the tick rule decides: unknown without a fill before, the sign of the last change otherwise
    assert last_fill(1, 5.3, 104) == 0
    assert last_fill(1, 5.3, 104, prior=103) == 10      # the previous fill was at 103: a rise to 104 is a buy
    assert last_fill(1, 5.3, 104, prior=105) == -10     # a fall to 104 a sell
    assert last_fill(1, 5.3, 104, prior=104) == 0       # no change of price yet: still unknown


def test_delta_quote_rule_does_not_know_a_fill_before_its_sample_exists():
    delta = st.TradeDelta(window=100.0, side="quote", quote_lag=2)
    trade(delta, 0.5, [(105, 4)], 1)                      # before any sample
    delta.sample(1, Stub(100.0))
    trade(delta, 1.5, [(105, 4)], 1)                      # lag 2 needs the sample of unit 0, which was not taken
    delta.sample(2, Stub(100.0))
    trade(delta, 2.5, [(105, 4)], 1)                      # unit 1's sample exists
    assert delta.value(100.0) == 4
    assert delta.values(100.0)["coverage"] == pytest.approx(1 / 3)
    delta.sample(3, Stub(None))                           # a one-sided book at the sample: no mid
    trade(delta, 3.5, [(105, 4)], 1)                      # lag 2 reads unit 2's sample: known
    trade(delta, 4.5, [(105, 4)], 1)                      # lag 2 reads unit 3's sample, which has no mid: unknown
    assert delta.value(100.0) == 8
    assert delta.values(100.0)["coverage"] == pytest.approx(2 / 5)


def test_delta_with_a_one_sided_sample_has_no_mid_to_compare_with():
    delta = st.TradeDelta(window=100.0, side="quote", quote_lag=1)
    delta.sample(1, Stub(None))
    trade(delta, 1.5, [(105, 4)], 1)
    assert delta.value(100.0) == 0 and delta.values(100.0)["coverage"] == 0.0


# ---- VPIN ----

def lots_of(events, sign_of):
    """Every lot of every fill in order, as (sign, index of the event it belongs to)."""
    lots = []
    for i, (fills, aggressor) in enumerate(events):
        for price, size in fills:
            sign = sign_of(i, price, aggressor)
            lots += [(sign, i)] * size
    return lots


def vpin_by_lots(events, v, n, sign_of):
    """VPIN made the plain way: list every lot, cut buckets of v lots, score each by its signed lots."""
    lots = lots_of(events, sign_of)
    imbalances = []
    for b in range(len(lots) // v):
        bucket = [s for s, _ in lots[b * v:(b + 1) * v]]
        imbalances.append(abs(bucket.count(1) - bucket.count(-1)) / v)
    return (sum(imbalances[-n:]) / n if len(imbalances) >= n else None), len(imbalances)


def random_events(rng, count, max_fills=3, max_size=25):
    events = []
    price = 100
    for _ in range(count):
        fills = []
        for _ in range(int(rng.integers(1, max_fills + 1))):
            price += int(rng.choice([-1, 0, 0, 1]))
            fills.append((price, int(rng.integers(1, max_size + 1))))
        events.append((fills, int(rng.choice([-1, 1]))))
    return events


@pytest.mark.parametrize("side", ["true", "tick"])
@pytest.mark.parametrize("v,n", [(10, 3), (7, 5), (40, 4)])
def test_vpin_equals_vpin_made_by_listing_every_lot(side, v, n):
    rng = np.random.default_rng(v * 100 + n)
    events = random_events(rng, 400)
    flat_prices = [price for fills, _ in events for price, _ in fills]
    tick = iter(tick_signs(flat_prices))
    tick_by_fill = {}
    for i, (fills, _) in enumerate(events):
        for j in range(len(fills)):
            tick_by_fill[(i, j)] = next(tick)

    def true_sign(i, price, aggressor):
        return aggressor

    # the tick sign of a fill, found by its place: rebuild the lots with that sign
    def lots_with_tick():
        out = []
        for i, (fills, _) in enumerate(events):
            for j, (_, size) in enumerate(fills):
                out += [(tick_by_fill[(i, j)], i)] * size
        return out

    vpin = st.VPIN(buckets=n, volume=v, side=side)
    seen = []
    for i, (fills, aggressor) in enumerate(events):
        see(vpin, float(i), fills=fills_of(*fills), aggressor=aggressor)
        seen.append((fills, aggressor))
        if side == "true":
            expected, closed = vpin_by_lots(seen, v, n, true_sign)
        else:
            lots = [lot for lot in lots_with_tick() if lot[1] <= i]
            imbalances = []
            for b in range(len(lots) // v):
                bucket = [s for s, _ in lots[b * v:(b + 1) * v]]
                imbalances.append(abs(bucket.count(1) - bucket.count(-1)) / v)
            expected = sum(imbalances[-n:]) / n if len(imbalances) >= n else None
            closed = len(imbalances)
        values = vpin.values()
        assert values["buckets"] == closed
        if expected is None:
            assert values["vpin"] is None
        else:
            assert values["vpin"] == pytest.approx(expected, abs=1e-12)
    assert closed > n and values["vpin"] is not None


def test_a_fill_that_crosses_a_bucket_end_is_split_between_the_buckets():
    vpin = st.VPIN(buckets=2, volume=10, side="true")
    see(vpin, 1.0, fills=fills_of((101, 6)), aggressor=1)
    assert vpin.values()["buckets"] == 0
    see(vpin, 2.0, fills=fills_of((100, 25)), aggressor=-1)       # 4 lots finish a bucket, then 10, 10, 1 left
    assert vpin.values()["buckets"] == 3
    # buckets: 6 buys + 4 sells (0.2), 10 sells (1), 10 sells (1); the last 2 count
    assert vpin.value() == pytest.approx((1 + 1) / 2)
    see(vpin, 3.0, fills=fills_of((101, 9)), aggressor=1)         # 1 sell + 9 buys: imbalance 0.8
    assert vpin.values()["buckets"] == 4
    assert vpin.value() == pytest.approx((1 + 0.8) / 2)


def test_vpin_is_none_until_n_buckets_are_full():
    vpin = st.VPIN(buckets=3, volume=10, side="true")
    for k in range(1, 4):
        see(vpin, float(k), fills=fills_of((101, 10)), aggressor=1)
        assert (vpin.value() is None) == (k < 3)
    assert vpin.value() == 1.0
    see(vpin, 4.0, fills=fills_of((100, 10)), aggressor=-1)
    assert vpin.value() == 1.0   # |imbalance| of a one-sided bucket is 1 whichever side


def test_vpin_tick_rule_leaves_unknown_fills_out_of_both_sides():
    vpin = st.VPIN(buckets=1, volume=10, side="tick", log_buckets=True)
    see(vpin, 1.0, fills=fills_of((100, 4)), aggressor=1)         # unknown: fills the bucket, enters neither side
    see(vpin, 2.0, fills=fills_of((100, 3)), aggressor=-1)        # unknown: the same price after an unknown
    see(vpin, 3.0, fills=fills_of((101, 3)), aggressor=-1)        # a rise: a buy of 3
    assert vpin.value() == pytest.approx(3 / 10)
    # imbalance, inferred buy share, true buy share
    assert vpin.log[0][1:] == (pytest.approx(0.3), pytest.approx(1.0), pytest.approx(0.4))


def test_vpin_calibrates_its_bucket_on_the_first_200_fills_and_those_fills_fill_no_bucket():
    vpin = st.VPIN(buckets=2)
    assert vpin.volume_in_use is None
    sizes = [3, 5, 5, 7] * 50          # median 5
    for i, size in enumerate(sizes):
        see(vpin, float(i), fills=fills_of((100 + i % 2, size)), aggressor=1)
        assert (vpin.volume_in_use is None) == (i < 199)
    assert vpin.volume_in_use == 250.0
    assert vpin.values()["buckets"] == 0 and vpin.value() is None
    for i in range(10):                  # 10 fills of 25 lots: exactly one bucket
        see(vpin, 300.0 + i, fills=fills_of((100, 25)), aggressor=1)
    assert vpin.values()["buckets"] == 1 and vpin.volume_in_use == 250.0
    vpin.reset()
    assert vpin.volume_in_use is None   # reset: calibrate again
    vpin.set_volume(80)
    assert vpin.volume_in_use == 80.0
    with pytest.raises(ValueError):
        vpin.set_volume(0.5)
    with pytest.raises(ValueError):
        vpin.set_volume(float("nan"))
    fixed = st.VPIN(volume=33)
    assert fixed.volume_in_use == 33.0
    fixed.reset()
    assert fixed.volume_in_use == 33.0   # a fixed bucket does not need calibrating


def test_vpin_with_true_sides_on_fair_coin_flow_sits_near_sqrt_2_over_pi_n():
    rng = np.random.default_rng(3)
    vpin = st.VPIN(buckets=50, volume=50, side="true")
    for i in range(60_000):
        see(vpin, float(i), fills=fills_of((100, 1)), aggressor=int(rng.choice([-1, 1])))
    assert vpin.value() == pytest.approx(math.sqrt(2 / (math.pi * 50)), abs=0.02)   # 0.113


def bvc_by_numpy(events, v, n, mids_before, mids_after):
    """BVC-VPIN written with numpy and scipy: bucket mid changes, sigma of the last 100, Phi.

    A bucket's change is the mid after the event that closed it less the mid after the event that closed the one
    before (for the first, the mid before its first fill); a missing mid counts as a change of nothing.
    """
    lots = lots_of(events, lambda i, p, a: 0)
    closing = [lots[(b + 1) * v - 1][1] for b in range(len(lots) // v)]
    mid = mids_before[lots[0][1]]
    changes = []
    for i in closing:
        end = mids_after[i]
        changes.append(0.0 if end is None or mid is None else end - mid)
        mid = end if end is not None else mid
    changes = np.array(changes)
    imbalances, shares = [], []
    for b in range(len(changes)):
        window = changes[max(0, b - 99): b + 1]
        sigma = np.std(window, ddof=1) if len(window) > 1 else 0.0
        x = changes[b] / sigma if sigma > 0 else 0.0
        shares.append(stats.norm.cdf(x))
        imbalances.append(abs(2 * stats.norm.cdf(x) - 1))
    return (float(np.mean(imbalances[-n:])) if len(imbalances) >= n else None), imbalances, shares


def test_bvc_vpin_equals_numpy_phi_of_the_bucket_price_change_over_sigma():
    rng = np.random.default_rng(5)
    v, n = 30, 5
    events, mids_before, mids_after = [], [], []
    mid = 100.0
    vpin = st.VPIN(buckets=n, volume=v, side="bvc", log_buckets=True)
    for i in range(1500):
        fills = [(100, int(rng.integers(1, 25)))]
        before_mid = mid
        mid += float(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
        events.append((fills, int(rng.choice([-1, 1]))))
        mids_before.append(before_mid)
        mids_after.append(mid)
        vpin.observe(float(i), st.ARRIVAL_MARKET, BID, 0, 1, fills_of(*fills), events[-1][1],
                     quotes_at(before_mid), quotes_at(mid), None)
    expected, imbalances, shares = bvc_by_numpy(events, v, n, mids_before, mids_after)
    assert expected is not None and len(imbalances) > 100
    assert vpin.value() == pytest.approx(expected, abs=1e-9)
    np.testing.assert_allclose([row[1] for row in vpin.log], imbalances, atol=1e-9)
    # the buy share a bucket is given is Phi(x), and its true share is the buy lots in it over V
    np.testing.assert_allclose([row[2] for row in vpin.log], shares, atol=1e-9)
    lots = lots_of(events, lambda i, p, a: a)
    true_shares = [sum(1 for sign, _ in lots[b * v:(b + 1) * v] if sign == 1) / v for b in range(len(lots) // v)]
    np.testing.assert_allclose([row[3] for row in vpin.log], true_shares, atol=1e-12)


def test_bvc_vpin_is_about_one_half_whatever_the_flow():
    """|2 Phi(x) - 1| is uniform on (0, 1) when x is normal, so BVC-VPIN is 1/2 even for one-sided flow."""
    rng = np.random.default_rng(8)
    one_sided = st.VPIN(buckets=50, volume=40, side="bvc")
    coin = st.VPIN(buckets=50, volume=40, side="bvc")
    mid = 100.0
    for i in range(40_000):
        before_mid, mid = mid, mid + float(rng.normal(0, 0.4))
        for study, aggressor in ((one_sided, 1), (coin, int(rng.choice([-1, 1])))):
            study.observe(float(i), st.ARRIVAL_MARKET, BID, 0, 1, fills_of((100, 4)), aggressor,
                          quotes_at(round(before_mid * 2) / 2), quotes_at(round(mid * 2) / 2), None)
    assert one_sided.value() == pytest.approx(0.5, abs=0.08)
    assert coin.value() == pytest.approx(0.5, abs=0.08)


def test_vpin_mid_rule_is_the_true_side_with_lag_zero_and_reads_the_lagged_mid_otherwise():
    rng = np.random.default_rng(2)
    events = random_events(rng, 300, max_fills=1)
    true_side = st.VPIN(buckets=3, volume=20, side="true")
    mid_rule = st.VPIN(buckets=3, volume=20, side="mid", quote_lag=0)
    for i, (fills, aggressor) in enumerate(events):
        price = fills[0][0]
        # a buyer's fills are at asks and a seller's at bids: the pre-event mid is on the other side of the price
        before = q(price - 1, 5, price + 3, 5) if aggressor < 0 else q(price - 3, 5, price + 1, 5)
        for study in (true_side, mid_rule):
            see(study, float(i), before=before, fills=fills_of(*fills), aggressor=aggressor)
    assert mid_rule.value() == true_side.value() is not None

    lagged = st.VPIN(buckets=1, volume=10, side="mid", quote_lag=1)
    lagged.sample(1, Stub(100.0))
    see(lagged, 1.5, fills=fills_of((100, 4)), aggressor=-1)     # at the mid: VisualHFT counts a price >= mid a buy
    see(lagged, 1.6, fills=fills_of((99, 6)), aggressor=1)       # below the lagged mid: a sell
    assert lagged.value() == pytest.approx(abs(4 - 6) / 10)


# ---- Order-to-trade ratio and trade-to-order volume ----

def test_otr_counts_messages_per_trade_where_a_trade_is_an_order_that_filled():
    otr = st.OrderToTrade(window=10.0, scope="model")
    see(otr, 1.0, kind=st.ARRIVAL_LIMIT, size=5)
    see(otr, 2.0, kind=st.CANCEL, size=3)
    see(otr, 3.0, kind=st.ARRIVAL_MARKET, size=4, fills=fills_of((101, 2), (102, 2)), aggressor=1)  # 2 fills: 1 trade
    see(otr, 4.0, kind=st.ARRIVAL_LIMIT, size=10, own=True)                                          # not the model's
    see(otr, 5.0, kind=st.ARRIVAL_LIMIT, size=6, fills=fills_of((101, 3)), aggressor=1)             # marketable limit
    values = otr.values(10.0)
    assert values == {"otr": 4 / 2, "tto": 7 / 15, "messages": 4, "trades": 2}
    everyone = st.OrderToTrade(window=10.0, scope="everyone")
    for t, kind, size, fills, own in ((1.0, st.ARRIVAL_LIMIT, 5, (), False), (2.0, st.CANCEL, 3, (), False),
                                      (3.0, st.ARRIVAL_MARKET, 4, fills_of((101, 2), (102, 2)), False),
                                      (4.0, st.ARRIVAL_LIMIT, 10, (), True),
                                      (5.0, st.ARRIVAL_LIMIT, 6, fills_of((101, 3)), False)):
        see(everyone, t, kind=kind, size=size, fills=fills, aggressor=1, own=own)
    assert everyone.values(10.0) == {"otr": 5 / 2, "tto": 7 / 25, "messages": 5, "trades": 2}
    # the window moves with time: at t = 11.5 the events at 1.0 (and not 2.0) have left
    assert otr.values(11.5) == {"otr": 3 / 2, "tto": 7 / 10, "messages": 3, "trades": 2}
    assert otr.values(13.0) == {"otr": 1 / 1, "tto": 3 / 6, "messages": 1, "trades": 1}
    assert otr.values(15.0)["otr"] is None and otr.values(15.0)["tto"] is None   # nothing left in the window


def test_otr_is_none_before_its_window_is_full_and_with_no_trades():
    otr = st.OrderToTrade(window=10.0)
    see(otr, 1.0, kind=st.ARRIVAL_LIMIT, size=5, fills=fills_of((101, 5)), aggressor=1)
    assert set(otr.values(9.9).values()) == {None}
    assert otr.values(10.0)["otr"] == 1.0
    quiet = st.OrderToTrade(window=10.0)
    see(quiet, 1.0, kind=st.ARRIVAL_LIMIT, size=5)
    assert quiet.values(10.0) == {"otr": None, "tto": 0.0, "messages": 1, "trades": 0}


def test_otr_counts_a_strategy_market_order_and_the_cancel_of_its_remainder_as_two_messages_for_one_trade():
    otr = st.OrderToTrade(window=10.0, scope="everyone")
    see(otr, 1.0, kind=st.ARRIVAL_LIMIT, size=8, fills=fills_of((101, 5)), aggressor=1, own=True)  # protected price
    see(otr, 1.0, kind=st.CANCEL, size=3, own=True)                                                 # the remainder
    values = otr.values(10.0)
    assert values["messages"] == 2 and values["trades"] == 1 and values["otr"] == 2.0
    assert values["tto"] == 5 / 8


def test_otr_window_matches_a_look_at_every_event():
    rng = np.random.default_rng(4)
    otr = st.OrderToTrade(window=6.0, scope="everyone")
    seen = []
    t = 0.0
    for _ in range(3000):
        t += float(rng.choice([0.0, 0.01, 0.1, 0.5, 3.0]))
        traded = bool(rng.random() < 0.15)
        cancel = not traded and rng.random() < 0.4
        size = int(rng.integers(1, 20))
        fills = fills_of((101, int(rng.integers(1, size + 1)))) if traded else ()
        kind = st.CANCEL if cancel else st.ARRIVAL_LIMIT
        see(otr, t, kind=kind, size=size, fills=fills, aggressor=1)
        seen.append((t, 0 if cancel else size, sum(f.size for f in fills), bool(fills)))
        inside = [e for e in seen if e[0] > t - 6.0]
        trades = sum(e[3] for e in inside)
        if t >= 6.0:
            values = otr.values(t)
            assert values["messages"] == len(inside) and values["trades"] == trades
            assert values["otr"] == (len(inside) / trades if trades else None)
            arrived = sum(e[1] for e in inside)
            assert values["tto"] == (sum(e[2] for e in inside) / arrived if arrived else None)


# ---- Book imbalance ----

def make_book(bids, asks, tick_size=1):
    book = OrderBook(tick_size=tick_size)
    engine = PriceTimeEngine()
    for side, levels in ((BID, bids), (ASK, asks)):
        for price, lots in levels:
            engine.process(book, limit_order(price, lots, 0.0, side))
    return book


BIDS = [(100, 10), (99, 20), (98, 5)]
ASKS = [(102, 4), (103, 6), (105, 30)]


def test_equal_weight_imbalance_by_hand_and_against_the_metrics_function():
    book = make_book(BIDS, ASKS)
    for levels, expected in ((1, (10 - 4) / 14), (2, (30 - 10) / 40), (3, (35 - 40) / 75), (10, (35 - 40) / 75)):
        imbalance = st.BookImbalance(levels=levels)
        assert imbalance.compute(book) == pytest.approx(expected)
        assert imbalance.value(book=book) == pytest.approx(compute_order_imbalance(book, levels))
    mirror = make_book([(100 - (p - 102), n) for p, n in ASKS], [(102 + (100 - p), n) for p, n in BIDS])
    assert st.BookImbalance(levels=3).compute(mirror) == pytest.approx(-st.BookImbalance(levels=3).compute(book))


def test_imbalance_is_none_for_an_empty_or_one_sided_book():
    imbalance = st.BookImbalance()
    assert imbalance.compute(OrderBook()) is None
    assert imbalance.compute(make_book(BIDS, [])) is None
    assert imbalance.compute(make_book([], ASKS)) is None
    touch = st.BookImbalance(weighting="touch")
    assert touch.compute(make_book(BIDS, [])) is None
    assert imbalance.value(book=None) is None   # nothing sampled yet


def touch_weighted(book, levels, scale, tick=1):
    """(B - A) / (B + A) with weights (1 + d / S)^-2, d the distance in ticks from the side's own best quote."""
    bids = book.depth_profile(BID, levels)
    asks = book.depth_profile(ASK, levels)
    b = sum(n * (1 + (bids[0][0] - p) / tick / scale) ** -2 for p, n in bids)
    a = sum(n * (1 + (p - asks[0][0]) / tick / scale) ** -2 for p, n in asks)
    return (b - a) / (b + a)


def test_touch_weighted_imbalance_uses_the_median_spread_once_it_has_20_samples():
    book = make_book(BIDS, ASKS)                       # a spread of 2
    wide = make_book([(96, 5)], [(104, 5)])            # a spread of 8
    imbalance = st.BookImbalance(levels=3, weighting="touch")
    # warm-up: the current spread (at least one tick) is the scale
    assert imbalance.compute(book) == pytest.approx(touch_weighted(book, 3, 2.0))
    assert imbalance.compute(wide) == pytest.approx(touch_weighted(wide, 3, 8.0))
    for u in range(1, 20):
        imbalance.sample(u, wide)
    assert imbalance.compute(book) == pytest.approx(touch_weighted(book, 3, 2.0))   # 19 samples: still the warm-up
    imbalance.sample(20, wide)
    assert imbalance.scale(book) == 8.0
    assert imbalance.compute(book) == pytest.approx(touch_weighted(book, 3, 8.0))
    assert imbalance.values(book=book) == {"imbalance": pytest.approx(touch_weighted(book, 3, 8.0)), "scale": 8.0}
    # the median of the last 100: 90 narrow samples push the wide ones out of the middle
    for u in range(21, 111):
        imbalance.sample(u, book)
    assert imbalance.scale(book) == 2.0
    assert imbalance.compute(book) == pytest.approx(touch_weighted(book, 3, 2.0))
    assert imbalance.sample(111, book) == pytest.approx(touch_weighted(book, 3, 2.0))
    assert imbalance.value(book=None) == imbalance.sample(112, book)    # the last sample answers without a book


def test_touch_weighting_counts_distance_in_ticks():
    book = make_book([(100, 10), (95, 20), (90, 5)], [(110, 4), (115, 6)], tick_size=5)   # a spread of 2 ticks
    imbalance = st.BookImbalance(levels=3, weighting="touch")
    assert imbalance.compute(book) == pytest.approx(touch_weighted(book, 3, 2.0, tick=5))
    for u in range(1, 25):
        imbalance.sample(u, book)
    assert imbalance.scale(book) == 2.0


def test_touch_weighting_with_a_very_small_scale_is_the_touch_imbalance():
    book = make_book(BIDS, ASKS)
    imbalance = st.BookImbalance(levels=3, weighting="touch")
    tight = make_book([(100, 1)], [(101, 1)])
    for u in range(1, 25):
        imbalance.sample(u, tight)           # S = 1 tick
    assert imbalance.compute(book) == pytest.approx(touch_weighted(book, 3, 1.0))


# ---- Spread ----

class SpreadBook:
    def __init__(self, spread, tick_size=1):
        self.spread = spread
        self.tick_size = tick_size
        self.mid_price = None


def test_spread_z_score_is_against_the_last_100_samples_before_it():
    rng = np.random.default_rng(6)
    spread = st.Spread()
    samples = []
    for u in range(1, 260):
        s = int(rng.integers(1, 7))
        z = spread.values(book=SpreadBook(s))["z"]
        window = samples[-100:]
        if len(window) < 20 or np.std(window) == 0:
            assert z is None
        else:
            assert z == pytest.approx((s - np.mean(window)) / np.std(window))
        assert spread.sample(u, SpreadBook(s)) == s
        samples.append(s)
    assert len(spread._samples) == 100
    values = spread.values(book=SpreadBook(20))
    assert values["shock"] is True and values["spread"] == 20


def test_spread_has_no_z_score_when_the_spread_never_varies_or_a_side_is_empty():
    spread = st.Spread()
    for u in range(1, 40):
        spread.sample(u, SpreadBook(2))
    assert spread.values(book=SpreadBook(2)) == {"spread": 2, "z": None, "shock": None}
    assert spread.values(book=SpreadBook(3))["z"] is None
    assert spread.values(book=SpreadBook(None)) == {"spread": None, "z": None, "shock": None}
    assert spread.sample(40, SpreadBook(None)) is None
    assert len(spread._samples) == 39               # an empty side is not a sample
    assert spread.values() == {"spread": None, "z": None, "shock": None}


def test_spread_is_in_ticks_and_the_shock_level_is_a_setting():
    spread = st.Spread(shock=1.0)
    for u in range(1, 31):
        spread.sample(u, SpreadBook(10 if u % 2 else 20, tick_size=5))   # 2 and 4 ticks, mean 3, sd 1
    values = spread.values(book=SpreadBook(20, tick_size=5))
    assert values["spread"] == 4 and values["z"] == pytest.approx(1.0) and values["shock"] is True
    spread.configure(shock=1.5)
    assert spread.values(book=SpreadBook(20, tick_size=5)) == {"spread": 4, "z": None, "shock": None}  # started afresh


# ---- Depth depletion, recovery and the restoration of the quote ----

class Rig:
    """The session's hook, in miniature: an engine and a book, the last quotes, and one observe call per event.

    ``shortcut`` passes the same quotes object when nothing traded and the order is strictly behind its side's
    touch, as the dashboard does; without it every event gets fresh quotes.
    """

    def __init__(self, observe, book, engine=None, shortcut=True):
        self.observe, self.book, self.engine = observe, book, engine or PriceTimeEngine()
        self.shortcut = shortcut
        self.quotes = st.book_quotes(book)

    def send(self, order):
        book, before = self.book, self.quotes
        cancel = order.order_type.value == "cancel"
        canceled = book.get_order(order.order_id) if cancel else None
        size = order.size
        fills = self.engine.process(book, order)
        if cancel:
            if canceled is None:
                return fills
            side, price, size, kind, aggressor, fills = canceled.side, canceled.price, canceled.size, st.CANCEL, 0, ()
        else:
            side, price = order.side, order.price
            kind = st.ARRIVAL_MARKET if order.order_type.value == "market" else st.ARRIVAL_LIMIT
            aggressor = 1 if side is BID else -1
        touch = before[0] if side is BID else before[2]
        behind = touch is not None and (price < touch if side is BID else price > touch)
        if self.shortcut and not fills and behind and kind != st.ARRIVAL_MARKET:
            after = before
        else:
            after = st.book_quotes(book)
        self.observe(order.timestamp, kind, side, price, size, fills, aggressor, before, after, book)
        self.quotes = after
        return fills


def resilience_rig(shortcut=True, bid_prices=(100, 99, 98, 97), ask_prices=(101, 102, 103, 104), extra_bids=(),
                   settings=None, samples=25, tick_size=1):
    """Bids and asks of 100 lots at four prices each, and 25 unit samples: a baseline of 300 lots at three prices."""
    book, engine = OrderBook(tick_size=tick_size), PriceTimeEngine()
    ids = {}
    for price in bid_prices:
        order = limit_order(price, 100, 0.0, BID)
        engine.process(book, order)
        ids[("bid", price)] = order.order_id
    for price in ask_prices:
        order = limit_order(price, 100, 0.0, ASK)
        engine.process(book, order)
        ids[("ask", price)] = order.order_id
    for price, lots in extra_bids:
        engine.process(book, limit_order(price, lots, 0.0, BID))
    res = st.Resilience(**(settings or {}))
    for u in range(1, samples + 1):
        res.sample(u, book)
    return res, Rig(res.observe, book, engine, shortcut), ids


@pytest.mark.parametrize("shortcut", [True, False])
def test_a_take_that_leaves_the_near_touch_depth_low_starts_an_episode_that_ends_when_it_is_back(shortcut):
    res, rig, _ = resilience_rig(shortcut)
    assert res.values(26.0)["score"] == 1.0 and res.values(26.0)["baseline_bid"] == 300.0
    rig.send(market_order(160, 30.0, ASK))             # takes 100@100 and 60@99: 0 + 40 + 100 at 100, 99, 98
    values = res.values(30.5)
    assert values["started"] == 1 and values["depleted_bid"] == 0.5 and values["depleted_ask"] is None
    assert values["score"] is None                     # depleted, and no recovery to compare with yet
    rig.send(limit_order(100, 100, 33.0, BID))         # 240 of the 270 that recovery needs; the quote is back
    assert res.values(33.0)["recovered"] == 0
    assert res.values(33.0)["restored_share"] == 1.0 and res.values(33.0)["median_restoration"] == 3.0
    rig.send(limit_order(99, 60, 36.0, BID))           # 300: recovered, six units after it started
    values = res.values(36.0)
    assert (values["started"], values["recovered"], values["abandoned"]) == (1, 1, 0)
    assert values["median_recovery"] == 6.0 and values["depleted_bid"] is None and values["score"] == 1.0
    assert values["abandoned_share"] == 0.0


@pytest.mark.parametrize("shortcut", [True, False])
def test_a_cancel_inside_the_k_prices_can_start_an_episode_and_one_outside_cannot(shortcut):
    res, rig, ids = resilience_rig(shortcut)
    rig.send(cancel_order(ids[("bid", 97)], 30.0))     # 3 ticks from the touch: outside the three prices
    rig.send(cancel_order(ids[("bid", 99)], 31.0))     # 300 -> 200 of a line at 150: no
    assert res.started == 0
    rig.send(cancel_order(ids[("bid", 98)], 32.0))     # 100 lots left: below the line
    assert res.started == 1 and res.values(32.0)["depleted_bid"] == 0.0
    rig.send(cancel_order(12345, 33.0))                # a cancel of an order that is not there is not an event
    assert res.started == 1


@pytest.mark.parametrize("shortcut", [True, False])
def test_a_limit_order_that_improves_the_quote_never_starts_an_episode(shortcut):
    res, rig, ids = resilience_rig(shortcut, ask_prices=(108, 109, 110, 111))     # room inside the spread
    rig.send(limit_order(103, 5, 30.0, BID))           # the touch moves up to 103 and the three prices hold 5 lots
    book = rig.book
    assert sum(book.depth_at_price(p, BID) for p in (103, 102, 101)) == 5 < 0.5 * 300  # follows the touch
    assert res.started == 0 and res.values(30.0)["depleted_bid"] is None
    rig.send(cancel_order(ids[("bid", 97)], 31.0))     # far from the touch (6 ticks): not examined
    rig.send(cancel_order(ids[("bid", 100)], 32.0))    # 3 ticks from the new touch: outside its three prices
    assert res.started == 0
    # a take on the other side while the bid window is thin does not start a bid episode either
    rig.send(market_order(50, 33.0, BID))
    assert res.values(33.0)["started"] == 0 and res.values(33.0)["depleted_bid"] is None


@pytest.mark.parametrize("shortcut", [True, False])
def test_an_episode_is_abandoned_when_the_quote_moves_more_than_k_ticks_from_its_anchor(shortcut):
    # at once: a sweep through four prices leaves the touch at 95, five ticks from the anchor at 100
    res, rig, _ = resilience_rig(shortcut, extra_bids=[(95, 100)])
    rig.send(market_order(400, 30.0, ASK))
    values = res.values(30.0)
    assert (values["started"], values["abandoned"], values["recovered"]) == (1, 1, 0)
    assert values["depleted_bid"] is None and values["abandoned_share"] == 1.0 and values["median_recovery"] is None

    # later: the touch moves from 99 to 97 (within three ticks of 100) and then to 95
    res, rig, _ = resilience_rig(shortcut, extra_bids=[(95, 100)])
    rig.send(market_order(160, 30.0, ASK))
    rig.send(market_order(140, 31.0, ASK))             # takes 99 and 98: the touch is 97
    assert res.values(31.0)["depleted_bid"] == 1.0 and res.abandoned == 0
    rig.send(market_order(100, 32.0, ASK))             # takes 97: the touch is 95
    values = res.values(32.0)
    assert (values["started"], values["abandoned"], values["recovered"]) == (1, 1, 0)
    assert values["depleted_bid"] is None

    # the side runs dry
    res, rig, _ = resilience_rig(shortcut)
    rig.send(market_order(160, 30.0, ASK))
    rig.send(market_order(240, 31.0, ASK))
    assert res.values(31.0)["abandoned"] == 1


@pytest.mark.parametrize("shortcut", [True, False])
def test_the_count_of_open_episodes_that_lets_arrivals_pass_is_the_number_of_sides_depleted(shortcut):
    res, rig, _ = resilience_rig(shortcut, bid_prices=(100, 99, 98, 97, 96), extra_bids=[(90, 100)])
    def opened():
        return sum(episode is not None for episode in res._episode)

    assert res._open == opened() == 0
    rig.send(market_order(160, 30.0, ASK))             # a bid episode
    assert res._open == opened() == 1
    rig.send(market_order(160, 31.0, BID))             # an ask episode
    assert res._open == opened() == 2
    rig.send(limit_order(100, 100, 33.0, BID))         # the bid side recovers: 100 + 100 + 100 lots at 100, 99, 98
    rig.send(limit_order(99, 60, 34.0, BID))
    assert res.recovered_n == 1 and res._open == opened() == 1
    rig.send(market_order(300, 35.0, BID))             # the 240 lots left on the ask side are taken
    assert res.abandoned == 1 and res._open == opened() == 0


@pytest.mark.parametrize("shortcut", [True, False])
def test_the_share_of_abandoned_episodes_is_of_those_that_ended_and_none_before_any_has(shortcut):
    res, rig, _ = resilience_rig(shortcut, extra_bids=[(95, 100)])
    values = res.values(26.0)
    assert values["abandoned_share"] is None and values["restored_share"] is None
    for t0 in (30.0, 50.0):                                    # two recoveries
        rig.send(market_order(160, t0, ASK))
        rig.send(limit_order(100, 100, t0 + 1, BID))
        rig.send(limit_order(99, 60, t0 + 2, BID))
    assert res.values(60.0)["abandoned_share"] == 0.0
    rig.send(market_order(400, 70.0, ASK))                     # one abandoned at once: the touch falls to 95
    values = res.values(70.0)
    assert (values["recovered"], values["abandoned"]) == (2, 1)
    assert values["abandoned_share"] == pytest.approx(1 / 3)


def test_an_episode_ends_at_the_unit_sample_when_the_baseline_has_fallen_to_the_depth():
    res, rig, _ = resilience_rig(bid_prices=(100, 99, 98))       # nothing behind 98: the touch window stays at 140
    for u in range(26, 30):
        res.sample(u, rig.book)
    rig.send(market_order(160, 30.0, ASK))
    assert res.started == 1
    samples, expected = [300.0] * 29, None
    for u in range(31, 140):
        samples.append(140.0)
        if 140 >= 0.9 * np.median(samples[-100:]):
            expected = u
            break
    assert expected is not None and 31 < expected < 100
    for u in range(31, expected):
        res.sample(u, rig.book)
        assert res.values(float(u))["recovered"] == 0
    res.sample(expected, rig.book)
    values = res.values(float(expected))
    assert values["recovered"] == 1 and values["median_recovery"] == expected - 30
    assert values["baseline_bid"] == np.median(samples[-100:])


def test_the_resilience_score_is_t_hist_over_t_hist_plus_t_now():
    res, rig, _ = resilience_rig()
    for t0, d in ((30.0, 4.0), (50.0, 6.0), (70.0, 10.0)):
        rig.send(market_order(160, t0, ASK))
        if t0 == 30.0:
            assert res.values(t0 + 1)["score"] is None    # depleted, with nothing to compare its length with
        rig.send(limit_order(100, 100, t0 + d - 1, BID))
        rig.send(limit_order(99, 60, t0 + d, BID))
        assert res.values(t0 + d)["depleted_bid"] is None
    assert res.values(80.0)["median_recovery"] == 6.0           # the median of 4, 6, 10
    rig.send(market_order(160, 90.0, ASK))
    assert res.values(90.0)["score"] == 1.0                     # T_now = 0
    assert res.values(90.5)["score"] == pytest.approx(6 / (6 + 0.5))    # a depletion shorter than a unit is scored too
    assert res.values(93.0)["score"] == pytest.approx(6 / (6 + 3))
    assert res.values(96.0)["score"] == pytest.approx(6 / (6 + 6))


def test_the_score_of_the_worse_depleted_side_is_reported():
    res, rig, _ = resilience_rig()
    for t0 in (30.0, 50.0):                                     # one recovery on the bid side, of 2 units
        rig.send(market_order(160, t0, ASK))
        rig.send(limit_order(100, 100, t0 + 1, BID))
        rig.send(limit_order(99, 60, t0 + 2, BID))
    rig.send(market_order(160, 70.0, ASK))                      # the bid side, from 70
    rig.send(market_order(160, 73.0, BID))                      # the ask side, from 73: takes 101 and 102 partly
    values = res.values(75.0)
    assert values["depleted_bid"] == 5.0 and values["depleted_ask"] == 2.0
    assert values["score"] == pytest.approx(2 / (2 + 5))        # the bid side, the older one, is the lower


@pytest.mark.parametrize("shortcut", [True, False])
def test_large_s_restoration_of_the_quote_after_a_quote_moving_take(shortcut):
    res, rig, _ = resilience_rig(shortcut)
    rig.send(market_order(100, 30.0, BID))             # empties the ask at 101: the ask moves to 102
    assert res.quote_moves == 1 and res.started == 0
    rig.send(limit_order(101, 50, 35.0, ASK))          # the old quote is back after 5 units
    rig.send(market_order(50, 40.0, BID))              # empties it again
    assert res.quote_moves == 2
    values = res.values(75.0)                          # 35 units on, past the 30 it is given
    assert values["restored_share"] == 0.5 and values["median_restoration"] == 5.0
    rig.send(limit_order(101, 10, 80.0, ASK))          # too late: not counted
    values = res.values(80.0)
    assert values["restored_share"] == 0.5 and values["quote_moves"] == 2


@pytest.mark.parametrize("shortcut", [True, False])
def test_two_takes_waiting_are_restored_one_by_one(shortcut):
    res, rig, _ = resilience_rig(shortcut)
    rig.send(market_order(100, 30.0, BID))             # the ask goes 101 -> 102, to be put back to 101
    rig.send(market_order(100, 31.0, BID))             # 102 -> 103, to be put back to 102
    rig.send(limit_order(102, 10, 33.0, ASK))          # the touch is 102: the second is restored, the first is not
    assert res.values(33.0)["restored_share"] == 1.0 and res.restored == 1
    rig.send(limit_order(101, 10, 36.0, ASK))
    values = res.values(36.0)
    assert res.restored == 2 and values["median_restoration"] == 4.0    # 2 and 6 units


def test_a_marketable_limit_order_that_rests_on_the_other_side_restores_that_side():
    res, rig, _ = resilience_rig()
    rig.send(market_order(100, 30.0, ASK))             # the bid goes 100 -> 99
    rig.send(limit_order(102, 250, 32.0, BID))         # buys 100@101 and 100@102; the other 50 rest as the best bid
    assert rig.book.best_bid == 102 and rig.book.best_ask == 103
    values = res.values(32.0)
    assert res.restored == 1 and values["median_restoration"] == 2.0
    assert res.quote_moves == 2                         # and it moved the ask


@pytest.mark.parametrize("shortcut", [True, False])
def test_a_marketable_limit_order_that_rests_on_the_other_side_ends_that_sides_episode_if_it_moved_the_quote(shortcut):
    res, rig, _ = resilience_rig(shortcut)
    rig.send(market_order(160, 30.0, ASK))             # the bid side is depleted, anchored at 100
    assert res.values(30.0)["depleted_bid"] == 0.0
    rig.send(limit_order(104, 450, 31.0, BID))         # buys the 400 lots at 101 .. 104 and rests 50 at 104
    assert rig.book.best_bid == 104 and rig.book.best_ask is None
    values = res.values(31.0)                          # the ask side's episode began and ended at once: it is empty
    assert values["depleted_bid"] is None and (values["started"], values["abandoned"], values["recovered"]) == (2, 2, 0)

    res, rig, _ = resilience_rig(shortcut)             # resting no further than K ticks from the anchor: it goes on
    rig.send(market_order(160, 30.0, ASK))
    rig.send(limit_order(103, 350, 31.0, BID))         # buys the 300 lots at 101 .. 103 and rests 50 at 103
    assert rig.book.best_bid == 103
    values = res.values(31.0)                          # the ask side's episode goes on from 104, three ticks off
    assert values["depleted_bid"] == 1.0 and values["depleted_ask"] == 0.0
    assert (values["started"], values["abandoned"]) == (2, 0)


@pytest.mark.parametrize("shortcut", [True, False])
def test_a_cancel_outside_the_anchors_prices_that_moves_the_quote_too_far_ends_the_episode(shortcut):
    res, rig, ids = resilience_rig(shortcut, bid_prices=(100, 99, 98, 97, 96))
    rig.send(market_order(300, 30.0, ASK))             # 100, 99 and 98 are gone: the touch is 97, three ticks off
    assert res.values(30.0)["depleted_bid"] == 0.0 and res.abandoned == 0
    rig.send(cancel_order(ids[("bid", 96)], 31.0))     # behind the touch: it changes nothing
    assert res.values(31.0)["depleted_bid"] == 1.0 and res.abandoned == 0
    rig.send(cancel_order(ids[("bid", 97)], 32.0))     # the touch is empty: nothing left within three ticks
    assert res.values(32.0)["depleted_bid"] is None and res.abandoned == 1

    res, rig, ids = resilience_rig(shortcut, bid_prices=(100, 99, 98, 97, 96))
    rig.send(market_order(300, 30.0, ASK))
    rig.send(cancel_order(ids[("bid", 97)], 31.0))     # the touch moves to 96, four ticks from the anchor at 100
    values = res.values(31.0)
    assert values["depleted_bid"] is None and values["abandoned"] == 1


@pytest.mark.parametrize("shortcut", [True, False])
def test_the_crossing_onset_leaves_out_a_start_from_a_touch_that_was_already_thin(shortcut):
    for onset, expected in (("below", 1), ("crossing", 0)):
        res, rig, ids = resilience_rig(shortcut, ask_prices=(108, 109, 110, 111), settings={"onset": onset})
        rig.send(limit_order(103, 5, 30.0, BID))                    # a new, thin touch: 5 lots at 103, 102, 101
        rig.send(cancel_order(rig.book.orders_at(103, BID)[0].order_id, 31.0))   # the cancel leaves 0 of 150
        assert res.started == expected, onset
    # a take that does cross the line starts an episode under both
    for onset in ("below", "crossing"):
        res, rig, ids = resilience_rig(shortcut, settings={"onset": onset})
        rig.send(market_order(160, 30.0, ASK))
        assert res.started == 1 and res.values(30.0)["depleted_bid"] == 0.0
    # and so does a cancel that crosses it: 200 -> 100 of a line at 150
    for onset in ("below", "crossing"):
        res, rig, ids = resilience_rig(shortcut, settings={"onset": onset})
        rig.send(cancel_order(ids[("bid", 99)], 30.0))
        assert res.started == 0
        rig.send(cancel_order(ids[("bid", 98)], 31.0))
        assert res.started == 1


def test_a_take_across_the_k_prices_counts_only_the_lots_it_took_inside_them():
    # 100 lots at each of 100 .. 96; the touch window is 100, 99, 98 (300 lots, a line at 150)
    res, rig, _ = resilience_rig(bid_prices=(100, 99, 98, 97, 96), settings={"onset": "crossing"})
    rig.send(market_order(150, 30.0, ASK))             # 100@100 and 50@99: 150 left in the window: not below 150
    assert res.started == 0
    rig.send(market_order(10, 31.0, ASK))              # 140 left (the touch is 99 now: its window 99, 98, 97 holds 240)
    assert res.started == 0


@pytest.mark.parametrize("shortcut", [True, False])
@pytest.mark.parametrize("tick", [1, 5])
@pytest.mark.parametrize("side", [BID, ASK])
def test_the_crossing_onset_counts_the_lots_a_take_removed_inside_the_k_prices_and_no_others(shortcut, tick, side):
    """A touch that an arrival has improved holds some lots in its three prices, against a line of 150 lots."""
    cases = (([(0, 149)], 10, 0), ([(0, 150)], 10, 1),        # 139 left and 10 taken: 149 before, one short of the line
             ([(0, 149)], 200, 0), ([(0, 150)], 200, 1),      # the take goes on behind the three prices: 51 or 50 lots
             ([(0, 50), (2, 99)], 120, 0), ([(0, 51), (2, 99)], 120, 1))    # and one that reaches the third price
    for orders, take, crossing in cases:
        for onset, expected in (("below", 1), ("crossing", crossing)):
            near, far = [tuple(100 + i * tick for i in range(n, n + 4)) for n in (1, 8)]     # ask-side prices
            if side is BID:   # the new touch is at 100 + 3 ticks, its third price at 100 + 1 tick
                books = {"bid_prices": tuple(100 - i * tick for i in range(4)), "ask_prices": far}
                price_of, taker = (lambda k: 100 + (3 - k) * tick), ASK
            else:             # the new touch is at 100 - 2 ticks, its third price at 100
                books = {"bid_prices": tuple(200 - p for p in far), "ask_prices": near}
                price_of, taker = (lambda k: 100 - (2 - k) * tick), BID
            res, rig, _ = resilience_rig(shortcut, settings={"onset": onset}, tick_size=tick, **books)
            for k, lots in orders:
                rig.send(limit_order(price_of(k), lots, 30.0, side))
            assert res.started == 0
            rig.send(market_order(take, 31.0, taker))
            assert res.started == expected, (onset, orders, take)


def test_a_deep_book_never_depletes():
    res = st.Resilience()
    book, engine = OrderBook(), PriceTimeEngine()
    for price in (100, 99, 98, 97, 96):
        engine.process(book, limit_order(price, 5000, 0.0, BID))
    for price in (101, 102, 103, 104, 105):
        engine.process(book, limit_order(price, 5000, 0.0, ASK))
    for u in range(1, 26):
        res.sample(u, book)
    rig = Rig(res.observe, book, engine)
    for k in range(40):
        rig.send(market_order(100, 30.0 + k, ASK if k % 2 else BID))
    values = res.values(70.0)
    assert values["started"] == 0 and values["score"] == 1.0 and values["median_recovery"] is None


def test_a_baseline_needs_20_samples_and_an_empty_side_is_not_sampled():
    res = st.Resilience()
    book, engine = OrderBook(), PriceTimeEngine()
    engine.process(book, limit_order(101, 100, 0.0, ASK))
    for u in range(1, 30):
        res.sample(u, book)
    values = res.values(30.0)
    assert values["baseline_bid"] is None and values["baseline_ask"] == 100.0
    rig = Rig(res.observe, book, engine)
    rig.send(market_order(10, 31.0, ASK))              # nothing to hit: no event, nothing happens to the studies
    assert res.started == 0 and res.quote_moves == 0
    res2 = st.Resilience()
    for u in range(1, 19):
        res2.sample(u, book)
    assert res2.values(19.0)["baseline_ask"] is None
    res2.sample(19, book)
    assert res2.values(19.0)["baseline_ask"] is None   # 19 samples
    res2.sample(20, book)
    assert res2.values(20.0)["baseline_ask"] == 100.0


class LiteralResilience:
    """The same rules applied to every event with no shortcut, reading depth from ``depth_profile``."""

    def __init__(self, k=3, q=0.5, r=0.9, window=30.0, crossing=False):
        self.k, self.q, self.r, self.window, self.crossing = k, q, r, window, crossing
        self.shadow = {BID: None, ASK: None}        # the depth at the k prices from the touch after the last event
        self.samples = {BID: [], ASK: []}
        self.episode = {BID: None, ASK: None}
        self.started = self.recovered = self.abandoned = 0
        self.recoveries, self.restorations = [], []
        self.moves = self.restored = self.unrestored = 0
        self.pending = {BID: [], ASK: []}

    def depth(self, book, side, anchor):
        profile = dict(book.depth_profile(side, 500))
        step = -1 if side is BID else 1
        return sum(profile.get(anchor + step * i, 0) for i in range(self.k))

    def baseline(self, side):
        window = self.samples[side][-100:]
        return float(np.median(window)) if len(window) >= 20 else None

    def follow(self, t, side, best, book):
        t0, anchor = self.episode[side]
        if self.depth(book, side, anchor) >= self.r * self.baseline(side):
            self.recoveries.append(t - t0)
            self.episode[side] = None
            self.recovered += 1
        elif best is None or abs(best - anchor) > self.k:
            self.episode[side] = None
            self.abandoned += 1

    def expire(self, t):
        for side in (BID, ASK):
            stale = [p for p in self.pending[side] if t - p[0] > self.window]
            self.unrestored += len(stale)
            self.pending[side] = [p for p in self.pending[side] if t - p[0] <= self.window]

    def sample(self, u, book):
        for side in (BID, ASK):
            best = book.best_bid if side is BID else book.best_ask
            if best is not None:
                self.samples[side].append(self.depth(book, side, best))
            if self.episode[side] is not None:
                self.follow(float(u), side, best, book)
        self.expire(float(u))

    def event(self, t, kind, side, price, fills, aggressor, before, after, book):
        self.expire(t)
        for s in (BID, ASK):                                   # restorations first, then this event's own moves
            best = after[0] if s is BID else after[2]
            if best is None:
                continue
            kept = []
            for p in self.pending[s]:
                if best >= p[1] if s is BID else best <= p[1]:
                    self.restorations.append(t - p[0])
                    self.restored += 1
                else:
                    kept.append(p)
            self.pending[s] = kept
        touched = set()
        removed = set()
        if fills:
            maker = ASK if aggressor > 0 else BID
            touched.add(maker)
            removed.add(maker)
            if kind == st.ARRIVAL_LIMIT:
                touched.add(BID if aggressor > 0 else ASK)
            was, now = (before[0], after[0]) if maker is BID else (before[2], after[2])
            if now is None or (now < was if maker is BID else now > was):
                self.moves += 1
                self.pending[maker].append((t, was))
        elif kind == st.CANCEL:
            touched.add(side)
            removed.add(side)
        elif kind == st.ARRIVAL_LIMIT:
            touched.add(side)
        for s in (BID, ASK):
            if s not in touched:
                continue
            was, now = (before[0], after[0]) if s is BID else (before[2], after[2])
            if self.episode[s] is not None:
                self.follow(t, s, now, book)
            elif s in removed and was is not None and self.baseline(s) is not None:
                near = fills or (0 <= (was - price if s is BID else price - was) < self.k)
                already_low = self.crossing and (self.shadow[s] or 0) < self.q * self.baseline(s)
                if near and not already_low and self.depth(book, s, was) < self.q * self.baseline(s):
                    self.started += 1
                    if now is None or abs(now - was) > self.k:
                        self.abandoned += 1
                    else:
                        self.episode[s] = (t, was)
        for s in (BID, ASK):
            best = after[0] if s is BID else after[2]
            step = -1 if s is BID else 1
            self.shadow[s] = None if best is None else sum(book.depth_at_price(best + step * i, s)
                                                           for i in range(self.k))


def shortcut_observer(observe):
    """Pass the same quotes object when nothing traded and the order is behind the touch (the session's rule)."""
    def call(t, kind, side, price, size, fills, aggressor, before, after, book, own=False):
        if not fills:
            touch = before[0] if side is BID else before[2]
            if touch is not None and (price < touch if side is BID else price > touch) and kind != st.ARRIVAL_MARKET:
                after = before
        observe(t, kind, side, price, size, fills, aggressor, before, after, book, own)
    return call


def zi_model(**overrides):
    params = dict(mu=6.0, lam=1.0, delta=0.03, price_range=15, initial_mid=100, lot_size=5)
    params.update(overrides)
    return SmithFarmerModel(**params)


class Fan:
    """One observer that passes every event on to several, after sampling the units it begins as StudySet does."""

    def __init__(self, observers=(), literal=None):
        self.observers = list(observers)
        self.literal = literal
        self.unit = 0

    def __call__(self, t, kind, side, price, size, fills, aggressor, before, after, book, own=False):
        if self.literal is not None:
            if t >= self.unit + 1:
                for u in range(max(self.unit + 1, int(t) - st.HISTORY + 1), int(t) + 1):
                    self.literal.sample(u, book)
                self.unit = int(t)
            self.literal.event(t, kind, side, price, fills, aggressor, before, after, book)
        for observe in self.observers:
            observe(t, kind, side, price, size, fills, aggressor, before, after, book, own)


@pytest.mark.parametrize("shortcut", [True, False])
@pytest.mark.parametrize("settings", [dict(), dict(ticks=2, depleted=0.6, recovered=0.8, restore_window=3.0),
                                      dict(onset="crossing"), dict(onset="crossing", ticks=4, depleted=0.4)])
def test_resilience_on_a_recorded_run_equals_the_unfiltered_rule(shortcut, settings):
    reset_order_id_counter()
    study_set = st.StudySet([st.Resilience(**settings)])
    literal = LiteralResilience(k=settings.get("ticks", 3), q=settings.get("depleted", 0.5),
                                r=settings.get("recovered", 0.9), window=settings.get("restore_window", 30.0),
                                crossing=settings.get("onset") == "crossing")
    observe = Fan([study_set.observe], literal)
    if shortcut:
        observe = shortcut_observer(observe)
    sim = Simulation(zi_model(), config=SimulationConfig(n_events=60_000, warmup_events=0, seed=11), observer=observe)
    sim.run(progress=False)
    res = study_set["resilience"]
    end = res._t
    values = res.values(end)
    assert values["started"] == literal.started > 10
    assert values["recovered"] == literal.recovered > 3
    assert values["abandoned"] == literal.abandoned > 3
    assert values["quote_moves"] == literal.moves > 1000
    literal.expire(end)
    assert res.restored == literal.restored and res.unrestored == literal.unrestored
    assert list(res._recoveries) == pytest.approx(literal.recoveries[-20:])         # the last 20 recovery times
    assert list(res._restorations) == pytest.approx(literal.restorations[-20:])
    assert values["median_recovery"] == pytest.approx(np.median(literal.recoveries[-20:]))
    assert values["median_restoration"] == pytest.approx(np.median(literal.restorations[-20:]))
    assert res._episode[0] == literal.episode[BID] and res._episode[1] == literal.episode[ASK]


# ---- A set of studies, and the unit samples ----

class Probe(st.CVD):
    """A study that notes the units it is sampled for and the book it was given."""
    key = "probe"

    def __init__(self):
        self.sampled = []
        super().__init__()

    def sample(self, index, book):
        self.sampled.append((index, book))
        return super().sample(index, book)


def test_a_study_set_samples_each_unit_an_event_begins_and_fills_the_gaps():
    probe = Probe()
    study_set = st.StudySet([probe])
    book = object()
    for t in (0.5, 0.9, 1.2, 1.3, 4.7):
        study_set.observe(t, st.ARRIVAL_LIMIT, BID, 90, 1, (), 1, q(), q(), book)
    assert [unit for unit, _ in probe.sampled] == [1, 2, 3, 4]       # 1 at t = 1.2; 2, 3, 4 when the clock jumped
    assert all(b is book for _, b in probe.sampled) and study_set.unit == 4
    study_set.observe(4.9, st.ARRIVAL_LIMIT, BID, 90, 1, (), 1, q(), q(), book)
    assert len(probe.sampled) == 4                                     # still unit 4


def test_a_session_that_samples_for_itself_turns_the_automatic_sampling_off():
    probe = Probe()
    study_set = st.StudySet([probe], auto_sample=False)
    study_set.observe(7.5, st.ARRIVAL_LIMIT, BID, 90, 1, (), 1, q(), q(), object())
    assert probe.sampled == [] and study_set.unit == 0
    book = object()
    rows = study_set.sample_to(9, book, rows=True)
    assert [unit for unit, _ in rows] == list(range(1, 10)) and study_set.unit == 9
    assert rows[0] == (1, {"probe": 0}) and [unit for unit, _ in probe.sampled] == list(range(1, 10))
    assert study_set.sample_to(9, book, rows=True) == []             # nothing new
    # a late start: at most ``limit`` units are sampled, the last ones
    rows = study_set.sample_to(2000, book, limit=600, rows=True)
    assert [unit for unit, _ in rows] == list(range(1401, 2001))
    probe.sampled.clear()
    study_set.sample_to(5000, book)
    assert [unit for unit, _ in probe.sampled] == list(range(4901, 5001)) and study_set.unit == 5000


def test_a_study_set_reports_the_values_of_the_studies_that_are_on():
    study_set = st.StudySet(auto_sample=False)
    study_set.enable("vpin", False)
    values = study_set.values(1.0)
    assert set(values) == {"ofi", "cvd", "delta", "otr", "imbalance", "spread", "resilience"}
    assert values["cvd"] == {"cvd": 0} and values["ofi"] == {"ofi": None}
    json.dumps(values, allow_nan=False)
    assert study_set["spread"].key == "spread" and [s.key for s in study_set] == [s.key for s in study_set.studies]
    with pytest.raises(KeyError):
        study_set["momentum"]


def test_the_rows_of_the_samples_hold_each_study_s_main_value_at_the_unit():
    ofi = st.OFI(window=2.0)
    study_set = st.StudySet([ofi, st.Spread()], auto_sample=False)
    see(ofi, 0.5, after=q(100, 20))                       # +10
    book = SpreadBook(3)
    rows = study_set.sample_to(4, book, rows=True)
    assert rows == [(1, {"ofi": None, "spread": 3}), (2, {"ofi": 10, "spread": 3}),
                    (3, {"ofi": 0, "spread": 3}), (4, {"ofi": 0, "spread": 3})]   # the window drains with the units


# ---- The Simulation's observer ----

class Script:
    """A model that sends the orders it was given, one for each step."""

    def __init__(self, orders):
        self.orders = deque(orders)

    def generate_event(self, book, t, rng):
        return self.orders.popleft() if self.orders else None


class Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, t, kind, side, price, size, fills, aggressor, before, after, book, own=False):
        self.calls.append((t, kind, side, price, size, [(f.price, f.size) for f in fills], aggressor, before, after,
                           book))


def test_the_observer_gets_the_arguments_of_each_event_and_none_for_a_cancel_of_an_order_that_is_not_there():
    first = limit_order(100, 10, 1.0, BID)
    orders = [first, limit_order(102, 8, 2.0, ASK), limit_order(101, 40, 3.0, BID, display_size=10),
              market_order(25, 4.0, ASK), cancel_order(first.order_id, 5.0), cancel_order(99999, 6.0),
              market_order(3, 7.0, BID), limit_order(103, 50, 8.0, BID)]
    spy = Spy()
    sim = Simulation(Script(orders), config=SimulationConfig(n_events=len(orders), warmup_events=2), observer=spy)
    df = sim.run(progress=False)
    none = (None, 0, None, 0)
    assert [c[:9] for c in spy.calls] == [
        (1.0, st.ARRIVAL_LIMIT, BID, 100, 10, [], 1, none, (100, 10, None, 0)),
        (2.0, st.ARRIVAL_LIMIT, ASK, 102, 8, [], -1, (100, 10, None, 0), (100, 10, 102, 8)),
        (3.0, st.ARRIVAL_LIMIT, BID, 101, 40, [], 1, (100, 10, 102, 8), (101, 10, 102, 8)),    # 40 lots, a peak of 10
        (4.0, st.ARRIVAL_MARKET, ASK, 0, 25, [(101, 10), (101, 10), (101, 5)], -1, (101, 10, 102, 8),
         (101, 5, 102, 8)),
        (5.0, st.CANCEL, BID, 100, 10, [], 0, (101, 5, 102, 8), (101, 5, 102, 8)),
        (7.0, st.ARRIVAL_MARKET, BID, 0, 3, [(102, 3)], 1, (101, 5, 102, 8), (101, 5, 102, 5)),
        (8.0, st.ARRIVAL_LIMIT, BID, 103, 50, [(102, 5)], 1, (101, 5, 102, 5), (103, 45, None, 0)),
    ]
    assert all(call[9] is sim.book for call in spy.calls)
    assert len(df) == 6      # the warm-up's two events are observed but not logged, the cancel of nothing is logged


def test_the_observer_changes_nothing_in_the_log():
    def run(observer):
        reset_order_id_counter()
        sim = Simulation(zi_model(), config=SimulationConfig(n_events=4000, warmup_events=100, seed=3),
                         observer=observer)
        return sim.run(progress=False), sim.trades

    plain, plain_trades = run(None)
    study_set = st.StudySet()
    observed, observed_trades = run(study_set.observe)
    pd.testing.assert_frame_equal(plain, observed)
    pd.testing.assert_frame_equal(plain_trades, observed_trades)


# ---- Recorded runs ----

class Run:
    """A Simulation with a spy and several studies, and what is needed to compute them again by hand."""

    def __init__(self, model, engine, n_events, seed):
        reset_order_id_counter()
        self.sets = {
            "ofi": st.StudySet([st.OFI(window=5.0)]),
            "cvd": st.StudySet([st.CVD()]),
            "delta_true": st.StudySet([st.TradeDelta(window=10.0, side="true")]),
            "delta_tick": st.StudySet([st.TradeDelta(window=10.0, side="tick")]),
            "delta_q0": st.StudySet([st.TradeDelta(window=10.0, side="quote", quote_lag=0)]),
            "delta_q1": st.StudySet([st.TradeDelta(window=10.0, side="quote", quote_lag=1)]),
            "delta_q3": st.StudySet([st.TradeDelta(window=10.0, side="quote", quote_lag=3)]),
            "vpin_true": st.StudySet([st.VPIN(buckets=20, volume=100, side="true")]),
            "vpin_tick": st.StudySet([st.VPIN(buckets=20, volume=100, side="tick")]),
            "vpin_mid0": st.StudySet([st.VPIN(buckets=20, volume=100, side="mid", quote_lag=0)]),
            "vpin_bvc": st.StudySet([st.VPIN(buckets=20, volume=100, side="bvc")]),
            "vpin_auto": st.StudySet([st.VPIN(buckets=20, side="true")]),
            "otr": st.StudySet([st.OrderToTrade(window=10.0)]),
            "all": st.StudySet(),
        }
        self.calls = []
        self.checked = 0
        sim = Simulation(model, engine=engine, config=SimulationConfig(n_events=n_events, warmup_events=0, seed=seed),
                         observer=self.observe)
        self.df = sim.run(progress=False)
        self.trades = sim.trades
        self.book = sim.book
        self.end = self.calls[-1][0]

    def observe(self, t, kind, side, price, size, fills, aggressor, before, after, book, own=False):
        self.calls.append((t, kind, side, price, size, [(f.price, f.size) for f in fills], aggressor, before, after))
        for study_set in self.sets.values():
            study_set.observe(t, kind, side, price, size, fills, aggressor, before, after, book, own)
        if len(self.calls) % 499 == 0:    # the values are JSON at every stage of the run, the empty book included
            json.dumps(self.sets["all"].values(t, book), allow_nan=False)
            self.checked += 1

    def value(self, key, study):
        return self.sets[key][study].values(self.end, self.book)


@lru_cache(maxsize=None)
def recorded(kind):
    if kind == "zi":
        return Run(zi_model(), PriceTimeEngine(), 40_000, 7)
    return Run(ParlourModel(arrival_rate=13.0, delta=0.05, initial_mid=100, price_range=10, lot_size=5),
               ProRataEngine(), 25_000, 5)


def fills_by_event(run):
    return [(c[0], c[6], c[5]) for c in run.calls if c[5]]


def window_sum(items, end, window):
    return sum(x for t, x in items if end - window < t <= end)


@pytest.mark.parametrize("kind", ["zi", "parlour"])
def test_the_recorded_stream_is_the_simulation_s_own_log(kind):
    run = recorded(kind)
    df, calls = run.df, run.calls
    assert run.checked > 10
    assert len(calls) == len(df)                        # no cancel of a missing order in these models
    assert (df["n_fills"] > 1).any() == (kind == "parlour")
    codes = {"limit": st.ARRIVAL_LIMIT, "market": st.ARRIVAL_MARKET, "cancel": st.CANCEL}
    resting = {}
    previous = (None, 0, None, 0)
    for call, rec in zip(df.itertuples(), calls, strict=True):
        t, k, side, price, size, fills, aggressor, before, after = rec
        assert t == call.time and k == codes[call.event_type] and before == previous
        assert sum(n for _, n in fills) == call.fill_volume and len(fills) == call.n_fills
        quotes = tuple(None if isinstance(x, float) and math.isnan(x) else int(x)
                       for x in (call.best_bid, call.bid_size, call.best_ask, call.ask_size))
        assert (after[0], after[2]) == (quotes[0], quotes[2])
        assert (after[1] if after[0] is not None else 0) == quotes[1]
        assert (after[3] if after[2] is not None else 0) == quotes[3]
        if k == st.CANCEL:
            arrived_side, arrived_price, arrived_size = resting.pop(call.order_id)   # the canceled order's own record
            assert (side.value, price) == (arrived_side, arrived_price) and 0 < size <= arrived_size
            assert aggressor == 0
        else:
            assert (side.value, size) == (call.side, call.size) and aggressor == (1 if side is BID else -1)
            if k == st.ARRIVAL_LIMIT:
                resting[call.order_id] = (side.value, price, size)
                assert price == call.price
        previous = after


@pytest.mark.parametrize("kind", ["zi", "parlour"])
def test_flow_studies_on_a_recorded_run_equal_their_references(kind):
    run = recorded(kind)
    calls, end = run.calls, run.end
    times = np.array([c[0] for c in calls])

    # OFI: the metrics function over the logged quotes; the first event has the empty book before it
    e = numpy_increments([calls[0][7]] + [c[8] for c in calls])
    log_e = order_flow_imbalance(run.df["best_bid"], run.df["bid_size"], run.df["best_ask"], run.df["ask_size"])
    np.testing.assert_array_equal(e[1:], log_e)
    assert run.value("ofi", "ofi")["ofi"] == e[times > end - 5.0].sum()

    # CVD from the trade log
    signed = np.where(run.trades["aggressor"] == "bid", 1, -1) * run.trades["size"]
    assert run.value("cvd", "cvd")["cvd"] == signed.sum()

    events = fills_by_event(run)
    flat = [(t, a, p, n) for t, a, fills in events for p, n in fills]
    assert len(flat) == len(run.trades)

    # delta with true sides
    signed_by_event = [(t, a * sum(n for _, n in f)) for t, a, f in events]
    assert run.value("delta_true", "delta")["delta"] == window_sum(signed_by_event, end, 10.0)

    # delta with the tick rule
    signs = tick_signs([p for _, _, p, _ in flat])
    known = [(t, s, a, n) for (t, a, _, n), s in zip(flat, signs, strict=True) if s is not None]
    values = run.value("delta_tick", "delta")
    assert values["delta"] == window_sum([(t, s * n) for t, s, _, n in known], end, 10.0)
    assert values["coverage"] == pytest.approx(sum(n for *_, n in known) / sum(n for *_, n in flat))
    assert values["accuracy"] == pytest.approx(sum(n for _, s, a, n in known if s == a) / sum(n for *_, n in known))

    # delta with the quote rule: the exact mid just before the event, and the mids of the first event of each unit
    for key, lag in (("delta_q0", 0), ("delta_q1", 1), ("delta_q3", 3)):
        mids, nxt = {}, 1
        for c in calls:
            if c[0] >= nxt:
                after = c[8]
                mid = None if after[0] is None or after[2] is None else (after[0] + after[2]) / 2
                for u in range(max(nxt, int(c[0]) - st.HISTORY + 1), int(c[0]) + 1):
                    mids[u] = mid
                nxt = int(c[0]) + 1
        tick, prev, last = [], None, None
        signed_lots, classified, correct, total = [], 0, 0, 0
        for _, _, p, _ in flat:
            tick_sign = None if prev is None else (1 if p > prev else -1 if p < prev else last)
            prev, last = p, tick_sign
            tick.append(tick_sign)
        index = 0
        for t, a, fills in events:
            if lag == 0:
                before = next(c[7] for c in calls if c[0] == t and c[5])      # the first such event at this time
                ref = None if before[0] is None or before[2] is None else (before[0] + before[2]) / 2
            else:
                ref = mids.get(int(t) - lag + 1)
            for p, n in fills:
                total += n
                if ref is None:
                    sign = None
                else:
                    sign = 1 if p > ref else -1 if p < ref else tick[index]
                if sign is not None:
                    classified += n
                    signed_lots.append((t, sign * n))
                    correct += n if sign == a else 0
                index += 1
        values = run.value(key, "delta")
        assert values["delta"] == window_sum(signed_lots, end, 10.0)
        assert values["coverage"] == pytest.approx(classified / total)
        assert values["accuracy"] == pytest.approx(correct / classified)
        if lag == 0:
            assert values["coverage"] == 1.0 and values["accuracy"] == 1.0   # exact by construction
            assert values["delta"] == run.value("delta_true", "delta")["delta"]
        else:
            assert values["accuracy"] < 1.0

    # OTR and TTO
    values = run.value("otr", "otr")
    window = [c for c in calls if end - 10.0 < c[0] <= end]
    trades = sum(1 for c in window if c[5])
    assert values["messages"] == len(window) and values["trades"] == trades
    assert values["otr"] == len(window) / trades
    assert values["tto"] == sum(n for c in window for _, n in c[5]) / sum(c[4] for c in window if c[1] != st.CANCEL)
    assert len(events) == len(trades_by_order(run.trades))    # an order that filled is one trade, however many fills


@pytest.mark.parametrize("kind", ["zi", "parlour"])
def test_vpin_on_a_recorded_run_equals_vpin_made_by_listing_every_lot(kind):
    run = recorded(kind)
    events = [(fills, a) for _, a, fills in fills_by_event(run)]
    for key in ("vpin_true", "vpin_mid0"):      # the mid just before the event is on the true side of every fill
        expected, closed = vpin_by_lots(events, 100, 20, lambda i, p, a: a)
        values = run.value(key, "vpin")
        assert values["vpin"] == pytest.approx(expected, abs=1e-12) and values["buckets"] == closed > 40
    # tick rule
    signs = tick_signs([p for fills, _ in events for p, _ in fills])
    lots, k = [], 0
    for i, (fills, _) in enumerate(events):
        for _, n in fills:
            lots += [(signs[k], i)] * n
            k += 1
    imbalances = [abs([s for s, _ in lots[b * 100:(b + 1) * 100]].count(1)
                      - [s for s, _ in lots[b * 100:(b + 1) * 100]].count(-1)) / 100 for b in range(len(lots) // 100)]
    assert run.value("vpin_tick", "vpin")["vpin"] == pytest.approx(np.mean(imbalances[-20:]), abs=1e-12)

    # automatic bucket: 50 times the median of the first 200 fills; those 200 fill no bucket
    sizes = [n for fills, _ in events for _, n in fills]
    volume = 50 * float(np.median(sizes[:200]))
    values = run.value("vpin_auto", "vpin")
    assert values["volume"] == volume
    rest = []
    seen = 0
    for fills, a in events:
        kept = []
        for p, n in fills:
            seen += 1
            if seen > 200:
                kept.append((p, n))
        if kept:
            rest.append((kept, a))
    expected, closed = vpin_by_lots(rest, int(volume), 20, lambda i, p, a: a)
    assert values["buckets"] == closed and values["vpin"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("kind", ["zi", "parlour"])
def test_bulk_volume_classification_on_a_recorded_run_sits_near_one_half_beside_the_true_side(kind):
    run = recorded(kind)
    bvc = run.value("vpin_bvc", "vpin")["vpin"]
    true_side = run.value("vpin_true", "vpin")["vpin"]
    assert 0.3 < bvc < 0.7
    assert abs(bvc - 0.5) < abs(true_side - 0.5)
    # and numpy agrees on the whole series of buckets
    events = [(fills, a) for _, a, fills in fills_by_event(run)]
    hit = [i for i, c in enumerate(run.calls) if c[5]]
    mids_before = [_mid_of(run.calls[i][7]) for i in hit]
    mids_after, carried = [], None
    for i in hit:                                     # a side emptied by a take leaves no mid: the last one stands
        carried = _mid_of(run.calls[i][8]) if _mid_of(run.calls[i][8]) is not None else carried
        mids_after.append(carried)
    mids_before = [after if before is None else before for before, after in zip(mids_before, mids_after, strict=True)]
    expected, imbalances, shares = bvc_by_numpy(events, 100, 20, mids_before, mids_after)
    assert bvc == pytest.approx(expected, abs=1e-9)


def _mid_of(quotes):
    return None if quotes[0] is None or quotes[2] is None else (quotes[0] + quotes[2]) / 2


def test_the_default_set_of_studies_gives_json_on_a_recorded_run_from_the_empty_book():
    run = recorded("zi")
    values = run.sets["all"].values(run.end, run.book)
    json.dumps(values, allow_nan=False)
    assert values["imbalance"]["imbalance"] is not None and values["spread"]["spread"] is not None
    assert values["resilience"]["started"] > 20 and values["resilience"]["quote_moves"] > 1000
    assert values["vpin"]["vpin"] is not None and values["delta"]["delta"] is not None
    assert run.checked >= 70


def test_bulk_volume_classification_of_fair_coin_flow_in_zi_is_nowhere_near_the_true_side():
    run = recorded("zi")
    assert run.value("vpin_true", "vpin")["vpin"] < 0.25      # fair-coin sides: low
    assert 0.4 < run.value("vpin_bvc", "vpin")["vpin"] < 0.6  # whatever the flow: near one half


# ---- Edge cases ----

def test_resilience_counts_prices_in_ticks_when_the_tick_is_not_one():
    book, engine = OrderBook(tick_size=5), PriceTimeEngine()
    for price in (100, 95, 90, 85, 60):
        engine.process(book, limit_order(price, 100, 0.0, BID))
    for price in (105, 110, 115, 120):
        engine.process(book, limit_order(price, 100, 0.0, ASK))
    res = st.Resilience()
    for u in range(1, 26):
        res.sample(u, book)
    assert res.values(26.0)["baseline_bid"] == 300.0               # the prices 100, 95 and 90
    rig = Rig(res.observe, book, engine)
    rig.send(market_order(160, 30.0, ASK))                         # 100 gone, 60 of 95: 0 + 40 + 100 at 100, 95, 90
    assert res.started == 1 and res.values(30.0)["depleted_bid"] == 0.0
    rig.send(limit_order(100, 100, 33.0, BID))
    rig.send(limit_order(95, 60, 36.0, BID))
    assert res.recovered_n == 1 and res.values(36.0)["median_recovery"] == 6.0

    book2, engine2 = OrderBook(tick_size=5), PriceTimeEngine()
    for price in (100, 95, 90, 85, 60):
        engine2.process(book2, limit_order(price, 100, 0.0, BID))
    engine2.process(book2, limit_order(105, 100, 0.0, ASK))
    res2 = st.Resilience()
    for u in range(1, 26):
        res2.sample(u, book2)
    Rig(res2.observe, book2, engine2).send(market_order(400, 30.0, ASK))   # the touch falls to 60: 8 ticks, K is 3
    assert res2.started == 1 and res2.abandoned == 1


def test_a_vpin_bucket_smaller_than_a_lot_is_refused_and_recalibrate_starts_the_bucket_again():
    with pytest.raises(ValueError, match="at least 1"):
        st.VPIN(volume=0.5)
    vpin = st.VPIN(buckets=1)
    for i in range(200):
        see(vpin, float(i), fills=fills_of((100, 4)), aggressor=1)
    assert vpin.volume_in_use == 200.0
    see(vpin, 201.0, fills=fills_of((100, 200)), aggressor=1)
    assert vpin.values()["buckets"] == 1
    vpin.recalibrate()
    assert vpin.volume_in_use is None and vpin.values()["buckets"] == 0
    assert vpin.log is None and st.VPIN(log_buckets=True).log == []


def test_vpin_mid_rule_does_not_know_a_fill_when_the_book_was_one_sided():
    vpin = st.VPIN(buckets=1, volume=10, side="mid", quote_lag=0)
    see(vpin, 1.0, before=q(100, 5, None, 0), fills=fills_of((101, 6)), aggressor=1)
    see(vpin, 2.0, before=q(100, 5, 102, 5), fills=fills_of((102, 4)), aggressor=1)
    assert vpin.value() == pytest.approx(4 / 10)         # the first 6 lots filled the bucket and entered neither side


def test_a_bvc_bucket_with_no_price_change_is_balanced_and_the_values_stay_finite():
    vpin = st.VPIN(buckets=2, volume=10, side="bvc")
    for i in range(6):
        see(vpin, float(i), before=q(100, 5, 102, 5), after=q(100, 5, 102, 5), fills=fills_of((100, 10)), aggressor=1)
    assert vpin.value() == 0.0
    json.dumps(vpin.values(), allow_nan=False)


def test_the_study_set_passes_the_own_flag_on_and_restarts_a_study_it_configures_from_now():
    study_set = st.StudySet([st.OrderToTrade(window=5.0), st.OFI(window=5.0)], auto_sample=False)
    study_set.observe(1.0, st.ARRIVAL_LIMIT, BID, 99, 5, (), 1, q(), q(), None, True)    # a strategy's order
    study_set.observe(2.0, st.ARRIVAL_LIMIT, BID, 99, 5, (), 1, q(), q(), None, False)
    assert study_set.values(5.0)["otr"]["messages"] == 1
    study_set.observe(30.0, st.ARRIVAL_LIMIT, BID, 99, 5, (), 1, q(), q(), None)
    study_set.sample_to(30, None, limit=1)      # a study set whose sampling is switched on would do this itself
    assert study_set.configure("otr", scope="everyone") is True
    assert study_set["otr"].start == 30.0       # from now, not from the last event the study happened to see


def test_a_study_set_with_no_book_yet_has_no_book_state_values():
    study_set = st.StudySet(auto_sample=False)
    values = study_set.values(1.0)
    assert values["imbalance"] == {"imbalance": None, "scale": None}
    assert values["spread"] == {"spread": None, "z": None, "shock": None}
    json.dumps(values, allow_nan=False)


def test_book_state_values_without_a_book_are_the_last_sample_s():
    book = make_book(BIDS, ASKS)
    imbalance = st.BookImbalance(levels=3, weighting="touch")
    spread = st.Spread()
    for u in range(1, 30):
        wide = make_book([(100 - u % 3, 5)], [(104 + u % 4, 5)])
        imbalance.sample(u, wide)
        spread.sample(u, wide)
    expected = imbalance.compute(book), imbalance.scale(book), spread.values(book=book)["z"]   # before the sample joins
    imbalance.sample(30, book)
    spread.sample(30, book)
    assert imbalance.values() == {"imbalance": pytest.approx(expected[0]), "scale": expected[1]}
    values = spread.values()
    assert values["spread"] == 2 and values["z"] == pytest.approx(expected[2]) and values["z"] is not None


# ---- Defaults and ranges ----

SETTINGS_TABLE = {
    st.OFI: {"window": (5.0, 0.5, 100.0)},
    st.TradeDelta: {"window": (10.0, 0.5, 100.0), "side": ("true", ("true", "tick", "quote")),
                    "quote_lag": (1, 0, 100)},
    st.VPIN: {"buckets": (50, 1, 1000), "volume": (0, 0, 1_000_000), "side": ("true", ("true", "tick", "mid", "bvc")),
              "quote_lag": (1, 0, 100)},
    st.OrderToTrade: {"window": (10.0, 1.0, 100.0), "scope": ("model", ("model", "everyone"))},
    st.BookImbalance: {"levels": (10, 1, 20), "weighting": ("equal", ("equal", "touch"))},
    st.Spread: {"shock": (2.0, 0.5, 10.0)},
    st.Resilience: {"ticks": (3, 1, 20), "depleted": (0.5, 0.05, 0.95), "recovered": (0.9, 0.1, 1.0),
                    "restore_window": (30.0, 1.0, 600.0), "onset": ("below", ("below", "crossing"))},
}


@pytest.mark.parametrize("cls", list(SETTINGS_TABLE))
def test_each_study_has_the_documented_settings_defaults_and_ranges(cls):
    table = SETTINGS_TABLE[cls]
    study = cls()
    assert study.settings == {name: spec[0] for name, spec in table.items()}
    assert [spec.name for spec in cls.SETTINGS] == list(table)
    for spec in cls.SETTINGS:
        expected = table[spec.name]
        if spec.options:
            assert spec.options == expected[1]
            continue
        _, low, high = expected
        assert (spec.low, spec.high) == (low, high)
        assert spec.parse(low) == low and spec.parse(high) == high
        nudge = 1 if spec.integer else 1e-6
        for outside in (low - nudge, high + nudge):
            with pytest.raises(ValueError, match="must be between"):
                spec.parse(outside)


def test_keys_labels_and_which_studies_need_fills():
    studies = st.default_studies()
    assert [s.key for s in studies] == ["ofi", "cvd", "delta", "vpin", "otr", "imbalance", "spread", "resilience"]
    assert [s.key for s in studies if s.needs_fills] == ["cvd", "delta", "vpin"]
    assert all(s.label and s.headline for s in studies)
    for s in studies:
        assert s.headline in s.values(1.0)


def test_a_fill_in_a_unit_not_yet_sampled_reads_the_newest_sample():
    delta = st.TradeDelta(window=100.0, side="quote", quote_lag=1)
    for u in range(1, 4):
        delta.sample(u, Stub(100.0 + u))                  # 101, 102, 103
    trade(delta, 5.3, [(103, 10)], 1)                     # units 4, 5 not sampled: the mid is 103, the price at it
    trade(delta, 5.4, [(104, 10)], 1)                     # above 103: a buy
    assert delta.value(100.0) == 10                       # not 102 (a buy for the first) or 104
    assert delta.values(100.0)["coverage"] == pytest.approx(0.5)


def test_the_window_studies_are_full_a_window_after_their_own_start():
    delta = st.TradeDelta(window=4.0)
    otr = st.OrderToTrade(window=4.0)
    for study in (delta, otr):
        study.reset(100, 100.5)
        see(study, 101.0, kind=st.ARRIVAL_LIMIT, size=2, fills=fills_of((101, 2)), aggressor=1)
    assert delta.value(104.4) is None and delta.value(104.5) == 2
    assert otr.values(104.4)["otr"] is None and otr.values(104.5)["otr"] == 1.0


def test_vpin_buckets_are_at_least_one_lot():
    assert st.VPIN(volume=1).volume_in_use == 1.0
    vpin = st.VPIN()
    vpin.set_volume(1)
    assert vpin.volume_in_use == 1.0
    see(vpin, 1.0, fills=fills_of((100, 3)), aggressor=1)
    assert vpin.values()["buckets"] == 3


def test_a_bvc_bucket_closed_when_a_side_is_empty_takes_the_last_mid_there_was():
    vpin = st.VPIN(buckets=2, volume=10, side="bvc", log_buckets=True)

    def fill(t, lots, before, after):
        vpin.observe(t, st.ARRIVAL_MARKET, BID, 0, lots, fills_of((100, lots)), 1, before, after, None)

    fill(1.0, 10, quotes_at(100.0), quotes_at(101.0))        # bucket 1: the mid went 100 -> 101
    fill(2.0, 5, quotes_at(101.0), quotes_at(103.0))         # half of bucket 2: the mid is 103
    fill(3.0, 5, quotes_at(103.0), (110, 5, None, 0))        # bucket 2 closes, the ask gone: 103 stands as its end
    changes = np.array([1.0, 2.0])                           # 101 - 100, 103 - 101
    x = 2.0 / np.std(changes, ddof=1)
    assert vpin.log[1][1] == pytest.approx(abs(2 * stats.norm.cdf(x) - 1))
    fill(4.0, 10, (110, 5, None, 0), quotes_at(104.0))       # no mid before: the start is the last one, 103
    changes = np.array([1.0, 2.0, 1.0])                      # 104 - 103
    x = 1.0 / np.std(changes, ddof=1)
    assert vpin.log[2][1] == pytest.approx(abs(2 * stats.norm.cdf(x) - 1))


@pytest.mark.parametrize("restart", ["recalibrate", "set_volume", "reset"])
def test_a_new_vpin_bucket_keeps_the_mid_samples_of_the_mid_rule_and_only_a_reset_clears_them(restart):
    vpin = st.VPIN(buckets=1, volume=20, side="mid", quote_lag=1)
    vpin.sample(1, Stub(100.0))
    if restart == "recalibrate":
        vpin.recalibrate()
    elif restart == "set_volume":
        vpin.set_volume(20)
    else:
        vpin.reset(1, 1.0)
    see(vpin, 1.5, fills=fills_of((101, 20)), aggressor=-1)       # above the mid of the last whole unit: a buy
    assert vpin.value() == (0.0 if restart == "reset" else 1.0)  # a reset leaves no mid, so the fill has no side


def test_the_share_of_a_logged_bucket_is_its_buy_lots_among_those_with_a_side_and_none_when_there_are_none():
    true = st.VPIN(buckets=1, volume=10, side="true", log_buckets=True)
    see(true, 1.0, fills=fills_of((100, 6)), aggressor=1)
    see(true, 2.0, fills=fills_of((100, 4)), aggressor=-1)
    assert true.log == [(2.0, pytest.approx(0.2), pytest.approx(0.6), pytest.approx(0.6))]
    tick = st.VPIN(buckets=1, volume=10, side="tick", log_buckets=True)
    see(tick, 1.0, fills=fills_of((100, 10)), aggressor=1)          # the first fill has no tick to compare with
    assert tick.log == [(1.0, 0.0, None, 1.0)]


@pytest.mark.parametrize("one_sided", [False, True])
def test_the_first_bvc_bucket_starts_at_the_last_mid_there_was_when_the_calibration_finished(one_sided):
    vpin = st.VPIN(buckets=2, side="bvc")
    for i in range(200):
        after = (100, 5, None, 0) if one_sided and i == 199 else quotes_at(100.0)    # the ask gone: no mid after it
        see(vpin, float(i), before=quotes_at(100.0), after=after, fills=fills_of((101, 1)), aggressor=1)
    assert vpin.volume_in_use == 50.0
    # bucket 1 closes at a mid of 101 and opened at the 100 left by the last calibrating fill, not at the 100.5
    # the mid had moved to by the time of its first fill
    for i in range(50):
        mid = 100.5 if i < 49 else 101.0
        see(vpin, 201.0 + i, before=quotes_at(100.5 if i == 0 else mid), after=quotes_at(mid),
            fills=fills_of((101, 1)), aggressor=1)
    for i in range(50):                                             # bucket 2 ends back at 100
        mid = 101.0 if i < 49 else 100.0
        see(vpin, 251.0 + i, before=quotes_at(101.0), after=quotes_at(mid), fills=fills_of((101, 1)), aggressor=1)
    changes = np.array([1.0, -1.0])                                 # 101 - 100, 100 - 101
    x = changes[1] / np.std(changes, ddof=1)
    assert vpin.value() == pytest.approx((0.0 + abs(2 * stats.norm.cdf(x) - 1)) / 2)    # the first bucket has no sigma


def bvc_shares(changes):
    """The buy share Phi(c / sigma) of each bucket, sigma the deviation of the changes so far (0 for the first)."""
    return [0.5 if k == 0 else stats.norm.cdf(c / np.std(changes[:k + 1], ddof=1)) for k, c in enumerate(changes)]


@pytest.mark.parametrize("events, changes", [
    # no mid before the bucket's first fill: it opens at the mid after that fill, 101 (not the 102 of its next
    # fill's before), and closes at 103
    ([(5, (100, 5, None, 0), quotes_at(101.0)), (5, quotes_at(102.0), quotes_at(103.0)),
      (10, quotes_at(103.0), quotes_at(101.0))], [2.0, -2.0]),
    # no mid anywhere before the second fill gives the bucket: it opens and closes at 101
    ([(5, (100, 5, None, 0), (100, 5, None, 0)), (5, (100, 5, None, 0), quotes_at(101.0)),
      (10, quotes_at(101.0), quotes_at(100.0))], [0.0, -1.0]),
    # a bucket that closes with no mid (the mid before was 100, none ever after a fill): no change, and the next
    # bucket opens where this one did, at 100, not where its own first fill found the mid, 101
    ([(10, quotes_at(100.0), (100, 5, None, 0)), (10, quotes_at(101.0), quotes_at(102.0)),
      (10, quotes_at(102.0), quotes_at(101.0))], [0.0, 2.0, -1.0]),
])
def test_a_bvc_bucket_opens_and_closes_at_the_last_mid_there_was_when_a_side_is_empty(events, changes):
    vpin = st.VPIN(buckets=2, volume=10, side="bvc", log_buckets=True)
    for i, (lots, before, after) in enumerate(events, 1):
        see(vpin, float(i), kind=st.ARRIVAL_MARKET, price=0, before=before, after=after, fills=fills_of((100, lots)),
            aggressor=1)
    shares = bvc_shares(changes)
    assert [row[2] for row in vpin.log] == pytest.approx(shares)
    assert [row[1] for row in vpin.log] == pytest.approx([abs(2 * share - 1) for share in shares])


def test_a_mid_rule_bucket_logs_its_buy_share_with_a_fill_at_the_mid_a_buy():
    vpin = st.VPIN(buckets=1, volume=10, side="mid", quote_lag=0, log_buckets=True)
    see(vpin, 1.0, before=quotes_at(100.0), fills=fills_of((100, 4), (101, 3), (99, 3)), aggressor=-1)
    assert vpin.log == [(1.0, pytest.approx(0.4), pytest.approx(0.7), 0.0)]     # 7 lots bought, 3 sold; the taker sold


def test_samples_keep_the_last_n_numbers_with_their_median_mean_and_deviation():
    samples = st._Samples(4)
    assert len(samples) == 0 and samples.median() is None and samples.mean_sd() == (None, None)
    samples.add(3)
    assert samples.mean_sd() == (3.0, 0.0) and samples.mean_sd(ddof=1) == (3.0, None)
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    for i in range(1, len(numbers)):
        samples.add(numbers[i])
        last = numbers[max(0, i - 3):i + 1]
        assert len(samples) == len(last) and samples.median() == np.median(last)
        assert samples.mean_sd() == (pytest.approx(np.mean(last)), pytest.approx(np.std(last)))
        assert samples.mean_sd(ddof=1) == (pytest.approx(np.mean(last)), pytest.approx(np.std(last, ddof=1)))


def test_values_without_a_time_are_as_of_the_last_event_seen():
    for study in (st.OFI(window=2.0), st.OrderToTrade(window=2.0), st.TradeDelta(window=2.0)):
        for i in range(1, 12):
            see(study, i * 0.5, before=q(100, 10, 102, 10), after=q(100, 10 + i, 102, 10),
                fills=fills_of((102, 3)), aggressor=1)
        assert study.values() == study.values(5.5) != study.values(100.0)
    res, rig, _ = resilience_rig()
    rig.send(market_order(160, 30.0, ASK))             # an episode and a quote-moving take waiting
    assert res.values() == res.values(30.0) and res.values()["depleted_bid"] == 0.0
    study_set = st.StudySet([st.OFI(window=2.0), st.TradeDelta(window=2.0)], auto_sample=False)
    for i in range(1, 12):
        study_set.observe(i * 0.5, st.ARRIVAL_LIMIT, BID, 100, 1, fills_of((102, 3)), 1, q(100, 10), q(100, 10 + i),
                          None)
    assert study_set.values() == study_set.values(5.5) and study_set.values()["delta"]["delta"] == 12
    study_set.observe(9.0, st.ARRIVAL_LIMIT, BID, 99, 1, (), 1, q(), q(), None)   # no fill: the delta has not seen it
    assert study_set.values() == study_set.values(9.0) and study_set.values()["delta"]["delta"] == 0


def test_a_study_set_reset_without_a_time_starts_at_the_unit_and_a_study_switched_on_starts_from_its_clock():
    study_set = st.StudySet()
    study_set.enable("ofi", False)
    study_set.enable("ofi", True)
    assert study_set["ofi"].start == 0.0
    study_set.reset(7)
    assert study_set.unit == 7 and study_set["ofi"].start == 7.0
    study_set.enable("ofi", False)
    study_set.enable("ofi", True)
    assert study_set["ofi"].start == 7.0                # from the clock the reset left, which is the unit's start


def locked_book():
    """A book whose best bid and best ask are the same price, which only a direct add can make."""
    book = OrderBook()
    for side, price, lots in ((BID, 100, 10), (BID, 99, 5), (ASK, 100, 8), (ASK, 101, 4)):
        book.add_order(limit_order(price, lots, 0.0, side))
    return book


def test_touch_weighting_has_no_scale_without_a_spread_and_a_locked_book_has_a_scale_of_one_tick():
    one_sided = make_book([(100, 10), (99, 5)], [])
    touch = st.BookImbalance(levels=3, weighting="touch")
    assert touch.sample(1, one_sided) is None
    assert touch.values(book=one_sided) == {"imbalance": None, "scale": None}
    assert st.BookImbalance(levels=3).values(book=make_book(BIDS, ASKS))["scale"] is None    # equal weights: none
    locked = locked_book()
    assert locked.spread == 0
    assert touch.scale(locked) == 1.0 and touch.compute(locked) == pytest.approx(touch_weighted(locked, 3, 1.0))
    for u in range(1, 21):                              # and with the 20 samples of a median spread of 0
        touch.sample(u, locked)
    assert touch.scale(locked) == 1.0 and touch.compute(locked) == pytest.approx(touch_weighted(locked, 3, 1.0))


def test_a_window_nobody_reads_holds_no_more_than_one_update_may_hold():
    ofi = st.OFI(window=5.0)
    otr = st.OrderToTrade(window=5.0)
    for i in range(30_000):
        t = i * 0.001
        ofi.observe(t, st.ARRIVAL_LIMIT, BID, 100, 1, (), 0, q(100, 10), q(100, 11), None)
        otr.observe(t, st.ARRIVAL_LIMIT, BID, 100, 1, (), 0, q(), q(), None)
        assert len(ofi._window.items) <= 20_001 and len(otr._items) <= 20_001
    assert len(ofi._window.items) > 5_000 and len(otr._items) > 5_000      # the window of 5 units holds 5,000 events
    assert ofi.values()["ofi"] == len(ofi._window.items) and otr.values()["messages"] == len(otr._items)
