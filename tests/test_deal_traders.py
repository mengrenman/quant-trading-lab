"""Trading stock deals, and keeping pairs out of their way (lob/deal_traders.py; the deals are those of lob/deals.py)"""

import dataclasses
import functools
import math
from functools import partial
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from lob.deal_traders import DealRecord, DealTrader, NewsAwareTrader
from lob.deals import Deal, Deals
from lob.statarb import (WAIT, BookMarket, FactorValues, RelationTrader, StateTrader, pairs_by_distance,
                         sector_loadings)

FOUR = sector_loadings(2, 2)  # four books, two sectors: the books of a sector are the pairs that are truly related
SIX = sector_loadings(3, 2)  # six books, three sectors
SEEDS = range(1, 9)
BUSY = dict(rate=0.15, premium=60.0, length=30.0, breaks=0.3, priced=0.4, cool=10.0)  # a deal about every 7 units


def deal(target, acquirer, ratio=1.0, announced=0.0, ends=100.0, offered=60.0):
    return Deal(target, acquirer, ratio, announced, ends, offered)


class Stage:
    """A market made by hand for a trader: the deals on and those that ended, the clock, and every order filled in full
    at the mids it was asked at (``fill`` says how much of an order is filled, in lots, with its sign, and the orders
    in the books of ``skip`` are not filled at all)."""

    def __init__(self, trader, books=4, deals=True):
        self.trader = trader
        self.fair = SimpleNamespace(deals=[], ended=[], deal_settings=Deals() if deals else None)
        self.time = 0.0
        trader.check(self)
        trader.attach(books)

    def start(self, *deals):
        self.fair.deals.extend(deals)

    def end(self, deal, outcome="completed", at=None):
        deal.outcome, deal.ended_at = outcome, self.time if at is None else at
        self.fair.deals.remove(deal)
        self.fair.ended.append(deal)

    def decide(self, mids, dt=1.0, fill=None, skip=()):
        """The decision ``dt`` later at these mids: the trader observes and decides, and its moves fill at the mids."""
        self.time += dt
        mids = np.asarray(mids, dtype=float)
        self.trader.observe(self)
        moves = self.trader.decide(mids)
        for move in moves:
            for book in np.nonzero(move)[0]:
                lots = int(move[book])
                lots = 0 if book in skip else lots if fill is None else fill(lots)
                self.trader.position[book] += lots
                self.trader.book_cash[book] -= lots * mids[book]
                self.trader.cash -= lots * mids[book]
        return moves


def half(lots):
    """An order that fills half of what it asks (at least a lot), with its sign."""
    return int(np.sign(lots)) * max(1, abs(lots) // 2)


def in_a_market(trader=None, table=FOUR, seed=1, deals=None, **fair):
    """A market with deals (``BUSY`` over what is given) and the trader in it."""
    values = FactorValues(table, deals=Deals(**(BUSY | (deals or {}))), **fair)
    return BookMarket(values, seed=seed, traders=[trader] if trader is not None else [])


def decisions(market, trader, until):
    """Step the market, yielding the time after each step in which the trader decided."""
    seen = trader.next_time
    while market.time < until:
        market.step()
        if trader.next_time != seen:
            seen = trader.next_time
            yield market.time


# ---- The settings, and what a deal trader says of itself ----

def test_a_deal_trader_has_the_settings_the_spec_gives_after_a_state_traders_and_is_a_state_trader():
    trader = DealTrader()
    assert (trader.entry, trader.limit, trader.outcomes) == (8.0, 0, [])
    assert (trader.size, trader.interval, trader.rate, trader.band, trader.netted) == (5.0, 1.0, 1.0, 0.0, True)
    assert isinstance(trader, StateTrader)
    names = [f.name for f in dataclasses.fields(DealTrader)]
    assert names[-3:] == ["entry", "limit", "outcomes"] and names[:2] == ["size", "interval"]
    assert "book_cash" not in names and trader.outcomes is not DealTrader().outcomes
    assert [f.name for f in dataclasses.fields(DealRecord)] == ["target", "acquirer", "announced", "entered", "ended",
                                                                "outcome", "lots", "pnl", "flat"]


@pytest.mark.parametrize("bad", [
    {"size": 0.0}, {"size": -1.0}, {"size": float("nan")}, {"size": float("inf")}, {"entry": -0.5},
    {"entry": float("nan")}, {"entry": float("inf")}, {"limit": -1}, {"limit": 1.5}, {"limit": True},
    {"weights": [[1.0, -1.0, 0.0, 0.0]]}, {"states": [0]}, {"rate": 0.0}, {"rate": 1.5}, {"band": -1.0},
    {"netted": False, "rate": 0.5},
])
def test_a_deal_trader_refuses_settings_that_cannot_work(bad):
    with pytest.raises(ValueError):
        DealTrader(**bad)


def test_a_deal_trader_takes_a_limit_of_a_number_numpy_makes_and_an_entry_of_0():
    assert DealTrader(limit=np.int64(3), entry=0.0).limit == 3


def test_a_deal_trader_refuses_a_market_with_no_deals_when_added_and_is_not_added():
    market = BookMarket(FactorValues(FOUR), seed=1)
    trader = DealTrader()
    with pytest.raises(ValueError, match="^this market has no deals$"):
        market.add(trader)
    assert market.traders == [] and trader.position is None  # never attached
    with pytest.raises(ValueError, match="^this market has no deals$"):
        BookMarket(FactorValues(FOUR), seed=1, traders=[DealTrader()])


def test_a_deal_trader_is_added_to_a_market_that_has_deals_even_at_a_rate_of_0_and_starts_with_no_rows():
    trader = DealTrader()
    market = in_a_market(trader, deals={"rate": 0.0})
    assert market.traders == [trader]
    assert trader.weights.shape == (0, 4) and trader.states.shape == (0,) and trader.signal().shape == (0,)
    for _ in range(300):
        market.step()
    assert not trader.position.any() and trader.entries == 0 and trader.outcomes == []


def test_a_deal_trader_has_no_mids_to_learn_and_learn_takes_none_and_it_never_warms_up():
    trader = DealTrader(interval=2.5)
    stage = Stage(trader)
    mids = [np.array([1000.0, 1001.0, 1002.0, 1003.0])] * 30
    before = (trader.position.copy(), trader.entries, trader.cash, trader.weights.copy())
    assert trader.learn(mids) == 0 and trader.learn([]) == 0
    assert not hasattr(trader, "_mids")  # it keeps no mids
    assert (trader.position == before[0]).all() and trader.entries == before[1] and trader.cash == before[2]
    assert trader.weights.shape == before[3].shape
    assert trader.warming is None
    assert trader.warm_up == pytest.approx(2.5 + WAIT) == trader.taking(1)
    stage.decide(mids[0])
    assert trader.warming is None


def test_its_levels_are_its_entry_and_0_and_its_signal_is_the_gap_in_ticks_of_each_deal_on_in_the_rows_order():
    trader = DealTrader(entry=12.5)
    stage = Stage(trader)
    assert trader.levels == (12.5, 0.0)
    first, second = deal(0, 1, ratio=1.25), deal(2, 3, ratio=0.5)
    stage.start(first, second)
    stage.decide([1000.0, 800.0, 1000.0, 2000.0])
    assert trader.signal().tolist() == [1.25 * 800.0 - 1000.0, 0.5 * 2000.0 - 1000.0] == [0.0, 0.0]
    stage.decide([1000.0, 810.0, 990.0, 2040.0])
    assert trader.signal().tolist() == [12.5, 30.0]  # r * mid_A - mid_T, each in its own row
    stage.end(first)
    stage.decide([1000.0, 810.0, 990.0, 2040.0])
    assert trader.signal().tolist() == [30.0]
    stage.end(second)
    stage.decide([1000.0, 810.0, 990.0, 2040.0])
    assert trader.signal().shape == (0,)


# ---- Rows: one a deal on, +1 at the target and minus the ratio at the acquirer ----

def test_each_deal_on_is_a_row_of_plus_1_at_the_target_and_minus_its_ratio_at_the_acquirer_and_a_state_of_0_or_1():
    trader = DealTrader(entry=1e9)  # never enters
    stage = Stage(trader)
    stage.start(deal(1, 3, ratio=1.3, announced=2.0), deal(2, 0, ratio=0.85, announced=4.0))
    stage.decide([1000.0] * 4)
    assert trader.weights.tolist() == [[0.0, 1.0, 0.0, -1.3], [-0.85, 0.0, 1.0, 0.0]]  # not rescaled to a largest leg
    assert trader.states.tolist() == [0, 0] and trader.weights.shape == (2, 4)


@pytest.mark.parametrize("size, ratio", [(5, 1.34), (5, 1.26), (5, 1.3), (5, 1.5), (1, 1.4), (12, 0.9), (2.5, 1.1),
                                         (10, 1.02)])
def test_it_holds_size_lots_of_the_target_and_the_ratio_times_size_rounded_of_the_acquirer_short(size, ratio):
    trader = DealTrader(size=size, entry=0.0)
    stage = Stage(trader)
    stage.start(deal(1, 3, ratio=ratio))
    stage.decide([1000.0, 1000.0, 1000.0, 2000.0])  # r * 2000 is above 1000 for any of these ratios: a gap to enter at
    assert trader.position.tolist() == [0, int(np.rint(size)), 0, -int(np.rint(size * ratio))]


# ---- When it enters and when it leaves ----

@pytest.mark.parametrize("ratio, mid_target, mid_acquirer, enters", [
    (1.0, 1000.0, 1008.0, True),  # a gap of exactly 8 is at its entry
    (1.0, 1000.0, 1007.5, False), (1.0, 1000.0, 1030.0, True), (1.0, 1000.0, 990.0, False),
    (1.25, 1242.0, 1000.0, True), (1.25, 1242.5, 1000.0, False),  # the gap is r * mid_A - mid_T
    (0.5, 1000.0, 2016.0, True), (0.5, 1000.0, 2015.0, False),
])
def test_it_enters_when_the_gap_in_mids_is_at_least_its_entry_and_not_below_it(ratio, mid_target, mid_acquirer, enters):
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    stage.start(deal(0, 1, ratio=ratio))
    stage.decide([mid_target, mid_acquirer, 1000.0, 1000.0])
    assert trader.signal()[0] == ratio * mid_acquirer - mid_target
    assert trader.states.tolist() == [int(enters)] and trader.entries == int(enters)
    assert bool(trader.position[0]) == enters and bool(trader.position[1]) == enters


def test_it_enters_a_gap_of_exactly_its_entry_and_with_an_entry_of_0_a_gap_of_0_but_not_a_negative_one():
    for entry, gap, enters in ((8.0, 8.0, 1), (8.0, 7.999, 0), (0.0, 0.0, 1), (0.0, -0.001, 0), (30.0, 30.0, 1)):
        trader = DealTrader(entry=entry)
        stage = Stage(trader)
        stage.start(deal(0, 1))
        stage.decide([1000.0, 1000.0 + gap, 1000.0, 1000.0])
        assert trader.states.tolist() == [enters], (entry, gap)


def test_a_deal_whose_gap_is_below_its_entry_it_enters_at_the_first_decision_the_gap_reaches_it():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    stage.start(deal(0, 1))
    for gap in (2.0, 5.0, 7.5):
        assert not stage.decide([1000.0, 1000.0 + gap, 1000.0, 1000.0]) and not trader.position.any()
    moves = stage.decide([1000.0, 1009.0, 1000.0, 1000.0])
    assert [m.tolist() for m in moves] == [[5, -5, 0, 0]] and trader.entries == 1


def test_once_in_a_deal_it_stays_in_whatever_the_gap_does_until_the_deal_is_over():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    d = deal(0, 1)
    stage.start(d)
    stage.decide([1000.0, 1020.0, 1000.0, 1000.0])
    for gap in (30.0, 8.0, 3.0, 0.0, -25.0, -60.0):  # far below its entry, and negative
        assert not stage.decide([1000.0, 1000.0 + gap, 1000.0, 1000.0])
        assert trader.position.tolist() == [5, -5, 0, 0] and trader.states.tolist() == [1]
    stage.end(d)
    stage.decide([1000.0, 940.0, 1000.0, 1000.0])
    assert not trader.position.any() and trader.entries == 1


def test_with_a_limit_it_holds_that_many_deals_at_most_the_widest_gaps_first_and_with_none_any_number():
    mids = [1000.0, 1010.0, 1000.0, 1030.0, 1000.0, 1020.0]  # the gaps of the three deals: 10, 30 and 20
    for limit, expected in ((0, [1, 1, 1]), (3, [1, 1, 1]), (2, [0, 1, 1]), (1, [0, 1, 0])):
        trader = DealTrader(entry=8.0, limit=limit)
        stage = Stage(trader, books=6)
        stage.start(deal(0, 1, announced=1.0), deal(2, 3, announced=2.0), deal(4, 5, announced=3.0))
        stage.decide(mids)
        assert trader.states.tolist() == expected, limit
        assert [bool(trader.position[t]) for t in (0, 2, 4)] == [bool(s) for s in expected]


def test_a_limit_counts_the_deals_it_holds_and_never_gives_one_up_for_a_wider_gap_and_fills_a_room_that_opens():
    trader = DealTrader(entry=8.0, limit=2)
    stage = Stage(trader, books=6)
    d1, d2, d3 = deal(0, 1, announced=1.0), deal(2, 3, announced=2.0), deal(4, 5, announced=3.0)
    stage.start(d1, d2)
    stage.decide([1000.0, 1010.0, 1000.0, 1010.0, 1000.0, 1000.0])
    assert trader.states.tolist() == [1, 1]
    stage.start(d3)
    stage.decide([1000.0, 1010.0, 1000.0, 1010.0, 1000.0, 1100.0])  # a gap of 100: no room, and no one gives way
    assert trader.states.tolist() == [1, 1, 0] and not trader.position[4]
    stage.end(d1)
    stage.decide([1000.0, 1010.0, 1000.0, 1010.0, 1000.0, 1100.0])  # d1 is over: one room, and d3 takes it
    assert trader.states.tolist() == [1, 1] and trader.position[4] == 5 and trader.position[5] == -5


def test_with_a_limit_deals_of_equal_gaps_go_in_the_order_they_were_announced_and_a_gap_below_its_entry_never_does():
    trader = DealTrader(entry=8.0, limit=1)
    stage = Stage(trader, books=6)
    stage.start(deal(0, 1, announced=1.0), deal(2, 3, announced=2.0), deal(4, 5, announced=3.0))
    stage.decide([1000.0, 1020.0, 1000.0, 1020.0, 1000.0, 1004.0])
    assert trader.states.tolist() == [1, 0, 0]
    trader = DealTrader(entry=8.0, limit=1)
    stage = Stage(trader, books=6)
    stage.start(deal(0, 1, announced=1.0), deal(2, 3, announced=2.0))
    stage.decide([1000.0, 1004.0, 1000.0, 1007.9, 1000.0, 1000.0])  # the wider of two gaps below the entry: none
    assert trader.states.tolist() == [0, 0] and not trader.position.any()


# ---- Rows come and go: weights and states stay as long as one another, and a deal that goes on keeps its state ----

def test_weights_and_states_are_as_long_as_the_deals_on_as_they_come_and_go_and_one_that_goes_on_keeps_its_state():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    d1, d2, d3 = deal(0, 1, announced=1.0), deal(2, 3, announced=2.0), deal(0, 1, announced=40.0)
    stage.start(d1, d2)
    stage.decide([1000.0, 1000.0, 1000.0, 1020.0])  # d1 has no gap, and d2 is entered: rows [d1, d2], states [0, 1]
    assert trader.states.tolist() == [0, 1] and trader.entries == 1
    assert len(trader.weights) == len(trader.states) == 2
    stage.end(d1)  # d2 is now the first row, and keeps its state, its lots, and its place in what netting counts
    assert not stage.decide([1000.0, 1000.0, 1000.0, 1020.0])
    assert trader.states.tolist() == [1] and trader.weights.tolist() == [[0.0, 0.0, 1.0, -1.0]]
    assert trader.entries == 1 and trader.netted_lots == 0 and trader.position.tolist() == [0, 0, 5, -5]
    stage.start(d3)  # a deal comes after it: [d2, d3]
    moves = stage.decide([1000.0, 1050.0, 1000.0, 1020.0])
    assert trader.states.tolist() == [1, 1] and trader.entries == 2 and trader.netted_lots == 0
    assert [m.tolist() for m in moves] == [[5, -5, 0, 0]]  # only the new deal's lots
    stage.end(d2)
    stage.decide([1000.0, 1050.0, 1000.0, 1020.0])
    assert trader.states.tolist() == [1] and trader.weights.tolist() == [[1.0, -1.0, 0.0, 0.0]]
    assert trader.position.tolist() == [5, -5, 0, 0]  # the books of the one that ended are closed, the others kept


def test_a_deals_state_is_kept_by_its_target_its_acquirer_and_the_time_it_was_announced():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    first, second = deal(0, 1, announced=1.0), deal(0, 1, announced=60.0)  # one pair of books, two deals in turn
    stage.start(first)
    stage.decide([1000.0, 1020.0, 1000.0, 1000.0])
    assert trader.states.tolist() == [1]
    stage.end(first)
    stage.start(second)
    stage.decide([1000.0, 1000.0, 1000.0, 1000.0])  # the same books, a new deal and no gap: not the old one's state
    assert trader.states.tolist() == [0] and trader.entries == 1


def test_with_no_deal_on_it_has_no_rows_and_closes_everything_at_once_and_forgets_its_aim():
    trader = DealTrader(entry=0.0, rate=0.2)
    stage = Stage(trader)
    d = deal(0, 1)
    stage.start(d)
    for _ in range(30):
        stage.decide([1000.0, 1000.0, 1000.0, 1000.0])
    assert trader.position.tolist() == [5, -5, 0, 0] and trader._aim.any()
    stage.end(d)
    moves = stage.decide([1000.0, 1000.0, 1000.0, 1000.0])
    assert [m.tolist() for m in moves] == [[-5, 5, 0, 0]] and not trader.position.any()
    assert trader.weights.shape == (0, 4) and trader.states.shape == (0,) and not trader._aim.any()
    assert not stage.decide([1000.0, 1000.0, 1000.0, 1000.0])  # and nothing more to do


# ---- A deal that is over is closed all the way ----

@pytest.mark.parametrize("rate, band", [(1.0, 0.0), (0.2, 0.0), (1.0, 3.0), (0.3, 2.0)])
@pytest.mark.parametrize("another", [True, False])
def test_a_deal_that_is_over_is_closed_all_the_way_at_that_decision_whatever_the_rate_and_the_band(rate, band, another):
    trader = DealTrader(size=10, entry=0.0, rate=rate, band=band)
    stage = Stage(trader)
    d1, d2 = deal(0, 1), deal(2, 3)
    stage.start(d1, *([d2] if another else []))
    for _ in range(40):  # to what it holds in each: the whole size, or its edge of the band
        stage.decide([1000.0, 1000.0, 1000.0, 1000.0])
    held = trader.position.copy()
    assert held[0] == int(np.rint(10 - band)) > 0 and held[1] == -held[0]
    stage.end(d1)
    stage.decide([1000.0, 1000.0, 1000.0, 1000.0])
    assert not trader.position[:2].any()  # both books of the deal, at that one decision
    if another:
        assert trader.position[2:].tolist() == held[2:].tolist()  # the other deal goes on as it was
    else:
        assert not trader.position.any()


@pytest.mark.parametrize("rate", [1.0, 0.2])
def test_it_goes_on_closing_a_deal_that_is_over_at_each_decision_until_its_books_are_flat(rate):
    trader = DealTrader(size=10, entry=0.0, rate=rate)
    stage = Stage(trader)
    d1, d2 = deal(0, 1), deal(2, 3)
    stage.start(d1, d2)
    for _ in range(40):
        stage.decide([1000.0, 1000.0, 1000.0, 1000.0])
    assert trader.position.tolist() == [10, -10, 10, -10]
    stage.end(d1)
    seen = []
    for _ in range(8):  # each order fills half of what was asked
        stage.decide([1000.0, 1000.0, 1000.0, 1000.0], fill=half)
        seen.append(int(trader.position[0]))
    assert seen[:3] == [5, 3, 2] and seen[-1] == 0 and seen == sorted(seen, reverse=True)
    assert not trader.position[:2].any() and trader.position[2:].tolist() == [10, -10]


def test_the_lots_of_a_deal_still_on_are_asked_for_at_its_rate_and_within_its_band_as_for_any_trader():
    trader = DealTrader(size=10, entry=0.0, rate=0.5, band=2.0)
    stage = Stage(trader)
    stage.start(deal(0, 1))
    held = []
    for _ in range(12):
        stage.decide([1000.0, 1000.0, 1000.0, 1000.0])
        held.append(int(trader.position[0]))
    assert held[0] == 3 and held == sorted(held) and held[-1] == 8  # the aim goes half way: 5, 7.5, ...; the band is 2


def test_a_deal_trader_that_does_not_net_its_orders_holds_the_same_lots_and_closes_a_deal_that_is_over_all_the_same():
    for netted in (True, False):
        trader = DealTrader(size=5, entry=0.0, netted=netted)
        stage = Stage(trader, books=6)
        a, b = deal(0, 1, ratio=1.3), deal(2, 3, ratio=0.8)
        stage.start(a, b)
        stage.decide([1000.0, 2000.0, 1000.0, 2000.0, 1000.0, 1000.0])
        assert trader.position.tolist() == [5, -6, 5, -4, 0, 0], netted
        stage.end(a)
        stage.decide([1000.0, 2000.0, 1000.0, 2000.0, 1000.0, 1000.0])
        assert trader.position.tolist() == [0, 0, 5, -4, 0, 0], netted
        stage.end(b)
        stage.decide([1000.0, 2000.0, 1000.0, 2000.0, 1000.0, 1000.0])
        assert not trader.position.any(), netted


# ---- Records ----

def test_a_deal_is_a_record_once_it_is_over_and_both_its_books_are_flat_with_the_change_in_the_cash_of_the_two():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    d = deal(0, 1, announced=0.5)
    stage.start(d)
    trader.book_cash[0] = -777.0  # cash it had before from other trades in these books: not the deal's
    trader.book_cash[2] = 41.0
    stage.decide([1000.0, 1030.0, 1000.0, 1000.0])  # t = 1: 5 lots bought at 1000 and sold at 1030, which is 150
    assert trader.position.tolist() == [5, -5, 0, 0] and trader.outcomes == []
    stage.decide([1010.0, 1030.0, 1000.0, 1000.0])
    stage.end(d, "completed", at=3.5)
    stage.decide([1020.0, 1030.0, 1000.0, 1000.0])  # t = 3: sells 5 at 1020 and buys 5 at 1030, which is 100 less 150
    assert not trader.position.any() and trader.outcomes == []  # flat, but it is seen at the next decision
    stage.decide([1020.0, 1030.0, 1000.0, 1000.0])
    assert trader.outcomes == [DealRecord(0, 1, announced=0.5, entered=1.0, ended=3.5, outcome="completed", lots=5,
                                          pnl=100.0, flat=4.0)]
    stage.decide([1020.0, 1030.0, 1000.0, 1000.0])
    assert len(trader.outcomes) == 1  # once


def test_a_deal_that_is_over_but_not_flat_in_both_books_is_not_a_record_until_it_is():
    trader = DealTrader(size=10, entry=0.0)
    stage = Stage(trader)
    d = deal(0, 1)
    stage.start(d)
    stage.decide([1000.0, 1000.0, 1000.0, 1000.0])
    assert trader.position.tolist() == [10, -10, 0, 0]
    stage.end(d, "broken", at=1.5)
    held, records = [], []
    for _ in range(6):  # at 2 to 7: each order fills half of what it asks
        stage.decide([950.0, 1000.0, 1000.0, 1000.0], fill=half)
        held.append(int(trader.position[0]))
        records.append(len(trader.outcomes))
    assert held == [5, 3, 2, 1, 0, 0]  # flat after the decision at 6, and seen so at the next
    assert records == [0, 0, 0, 0, 0, 1]
    assert trader.outcomes == [DealRecord(0, 1, announced=0.0, entered=1.0, ended=1.5, outcome="broken", lots=10,
                                          pnl=-500.0, flat=7.0)]  # bought at 1000, sold at 950; the acquirer's even


def test_a_deal_that_is_over_before_the_trader_decides_while_it_is_on_is_never_entered_and_leaves_no_row():
    trader = DealTrader(entry=0.0)
    stage = Stage(trader)
    gone = deal(0, 1)
    stage.start(gone)
    stage.end(gone)  # on and over between two decisions
    assert not stage.decide([1000.0, 1100.0, 1000.0, 1000.0]) and trader.weights.shape == (0, 4)
    assert trader.entries == 0 and trader.outcomes == [] and not trader.position.any()


def test_a_deal_entered_whose_orders_never_fill_is_a_record_of_no_lots_and_no_profit_once_it_is_over():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    d = deal(0, 1, announced=0.5)
    stage.start(d)
    for _ in range(2):  # at 1 and 2: it enters, and no order of it fills in either book
        stage.decide([1000.0, 1030.0, 1000.0, 1000.0], skip={0, 1})
    assert not trader.position.any() and trader.entries == 1 and trader.outcomes == []  # the deal is still on
    stage.end(d, "broken", at=2.5)
    stage.decide([1000.0, 1030.0, 1000.0, 1000.0], skip={0, 1})
    assert trader.outcomes == [DealRecord(0, 1, announced=0.5, entered=1.0, ended=2.5, outcome="broken", lots=0,
                                          pnl=0.0, flat=3.0)]


def test_a_deal_held_in_other_books_than_the_one_entered_on_a_book_not_flat_stays_a_record_with_its_own_profit():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader, books=6)
    first, other, third = deal(0, 1, announced=1.0), deal(2, 3, announced=1.5), deal(0, 4, announced=3.0)
    stage.start(first, other)
    stage.decide([1000.0, 1030.0, 1000.0, 1030.0, 1000.0, 1000.0])  # t = 1: both are entered, 5 lots a book
    assert trader.position.tolist() == [5, -5, 5, -5, 0, 0] and trader.not_flat_at_entry == 0
    stage.end(first)  # over, and book 0 is not flat when a new deal takes it
    stage.start(third)
    stage.decide([1000.0, 1030.0, 1000.0, 1030.0, 1030.0, 1000.0])
    assert trader.position.tolist() == [5, 0, 5, -5, -5, 0]
    assert trader.not_flat_at_entry == 2  # the third, and the first still being closed in book 0: not the other deal
    stage.end(other, at=2.5)
    stage.end(third)
    stage.decide([1000.0, 1030.0, 1020.0, 1030.0, 1030.0, 1000.0])  # t = 3: everything is closed
    assert not trader.position.any()
    stage.decide([1000.0, 1030.0, 1020.0, 1030.0, 1030.0, 1000.0])
    # the other deal: bought 5 at 1000 and sold 5 at 1030, then sold 5 at 1020 and bought 5 at 1030, which is 100
    assert trader.outcomes == [DealRecord(2, 3, announced=1.5, entered=1.0, ended=2.5, outcome="completed", lots=5,
                                          pnl=100.0, flat=4.0)]
    assert trader.not_flat_at_entry == 2


def test_a_recording_deal_trader_keeps_the_gaps_it_saw_a_deal_each_as_they_were_then():
    trader = DealTrader(entry=1e9, interval=0.5, record=True)
    market = in_a_market(trader, table=SIX, seed=8)
    seen = []
    for _ in decisions(market, trader, 200.0):
        seen.append(len(market.fair.deals))
    assert len(trader.history) == len(seen) > 200 and sum(seen) > 100
    for (_, mids, position, signal), count in zip(trader.history, seen, strict=True):
        assert len(signal) == count and len(mids) == len(position) == 6
    assert len({id(row[3]) for row in trader.history}) == len(trader.history)  # an array of its own each time


def test_a_deal_it_never_entered_has_no_record_and_a_record_holds_the_most_of_the_target_it_held():
    trader = DealTrader(entry=8.0, rate=0.5, size=10)
    stage = Stage(trader)
    never, entered = deal(0, 1, announced=1.0), deal(2, 3, announced=2.0)
    stage.start(never, entered)
    for _ in range(10):
        stage.decide([1000.0, 1002.0, 1000.0, 1040.0])
    stage.end(never)
    stage.end(entered)
    for _ in range(3):
        stage.decide([1000.0, 1002.0, 1000.0, 1040.0])
    assert [(r.target, r.acquirer) for r in trader.outcomes] == [(2, 3)] and trader.outcomes[0].lots == 10


def test_a_deal_entered_while_a_book_was_not_flat_is_counted_and_not_recorded_nor_is_the_deal_still_being_closed():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader, books=4)
    first, second, clean = deal(0, 1, announced=1.0), deal(0, 2, announced=2.0), deal(1, 3, announced=3.0)
    stage.start(first)
    stage.decide([1000.0, 1020.0, 1000.0, 1000.0])
    assert trader.position.tolist() == [5, -5, 0, 0] and trader.not_flat_at_entry == 0
    stage.end(first)  # over, and a new deal is on in book 0 before it is flat there (a cooling time too short)
    stage.start(second)
    stage.decide([1000.0, 1020.0, 1020.0, 1000.0], fill=half)
    assert trader.not_flat_at_entry == 2  # the new deal, and the one that was still being closed in book 0
    stage.end(second)
    for _ in range(5):
        stage.decide([1000.0, 1020.0, 1020.0, 1000.0])
    assert not trader.position.any() and trader.outcomes == [] and trader.entries == 2
    stage.start(clean)  # books 1 and 3 are flat: this one is a record
    stage.decide([1000.0, 1000.0, 1000.0, 1030.0])
    stage.end(clean)
    stage.decide([1000.0, 1000.0, 1000.0, 1030.0])
    stage.decide([1000.0, 1000.0, 1000.0, 1030.0])
    assert [(r.target, r.acquirer) for r in trader.outcomes] == [(1, 3)] and trader.not_flat_at_entry == 2


def test_a_deal_is_not_a_record_while_either_book_is_not_flat_the_acquirers_as_much_as_the_targets():
    for stuck, remaining in ((1, [0, -5, 0, 0]), (0, [5, 0, 0, 0])):  # the orders of one book never fill
        trader = DealTrader(entry=8.0)
        stage = Stage(trader)
        d = deal(0, 1)
        stage.start(d)
        stage.decide([1000.0, 1030.0, 1000.0, 1000.0])
        assert trader.position.tolist() == [5, -5, 0, 0]
        stage.end(d)
        for _ in range(3):  # the orders in one book never fill: the other is flat after the first
            stage.decide([1000.0, 1030.0, 1000.0, 1000.0], skip={stuck})
            assert trader.position.tolist() == remaining and trader.outcomes == []
        stage.decide([1000.0, 1030.0, 1000.0, 1000.0])  # it fills now
        assert not trader.position.any() and trader.outcomes == []
        stage.decide([1000.0, 1030.0, 1000.0, 1000.0])
        assert [(r.target, r.acquirer, r.flat) for r in trader.outcomes] == [(0, 1, 6.0)]


def test_a_deal_entered_while_only_its_acquirers_book_is_not_flat_is_counted_and_so_is_the_deal_closed_there():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    first, second = deal(0, 1, announced=1.0), deal(2, 1, announced=2.0)  # the second one's acquirer is the first's
    stage.start(first)
    stage.decide([1000.0, 1030.0, 1000.0, 1000.0])
    stage.end(first)
    stage.start(second)
    stage.decide([1000.0, 1030.0, 1000.0, 1000.0], skip={1})  # book 0 is flat, the target of the second is flat too
    assert trader.position.tolist() == [0, -5, 5, 0] and trader.not_flat_at_entry == 2
    stage.end(second)
    for _ in range(4):
        stage.decide([1000.0, 1030.0, 1000.0, 1000.0])
    assert not trader.position.any() and trader.outcomes == [] and trader.entries == 2


def test_two_deals_held_at_once_have_records_each_of_the_cash_of_its_own_two_books():
    trader = DealTrader(entry=8.0)
    stage = Stage(trader)
    a, b = deal(0, 1, announced=1.0), deal(2, 3, announced=2.0)
    stage.start(a, b)
    stage.decide([1000.0, 1020.0, 2000.0, 2050.0])  # a: 5 at 1000, -5 at 1020; b: 5 at 2000, -5 at 2050
    stage.end(a, "completed", at=1.5)
    stage.decide([1010.0, 1020.0, 2100.0, 2050.0])  # a closes at 1010 and 1020; b is held
    stage.end(b, "broken", at=2.5)
    stage.decide([1010.0, 1020.0, 1900.0, 2050.0])  # b closes at 1900 and 2050
    stage.decide([1010.0, 1020.0, 1900.0, 2050.0])
    records = {(r.target, r.acquirer): r for r in trader.outcomes}
    assert records[(0, 1)].pnl == pytest.approx(5 * (1010.0 - 1000.0) + 5 * (1020.0 - 1020.0))
    assert records[(2, 3)].pnl == pytest.approx(5 * (1900.0 - 2000.0) + 5 * (2050.0 - 2050.0))
    assert sum(r.pnl for r in trader.outcomes) == pytest.approx(trader.cash)


# ---- In a market ----

def test_in_a_market_it_holds_the_lots_its_rows_ask_through_every_deal_and_none_in_a_book_in_no_row_after_a_decision():
    trader = DealTrader(size=5, entry=8.0, interval=0.5)
    market = in_a_market(trader, table=SIX, seed=2)
    held, checked = 0, 0
    for _ in decisions(market, trader, 500.0):
        fair = market.fair
        assert len(trader.weights) == len(trader.states) == len(fair.deals)
        for row, state, d in zip(trader.weights, trader.states, fair.deals, strict=True):  # a row a deal, in order
            assert row[d.target] == 1.0 and row[d.acquirer] == -d.ratio and np.count_nonzero(row) == 2
            assert state in (0, 1)
        asked = np.rint(trader.size * (trader.states @ trader.weights)).astype(int)
        assert trader.position.tolist() == asked.tolist()
        held += int(trader.states.sum())
        checked += 1
    assert checked > 600 and held > 300 and len(trader.outcomes) > 20


def test_the_deals_it_enters_are_those_whose_gap_in_mids_is_at_least_its_entry_then_and_it_never_leaves_one():
    @dataclasses.dataclass
    class Spy(DealTrader):
        """Notes for each deal on at a decision its state before and after it, and its gap in mids."""
        log: list = dataclasses.field(default_factory=list)

        def attach(self, n):
            super().attach(n)
            self.rows = []  # the deals its states are of, as the last decision left them

        def observe(self, market):
            super().observe(market)
            self.on = list(market.fair.deals)

        def decide(self, mids):
            before = {id(d): int(state) for d, state in zip(self.rows, self.states, strict=True)}
            moves = super().decide(mids)
            self.rows = self.on
            for d, state in zip(self.rows, self.states, strict=True):
                self.log.append((before.get(id(d), 0), int(state), d.ratio * mids[d.acquirer] - mids[d.target]))
            return moves

    trader = Spy(entry=24.0, interval=0.5)  # about the gap a deal opens at, so that some are entered later or never
    market = in_a_market(trader, table=SIX, seed=3)
    for _ in decisions(market, trader, 600.0):
        pass
    entered = [(before, after, gap) for before, after, gap in trader.log if before == 0 and after == 1]
    waiting = [(before, after, gap) for before, after, gap in trader.log if before == 0 and after == 0]
    assert len(entered) > 15 and len(waiting) > 40
    assert all(gap >= 24.0 for _, _, gap in entered) and all(gap < 24.0 for _, _, gap in waiting)
    assert all(after == 1 for before, after, _ in trader.log if before == 1)  # in, it stays in while the deal is on


def test_with_a_limit_it_never_holds_more_deals_than_that_and_does_hold_that_many_when_there_are_deals_to_hold():
    for limit in (1, 2):
        trader = DealTrader(entry=0.0, limit=limit, interval=0.5)
        market = in_a_market(trader, table=SIX, seed=4)
        most = 0
        for _ in decisions(market, trader, 500.0):
            held = int(trader.states.sum())
            assert held <= limit
            most = max(most, held)
        assert most == limit and len(trader.outcomes) > 10


@pytest.mark.parametrize("rate, band", [(0.3, 0.0), (1.0, 2.0), (0.3, 2.0)])
def test_with_a_rate_or_a_band_a_finished_deal_is_closed_all_the_way_at_the_first_decision_that_sees_it(rate, band):
    trader = DealTrader(size=5, entry=0.0, rate=rate, band=band, interval=0.5)
    market = in_a_market(trader, seed=5)
    flats = 0
    for _ in decisions(market, trader, 600.0):
        on = {book for d in market.fair.deals for book in (d.target, d.acquirer)}
        for book in range(4):
            if book not in on:  # in no deal on: whatever it held there when the deal ended is closed by now
                assert trader.position[book] == 0
                flats += 1
    assert flats > 500 and len(trader.outcomes) > 15 and trader.not_flat_at_entry == 0
    assert max(r.lots for r in trader.outcomes) <= 5 - int(band) and all(r.lots <= 5 for r in trader.outcomes)


@pytest.mark.parametrize("rate, band", [(1.0, 0.0), (0.3, 0.0), (1.0, 2.0)])
def test_the_pnl_of_its_records_add_up_to_its_whole_pnl_once_it_is_flat_at_the_end_with_nothing_open(rate, band):
    trader = DealTrader(size=5, entry=8.0, rate=rate, band=band, interval=0.5)
    market = in_a_market(trader, table=SIX, seed=6, deals={"cool": 20.0})
    market.run(500.0)
    assert len(trader.outcomes) > 20
    trader.entry = 1e9  # no new deal, from here: it sees out those it holds, and is flat when the last is closed
    while trader.entries != len(trader.outcomes) or trader.position.any():
        market.step()
        assert market.time < 900.0
    assert not trader.position.any() and not trader.states.any()
    marks = market.marks()
    assert trader.value(marks) == trader.cash == pytest.approx(float(trader.book_cash.sum()))
    assert sum(r.pnl for r in trader.outcomes) == pytest.approx(trader.cash, abs=1e-6)
    assert trader.entries == len(trader.outcomes) and trader.not_flat_at_entry == 0  # each deal entered is a record
    for r in trader.outcomes:
        assert r.announced < r.entered <= r.ended and r.ended <= r.flat and r.outcome in ("completed", "broken")
        assert 0 <= r.lots <= 5


def test_it_has_a_record_for_each_deal_it_entered_once_each_with_the_deals_own_outcome_and_the_time_it_was_over():
    trader = DealTrader(size=5, entry=0.0, interval=0.5)
    market = in_a_market(trader, table=SIX, seed=7, deals={"cool": 20.0})
    market.run(400.0)
    trader.entry = 1e9
    while trader.entries != len(trader.outcomes) or trader.position.any():
        market.step()
        assert market.time < 800.0
    keys = [(r.target, r.acquirer, r.announced) for r in trader.outcomes]
    assert len(set(keys)) == len(keys) == trader.entries
    ended = {(d.target, d.acquirer, d.announced): d for d in market.fair.ended}
    for r in trader.outcomes:
        d = ended[(r.target, r.acquirer, r.announced)]
        assert (r.outcome, r.ended) == (d.outcome, d.ended_at)
    assert float(trader.book_cash.sum()) == pytest.approx(sum(r.pnl for r in trader.outcomes), abs=1e-6)


# ---- One lot of the target against one of the acquirer, at fair values and in the mids ----

@functools.lru_cache(maxsize=None)
def lot_pairs(priced=0.4, breaks=0.2, length=100.0, rate=0.2, units=1500.0):
    """One lot of the target and one of the acquirer, entered at the first decision after each announcement and closed
    after the deal, over the deals of seeds 1 to 8 that were over by the end (those announced at least a length and a
    margin before it): four books with no own parts and no impact, a premium of 60. A list of dicts, one a deal: its
    outcome, ``fair`` what its four orders made at the books' fair values when they were made, ``before`` what it made
    before costs (the change in its cash from the two books, and the spread it paid on them), ``mids`` the same from the
    mids before each order (which is ``before``), ``cost`` the spread paid, and ``wait`` the time from the announcement
    to its entry."""
    rows = []
    log = []
    original = BookMarket._trade

    def noting(self, trader, book, need):
        paid = trader.spread_paid
        mid = self.books[book].mid_price
        original(self, trader, book, need)
        log.append((self.time, book, need, self.fair.fairs[book], mid, trader.spread_paid - paid))

    settings = dict(rate=rate, premium=60.0, length=length, breaks=breaks, priced=priced, cool=2.0)
    with mock.patch.object(BookMarket, "_trade", noting):
        for seed in SEEDS:
            log.clear()
            trader = DealTrader(size=1, entry=0.0, interval=0.25)
            fair = FactorValues(FOUR, own_volatility=0.0, impact=0.0, deals=Deals(**settings))
            market = BookMarket(fair, seed=seed, traders=[trader])
            while market.time < units:
                market.step()
            assert trader.not_flat_at_entry == 0
            for r in trader.outcomes:
                if r.announced > units - length - 10.0:
                    continue
                mine = [x for x in log if r.entered <= x[0] <= r.flat and x[1] in (r.target, r.acquirer)]
                orders = [(x[1], x[2]) for x in mine]  # in, and out: a lot of each book, the acquirer's short
                assert set(orders[:2]) == {(r.target, 1), (r.acquirer, -1)} and len(orders) == 4
                assert set(orders[2:]) == {(r.target, -1), (r.acquirer, 1)}
                rows.append({"outcome": r.outcome, "fair": sum(-x[2] * x[3] for x in mine),
                             "before": r.pnl + sum(x[5] for x in mine), "mids": sum(-x[2] * x[4] for x in mine),
                             "wait": r.entered - r.announced, "cost": sum(x[5] for x in mine)})
    return rows


def standard_error(values):
    values = np.asarray(values, dtype=float)
    return float(values.std() / math.sqrt(len(values)))


def test_one_lot_pair_entered_at_the_first_decision_after_an_announcement_earns_at_fair_values_what_it_works_out_to():
    rows = lot_pairs()
    assert len(rows) > 200
    assert max(r["wait"] for r in rows) < 1.0 and min(r["wait"] for r in rows) > 0.0  # the first decision after
    done = np.array([r["fair"] for r in rows if r["outcome"] == "completed"])
    broke = np.array([r["fair"] for r in rows if r["outcome"] == "broken"])
    assert len(done) > 150 and len(broke) > 40
    # a completed deal makes (1 - pi_0) * 60 = 24 and a broken one loses pi_0 * 60 = 36, hedged with a lot, not r
    assert abs(done.mean() - 24.0) < 0.5 and abs(broke.mean() + 36.0) < 0.2
    # so (1 - breaks) * 24 - breaks * 36 = 12 on average over deals, 20% of which break
    everything = np.concatenate([done, broke])
    se = standard_error(everything)
    assert 1.0 < se < 2.5  # the standard error of the mean over the deals is about 1.6 ticks
    assert abs(everything.mean() - 12.0) < 3 * se
    assert abs(len(broke) / len(everything) - 0.2) < 4 * math.sqrt(0.2 * 0.8 / len(everything))


def test_the_same_lot_pair_earns_before_costs_in_the_mids_the_same_less_the_lag_behind_a_fair_value_on_the_move():
    rows = lot_pairs()
    before = np.array([r["before"] for r in rows])
    fair = np.array([r["fair"] for r in rows])
    se = standard_error(before)
    assert 1.0 < se < 2.5  # about 1.6 ticks: a standard deviation of 25 ticks a deal over about 240 deals
    assert abs(before.mean() - 12.0) < 3 * se  # what the arithmetic says, within three standard errors
    # the mids are what the cash and the spread paid say they are
    assert max(abs(r["before"] - r["mids"]) for r in rows) < 1e-6
    # they trail the fair values by about 2 ticks, a lag the quotes keep behind a fair value that drifts (the deal's)
    lag = (before - fair).mean()
    assert -4.0 < lag < -1.0 and standard_error(before - fair) < 0.3
    # a round trip crosses four spreads of about a tick each
    assert 3.0 < np.mean([r["cost"] for r in rows]) < 5.0


def test_with_priced_equal_to_breaks_the_lot_pair_earns_nothing_on_average_at_fair_values_and_each_outcome_what_it_is():
    rows = lot_pairs(priced=0.3, breaks=0.3, length=40.0, rate=0.4, units=900.0)
    done = np.array([r["fair"] for r in rows if r["outcome"] == "completed"])
    broke = np.array([r["fair"] for r in rows if r["outcome"] == "broken"])
    assert len(done) > 200 and len(broke) > 80
    assert abs(done.mean() - 18.0) < 0.5 and abs(broke.mean() + 42.0) < 0.2  # 0.3 * 60, and -0.7 * 60
    everything = np.concatenate([done, broke])
    se = standard_error(everything)
    assert 0.5 < se < 2.0
    assert abs(everything.mean()) < 3 * se  # (1 - 0.3) * 18 - 0.3 * 42 = 0


# ---- The news-aware pairs trader ----

PAIRS = [[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]]


def pair_trader(cls=NewsAwareTrader, **kwargs):
    return cls(weights=PAIRS, window=4, entry=1.0, exit=0.0, size=2, interval=1.0, **kwargs)


def spikes(at):
    """The mids of a decision: both pairs' spreads 0, or 10 at a decision in ``at`` (the books of one are 1000)."""
    return [1010.0, 1000.0, 1010.0, 1000.0] if at else [1000.0, 1000.0, 1000.0, 1000.0]


def test_a_news_aware_trader_has_a_quarantine_of_the_time_its_window_takes_unless_given_one():
    # each decision waits its interval and then for the next event (``taking``), so a little more than the interval
    for kwargs in (dict(window=4, interval=2.5), dict(), dict(window=7), dict(window=400, interval=5.0)):
        trader = NewsAwareTrader(weights=PAIRS, **kwargs)
        assert trader.quarantine == trader.taking(trader.window) > trader.window * trader.interval
    assert NewsAwareTrader(weights=PAIRS, window=4, interval=2.5).quarantine == pytest.approx(4 * 2.53)
    assert NewsAwareTrader(weights=PAIRS).quarantine == pytest.approx(400 * 1.03)
    assert NewsAwareTrader(weights=PAIRS, window=400, interval=5.0).quarantine == pytest.approx(2012.0)
    assert NewsAwareTrader(weights=PAIRS, window=4, interval=2.5, quarantine=3.0).quarantine == 3.0
    assert NewsAwareTrader(weights=PAIRS, quarantine=0.0).quarantine == 0.0
    assert isinstance(NewsAwareTrader(weights=PAIRS, window=7).quarantine, float)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            NewsAwareTrader(weights=PAIRS, quarantine=bad)
    assert issubclass(NewsAwareTrader, RelationTrader)
    assert [f.name for f in dataclasses.fields(NewsAwareTrader)][-2:] == ["quarantine", "acquirers"]
    assert NewsAwareTrader(weights=PAIRS).acquirers is False  # only the targets of deals, unless told
    assert NewsAwareTrader(weights=PAIRS, acquirers=True).acquirers is True


def test_it_keeps_the_settings_and_refusals_of_a_relation_trader():
    with pytest.raises(ValueError):
        NewsAwareTrader(weights=PAIRS, window=2)
    with pytest.raises(ValueError):
        NewsAwareTrader()
    with pytest.raises(ValueError):
        NewsAwareTrader(weights=PAIRS, rate=0.5, netted=False)
    trader = NewsAwareTrader(choose=pairs_by_distance, window=20, formation=50)
    assert trader.warm_up == RelationTrader(choose=pairs_by_distance, window=20, formation=50).warm_up


def test_a_news_aware_trader_that_has_not_chosen_its_relations_yet_asks_for_nothing_with_a_deal_on():
    trader = NewsAwareTrader(choose=pairs_by_distance, window=4, formation=8, entry=1.0, size=2, interval=1.0)
    stage = Stage(trader)
    stage.start(deal(0, 1))  # a deal is on through the whole formation period, when it has no relation to hold
    for t in range(1, 8):
        assert stage.decide(spikes(t % 3 == 0)) == []
        assert trader.weights is None and trader._quiet == [0] and not trader.position.any()  # the target's book


def run_stage(trader, spiked, on=None, ended_at=None, between=(0, 1)):
    """The decisions at times 1 to 14, one unit apart, with spikes in the mids at those in ``spiked``. With ``on`` (the
    first and the last decision it is on at) a deal from the acquirer ``between[1]`` for the target ``between[0]`` is
    on for those and over for the next, having ended at ``ended_at`` in between. The states and the lots after each
    decision, by its time."""
    stage = Stage(trader)
    news = deal(*between)
    out = {}
    for t in range(1, 15):
        if on and t == on[0]:
            stage.start(news)
        if on and t == on[1] + 1:
            stage.end(news, at=ended_at)
        stage.decide(spikes(t in spiked))
        out[t] = (trader.states.tolist(), trader.position.tolist())
    return out


def test_a_relation_with_a_book_in_a_deal_is_flat_and_stays_flat_until_the_quarantine_after_it_ends_and_others_trade():
    spiked = {4, 8, 12}  # a spread far above its window at each: the plain trader goes short both pairs
    news = run_stage(pair_trader(quarantine=4.0), spiked, on=(4, 5), ended_at=5.5)  # on at 4 and 5, over by 6
    plain = run_stage(pair_trader(RelationTrader), spiked)
    assert plain[4] == ([-1, -1], [-2, 2, -2, 2])
    # at 4 the deal is on: the pair of books 0 and 1 is flat, and the other pair is short
    assert news[4] == ([0, -1], [0, 0, -2, 2])
    # at 8 the deal ended 2.5 ago, less than the 4 it is quarantined for: still flat, though its spread says short
    assert plain[8][0] == [-1, -1] and news[8][0] == [0, -1] and news[8][1][:2] == [0, 0]
    # at 12 it ended 6.5 ago: it trades as the plain trader does
    assert news[12] == plain[12] == ([-1, -1], [-2, 2, -2, 2])
    # the pair that holds no book in a deal trades as the plain trader's does, at every decision
    assert all(news[t][0][1] == plain[t][0][1] and news[t][1][2:] == plain[t][1][2:] for t in news)
    assert sum(news[t][0][1] != 0 for t in news) >= 3  # and it trades: short at 4, 8 and 12


@pytest.mark.parametrize("ended_at, free", [(8.0, True), (7.5, True), (8.001, False), (9.0, False), (11.0, False)])
def test_a_deal_that_ended_exactly_the_quarantine_ago_is_not_in_it_and_one_that_ended_less_ago_is(ended_at, free):
    trader = pair_trader(quarantine=4.0)  # a quarantine of 4 units, given: its default would be 4.12
    stage = Stage(trader)
    news = deal(0, 1)
    stage.start(news)
    for _ in range(11):
        stage.decide(spikes(False))
    stage.end(news, at=ended_at)
    stage.decide(spikes(True))  # at 12 the spread of each pair is far above its window's
    assert trader.states.tolist() == [-1 if free else 0, -1]


def test_a_relation_with_a_book_that_goes_into_a_deal_closes_the_lots_it_holds_at_once_and_others_keep_theirs():
    trader = pair_trader()
    stage = Stage(trader)
    for _ in range(3):
        stage.decide(spikes(False))
    stage.decide(spikes(True))
    assert trader.position.tolist() == [-2, 2, -2, 2]
    stage.start(deal(0, 1, announced=4.5))  # a deal in a book of the first pair
    stage.decide(spikes(True))  # the spreads are as far above their windows' as they were: the other pair stays short
    assert trader.states.tolist() == [0, -1] and trader.position.tolist() == [0, 0, -2, 2]
    stage.decide(spikes(True))
    assert trader.states.tolist() == [0, -1] and trader.position.tolist() == [0, 0, -2, 2]  # and the first stays flat


# the deal's (target, acquirer) in pairs (0, 1) and (2, 3), and the states of the two pairs (both are told to go short
# by a spike) with the target's relation quiet alone, and with the acquirer's quiet too
QUIET = (
    (0, 2, [0, -1], [0, 0]),  # the target is in the first pair, the acquirer in the second
    (2, 0, [-1, 0], [0, 0]),
    (1, 3, [0, -1], [0, 0]),
    (0, 1, [0, -1], [0, -1]),  # both books in one pair: it is quiet, and the other is not, whichever rule
    (2, 3, [-1, 0], [-1, 0]),
)


@pytest.mark.parametrize("acquirers", [False, True])
def test_a_relation_with_the_target_of_a_deal_is_flat_one_with_only_its_acquirer_is_so_only_if_acquirers(acquirers):
    for target, acquirer, by_default, with_acquirers in QUIET:
        expected = with_acquirers if acquirers else by_default
        trader = pair_trader(acquirers=acquirers)
        stage = Stage(trader)
        for _ in range(3):
            stage.decide(spikes(False))
        news = deal(target, acquirer)
        stage.start(news)
        stage.decide(spikes(True))
        assert trader.states.tolist() == expected, (target, acquirer)
        later = pair_trader(acquirers=acquirers)  # and the books of a deal that ended less than the quarantine ago
        stage = Stage(later)
        news = deal(target, acquirer)
        stage.start(news)
        for _ in range(3):
            stage.decide(spikes(False))
        stage.end(news, at=3.5)
        stage.decide(spikes(True))  # at 4, half a unit after it ended
        assert later.states.tolist() == expected, (target, acquirer)
    plain = pair_trader(RelationTrader)  # a pair with neither book of the deal, as a relation trader's does
    stage = Stage(plain)
    for _ in range(3):
        stage.decide(spikes(False))
    stage.decide(spikes(True))
    assert plain.states.tolist() == [-1, -1]


def test_a_pair_that_holds_only_the_acquirer_of_a_deal_trades_as_a_relation_trader_would_by_default():
    spiked = {4, 8, 12}
    deal_on = dict(on=(5, 9), ended_at=9.5, between=(0, 2))  # target in the first pair, acquirer in the second
    plain = run_stage(pair_trader(RelationTrader), spiked)
    target_only = run_stage(pair_trader(), spiked, **deal_on)
    both = run_stage(pair_trader(acquirers=True), spiked, **deal_on)
    assert plain[8] == ([-1, -1], [-2, 2, -2, 2])
    # the quarantine, 4.12 units, ends between the decisions at 13 and 14
    for t in range(5, 14):
        assert target_only[t][0][0] == 0 and target_only[t][1][:2] == [0, 0]  # the target's pair is flat
        assert both[t][0] == [0, 0] and both[t][1] == [0, 0, 0, 0]  # and so is the acquirer's with acquirers=True
    # the acquirer's pair does what the plain trader's does, at every decision, the lots it holds as much as its state
    assert all(target_only[t][0][1] == plain[t][0][1] and target_only[t][1][2:] == plain[t][1][2:] for t in plain)
    assert target_only[8][0] == [0, -1] and target_only[8][1] == [0, 0, -2, 2]  # short at 8 in the acquirer's pair
    assert both[8] == ([0, 0], [0, 0, 0, 0])
    # when the quarantine is over both trade as the plain trader does
    assert target_only[14] == both[14] == plain[14]
    assert both[4] == plain[4] == target_only[4]  # and before the deal, too


def test_a_pair_that_holds_the_target_of_a_deal_is_flat_for_either_value_of_acquirers_and_for_its_whole_life():
    spiked = {4, 8, 12}
    deal_on = dict(on=(5, 9), ended_at=9.5, between=(1, 2))  # target in the first pair, acquirer in the second
    for acquirers in (False, True):
        news = run_stage(pair_trader(acquirers=acquirers), spiked, **deal_on)
        assert all(news[t][0][0] == 0 and news[t][1][:2] == [0, 0] for t in range(5, 14))
        assert news[4][0] == [-1, -1] and news[4][1][:2] == [-2, 2]  # short before the deal is announced
    # the other pair is the only difference between the two values: it trades at 8 only by default
    by_default = run_stage(pair_trader(), spiked, **deal_on)
    with_acquirers = run_stage(pair_trader(acquirers=True), spiked, **deal_on)
    assert by_default[8][0] == [0, -1] and with_acquirers[8][0] == [0, 0]


def test_a_news_aware_trader_does_number_for_number_what_a_relation_trader_does_in_a_market_with_no_deals():
    settings = (
        dict(weights=PAIRS, window=20, entry=0.5, size=3, interval=1.0),
        dict(weights=PAIRS, window=20, entry=0.5, size=4, interval=1.0, rate=0.4, band=1.0),
        dict(choose=partial(pairs_by_distance, count=2), formation=120, window=40, entry=1.0, size=3, interval=1.0,
             refit=60),
    )
    for fair_settings in ({}, {"deals": Deals(rate=0.0)}):  # no deals at all, and deals that never come
        for kwargs in settings:
            results = []
            for cls in (RelationTrader, NewsAwareTrader):
                trader = cls(record=True, **kwargs)
                market = BookMarket(FactorValues(FOUR, **fair_settings), seed=3, traders=[trader])
                market.run(500.0)
                results.append((trader, market))
            (plain, plain_market), (aware, aware_market) = results
            assert plain.lots > 20 and plain.entries > 3
            assert (aware.position == plain.position).all() and aware.cash == plain.cash and aware.lots == plain.lots
            assert aware.spread_paid == plain.spread_paid and aware.entries == plain.entries
            assert aware.netted_lots == plain.netted_lots and (aware.book_cash == plain.book_cash).all()
            assert (aware.states == plain.states).all() and np.array_equal(aware.weights, plain.weights)
            assert aware.fits == plain.fits and aware_market.events == plain_market.events
            assert len(aware.history) == len(plain.history)
            assert all(a[0] == p[0] and (a[1] == p[1]).all() and (a[2] == p[2]).all()
                       for a, p in zip(aware.history, plain.history, strict=True))


@pytest.mark.parametrize("within", [True, False])
@pytest.mark.parametrize("acquirers", [False, True])
def test_beside_nothing_else_in_a_deals_market_it_holds_no_lots_of_a_book_from_a_deal_to_the_quarantines_end(
        acquirers, within):
    sectors = [[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]]  # the pairs of SIX's three sectors

    def run(cls):
        trader = cls(weights=sectors, window=20, entry=0.5, size=3, interval=1.0,
                     **({"acquirers": acquirers} if cls is NewsAwareTrader else {}))
        market = in_a_market(trader, table=SIX, seed=1,
                             deals={"rate": 0.05, "length": 30.0, "cool": 5.0, "within": within})
        quiet, checked, quarantined, held_acquirers = [], 0, 0, 0
        for now in decisions(market, trader, 900.0):
            near = list(market.fair.deals) + [d for d in market.fair.ended if now - d.ended_at < trader.taking(20)]
            targets = {d.target for d in near}
            books = targets | {d.acquirer for d in near} if acquirers else targets
            quiet.append([b for b in books if trader.position[b]])
            checked += 1
            quarantined += len(books)
            held_acquirers += any(trader.position[d.acquirer] for d in near if d.acquirer not in targets)
        return trader, market, quiet, checked, quarantined, held_acquirers

    aware, market, quiet, checked, quarantined, held_acquirers = run(NewsAwareTrader)
    assert aware.quarantine == aware.taking(20) and len(market.fair.ended) > 20 and checked > 800
    assert quarantined > 600 and aware.entries > 40 and aware.lots > 200  # a market it traded in, with deals about
    assert all(not held for held in quiet)  # no lots of a book in a deal, or within its window's time of one's end
    # the acquirer's book is held when its relation is not the target's (a pair is quiet for a target in it), unless
    # it is kept out of as well: in a sector the acquirer is the target's own pair's, and then it is flat anyway
    assert (held_acquirers > 20) == (not acquirers and not within)
    plain, _, quiet_plain, _, _, _ = run(RelationTrader)  # the same market, a trader that does not read the news
    assert sum(bool(held) for held in quiet_plain) > 100
