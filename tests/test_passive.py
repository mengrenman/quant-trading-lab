"""Resting orders for the traders of ``BookMarket`` (lob/passive.py).

The rules are tested on books made by hand: two or more books of a scripted model that sends only the orders it is given
(``Scripted``), so that every price, size and fill is known. A book starts as 2997 to 2999 bids and 3001 to 3003 asks,
10 lots at each (``hand_market``), the mid 3000. A decision is made at a time the test chooses (``decide``) and a
model's order is sent through ``market.submit``, as ``step`` sends it, so that the time is the test's. The runs of the
real order flow are at the end: the accounts of traders that work orders for a few hundred units add up.
"""

import bisect
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

import lob.crosssection as cross_now
import lob.statarb as statarb_now
from lob.crosssection import FlowTrader, ReversalTrader
from lob.deal_traders import DealTrader
from lob.deals import Deals
from lob.order import OrderType, Side, cancel_order, limit_order, market_order
from lob.pairs import AnchoredZIModel
from lob.passive import (Account, Execution, Lot, PassiveBookMarket, ShortfallParts, Since, score_execution,
                         shortfall_parts)
from lob.statarb import (BookMarket, BookTrader, FactorValues, RelationTrader, StateTrader, score_holdings,
                         sector_loadings)
from tests.test_stage4_identity import KINDS, SECTORS

BID, ASK = Side.BID, Side.ASK
BUSY = dict(rate=0.15, premium=60.0, length=30.0, breaks=0.3, priced=0.4, cool=10.0)  # a deal about every 7 units


def adds_up(market, trader):
    """What a trader's account must satisfy whatever it did: its position is the signed lots of its fills, its cash
    minus their lots times their prices, its books' cash add up to its cash, and its lots are passive and crossed."""
    account = market.account(trader)
    position = np.zeros(market.n, dtype=int)
    cash = np.zeros(market.n)
    for _, book, lots, price, _, _ in account.fills:
        position[book] += lots
        cash[book] -= lots * price
    assert (position == trader.position).all()
    assert np.allclose(cash, trader.book_cash) and abs(cash.sum() - trader.cash) < 1e-6
    assert abs(trader.book_cash.sum() - trader.cash) < 1e-6
    assert trader.lots == account.passive_lots + account.crossed_lots
    assert sum(abs(fill[2]) for fill in account.fills) == trader.lots


# ---- Books made by hand ----

class Scripted:
    """An order-flow model that sends the orders it is given, one an event, in its book, and nothing else. Its rate is
    high, so that a step is a few thousandths of a unit of time (a test that steps sets the time first)."""

    def __init__(self, rate=1000.0):
        self.orders, self._rate = [], rate

    def rate(self, book):
        return self._rate

    def event(self, book, t, fair, rng):
        return self.orders.pop(0) if self.orders else None


def hand_market(books=2, depth=3, lots=10, **kwargs):
    """``books`` books of scripted models, each with ``depth`` prices a side (bids 2999 down, asks 3001 up) of
    ``lots``."""
    models = [Scripted() for _ in range(books)]
    fair = FactorValues(np.ones((books, 1)))
    market = PassiveBookMarket(fair, models=models, seed=1, depth=depth, lots=lots, **kwargs)
    return market, models


@dataclass
class Wants(BookTrader):
    """Wants these lots in each book, and asks for the gap to them at every decision. It keeps the mids it was given and
    what ``observe`` showed it (``seen``: a function of the market it is given, if ``look`` is)."""
    goal: Optional[np.ndarray] = None
    look: Optional[object] = None
    mids: list = field(default_factory=list)
    seen: list = field(default_factory=list)

    def observe(self, market):
        if self.look is not None:
            self.seen.append(self.look(market))

    def decide(self, mids):
        self.mids.append(mids.copy())
        gap = np.asarray(self.goal) - self.position
        return [gap] if gap.any() else []


def decide(market, trader, at):
    """The trader's decision at a time of the test's choosing."""
    market.time = at
    market.trade_for(trader)


def work_of(market, trader):
    return market._works[id(trader)]


def working(market, trader, book):
    """The trader's working order in a book: ``(price, size, id)``, or None."""
    order_id = work_of(market, trader).order.get(book)
    if order_id is None:
        return None
    order = market.books[book].get_order(order_id)
    return order.price, order.size, order_id


def clock_of(market, trader, book):
    return work_of(market, trader).clock.get(book)


def rest(market, book, side, price, lots):
    """Another order resting in a book (put there directly: no event, no draw). Its id."""
    order = limit_order(price, lots, 0.0, side)
    market.books[book].add_order(order)
    return order.order_id


def clear(market, book, side, price, keep=()):
    """Every order at a price but those of ``keep`` (ids), taken away."""
    for order in market.books[book].orders_at(price, side):
        if order.order_id not in keep:
            market.books[book].cancel_order(order.order_id)


def queue(market, book, side, price):
    """The ids of the orders at a price in priority order."""
    return [order.order_id for order in market.books[book].orders_at(price, side)]


def hit(market, book, side, lots, kind=OrderType.MARKET, price=0, impact=True):
    """A model's order, sent as ``step`` sends it, that sells (``side`` ASK) or buys (BID) these lots."""
    if kind == OrderType.MARKET:
        order = market_order(lots, market.time, side)
    else:
        order = limit_order(price, lots, market.time, side)
    return market.submit(book, order, impact=impact)


def worked(goal, patience=10.0, reprice=None, books=2, depth=3, lots=10, interval=5.0, urgent_closes=True, **kwargs):
    """A hand-made market with a trader that wants ``goal`` and works its needs. The market, the models, the trader."""
    market, models = hand_market(books, depth, lots)
    trader = Wants(goal=np.array(goal), interval=interval, **kwargs)
    market.add(trader, Execution(patience, reprice, urgent_closes))
    return market, models, trader


# ---- The settings ----

def test_an_execution_is_a_patience_of_20_units_a_reprice_every_unit_and_urgent_closes_unless_told():
    execution = Execution()
    assert (execution.patience, execution.reprice, execution.urgent_closes) == (20.0, 1.0, True)
    assert [f.name for f in dataclasses.fields(Execution)] == ["patience", "reprice", "urgent_closes"]
    assert Execution(0).patience == 0 and Execution(patience=4.5, reprice=0.5, urgent_closes=False).reprice == 0.5
    assert Execution(reprice=None).reprice is None  # at the trader's decisions alone


def test_an_execution_is_not_changed_once_made_so_that_one_may_be_given_to_many_traders():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Execution(reprice=None).patience = 5.0


@pytest.mark.parametrize("patience", [-1.0, -1e-9, math.inf, -math.inf, math.nan])
def test_a_patience_that_is_negative_or_not_finite_is_refused(patience):
    with pytest.raises(ValueError, match="patience"):
        Execution(patience=patience, reprice=None)


@pytest.mark.parametrize("reprice", [0.0, -1.0, math.inf, math.nan])
def test_a_reprice_that_is_not_none_or_a_finite_time_above_0_is_refused(reprice):
    with pytest.raises(ValueError, match="reprice"):
        Execution(reprice=reprice)


# ---- Adding a trader, and its account ----

def test_every_trader_has_an_account_with_or_without_an_execution_and_one_not_in_the_market_has_none():
    market, _ = hand_market()
    bare, worker = Wants(goal=np.zeros(2, dtype=int)), Wants(goal=np.zeros(2, dtype=int))
    market.add(bare)
    market.add(worker, Execution(reprice=None))
    built = PassiveBookMarket(FactorValues(np.ones((2, 1))), traders=[Wants(goal=np.zeros(2, dtype=int))])
    for made, trader in ((market, bare), (market, worker), (built, built.traders[0])):
        account = made.account(trader)
        assert isinstance(account, Account) and made.account(trader) is account
        assert account.books == 2 and not account.asked.any() and account.asked.dtype == int
        assert (account.passive_lots, account.crossed_lots, account.urgent_lots, account.orders_placed) == (0, 0, 0, 0)
        assert (account.orders_withdrawn, account.orders_crossed, account.asked_lots) == (0, 0, 0)
        assert account.spread_earned == 0.0 and account.paper_cash == 0.0
        assert account.fills == [] and account.waits == [] and account.decisions == []
    assert market.account(bare) is not market.account(worker) and market.account(bare) != market.account(worker)
    with pytest.raises(ValueError, match="not in this market"):
        market.account(Wants(goal=np.zeros(2, dtype=int)))


def test_traders_given_at_construction_have_no_execution_and_are_added_as_a_book_market_adds_them():
    first, second = Wants(goal=np.zeros(2, dtype=int)), Wants(goal=np.zeros(2, dtype=int))
    market = PassiveBookMarket(FactorValues(np.ones((2, 1))), seed=1, traders=[first, second])
    assert market.traders == [first, second] and not market._works
    assert first.position.shape == (2,) and not first.book_cash.any()  # attached, as add does


def test_a_trader_that_does_not_net_its_orders_is_refused_an_execution_in_a_sentence_and_is_not_added():
    weights = np.array([[1.0, -1.0]])
    for patience in (20.0, 0.0):
        market, _ = hand_market()
        trader = RelationTrader(weights=weights, window=3, netted=False)
        with pytest.raises(ValueError, match="does not net its orders.*working order is one in a book"):
            market.add(trader, Execution(patience, reprice=None))
        assert market.traders == [] and not hasattr(trader, "book_cash")  # refused before it was attached
    market.add(trader)  # with none it is added: it sends its own orders, as it always did
    assert market.traders == [trader] and market.account(trader).orders_placed == 0


def test_an_execution_that_is_not_an_execution_is_refused():
    market, _ = hand_market()
    with pytest.raises(ValueError, match="an Execution"):
        market.add(Wants(goal=np.zeros(2, dtype=int)), 20.0)
    assert market.traders == []


def test_a_patience_above_0_however_short_is_an_execution_and_only_0_is_none():
    market, _ = hand_market()
    short, one, none = (Wants(goal=np.zeros(2, dtype=int)) for _ in range(3))
    market.add(short, Execution(patience=0.5, reprice=None))
    market.add(one, Execution(patience=1.0, reprice=None))
    market.add(none, Execution(patience=0.0, reprice=None))
    assert id(short) in market._works and id(one) in market._works and id(none) not in market._works


def test_a_trader_added_with_a_patience_of_0_is_kept_as_one_with_no_execution_and_sends_market_orders():
    market, _ = hand_market()
    trader = Wants(goal=np.array([3, -2]), interval=5.0)
    market.add(trader, Execution(patience=0.0, reprice=None))
    assert not market._works
    before = market.events
    decide(market, trader, 1.0)
    account = market.account(trader)
    assert (trader.position == [3, -2]).all() and market.events == before + 2  # two market orders, nothing placed
    assert (account.orders_placed, account.orders_withdrawn, account.orders_crossed) == (0, 0, 0)
    assert (account.crossed_lots, account.passive_lots) == (5, 0) and not any(market._working)
    assert [fill[5] for fill in account.fills] == ["cross", "cross"]


# ---- The books without a trader's own orders ----

def test_with_nothing_resting_the_best_prices_and_the_mid_are_the_books_own():
    market, _, trader = worked([3, -2])
    assert market.best_for(trader, 0, BID) == 2999 and market.best_for(trader, 0, ASK) == 3001
    assert market.mid_for(trader, 0) == 3000.0 == market.books[0].mid_price
    stranger = Wants(goal=np.zeros(2, dtype=int))  # not in the market: nothing of it rests
    assert market.best_for(stranger, 1, ASK) == 3001 and market.mid_for(stranger, 1) == 3000.0


def test_an_order_of_the_trader_alone_at_the_best_price_is_left_out_and_the_next_price_is_the_best():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    assert working(market, trader, 0)[:2] == (2999, 3)
    clear(market, 0, BID, 2999, keep=[working(market, trader, 0)[2]])  # the others at 2999 go: it is alone at the best
    assert market.books[0].best_bid == 2999 and market.books[0].depth_at_price(2999, BID) == 3
    assert market.best_for(trader, 0, BID) == 2998 and market.best_for(trader, 0, ASK) == 3001
    assert market.mid_for(trader, 0) == 2999.5 and market.books[0].mid_price == 3000.0
    assert market.mid_for(trader, 1) == 3000.0  # nothing rests in book 1


def test_an_order_of_the_trader_with_others_at_its_price_leaves_the_best_price_and_the_mid_where_they_are():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    assert market.books[0].depth_at_price(2999, BID) == 13  # 10 and its 3
    assert market.best_for(trader, 0, BID) == 2999 and market.mid_for(trader, 0) == 3000.0


def test_an_order_of_the_trader_that_is_not_at_the_best_price_changes_nothing():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    rest(market, 0, BID, 3000, 1)  # the best bid is now above it
    assert working(market, trader, 0)[0] == 2999
    assert market.best_for(trader, 0, BID) == 3000 and market.mid_for(trader, 0) == 3000.5 == market.books[0].mid_price


def test_the_best_price_is_none_when_the_traders_own_order_is_all_there_is_on_that_side():
    market, _, trader = worked([0, -2])
    decide(market, trader, 1.0)
    own = working(market, trader, 1)[2]
    for price in (3001, 3002, 3003):
        clear(market, 1, ASK, price, keep=[own])
    assert market.books[1].best_ask == 3001 and market.best_for(trader, 1, ASK) is None
    assert market.mid_for(trader, 1) is None and market.books[1].mid_price == 3000.0
    assert market.best_for(trader, 1, BID) == 2999


def test_the_best_bid_and_the_mid_are_none_when_the_traders_own_bid_is_all_there_is_on_the_bid_side():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    own = working(market, trader, 0)[2]
    for price in (2999, 2998, 2997):
        clear(market, 0, BID, price, keep=[own])  # its bid is the only one: the others are gone
    assert market.books[0].best_bid == 2999 and market.best_for(trader, 0, BID) is None
    assert market.mid_for(trader, 0) is None and market.books[0].mid_price == 3000.0
    assert market.best_for(trader, 0, ASK) == 3001


def test_the_mid_is_none_if_either_side_has_none_and_the_books_own_mid_when_nothing_rests():
    market, _, trader = worked([3, 0])
    for price in (3001, 3002, 3003):
        clear(market, 1, ASK, price)
    assert market.books[1].mid_price is None and market.mid_for(trader, 1) is None  # nothing rests: the book's own
    assert market.mid_for(trader, 0) == 3000.0


def test_another_traders_orders_are_not_left_out_but_the_traders_own_are():
    market, _, first = worked([3, 0])
    second = Wants(goal=np.array([2, 0]), interval=5.0)
    market.add(second, Execution(10.0, reprice=None))
    decide(market, first, 1.0)
    decide(market, second, 1.0)
    one, two = working(market, first, 0)[2], working(market, second, 0)[2]
    clear(market, 0, BID, 2999, keep=[one, two])  # two orders at 2999 and no others
    assert market.books[0].depth_at_price(2999, BID) == 5
    assert market.best_for(first, 0, BID) == 2999 and market.best_for(second, 0, BID) == 2999  # each sees the other's
    clear(market, 0, BID, 2999, keep=[one])  # now the second's goes: the first is alone
    assert market.best_for(first, 0, BID) == 2998 and market.best_for(second, 0, BID) == 2999


def test_the_marks_of_a_trader_leave_out_its_own_orders_and_the_markets_own_marks_do_not():
    market, _, trader = worked([3, -2])
    decide(market, trader, 1.0)
    for book, side, price in ((0, BID, 2999), (1, ASK, 3001)):  # each of its orders alone at the best price
        clear(market, book, side, price, keep=[working(market, trader, book)[2]])
    assert list(market.marks()) == [3000.0, 3000.0]  # the books' own: the orders are in them
    assert list(market.marks_for(trader)) == [(2998 + 3001) / 2.0, (2999 + 3002) / 2.0]


def test_the_marks_of_a_trader_leave_out_its_own_order_and_not_another_traders():
    market, _, first = worked([3, 0])
    second = Wants(goal=np.array([2, 0]), interval=5.0)
    market.add(second, Execution(10.0, reprice=None))
    decide(market, first, 1.0)
    decide(market, second, 1.0)
    one, two = working(market, first, 0)[2], working(market, second, 0)[2]
    clear(market, 0, BID, 2999, keep=[one, two])  # two orders at 2999 and no others: each is the other's best bid
    assert list(market.marks_for(first)) == [3000.0, 3000.0] == list(market.marks_for(second))
    clear(market, 0, BID, 2999, keep=[one])  # the second's goes: the first is alone at 2999, and its mark leaves it out
    assert market.marks_for(first)[0] == (2998 + 3001) / 2.0 and market.marks_for(second)[0] == 3000.0
    assert market.marks()[0] == 3000.0 and market.marks_for(Wants(goal=np.zeros(2, dtype=int)))[0] == 3000.0


def test_the_marks_of_a_trader_are_the_markets_last_mid_where_its_mids_are_none():
    market, _, trader = worked([0, -2])
    decide(market, trader, 1.0)
    own = working(market, trader, 1)[2]
    for price in (3001, 3002, 3003):
        clear(market, 1, ASK, price, keep=[own])
    assert market.mid_for(trader, 1) is None and market.books[1].mid_price == 3000.0
    marks = market.marks_for(trader)
    assert marks[1] == market.mark(1) == 3000.0 and marks[0] == 3000.0  # the book's own mark where there is no mid
    for price in (2999, 2998, 2997):
        clear(market, 0, BID, price)  # book 0 has no bid, no mid, and its last mid is the mark
    assert market.books[0].mid_price is None and market.marks_for(trader)[0] == 3000.0


# ---- A decision: the mids, what it observes, what it needs, the paper account ----

def play(market, models, script):
    """Send each book its orders, in turn, through ``step``, until every one has been through."""
    for book, orders in script.items():
        models[book].orders.extend(orders)
    while any(model.orders for model in models):
        market.step()


@dataclass
class Splits(BookTrader):
    """Gives each of ``parts`` as a move, at every decision: its need is their sum."""
    parts: list = field(default_factory=list)

    def decide(self, mids):
        return [np.array(part) for part in self.parts]


@dataclass
class Switch(StateTrader):
    """Wants ``goal`` while it is not ``closing``, and then has no relations, which a ``StateTrader`` closes at once."""
    goal: Optional[np.ndarray] = None
    closing: bool = False

    def decide(self, mids):
        if self.closing:
            self.weights = None
            return self._become(np.zeros(0, dtype=int))
        self.weights = np.eye(len(mids))[:1]
        gap = np.asarray(self.goal) - self.position
        return [gap] if gap.any() else []


def test_a_decision_when_a_book_has_no_mid_does_nothing_and_leaves_the_orders_and_the_paper_account_as_they_are():
    market, _, trader = worked([3, 2], record=True)
    decide(market, trader, 1.0)
    placed = (working(market, trader, 0), working(market, trader, 1))
    events, asked = market.events, market.account(trader).asked.copy()
    for price in (3001, 3002, 3003):
        clear(market, 1, ASK, price)  # book 1 has no ask
    decide(market, trader, 2.0)
    assert (working(market, trader, 0), working(market, trader, 1)) == placed and market.events == events
    assert trader.next_time == 2.0 + trader.interval  # as the base sets it, before it looks
    assert len(trader.mids) == 1 and len(trader.history) == 1  # it did not decide, and left no row
    assert (market.account(trader).asked == asked).all() and market.account(trader).asked_lots == 5


def test_a_decision_is_not_made_when_the_mid_without_the_traders_own_order_is_none_though_the_books_is_not():
    market, _, trader = worked([0, -2])
    decide(market, trader, 1.0)
    own = working(market, trader, 1)
    for price in (3001, 3002, 3003):
        clear(market, 1, ASK, price, keep=[own[2]])
    assert market.books[1].mid_price == 3000.0  # the book has a mid: its trader's own ask is the only ask
    decide(market, trader, 2.0)
    assert len(trader.mids) == 1 and working(market, trader, 1) == own


def test_the_mids_decide_is_given_leave_out_the_traders_own_order_and_no_ones_else():
    market, _, first = worked([3, 0], reprice=100.0)  # a reprice that is not due: an order alone at the best stays
    second = Wants(goal=np.array([0, 2]), interval=5.0)
    market.add(second, Execution(10.0, reprice=100.0))
    decide(market, first, 1.0)
    decide(market, second, 1.0)
    clear(market, 0, BID, 2999, keep=[working(market, first, 0)[2]])  # each is alone at the best price in its book
    clear(market, 1, BID, 2999, keep=[working(market, second, 1)[2]])
    assert [b.mid_price for b in market.books] == [3000.0, 3000.0]
    decide(market, first, 2.0)
    decide(market, second, 2.0)
    assert list(first.mids[-1]) == [2999.5, 3000.0]  # its own bid is out of book 0, the other's is in book 1
    assert list(second.mids[-1]) == [3000.0, 2999.5]


def snapshot(market):
    """What a trader's ``observe`` reads of a market: the books' best prices, mid, spread and queues, and the rest."""
    return {"books": [(b.best_bid, b.best_ask, b.mid_price, b.spread, b.depth_at_price(2999, BID),
                       b.depth_at_price(2998, BID), b.depth_at_price(3001, ASK), b.num_orders(), b.total_depth(BID),
                       b.depth_profile(BID, 3)) for b in market.books],
            "market": (market.time, market.n, market.flow, market.volume, market.fair, market.models)}


def test_the_view_a_trader_observes_is_the_market_with_books_that_leave_out_its_own_order_and_no_ones_else():
    market, _, trader = worked([3, 0], look=snapshot, reprice=100.0)
    other = Wants(goal=np.array([0, 2]), interval=5.0)
    market.add(other, Execution(10.0, reprice=100.0))
    decide(market, trader, 1.0)
    decide(market, other, 1.0)
    clear(market, 0, BID, 2999, keep=[working(market, trader, 0)[2]])  # its bid alone at 2999
    clear(market, 1, BID, 2999, keep=[working(market, other, 1)[2]])  # the other's alone at 2999 in book 1
    decide(market, trader, 2.0)
    books, rest_of = trader.seen[-1]["books"], trader.seen[-1]["market"]
    # book 0: no bid at 2999 (its own, alone), so 2998 is the best, the mid 2999.5, the spread 3; the book's own methods
    # that the view does not answer (num_orders, total_depth, depth_profile) are the book's, with its order in
    assert books[0] == (2998, 3001, 2999.5, 3, 0, 10, 10, 6, 23, [(2999, 3), (2998, 10), (2997, 10)])
    # book 1: the other trader's bid is there, alone at 2999 with 2 lots, so it is in the view
    assert books[1] == (2999, 3001, 3000.0, 2, 2, 10, 10, 6, 22, [(2999, 2), (2998, 10), (2997, 10)])
    time, n, flow, volume, fair, models = rest_of  # everything else is the market's, as it is
    assert (time, n) == (2.0, 2) and flow is market.flow and volume is market.volume and fair is market.fair
    assert models is market.models


def test_a_view_of_a_book_with_a_side_empty_has_no_spread_and_no_mid_and_a_best_price_on_the_other_side_only():
    market, _, trader = worked([3, 0])
    book = work_of(market, trader).view.books[1]
    assert (book.best_bid, book.best_ask, book.spread, book.mid_price) == (2999, 3001, 2, 3000.0)
    for price in (3001, 3002, 3003):
        clear(market, 1, ASK, price)
    assert (book.best_bid, book.best_ask, book.spread, book.mid_price) == (2999, None, None, None)
    for price in (2999, 2998, 2997):
        clear(market, 1, BID, price)
    assert (book.best_bid, book.best_ask, book.spread, book.mid_price) == (None, None, None, None)
    assert book.depth_at_price(2999, BID) == 0 and book.num_orders() == 0  # the book's own for the rest


def test_a_view_of_a_book_with_no_bid_has_no_spread_and_no_mid_and_the_ask_it_has():
    market, _, trader = worked([3, 0])
    book = work_of(market, trader).view.books[1]
    for price in (2999, 2998, 2997):
        clear(market, 1, BID, price)
    assert (book.best_bid, book.best_ask, book.spread, book.mid_price) == (None, 3001, None, None)


def test_a_passive_traders_own_fills_are_in_the_flow_and_volume_it_reads():
    market, models, trader = worked([3, 0], look=lambda m: (m.flow.copy(), m.volume.copy()))
    decide(market, trader, 1.0)
    play(market, models, {0: [market_order(12, 0.0, ASK)]})  # sells 12: the 10 ahead of it and 2 of its 3
    assert trader.position[0] == 2 and market.account(trader).passive_lots == 2
    decide(market, trader, 2.0)
    flow, volume = trader.seen[-1]
    assert (flow == [-12, 0]).all() and (volume == [12, 0]).all()  # its own 2 lots are in them, as the models' orders


def test_a_queue_flow_trader_does_not_read_its_own_lots_at_the_best_quote():
    market, _ = hand_market(books=3)
    rest(market, 0, BID, 2999, 10)  # 20 lots at the best bid of book 0
    trader = FlowTrader(lookback=1, factors=0, measure="queues", hold=1, size=4, interval=1.0)
    market.add(trader, Execution(10.0, reprice=None))
    decide(market, trader, 1.0)
    assert working(market, trader, 0)[:2] == (2999, 4)  # a score of 1/3 in book 0: it joins the 20 with 4 of its own
    assert market.books[0].depth_at_price(2999, BID) == 24
    decide(market, trader, 2.0)
    bids, asks = trader._lots
    assert list(bids) == [20, 10, 10] and list(asks) == [10, 10, 10]  # not 24: its own lots are not read


def test_the_need_is_the_sum_of_the_moves_a_decision_gives_and_none_if_it_gives_none():
    market, _ = hand_market()
    trader = Splits(parts=[[2, 0], [1, -3]], interval=5.0)
    market.add(trader, Execution(10.0, reprice=None))
    decide(market, trader, 1.0)
    assert working(market, trader, 0)[:2] == (2999, 3) and working(market, trader, 1)[:2] == (3001, 3)
    assert (market.account(trader).asked == [3, -3]).all()
    trader.parts = []
    decide(market, trader, 2.0)  # it gives none: no need anywhere, and what it worked is withdrawn
    assert working(market, trader, 0) is None and working(market, trader, 1) is None and not any(market._working)
    assert (market.account(trader).asked == [0, 0]).all()


def reversal(rate=1.0, hold=3, size=2.0):
    """A cross-sectional trader of four books whose scores are the same at each rebalance."""
    market, models = hand_market(books=4)
    trader = ReversalTrader(lookback=1, factors=0, hold=hold, size=size, interval=1.0, rate=rate)
    trader.scores = lambda: np.array([1.0, -1.0, 0.5, -0.5])  # a target of size times these, to the lot
    market.add(trader, Execution(50.0, reprice=None))
    return market, models, trader


def test_a_cross_sectional_trader_with_a_hold_above_1_keeps_working_toward_its_last_target_between_rebalances():
    market, models, trader = reversal()
    decide(market, trader, 1.0)  # warming up: it has one set of mids and needs two
    assert not any(market._working) and trader._target is None
    decide(market, trader, 2.0)  # a rebalance: the target is size times the scores
    assert list(trader._target) == [2, -2, 1, -1]
    ids = [working(market, trader, book) for book in range(4)]
    assert [w[:2] for w in ids] == [(2999, 2), (3001, 2), (2999, 1), (3001, 1)]
    rest(market, 0, BID, 2999, 5)  # others behind its bid in book 0: it is not alone at the best price
    events = market.events
    decide(market, trader, 3.0)  # between rebalances ``decide`` gives nothing: the need is target less position
    assert [working(market, trader, book) for book in range(4)] == ids and market.events == events
    play(market, models, {0: [market_order(11, 0.0, ASK)]})  # the 10 ahead and 1 of its 2 lots
    assert list(trader.position) == [1, 0, 0, 0] and working(market, trader, 0)[:2] == (2999, 1)
    decide(market, trader, 4.0)  # still between rebalances: what is left of the target is the order it keeps
    assert working(market, trader, 0) == (2999, 1, ids[0][2]) and trader._since == 3
    assert market.account(trader).asked.tolist() == [2, -2, 1, -1]


def test_a_cross_sectional_trader_with_no_execution_asks_for_its_position_and_what_it_gives_between_rebalances():
    # the sheet's rule for a trader with no execution is its position and the sum of its moves: that it holds less than
    # its target (a thin book took half of an order) is the paper account's to remember at a rebalance, not between
    market, models = hand_market(books=4, depth=1, lots=1)
    trader = ReversalTrader(lookback=1, factors=0, hold=3, size=2.0, interval=1.0)
    trader.scores = lambda: np.array([1.0, -1.0, 0.5, -0.5])
    market.add(trader)  # no execution
    account = market.account(trader)
    decide(market, trader, 1.0)  # warming up
    decide(market, trader, 2.0)  # a rebalance to 2, -2, 1, -1: a lot at each touch, so it holds 1, -1, 1, -1
    assert list(trader._target) == [2, -2, 1, -1] and trader.position.tolist() == [1, -1, 1, -1]
    assert account.asked.tolist() == [2, -2, 1, -1]  # at a rebalance: its position and what it gave
    for book, side in ((0, ASK), (1, BID), (2, ASK), (3, BID)):  # the sides the orders took empty, so it decides again
        rest(market, book, side, 3001 if side == ASK else 2999, 5)
    decide(market, trader, 3.0)  # between rebalances it gives nothing: asked is its position, as the sheet words it
    assert trader.position.tolist() == [1, -1, 1, -1] and account.asked.tolist() == [1, -1, 1, -1]
    assert account.asked_lots == 6 + 2  # the 6 of the rebalance (2, 2, 1, 1), and 1 and 1 back in the first two books


def test_a_cost_aware_cross_sectional_trader_works_what_it_gives_at_each_decision_and_not_the_whole_target():
    market, _, trader = reversal(rate=0.5, size=4.0)
    decide(market, trader, 1.0)
    decide(market, trader, 2.0)  # a rebalance: the target 4, -4, 2, -2, and the aim half the way: 2, -2, 1, -1
    assert list(trader._target) == [4, -4, 2, -2]
    assert [working(market, trader, book)[1] for book in range(4)] == [2, 2, 1, 1]
    decide(market, trader, 3.0)  # not a rebalance, and the aim goes half of what is left of the way again: 3, -3
    assert [working(market, trader, book)[1] for book in range(4)] == [3, 3, 2, 2]  # 1.5 and 0.5 rounded to even
    # the paper asks as a trader holding 2, -2, 1, -1 (what it asked before) would: 0.5 more rounds to even, to none
    assert market.account(trader).asked.tolist() == [3, -3, 1, -1]


def test_the_paper_account_is_the_position_asked_for_at_each_decision_got_at_that_decisions_mids_with_no_cost():
    market, _, trader = worked([3, -2])
    account = market.account(trader)
    decide(market, trader, 1.0)  # mids 3000 and 3000
    assert account.asked.tolist() == [3, -2] and account.paper_cash == -(3 * 3000.0 - 2 * 3000.0)
    assert account.asked_lots == 5
    rest(market, 0, BID, 3000, 1)  # book 0's mid 3000.5
    trader.goal = np.array([1, -2])
    decide(market, trader, 2.0)  # asks 2 lots less in book 0: got at 3000.5, back
    assert account.asked.tolist() == [1, -2] and account.asked_lots == 7
    assert account.paper_cash == pytest.approx(-3000.0 + 2 * 3000.5)
    trader.goal = np.array([1, -5])  # three lots more of the ask in book 1: sold at its mid
    decide(market, trader, 3.0)
    assert account.paper_cash == pytest.approx(-3000.0 + 2 * 3000.5 + 3 * 3000.0) and account.asked_lots == 10
    assert (trader.position == 0).all() and trader.cash == 0.0  # nothing was filled: the paper account is its own
    assert not market.keep_decisions and account.decisions == []  # the decisions are not kept unless asked


def test_with_keep_decisions_on_each_account_keeps_the_time_the_mids_and_what_was_asked_at_each_decision():
    market, _, trader = worked([3, -2])
    market.keep_decisions = True
    bystander = Wants(goal=np.array([1, 0]), interval=5.0)
    market.add(bystander)  # an account of a trader with no execution keeps them too
    decide(market, trader, 1.0)
    rest(market, 0, BID, 3000, 1)
    trader.goal = np.array([1, -2])
    decide(market, trader, 2.0)
    decide(market, bystander, 3.0)
    (t1, mids1, asked1), (t2, mids2, asked2) = market.account(trader).decisions
    assert (t1, t2) == (1.0, 2.0) and list(mids1) == [3000.0, 3000.0] and list(mids2) == [3000.5, 3000.0]
    assert asked1.tolist() == [3, -2] and asked2.tolist() == [1, -2]
    assert asked1 is not market.account(trader).asked and mids1 is not trader.mids[0]  # copies
    (t3, mids3, asked3), = market.account(bystander).decisions
    assert t3 == 3.0 and list(mids3) == [3000.5, 3000.0] and asked3.tolist() == [1, 0]


def test_the_paper_account_of_a_trader_with_no_execution_takes_its_position_and_its_moves_by_the_same_rule():
    market, _ = hand_market()
    trader = Wants(goal=np.array([3, -2]), interval=5.0)
    market.add(trader)
    account = market.account(trader)
    decide(market, trader, 1.0)  # crossed at once: 3 lots at 3001 and 2 at 2999
    assert trader.position.tolist() == [3, -2] and account.asked.tolist() == [3, -2]
    assert account.paper_cash == -(3 * 3000.0 - 2 * 3000.0) and account.asked_lots == 5
    assert trader.cash == -(3 * 3001 - 2 * 2999)  # the cost of crossing is in its cash, and not in the paper account
    trader.goal = np.array([3, 0])
    decide(market, trader, 7.0)  # it holds 3 and -2, and moves book 1 by +2: got at the mid of book 1 then, 3000
    assert account.asked.tolist() == [3, 0] and account.asked_lots == 7 and trader.position.tolist() == [3, 0]
    assert account.paper_cash == -3000.0 - 2 * 3000.0


def test_a_trader_that_asks_nothing_has_asked_for_the_position_it_holds():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 13)  # all of its 3 lots, and the 10 ahead
    assert trader.position.tolist() == [3, 0]
    decide(market, trader, 2.0)  # its goal is its position: no move, no need
    assert market.account(trader).asked.tolist() == [3, 0] and not any(market._working)


def test_a_trader_that_records_has_a_row_a_decision_with_the_mids_it_decided_on_and_none_for_a_decision_not_made():
    market, _, trader = worked([3, 0], record=True)
    decide(market, trader, 1.0)
    time, mids, position, signal = trader.history[0]
    assert time == 1.0 and list(mids) == [3000.0, 3000.0] and position.tolist() == [0, 0] and signal is None
    rest(market, 0, BID, 3000, 1)
    decide(market, trader, 2.0)
    assert len(trader.history) == 2 and list(trader.history[1][1]) == [3000.5, 3000.0]
    assert trader.history[1][1] is not trader.mids[1] and trader.history[1][2] is not trader.position  # copies
    trader.position[0] = 99
    assert trader.history[1][2][0] != 99


def test_a_trader_with_no_execution_records_the_books_mids_and_a_row_a_decision_as_a_book_market_does():
    market, _ = hand_market()
    trader = Wants(goal=np.array([1, 0]), interval=5.0, record=True)
    market.add(trader)
    decide(market, trader, 1.0)
    (time, mids, position, signal), = trader.history
    assert time == 1.0 and list(mids) == [3000.0, 3000.0] and position.tolist() == [1, 0] and signal is None


# ---- A trader with no relations ----

def switch(goal, urgent_closes=True):
    market, models = hand_market()
    trader = Switch(goal=np.array(goal), interval=5.0)
    market.add(trader, Execution(10.0, urgent_closes=urgent_closes, reprice=None))
    return market, models, trader


def test_a_trader_with_no_relations_withdraws_its_orders_and_closes_what_it_holds_at_once_with_market_orders():
    market, models, trader = switch([3, -2])
    decide(market, trader, 1.0)
    play(market, models, {0: [market_order(11, 0.0, ASK)]})  # 1 of its 3 bought, 2 left resting alone at 2999
    assert trader.position.tolist() == [1, 0] and working(market, trader, 0)[:2] == (2999, 2)
    account = market.account(trader)
    trader.closing = True
    decide(market, trader, 2.0)
    assert trader.position.tolist() == [0, 0] and not any(market._working) and not work_of(market, trader).clock
    assert account.urgent_lots == 1 and account.crossed_lots == 1 and account.passive_lots == 1
    assert account.orders_withdrawn == 2 and account.orders_crossed == 0  # an urgent close is no run-out of patience
    time, book, lots, price, mid, kind = account.fills[-1]
    assert (time, book, lots, kind) == (2.0, 0, -1, "urgent") and price == 2998  # its own bid was withdrawn first
    assert mid == (2998 + 3001) / 2.0 and work_of(market, trader).left.tolist() == [0, 0]
    adds_up(market, trader)


def test_an_urgent_close_still_leaves_the_history_row_of_a_decision_with_its_mids_and_the_position_after():
    market, models, trader = switch([3, -2])
    trader.record = True
    decide(market, trader, 1.0)
    play(market, models, {0: [market_order(11, 0.0, ASK)]})  # 1 of its 3 bought, its other 2 resting alone at 2999
    trader.closing = True
    decide(market, trader, 2.0)  # step 5 goes on to step 7: a decision that closed at once is a row all the same
    assert trader.position.tolist() == [0, 0] and market.account(trader).urgent_lots == 1
    assert len(trader.history) == 2
    time, mids, position, signal = trader.history[1]
    assert time == 2.0 and list(mids) == [(2998 + 3001) / 2.0, 3000.0]  # the mids without its own bid, before the close
    assert position.tolist() == [0, 0] and position is not trader.position and signal is None  # a copy, after the close
    assert trader.history[0][2].tolist() == [0, 0] and list(trader.history[0][1]) == [3000.0, 3000.0]


def test_with_urgent_closes_off_a_trader_with_no_relations_works_its_closes_like_any_need():
    market, models, trader = switch([3, -2], urgent_closes=False)
    decide(market, trader, 1.0)
    play(market, models, {0: [market_order(11, 0.0, ASK)]})
    trader.closing = True
    decide(market, trader, 2.0)  # it needs to sell 1 in book 0: its bid is out, an ask is placed; book 1 needs none
    assert trader.position.tolist() == [1, 0] and market.account(trader).urgent_lots == 0
    assert working(market, trader, 0)[:2] == (3001, 1) and working(market, trader, 1) is None
    assert clock_of(market, trader, 0) == (2.0, ASK) and clock_of(market, trader, 1) is None


def test_a_deal_trader_with_no_deal_on_has_no_relations_and_closes_at_once():
    market = PassiveBookMarket(FactorValues(sector_loadings(2, 2), deals=Deals(rate=0.0)), seed=1)
    trader = DealTrader(entry=8.0, size=5)
    market.add(trader, Execution(20.0, reprice=None))
    trader.position[:2] = [3, -2]
    assert len(trader.weights) == 0 and not market.fair.deals
    market.trade_for(trader)
    account = market.account(trader)
    assert trader.position.tolist() == [0, 0, 0, 0] and account.urgent_lots == 5 and account.orders_placed == 0
    assert {fill[5] for fill in account.fills} == {"urgent"} and [fill[2] for fill in account.fills][-1] == 2


# ---- Working a need: the clock, the order, the patience ----

def test_a_need_with_no_clock_starts_one_now_on_its_side_and_its_order_joins_the_best_price_at_the_back():
    market, _, trader = worked([3, -2])
    account, fairs, events = market.account(trader), list(market.fair.fairs), market.events
    decide(market, trader, 7.0)
    assert working(market, trader, 0)[:2] == (2999, 3) and working(market, trader, 1)[:2] == (3001, 2)
    assert all(type(order_id) is int for order_id in work_of(market, trader).order.values())  # ids, not orders
    assert clock_of(market, trader, 0) == (7.0, BID) and clock_of(market, trader, 1) == (7.0, ASK)
    assert queue(market, 0, BID, 2999)[-1] == working(market, trader, 0)[2]  # behind the 10 lots there
    assert queue(market, 1, ASK, 3001)[-1] == working(market, trader, 1)[2]
    assert (market.events, account.orders_placed, account.orders_withdrawn) == (events + 2, 2, 0)
    assert list(market.fair.fairs) == fairs and not market.flow.any() and not market.volume.any()  # no impact, no flow
    assert trader.position.tolist() == [0, 0] and trader.cash == 0.0 and trader.lots == 0
    assert work_of(market, trader).left.tolist() == [3, -2]


def test_a_book_with_no_need_has_its_working_order_withdrawn_and_its_clock_cleared_and_the_others_are_kept():
    market, _, trader = worked([3, -2])
    decide(market, trader, 1.0)
    kept, events = working(market, trader, 1), market.events
    trader.goal = np.array([0, -2])
    decide(market, trader, 2.0)
    assert working(market, trader, 0) is None and clock_of(market, trader, 0) is None
    assert working(market, trader, 1) == kept and clock_of(market, trader, 1) == (1.0, ASK)
    assert market.events == events + 1 and market.account(trader).orders_withdrawn == 1
    assert market.books[0].num_orders(BID) == 3  # the 3 prices of the starting book: its order is out of the book


def test_a_change_of_side_withdraws_the_order_on_the_old_side_and_starts_the_clock_again_on_the_new_one():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    old, events = working(market, trader, 0)[2], market.events
    trader.goal = np.array([-2, 0])
    decide(market, trader, 4.0)
    assert market.books[0].get_order(old) is None and not any(
        order.order_id == old for order in market.books[0].orders_at(2999, BID))
    assert working(market, trader, 0)[:2] == (3001, 2) and clock_of(market, trader, 0) == (4.0, ASK)
    assert market.events == events + 2 and market.account(trader).orders_withdrawn == 1
    assert market.account(trader).orders_placed == 2


def test_the_clock_starts_again_on_a_change_of_side_and_not_on_a_move_of_the_order():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    first = working(market, trader, 0)[2]
    rest(market, 0, BID, 3000, 1)  # the best bid is now above its order: it is moved to 3000 at the next decision
    decide(market, trader, 2.0)
    moved = working(market, trader, 0)
    assert moved[:2] == (3000, 3) and moved[2] != first and market.books[0].get_order(first) is None
    assert clock_of(market, trader, 0) == (1.0, BID)  # the first decision's, though the order is a new one
    trader.goal = np.array([-1, 0])
    decide(market, trader, 3.0)
    assert clock_of(market, trader, 0) == (3.0, ASK)  # a change of side starts it again


@pytest.mark.parametrize("reprice", [None, 100.0])
def test_a_need_that_grows_or_shrinks_on_the_same_side_keeps_its_clock_and_its_order_is_placed_again(reprice):
    market, _, trader = worked([3, 0], reprice=reprice)
    decide(market, trader, 1.0)
    for goal, at in ((5, 2.0), (2, 3.0)):  # a size that is not the need's is put right at once, due or not
        old = working(market, trader, 0)[2]
        trader.goal = np.array([goal, 0])
        decide(market, trader, at)
        assert working(market, trader, 0)[:2] == (2999, goal) and working(market, trader, 0)[2] != old
        assert clock_of(market, trader, 0) == (1.0, BID) and market.books[0].get_order(old) is None
    assert market.account(trader).orders_placed == 3 and market.account(trader).orders_withdrawn == 2


def test_an_order_with_the_size_needed_at_the_best_price_is_kept_in_its_place_in_the_queue_and_costs_no_event():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    own, events = working(market, trader, 0), market.events
    behind = rest(market, 0, BID, 2999, 4)  # an order arrives behind it
    decide(market, trader, 2.0)
    assert working(market, trader, 0) == own and market.events == events
    assert queue(market, 0, BID, 2999)[-2:] == [own[2], behind]  # still ahead of the newcomer
    assert market.account(trader).orders_placed == 1 and market.account(trader).orders_withdrawn == 0


def test_an_order_of_the_size_needed_at_another_price_is_moved_to_the_best_price_without_its_own_behind_or_ahead():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    rest(market, 0, BID, 3000, 1)  # behind the best price: moved to it, to the back of the queue there
    decide(market, trader, 2.0)
    assert working(market, trader, 0)[:2] == (3000, 3)
    assert queue(market, 0, BID, 3000)[-1] == working(market, trader, 0)[2]
    other = queue(market, 0, BID, 3000)[0]
    clear(market, 0, BID, 3000, keep=[working(market, trader, 0)[2]])  # now it is alone at 3000, ahead of everyone
    assert other not in queue(market, 0, BID, 3000) and market.best_for(trader, 0, BID) == 2999
    decide(market, trader, 3.0)  # the best price without its own is 2999: it joins the back of it
    assert working(market, trader, 0)[:2] == (2999, 3)
    assert queue(market, 0, BID, 2999)[-1] == working(market, trader, 0)[2]
    assert market.account(trader).orders_withdrawn == 2 and clock_of(market, trader, 0) == (1.0, BID)


def test_with_a_reprice_a_stale_order_of_the_size_needed_is_left_until_the_clock_is_due_and_then_moved():
    market, _, trader = worked([3, 0], patience=50.0, reprice=10.0)
    decide(market, trader, 1.0)  # the clock was due at the first decision (nothing to move): it runs from 1
    own = working(market, trader, 0)
    rest(market, 0, BID, 3000, 1)
    for at in (2.0, 10.9):  # not due: it stays where it is, behind the best price
        decide(market, trader, at)
        assert working(market, trader, 0) == own
    decide(market, trader, 11.0)  # 10 units after the first: due, and the clock runs again from now
    assert working(market, trader, 0)[:2] == (3000, 3) and work_of(market, trader).next_look == 21.0
    assert market.account(trader).orders_withdrawn == 1


def test_with_a_reprice_not_due_an_order_of_another_size_is_still_put_right_and_a_need_with_no_order_is_placed():
    market, _, trader = worked([3, 0], reprice=10.0)
    decide(market, trader, 1.0)
    rest(market, 0, BID, 3000, 1)
    trader.goal = np.array([5, 2])
    decide(market, trader, 2.0)  # book 0: its size is wrong, so it is placed again, at the best price; book 1: new
    assert working(market, trader, 0)[:2] == (3000, 5) and working(market, trader, 1)[:2] == (2999, 2)
    assert work_of(market, trader).next_look == 11.0  # the clock that was not due did not run again


def test_after_its_patience_the_order_is_withdrawn_and_what_is_needed_is_crossed_and_the_need_is_over():
    market, _, trader = worked([3, -2], patience=10.0)
    decide(market, trader, 7.0)
    decide(market, trader, 16.9)  # 9.9 of its 10 units
    assert working(market, trader, 0) is not None and market.account(trader).orders_crossed == 0
    decide(market, trader, 17.0)
    account = market.account(trader)
    assert trader.position.tolist() == [3, -2] and not any(market._working)
    assert (account.orders_crossed, account.crossed_lots, account.passive_lots) == (2, 5, 0)
    assert account.orders_withdrawn == 2
    assert account.waits == [(17.0, 10.0), (17.0, 10.0)]
    assert account.fills == [(17.0, 0, 3, 3001, 3000.0, "cross"), (17.0, 1, -2, 2999, 3000.0, "cross")]
    assert clock_of(market, trader, 0) is None and clock_of(market, trader, 1) is None
    assert trader.spread_paid == 3 * 1.0 + 2 * 1.0 and trader.lots == 5
    adds_up(market, trader)


def test_a_clock_within_a_billionth_of_its_patience_has_run_and_one_a_millionth_short_has_not():
    market, _, trader = worked([3, 0], patience=10.0)
    decide(market, trader, 1.0)
    decide(market, trader, 11.0 - 1e-6)
    assert working(market, trader, 0) is not None and not trader.position.any()
    decide(market, trader, 11.0 - 5e-10)
    assert working(market, trader, 0) is None and trader.position.tolist() == [3, 0]


def test_the_least_a_clock_may_be_short_of_its_patience_or_of_its_reprice_is_a_billionth_and_not_two():
    market, _, trader = worked([3, 0], patience=10.0)
    decide(market, trader, 1.0)
    decide(market, trader, 11.0 - 1.5e-9)  # a billion and a half short: the patience has not run
    assert working(market, trader, 0) is not None and not trader.position.any()
    market, _, trader = worked([3, 0], patience=50.0, reprice=10.0)
    decide(market, trader, 1.0)  # the reprice clock runs from 1 and is due at 11
    own = working(market, trader, 0)
    rest(market, 0, BID, 3000, 1)  # its order is stale
    decide(market, trader, 11.0 - 1.5e-9)
    assert working(market, trader, 0) == own  # not due: a billion and a half short
    decide(market, trader, 11.0 - 5e-10)
    assert working(market, trader, 0)[:2] == (3000, 3)  # within a billionth: due, and moved


def test_the_working_order_is_withdrawn_before_the_market_order_so_the_mid_it_pays_against_has_none_of_its_own():
    market, _, trader = worked([2, 0], patience=10.0)
    decide(market, trader, 1.0)
    clear(market, 0, BID, 2999, keep=[working(market, trader, 0)[2]])  # its bid alone at the best: the mid is 3000
    assert market.books[0].mid_price == 3000.0
    decide(market, trader, 11.0)
    # its bid is out first: the book's mid is then (2998 + 3001) / 2, and a lot at 3001 costs 1.5 against it
    assert trader.position.tolist() == [2, 0] and trader.spread_paid == 2 * 1.5
    assert market.account(trader).fills == [(11.0, 0, 2, 3001, 2999.5, "cross")]


def test_a_crossing_that_does_not_fill_all_that_is_needed_leaves_the_clock_and_the_rest_is_crossed_at_the_next_look():
    market, _, trader = worked([6, 0], depth=1, lots=4, patience=10.0)
    decide(market, trader, 1.0)
    decide(market, trader, 11.0)  # the book has 4 lots to sell: 4 are bought, the ask side is empty, 2 are left
    account = market.account(trader)
    assert trader.position.tolist() == [4, 0] and clock_of(market, trader, 0) == (1.0, BID)
    assert work_of(market, trader).left.tolist() == [2, 0] and account.waits == [] and account.orders_crossed == 1
    assert working(market, trader, 0) is None
    decide(market, trader, 12.0)  # no ask in book 0: no mid, and no decision
    assert account.orders_crossed == 1 and len(trader.mids) == 2
    rest(market, 0, ASK, 3002, 5)
    decide(market, trader, 13.0)  # the patience has run: the rest is crossed, and the need is over
    assert trader.position.tolist() == [6, 0] and account.orders_crossed == 1 and account.waits == [(13.0, 12.0)]
    assert clock_of(market, trader, 0) is None and account.crossed_lots == 6


def test_a_passive_fill_that_completes_a_need_clears_its_clock_and_a_later_need_on_the_same_side_starts_a_new_one():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    market.time = 4.0
    hit(market, 0, ASK, 13)  # the 10 ahead of it and its 3
    account = market.account(trader)
    assert trader.position.tolist() == [3, 0] and working(market, trader, 0) is None
    assert clock_of(market, trader, 0) is None and account.waits == [(4.0, 3.0)]
    trader.goal = np.array([5, 0])
    decide(market, trader, 6.0)
    assert clock_of(market, trader, 0) == (6.0, BID) and working(market, trader, 0)[:2] == (2998, 2)


def test_what_is_left_of_a_need_falls_with_each_passive_fill_and_is_the_size_of_the_order_that_rests():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 12)  # 10 ahead, 2 of its 3
    assert work_of(market, trader).left.tolist() == [1, 0] and working(market, trader, 0)[:2] == (2999, 1)
    assert clock_of(market, trader, 0) == (1.0, BID) and market.account(trader).waits == []  # not over
    hit(market, 0, ASK, 1)
    assert work_of(market, trader).left.tolist() == [0, 0] and working(market, trader, 0) is None
    assert clock_of(market, trader, 0) is None and len(market.account(trader).waits) == 1
    adds_up(market, trader)


# ---- Between decisions ----

def test_between_decisions_a_look_when_the_reprice_clock_is_due_moves_a_stale_order_and_runs_the_clock_again():
    market, _, trader = worked([3, 0], patience=50.0, reprice=2.0, interval=100.0)
    decide(market, trader, 1.0)
    own = working(market, trader, 0)
    rest(market, 0, BID, 3000, 1)
    market.time = 1.5
    market.step()  # the clock runs from the first decision: due at 3
    assert working(market, trader, 0) == own
    market.time = 3.0
    market.step()
    moved = working(market, trader, 0)
    assert moved[:2] == (3000, 3) and moved[2] != own[2] and clock_of(market, trader, 0) == (1.0, BID)
    assert work_of(market, trader).next_look == market.time + 2.0  # run again from the look
    clear(market, 0, BID, 3000, keep=[moved[2]])  # alone at the best price: stale again, 2999 is the best without it
    market.time += 1.0
    market.step()  # the clock runs again from the look: 1 of its 2 units
    assert working(market, trader, 0) == moved
    market.time += 2.0
    market.step()
    assert working(market, trader, 0)[:2] == (2999, 3)


def test_a_step_that_falls_exactly_on_the_time_the_reprice_clock_is_due_looks_and_one_before_it_does_not():
    market, _, trader = worked([3, 0], patience=50.0, reprice=2.0, interval=100.0)
    decide(market, trader, 1.0)  # the reprice clock runs from 1: due at 3
    own = working(market, trader, 0)
    rest(market, 0, BID, 3000, 1)  # its order is stale
    step_to(market, 2.75)  # steps of a quarter: 1.25, 1.5, ... 2.75
    assert working(market, trader, 0) == own
    step_to(market, 3.0)  # the step that falls on 3.0 is a look
    assert working(market, trader, 0)[:2] == (3000, 3) and work_of(market, trader).next_look == 5.0


def test_a_clock_exactly_a_billionth_short_of_its_patience_has_run_and_a_reprice_exactly_a_billionth_short_is_due():
    # clocks that began at 0.0, so that the sums are exact to the last digit: "at least" includes the boundary itself
    market, _, trader = worked([3, 0], patience=10.0)
    decide(market, trader, 0.0)
    decide(market, trader, 10.0 - 1e-9)
    assert working(market, trader, 0) is None and trader.position.tolist() == [3, 0]  # crossed
    market, _, trader = worked([3, 0], patience=50.0, reprice=10.0)
    decide(market, trader, 0.0)  # the reprice clock runs from 0: due at 10
    rest(market, 0, BID, 3000, 1)  # its order is stale
    decide(market, trader, 10.0 - 1e-9)
    assert working(market, trader, 0)[:2] == (3000, 3)  # due, and moved


def test_a_step_exactly_a_billionth_short_of_the_time_the_reprice_clock_is_due_looks():
    market, _, trader = worked([3, 0], patience=50.0, reprice=2.0, interval=100.0)
    decide(market, trader, 0.0)  # due at 2
    rest(market, 0, BID, 3000, 1)  # its order is stale
    market.rng = Beat(market.rng, 2.0 - 1e-9)  # one step of exactly that long
    market.step()
    assert market.time == 2.0 - 1e-9 and working(market, trader, 0)[:2] == (3000, 3)


def test_a_look_exactly_on_the_patience_or_a_billionth_short_of_it_crosses_what_is_left():
    market, _, trader = worked([3, 0], patience=5.0, reprice=1.0, interval=100.0)
    decide(market, trader, 1.0)  # the need began at 1.0: its patience runs out at 6.0, and the looks are at each unit
    step_to(market, 5.75)
    assert not trader.position.any() and market.account(trader).orders_crossed == 0
    step_to(market, 6.0)  # the look that falls on 6.0 crosses
    account = market.account(trader)
    assert trader.position.tolist() == [3, 0] and account.orders_crossed == 1 and account.fills[-1][0] == 6.0
    market, _, trader = worked([3, 0], patience=5.0, reprice=5.0 - 1e-9, interval=100.0)
    decide(market, trader, 0.0)  # and a look a billionth short of it, to the last digit (the sums are exact from 0.0)
    market.rng = Beat(market.rng, 5.0 - 1e-9)
    market.step()
    assert trader.position.tolist() == [3, 0] and market.account(trader).orders_crossed == 1


def test_a_look_between_decisions_leaves_an_order_that_is_at_the_best_price_where_it_is_and_costs_no_event():
    market, _, trader = worked([3, 0], patience=50.0, reprice=2.0, interval=100.0)
    decide(market, trader, 1.0)
    own, events = working(market, trader, 0), market.events
    behind = rest(market, 0, BID, 2999, 2)  # an order behind it: it keeps its place in the queue
    market.time = 3.5
    market.step()  # the clock is due, and the look finds it at the best price without its own
    assert working(market, trader, 0) == own and market.events == events
    assert queue(market, 0, BID, 2999)[-2:] == [own[2], behind]
    assert work_of(market, trader).next_look == market.time + 2.0 and market.account(trader).orders_placed == 1


def test_with_no_reprice_the_market_does_not_look_between_decisions_and_the_patience_is_met_at_a_decision():
    market, _, trader = worked([3, 0], patience=5.0, interval=100.0)
    decide(market, trader, 1.0)
    own = working(market, trader, 0)
    rest(market, 0, BID, 3000, 1)
    market.time = 50.0
    market.step()  # far past its patience, and nobody looks
    assert working(market, trader, 0) == own and not trader.position.any()


def test_a_look_between_decisions_crosses_what_is_left_of_a_need_whose_patience_has_run_out():
    market, _, trader = worked([3, 0], patience=5.0, reprice=2.0, interval=100.0)
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 12)  # 2 of its 3 are bought: 1 is left
    market.time = 5.0  # 4 of its 5 units, and the look is due (its order alone at 2999 is moved to 2998)
    market.step()
    assert working(market, trader, 0)[:2] == (2998, 1) and not market.account(trader).orders_crossed
    market.time = 7.5  # the look restarted at 5 is due again, and 6.5 of its 5 units have run
    market.step()  # crossed, the lot that is left, at the best ask, and the need is over
    account = market.account(trader)
    assert trader.position.tolist() == [3, 0] and working(market, trader, 0) is None
    assert account.orders_crossed == 1 and account.crossed_lots == 1 and account.passive_lots == 2
    assert clock_of(market, trader, 0) is None and len(account.waits) == 1
    assert account.waits[0][1] == pytest.approx(6.5, abs=0.05)
    time, book, lots, price, mid, kind = account.fills[-1]
    assert (book, lots, price, kind) == (0, 1, 3001, "cross")


def test_a_look_between_decisions_places_what_is_left_of_the_need_where_nothing_rests_after_a_jump():
    market, _, trader = worked([3, 0], patience=50.0, reprice=2.0, interval=100.0)
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 12)  # 2 of its 3 lots are bought: 1 is left of the need at the last decision
    market._shift(0, 2)  # a jump up: its order is withdrawn, and the others are 2 ticks higher
    assert working(market, trader, 0) is None and clock_of(market, trader, 0) == (1.0, BID)
    assert market.books[0].best_bid == 3000 and market.books[0].best_ask == 3003  # the book is 2 ticks up
    market.time = 3.5
    market.step()
    assert working(market, trader, 0)[:2] == (3000, 1)  # what is left (1), not the need it had (3), at the new best bid
    assert queue(market, 0, BID, 3000)[-1] == working(market, trader, 0)[2]


def test_a_look_leaves_what_rests_and_places_nothing_where_the_side_has_no_other_order():
    market, _, trader = worked([0, -2], patience=50.0, reprice=1.0, interval=100.0)
    decide(market, trader, 1.0)
    own = working(market, trader, 1)
    for price in (3001, 3002, 3003):
        clear(market, 1, ASK, price, keep=[own[2]])
    events = market.events
    market.time = 3.0
    market.step()  # its ask is all there is: the best price without it is none, and it stays
    assert working(market, trader, 1) == own and market.events == events
    market._shift(1, 1)  # a jump withdraws it; there is no other ask
    market.time = 6.0
    market.step()
    assert working(market, trader, 1) is None and clock_of(market, trader, 1) == (1.0, ASK)  # nothing placed


def test_a_trader_that_is_no_longer_active_has_the_clocks_and_needs_of_withdrawn_orders_dropped_too():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    market._shift(0, 2)  # a jump: its order is withdrawn, its need and clock are kept
    assert not work_of(market, trader).order and clock_of(market, trader, 0) == (1.0, BID)
    trader.active = False
    market.step()
    assert clock_of(market, trader, 0) is None and work_of(market, trader).left.tolist() == [0, 0]
    assert market.account(trader).orders_withdrawn == 1  # the jump's: nothing was left to withdraw


def test_a_trader_that_is_no_longer_active_has_its_orders_withdrawn_at_the_next_step_and_its_needs_dropped():
    market, _, trader = worked([3, -2])
    decide(market, trader, 1.0)
    events = market.events
    trader.active = False
    trader.next_time = 0.0  # it would be due, if it were active
    market.step()
    account = market.account(trader)
    assert not any(market._working) and not work_of(market, trader).clock and not work_of(market, trader).left.any()
    assert account.orders_withdrawn == 2 and market.events == events + 2 and market.books[0].num_orders(BID) == 3
    market.step()
    assert account.orders_placed == 2 and account.orders_withdrawn == 2  # it is not asked again, and nothing is left


# ---- A fill of a working order ----

def test_a_model_order_that_hits_a_working_order_books_the_trader_at_the_orders_price_with_the_makers_sign():
    market, _, trader = worked([3, -2])
    decide(market, trader, 1.0)
    market.time = 2.0
    hit(market, 0, ASK, 12)  # a model sells 12: the 10 ahead of the bid, and 2 of its 3
    hit(market, 1, BID, 11)  # a model buys 11: the 10 ahead of the ask, and 1 of its 2
    account = market.account(trader)
    assert trader.position.tolist() == [2, -1] and trader.cash == -2 * 2999 + 1 * 3001
    assert trader.book_cash.tolist() == [-2 * 2999, 3001] and trader.lots == 3 and trader.spread_paid == 0.0
    assert (account.passive_lots, account.crossed_lots) == (3, 0)
    assert account.fills == [(2.0, 0, 2, 2999, 3000.0, "market"), (2.0, 1, -1, 3001, 3000.0, "market")]
    assert account.spread_earned == (3000 - 2999) * 2 + (3001 - 3000) * 1  # a bid below the mid and an ask above it
    adds_up(market, trader)


def test_a_model_limit_order_priced_across_the_quote_that_hits_a_working_order_is_a_fill_of_the_kind_limit():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    before = list(market.fair.fairs)
    hit(market, 0, ASK, 12, kind=OrderType.LIMIT, price=2999, impact=False)  # as ``step`` sends the book's repricing
    assert trader.position.tolist() == [2, 0] and market.account(trader).fills[0][3:] == (2999, 3000.0, "limit")
    assert list(market.fair.fairs) == before  # a model's crossing limit order moves no fair value, whoever it hits


def test_the_spread_earned_is_told_against_the_mid_before_the_order_that_hit_it_and_not_the_one_after():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 13)  # the whole of the level: the mid goes from 3000 to 2999.5
    assert market.books[0].mid_price == 2999.5
    assert market.account(trader).spread_earned == (3000.0 - 2999) * 3 and market.account(trader).fills[0][4] == 3000.0


def test_the_spread_earned_is_told_against_the_mid_without_the_traders_own_order():
    market, _, trader = worked([3, 0], reprice=100.0)
    decide(market, trader, 1.0)
    clear(market, 0, BID, 2999, keep=[working(market, trader, 0)[2]])  # its bid alone at the best price
    hit(market, 0, ASK, 3)
    # the mid before, without its own bid, is (2998 + 3001) / 2: its fill at 2999 earned half a tick a lot, not a tick
    assert market.account(trader).spread_earned == 0.5 * 3 and market.account(trader).fills[0][4] == 2999.5


def test_with_no_mid_before_the_fill_a_passive_fill_is_booked_and_earns_nothing_against_it():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    for price in (3001, 3002, 3003):
        clear(market, 0, ASK, price)  # no ask: the book has no mid
    hit(market, 0, ASK, 12)
    account = market.account(trader)
    assert trader.position.tolist() == [2, 0] and account.passive_lots == 2 and account.spread_earned == 0.0
    assert account.fills[0][:4] == (1.0, 0, 2, 2999) and math.isnan(account.fills[0][4])


def test_an_order_sent_with_no_word_about_impact_has_its_impact_as_a_book_markets_does():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    before = list(market.fair.fairs)
    market.submit(0, market_order(12, 2.0, ASK))  # as ``submit(book, order)``, with the base's default of impact on
    assert market.fair.fairs[0] == pytest.approx(before[0] - 0.05 * 12)
    plain, _ = hand_market()
    plain.submit(0, market_order(5, 2.0, BID))  # and with nothing resting
    assert plain.fair.fairs[0] == pytest.approx(3000.0 + 0.05 * 5)


def test_a_model_market_order_that_hits_a_working_order_moves_the_fair_value_by_all_the_lots_it_took():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    before = list(market.fair.fairs)
    hit(market, 0, ASK, 12)  # 10 lots of others, 2 of the trader's: the order's impact is its own, whoever it hits
    assert market.fair.fairs[0] == pytest.approx(before[0] - 0.05 * 12) and market.fair.fairs[1] == before[1]


def test_a_trader_that_crosses_into_another_traders_working_order_pays_one_and_credits_the_other():
    market, _ = hand_market()
    maker, taker = Wants(goal=np.array([3, 0]), interval=5.0), Wants(goal=np.array([-13, 0]), interval=5.0)
    market.add(maker, Execution(10.0, reprice=None))
    market.add(taker)  # no execution: it sells 13 with a market order
    decide(market, maker, 1.0)
    decide(market, taker, 2.0)  # the 10 lots ahead of the maker's bid and its 3
    made, took = market.account(maker), market.account(taker)
    assert maker.position.tolist() == [3, 0] and taker.position.tolist() == [-13, 0]
    assert (made.passive_lots, made.crossed_lots, took.passive_lots, took.crossed_lots) == (3, 0, 0, 13)
    assert made.fills == [(2.0, 0, 3, 2999, 3000.0, "market")] and made.spread_earned == 3.0
    assert [fill[:4] for fill in took.fills] == [(2.0, 0, -10, 2999), (2.0, 0, -3, 2999)]
    assert {fill[5] for fill in took.fills} == {"cross"} and taker.spread_paid == 13.0  # a tick below the mid, 13 lots
    assert maker.cash == -3 * 2999 and taker.cash == 13 * 2999
    assert not work_of(market, maker).clock and not any(market._working)  # its need is over: its order is gone
    adds_up(market, maker)
    adds_up(market, taker)


def test_a_worked_traders_market_order_after_its_patience_fills_anothers_working_order_and_both_accounts_are_right():
    market, _ = hand_market()
    maker, taker = Wants(goal=np.array([3, 0]), interval=5.0), Wants(goal=np.array([-13, 0]), interval=5.0)
    market.add(maker, Execution(100.0, reprice=None))
    market.add(taker, Execution(10.0, reprice=None))
    decide(market, maker, 1.0)  # a bid of 3 at 2999, behind the 10 there
    decide(market, taker, 1.0)  # an ask of 13 at 3001: nobody trades
    assert maker.lots == taker.lots == 0 and working(market, taker, 0)[:2] == (3001, 13)
    decide(market, taker, 11.0)  # its patience has run: its ask is out, and it sells 13: the 10 and the maker's 3
    made, took = market.account(maker), market.account(taker)
    assert maker.position.tolist() == [3, 0] and taker.position.tolist() == [-13, 0]
    assert (made.passive_lots, made.crossed_lots, took.passive_lots, took.crossed_lots) == (3, 0, 0, 13)
    assert made.fills == [(11.0, 0, 3, 2999, 3000.0, "market")] and made.spread_earned == 3.0
    assert [fill[:4] for fill in took.fills] == [(11.0, 0, -10, 2999), (11.0, 0, -3, 2999)]
    assert took.orders_crossed == 1 and made.orders_crossed == 0 and took.waits == [(11.0, 10.0)]
    assert made.waits == [(11.0, 10.0)] and not work_of(market, maker).clock and not work_of(market, taker).clock
    assert not any(market._working) and maker.cash == -3 * 2999 and taker.cash == 13 * 2999
    adds_up(market, maker)
    adds_up(market, taker)


def test_two_traders_working_opposite_sides_never_trade_with_each_other_because_each_joins_its_own_side():
    market, _ = hand_market()
    buyer, seller = Wants(goal=np.array([3, 0]), interval=5.0), Wants(goal=np.array([-3, 0]), interval=5.0)
    market.add(buyer, Execution(10.0, reprice=None))
    market.add(seller, Execution(10.0, reprice=None))
    decide(market, buyer, 1.0)
    decide(market, seller, 1.0)
    assert working(market, buyer, 0)[:2] == (2999, 3) and working(market, seller, 0)[:2] == (3001, 3)
    assert buyer.lots == seller.lots == 0 and market.books[0].spread == 2


# ---- What a model's cancel does to a working order ----

def test_a_models_cancel_that_names_a_working_order_is_dropped_and_counts_no_event():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    own, events = working(market, trader, 0), market.events
    rates = list(market._rates)
    assert market.submit(0, cancel_order(own[2], 2.0)) == []
    assert working(market, trader, 0) == own and market.events == events and market._rates == rates
    assert market.books[0].get_order(own[2]) is not None


def test_a_models_cancel_of_an_order_that_is_not_a_traders_is_made_and_counts_an_event_as_before():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    events = market.events
    ahead = queue(market, 0, BID, 2999)[0]  # the model's own order at the best bid
    market.submit(0, cancel_order(ahead, 2.0))
    assert market.books[0].get_order(ahead) is None and market.events == events + 1
    assert working(market, trader, 0) is not None


def test_a_cancel_the_models_event_sends_through_step_is_dropped_the_same_way_and_the_others_are_not():
    market, models, trader = worked([3, 0])
    decide(market, trader, 1.0)
    own, ahead = working(market, trader, 0), queue(market, 0, BID, 2999)[0]
    events = market.events
    play(market, models, {0: [cancel_order(own[2], 0.0)]})
    assert working(market, trader, 0) == own and market.events == events and not market.volume.any()
    play(market, models, {0: [cancel_order(ahead, 0.0)]})
    assert market.books[0].get_order(ahead) is None and market.events == events + 1


def test_a_traders_own_withdrawal_is_never_dropped_and_counts_an_event():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    own, events = working(market, trader, 0)[2], market.events
    trader.goal = np.array([0, 0])
    decide(market, trader, 2.0)
    assert market.books[0].get_order(own) is None and market.events == events + 1


# ---- What placing and withdrawing do to the market ----

def test_a_placement_and_a_withdrawal_each_count_an_event_and_take_the_books_rate_again_and_nothing_else_moves():
    market = PassiveBookMarket(FactorValues(np.ones((2, 1))), seed=1)  # the order flow's own models: 16 orders a book
    trader = Wants(goal=np.array([3, 0]), interval=5.0)
    market.add(trader, Execution(10.0, reprice=None))
    rate = AnchoredZIModel().rate
    assert market._rates[0] == pytest.approx(10 + 1.5 + 0.05 * 16)
    fairs, events = list(market.fair.fairs), market.events
    decide(market, trader, 1.0)
    assert market.events == events + 1 and market._rates[0] == pytest.approx(10 + 1.5 + 0.05 * 17)
    assert market._rates[0] == rate(market.books[0]) and market._rates[1] == pytest.approx(10 + 1.5 + 0.05 * 16)
    trader.goal = np.array([0, 0])
    decide(market, trader, 2.0)
    assert market.events == events + 2 and market._rates[0] == pytest.approx(10 + 1.5 + 0.05 * 16)
    assert list(market.fair.fairs) == fairs and not market.flow.any() and not market.volume.any()


def test_a_fill_takes_the_books_rate_again():
    market = PassiveBookMarket(FactorValues(np.ones((2, 1))), seed=1)
    trader = Wants(goal=np.array([3, 0]), interval=5.0)
    market.add(trader, Execution(10.0, reprice=None))
    decide(market, trader, 1.0)
    market.time = 2.0
    hit(market, 0, ASK, 1000)  # every bid in the book: 8 orders and the trader's, which are then all gone
    assert market.books[0].num_orders(BID) == 0 and market._rates[0] == pytest.approx(10 + 1.5 + 0.05 * 8)
    assert trader.position.tolist() == [3, 0]


def test_a_placement_that_trades_raises_an_error_since_joining_the_best_price_on_its_own_side_cannot_cross(monkeypatch):
    market, _, trader = worked([3, 0])
    monkeypatch.setattr(market, "_best", lambda work, book, side: 3001)  # a price that crosses: it is the best ask
    fairs = list(market.fair.fairs)
    with pytest.raises(RuntimeError, match="traded as it was placed"):
        decide(market, trader, 1.0)
    assert list(market.fair.fairs) == fairs  # a placement has no impact, even one that went wrong


# ---- A jump ----

def test_a_jump_withdraws_every_traders_working_orders_in_that_book_and_keeps_their_needs_and_clocks():
    market, _, first = worked([3, -2])
    second = Wants(goal=np.array([-4, 1]), interval=5.0)
    market.add(second, Execution(10.0, reprice=None))
    decide(market, first, 1.0)
    decide(market, second, 2.0)
    events, own = market.events, working(market, first, 1)
    market._shift(0, 3)  # book 0 jumps up 3 ticks
    assert working(market, first, 0) is None and working(market, second, 0) is None
    assert working(market, first, 1) == own and working(market, second, 1) is not None  # book 1 does not jump
    assert market.events == events + 2 and market.account(first).orders_withdrawn == 1
    assert market.account(second).orders_withdrawn == 1 and not market._working[0]
    assert clock_of(market, first, 0) == (1.0, BID) and clock_of(market, second, 0) == (2.0, ASK)
    assert work_of(market, first).left.tolist() == [3, -2] and work_of(market, second).left.tolist() == [-4, 1]
    book = market.books[0]  # the others went with the book: 3000 to 3002 bids, 3004 to 3006 asks, and it is not crossed
    assert (book.best_bid, book.best_ask) == (3002, 3004) and book.num_orders() == 6


def test_after_a_jump_the_next_look_places_the_need_again_at_the_best_price_of_the_shifted_book():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    market._shift(0, -3)
    assert working(market, trader, 0) is None and market.books[0].best_bid == 2996
    decide(market, trader, 2.0)
    assert working(market, trader, 0)[:2] == (2996, 3) and clock_of(market, trader, 0) == (1.0, BID)
    assert queue(market, 0, BID, 2996)[-1] == working(market, trader, 0)[2]


def test_a_jump_of_half_a_tick_that_moves_no_order_withdraws_the_working_orders_all_the_same():
    market, _, trader = worked([3, 0])
    decide(market, trader, 1.0)
    market._shift(0, 0)  # a jump that is half a tick rounds to no tick: the others stay, and it was seen all the same
    assert working(market, trader, 0) is None and clock_of(market, trader, 0) == (1.0, BID)
    assert market.books[0].best_bid == 2999


def test_a_jump_that_step_finds_in_the_fair_values_is_shifted_with_the_orders_withdrawn_first():
    market, _, trader = worked([3, -2])
    decide(market, trader, 1.0)
    market.fair.gapped.append((0, 3.0))  # as a deal's announcement leaves it
    market.step()
    assert working(market, trader, 0) is None and working(market, trader, 1) is not None
    assert market.books[0].best_bid == 3002 and market.fair.gapped == []
    assert market.account(trader).orders_withdrawn == 1


# ---- What a passive fill is worth later: the markouts ----

class Beat:
    """The market's generator with every step exactly ``dt`` units of time long (the draw that picks the book is the
    real one), so that a step can fall on a time the test chooses."""

    def __init__(self, rng, dt):
        self._rng, self.dt = rng, dt

    def exponential(self, scale):
        return self.dt

    def random(self):
        return self._rng.random()


def step_to(market, until, dt=0.25):
    """Step, ``dt`` of time each, until the time is ``until`` (a whole number of ``dt`` from where it is; exactly it
    if ``dt`` is a binary fraction)."""
    if not isinstance(market.rng, Beat):
        market.rng = Beat(market.rng, dt)
    while market.time < until - 1e-9:
        market.step()
    assert abs(market.time - until) < 1e-9


def test_a_passive_fill_is_marked_1_5_and_20_units_later_at_the_first_step_at_or_after_each_time_at_the_mid_then():
    market, _, trader = worked([1, -1], patience=100.0)
    decide(market, trader, 2.0)  # a bid for 1 at 2999 and an ask for 1 at 3001, each behind 10 lots
    trader.next_time = 1e9  # it does not decide again
    hit(market, 0, ASK, 11)  # the 10 ahead and its bid, at 2999: book 0's mid is then 2999.5
    hit(market, 1, BID, 11)  # the 10 ahead and its ask, at 3001: book 1's mid is then 3000.5
    account = market.account(trader)
    assert [(f[0], f[1], f[2], f[3], f[5]) for f in account.fills] == [(2.0, 0, 1, 2999, "market"),
                                                                       (2.0, 1, -1, 3001, "market")]
    assert account.markouts == []
    step_to(market, 2.75)
    assert account.markouts == []  # a unit has not passed
    step_to(market, 3.0)  # a step that falls on the time itself is at it
    assert account.markouts == [(0, 1.0, 0.5), (1, 1.0, 0.5)]  # a bought lot at 2999 is worth 2999.5, a sold one 3000.5
    rest(market, 0, BID, 3000, 1)  # book 0's mid is 3000.5 now, book 1's 3001.5
    rest(market, 1, BID, 3001, 1)
    step_to(market, 6.75)
    assert len(account.markouts) == 2  # 5 units have not passed
    step_to(market, 7.0)
    assert account.markouts[2:] == [(0, 5.0, 1.5), (1, 5.0, -0.5)]  # 3000.5 less 2999, and minus 3001.5 less 3001
    clear(market, 0, ASK, 3001)  # book 0's mid is 3001.0, book 1's 3002.0
    clear(market, 1, ASK, 3002)
    step_to(market, 21.75)
    assert len(account.markouts) == 4
    step_to(market, 22.0)
    assert account.markouts[4:] == [(0, 20.0, 2.0), (1, 20.0, -1.0)]
    assert not any(market._pending)  # all taken, and each once


def test_a_markout_is_taken_at_the_first_step_after_its_time_when_no_step_falls_on_it():
    market, _, trader = worked([1, 0], patience=100.0)
    decide(market, trader, 2.0)
    trader.next_time = 1e9
    hit(market, 0, ASK, 11)  # filled at 2999 at time 2.0: marked at 3.0, 7.0 and 22.0
    step_to(market, 2.9, dt=0.3)  # 2.3, 2.6, 2.9
    assert market.account(trader).markouts == []
    rest(market, 0, BID, 3000, 1)  # the mid is 3000.5 from here
    step_to(market, 3.2)  # the first step past 3.0
    assert market.account(trader).markouts == [(0, 1.0, pytest.approx(1.5))]


def test_a_markout_is_taken_at_the_mid_without_the_traders_own_order_and_at_the_markets_mark_where_that_is_none():
    market, _, trader = worked([1, 0], patience=100.0)
    decide(market, trader, 2.0)
    hit(market, 0, ASK, 11)  # filled at 2999 at time 2.0
    trader.goal = np.array([2, 0])
    decide(market, trader, 2.5)  # one more: a bid at 2998, the best now, behind 10
    clear(market, 0, BID, 2998, keep=[working(market, trader, 0)[2]])  # alone at the best bid
    trader.next_time = 1e9
    assert market.books[0].mid_price == 2999.5 and market.mid_for(trader, 0) == 2999.0
    account = market.account(trader)
    step_to(market, 3.0)
    assert account.markouts == [(0, 1.0, 0.0)]  # at 2999.0 without its own bid, not at 2999.5: a lot made nothing
    for price in (3001, 3002, 3003):
        clear(market, 0, ASK, price)
    last = market.mark(0)  # no mid now: the market's mark is the last one the book had, as its last order left it
    assert market.mid_for(trader, 0) is None and market.books[0].mid_price is None and last == 2999.5
    step_to(market, 7.0)
    assert account.markouts[1:] == [(0, 5.0, last - 2999)]
    step_to(market, 22.0)
    assert account.markouts[2:] == [(0, 20.0, 0.5)]


def test_the_markout_of_a_fill_by_a_limit_order_is_kept_apart_from_that_of_a_market_order():
    market, _, trader = worked([1, 0], patience=100.0)
    decide(market, trader, 2.0)
    trader.next_time = 1e9
    hit(market, 0, ASK, 11)  # 1 lot at 2999, by a market order
    trader.goal = np.array([4, 0])
    decide(market, trader, 2.5)
    trader.next_time = 1e9
    hit(market, 0, ASK, 13, kind=OrderType.LIMIT, price=2998, impact=False)  # 3 lots at 2998, by a limit order
    assert [f[5] for f in market.account(trader).fills] == ["market", "limit"]
    step_to(market, 3.5)  # book 0's mid is 2999.0 now: bids 2997 and 2996, asks 3001
    account = market.account(trader)
    assert account.markout(1.0, "market") == pytest.approx(0.0)  # 2999.0 less 2999
    assert account.markout(1.0, "limit") == pytest.approx(1.0)  # 2999.0 less 2998
    assert account.markout(5.0, "market") == 0.0 and account.markout(1.0, "other") == 0.0  # none yet; no such kind


def test_the_markout_a_lot_is_the_mean_of_the_markouts_of_the_fills_weighted_by_their_lots_from_a_row_on():
    market, _, trader = worked([1, 0], patience=100.0)
    decide(market, trader, 2.0)
    trader.next_time = 1e9
    hit(market, 0, ASK, 11)  # row 0: 1 lot at 2999
    trader.goal = np.array([4, 0])
    decide(market, trader, 2.5)
    trader.next_time = 1e9
    hit(market, 0, ASK, 13)  # row 1: 3 lots at 2998, both by market orders
    step_to(market, 3.0)  # the mid is 2999.0 (bids 2997, asks 3001): row 0 is worth 0.0
    rest(market, 0, BID, 3000, 1)  # the mid is 3000.5
    step_to(market, 3.5)  # row 1 is worth 3000.5 less 2998: 2.5
    account = market.account(trader)
    assert [m[:2] for m in account.markouts] == [(0, 1.0), (1, 1.0)]
    assert account.markout(1.0, "market") == pytest.approx((1 * 0.0 + 3 * 2.5) / 4)  # by the lots, not by the fills
    assert account.markout(1.0, "market", since=1) == pytest.approx(2.5)  # the fills from row 1 on
    assert account.markout(1.0, "market", since=2) == 0.0  # none


def test_the_fills_of_a_traders_own_market_orders_are_not_marked_whether_it_crossed_or_closed():
    market, _ = hand_market()
    trader = Wants(goal=np.array([3, 0]), interval=5.0)
    market.add(trader, Execution(10.0, reprice=None))
    other = Wants(goal=np.array([-1, 0]), interval=5.0)
    market.add(other)  # no execution: it sells at once
    decide(market, other, 1.0)
    decide(market, trader, 2.0)
    decide(market, trader, 30.0)  # its patience has run out: what is left of its need is crossed
    assert market.account(trader).crossed_lots == 3 and market.account(other).crossed_lots == 1
    assert not market.account(trader).markouts and not market.account(other).markouts and not any(market._pending)


# ---- score_execution ----

class Timed:
    """An order-flow model that sends, at the first event at or after each time it is given, the order that time's
    function gives (or none), and nothing else. Its rate is a few hundred an unit, so a step is a few thousandths."""

    def __init__(self, rate=200.0):
        self.orders, self._rate = [], rate

    def rate(self, book):
        return self._rate

    def event(self, book, t, fair, rng):
        if self.orders and self.orders[0][0] <= t:
            return self.orders.pop(0)[1](t)
        return None


def staged_market(execution):
    """Two books of timed models and a trader that wants 1 lot in the first, then 4 from time 2 and 5 from time 7,
    and the orders that fill it: a market sell of 11 at 0.5 (its first lot, at 2999), of 13 at 6 (3 lots at 2998), and
    a limit sell of 11 at 2997 at 12 (its last lot). The market, the trader."""
    models = [Timed(), Timed()]
    market = PassiveBookMarket(FactorValues(np.ones((2, 1))), models=models, seed=1, depth=6, lots=10)
    trader = Wants(goal=np.array([1, 0]), interval=5.0)
    market.add(trader, execution)

    def want(lots):
        def change(t):
            trader.goal = np.array([lots, 0])
        return change

    models[0].orders = [(0.5, lambda t: market_order(11, t, ASK)), (2.0, want(4)),
                        (6.0, lambda t: market_order(13, t, ASK)), (7.0, want(5)),
                        (12.0, lambda t: limit_order(2997, 11, t, ASK))]
    return market, trader


def test_score_execution_of_a_worked_trader_over_a_span_is_what_its_account_did_in_the_span_alone():
    market, trader = staged_market(Execution(100.0, reprice=None))
    score = score_execution(market, trader, 2.0, 40.0)
    # the first lot filled at 0.5, before the span; in it, 3 lots at 2998 (a market sell) and 1 at 2997 (a limit sell)
    assert trader.position.tolist() == [5, 0] and market.time > 41.9
    assert score["pnl"] == pytest.approx(2.0) and score["costs"] == 0.0 and score["lots"] == 4
    # paper: asked 1 at 3000 (before), then 3 at 2999.5 and 1 at 2999.0, and at the end held at 2998.5; against 2999.5
    assert score["paper"] == pytest.approx(-4.5) and score["asked_lots"] == 4
    assert score["shortfall"] == pytest.approx(-6.5) and score["shortfall_a_lot"] == pytest.approx(-6.5 / 4)
    assert score["earned"] == pytest.approx((2999.5 - 2998) * 3 + (2999.0 - 2997) * 1)
    assert (score["passive_lots"], score["crossed_lots"], score["urgent_lots"]) == (4, 0, 0)
    assert (score["orders_placed"], score["orders_withdrawn"], score["orders_crossed"]) == (2, 0, 0)
    assert score["wait"] == pytest.approx((1.0 + 2.0) / 2, abs=0.05)  # its needs of 3 and 1 lots ended 1 and 2 later
    # the markouts of the fills of the span only (the first lot's would make the first 0.875), by kind
    assert score["markout_1_market"] == pytest.approx(1.0) and score["markout_5_market"] == pytest.approx(1.0)
    assert score["markout_20_market"] == pytest.approx(0.5)  # 2999.0 twice, then 2998.5 less 2998
    assert [score[f"markout_{h}_limit"] for h in (1, 5, 20)] == pytest.approx([1.5, 1.5, 1.5])  # 2998.5 less 2997
    assert score.keys() >= {"deviation_long", "risk_adjusted_long", "factor_part", "pnl", "before_costs", "costs"}


def test_a_markout_whose_later_time_has_not_come_by_the_end_of_the_span_is_left_out_of_its_score():
    market, trader = staged_market(Execution(100.0, reprice=None))
    score = score_execution(market, trader, 2.0, 19.0)  # it ends at about 21: the fill at 6 is 15 units old
    assert score["markout_1_market"] == pytest.approx(1.0) and score["markout_5_market"] == pytest.approx(1.0)
    assert score["markout_20_market"] == 0.0  # due at about 26
    assert score["markout_1_limit"] == pytest.approx(1.5) and score["markout_20_limit"] == 0.0  # the limit fill at 12


def test_for_a_trader_with_no_execution_score_execution_is_what_score_holdings_is_and_shortfall_is_what_crossing_cost():
    scored = {}
    for name, kind in (("book", BookMarket), ("passive", PassiveBookMarket)):
        models = [Scripted(), Scripted()]
        market = kind(FactorValues(np.ones((2, 1))), models=models, seed=1, depth=3, lots=10)
        trader = Wants(goal=np.array([2, -1]), interval=5.0)
        market.add(trader)
        score = (score_holdings if kind is BookMarket else score_execution)(market, trader, 0.0, 10.0, 5.0)
        scored[name] = (score, market, trader)
    (plain, book, _), (score, market, trader) = scored["book"], scored["passive"]
    assert {key: score[key] for key in plain} == plain and plain["lots"] == 3  # every number, to the last digit
    assert (market.time, market.events, list(market._last_mid)) == (book.time, book.events, list(book._last_mid))
    assert market.rng.bit_generator.state == book.rng.bit_generator.state and list(market.flow) == list(book.flow)
    # it crossed 2 lots at 3001 and 1 at 2999 against mids of 3000: the paper account asked for them at 3000
    assert score["paper"] == 0.0 and score["pnl"] == -3.0 and score["costs"] == 3.0
    assert score["shortfall"] == 3.0 and score["asked_lots"] == 3 and score["shortfall_a_lot"] == 1.0
    assert (score["crossed_lots"], score["passive_lots"], score["urgent_lots"], score["orders_placed"]) == (3, 0, 0, 0)
    assert score["earned"] == 0.0 and score["wait"] == 0.0 and score["markout_1_market"] == 0.0


def test_score_execution_has_the_horizon_score_holdings_has_by_default_and_passes_one_it_is_given_on():
    scores = {}
    for name, horizon in (("default", None), ("short", 50.0)):
        for kind, score_of in ((BookMarket, score_holdings), (PassiveBookMarket, score_execution)):
            market = kind(FactorValues(sector_loadings(1, 3)), seed=3)
            trader = Drifter(seed=3, interval=5.0, every=5.0)
            market.add(trader)
            args = (market, trader, 50.0, 450.0) + (() if horizon is None else (horizon,))
            scores[(name, kind)] = score_of(*args)
    for name in ("default", "short"):
        plain, score = scores[(name, BookMarket)], scores[(name, PassiveBookMarket)]
        assert {key: score[key] for key in plain} == plain
    # the long-run deviation is 0 unless the span has a few stretches of the horizon: it does at 100 and 50, and differs
    default, short = (scores[(name, PassiveBookMarket)]["deviation_long"] for name in ("default", "short"))
    assert default > 0 and short > 0 and default != short


def test_a_shortfall_a_lot_is_0_for_a_trader_that_asked_for_nothing_in_the_span():
    market, _ = hand_market()
    trader = Wants(goal=np.array([0, 0]), interval=5.0)
    market.add(trader, Execution(10.0, reprice=None))
    score = score_execution(market, trader, 1.0, 10.0, 5.0)
    assert score["asked_lots"] == 0 and score["shortfall_a_lot"] == 0.0 and score["shortfall"] == 0.0


def test_score_execution_of_a_trader_that_is_not_in_the_market_is_refused_before_the_market_runs():
    market, _ = hand_market()
    with pytest.raises(ValueError, match="not in this market"):
        score_execution(market, Wants(goal=np.array([1, 0])), 5.0, 10.0)
    assert market.time == 0.0


def test_score_execution_marks_the_trader_at_the_mids_without_its_own_orders_and_the_market_is_left_as_it_was():
    market, _, trader = worked([1, 0], patience=1000.0)
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 11)  # it holds 1 lot, bought at 2999
    trader.goal = np.array([2, 0])
    decide(market, trader, 1.5)  # a bid for one more at 2998, behind 10
    clear(market, 0, BID, 2998, keep=[working(market, trader, 0)[2]])  # alone at the best bid
    assert market.marks().tolist() == [2999.5, 3000.0] and market.marks_for(trader).tolist() == [2999.0, 3000.0]
    trader.goal = np.array([1, 0])
    trader.next_time = 3.0  # it decides in the span, wants no more, and its bid goes: the marks do not change
    score = score_execution(market, trader, 1.5, 5.0, 5.0)
    assert working(market, trader, 0) is None and market.marks().tolist() == [2999.0, 3000.0]
    assert score["pnl"] == 0.0  # marked at 2999.0 at both ends; at the book's own marks it would have lost 0.5
    assert score["paper"] == 0.0 and score["asked_lots"] == 1 and score["shortfall_a_lot"] == 0.0
# ---- What the judgment review found: a band, a stop, a missing mid, a thin book, own orders in a mark ----

@dataclass
class Banded(StateTrader):
    """A trader of one relation, long in book 0, that asks (through the library's own ``_become``) for ``size`` lots of
    it, as the test sets ``size`` before each decision: with a rate of 1 its aim is ``size``, and it trades to the edge
    of its band."""

    def decide(self, mids):
        self.weights = np.eye(len(mids))[:1]
        return self._become(np.array([1]))


BAND_SIZES = [3.0, 2.0, 4.0, 3.0, 1.0, 0.0]


def banded(execution, hits=()):
    """A ``Banded`` trader of band 1 decides at each of ``BAND_SIZES``, and after the decision with an index in
    ``hits`` a model sells that many lots into book 0. The market, the trader."""
    market, _ = hand_market()
    market.keep_decisions = True
    trader = Banded(band=1.0, interval=5.0)
    market.add(trader, execution)
    for k, size in enumerate(BAND_SIZES):
        trader.size = size
        decide(market, trader, k + 1.0)
        for index, lots in hits:
            if index == k:
                hit(market, 0, ASK, lots)
    return market, trader


CROSS_SCORES = [[1.0, -1.0, 0.5, -0.5], [0.5, -0.5, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [-1.0, 1.0, -0.5, 0.5]]


def crossed_by_band(execution, hits=()):
    """A cross-sectional trader of four books with a band of 1 that rebalances at each decision, to each of
    ``CROSS_SCORES`` in turn (the third gives it nothing to hold), after a decision of warming up; and, after the
    decision with an index in ``hits``, a model sells that many lots into book 0. The market, the trader."""
    market, _ = hand_market(books=4)
    market.keep_decisions = True
    trader = ReversalTrader(lookback=1, factors=0, hold=1, size=4.0, interval=1.0, band=1.0)
    scores = [np.array(CROSS_SCORES[0])]
    trader.scores = lambda: scores[0]
    market.add(trader, execution)
    decide(market, trader, 0.5)
    for k, new in enumerate(CROSS_SCORES):
        scores[0] = np.array(new)
        decide(market, trader, k + 1.0)
        for index, lots in hits:
            if index == k:
                hit(market, 0, ASK, lots)
    return market, trader


@pytest.mark.parametrize("build, asked", [
    (banded, [[2, 0], [2, 0], [3, 0], [3, 0], [2, 0], [0, 0]]),
    (crossed_by_band, [[0] * 4, [3, -3, 1, -1], [3, -3, 3, -3], [0] * 4, [-3, 3, -1, 1]])])
def test_the_paper_account_of_a_trader_with_a_band_does_not_follow_its_fills_and_is_its_crossing_twins(build, asked):
    # inside its band a trader asks for nothing, which is the position it holds: the paper took that, so a lot asked
    # and not yet filled was given up on paper at the next decision. The paper is what the trader's own rule asks of
    # the paper's last position (a trader filled at once holds it), whatever its orders did. By hand for the first
    # trader: a band of 1 around an aim of 3 asks for 2; aims of 2, 4, 3, 1 and 0 ask for nothing, 1 more, nothing,
    # 1 less (the aim 1 is 2 below the 3 held: its edge is 2) and all of the rest
    slow = Execution(100.0, reprice=None)
    runs = [build(None), build(slow), build(slow, [(0, 11), (2, 12)])]
    twin, quiet, busy = (market.account(trader) for market, trader in runs)
    assert busy.passive_lots > 0 and quiet.passive_lots == 0 and twin.passive_lots == 0  # the fills differ
    assert [row[2].tolist() for row in twin.decisions] == asked
    assert [row[2].tolist() for row in quiet.decisions] == asked  # no lot filled: the same asks
    assert [row[2].tolist() for row in busy.decisions] == asked  # some filled: the same asks
    lots = int(np.abs(np.diff(np.array([[0] * len(asked[0])] + asked), axis=0)).sum())  # what the paper changed by
    assert twin.asked_lots == quiet.asked_lots == busy.asked_lots == lots
    assert twin.paper_cash == quiet.paper_cash == 0.0  # the same asks at the same mids (the book's mid was 3000)


def test_a_band_traders_paper_keeps_the_lot_it_asked_for_through_a_jump_that_its_fills_did_not_get():
    # target 3 and a band of 1: it asks for 2. One lot fills. The target falls to 2: its position, 1, is inside the
    # band, so it asks for nothing; a trader filled at once holds 2, inside the band too, and so does its paper. After a
    # jump of 10 the paper has made 2 x 9.5 and the trader 1 x 10.5 - 1 x 0.5: the lot not filled is in the shortfall
    market, _ = hand_market()
    trader = Banded(size=3.0, band=1.0, interval=5.0)
    market.add(trader, Execution(50.0, reprice=None))
    account = market.account(trader)
    decide(market, trader, 1.0)
    market.time = 2.0
    hit(market, 0, ASK, 11)  # 10 ahead and one of its 2, at 2999
    trader.size = 2.0
    decide(market, trader, 3.0)
    assert account.asked.tolist() == [2, 0] and trader.position.tolist() == [1, 0]
    assert account.paper_cash == -2 * 3000.0 and account.asked_lots == 2
    market._shift(0, 10)
    marks = market.marks_for(trader)  # book 0's mid is (3008 + 3011) / 2
    paper = account.paper_cash + float(account.asked @ marks)
    real = trader.cash + float(trader.position @ marks)
    assert (paper, real) == (19.0, 10.5) and paper - real == 8.5


def test_a_trader_that_does_not_go_through_the_band_rule_keeps_its_position_and_its_need_as_its_paper():
    market, _, trader = worked([3, -2])
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 11)  # one lot of 3
    decide(market, trader, 2.0)
    assert trader.position.tolist() == [1, 0] and market.account(trader).asked.tolist() == [3, -2]


def test_the_before_costs_of_a_worked_trader_is_nan_and_paper_is_what_to_set_beside_a_crossing_traders():
    market, trader = staged_market(Execution(100.0, reprice=None))
    score = score_execution(market, trader, 2.0, 40.0)
    assert math.isnan(score["before_costs"]) and score["costs"] == 0.0
    assert score["pnl"] == pytest.approx(2.0) and score["paper"] == pytest.approx(-4.5)  # the numbers beside it stay
    market, _ = hand_market()  # a trader with no execution keeps the number score_holdings gives, its paper's
    twin = Wants(goal=np.array([1, 0]), interval=5.0)
    market.add(twin)
    score = score_execution(market, twin, 0.0, 10.0, 5.0)
    assert score["costs"] == 1.0 and score["pnl"] == -1.0 and score["before_costs"] == 0.0 == score["paper"]


def test_a_need_whose_patience_has_run_is_crossed_at_a_decision_that_cannot_be_made_for_want_of_a_mid():
    market, _, trader = worked([1, 0], patience=2.0, interval=1.0, record=True)
    decide(market, trader, 1.0)  # a bid for 1, behind 10
    own = working(market, trader, 0)
    for price in (2999, 2998, 2997):
        clear(market, 0, BID, price, keep=[own[2]])  # its bid is all the bids there are: no mid without it
    account = market.account(trader)
    decide(market, trader, 2.0)  # 1 of its 2 units, and no decision
    assert working(market, trader, 0) == own and account.orders_crossed == 0 and len(trader.mids) == 1
    decide(market, trader, 3.0)  # its patience has run: still no decision, but what is left is crossed
    assert trader.position.tolist() == [1, 0] and working(market, trader, 0) is None
    assert account.orders_crossed == 1 and account.crossed_lots == 1 and account.waits == [(3.0, 2.0)]
    assert clock_of(market, trader, 0) is None and work_of(market, trader).left.tolist() == [0, 0]
    assert len(trader.mids) == 1 and len(trader.history) == 1 and account.asked.tolist() == [1, 0]  # none made
    assert trader.next_time == 3.0 + trader.interval and math.isnan(account.fills[-1][4])  # no mid before
    adds_up(market, trader)


def test_an_inactive_trader_that_is_stopped_gives_up_the_lots_it_did_not_get_on_paper_at_the_marks_then():
    market, _, trader = worked([3, 0])
    market.keep_decisions = True
    account = market.account(trader)
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 11)  # 10 ahead and one of its 3: 2 left, alone at 2999 (the mid without it 2999.5)
    trader.active = False
    market.time = 2.0
    market.step()
    assert not any(market._working) and account.asked.tolist() == [1, 0] == trader.position.tolist()
    assert account.paper_cash == pytest.approx(-3 * 3000.0 + 2 * 2999.5)  # the two given back at the marks of then
    assert [row[2].tolist() for row in account.decisions] == [[3, 0], [1, 0]] and account.decisions[1][0] > 1.99

    def shortfall():
        marks = market.marks_for(trader)
        return (account.paper_cash + float(account.asked @ marks)) - (trader.cash + float(trader.position @ marks))

    before = shortfall()
    market._shift(0, 5)
    market._shift(0, 5)  # the price goes on: a trader that has stopped asks for nothing, its shortfall does not move
    assert shortfall() == pytest.approx(before) and before == pytest.approx(-2.0)
    market.step()
    assert len(account.decisions) == 2  # and it is given up once


def test_a_trader_stopped_with_nothing_resting_and_no_need_leaves_its_paper_as_it_was():
    market, _, trader = worked([2, 0])
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 12)  # all of it
    assert trader.position.tolist() == [2, 0]
    trader.active = False
    market.step()
    assert market.account(trader).asked.tolist() == [2, 0] and market.account(trader).paper_cash == -2 * 3000.0


def test_a_stopped_trader_with_a_band_and_nothing_resting_gives_up_the_lots_its_paper_holds():
    # inside its band it has no need, no order and no clock, but its paper holds a lot its position lacks: the stop
    # gives it up all the same
    market, _ = hand_market()
    trader = Banded(size=3.0, band=1.0, interval=5.0)
    market.add(trader, Execution(50.0, reprice=None))
    account = market.account(trader)
    decide(market, trader, 1.0)
    market.time = 2.0
    hit(market, 0, ASK, 11)  # 10 ahead and one of its 2
    trader.size = 2.0
    decide(market, trader, 3.0)  # its position 1 is inside the band of the aim 2: no need; the paper holds 2
    assert not any(market._working) and not work_of(market, trader).clock
    assert account.asked.tolist() == [2, 0] and trader.position.tolist() == [1, 0]
    trader.active = False
    market.time = 4.0
    market.step()
    assert account.asked.tolist() == [1, 0] and account.paper_cash == pytest.approx(-2 * 3000.0 + 2999.5)


def test_a_need_that_a_thin_book_lets_cross_twice_is_one_need_crossed_and_counted_once():
    market, _, trader = worked([3, 0], depth=2, lots=1, patience=5.0)
    decide(market, trader, 1.0)
    decide(market, trader, 6.0)  # the patience has run: 3 lots wanted, 2 to buy at 3001 and 3002, 1 left
    account = market.account(trader)
    assert (account.orders_crossed, account.crossed_lots) == (1, 2) and work_of(market, trader).left.tolist() == [1, 0]
    rest(market, 0, ASK, 3005, 1)
    decide(market, trader, 11.0)  # and the last lot is crossed at the next decision: the same need
    assert trader.position.tolist() == [3, 0] and account.crossed_lots == 3
    assert account.orders_crossed == 1 and len(account.waits) == 1  # one need crossed, and one need ended
    rest(market, 0, ASK, 3006, 1)
    trader.goal = np.array([4, 0])
    decide(market, trader, 12.0)  # a need that comes after is a need of its own, with a clock of its own
    decide(market, trader, 18.0)
    assert account.orders_crossed == 2 and trader.position.tolist() == [4, 0]


def alone_with_a_mark_of_its_own(patience=50.0):
    """A bid for 1 behind 10 at 2999 (asks 3001); a model's bid of 1 at 3000 moves the mid to 3000.5; the models then
    sell the 1 at 3000 and the 10 ahead, so that the trader's bid is all the bids, and then have the asks bought and
    one placed at 3009. The market, the trader, and the trader's mid, the market's mark and the trader's mark after the
    bids go and after the asks do."""
    market, _, trader = worked([1, 0], patience=patience, depth=1, lots=10, interval=1.0)
    decide(market, trader, 1.0)
    market.time = 1.2
    market.submit(0, limit_order(3000, 1, 1.2, BID), impact=False)
    assert market.mid_for(trader, 0) == 3000.5
    market.time = 1.5
    hit(market, 0, ASK, 11)  # the 1 at 3000 and the 10 ahead of the trader: its bid is all the bids, it has no mid
    first = (market.mid_for(trader, 0), market.mark(0), float(market.marks_for(trader)[0]))
    market.time = 1.6
    hit(market, 0, BID, 10)  # the asks go, and a model's ask at 3009 is all the asks there are
    market.submit(0, limit_order(3009, 5, 1.7, ASK), impact=False)
    second = (market.mid_for(trader, 0), market.mark(0), float(market.marks_for(trader)[0]))
    return market, trader, first, second


def test_the_marks_of_a_trader_whose_mid_is_none_are_its_last_mid_without_its_own_orders_and_not_the_markets_last():
    market, trader, first, second = alone_with_a_mark_of_its_own()
    assert first == (None, 3000.0, 3000.5)  # the market's last mid is 2999 and 3001 over two: made of its own bid
    assert second == (None, 3004.0, 3000.5)  # (2999 + 3009) / 2: made of its own bid; the trader's last was 3000.5
    assert market.marks_for(trader)[1] == 3000.0  # a book it has no order in is the market's own


def test_a_trader_stopped_with_its_order_alone_gives_up_what_it_did_not_get_at_the_last_mid_without_its_own_order():
    market, trader, _, second = alone_with_a_mark_of_its_own()
    market.keep_decisions = True
    trader.active = False
    market.time = 2.0
    market.step()  # its bid goes: the bids are none, and the market's last mid (3004) is made of that bid
    assert not any(market._working) and market.books[0].mid_price is None and market.mark(0) == 3004.0
    account = market.account(trader)
    assert account.asked.tolist() == [0, 0] and account.paper_cash == 0.0 - 3000.0 + 3000.5  # at the mid it had


def test_the_mid_a_markout_is_taken_at_is_the_traders_last_without_its_own_orders_where_it_has_none():
    market, _, trader = worked([2, 0], patience=100.0)
    decide(market, trader, 1.0)
    trader.next_time = 1e9
    hit(market, 0, ASK, 11)  # the ten ahead and one of its 2, at 2999; the other is alone at 2999: the mid is 2999.5
    account = market.account(trader)
    assert market.mid_for(trader, 0) == 2999.5 and account.fills[0][2:4] == (1, 2999)
    for price in (2998, 2997):
        clear(market, 0, BID, price)  # now its bid is all the bids: no mid without it; the book's own is 3000
    assert market.mid_for(trader, 0) is None and market.books[0].mid_price == 3000.0
    step_to(market, 2.0)
    assert account.markouts == [(0, 1.0, 2999.5 - 2999)]  # not 3000.0 less 2999


def test_a_frame_of_run_values_a_worked_trader_at_the_mids_without_its_own_orders_and_one_with_none_as_before():
    market, _, trader = worked([2, 0], patience=50.0, interval=100.0)
    bystander = Wants(goal=np.array([0, 0]), interval=100.0)
    market.add(bystander)
    decide(market, trader, 1.0)
    market.time = 1.5
    hit(market, 0, ASK, 11)  # 10 ahead and one of its 2: the other is alone at 2999, book 0's own mid is 3000
    assert trader.position.tolist() == [1, 0] and market.marks().tolist()[0] == 3000.0
    frame = market.run(1.7, every=0.1)
    assert len(frame) >= 1 and frame["value_0"].iloc[-1] == -2999 + 2999.5  # its mark leaves its own order out
    assert frame["value_1"].iloc[-1] == 0.0


def test_the_market_a_trader_observes_answers_mid_mark_and_marks_without_its_own_orders():
    market, _, trader = worked([3, 0], look=lambda m: (m.mid(0), m.mark(0), list(m.marks()), m.books[0].mid_price),
                               reprice=100.0)
    decide(market, trader, 1.0)
    clear(market, 0, BID, 2999, keep=[working(market, trader, 0)[2]])  # its bid alone at 2999
    assert market.mid(0) == 3000.0
    decide(market, trader, 2.0)
    mid, mark, marks, own_mid = trader.seen[-1]
    assert (mid, mark, own_mid) == (2999.5, 2999.5, 2999.5) and marks == [2999.5, 3000.0]
    assert market.mid(0) == 3000.0 and list(market.marks()) == [3000.0, 3000.0]  # the market's own are as they were


def test_a_need_that_grows_keeps_its_first_clock_so_the_lots_asked_late_are_crossed_before_their_own_patience():
    # the clock of a need is when the need began, not when each lot was asked: at a patience of 4 the lot asked at 5
    # is crossed with the one asked at 1. The mean age of the 5 lots at the crossing is 2 units, the need's is 4.
    market, _, trader = worked([0, 0], patience=4.0, interval=1.0)
    crossed = []
    for k in range(5):
        trader.goal = np.array([k + 1, 0])
        decide(market, trader, k + 1.0)
        crossed.append(market.account(trader).crossed_lots)
    account = market.account(trader)
    assert crossed == [0, 0, 0, 0, 5] and {fill[0] for fill in account.fills} == {5.0}
    assert account.waits == [(5.0, 4.0)] and account.orders_crossed == 1


def test_a_need_that_goes_to_nothing_and_comes_back_starts_a_clock_of_its_own_so_its_patience_may_never_run():
    market, _, trader = worked([0, 0], patience=4.0, interval=1.0)
    for at, goal in ((1.0, 1), (2.0, 0), (3.0, 1), (4.0, 0), (5.0, 1)):
        trader.goal = np.array([goal, 0])
        decide(market, trader, at)
    account = market.account(trader)
    assert clock_of(market, trader, 0) == (5.0, BID) and account.orders_crossed == 0 and account.crossed_lots == 0


# ---- Real order flow: the accounts add up ----

@dataclass
class Drifter(BookTrader):
    """No signal: every ``every`` units it adds 1 or -1 lot, at random, to its goal in each book, and asks the gap."""
    seed: int = 0
    every: float = 25.0
    goal: Optional[np.ndarray] = None
    goals: list = field(default_factory=list)

    def attach(self, n):
        super().attach(n)
        self._rng = np.random.default_rng([self.seed, 777])
        self.goal, self._next = np.zeros(n, dtype=int), 0.0
        self._now = 0.0

    def observe(self, market):
        self._now = market.time

    def decide(self, mids):
        if self._now >= self._next:
            self.goal = self.goal + self._rng.choice([-1, 1], size=len(mids))
            self._next += self.every
        self.goals.append(self.goal.copy())
        gap = self.goal - self.position
        return [gap] if gap.any() else []


def assert_orders_match_needs(market):
    """Whenever a working order rests, the book has it, and what is left of its need is its size, on its side."""
    resting = 0
    for work in market._works.values():
        for book, order_id in work.order.items():
            order = market.books[book].get_order(order_id)
            assert order is not None and market._working[book][order_id] is work
            assert order.size == abs(work.left[book]) and (order.side == BID) == (work.left[book] > 0)
            assert work.clock[book][1] == order.side
            resting += 1
    assert resting == sum(len(orders) for orders in market._working)


def test_the_accounts_of_traders_working_orders_in_the_real_order_flow_add_up_and_their_paper_accounts_replay():
    market = PassiveBookMarket(FactorValues(sector_loadings(2, 3)), seed=3)
    market.keep_decisions = True
    slow, quick, crossing = Drifter(seed=1, interval=5.0), Drifter(seed=2, interval=1.0), Drifter(seed=3, interval=5.0)
    market.add(slow, Execution(20.0, reprice=1.0))
    market.add(quick, Execution(4.0, reprice=None))
    market.add(crossing)
    named, dropped, submit = [], [], market.submit

    def spy(book, order, impact=True):
        """The cancels that name a working order, and the ones that counted no event."""
        if order.order_type == OrderType.CANCEL and any(order.order_id in work.order.values()
                                                       for work in market._works.values()):
            named.append(order.order_id)
        events, fills = market.events, submit(book, order, impact)
        if order.order_type == OrderType.CANCEL and market.events == events:
            dropped.append(order.order_id)
        return fills

    market.submit = spy
    while market.time < 400.0:
        market.step()
        assert_orders_match_needs(market)
    for trader in (slow, quick, crossing):
        adds_up(market, trader)
        account = market.account(trader)
        goals = trader.goals  # a decision's goal is what was asked: the position, and the need that is its gap to it
        assert len(account.decisions) == len(goals) > 50
        assert all((row[2] == goal).all() for row, goal in zip(account.decisions, goals, strict=True))
        # a replay, written apart: the paper cash is what asking each change of goal at that decision's mids paid
        previous, cash, lots = np.zeros(market.n, dtype=int), 0.0, 0
        for (_, mids, _), goal in zip(account.decisions, goals, strict=True):
            cash -= float((goal - previous) @ mids)
            lots += int(np.abs(goal - previous).sum())
            previous = goal
        assert account.paper_cash == pytest.approx(cash) and account.asked_lots == lots
        assert (account.asked == goals[-1]).all()
    assert market.account(slow).passive_lots > 50 and market.account(quick).passive_lots > 50
    assert market.account(slow).orders_crossed > 0 or market.account(quick).orders_crossed > 0
    assert market.account(crossing).passive_lots == 0 and market.account(crossing).orders_placed == 0
    assert len(dropped) > 20 and dropped == named  # the model's draws named its orders: all dropped, and no other


def run_with_orders(market, until):
    """Step to ``until``, checking after each step that every working order is what is left of its need."""
    while market.time < until:
        market.step()
        assert_orders_match_needs(market)


@pytest.mark.parametrize("kind", [name for name in KINDS if name != "pairs not netted"])
def test_every_kind_of_trader_works_its_orders_in_the_real_order_flow_and_its_account_adds_up(kind):
    market = PassiveBookMarket(FactorValues(SECTORS), seed=4)
    trader = KINDS[kind](statarb_now, cross_now)
    market.add(trader, Execution(10.0, reprice=2.0))
    run_with_orders(market, trader.warm_up + 251.0)
    account = market.account(trader)
    adds_up(market, trader)
    assert account.orders_placed > 0 and account.passive_lots + account.crossed_lots == trader.lots > 0
    assert account.asked_lots > 0 and account.waits  # it worked orders and some needs were met


def test_a_deal_trader_works_its_orders_through_the_announcements_and_breaks_of_deals_and_its_account_adds_up():
    fair = FactorValues(sector_loadings(2, 2), deals=Deals(**BUSY))
    market = PassiveBookMarket(fair, seed=7)
    trader = DealTrader(entry=8.0, size=5)
    market.add(trader, Execution(10.0, reprice=2.0))
    run_with_orders(market, 400.0)
    account = market.account(trader)
    adds_up(market, trader)
    assert fair.ended and trader.outcomes  # deals came and went, and it held and closed some
    assert account.orders_placed > 0 and account.orders_withdrawn > 0 and trader.lots > 20


def shortfall_a_lot(market, trader):
    """What the trader's execution cost it a lot of what it asked: its paper P&L less its P&L, over the lots asked."""
    account, marks = market.account(trader), market.marks_for(trader)
    paper = account.paper_cash + float(account.asked @ marks)
    pnl = trader.cash + float(trader.position @ marks)
    return (paper - pnl) / account.asked_lots


def test_every_markout_in_the_real_flow_is_the_makers_sign_times_the_traders_mark_at_the_first_step_after_its_time():
    market = PassiveBookMarket(FactorValues(sector_loadings(1, 3)), seed=2)
    trader = Drifter(seed=2, interval=2.0, every=5.0)
    market.add(trader, Execution(20.0, reprice=1.0))
    seen = []  # the time of each step and the trader's marks at its end, as the test sees them
    while market.time < 300.0:
        market.step()
        seen.append((market.time, market.marks_for(trader)))
    account = market.account(trader)
    times = [time for time, _ in seen]
    passive = [row for row, fill in enumerate(account.fills) if fill[5] in ("market", "limit")]
    taken = {(row, horizon): value for row, horizon, value in account.markouts}
    assert len(passive) > 100 and len(taken) == len(account.markouts) and all(row in passive for row, _ in taken)
    waiting = 0
    for row in passive:
        time, book, lots, price = account.fills[row][:4]
        for horizon in (1.0, 5.0, 20.0):
            step = bisect.bisect_left(times, time + horizon)  # the first step at or after the time
            if step == len(times):
                assert (row, horizon) not in taken
                waiting += 1
            else:
                assert taken[(row, horizon)] == pytest.approx((1 if lots > 0 else -1) * (seen[step][1][book] - price))
    assert waiting > 0 and waiting == sum(len(queue) for queue in market._pending)  # the rest are still to be taken
    assert len(taken) + waiting == 3 * len(passive)
    assert {fill[5] for fill in account.fills} >= {"market", "limit"}  # both kinds of order hit it in 300 units


def paper_replayed(rows):
    """The paper account made again from the decisions a market kept, ``(time, mids, asked)`` each, and nothing else:
    ``(time, cash, held, lots)`` after each, where it asked each change of what it held at that decision's mids."""
    cash, held, lots, out = 0.0, np.zeros(len(rows[0][1]), dtype=int), 0, []
    for time, mids, asked in rows:
        change = np.asarray(asked) - held
        cash -= float(sum(int(c) * float(m) for c, m in zip(change, mids, strict=True)))
        lots += int(sum(abs(int(c)) for c in change))
        held = np.asarray(asked)
        out.append((time, cash, held, lots))
    return out


def test_the_paper_account_is_replayed_from_the_decisions_the_market_kept_and_score_execution_agrees_over_a_span():
    market = PassiveBookMarket(FactorValues(sector_loadings(3, 4)), seed=5)
    market.keep_decisions = True
    walkers = {"worked": Drifter(seed=5, interval=5.0), "repriced": Drifter(seed=6, interval=1.0),
               "crossing": Drifter(seed=7, interval=5.0)}
    market.add(walkers["worked"], Execution(20.0, reprice=None))
    market.add(walkers["repriced"], Execution(4.0, reprice=1.0))
    market.add(walkers["crossing"])
    while market.time < 150.0:
        market.step()
    for name, trader in walkers.items():  # each is scored over a span of its own, one after the other
        start = market.time
        cash0, position0, marks0 = trader.cash, trader.position.copy(), market.marks_for(trader).copy()
        score = score_execution(market, trader, start, 150.0)
        account = market.account(trader)
        rows = account.decisions
        times = [row[0] for row in rows]
        assert times == sorted(times) and len(set(times)) == len(times) > 50  # a decision each, in order
        replay = paper_replayed(rows)
        assert replay[-1][1] == pytest.approx(account.paper_cash) and replay[-1][3] == account.asked_lots
        assert (replay[-1][2] == account.asked).all()
        # the span's paper, from the replay: what it is worth at the end less at the start, at the trader's own marks
        _, cash_start, held_start, lots_start = [row for row in replay if row[0] <= start][-1]
        _, cash_end, held_end, lots_end = replay[-1]
        marks1 = market.marks_for(trader)
        paper = (cash_end + float(held_end @ marks1)) - (cash_start + float(held_start @ marks0))
        pnl = (trader.cash + float(trader.position @ marks1)) - (cash0 + float(position0 @ marks0))
        assert lots_end - lots_start > 50 and score["asked_lots"] == lots_end - lots_start, name
        assert score["paper"] == pytest.approx(paper) and score["pnl"] == pytest.approx(pnl), name
        assert score["shortfall"] == pytest.approx(paper - pnl), name
        assert score["shortfall_a_lot"] == pytest.approx((paper - pnl) / (lots_end - lots_start)), name
        adds_up(market, trader)


def test_a_worked_trader_has_a_shortfall_a_lot_more_than_half_a_tick_below_its_market_order_twins_in_score_execution():
    scores = {}
    for name, execution in (("worked", Execution(20.0, reprice=None)), ("crossed", None)):
        market = PassiveBookMarket(FactorValues(sector_loadings(3, 4)), seed=1)
        trader = Drifter(seed=1, interval=5.0)
        market.add(trader, execution)
        scores[name] = score_execution(market, trader, 0.0, 1000.0)  # from the start, so the account is the span's
        assert scores[name]["shortfall_a_lot"] == pytest.approx(shortfall_a_lot(market, trader))  # the by-hand figure
    worked_, crossed = scores["worked"], scores["crossed"]
    assert 0.8 < crossed["shortfall_a_lot"] < 1.2  # a market order costs about the half spread, 1 tick a lot
    assert worked_["shortfall_a_lot"] < crossed["shortfall_a_lot"] - 0.5
    assert crossed["passive_lots"] == 0 and crossed["crossed_lots"] == crossed["asked_lots"] > 400
    assert worked_["passive_lots"] > 10 * worked_["crossed_lots"] and worked_["earned"] > 0 and worked_["wait"] > 0
    assert worked_["asked_lots"] == crossed["asked_lots"] and worked_["costs"] < crossed["costs"] / 10


# ---- What a shortfall is made of: the lots got, taken back, traded unasked, and open ----
# Each by a hand-made book: a book is 2997 to 2999 bids and 3001 to 3003 asks, 10 lots at each, the mid 3000.

NO_FILLS = {"market": 0, "limit": 0, "cross": 0, "urgent": 0}  # the lots got by each kind of fill, none
NO_COST = {"market": 0.0, "limit": 0.0, "cross": 0.0, "urgent": 0.0}


def started(goal, **kwargs):
    """``worked`` with the market keeping its decisions and the start of the span noted at time 0: the market, the
    trader, the start."""
    market, _, trader = worked(goal, **kwargs)
    market.keep_decisions = True
    return market, trader, market.since(trader)


def noted(market, trader):
    """What the trader and its paper account hold now, noted apart from ``market.since``: position, cash, what was
    asked, the paper's cash and the trader's marks."""
    account = market.account(trader)
    return (trader.position.copy(), trader.cash, account.asked.copy(), account.paper_cash,
            market.marks_for(trader).copy())


def shortfall_since(market, trader, start):
    """Paper P&L less P&L from a start noted by ``noted`` to now, each at the trader's marks: written apart from
    ``shortfall_parts``."""
    position, cash, asked, paper_cash, marks0 = start
    account, marks = market.account(trader), market.marks_for(trader)
    paper = (account.paper_cash - paper_cash) + float(account.asked @ marks) - float(asked @ marks0)
    pnl = (trader.cash - cash) + float(trader.position @ marks) - float(position @ marks0)
    return paper - pnl


def listed(parts):
    """The lots, with the closing time of an open lot (nan, which is not equal to itself) as None."""
    return [lot._replace(closed=None) if math.isnan(lot.closed) else lot for lot in parts.lots]


def test_a_lot_asked_and_filled_by_its_resting_order_is_got_at_the_fills_price_less_the_decisions_mid():
    market, trader, since = started([1, 0], patience=50.0)
    decide(market, trader, 1.0)  # asked at the mid, 3000: a bid for 1 at 2999 behind 10
    hit(market, 0, ASK, 11)  # the 10 ahead of it, and its lot, at 2999
    parts = shortfall_parts(market, trader, since)
    assert parts.lots == [Lot("got", 0, 1, -1.0, "ask", "market", 1.0, 1.0)]  # 2999 less 3000
    assert (parts.got_lots, parts.got_cost, parts.got_a_lot) == (1, -1.0, -1.0)
    assert (parts.back_lots, parts.unasked_lots, parts.open_lots) == (0, 0, 0)
    assert parts.got_lots_by_fill == NO_FILLS | {"market": 1} and parts.got_cost_by_fill == NO_COST | {"market": -1.0}
    # by hand: the book's mid is 2999.5 now (2998 and 3001): the paper made -3000 + 2999.5, the trader -2999 + 2999.5
    assert parts.shortfall == -1.0 == parts.total and parts.residual == 0.0 and parts.notes == []


def test_a_lot_asked_and_crossed_when_its_patience_ran_out_is_got_at_the_price_it_crossed_at():
    market, trader, since = started([1, 0], patience=2.0)
    decide(market, trader, 1.0)  # a bid for 1 at 2999, asked at 3000
    decide(market, trader, 3.0)  # 2 units on: withdrawn, and 1 bought at the best ask, 3001, a cost of 1
    parts = shortfall_parts(market, trader, since)
    assert parts.lots == [Lot("got", 0, 1, 1.0, "ask", "cross", 1.0, 3.0)]
    assert parts.got_lots_by_fill == NO_FILLS | {"cross": 1} and parts.got_cost_by_fill == NO_COST | {"cross": 1.0}
    # by hand: the mid is 3000 still (9 lots are left at 3001): the paper made -3000 + 3000, the trader -3001 + 3000
    assert parts.shortfall == 1.0 == parts.total and parts.back_lots == parts.open_lots == 0


def test_a_lot_asked_and_taken_back_by_a_later_ask_before_any_fill_is_what_the_paper_made_on_it():
    market, trader, since = started([1, 0], patience=50.0)
    decide(market, trader, 1.0)  # asked at 3000, a bid resting
    market._shift(0, 4)  # a jump of 4 ticks: its bid is withdrawn, and the mid is 3004
    trader.goal = np.array([0, 0])
    decide(market, trader, 2.0)  # it asks for nothing: the lot is taken back at 3004, never got
    parts = shortfall_parts(market, trader, since)
    assert parts.lots == [Lot("back", 0, 1, 4.0, "ask", "", 1.0, 2.0)]
    assert (parts.back_lots, parts.back_drift) == (1, 4.0) and parts.got_lots == parts.open_lots == 0
    # by hand: the paper bought at 3000 and sold at 3004, 4 made; the trader did nothing: 4 of shortfall
    assert trader.position.tolist() == [0, 0] and parts.shortfall == 4.0 == parts.total


def test_a_lot_of_the_gap_left_at_the_end_is_open_at_the_traders_marks_then():
    market, trader, since = started([2, 0], patience=50.0)
    decide(market, trader, 1.0)  # asked 2 at 3000, a bid for 2 at 2999 behind 10
    hit(market, 0, ASK, 11)  # the 10 ahead and 1 lot of its 2, at 2999; its other lot is alone at 2999
    parts = shortfall_parts(market, trader, since)
    assert market.marks_for(trader).tolist() == [2999.5, 3000.0]  # its own order is left out: 2998 and 3001
    assert listed(parts) == [Lot("got", 0, 1, -1.0, "ask", "market", 1.0, 1.0),
                             Lot("open", 0, 1, -0.5, "ask", "", 1.0, None)]  # an open lot has no closing time
    assert (parts.got_lots, parts.got_cost, parts.open_lots, parts.open_drift) == (1, -1.0, 1, -0.5)
    # by hand: the paper made -6000 + 2 x 2999.5 = -1, the trader -2999 + 2999.5 = 0.5: -1.5 = -1 (got) - 0.5 (open)
    assert parts.shortfall == -1.5 == parts.total


def test_a_gap_carried_in_at_the_start_of_the_span_is_closed_by_a_fill_at_the_traders_marks_then():
    market, _, trader = worked([2, 0], patience=50.0)
    market.keep_decisions = True
    decide(market, trader, 1.0)  # asked 2 at 3000
    hit(market, 0, ASK, 11)  # 1 lot of the 2 at 2999: before the span, so that its cost is not in it
    since = market.since(trader)
    assert since.position.tolist() == [1, 0] and since.asked.tolist() == [2, 0]
    assert since.marks.tolist() == [2999.5, 3000.0]
    hit(market, 0, ASK, 1)  # in the span: its other lot, at 2999, now first in the queue
    parts = shortfall_parts(market, trader, since)
    # the lot of the gap is valued at the marks of the start, 2999.5 (its own order left out), not at the 3000 it was
    # asked at: a cost of 2999 less 2999.5
    assert parts.lots == [Lot("got", 0, 1, -0.5, "carried", "market", 1.0, 1.0)]
    # by hand: the paper's 2 lots are 2 x 2999.5 at both ends, its cash unchanged; the trader's cash fell by 2999, and
    # its 2 lots (from 1) are worth 2999.5 more: -2999 + 2999.5 = 0.5 made, the shortfall -0.5
    assert parts.shortfall == -0.5 == parts.total and market.marks_for(trader).tolist() == [2999.5, 3000.0]
    assert shortfall_parts(market, trader, market.since(trader)).lots == []  # a span with nothing in it has no lots


def test_the_lots_of_two_books_are_told_apart_first_in_first_out_in_each_book():
    market, trader, since = started([1, -1], patience=50.0)
    decide(market, trader, 1.0)  # a bid for 1 in book 0 at 2999, an offer for 1 in book 1 at 3001: asked at 3000
    hit(market, 1, BID, 11)  # the 10 ahead of its offer and its lot: it sold 1 at 3001, a cost of -1 (3001 above 3000)
    market._shift(0, 2)  # book 0 jumps 2 ticks: its bid is withdrawn, the mid is 3002
    trader.goal = np.array([0, -1])
    decide(market, trader, 2.0)  # book 0's lot is taken back at 3002: it asks for nothing there
    parts = shortfall_parts(market, trader, since)
    assert listed(parts) == [Lot("got", 1, -1, -1.0, "ask", "market", 1.0, 1.0),
                             Lot("back", 0, 1, 2.0, "ask", "", 1.0, 2.0)]  # book 1's lot is not book 0's to close
    # by hand: paper cash 0 (book 0 bought at 3000, book 1 sold at 3000) then +3002; it holds -1 of book 1, worth
    # 3000.5 (2999 and 3002 now): 1.5. The trader sold 1 at 3001: 3001 - 3000.5 = 0.5 made: 1.5 - 0.5 = 1 = 2 - 1
    assert parts.shortfall == 1.0 == parts.total


def test_the_lots_a_band_keeps_the_position_short_of_are_open_until_the_paper_takes_them_back():
    market, _ = hand_market()
    market.keep_decisions = True
    trader = Banded(band=1.0, interval=5.0)
    market.add(trader, Execution(1000.0, reprice=None))
    since = market.since(trader)
    trader.size = 3.0
    decide(market, trader, 1.0)  # an aim of 3 and a band of 1: it asks for 2, at 3000
    market._shift(0, 10)  # the mid is 3010
    trader.size = 2.0
    decide(market, trader, 2.0)  # an aim of 2 is inside the band of the 2 it holds on paper: it asks for nothing
    trader.size = 4.0
    decide(market, trader, 3.0)  # an aim of 4 is 2 away from 2: it asks for 1 more, at 3010
    account = market.account(trader)
    assert account.asked.tolist() == [3, 0] and trader.position.tolist() == [0, 0]  # nothing filled: short of the paper
    parts = shortfall_parts(market, trader, since)
    # open: the 2 lots asked at 3000 are 10 each above it, the one asked at 3010 is not: 20
    assert (parts.open_lots, parts.open_drift, parts.got_lots, parts.back_lots) == (3, 20.0, 0, 0)
    assert parts.shortfall == 20.0 == parts.total  # by hand: -6000 - 3010 + 3 x 3010 = 20, and the trader made 0
    trader.size = 0.0
    decide(market, trader, 4.0)  # all of it is asked back, at 3010: the oldest first, 10 + 10 + 0
    parts = shortfall_parts(market, trader, since)
    assert (parts.open_lots, parts.open_drift, parts.back_lots, parts.back_drift) == (0, 0.0, 3, 20.0)
    assert [lot.cost for lot in parts.lots] == [10.0, 10.0, 0.0]
    assert parts.shortfall == 20.0 == parts.total and account.asked.tolist() == [0, 0]


def test_a_stopped_traders_lots_it_did_not_get_are_taken_back_at_the_stop_and_the_price_after_it_is_not_in_them():
    market, _, trader = worked([3, 0], patience=50.0)
    market.keep_decisions = True
    since = market.since(trader)
    decide(market, trader, 1.0)  # asked 3 at 3000
    hit(market, 0, ASK, 11)  # 1 of its 3 at 2999: its other 2 rest alone at 2999 (the mid without them 2999.5)
    trader.active = False
    market.time = 2.0
    market.step()  # the stop: the 2 not got are given up on paper, at 2999.5
    parts = shortfall_parts(market, trader, since)
    assert [(lot.kind, lot.cost) for lot in parts.lots] == [("got", -1.0), ("back", -0.5), ("back", -0.5)]
    assert (parts.got_lots, parts.back_lots, parts.open_lots) == (1, 2, 0) and parts.back_drift == -1.0
    assert parts.shortfall == -2.0 == parts.total  # by hand: paper -9000 + 2 x 2999.5 + 2999.5 less -2999 + 2999.5
    market._shift(0, 5)
    market._shift(0, 5)  # the price goes on: it asks for nothing, and its shortfall does not move
    again = shortfall_parts(market, trader, since)
    assert (again.got_cost, again.back_drift, again.open_lots, again.shortfall) == (-1.0, -1.0, 0, -2.0)


def test_a_partial_fill_gets_some_lots_of_an_order_and_the_rest_are_open_until_they_are_crossed():
    market, trader, since = started([3, 0], patience=5.0)
    decide(market, trader, 1.0)  # a bid for 3 at 2999 behind 10
    hit(market, 0, ASK, 12)  # the 10 ahead and 2 of its 3 lots, at 2999: its third rests alone
    parts = shortfall_parts(market, trader, since)
    assert (parts.got_lots, parts.got_cost, parts.open_lots, parts.open_drift) == (2, -2.0, 1, -0.5)  # marks 2999.5
    # by hand: the paper -9000 + 3 x 2999.5 = -1.5, the trader -2 x 2999 + 2 x 2999.5 = 1: -2.5 = -2 - 0.5
    assert parts.shortfall == -2.5 == parts.total
    decide(market, trader, 6.0)  # 5 units on: its third lot is crossed at 3001, a cost of 1
    parts = shortfall_parts(market, trader, since)
    assert (parts.got_lots, parts.got_cost, parts.open_lots) == (3, -1.0, 0)
    assert parts.got_lots_by_fill == NO_FILLS | {"market": 2, "cross": 1}
    assert parts.got_cost_by_fill == NO_COST | {"market": -2.0, "cross": 1.0}
    assert parts.shortfall == -1.0 == parts.total  # by hand: paper -1.5 (marks 2999.5) less the trader's -0.5


def test_the_cost_of_the_lots_got_is_told_by_the_kind_of_fill_that_got_them_market_limit_and_cross():
    market, trader, since = started([4, 0], patience=5.0)
    decide(market, trader, 1.0)  # a bid for 4 at 2999 behind 10
    hit(market, 0, ASK, 11)  # a market sell: the 10 ahead, 1 lot of its 4, at 2999
    hit(market, 0, ASK, 1, kind=OrderType.LIMIT, price=2999)  # a limit sell at 2999: a second lot
    decide(market, trader, 6.0)  # 5 units on: the other 2 are crossed at 3001
    parts = shortfall_parts(market, trader, since)
    assert parts.got_lots_by_fill == {"market": 1, "limit": 1, "cross": 2, "urgent": 0}
    assert parts.got_cost_by_fill == {"market": -1.0, "limit": -1.0, "cross": 2.0, "urgent": 0.0}  # 1 each at 3001
    assert parts.got_cost == 0.0 == parts.shortfall == parts.total and parts.got_lots == 4


def test_the_lots_a_trader_with_no_relations_closes_at_once_are_got_by_urgent_fills():
    market, models, trader = switch([3, -2])
    market.keep_decisions = True
    since = market.since(trader)
    decide(market, trader, 1.0)  # asked 3 in book 0 and -2 in book 1, at 3000
    play(market, models, {0: [market_order(11, 0.0, ASK)]})  # 1 of its 3 bought at 2999; 2 rest alone at 2999
    trader.closing = True
    decide(market, trader, 2.0)  # it asks for nothing: 2 lots taken back at 2999.5, and 1 sold at 2998, urgent
    parts = shortfall_parts(market, trader, since)
    assert parts.got_lots_by_fill == NO_FILLS | {"market": 1, "urgent": 1}
    # the lot bought is -1 (2999 less 3000); the urgent sale is of a lot the paper asked to be sold, 3 asked and 1 held:
    # -1 x (2998 - 2999.5); of book 0's 3 asked: 1 got, 2 taken back at 2999.5 (-0.5 each); book 1's 2 offers: 0
    assert parts.got_cost_by_fill == NO_COST | {"market": -1.0, "urgent": 1.5}
    assert (parts.back_lots, parts.back_drift, parts.open_lots) == (4, -1.0, 0)
    # by hand: the paper -9000 + 3 x 2999.5 and 6000 - 2 x 3000 = -1.5; the trader bought at 2999 and sold at 2998: -1
    assert parts.shortfall == -0.5 == parts.total


def test_a_twin_that_crosses_every_lot_at_once_gets_each_for_its_spread_paid():
    market, _ = hand_market()
    market.keep_decisions = True
    twin = Wants(goal=np.array([12, -2]), interval=5.0)  # 10 lots at 3001 and 2 at 3002; 2 lots at 2999
    market.add(twin)
    since = market.since(twin)
    decide(market, twin, 1.0)
    parts = shortfall_parts(market, twin, since)
    assert parts.got_lots == 14 and parts.got_cost == 16.0 == twin.spread_paid  # 10 x 1 + 2 x 2, and 2 x 1
    assert parts.got_lots_by_fill == NO_FILLS | {"cross": 14} and parts.got_cost_by_fill == NO_COST | {"cross": 16.0}
    assert parts.back_lots == parts.unasked_lots == parts.open_lots == 0
    assert parts.shortfall == 16.0 == parts.total  # by hand: the paper's cash less the trader's, -30000 + 30016


def test_a_lot_the_trader_traded_and_traded_back_with_no_ask_between_is_unasked_and_the_notes_say_so():
    market, _ = hand_market()
    market.keep_decisions = True
    trader = Wants(goal=np.array([0, 0]), interval=5.0)
    market.add(trader)
    since = market.since(trader)
    account = market.account(trader)
    market.time = 1.0
    market._cross(trader, account, 0, 1, "cross")  # bought at 3001 with nothing asked: ahead of the paper
    market.time = 2.0
    market._cross(trader, account, 0, -1, "cross")  # and sold at 2999: its round trip, 2 ticks
    parts = shortfall_parts(market, trader, since)
    assert parts.lots == [Lot("unasked", 0, -1, 2.0, "fill", "", 1.0, 2.0)]
    assert (parts.unasked_lots, parts.unasked_cost, parts.got_lots) == (1, 2.0, 0)
    assert parts.shortfall == 2.0 == parts.total  # by hand: the paper asked for nothing, the trader lost 3001 - 2999
    assert len(parts.notes) == 1 and "traded back with no ask between" in parts.notes[0] and "2.0" in parts.notes[0]


def test_a_lot_the_trader_traded_before_the_paper_asked_for_it_is_got_when_it_is_asked():
    market, _ = hand_market()
    market.keep_decisions = True
    trader = Wants(goal=np.array([1, 0]), interval=5.0)
    market.add(trader)
    since = market.since(trader)
    account = market.account(trader)
    market.time = 1.0
    market._cross(trader, account, 0, 1, "cross")  # bought at 3001, nothing asked yet
    decide(market, trader, 2.0)  # the paper asks for 1 at 3000: the lot it holds is got, 3001 less 3000
    parts = shortfall_parts(market, trader, since)
    assert parts.lots == [Lot("got", 0, -1, 1.0, "fill", "cross", 1.0, 2.0)]
    assert parts.got_lots_by_fill == NO_FILLS | {"cross": 1} and parts.unasked_lots == parts.open_lots == 0
    assert parts.shortfall == 1.0 == parts.total and parts.notes == []  # by hand: the paper 0, the trader -3001 + 3000


# ---- The order of events: a fill of a step is before a decision of that step or after it, as it happened ----

def test_a_fill_that_came_before_a_decision_of_the_same_time_is_got_before_the_decision_asks_what_it_asks():
    market, trader, since = started([1, 0], patience=50.0)
    decide(market, trader, 1.0)  # asked 1 at 3000, a bid for 1 at 2999
    market.time = 2.0
    hit(market, 0, ASK, 11)  # a model's order at 2.0 fills it ...
    trader.goal = np.array([0, 0])
    decide(market, trader, 2.0)  # ... and then it decides to hold nothing: an ask of -1, of a lot it already got
    account = market.account(trader)
    assert account.fills_before == [0, 1]  # the second decision came after the fill, at the same time
    parts = shortfall_parts(market, trader, since)
    # got first (2999 less 3000); then a lot the paper asked to sell while the trader held 1: open at 2999.5 less the
    # 2999.5 it was asked at. Had the decision come first, the lot would be taken back (a drift of -0.5), the fill
    # would be a lot traded ahead of the paper, and nothing would be got
    assert listed(parts) == [Lot("got", 0, 1, -1.0, "ask", "market", 1.0, 2.0),
                             Lot("open", 0, -1, 0.0, "ask", "", 2.0, None)]
    assert parts.shortfall == -1.0 == parts.total  # by hand: the paper -3000 + 2999.5, the trader -2999 + 2999.5


def test_a_fill_by_another_traders_order_after_this_traders_decision_of_the_same_time_comes_after_it():
    market, _, trader = worked([1, 0], patience=50.0)
    market.keep_decisions = True
    other = Wants(goal=np.array([0, 0]), interval=5.0)
    market.add(other)  # no execution: it sells with market orders; it decides after the trader at each time
    since = market.since(trader)
    decide(market, trader, 1.0)  # asked 1 at 3000, a bid at 2999
    market._shift(0, 10)  # its bid is withdrawn; the mid is 3010
    trader.goal = np.array([2, 0])
    decide(market, trader, 2.0)  # 1 more asked, at 3010: a bid for 2 at 3009 behind 10
    trader.goal = np.array([1, 0])
    other.goal = np.array([-11, 0])
    decide(market, trader, 3.0)  # it asks 1 back: the oldest lot, asked at 3000 (a drift of 10); its bid is for 1
    decide(market, other, 3.0)  # and then the other sells 11 at 3009: the 10 ahead and the trader's lot, at 3.0
    account = market.account(trader)
    assert account.fills_before == [0, 0, 0] and len(account.fills) == 1 and account.fills[0][0] == 3.0
    parts = shortfall_parts(market, trader, since)
    # the fill got the lot asked at 3010 (3009 less 3010: -1), the one the ask at 3.0 left; had the fill been put
    # before the decision, it would have got the oldest (3009 less 3000: 9) and the decision would take back the
    # other (3010 less 3010: 0): the same sum, 9, and a different split
    assert listed(parts) == [Lot("back", 0, 1, 10.0, "ask", "", 1.0, 3.0),
                             Lot("got", 0, 1, -1.0, "ask", "market", 2.0, 3.0)]
    assert parts.shortfall == 9.0 == parts.total  # by hand: the paper made -3000 + 3009.5 (marks), the trader 0.5


def test_a_fill_of_a_look_between_decisions_is_got_in_its_place_among_the_decisions():
    market, _, trader = worked([3, 0], patience=5.0, reprice=2.0, interval=100.0)
    market.keep_decisions = True
    since = market.since(trader)
    decide(market, trader, 1.0)  # asked 3 at 3000, a bid for 3
    hit(market, 0, ASK, 12)  # 2 of its 3 at 2999
    market.time = 5.0
    market.step()  # a look: its last lot, alone at 2999, is moved to 2998
    market.time = 7.5
    market.step()  # a look past its patience: the last lot crossed at 3001
    account = market.account(trader)
    assert account.fills_before == [0] and [fill[5] for fill in account.fills] == ["market", "cross"]
    parts = shortfall_parts(market, trader, since)
    assert parts.got_lots_by_fill == NO_FILLS | {"market": 2, "cross": 1} and parts.open_lots == 0
    assert parts.got_cost == -1.0 == parts.shortfall  # 2 x (2999 - 3000) and 3001 - 3000


def test_a_trader_whose_position_was_changed_by_something_the_account_does_not_record_leaves_a_residual_the_notes_say():
    market, trader, since = started([1, 0], patience=50.0)
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 11)  # a lot got, at 2999
    assert shortfall_parts(market, trader, since).notes == []
    trader.position[1] += 1  # a lot of book 1 from nowhere: no fill, no cash
    parts = shortfall_parts(market, trader, since)
    assert parts.total == -1.0 and parts.residual == -3000.0  # its P&L is 3000 more, and no lot says why
    assert len(parts.notes) == 1 and "does not record" in parts.notes[0]


# ---- What a since is, and what is refused ----

def test_a_start_is_the_account_as_it_was_and_does_not_change_when_the_trader_trades_after_it():
    market, _, trader = worked([2, 0], patience=50.0)
    market.keep_decisions = True
    decide(market, trader, 1.0)
    hit(market, 0, ASK, 11)
    since = market.since(trader)
    assert isinstance(since, Since) and since.time == 1.0 and since.account is market.account(trader)
    assert since.position.tolist() == [1, 0] and since.asked.tolist() == [2, 0] and since.cash == -2999.0
    assert since.paper_cash == -6000.0 and since.marks.tolist() == [2999.5, 3000.0]
    assert (since.fills, since.decisions, since.unkept) == (1, 1, 0)
    hit(market, 0, ASK, 1)
    assert since.position.tolist() == [1, 0] and since.cash == -2999.0 and trader.position.tolist() == [2, 0]
    with pytest.raises(ValueError, match="not in this market"):
        market.since(Wants(goal=np.zeros(2, dtype=int)))


def test_the_parts_are_refused_in_a_sentence_if_the_market_did_not_keep_the_decisions_of_the_span():
    market, _, trader = worked([1, 0], patience=50.0)
    since = market.since(trader)  # keep_decisions is off: the decisions that follow are not kept
    decide(market, trader, 1.0)
    with pytest.raises(ValueError, match=r"1 of this trader's decisions .* keep_decisions off"):
        shortfall_parts(market, trader, since)
    market.keep_decisions = True  # on again: a span from here on is whole
    later = market.since(trader)
    trader.goal = np.array([0, 0])
    decide(market, trader, 2.0)
    assert shortfall_parts(market, trader, later).back_lots == 1  # the decisions before it do not matter
    market.keep_decisions = False
    decide(market, trader, 3.0)  # a decision that was not kept: the span from `later` is not whole
    with pytest.raises(ValueError, match="keep_decisions"):
        shortfall_parts(market, trader, later)


def test_the_parts_are_refused_for_a_start_that_is_not_one_or_is_another_traders():
    market, _, trader = worked([1, 0], patience=50.0)
    market.keep_decisions = True
    other = Wants(goal=np.zeros(2, dtype=int), interval=5.0)
    market.add(other)
    for since in (None, 0.0, market.since(other)):
        with pytest.raises(ValueError, match=r"market\.since\(trader\)"):
            shortfall_parts(market, trader, since)
    with pytest.raises(ValueError, match="not in this market"):
        shortfall_parts(market, Wants(goal=np.zeros(2, dtype=int)), market.since(trader))


# ---- score_execution says what the shortfall is made of ----

HOLDINGS_KEYS = {"pnl", "before_costs", "costs", "lots", "entries", "gross", "exposure", "deviation", "netted",
                 "risk_adjusted", "deviation_long", "risk_adjusted_long", "factor_part"}
EXECUTION_KEYS = {"paper", "shortfall", "asked_lots", "shortfall_a_lot", "earned", "passive_lots", "crossed_lots",
                  "urgent_lots", "orders_placed", "orders_withdrawn", "orders_crossed", "wait", "markout_1_market",
                  "markout_5_market", "markout_20_market", "markout_1_limit", "markout_5_limit", "markout_20_limit"}
PARTS_KEYS = {"got_lots", "got_cost", "got_a_lot", "back_lots", "back_drift", "unasked_lots", "unasked_cost",
              "open_lots", "open_drift"}


@pytest.mark.parametrize("kept", [False, True])
@pytest.mark.parametrize("execution", [None, Execution(20.0, reprice=1.0)])
def test_score_execution_has_the_parts_of_the_shortfall_when_the_market_keeps_its_decisions_and_the_keys_it_had_if_not(
        kept, execution):
    market = PassiveBookMarket(FactorValues(sector_loadings(1, 3)), seed=2)
    market.keep_decisions = kept
    trader = Drifter(seed=2, interval=2.0, every=5.0)
    market.add(trader, execution)
    score = score_execution(market, trader, 30.0, 120.0)
    assert set(score) == HOLDINGS_KEYS | EXECUTION_KEYS | (PARTS_KEYS if kept else set())  # no key is added if off


def test_score_execution_does_not_change_a_number_it_had_because_the_decisions_are_kept():
    scores = []
    for kept in (False, True):
        market = PassiveBookMarket(FactorValues(sector_loadings(1, 3)), seed=2)
        market.keep_decisions = kept
        trader = Drifter(seed=2, interval=2.0, every=5.0)
        market.add(trader, Execution(20.0, reprice=1.0))
        scores.append((score_execution(market, trader, 30.0, 120.0), market))
    (off, plain), (on, kept) = scores
    assert {key: on[key] for key in off} == off  # every number it had, to the last digit
    assert (kept.time, kept.events) == (plain.time, plain.events)
    assert kept.rng.bit_generator.state == plain.rng.bit_generator.state


def test_score_execution_parts_of_a_scripted_flow_are_the_lots_got_by_hand_and_add_up_to_its_shortfall():
    market, trader = staged_market(Execution(100.0, reprice=None))
    market.keep_decisions = True
    score = score_execution(market, trader, 2.0, 40.0)
    # as the test of its shortfall above: the first lot was got at 0.5, before the span; in it, 3 lots asked at 2999.5
    # and filled at 2998 by a market sell (-1.5 each), 1 asked at 2999.0 and filled at 2997 by a limit sell (-2.0)
    assert score["shortfall"] == pytest.approx(-6.5)
    assert (score["got_lots"], score["got_cost"], score["got_a_lot"]) == (4, pytest.approx(-6.5), pytest.approx(-1.625))
    assert (score["back_lots"], score["back_drift"], score["unasked_lots"], score["unasked_cost"]) == (0, 0.0, 0, 0.0)
    assert (score["open_lots"], score["open_drift"]) == (0, 0.0)
    market, trader = staged_market(Execution(100.0, reprice=None))
    market.keep_decisions = True
    while market.time < 2.0:
        market.step()
    since = market.since(trader)
    while market.time < 42.0:
        market.step()
    parts = shortfall_parts(market, trader, since)
    assert parts.got_lots_by_fill == NO_FILLS | {"market": 3, "limit": 1}
    assert parts.got_cost_by_fill == {"market": pytest.approx(-4.5), "limit": pytest.approx(-2.0), "cross": 0.0,
                                      "urgent": 0.0}
    assert [lot.cost for lot in parts.lots] == pytest.approx([-1.5, -1.5, -1.5, -2.0])


# ---- Real order flow: the parts of the shortfall of every kind of trader add up to it ----

REAL_RUNS = [(kind, True) for kind in KINDS if kind != "pairs not netted"] + [(kind, False) for kind in KINDS]


def real_run(kind, worked_, seed, stop=None):
    """A trader of this kind, with ``Execution(10, reprice=2)`` or none, in the sectors market for its warm-up and 100
    units, and then 500 more, the market keeping its decisions; made inactive ``stop`` units into the span if told.
    The market, the trader, the start as ``noted`` made it and as ``since`` did, the rows of the trader's fills and the
    lots its paper had asked at the start."""
    market = PassiveBookMarket(FactorValues(SECTORS), seed=seed)
    market.keep_decisions = True
    trader = KINDS[kind](statarb_now, cross_now)
    market.add(trader, Execution(10.0, reprice=2.0) if worked_ else None)
    burn = trader.warm_up + 100.0
    while market.time < burn:
        market.step()
    account = market.account(trader)
    began = noted(market, trader), market.since(trader), len(account.fills), account.asked_lots
    if stop is not None:
        while market.time < burn + stop:
            market.step()
        trader.active = False
    while market.time < burn + 500.0:
        market.step()
    return (market, trader) + began


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("kind, worked_", REAL_RUNS)
def test_the_parts_of_a_real_traders_shortfall_add_up_to_it_exactly_worked_or_not(kind, worked_, seed):
    market, trader, start, since, rows, asked0 = real_run(kind, worked_, seed)
    parts = shortfall_parts(market, trader, since)
    account = market.account(trader)
    shortfall = shortfall_since(market, trader, start)  # written apart, from what was noted
    assert parts.total == pytest.approx(shortfall, abs=1e-6)
    assert parts.got_cost + parts.back_drift + parts.unasked_cost + parts.open_drift == pytest.approx(
        shortfall, abs=1e-6)
    assert parts.shortfall == pytest.approx(shortfall, abs=1e-6) and abs(parts.residual) < 1e-6 and parts.notes == []
    # these traders trade only what the paper asked: no lot is traded ahead of it (a band of half a lot can)
    assert parts.unasked_lots == 0 and {lot.origin for lot in parts.lots} <= {"ask", "carried"}
    # each lot traded got one lot of the gap; each lot asked opened a lot of it or took one back (two lots in the span
    # for each, but not for a lot carried in, which was asked before it)
    traded = sum(abs(fill[2]) for fill in account.fills[rows:])
    carried = int(np.abs(since.asked - since.position).sum())
    asked = account.asked_lots - asked0
    assert parts.got_lots == traded > 50
    assert asked == parts.got_lots + 2 * parts.back_lots + parts.open_lots - carried > 100
    assert len(parts.lots) == parts.got_lots + parts.back_lots + parts.open_lots
    assert parts.got_cost == pytest.approx(sum(parts.got_cost_by_fill.values()))
    assert sum(parts.got_lots_by_fill.values()) == parts.got_lots
    if worked_:
        assert parts.back_lots + parts.open_lots > 0 and parts.got_lots_by_fill["market"] > 0
    else:  # a trader that crosses has every lot filled as it is asked
        assert parts.back_lots == 0 and parts.got_lots_by_fill["market"] == parts.got_lots_by_fill["limit"] == 0


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("kind", ["pairs with a rate and a band", "residuals", "reversal", "flow of queues"])
def test_the_parts_of_a_real_traders_shortfall_add_up_to_it_when_it_is_stopped_part_way_through_the_span(kind, seed):
    market, trader, start, since, rows, asked0 = real_run(kind, True, seed, stop=250.0)
    parts = shortfall_parts(market, trader, since)
    account = market.account(trader)
    assert parts.total == pytest.approx(shortfall_since(market, trader, start), abs=1e-6)
    assert abs(parts.residual) < 1e-6 and parts.unasked_lots == 0
    # a stopped trader asks for the position it holds: nothing of the gap is open at the end, and the stop is a decision
    assert parts.open_lots == 0 and (account.asked == trader.position).all()
    assert (account.decisions[-1][2] == trader.position).all() and not any(market._working)  # the last is the stop
    assert parts.got_lots > 50 and parts.back_lots > 0


@pytest.mark.parametrize("seed", [1, 2])
def test_the_parts_of_each_trader_of_a_market_of_five_add_up_when_their_orders_fill_each_other_in_the_same_step(seed):
    # five traders that decide every 2 units, so at the same step: three work their orders and two cross, and the
    # crossing ones fill the working orders after the decisions of that step
    names = ["residuals", "reversal", "flow of trades", "momentum", "lead-lag"]
    market = PassiveBookMarket(FactorValues(SECTORS), seed=seed)
    market.keep_decisions = True
    traders = {name: KINDS[name](statarb_now, cross_now) for name in names}
    for name, trader in traders.items():
        market.add(trader, Execution(10.0, reprice=2.0) if name in names[:3] else None)
    burn = max(trader.warm_up for trader in traders.values()) + 100.0
    while market.time < burn:
        market.step()
    began = {name: (noted(market, trader), market.since(trader), len(market.account(trader).fills),
                    market.account(trader).asked_lots) for name, trader in traders.items()}
    while market.time < burn + 400.0:
        market.step()
    for name, trader in traders.items():
        start, since, rows, asked0 = began[name]
        account, parts = market.account(trader), shortfall_parts(market, trader, since)
        assert parts.total == pytest.approx(shortfall_since(market, trader, start), abs=1e-6), name
        assert parts.unasked_lots == 0 and {lot.origin for lot in parts.lots} <= {"ask", "carried"}, name
        traded = sum(abs(fill[2]) for fill in account.fills[rows:])
        carried = int(np.abs(since.asked - since.position).sum())
        assert parts.got_lots == traded > 100, name
        assert account.asked_lots - asked0 == parts.got_lots + 2 * parts.back_lots + parts.open_lots - carried, name
        decided = {row[0]: index for index, row in enumerate(account.decisions)}
        after = sum(1 for index, fill in enumerate(account.fills[rows:], start=rows)
                    if fill[0] in decided and fill[5] in ("market", "limit")
                    and index >= account.fills_before[decided[fill[0]]])
        assert (after > 20) == (name in names[:3]), name  # a passive fill after a decision of the same time, often


# ---- What the mechanical sweep of the lines of shortfall_parts found the tests did not say (an audit, 2026-11-14) ----

def parts_of(shortfall, got_cost):
    """A ``ShortfallParts`` made by hand with no lots but a ``got_cost`` and a ``shortfall``: its residual is the
    difference, and nothing else is in it."""
    return ShortfallParts(lots=[], shortfall=shortfall, got_lots=0, got_cost=got_cost, back_lots=0, back_drift=0.0,
                          unasked_lots=0, unasked_cost=0.0, open_lots=0, open_drift=0.0,
                          got_lots_by_fill=dict(NO_FILLS), got_cost_by_fill=dict(NO_COST))


def test_the_decisions_made_while_the_market_did_not_keep_them_are_counted_and_the_refusal_says_how_many():
    market, _, trader = worked([1, 0], patience=50.0)
    account = market.account(trader)
    since = market.since(trader)  # keep_decisions is off
    decide(market, trader, 1.0)
    trader.goal = np.array([2, 0])
    decide(market, trader, 2.0)
    assert account.unkept == 2 and account.decisions == [] and account.fills_before == []  # counted, and not kept
    with pytest.raises(ValueError, match=r"(?<![-\d])2 of this trader's decisions in the span were made with keep_"):
        shortfall_parts(market, trader, since)
    market.keep_decisions = True
    trader.goal = np.array([3, 0])
    decide(market, trader, 3.0)  # a kept one does not add to the count, and has its place among the fills
    assert account.unkept == 2 and len(account.decisions) == 1 and account.fills_before == [len(account.fills)]


def test_a_start_cannot_be_changed_once_it_is_noted():
    market, trader, since = started([1, 0], patience=50.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        since.time = 5.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        since.unkept = 3
    assert since.time == 0.0 and since.unkept == 0


def test_the_mean_cost_of_the_lots_got_is_0_when_none_was_got_and_the_mean_of_their_costs_when_some_were():
    market, trader, since = started([1, 0], patience=50.0)
    decide(market, trader, 1.0)
    market._shift(0, 4)
    trader.goal = np.array([0, 0])
    decide(market, trader, 2.0)  # the lot is taken back: none was got
    parts = shortfall_parts(market, trader, since)
    assert parts.got_lots == 0 and parts.got_a_lot == 0.0 and parts.back_lots == 1
    assert parts_of(0.0, -7.0).got_a_lot == 0.0  # no lots, and a cost (which cannot be): still 0, not a division
    twin_market, _ = hand_market()
    twin_market.keep_decisions = True
    twin = Wants(goal=np.array([12, 0]), interval=5.0)  # 10 lots at 3001 (1 each) and 2 at 3002 (2 each)
    twin_market.add(twin)
    start = twin_market.since(twin)
    decide(twin_market, twin, 1.0)
    got = shortfall_parts(twin_market, twin, start)
    assert (got.got_lots, got.got_cost) == (12, 14.0) and got.got_a_lot == 14.0 / 12


def test_the_note_on_lots_traded_ahead_of_the_paper_says_how_many_in_the_singular_and_the_plural():
    sentence = ("{} traded and traded back with no ask between, a position ahead of the paper's, at a cost of {} "
                "ticks: not what a signal asked for")
    market, _ = hand_market()
    market.keep_decisions = True
    trader = Wants(goal=np.array([0, 0]), interval=5.0)
    market.add(trader)
    since = market.since(trader)
    account = market.account(trader)
    for at, lots in ((1.0, 1), (2.0, -1)):  # bought at 3001 and sold at 2999: a round trip of 2 ticks
        market.time = at
        market._cross(trader, account, 0, lots, "cross")
    assert shortfall_parts(market, trader, since).notes == [sentence.format("1 lot was", "2.0")]
    for at, lots in ((3.0, 1), (4.0, -1)):  # and another, the same: 2 lots, 4 ticks
        market.time = at
        market._cross(trader, account, 0, lots, "cross")
    assert shortfall_parts(market, trader, since).notes == [sentence.format("2 lots were", "4.0")]


def test_a_residual_is_noted_when_it_is_more_than_a_millionth_of_one_and_the_shortfall():
    assert parts_of(0.0, 1e-6).residual == -1e-6 and parts_of(0.0, 1e-6).notes == []  # on the line: not noted
    assert parts_of(0.0, 7e-7).notes == [] and parts_of(0.0, -9e-7).notes == []  # under it, either way
    assert len(parts_of(0.0, 1.5e-6).notes) == 1  # over it
    assert len(parts_of(0.0, -1.5e-6).notes) == 1
    assert parts_of(1000.0, 1000.0005).notes == []  # 5e-4 of a shortfall of 1000: under 1e-6 x 1001
    assert len(parts_of(1000.0, 1000.002).notes) == 1  # 2e-3: over it
    assert len(parts_of(-1000.0, -999.998).notes) == 1 and parts_of(-1000.0, -999.9995).notes == []
    note = parts_of(1000.0, 1000.002).notes[0]
    assert note.startswith("the parts miss the shortfall by ") and "does not record" in note


def test_a_lot_the_trader_traded_before_the_paper_asked_for_it_is_got_at_the_distance_of_the_fill_from_the_mid_asked():
    market, _ = hand_market()
    market.keep_decisions = True
    trader = Wants(goal=np.array([1, 0]), interval=5.0)
    market.add(trader)
    since = market.since(trader)
    account = market.account(trader)
    market.time = 1.0
    market._cross(trader, account, 0, 1, "cross")  # bought at 3001, nothing asked yet
    market._shift(0, 3)  # the book jumps: bids 3000 to 3002, asks 3004 to 3006, the mid 3003
    decide(market, trader, 2.0)  # the paper asks for the lot at 3003: the fill was 2 below that
    parts = shortfall_parts(market, trader, since)
    assert parts.lots == [Lot("got", 0, -1, -2.0, "fill", "cross", 1.0, 2.0)]  # 3001 less 3003, not one over it
    assert parts.shortfall == -2.0 == parts.total  # by hand: the paper -3003 + 3003, the trader -3001 + 3003

