"""Resting limit orders for the traders of ``lob.statarb.BookMarket``: a trader works what it needs at the best price of
each book, and crosses what is left with a market order once its patience has run out.

``PassiveBookMarket`` is a ``BookMarket``. A trader added with an ``Execution`` has, in each book, one working order:
what it needs there (the moves its ``decide`` gives, added up), at the best price on its side, which it moves back to
the best price as often as ``reprice`` says. A trader added with none, or with a patience of 0, trades as it does in a
``BookMarket``, to the last digit. No trader class changes and nothing in a trader object is added: the market keeps
each trader's execution, its working orders and its account (``account``).

**The yardstick.** Under passive execution most of an order's cost is the drift of the price between the decision and
the fill, which a before-costs column would book as the signal's: a worked trader's ``before_costs`` is nan, and
``paper`` is what to set beside a crossing trader's. So the market keeps, for every trader in it, a paper account: the
position it asked for at each decision, got at once at that decision's mid, with no cost. Paper P&L is what the signal
made, and shortfall, paper P&L less P&L, is what execution cost. Spread earned and paid are diagnostics.

The position a ``StateTrader`` or a ``CrossSectionTrader`` asked for is where its own rule (its aim, its target and its
band) takes the paper's last position, which is what the same trader filled at once would ask. So the same signal has
the same paper whether its orders fill or not, and a band does not make the paper follow the fills (inside its band a
trader asks for nothing, which for its position is what it holds). Any other trader asks its position and its need. A
trader made inactive gives up on paper, at the marks then, what it had not got.

A shortfall is the sum of its lots' costs, each a fill against its decision's mid, only where every lot asked is got. In
general it is the sum of four parts, which ``shortfall_parts`` tells lot by lot, first in first out in each book (and
``score_execution`` sums them when the market keeps its decisions): the cost of the lots *got* (asked and filled, or
filled and asked: a fill against the mid at the decision); the paper's gain on the lots *taken back* (asked, and taken
back by a later ask of the other sign before a fill reached them: no working order cost that, since the trader never had
them); the cost of the lots *unasked* (traded and traded back with no ask between, a position ahead of the paper's:
rare, and seen only where a band of half a lot makes a trader that fills in parts round up where the same trader filled
at once rounds down, as pairs with a band of 0.5 do); and the drift of the lots *open* (left in the gap between the
paper and the position at the end: a need not yet met, or a band that keeps the position short of the paper's). So a
shortfall a lot is what working an order cost only where nothing is taken back, unasked or open. A queue follower at a
rate of 0.02 (10,000 units, seeds 1 to 8) got 2,380 of the 5,108 lots a seed its paper asked, at -0.17 a lot against
their decisions' mids, and its shortfall of +0.15 a lot read as a cost of working an order: the rest was the paper's
gain on lots it asked and took back before a fill reached them (1,364 a seed, +0.87 a lot). The split is exact (to
rounding) for every trader whose trades the account records, worked or not, with a band or without, stopped or not; it
does not say what the drift on lots taken back means for a signal.

**Own orders.** What a trader sees and is marked at leaves out its own working orders, and no one else's: the mids
``decide`` gets, the books ``observe`` gets and the market's ``mid``, ``mark`` and ``marks`` there (through a view),
``marks_for`` and the frames of ``run``. Where a mid is missing without its own orders, the mark is the last one it
had without them. ``flow`` and ``volume`` are the market's: the lots a model's order traded against a trader's working
order are in the flow a ``FlowTrader`` reads, as the lots it traded against anyone are. That is not feedback from the
trader's orders: the models' orders arrive at the same rates with or without them, and the flow differs only where a
thin book lets an order meet more or less depth.

**What it scores.** ``score_execution`` is ``score_holdings`` with the trader marked at the mids without its own orders,
and with what its execution cost (``shortfall``), what its passive fills earned of the spread and what they were worth
1, 5 and 20 units later (their markouts, by the kind of order that hit them). With the market's ``keep_decisions`` on
it also says what the shortfall is made of (above). A trader made inactive in the span has its paper give up the lots
it had not got at the stop, so that its shortfall does not go on growing with the price.

**What this market does for a passive order that a real one does not.** The order flow does not react to it, nobody
steps in front of it, cancel and replace are free and instant, a jump is seen in time to withdraw, and there are no
fees or rebates. Three more things it does for it, the first two measured on one lot a book every 25 units:

- Its takers are nearly uninformed. A model's market order carries an impact of 0.05 a lot (``FactorValues``'
  ``impact``), and about 69% of the passive lots were filled by one, worth +1.5 a lot 20 units later; the lots filled
  by a model's limit order across the quote were worth -2.9. Taking the impact to 0.5 moved a passive trader's
  shortfall a lot from -0.09 to +0.34, 0.43 of the 1.07 a lot it saved over crossing. So a table of what passive
  execution saves should carry a row at ``impact=0.5``, and read the saving as a property of this market's takers and
  not of resting orders.
- An order left alone at the best price (the others there gone) is put back at the next price at the next look, since
  the best price is told without its own order: a step back that costs it nothing. It was 7% of the placements at an
  interval of 1, and 0.3% at 5.
- A jump is always seen in time. There is no row for one that fills a resting order, and the touch of an index book is
  not measured; read nothing from a run with deals or an index book.

What it does not flatter: a model's order that hits it moves the fair value as it moves it for anyone (at an impact of
0.05, see above), a model's order priced across a stale quote is a limit order that trades at the trader's price, and a
working order goes to the back of its queue when it moves.

**What patience and reprice mean here.** They are looked at where the market looks for the trader. By default that is
every unit of time (a ``reprice`` of 1), whatever the trader's interval, so that a patience means the same for a trader
that decides every unit and for one that decides every 5. With a ``reprice`` of None it is at the trader's decisions
alone: a need is then crossed at the first decision at or after its patience (21 for an interval of 3 or 7 and a
patience of 20), and, at a decision that cannot be made for a missing mid, its patience still runs and what is left is
crossed. How often the order is moved decides more than the patience: at a patience of 20 the acceptance's trader of
interval 5 had a shortfall a lot of -0.105 with a reprice of None, -0.194 with a reprice of 1, and at an interval of 1
-0.223. So compare traders at the same reprice, which the default gives. A need's clock starts when the need begins
and keeps going while it grows or shrinks on its side, so lots asked late are crossed with the first: the patience is
the age of a need's oldest lot (for a goal that creeps at 0.1 a unit, the lots crossed at a patience of 20 were 11 to
12 units old). A need that falls to nothing, even for one decision, clears its clock: a trader with a band whose need
keeps going to nothing may never cross (notebook 13's queue follower with a band of 4 crossed no lot in 8 runs of 5,000
units).

What it leaves out. A trader of relations works every leg and never hedges a leg that has filled; an order always
joins the best price on its side (it does not improve it, and where it rests is not a setting); a trader that does not
net its orders is refused. A model's cancel that names a working order is dropped: it counts no event and draws
nothing more, so the model's own orders are canceled as before (exact thinning of the cancel draw: measured, 0.0501 a
unit with the trader and without it).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from lob.crosssection import CrossSectionTrader
from lob.order import Fill, Order, OrderType, Side, cancel_order, limit_order
from lob.statarb import BookMarket, BookTrader, StateTrader, score_holdings

TOLERANCE = 1e-9  # a clock has run its time if it is within this of it
MARKOUT_HORIZONS = (1.0, 5.0, 20.0)  # units of time after a passive fill at which it is marked


def _nan_for(mid: Optional[float]) -> float:
    """A mid for a row of numbers: ``nan`` where the book had none."""
    return math.nan if mid is None else mid


@dataclass(frozen=True)
class Execution:
    """How a trader works its orders: settings and nothing else.

    ``patience`` is the units of time a need is worked before what is left of it is crossed with a market order. The
    clock of a need starts when the need does, on its side, and goes on while the need grows or shrinks on that side:
    the patience is the age of the need's oldest lot. A need that falls to nothing clears it. A patience of 0 is no
    execution at all: ``PassiveBookMarket.add`` keeps such a trader as one with none, which is a ``BookMarket``'s
    trader to the last digit. ``reprice`` is the least time between moves of a working order back to the best price:
    a clock of that period (1 unit unless told, whatever the trader's interval), which looks between decisions if it
    is shorter than the trader's interval and leaves a stale order where it is at the decisions it is not due at if it
    is longer; None moves the order at each of the trader's decisions and looks at no other time. How often an order is
    moved decides more of what it costs than the patience does, so compare traders at the same ``reprice``.
    ``urgent_closes`` keeps as market orders the one case where a
    trader closes everything at once whatever its settings: a ``StateTrader`` with no relations, which is also a
    ``DealTrader`` with no deal on.
    """
    patience: float = 20.0
    reprice: Optional[float] = 1.0
    urgent_closes: bool = True

    def __post_init__(self):
        if not (math.isfinite(self.patience) and self.patience >= 0):
            raise ValueError("a patience of 0 or more units of time")
        if self.reprice is not None and not (math.isfinite(self.reprice) and self.reprice > 0):
            raise ValueError("a reprice of None, or a time above 0")


@dataclass(eq=False)
class Account:
    """What the market keeps of one trader's trading, apart from what the trader keeps of itself.

    ``passive_lots`` are the lots its working orders traded, ``crossed_lots`` those that went through the market's
    ``_trade`` (``urgent_lots`` of them, the closes of a trader with no relations, are urgent), and the trader's own
    ``lots`` is their sum. ``spread_earned`` is, over passive fills, the maker's sign times the mid before the order
    that hit it less the fill's price, times the lots: positive for a bid filled below the mid. The orders it placed,
    withdrew and crossed (a need whose patience ran out) are counted, and ``waits`` has ``(time, wait)`` for each need
    that ended, filled in full by its working order or by crossing what was left; a need that was dropped has none.

    The paper account is ``asked``, the position the trader asked for at its last decision (its position then plus
    its need; for a ``StateTrader`` or a ``CrossSectionTrader``, where its own rule takes the paper's last position,
    which is not its need when a band keeps its position short of the paper's), ``paper_cash``, what getting each change
    of it at once at that decision's mids paid, and ``asked_lots``, the lots that traded. A trader made inactive has its
    ``asked`` set to its position at the marks then. ``fills`` has a row for each fill,
    ``(time, book, signed lots, price, mid before, kind)``: the kind is ``"market"`` or ``"limit"`` for a passive fill,
    by the kind of order that hit it, and ``"cross"`` or ``"urgent"`` for the trader's own market orders. The mid before
    is the book's, as the trader was marked at it (its own orders left out), and ``nan`` where there was none. With the
    market's ``keep_decisions`` on, ``decisions`` has ``(time, mids, asked)`` for each decision (and each stop that gave
    up lots), which is all the paper account is made of; ``fills_before`` has, for each of them, the number of rows
    ``fills`` had when it was made, so that the decisions and the fills are in the order they happened, a fill of the
    same step before a decision or after it; ``unkept`` counts the decisions (and such stops) made while it was off.

    ``markouts`` has a row ``(index, horizon, value)`` for each passive fill (the row ``index`` of ``fills``) and each
    of 1, 5 and 20 units (``MARKOUT_HORIZONS``) once that time has come: the maker's sign times the mid then, as the
    trader is marked at it (``mid_for``, or where that is None the last it had), less the fill's price, which is what a
    lot of the fill is worth then. ``markout`` averages them by kind of the order that hit the fill.
    """
    books: int
    passive_lots: int = 0
    crossed_lots: int = 0
    urgent_lots: int = 0
    spread_earned: float = 0.0
    orders_placed: int = 0
    orders_withdrawn: int = 0
    orders_crossed: int = 0
    paper_cash: float = 0.0
    asked_lots: int = 0
    fills: list = field(default_factory=list)
    waits: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    fills_before: list = field(default_factory=list)
    unkept: int = 0
    markouts: list = field(default_factory=list)
    asked: np.ndarray = field(init=False)

    def __post_init__(self):
        self.asked = np.zeros(self.books, dtype=int)

    def markout(self, horizon: float, kind: str, since: int = 0) -> float:
        """What a lot of the passive fills made from row ``since`` of ``fills`` on, that an order of this kind hit
        (``"market"`` or ``"limit"``), was worth ``horizon`` units later: the mean of their markouts, weighted by the
        lots. 0 with none, and a fill whose later time has not come is not one."""
        lots = total = 0.0
        for index, later, value in self.markouts:
            if later == horizon and index >= since and self.fills[index][5] == kind:
                lots += abs(self.fills[index][2])
                total += abs(self.fills[index][2]) * value
        return total / lots if lots else 0.0


@dataclass(frozen=True, eq=False)
class Since:
    """Where a trader's account stood at the start of a span, as ``PassiveBookMarket.since`` notes it: the time, the
    position and cash, what the paper account asked and held in cash, the trader's marks (``marks_for``) and how many
    rows of ``fills`` and ``decisions`` the account had. ``shortfall_parts`` splits what happened after it."""
    account: Account
    time: float
    position: np.ndarray
    cash: float
    asked: np.ndarray
    paper_cash: float
    marks: np.ndarray
    fills: int
    decisions: int
    unkept: int


class _Work:
    """What the market keeps of one trader that has an execution: its working order in each book (an order id), what
    is left of its need there, its clock there (the time it started and the side) and its reprice clock."""

    def __init__(self, trader: BookTrader, execution: Execution, account: Account, market: PassiveBookMarket):
        n = market.n
        self.trader, self.execution, self.account = trader, execution, account
        self.order: dict[int, int] = {}
        self.left = np.zeros(n, dtype=int)
        self.clock: dict[int, tuple[float, Side]] = {}
        self.counted: set[int] = set()  # the books whose need has been counted as crossed (a new clock clears one)
        self.last: dict[int, float] = {  # the last mid it had in each book without its own orders
            book: mid for book, mid in enumerate(b.mid_price for b in market.books) if mid is not None}
        self.next_look = -math.inf
        self.view = _MarketView(market, self)


class _BookView:
    """A book as one trader sees it: the best prices, the mid, the spread and the depth at a price without that
    trader's own working order. Everything else is the book's."""

    def __init__(self, market: PassiveBookMarket, work: _Work, book: int):
        self._market, self._work, self._book = market, work, book

    def __getattr__(self, name):
        return getattr(self.__dict__["_market"].books[self.__dict__["_book"]], name)

    @property
    def best_bid(self) -> Optional[int]:
        return self._market._best(self._work, self._book, Side.BID)

    @property
    def best_ask(self) -> Optional[int]:
        return self._market._best(self._work, self._book, Side.ASK)

    @property
    def mid_price(self) -> Optional[float]:
        return self._market._mid(self._work, self._book)

    @property
    def spread(self) -> Optional[int]:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else ask - bid

    def depth_at_price(self, price: int, side: Side) -> int:
        target = self._market.books[self._book]
        depth = target.depth_at_price(price, side)
        order = target.get_order(self._work.order[self._book]) if self._book in self._work.order else None
        if order is not None and order.side == side and order.price == price:
            depth -= order.size
        return depth


class _MarketView:
    """The market as one trader's ``observe`` sees it: every attribute is the market's, as it is now, except ``books``,
    which are views of the books without that trader's own working orders, and ``mid``, ``mark`` and ``marks``, which
    answer for the books as that trader sees them."""

    def __init__(self, market: PassiveBookMarket, work: _Work):
        self._market, self._work = market, work
        self.books = [_BookView(market, work, book) for book in range(market.n)]

    def __getattr__(self, name):
        return getattr(self.__dict__["_market"], name)

    def mid(self, book: int) -> Optional[float]:
        return self._market._mid(self._work, book)

    def mark(self, book: int) -> float:
        return self._market._mark_for(self._work, book)

    def marks(self) -> np.ndarray:
        return self._market._marks(self._work)


class PassiveBookMarket(BookMarket):
    """A ``BookMarket`` in which traders can rest orders: ``add(trader, execution)`` says how a trader works its needs.

    A trader decides at its interval as in a ``BookMarket``, and with an execution its needs are worked, not crossed,
    in this order. The mids are those without its own orders (``mid_for``; if one is missing it does not decide, and its
    orders stay as they are, but what is left of a need whose patience has run out is crossed). It observes (a view of
    the market without its own orders in the books) and decides. A need is the sum of the moves it gives, signed lots a
    book, except for a cross-sectional trader that does not trade cost-aware, which gives none between rebalances and
    whose need is its last target less its position at every decision from its first rebalance. The paper account takes
    the position it asks for (see the module's docstring). Then, in each book, a need
    starts a clock on its side if there is none (a change of side starts it again, and withdraws the order on the old
    side; growing or shrinking on the same side keeps it); once the clock has run ``patience`` the order is withdrawn
    and what is needed is sent as a market order, and until then the order is kept if it rests with the size needed at
    the best price without its own, put right otherwise, and placed if none rests. A passive fill completes a need if
    it takes the last of it, and the next need in that book starts a clock of its own. A need is counted as crossed
    (``orders_crossed``) once, however many looks its crossing takes in a thin book. A trader with no relations (the
    kind ``Execution.urgent_closes`` names) withdraws everything and sends its closes as market orders at once.

    With a ``reprice`` the market looks again between decisions, at the end of any step in which the trader's reprice
    clock is due: it crosses the books whose patience has run out, moves orders whose price is not the best, and places
    what is left of a need where nothing rests. What is left of a need is the need at the last decision less the
    passive fills since. A trader that is no longer ``active`` has its orders withdrawn at the next step, and its
    clocks and needs dropped, and its paper account gives up the lots it did not get at the marks then.

    Placing and withdrawing are ``BookMarket.submit``'s, with no impact: each counts an event and takes the book's
    rate again, and neither counts in ``flow`` or ``volume`` or moves a fair value. A model's cancel that names a
    working order is dropped. A fill of a working order, by whatever order, books the maker as ``_trade`` books a
    taker, with the maker's sign; an order's impact is its own, whoever it hits. A jump of a book's fair value
    withdraws every working order in it (needs and clocks are kept), and they are placed again at the next look.

    Each passive fill is marked 1, 5 and 20 units later (``Account.markouts``), at the first step at or after that
    time, at the mid the trader is marked at then. ``keep_decisions`` (off) makes each account keep the decisions its
    paper account was made from, which ``shortfall_parts`` needs: ``since(trader)`` notes the start of a span, and
    ``shortfall_parts(market, trader, since)`` says what the shortfall since is made of.

    What it leaves out is in the module's docstring.
    """

    keep_decisions = False

    def __init__(self, fair, *args, **kwargs):
        self._working: list[dict[int, _Work]] = [{} for _ in range(fair.n)]  # a book's working orders, by order id
        self._accounts: dict[int, Account] = {}
        self._works: dict[int, _Work] = {}
        self._tap: Optional[list[Fill]] = None  # the fills of the trader's own market order being sent
        self._pending = [deque() for _ in MARKOUT_HORIZONS]  # a horizon's markouts to take: (time, work, account, row)
        super().__init__(fair, *args, **kwargs)

    def add(self, trader: BookTrader, execution: Optional[Execution] = None) -> None:
        """As ``BookMarket.add``, and keeps the trader's execution. A trader that does not net its orders is refused one
        (it sends an order for each relation, and a working order is one a book), a patience of 0 included. A patience
        of 0 is no execution: the trader is kept as one with none."""
        if execution is not None and not isinstance(execution, Execution):
            raise ValueError("an execution is an Execution, or None")
        if execution is not None and not trader.netted:
            raise ValueError("a trader that does not net its orders cannot work orders: it sends an order for each of "
                             "its relations, and a working order is one in a book; net its orders or give it no "
                             "execution")
        super().add(trader)
        account = self._accounts[id(trader)] = Account(self.n)
        if execution is not None and execution.patience > 0:
            self._works[id(trader)] = _Work(trader, execution, account, self)

    def account(self, trader: BookTrader) -> Account:
        """The account of a trader in this market."""
        if id(trader) not in self._accounts:
            raise ValueError("this trader is not in this market")
        return self._accounts[id(trader)]

    def since(self, trader: BookTrader) -> Since:
        """Where the trader's account stands now, to be given to ``shortfall_parts`` at the end of a span. Set
        ``keep_decisions`` before the span's first decision: the parts are made of the decisions it keeps."""
        account = self.account(trader)
        return Since(account, self.time, trader.position.copy(), trader.cash, account.asked.copy(), account.paper_cash,
                     self.marks_for(trader), len(account.fills), len(account.decisions), account.unkept)

    # ---- What a trader sees and is marked at: the books without its own orders ----

    def _best(self, work: Optional[_Work], book: int, side: Side) -> Optional[int]:
        target = self.books[book]
        price = target.best_bid if side == Side.BID else target.best_ask
        order_id = work.order.get(book) if work is not None else None
        if price is None or order_id is None:
            return price
        order = target.get_order(order_id)
        if order is None or order.side != side or order.price != price:
            return price
        if target.depth_at_price(price, side) > order.size:
            return price  # others are at the price too
        deeper = target.depth_profile(side, 2)
        return deeper[1][0] if len(deeper) > 1 else None

    def _mid(self, work: Optional[_Work], book: int) -> Optional[float]:
        if work is None or book not in work.order:
            return self.books[book].mid_price
        bid, ask = self._best(work, book, Side.BID), self._best(work, book, Side.ASK)
        return None if bid is None or ask is None else (bid + ask) / 2.0

    def best_for(self, trader: BookTrader, book: int, side: Side) -> Optional[int]:
        """The best price on a side of a book among the orders that are not that trader's working order there (it has at
        most one in a book): the book's best price, unless its own order is at it and is all the depth there, and then
        the next price. None if there is none."""
        return self._best(self._works.get(id(trader)), book, side)

    def mid_for(self, trader: BookTrader, book: int) -> Optional[float]:
        """The mean of the best bid and the best ask without that trader's own order, None if either is None: the
        book's own mid when the trader has nothing resting there."""
        return self._mid(self._works.get(id(trader)), book)

    def marks_for(self, trader: BookTrader) -> np.ndarray:
        """What a position in each book is worth to that trader a lot: ``mid_for``, and where that is None the last
        ``mid_for`` it had (the market's own ``mark``, the last mid the book had, if it has had none: that one may be
        made of the trader's own order). ``marks`` stays the books' own."""
        return self._marks(self._works.get(id(trader)))

    def _marks(self, work: Optional[_Work]) -> np.ndarray:
        return np.array([self._mark_for(work, book) for book in range(self.n)])

    def _mark_for(self, work: Optional[_Work], book: int) -> float:
        mid = self._mid(work, book)
        if mid is not None:
            return mid
        last = None if work is None else work.last.get(book)
        return self.mark(book) if last is None else last

    def mark(self, book: int) -> float:
        """As ``BookMarket.mark``, the book's own; and each trader with an execution notes the mid it has in the book
        now without its own orders (``marks_for`` falls back on the last it noted). The base class calls this after
        every order a book gets and every jump, which is every time a book's mid can change."""
        mark = super().mark(book)
        for work in self._works.values():
            mid = self._mid(work, book)
            if mid is not None:
                work.last[book] = mid
        return mark

    # ---- What the models send, and what a trader's order meets ----

    def submit(self, book: int, order: Order, impact: bool = True) -> list[Fill]:
        """``BookMarket.submit``, and the fills of working orders are booked. A model's cancel that names a working
        order is dropped: it returns no fills, counts no event and changes nothing. A trader's own placements and
        withdrawals call the base's ``submit`` and so are never dropped."""
        working = self._working[book]
        if not working and self._tap is None:
            return super().submit(book, order, impact)
        if order.order_type == OrderType.CANCEL and order.order_id in working:
            return []
        mids = {id(work): self._mid(work, book) for work in working.values()}
        fills = super().submit(book, order, impact)
        for fill in fills:
            work = working.get(fill.maker_order_id)
            if work is not None:
                self._credit(work, book, fill, order, mids[id(work)])
        if self._tap is not None:
            self._tap.extend(fills)
        return fills

    def _credit(self, work: _Work, book: int, fill: Fill, hit: Order, mid: Optional[float]) -> None:
        """A working order's fill, booked as ``_trade`` books a taker's, with the maker's sign: a resting bid buys."""
        trader, account = work.trader, work.account
        sign = 1 if hit.side == Side.ASK else -1
        trader.position[book] += sign * fill.size
        trader.cash -= sign * fill.price * fill.size
        trader.book_cash[book] -= sign * fill.price * fill.size
        trader.lots += fill.size
        account.passive_lots += fill.size
        if mid is not None:
            account.spread_earned += sign * (mid - fill.price) * fill.size
        kind = "market" if hit.order_type == OrderType.MARKET else "limit"
        account.fills.append((self.time, book, sign * fill.size, fill.price, _nan_for(mid), kind))
        work.left[book] -= sign * fill.size
        if self.books[book].get_order(fill.maker_order_id) is None:  # the order is gone
            del work.order[book]
            del self._working[book][fill.maker_order_id]
            if work.left[book] == 0:
                clock = work.clock.pop(book, None)
                if clock is not None:
                    account.waits.append((self.time, self.time - clock[0]))
        self._mark_later(work, account)

    def _shift(self, book: int, ticks: int) -> None:
        """As ``BookMarket._shift``, after every trader's working order in that book is withdrawn (its need and clock
        are kept): a trader sees a jump in time, and a stale order would be moved with the book for free."""
        for work in list(self._working[book].values()):
            self._withdraw(work, book)
        super()._shift(book, ticks)

    def step(self) -> None:
        """As ``BookMarket.step``, and then, for each trader with an execution: if it is no longer active, its orders
        are withdrawn; if it has a ``reprice`` and its reprice clock is due, a look at its books. Then the markouts
        whose time has come are taken."""
        super().step()
        for work in self._works.values():
            if not work.trader.active:
                if work.order or work.clock or (work.account.asked != work.trader.position).any():
                    self._stop(work)
            elif work.clock and work.execution.reprice is not None and self.time >= work.next_look - TOLERANCE:
                self._look(work)
        self._markouts_due()

    # ---- What a passive fill is worth later ----

    def _mark_later(self, work: _Work, account: Account) -> None:
        """A passive fill was just booked (the last row of ``account.fills``): it is to be marked at each horizon."""
        row, now = len(account.fills) - 1, self.time
        for queue, horizon in zip(self._pending, MARKOUT_HORIZONS, strict=True):
            queue.append((now + horizon, work, account, row))

    def _markouts_due(self) -> None:
        """Take the markout of each passive fill whose later time has come, at the first step at or after it."""
        now = self.time
        for queue, horizon in zip(self._pending, MARKOUT_HORIZONS, strict=True):
            while queue and queue[0][0] <= now:
                _, work, account, row = queue.popleft()
                _, book, lots, price, _, _ = account.fills[row]
                later = self._mark_for(work, book)
                account.markouts.append((row, horizon, (1 if lots > 0 else -1) * (later - price)))

    # ---- A decision ----

    def trade_for(self, trader: BookTrader) -> None:
        """Ask a trader what it wants, and trade it, book by book. With no execution it does what
        ``BookMarket.trade_for`` does, with market orders; with one, it works its needs."""
        work = self._works.get(id(trader))
        account = self._accounts[id(trader)]
        trader.next_time = self.time + trader.interval
        aim = trader._aim
        if work is None:
            mids = [book.mid_price for book in self.books]
            if None in mids:
                return
            before = np.array(mids)
            trader.observe(self)
            moves = trader.decide(before)
            self._ask(account, trader, self._sum(moves), before, aim)
            for move in moves:
                for book in np.nonzero(move)[0]:
                    self._cross(trader, account, int(book), int(move[book]), "cross")
        else:
            mids = [self._mid(work, book) for book in range(self.n)]
            if None in mids:
                self._expire(work)  # it cannot decide, and its patience goes on running
                return
            before = np.array(mids)
            trader.observe(work.view)
            need = self._need(trader, trader.decide(before))
            self._ask(account, trader, need, before, aim)
            if work.execution.urgent_closes and isinstance(trader, StateTrader) and (
                    trader.weights is None or not len(trader.weights)):
                self._urgent(work, need)
            else:
                self._work(work, need)
        if trader.record:
            trader.history.append((self.time, before, trader.position.copy(), trader.signal()))

    def _sum(self, moves: list[np.ndarray]) -> np.ndarray:
        """The moves added up, signed lots a book."""
        need = np.zeros(self.n, dtype=int)
        for move in moves:
            need = need + np.asarray(move, dtype=int)
        return need

    def _need(self, trader: BookTrader, moves: list[np.ndarray]) -> np.ndarray:
        """The signed lots a book a trader with an execution needs: its moves added up. A cross-sectional trader that
        is not cost-aware gives none between rebalances without meaning that it needs none: its need is its last target
        less its position, which at a rebalance is what it gave."""
        if (isinstance(trader, CrossSectionTrader) and not trader._cost_aware and trader._target is not None
                and trader.warming is None):
            return np.asarray(trader._target - trader.position, dtype=int)
        return self._sum(moves)

    def _ask(self, account: Account, trader: BookTrader, need: np.ndarray, mids: np.ndarray,
             aim: np.ndarray) -> None:
        """The paper account: the position the trader asks for at this decision, got now at these mids. It is its
        position and its need, except for a ``StateTrader`` or a ``CrossSectionTrader`` that went toward its aim at
        this decision (``_toward`` made a new aim; ``aim`` is the one before it). Its own rule says where it goes, and
        asks it of the position the paper holds, not of the one its fills left it: inside its band a trader asks for
        nothing, which for its position is what it holds, and for a paper that follows the fills would give up a lot it
        asked for and has not got yet. So the paper asks what a trader filled at once would."""
        if trader._aim is not aim and isinstance(trader, (StateTrader, CrossSectionTrader)):
            target = trader._target if isinstance(trader, CrossSectionTrader) else trader._asked
            gap = trader._aim - account.asked
            beyond = np.sign(gap) * np.maximum(np.abs(gap) - trader.band, 0)
            asked = account.asked + np.rint(np.where(target == 0, gap, beyond)).astype(int)
        else:
            asked = trader.position + need
        self._book(account, asked, mids)

    def _book(self, account: Account, asked: np.ndarray, mids: np.ndarray) -> None:
        """The paper account takes the position ``asked``, got now at these mids."""
        change = asked - account.asked
        account.paper_cash -= float(change @ mids)
        account.asked_lots += int(np.abs(change).sum())
        account.asked = asked
        if self.keep_decisions:
            account.decisions.append((self.time, mids.copy(), asked.copy()))
            account.fills_before.append(len(account.fills))
        else:
            account.unkept += 1

    def _cross(self, trader: BookTrader, account: Account, book: int, lots: int, kind: str) -> int:
        """A market order for these signed lots, by ``_trade``; its fills are in the account. The signed lots it got."""
        mid = self.books[book].mid_price
        self._tap = []
        try:
            self._trade(trader, book, lots)
        finally:
            tapped, self._tap = self._tap, None
        sign = 1 if lots > 0 else -1
        got = 0
        for fill in tapped:
            account.fills.append((self.time, book, sign * fill.size, fill.price, _nan_for(mid), kind))
            got += fill.size
        account.crossed_lots += got
        if kind == "urgent":
            account.urgent_lots += got
        return sign * got

    def _urgent(self, work: _Work, need: np.ndarray) -> None:
        """A trader with no relations: every order withdrawn, every clock cleared, and what it holds closed at once."""
        for book in list(work.order):
            self._withdraw(work, book)
        work.clock.clear()
        work.left[:] = 0
        for book in np.nonzero(need)[0]:
            self._cross(work.trader, work.account, int(book), int(need[book]), "urgent")

    def _work(self, work: _Work, need: np.ndarray) -> None:
        """Each book's need, worked: see the class's docstring."""
        execution, now = work.execution, self.time
        due = execution.reprice is None or now >= work.next_look - TOLERANCE
        work.left = np.array(need, dtype=int)
        for book in range(self.n):
            lots = int(need[book])
            if lots == 0:
                self._withdraw(work, book)
                work.clock.pop(book, None)
                continue
            side = Side.BID if lots > 0 else Side.ASK
            clock = work.clock.get(book)
            if clock is None or clock[1] != side:
                self._withdraw(work, book)  # an order on the other side
                work.clock[book] = (now, side)
                work.counted.discard(book)
            elif self._patience_run(work, clock[0]):
                self._run_out(work, book)
                continue
            self._rest(work, book, side, abs(lots), due)
        if execution.reprice is not None and due:
            work.next_look = now + execution.reprice

    def _rest(self, work: _Work, book: int, side: Side, lots: int, due: bool) -> None:
        """Put a book's working order right: kept if it has the size and the best price (without its own), moved if it
        has the size and a move is ``due``, else placed again with the size needed."""
        price = self._best(work, book, side)
        if price is None:
            return
        order = self.books[book].get_order(work.order[book]) if book in work.order else None
        if order is not None:
            if order.size == lots and (order.price == price or not due):
                return
            self._withdraw(work, book)
        self._place(work, book, side, lots, price)

    def _look(self, work: _Work) -> None:
        """Between decisions, when the reprice clock is due: each book with a clock is crossed if its patience has run
        out, else its order is moved to the best price, or placed if none rests; then the clock is started again."""
        now = self.time
        for book in sorted(work.clock):
            start, side = work.clock[book]
            if self._patience_run(work, start):
                self._run_out(work, book)
                continue
            price = self._best(work, book, side)
            if price is None:
                continue
            if book in work.order:
                if self.books[book].get_order(work.order[book]).price == price:
                    continue
                self._withdraw(work, book)
            self._place(work, book, side, abs(int(work.left[book])), price)
        work.next_look = now + work.execution.reprice

    def _patience_run(self, work: _Work, start: float) -> bool:
        """Whether a clock that started at ``start`` has run the patience."""
        return self.time - start >= work.execution.patience - TOLERANCE

    def _expire(self, work: _Work) -> None:
        """At a time the trader cannot decide (a book has no mid without its own order): the books whose patience has
        run out are crossed, and nothing else is done."""
        for book in sorted(work.clock):
            if self._patience_run(work, work.clock[book][0]):
                self._run_out(work, book)

    def _run_out(self, work: _Work, book: int) -> None:
        """A need whose patience has run out: its order withdrawn, and what is left of it sent as a market order. If
        that took all of it the need is over (its clock is cleared and its wait kept), else the rest is crossed at the
        next look. A need is counted as crossed once, however many looks it takes."""
        self._withdraw(work, book)
        if book not in work.counted:
            work.counted.add(book)
            work.account.orders_crossed += 1
        work.left[book] -= self._cross(work.trader, work.account, book, int(work.left[book]), "cross")
        if work.left[book] == 0:
            start = work.clock.pop(book)[0]
            work.account.waits.append((self.time, self.time - start))

    def _place(self, work: _Work, book: int, side: Side, lots: int, price: int) -> None:
        """A working order for these lots at this price. Joining the best price on its own side cannot trade."""
        order = limit_order(int(price), int(lots), self.time, side)
        if super().submit(book, order, impact=False):
            raise RuntimeError("a working order traded as it was placed")
        work.order[book] = order.order_id  # the id, not the order: the book may hold another object for its rest
        self._working[book][order.order_id] = work
        work.account.orders_placed += 1

    def _withdraw(self, work: _Work, book: int) -> None:
        """A working order out of its book, if there is one."""
        order_id = work.order.pop(book, None)
        if order_id is not None:
            del self._working[book][order_id]
            super().submit(book, cancel_order(order_id, self.time), impact=False)
            work.account.orders_withdrawn += 1

    def _stop(self, work: _Work) -> None:
        """A trader that is no longer active: its orders withdrawn, and its needs and clocks dropped. What its paper
        account held that it did not get is given up at the marks now: a trader that has stopped asks for the position
        it holds, and the paper would otherwise go on holding the lots it dropped, whatever the price did."""
        for book in list(work.order):
            self._withdraw(work, book)
        work.clock.clear()
        work.left[:] = 0
        if (work.account.asked != work.trader.position).any():
            self._book(work.account, work.trader.position.copy(), self.marks_for(work.trader))

    def run(self, until: float, every: float = 1.0) -> pd.DataFrame:
        """As ``BookMarket.run``, with a trader that has an execution valued at the marks without its own orders
        (``marks_for``)."""
        rows, next_sample = [], self.time + every
        while self.time < until:
            self.step()
            while self.time >= next_sample:
                row = {"time": next_sample}
                row |= {f"mid_{i}": book.mid_price for i, book in enumerate(self.books)}
                row |= {f"fair_{i}": value for i, value in enumerate(self.fair.fairs)}
                marks = self.marks()
                row |= {f"value_{j}": trader.value(self.marks_for(trader) if id(trader) in self._works else marks)
                        for j, trader in enumerate(self.traders)}
                rows.append(row)
                next_sample += every
        return pd.DataFrame(rows)


class _Marked:
    """A market as one trader is marked in it: every attribute is the market's, as it is now, except ``marks()``, which
    is ``marks_for`` that trader. So ``score_holdings`` scores it as it scores any market, at the marks without the
    trader's own orders."""

    def __init__(self, market: PassiveBookMarket, trader: BookTrader):
        self._market, self._trader = market, trader

    def marks(self) -> np.ndarray:
        return self._market.marks_for(self._trader)

    def __getattr__(self, name):
        return getattr(self.__dict__["_market"], name)


FILL_KINDS = ("market", "limit", "cross", "urgent")  # the kinds of fill a row of ``Account.fills`` has


class Lot(NamedTuple):
    """One lot of a shortfall (see ``shortfall_parts``): its ``kind`` (``"got"``, ``"back"``, ``"unasked"`` or
    ``"open"``), ``book``, ``sign`` (+1 if the paper held it above the position, a lot to buy, -1 if below) and
    ``cost``, its part of the shortfall in ticks. ``origin`` is how it came to be: ``"ask"`` (the paper asked it),
    ``"fill"`` (the trader traded it, ahead of the paper) or ``"carried"`` (in the gap when the span began, valued at
    the trader's marks then). ``fill`` is, for a got lot, the kind of fill that got it, else ``""``. ``opened`` and
    ``closed`` are times (``closed`` is nan for an open lot)."""
    kind: str
    book: int
    sign: int
    cost: float
    origin: str
    fill: str
    opened: float
    closed: float


@dataclass(eq=False)
class ShortfallParts:
    """What a shortfall is made of (see ``shortfall_parts``): every ``lot``, and for each kind of lot the count and the
    sum of its costs in ticks a lot. ``got_lots_by_fill`` and ``got_cost_by_fill`` split the lots got by the kind of
    fill that got them (all four kinds are there, with 0 for those with no lot). ``shortfall`` is the account's own
    paper P&L less P&L over the span, worked out apart from the lots."""
    lots: list
    shortfall: float
    got_lots: int
    got_cost: float
    back_lots: int
    back_drift: float
    unasked_lots: int
    unasked_cost: float
    open_lots: int
    open_drift: float
    got_lots_by_fill: dict
    got_cost_by_fill: dict

    @property
    def total(self) -> float:
        """The four sums added up: what the shortfall is, lot by lot."""
        return self.got_cost + self.back_drift + self.unasked_cost + self.open_drift

    @property
    def residual(self) -> float:
        """``shortfall`` less ``total``: 0 to rounding unless something changed the trader's position or cash that the
        account does not record (a trader changed by hand, say)."""
        return self.shortfall - self.total

    @property
    def got_a_lot(self) -> float:
        """The mean cost of the lots got (0 with none): what working an order cost, a lot that was worked."""
        return self.got_cost / self.got_lots if self.got_lots else 0.0

    @property
    def notes(self) -> list[str]:
        """Sentences about what the parts show that a reader should not miss: lots the trader traded ahead of the paper
        and traded back, and a residual."""
        notes = []
        if self.unasked_lots:
            verb = "lot was" if self.unasked_lots == 1 else "lots were"
            notes.append(f"{self.unasked_lots} {verb} traded and traded back with no ask between, a position ahead "
                         f"of the paper's, at a cost of {self.unasked_cost:.1f} ticks: not what a signal asked for")
        if abs(self.residual) > 1e-6 * (1.0 + abs(self.shortfall)):
            notes.append(f"the parts miss the shortfall by {self.residual:.6g}: the trader's position or cash changed "
                         "in a way its account does not record")
        return notes


def shortfall_parts(market: PassiveBookMarket, trader: BookTrader, since: Since) -> ShortfallParts:
    """What the shortfall of a trader, its paper P&L less its P&L, is made of, from ``since`` (``market.since(trader)``,
    taken at the start of the span) to the market's time now, lot by lot.

    A shortfall is the sum of its lots' costs only where every lot asked is got. So the lots are told apart, in each
    book and first in first out. Each book has a gap between the paper and the position: ``asked`` less ``position``. It
    starts as the lots between the two at ``since`` (the paper ahead of the position, or the position ahead of the
    paper), valued at the trader's marks then (``marks_for``). Every lot the paper asks (a change of ``asked`` at a
    decision, at that decision's mid) and every lot the trader trades (a fill, at its price) either adds a lot to the
    gap or closes its oldest lot. A lot closed is:

    - ``got``: a lot asked and then filled, or filled and then asked. Its cost is the fill's sign (a taker's, a maker's)
      times the fill's price less the decision's mid: what working that lot cost against the mid when it was asked.
    - ``back``: a lot asked and taken back by a later ask of the other sign before a fill reached it (fills of older
      lots may have come between). The trader never had it:
      its cost is what the paper made on it, the sign it asked times the later decision's mid less the earlier one's,
      which the paper made and the trader did not, so it is part of the shortfall.
    - ``unasked``: a lot the trader traded and traded back with no ask between, a position ahead of the paper's. Its
      cost is the second fill's sign times its price less the first's. It is rare: seen for a trader whose band is half
      a lot (pairs with a band of 0.5), where rounding a half makes the trader that fills in parts take a lot more than
      its paper asks, and ``notes`` says when there are any.
    - ``open``: a lot of the gap left at the end, marked at ``marks_for`` now: the sign it has times the mark less the
      mid it was asked at (or the price it was traded at).

    So ``got_cost + back_drift + unasked_cost + open_drift`` is the shortfall, for every trader whose trades the account
    records: worked or not, with a band or without, stopped or not (a stop is a decision that gives up what was not got,
    at the marks then). It is exact to rounding, and ``shortfall`` is computed apart from the lots to say so
    (``residual``, with a line in ``notes`` where a position or cash changed in a way the account did not record). It is
    the sum of the lots got alone only if nothing is taken back, unasked or open, and then a shortfall a lot is the cost
    of a working order. A trader that asks for lots and takes them back before they fill has the paper's gain on them in
    its shortfall, which no working order cost. On a queue follower at a rate of 0.02 (10,000 units, seeds 1 to 8) the
    trader got 2,380 of the 5,108 lots a seed its paper asked, at -0.17 a lot, and the rest of its shortfall of +0.15 a
    lot was the paper's gain on lots taken back.

    The order of events is the order the account kept, not the order of the times. In a step a model's order fills the
    working orders it meets, then each trader decides and crosses (so another trader's market order can fill this
    trader's working order after its decision, at the same time), then a look between decisions crosses, and a stop
    comes last. ``Account.fills_before`` has each decision's place among the fills.

    The market's ``keep_decisions`` must have been on from ``since`` for every decision: the lots are made of them. If
    it was off for any, a ValueError says so. A fill row of several lots is that many lots at its price.
    """
    account = market.account(trader)
    if not isinstance(since, Since) or since.account is not account:
        raise ValueError("since is what market.since(trader) gave for this trader at the start of the span")
    if account.unkept != since.unkept:
        raise ValueError(f"the shortfall's parts are made of the decisions the market keeps, and "
                         f"{account.unkept - since.unkept} of this trader's decisions in the span were made with "
                         "keep_decisions off: set market.keep_decisions = True before the span's first decision")
    marks = market.marks_for(trader)
    lots: list[Lot] = []
    # a book's gap, oldest lot first: (sign, origin, value, kind of fill, time). Its lots all have the sign of the gap
    gaps = [deque() for _ in range(market.n)]
    gap = since.asked - since.position
    for book in range(market.n):
        lot = (1 if gap[book] > 0 else -1, "carried", float(since.marks[book]), "", since.time)
        gaps[book].extend([lot] * abs(int(gap[book])))

    def trade(row) -> None:
        """A fill: each lot of it closes the oldest lot of the gap with its sign, else adds one of the other sign."""
        time, book, signed, price, _, kind = row
        sign, queue = (1 if signed > 0 else -1), gaps[book]
        for _ in range(abs(int(signed))):
            if queue and queue[0][0] == sign:
                side, origin, value, _, opened = queue.popleft()
                if origin == "fill":  # a lot the trader traded ahead of the paper, traded back
                    lots.append(Lot("unasked", book, side, sign * (price - value), origin, "", opened, time))
                else:  # a lot asked, filled
                    lots.append(Lot("got", book, side, sign * (price - value), origin, kind, opened, time))
            else:
                queue.append((-sign, "fill", float(price), kind, time))

    def ask(time: float, book: int, mid: float, change: int) -> None:
        """A change of the position asked for: each lot of it takes back the oldest lot of the gap that has the other
        sign, else adds one of its own."""
        sign, queue = (1 if change > 0 else -1), gaps[book]
        for _ in range(abs(change)):
            if queue and queue[0][0] != sign:
                side, origin, value, fill, opened = queue.popleft()
                if origin == "fill":  # a lot the trader traded ahead of the paper, now asked
                    lots.append(Lot("got", book, side, sign * (value - mid), origin, fill, opened, time))
                else:  # a lot asked, taken back before it filled: what the paper made on it
                    lots.append(Lot("back", book, side, side * (mid - value), origin, "", opened, time))
            else:
                queue.append((sign, "ask", mid, "", time))

    previous, seen = since.asked, since.fills
    for index in range(since.decisions, len(account.decisions)):
        for row in account.fills[seen:account.fills_before[index]]:
            trade(row)
        seen = account.fills_before[index]
        time, mids, asked = account.decisions[index]
        change = asked - previous
        previous = asked
        for book in np.nonzero(change)[0]:
            ask(time, int(book), float(mids[book]), int(change[book]))
    for row in account.fills[seen:]:
        trade(row)
    for book, queue in enumerate(gaps):
        for sign, origin, value, _, opened in queue:
            lots.append(Lot("open", book, sign, sign * (float(marks[book]) - value), origin, "", opened, math.nan))
    paper = (account.paper_cash - since.paper_cash) + float(account.asked @ marks) - float(since.asked @ since.marks)
    pnl = (trader.cash - since.cash) + float(trader.position @ marks) - float(since.position @ since.marks)
    by_kind = {kind: [lot.cost for lot in lots if lot.kind == "got" and lot.fill == kind] for kind in FILL_KINDS}
    kinds = ("got", "back", "unasked", "open")
    sums = {kind: math.fsum(lot.cost for lot in lots if lot.kind == kind) for kind in kinds}
    counts = {kind: sum(1 for lot in lots if lot.kind == kind) for kind in kinds}
    return ShortfallParts(
        lots=lots, shortfall=paper - pnl, got_lots=counts["got"], got_cost=sums["got"], back_lots=counts["back"],
        back_drift=sums["back"], unasked_lots=counts["unasked"], unasked_cost=sums["unasked"],
        open_lots=counts["open"], open_drift=sums["open"], got_lots_by_fill={k: len(v) for k, v in by_kind.items()},
        got_cost_by_fill={k: math.fsum(v) for k, v in by_kind.items()})


def score_execution(market: PassiveBookMarket, trader: BookTrader, burn: float, span: float,
                    horizon: float = 100.0) -> dict[str, float]:
    """Run a market on for ``burn`` and then ``span`` of time, and how a trader in it did over ``span`` alone: what
    ``score_holdings`` gives, with the trader marked at the mids without its own orders, and what its execution cost.

    ``paper`` is the change over the span in what its paper account is worth (``paper_cash`` and ``asked`` at those
    marks): what its signal made. ``shortfall`` is ``paper`` less ``pnl``, what execution cost, and ``shortfall_a_lot``
    is that over ``asked_lots``, the lots the paper account traded (0 if it traded none). ``earned`` is the spread
    earned on passive fills. ``passive_lots``, ``crossed_lots`` (the urgent ones included), ``urgent_lots``,
    ``orders_placed``, ``orders_withdrawn`` and ``orders_crossed`` (needs, each once) are counts over the span, and
    ``wait`` is the mean wait of the needs that ended in it (0 with none). ``markout_1_market`` and the other five are
    the markouts of the passive fills made in the span whose later time has come, a lot, by the kind of order that hit
    them (0 with none). ``costs`` stays the spread paid by crossing, as ``score_holdings`` gives it. ``before_costs`` is
    nan for a trader with an execution: its P&L and its costs leave out what the price did between a decision and a
    fill, which is the signal's in a crossing trader's and not in a worked one's, so ``paper`` is what to set beside a
    crossing trader's ``before_costs``. For a trader with no execution the keys ``score_holdings`` has are its numbers,
    to the last digit, and the market ends in the state it would be in.

    If the market's ``keep_decisions`` is on when this is called, the score also says what the shortfall is made of
    (``shortfall_parts``, over the span, lot by lot and first in first out in each book): ``got_lots`` and ``got_cost``,
    the lots asked and filled (or filled and asked) and what working them cost against the mid at their decisions, and
    ``got_a_lot`` their mean (0 with none); ``back_lots`` and ``back_drift``, the lots asked and taken back by a later
    ask before a fill reached them, and what the paper made on them; ``unasked_lots`` and ``unasked_cost``, lots the
    trader traded and traded back with no ask between (0 but for a trader with a band of half a lot); ``open_lots`` and
    ``open_drift``, the lots of the gap between the paper and the position left at the end, at the marks then.
    ``got_cost + back_drift + unasked_cost + open_drift`` is ``shortfall``, to rounding, for every trader. A shortfall a
    lot is what working an order cost only where every lot asked is got: a trader whose asks are taken back before they
    fill has the paper's gain on them in its shortfall (a queue follower at a rate of 0.02 got -0.17 a lot on the 2,380
    of 5,108 lots it asked, and its shortfall a lot was +0.15). With it off the score has no such keys, as before.
    """
    account = market.account(trader)
    parts = market.keep_decisions
    while market.time < burn:
        market.step()
    since = market.since(trader) if parts else None
    marks = market.marks_for(trader)
    paper = account.paper_cash + float(account.asked @ marks)
    counts = {name: getattr(account, name) for name in (
        "asked_lots", "spread_earned", "passive_lots", "crossed_lots", "urgent_lots", "orders_placed",
        "orders_withdrawn", "orders_crossed")}
    fills, waits = len(account.fills), len(account.waits)
    score = score_holdings(_Marked(market, trader), trader, burn, span, horizon)
    marks = market.marks_for(trader)
    paper = account.paper_cash + float(account.asked @ marks) - paper
    asked_lots = account.asked_lots - counts["asked_lots"]
    ended = [wait for _, wait in account.waits[waits:]]
    if id(trader) in market._works:
        score["before_costs"] = math.nan  # the drift between the decision and the fill is not the signal's
    score |= {"paper": paper, "shortfall": paper - score["pnl"], "asked_lots": asked_lots,
              "shortfall_a_lot": (paper - score["pnl"]) / asked_lots if asked_lots else 0.0,
              "earned": account.spread_earned - counts["spread_earned"]}
    for name in ("passive_lots", "crossed_lots", "urgent_lots", "orders_placed", "orders_withdrawn", "orders_crossed"):
        score[name] = getattr(account, name) - counts[name]
    score["wait"] = sum(ended) / len(ended) if ended else 0.0
    for kind in ("market", "limit"):
        for later in MARKOUT_HORIZONS:
            score[f"markout_{later:g}_{kind}"] = account.markout(later, kind, fills)
    if parts:
        made = shortfall_parts(market, trader, since)
        score |= {"got_lots": made.got_lots, "got_cost": made.got_cost, "got_a_lot": made.got_a_lot,
                  "back_lots": made.back_lots, "back_drift": made.back_drift, "unasked_lots": made.unasked_lots,
                  "unasked_cost": made.unasked_cost, "open_lots": made.open_lots, "open_drift": made.open_drift}
    return score
