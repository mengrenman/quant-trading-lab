"""A feed in LOBSTER's format as the stream of events the study classes take, and a runner for them.

``lob.studies`` reads a market through ``observe`` calls (``observe(t, kind, side, price, size, fills, aggressor,
quotes_before, quotes_after, book)``), which ``lob.simulation.Simulation`` makes for its own events. This module
makes the same calls from frames in LOBSTER's format, the messages and the book after each (``lob.lobster``'s
readers, ``lob.itch.lobster_frames``), so the studies run unchanged on a recorded day.

- ``PriceLevelBook`` is what the studies read of a book (the best quotes, the levels, the depth by price), loaded
  from a row of a book frame, in ticks.
- ``lobster_events`` turns the frames into ``Event`` tuples, whose first ten fields are ``observe``'s arguments, and
  ``Break`` marks where the stream stops being one market (a cross, a trading action).
- ``run`` feeds a ``StudySet`` from them, taking each unit's sample from the book before the event that opens the
  unit, as notebook 14 does (``Tap.on_pre``) and the dashboard's session does.

What an event is, and what that leaves out:

- An add is the arrival of a limit order; a cancel of part or a deletion is a cancel, with the side, price and
  size taken out. Executions that make up one aggressive order are one event: rows of one ``group`` (a column of
  ``lob.itch``'s messages frame) or, without it, consecutive executions at one time against one side, as
  ``lob.lobster.lobster_orders`` groups them. Its aggressor is minus the resting side and its fills are the
  executions' prices and sizes. It is a market order's arrival unless an add of the aggressor's side follows at the
  same time priced at or beyond its worst fill: that is the rest of a limit order and belongs to the event, which is
  then a limit arrival with fills; the worst fill counts the hidden fills of the group too. An add priced behind
  the fills is another order and stays an event of its own (``lob.lobster.lobster_orders`` takes it for the
  remainder). A limit order that crosses the spread and fills in full leaves only its executions, so it is a market
  arrival here (price 0), where a simulator's observer calls it a limit arrival.
- The rows of a group may have deletions, cancels of part and paired reductions between them: ``lob.itch`` joins
  executions across them (a deletion in the middle of a sweep is a self-match-prevention cancel in all
  likelihood). They are inside the event, not events of their own (``interleaved_rows``: 8 on NVDA and 122 on
  CMCSA from 09:30 to 16:00), so an order-to-trade ratio does not count those cancels as messages; any other row
  between a group's executions raises. A paired reduction (the non-printing half of a trade between two displayed
  orders: ``lob.itch``'s ``paired`` column) that leads a group's first execution begins its event, so the
  displayed lots it took out are in that event's quote change; one that leads none is a cancel event of its own,
  flagged ``paired`` (a trade whose printed half is hidden). It is never a break: until the review of the ``C``
  rows every one was taken for a cross execution, which reset the studies 123 times in MSFT's session.
- Hidden executions (``lob.itch``'s ``hidden`` frame) are fills of the group they carry, in match-number order, when
  the group has displayed executions; a group of hidden executions alone has no event, since nothing displayed
  changes, and is counted. What that makes of the tape depends on ``hidden_mode``. As the stream gives it
  (``signed``, the default) the tape holds the displayed executions and the hidden ones that a run of displayed
  executions signs, and leaves out the unsigned runs: 11.4 percent of NVDA's and 6.5 percent of CMCSA's executed
  shares in the session. The studies cannot tell a hidden fill from a displayed one (``Fill.hidden`` says which it
  is, and ``Fill.priced`` which are executions with price), ``TradeDelta``'s ``coverage`` reads 1.0 whatever is
  left out, and its accuracy mixes the two kinds of lot (the quote rule at lag 0 on NVDA: 99.59 percent on the lots
  it scores, 99.80 on the displayed ones). The tape changes what is measured: the tick rule's accuracy on the
  same displayed lots of CMCSA is 62.1 percent on the displayed tape, 65.0 on the stream's and 68.5 on the full
  tape; and CMCSA's VPIN with true sides, at a fixed V of 5,000, is 0.401 on the stream's tape against a fair-coin
  null of 0.383, and 0.379 against 0.373 on the displayed tape: 1.8 and 0.4 standard deviations of the null over
  eight permutations (3.2 and 0.75 before executions were joined across deletions, which moved the null: the
  figure is rough). A figure that reads the flow names the mode beside it, computes its null in the same mode,
  and prints the unsigned share (``hidden_unsigned_shares`` against ``displayed_shares`` and
  ``hidden_fill_shares``, which ``Run.stream_counts`` carries).
- A cross or a trading action is no event. It is a ``Break`` and the runner starts the studies afresh at a cross
  trade and a trading action (``DEFAULT_RESET``). The book can change at a cross (a displayed order executed in
  it), so the next event's quotes before are the touch the break row shows. A ``cross execution`` break is a
  displayed order executed at the nanosecond of a cross trade, which has its own break: it does not reset by
  default.
- ``quotes_before`` is the previous event's ``quotes_after``; ``quotes_after`` is the touch in the book row of the
  event's last message (a row is the book after one message, not after one event). A crossed or locked touch is
  passed as no quote at all, ``(None, 0, None, 0)``, and counted; ``PriceLevelBook`` treats such a book the same
  way. The first event of a file that starts from a book already full has no quotes before: they are its quotes
  after (the same object, which says that nothing is known to have changed). Several events share a nanosecond
  (81,783 of NVDA's 946,327 in the session), and an event's quotes before are then a state inside the nanosecond:
  for 0.3 percent of NVDA's events they differ from the touch before the nanosecond, and one increment of order
  flow imbalance a nanosecond leaves its R2 against the mid change at 0.2341 from 0.2338.
- A replace is a deletion and a new order. They are two events, flagged ``replace`` 1 and 2, so a count of messages
  can take the pair as one (``run(..., replaces_as_one=True)``, which also hands the deletion the touch and the
  book after the new order, as one change) or as two. As two, the state between them reaches the studies: 4 zero-length
  recoveries and 195 more episode starts of 11,235 in ``Resilience`` on NVDA.
- ``size`` of an arrival is the whole size of the order: the fills and the remainder.

What the figures read from these events do not say by themselves (measured on NVDA, a spread of 6 ticks at the
median, and CMCSA, 1 tick, 09:40 to 15:50):

- Distances are in ticks of a cent, so ``Resilience``'s reach (``ticks`` 3) is a different thing on each stock: it
  holds 2 occupied ask levels at the median on NVDA and 3 on CMCSA, and its abandoned share is 0.66 on NVDA and
  0.13 on CMCSA (a score under 1 in 68 and 5 percent of units). A difference in resilience between stocks measures
  the tick-to-spread ratio; print ``ticks`` in spreads and the abandoned share beside every such figure.
- A book frame has ten occupied levels, which span a median 11 ticks on NVDA and 9 on CMCSA; ``BookImbalance`` counts
  occupied levels, as the simulator's does. Sizes are shares: nothing here is in the simulator's lots.
- The grouping of executions into aggressive orders moves a count of them, and so a figure per order, by up to 11
  percent (``merge_within`` of 100 microseconds on NVDA: 15,612 aggressive orders in the session become 13,856, and an
  order-to-trade ratio rises 10.5 percent): report both groupings.

The counts that the stream keeps (``EventStream.counts``) are in ``COUNTS``. Left out: the order of events inside
one nanosecond beyond the file's order, the identity of orders (fills carry only price and size), and iceberg
orders (a file that shows only displayed sizes does not show them).
"""

from __future__ import annotations

from typing import Any, Callable, Collection, Iterable, Iterator, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from lob.lobster import (DELETION, EXECUTION, HIDDEN_EXECUTION, LOBSTER_TICK, MESSAGE_COLUMNS, PARTIAL_CANCEL,
                        SUBMISSION, book_columns)
from lob.order import Side
from lob.studies import ARRIVAL_LIMIT, ARRIVAL_MARKET, CANCEL, Study, StudySet

CROSS, HALT = 6, 7  # LOBSTER's types for a cross trade and a trading halt
BREAK_REASONS = ("cross", "cross execution", "halt")
DEFAULT_RESET = ("cross", "halt")  # a cross execution is at a cross: its cross trade (Q) is the break
HIDDEN_MODES = ("signed", "none", "unknown")
NO_QUOTE = (None, 0, None, 0)
_PLACEHOLDER = 9_000_000_000  # LOBSTER writes 9,999,999,999 and -9,999,999,999 for a level that is empty
COUNTS = (
    "messages", "hidden_trades", "hidden_signed_trades", "hidden_signed_shares", "hidden_unsigned_runs",
    "hidden_unsigned_trades", "hidden_unsigned_shares",
    "hidden_orphans",                                   # known before the first event, from the whole hidden frame
    "events", "limit", "market", "cancel",              # events, and by kind
    "aggressive", "remainders", "fills", "hidden_fills",  # events that traded; those that also rest; fills in them
    "displayed_shares", "hidden_fill_shares",           # lots in the fills of the events, by kind of fill
    "hidden_unknown_events",                            # hidden_mode "unknown": runs of hidden trades alone
    "replace_deletions", "replace_adds", "adds_behind_the_fills",
    "paired_rows", "paired_events", "interleaved_rows",
    "breaks", "break_cross", "break_cross_execution", "break_halt",
    "crossed_quotes", "unknown_before",
)


class Fill(NamedTuple):
    """What the studies read of a fill (the price and the size), and where it came from."""

    price: Any  # in ticks: an int where the price is a whole number of ticks
    size: int
    hidden: bool = False  # a trade against a hidden order (a ``P``), not a displayed one
    priced: bool = False  # an execution with price (a printable ``C``): its price is a cent off its order's


class Event(NamedTuple):
    """One call of ``observe``: its first ten fields are its arguments; the rest say where the event is from."""

    t: float  # seconds after midnight, of the event's first message
    kind: int
    side: Side
    price: Any
    size: int
    fills: tuple
    aggressor: int
    quotes_before: tuple
    quotes_after: tuple
    book: Any  # a PriceLevelBook at the book row after the event; valid until the stream moves on
    first: int  # the first and last message rows of the event, in the messages frame
    last: int  # the book row after the event is ``last`` (an event of hidden trades alone has no row: first - 1)
    replace: int  # 0, or 1 for the deletion of a replace and 2 for its new order
    hidden: int  # hidden executions among the fills
    ns: Optional[int]  # nanoseconds after midnight when the frame has them
    paired: int = 0  # 1 for a paired reduction that is no part of an aggressive order: a cancel of its own


class Break(NamedTuple):
    """A row that is no event: a cross (``reason`` is ``cross`` for a cross trade that names no order,
    ``cross execution`` for a displayed order executed in a cross) or a trading action (``halt``)."""

    t: float
    ns: Optional[int]
    row: int
    reason: str


def _ticks(price: float, tick: int):
    """A price in LOBSTER's units as ticks: an int if it is a whole number of ticks, else a float."""
    value = price / tick
    whole = round(value)
    return whole if whole == value else value


# ---- The book ----

class BookRows:
    """The levels of a book frame as arrays: the prices in ticks (NaN where a level is empty) and the sizes.

    ``lob.lobster``'s frames have the columns of ``book_columns(levels)``, an empty level with a NaN price (or its
    file placeholder) and size 0. Levels are listed best first, ask prices rising and bid prices falling.
    """

    def __init__(self, book: pd.DataFrame, tick: int = LOBSTER_TICK):
        if tick <= 0:
            raise ValueError("the tick must be positive")
        width = book.shape[1]
        if width == 0 or width % 4:
            raise ValueError(f"a book frame has 4 columns a level (ask and bid price and size); this one has {width}")
        levels = width // 4
        missing = [name for name in book_columns(levels) if name not in book.columns]
        if missing:
            raise ValueError(f"the book frame has no column {missing[0]!r}: {width} columns are {levels} levels, "
                             f"{book_columns(levels)[:4]}, ...")
        self.levels, self.tick, self.n = levels, tick, len(book)
        arrays = {}
        for kind in ("ask_price", "ask_size", "bid_price", "bid_size"):
            columns = [book[f"{kind}_{i}"].to_numpy(np.float64) for i in range(1, levels + 1)]
            arrays[kind] = np.column_stack(columns) if len(book) else np.zeros((0, levels))
        for side in ("ask", "bid"):
            price, size = arrays[f"{side}_price"], arrays[f"{side}_size"]
            empty = ~(size > 0) | ~np.isfinite(price) | (np.abs(price) >= _PLACEHOLDER)
            price[empty] = np.nan
            size[empty] = 0.0
            price /= tick
        self.ask_price, self.ask_size = arrays["ask_price"], arrays["ask_size"]
        self.bid_price, self.bid_size = arrays["bid_price"], arrays["bid_size"]
        self._top: Optional[tuple] = None
        self._crossed: Optional[list] = None

    def _tops(self) -> tuple:
        if self._top is None:
            self._top = (self.bid_price[:, 0].tolist(), self.bid_size[:, 0].tolist(),
                         self.ask_price[:, 0].tolist(), self.ask_size[:, 0].tolist())
            self._crossed = (self.bid_price[:, 0] >= self.ask_price[:, 0]).tolist()  # False where either is NaN
        return self._top

    def crossed(self, i: int) -> bool:
        """Whether the touch of row ``i`` is crossed or locked (best bid at or above best ask)."""
        self._tops()
        return self._crossed[i]

    def quotes(self, i: int) -> tuple:
        """The touch of row ``i`` as the studies take it: (best bid, lots there, best ask, lots there)."""
        bid_price, bid_size, ask_price, ask_size = self._tops()
        bid, ask = bid_price[i], ask_price[i]
        if bid == bid:  # not NaN
            whole = int(bid)
            bid, lots_bid = (whole if whole == bid else bid), int(bid_size[i])
        else:
            bid, lots_bid = None, 0
        if ask == ask:
            whole = int(ask)
            ask, lots_ask = (whole if whole == ask else ask), int(ask_size[i])
        else:
            ask, lots_ask = None, 0
        return bid, lots_bid, ask, lots_ask

    def side(self, i: int, bids: bool) -> list:
        """The occupied levels of row ``i`` on one side as (price in ticks, lots), best first."""
        if bids:
            prices, sizes = self.bid_price[i].tolist(), self.bid_size[i].tolist()
        else:
            prices, sizes = self.ask_price[i].tolist(), self.ask_size[i].tolist()
        out = []
        for price, size in zip(prices, sizes, strict=True):
            if size > 0:
                whole = int(price)
                out.append((whole if whole == price else price, int(size)))
        return out


class PriceLevelBook:
    """What the study classes read of a book, and nothing more: the quotes, the levels, the depth by price.

    Prices are in ticks (a cent is ``lob.lobster.LOBSTER_TICK`` in LOBSTER's units), so ``tick_size`` is 1; the
    book keeps the levels its frame has (10 for LOBSTER's usual files), so a study that reads past them sees less
    than a book that holds every order. A crossed or locked book (best bid at or above best ask) has no quotes:
    ``best_bid``, ``best_ask``, ``mid_price`` and ``spread`` are ``None`` and ``depth_profile`` is empty.

    Load it with explicit levels, ``PriceLevelBook(bids, asks)`` or ``load``, lists of (price, lots) best first, or
    point it at a row of ``BookRows`` with ``read`` (cheap: the levels are only built when something asks for them).
    """

    tick_size = 1

    def __init__(self, bids: Iterable[tuple] = (), asks: Iterable[tuple] = ()):
        self.load(bids, asks)

    def load(self, bids: Iterable[tuple], asks: Iterable[tuple]) -> "PriceLevelBook":
        """Use these levels, lists of (price in ticks, lots) best first."""
        self.rows: Optional[BookRows] = None
        self._index = -1
        self._set(list(bids), list(asks))
        return self

    def read(self, rows: BookRows, index: int) -> "PriceLevelBook":
        """Point at row ``index`` of ``rows``."""
        if not 0 <= index < rows.n:
            raise IndexError(f"row {index} of a book frame of {rows.n} rows")
        self.rows, self._index, self._ready = rows, index, False
        return self

    def at(self, index: int) -> "PriceLevelBook":
        """Point at another row of the rows this book reads."""
        if self.rows is None:
            raise ValueError("this book was loaded with levels, not from a frame")
        return self.read(self.rows, index)

    @property
    def row(self) -> int:
        """The row of the frame this book shows (-1 if it was loaded with levels)."""
        return self._index

    def _set(self, bids: list, asks: list) -> None:
        self._bids, self._asks = bids, asks
        self._bid_depth, self._ask_depth = dict(bids), dict(asks)
        bid = bids[0][0] if bids else None
        ask = asks[0][0] if asks else None
        self._crossed = bid is not None and ask is not None and bid >= ask
        self._best = (None, None) if self._crossed else (bid, ask)
        self._ready = True

    def _need(self) -> None:
        if not self._ready:
            rows, i = self.rows, self._index
            self._set(rows.side(i, True), rows.side(i, False))

    @property
    def best_bid(self):
        self._need()
        return self._best[0]

    @property
    def best_ask(self):
        self._need()
        return self._best[1]

    @property
    def spread(self):
        self._need()
        bid, ask = self._best
        return None if bid is None or ask is None else ask - bid

    @property
    def mid_price(self) -> Optional[float]:
        self._need()
        bid, ask = self._best
        return None if bid is None or ask is None else (bid + ask) / 2.0

    @property
    def _bid_depth_at(self) -> dict:
        self._need()
        return self._bid_depth

    @property
    def _ask_depth_at(self) -> dict:
        self._need()
        return self._ask_depth

    def depth_profile(self, side: Side, n_levels: int = 10) -> list:
        """The best ``n_levels`` occupied levels of a side as (price, lots), best first."""
        self._need()
        if self._crossed:
            return []
        return (self._bids if side == Side.BID else self._asks)[:n_levels]

    def __repr__(self) -> str:
        self._need()
        return f"PriceLevelBook(bids={self._bids[:3]}, asks={self._asks[:3]})"


# ---- The stream ----

class EventStream:
    """The events of a pair of frames, as ``lobster_events`` makes them: iterate it for ``Event`` and ``Break``.

    ``counts`` (the names are ``COUNTS``) holds what is known before the first event and, as the iteration goes,
    what has been yielded; iterating again starts the count again. The counts of events are of what the generator
    has produced: it counts an event just before it yields it, so a consumer that stops at a time (``takewhile``)
    has pulled one event more than it used.
    """

    def __init__(self, messages: pd.DataFrame, book: pd.DataFrame, hidden: Optional[pd.DataFrame] = None,
                 tick: int = LOBSTER_TICK, from_empty: bool = False, hidden_mode: str = "signed"):
        if hidden_mode not in HIDDEN_MODES:
            raise ValueError(f"hidden_mode is one of {HIDDEN_MODES}, not {hidden_mode!r}")
        missing = [name for name in MESSAGE_COLUMNS if name not in messages.columns]
        if missing:
            raise ValueError(f"the messages frame has no column {missing[0]!r}")
        if len(messages) != len(book):
            raise ValueError(f"{len(messages)} messages and {len(book)} book rows: one row after each message")
        self.tick, self.from_empty, self.hidden_mode = tick, from_empty, hidden_mode
        self.rows = BookRows(book, tick)
        n = len(messages)
        self._time = messages["time"].to_numpy(np.float64)
        self._ns = messages["ns"].to_numpy(np.int64) if "ns" in messages else None
        if n and not np.isfinite(self._time).all():
            raise ValueError("a message has no time")
        clock = self._ns if self._ns is not None else self._time
        self._clock = clock
        if n > 1 and (np.diff(clock) < 0).any():
            raise ValueError("message times go backwards: a stream's clock never decreases")
        self._kind = messages["type"].to_numpy(np.int64)
        known = np.isin(self._kind, (SUBMISSION, PARTIAL_CANCEL, DELETION, EXECUTION, HIDDEN_EXECUTION, CROSS, HALT))
        if not known.all():
            raise ValueError(f"message type {int(self._kind[~known][0])} is not one of LOBSTER's 1 to 7")
        self._size = messages["size"].to_numpy(np.int64)
        self._price = messages["price"].to_numpy(np.int64)
        self._direction = messages["direction"].to_numpy(np.int64)
        self._order_id = messages["order_id"].to_numpy(np.int64)
        book_rows = self._kind <= EXECUTION
        if not np.isin(self._direction[book_rows | (self._kind == HIDDEN_EXECUTION)], (-1, 1)).all():
            raise ValueError("the direction of an order or an execution is 1 (buy) or -1 (sell)")
        self._group = messages["group"].to_numpy(np.int64) if "group" in messages else None
        self._match = messages["match"].to_numpy(np.int64) if "match" in messages else None
        self._replace = messages["replace"].to_numpy(bool) if "replace" in messages else None
        self._paired = messages["paired"].to_numpy(bool) if "paired" in messages else None
        self._priced = messages["priced"].to_numpy(bool) if "priced" in messages else None
        self._span = self._group_spans()
        self._hidden_by_group, self._unknown, static = self._join_hidden(hidden)
        static["messages"] = n
        self._static = static
        self.counts = dict.fromkeys(COUNTS, 0) | static

    def _group_spans(self) -> dict:
        """The first and last row of each group of displayed executions: where its event begins and ends."""
        if self._group is None:
            return {}
        rows = np.flatnonzero((self._kind == EXECUTION) & (self._group >= 0))
        if not len(rows):
            return {}
        groups = self._group[rows]
        ordered, first = np.unique(groups, return_index=True)
        _, last = np.unique(groups[::-1], return_index=True)
        return {int(g): (int(rows[a]), int(rows[-1 - b])) for g, a, b in zip(ordered, first, last, strict=True)}

    def _join_hidden(self, hidden: Optional[pd.DataFrame]) -> tuple[dict, list, dict]:
        """The hidden executions of each group that has displayed ones, the runs of hidden executions alone that
        ``hidden_mode`` ``unknown`` makes events of, and the counts of what is in the hidden frame."""
        static = {"hidden_trades": 0, "hidden_signed_trades": 0, "hidden_signed_shares": 0, "hidden_unsigned_runs": 0,
                  "hidden_unsigned_trades": 0, "hidden_unsigned_shares": 0, "hidden_orphans": 0}
        by_group: dict = {}
        alone: dict = {}
        if hidden is None or not len(hidden):
            return by_group, [], static
        missing = [name for name in ("group", "match", "price", "size", "aggressor") if name not in hidden.columns]
        if missing:
            raise ValueError(f"the hidden frame has no column {missing[0]!r}")
        if self._group is None or self._match is None:
            raise ValueError("hidden executions join the messages by group and match number: the messages frame "
                             "needs both columns")
        clock_name = "ns" if self._ns is not None else "time"
        if self.hidden_mode == "unknown" and clock_name not in hidden.columns:
            raise ValueError(f"the hidden frame has no column {clock_name!r}, which hidden_mode 'unknown' places "
                             "its events by")
        visible = set(np.unique(self._group[(self._kind == EXECUTION) & (self._group >= 0)]).tolist())
        unsigned = set()
        names = ["group", "match", "price", "size", "aggressor"] + ([clock_name] if clock_name in hidden else [])
        columns = (hidden[name].tolist() for name in names)
        for group, match, price, size, aggressor, *when in zip(*columns, strict=True):
            static["hidden_trades"] += 1
            if group in visible:
                static["hidden_signed_trades"] += 1
                static["hidden_signed_shares"] += size
                if self.hidden_mode != "none":
                    by_group.setdefault(group, []).append((match, price, size))
            elif aggressor == 0:
                unsigned.add(group)
                static["hidden_unsigned_trades"] += 1
                static["hidden_unsigned_shares"] += size
                if self.hidden_mode == "unknown":
                    alone.setdefault(group, []).append((match, price, size, when[0]))
            else:
                static["hidden_orphans"] += 1  # signed by a run whose displayed executions are not in the frame
        static["hidden_unsigned_runs"] = len(unsigned)
        unknown = []
        for fills in alone.values():
            fills.sort(key=lambda item: item[0])
            when = fills[0][3]
            position = int(np.searchsorted(self._clock, when, side="right"))  # after every row at or before its time
            unknown.append((position, when, tuple((m, p, s) for m, p, s, _ in fills)))
        unknown.sort(key=lambda item: (item[0], item[1], item[2][0][0]))
        return by_group, unknown, static

    def __iter__(self) -> Iterator:
        self.counts = dict.fromkeys(COUNTS, 0) | self._static
        return self._generate(self.counts)

    def _generate(self, counts: dict) -> Iterator:
        n = len(self._time)
        time = self._time.tolist()
        ns = self._ns.tolist() if self._ns is not None else None
        kind, size, price = self._kind.tolist(), self._size.tolist(), self._price.tolist()
        direction = self._direction.tolist()
        group = self._group.tolist() if self._group is not None else None
        match = self._match.tolist() if self._match is not None else None
        replace = self._replace.tolist() if self._replace is not None else None
        paired = self._paired.tolist() if self._paired is not None else None
        priced = self._priced.tolist() if self._priced is not None else None
        exact = ns if ns is not None else time
        hidden_by_group, span = self._hidden_by_group, self._span
        unknown = self._unknown if n else []
        rows, tick = self.rows, self.tick
        book = PriceLevelBook()
        if n:
            book.read(rows, 0)
        BID, ASK = Side.BID, Side.ASK

        def touch(r: int) -> tuple:
            if rows.crossed(r):
                counts["crossed_quotes"] += 1
                return (None, 0, None, 0)
            return rows.quotes(r)

        previous = NO_QUOTE if self.from_empty else None
        waiting = 0  # the next run of hidden executions alone to place (hidden_mode "unknown")
        i = 0
        while True:
            while waiting < len(unknown) and unknown[waiting][0] <= i:
                _, when, found = unknown[waiting]
                waiting += 1
                fills = tuple(Fill(_ticks(p, tick), s, True) for _, p, s in found)
                lots = sum(fill.size for fill in fills)
                if previous is None:
                    after = touch(max(i - 1, 0))
                    before = after
                    counts["unknown_before"] += 1
                else:
                    before = after = previous
                counts["events"] += 1
                counts["market"] += 1
                counts["aggressive"] += 1
                counts["fills"] += len(fills)
                counts["hidden_fills"] += len(fills)
                counts["hidden_fill_shares"] += lots
                counts["hidden_unknown_events"] += 1
                book.read(rows, max(i - 1, 0))
                yield Event(when / 1e9 if ns is not None else when, ARRIVAL_MARKET, BID, 0, lots, fills, 0, before,
                            after, book, i, i - 1, 0, len(fills), when if ns is not None else None)
            if i >= n:
                return
            k = kind[i]
            when = ns[i] if ns is not None else None
            if k == CROSS or k == HALT:
                if k == HALT:
                    reason = "halt"
                else:
                    reason = "cross" if int(self._order_id[i]) == 0 else "cross execution"
                previous = touch(i)
                counts["breaks"] += 1
                counts["break_" + reason.replace(" ", "_")] += 1
                yield Break(time[i], when, i, reason)
                i += 1
                continue
            # the rows of an aggressive order: its group's executions, and the paired reductions that lead them
            lead = i
            if paired is not None and paired[i] and group is not None:
                while lead < n and paired[lead] and kind[lead] != EXECUTION:
                    lead += 1
            starts = lead < n and kind[lead] == EXECUTION and (lead == i or exact[lead] == exact[lead - 1])
            if k == SUBMISSION or ((k == PARTIAL_CANCEL or k == DELETION) and not (starts and lead > i)):
                after = touch(i)
                if previous is None:
                    before = after
                    counts["unknown_before"] += 1
                else:
                    before = previous
                d = direction[i]
                side = BID if d == 1 else ASK
                flag = 0
                if replace is not None and replace[i]:
                    flag = 2 if k == SUBMISSION else 1 if k == DELETION else 0
                standing = 1 if paired is not None and paired[i] else 0
                if k == SUBMISSION:
                    event_kind, aggressor = ARRIVAL_LIMIT, d
                    counts["limit"] += 1
                else:
                    event_kind, aggressor = CANCEL, 0
                    counts["cancel"] += 1
                if flag == 1:
                    counts["replace_deletions"] += 1
                elif flag == 2:
                    counts["replace_adds"] += 1
                counts["paired_rows"] += standing
                counts["paired_events"] += standing
                counts["events"] += 1
                book.read(rows, i)
                previous = after
                yield Event(time[i], event_kind, side, _ticks(price[i], tick), size[i], (), aggressor, before, after,
                            book, i, i, flag, 0, when, standing)
                i += 1
                continue
            # executions: one event for the rows of one aggressive order, and the rest of its limit order
            counts["paired_rows"] += lead - i
            first_row, resting = i, direction[lead]
            g = group[lead] if group is not None else -1
            j = lead + 1
            displayed = [lead]
            if g >= 0 and g in span:
                end_exec = span[g][1]
                for r in range(lead + 1, end_exec + 1):
                    if kind[r] == EXECUTION and group[r] == g:
                        if direction[r] != resting:
                            raise ValueError(f"group {g} has executions against both sides (row {r})")
                        displayed.append(r)
                    elif kind[r] == EXECUTION and group[r] >= 0:
                        raise ValueError(f"group {g} is interrupted at row {r} by an execution of group {group[r]}")
                    elif (kind[r] == PARTIAL_CANCEL or kind[r] == DELETION) and not (
                            replace is not None and replace[r]):
                        counts["interleaved_rows"] += 1
                        counts["paired_rows"] += 1 if paired is not None and paired[r] else 0
                    else:
                        raise ValueError(f"group {g} is interrupted at row {r} by a row of type {kind[r]}: only "
                                         "deletions, cancels of part and paired reductions may lie between its "
                                         "executions")
                j = end_exec + 1
            else:
                while (j < n and (kind[j] == EXECUTION or kind[j] == HIDDEN_EXECUTION) and exact[j] == exact[lead]
                       and direction[j] == resting):
                    displayed.append(j)
                    j += 1
            fills = [Fill(_ticks(price[x], tick), size[x], False, bool(priced[x]) if priced is not None else False)
                     for x in displayed]
            displayed_lots = sum(fill.size for fill in fills)
            hidden = hidden_by_group.get(g) if g >= 0 else None
            if hidden:
                joined = [(match[x], fill) for x, fill in zip(displayed, fills, strict=True)]
                joined += [(m, Fill(_ticks(p, tick), s, True)) for m, p, s in hidden]
                joined.sort(key=lambda pair: pair[0])
                fills = [fill for _, fill in joined]
            buying = resting == -1
            prices = [fill.price for fill in fills]
            worst = max(prices) if buying else min(prices)
            last = j - 1
            rest = -1
            if (j < n and kind[j] == SUBMISSION and direction[j] == -resting and exact[j] == exact[last]
                    and not (replace is not None and replace[j])):
                add = _ticks(price[j], tick)
                if (add >= worst) if buying else (add <= worst):
                    rest = j
                else:
                    counts["adds_behind_the_fills"] += 1
            end = rest if rest >= 0 else last
            after = touch(end)
            if previous is None:
                before = after
                counts["unknown_before"] += 1
            else:
                before = previous
            lots = sum(fill.size for fill in fills)
            counts["events"] += 1
            counts["aggressive"] += 1
            counts["fills"] += len(fills)
            counts["hidden_fills"] += len(hidden) if hidden else 0
            counts["displayed_shares"] += displayed_lots
            counts["hidden_fill_shares"] += lots - displayed_lots
            if rest >= 0:
                counts["limit"] += 1
                counts["remainders"] += 1
                event_kind, event_price, lots = ARRIVAL_LIMIT, _ticks(price[rest], tick), lots + size[rest]
            else:
                counts["market"] += 1
                event_kind, event_price = ARRIVAL_MARKET, 0
            book.read(rows, end)
            previous = after
            yield Event(time[first_row], event_kind, BID if buying else ASK, event_price, lots, tuple(fills), -resting,
                        before, after, book, first_row, end, 0, len(hidden) if hidden else 0, when)
            i = end + 1


def lobster_events(messages: pd.DataFrame, book: pd.DataFrame, hidden: Optional[pd.DataFrame] = None,
                   tick: int = LOBSTER_TICK, from_empty: bool = False, hidden_mode: str = "signed") -> EventStream:
    """The stream of events (and breaks) of frames in LOBSTER's format: the messages, the book after each, and
    ``lob.itch``'s hidden executions if there are any.

    ``messages`` needs LOBSTER's six columns and may have ``lob.itch``'s ``ns``, ``match``, ``replace`` and ``group``;
    ``hidden`` needs ``group``, ``match``, ``price``, ``size`` and ``aggressor`` and joins the messages by group.
    ``tick`` is the price step in the frames' units. ``from_empty`` says that the file starts from an empty book (a
    simulator's does, a real day's does not): then the first event's quotes before are empty rather than unknown.
    ``hidden_mode`` says what the hidden executions are in the stream: ``signed`` (the default: those of a group that
    has displayed executions are fills of its event, those of a run alone are in no event), ``none`` (no hidden
    execution is a fill: the tape is the displayed one) or ``unknown`` (as ``signed``, and each run of hidden
    executions alone is an event of its own, a market arrival of aggressor 0 and no quote change, placed after
    the rows at or before its nanosecond; a study that signs by the aggressor adds nothing for it, and
    ``Resilience`` reads aggressor 0 as a sell). The frames are read positionally; times must not decrease.
    """
    return EventStream(messages, book, hidden, tick, from_empty, hidden_mode)


# ---- The runner ----

class Run:
    """What ``run`` did: the unit samples, the counts, and the ``StudySet`` (read its ``values`` for the rest).

    ``counts`` is what the runner saw (events, breaks, resets, samples). ``stream_counts`` is a copy of the
    stream's own counts (``COUNTS``: what hidden volume the frames had and how much of it reached the studies,
    aggressive orders, paired and interleaved rows, replaces, breaks by reason) when ``run`` was given an
    ``EventStream``, else empty. The studies cannot tell a hidden fill from a displayed one and ``TradeDelta``'s
    ``coverage`` counts only the fills it was given, so these counts are where a figure's hidden share is read.
    """

    def __init__(self, studies: StudySet):
        self.studies = studies
        self.samples: list = []  # (unit, {study key: its main value}) for every unit sampled
        self.counts = {"events": 0, "breaks": 0, "resets": 0, "samples": 0}
        self.stream_counts: dict = {}


def run(events: Iterable, studies: "StudySet | Sequence[Study] | None" = None, unit: float = 1.0,
        origin: float = 0.0, listeners: Sequence[Callable] = (), reset_on: Collection[str] = DEFAULT_RESET,
        replaces_as_one: bool = False) -> Run:
    """Feed a ``StudySet`` from a stream of events, as notebook 14 feeds one.

    The clock is ``(time - origin) / unit`` with ``time`` in seconds after midnight (from the nanoseconds if the
    events have them), so unit ``k`` is the ``k``-th ``unit`` seconds from ``origin``, which must not be after the
    first event. When an event opens a new unit the sample of each unit not yet sampled is taken from the book
    *before* that event, the row before its first message (which is how a session that samples on its own and the
    notebooks' ``Tap`` take them), and then the studies observe the event. A set that samples itself would sample
    from the book after the event, so ``studies`` must have ``auto_sample`` False (a list of studies is wrapped in
    a set that has). Two limits of ``StudySet.sample_to`` apply: a gap of more than 100 units drops the units
    before the last 100, and a reset marks every unit up to its own as sampled, so the units between the last
    one sampled and the break's are not sampled.

    A ``Break`` whose reason is in ``reset_on`` starts every study afresh at that time. By default that is a cross
    trade (``cross``) and a trading action (``halt``): a ``cross execution`` is a displayed order executed at a
    cross, which has its cross trade beside it, so it is not a break of its own.

    With ``replaces_as_one`` a replace is one change of the book to every study. Its deletion is passed with the
    touch and the book after its new order, so the state between the two rows, which never existed, reaches no
    study (the review found it starting 195 more recovery episodes of 11,235 on NVDA and 4 of zero length), and
    the new order is passed after it as an event that changes nothing, with ``own=True``, which an order-to-trade
    ratio of scope ``model`` (the default) leaves out, so a replace is one message there. Each ``listener`` is
    called with every event, as the stream made it, after the studies have seen it. A stream that is given to
    ``run`` itself (an ``EventStream``, not a list or a ``takewhile`` of it) has its counts copied into
    ``Run.stream_counts``.
    """
    if unit <= 0:
        raise ValueError("a unit is a positive number of seconds")
    if studies is None or not isinstance(studies, StudySet):
        studies = StudySet(studies, auto_sample=False)
    if studies.auto_sample:
        raise ValueError("the StudySet samples itself after each event; use auto_sample=False so that run can "
                         "sample from the book before the event that opens a unit")
    unknown = set(reset_on) - set(BREAK_REASONS)
    if unknown:
        raise ValueError(f"no such break reason: {sorted(unknown)}")
    result = Run(studies)
    counts, samples = result.counts, result.samples
    origin_ns, unit_ns = round(origin * 1e9), unit * 1e9
    observe, sample_to = studies.observe, studies.sample_to
    before = PriceLevelBook()
    empty = PriceLevelBook()

    def clock(item) -> float:
        t = (item.ns - origin_ns) / unit_ns if item.ns is not None else (item.t - origin) / unit
        if t < 0:
            raise ValueError(f"an event at {item.t} seconds is before the origin, {origin}")
        return t

    def take(item, quotes_before, quotes_after, row, own) -> None:
        t = clock(item)
        counts["events"] += 1
        index = int(t)
        if index > studies.unit:
            if item.first > 0:
                book_before = before.read(item.book.rows, item.first - 1)
            elif item.quotes_before is item.quotes_after:
                book_before = item.book.at(row)  # nothing is known of the book before the file's first row
            else:
                book_before = empty  # a file that starts from an empty book
            for unit_index, values in sample_to(index, book_before, rows=True):
                samples.append((unit_index, values))
                counts["samples"] += 1
        item.book.at(row)
        observe(t, item.kind, item.side, item.price, item.size, item.fills, item.aggressor, quotes_before,
                quotes_after, item.book, own)
        for listener in listeners:
            listener(item)

    def plain(item) -> None:
        take(item, item.quotes_before, item.quotes_after, max(item.last, 0), replaces_as_one and item.replace == 2)

    held = None  # a replace's deletion, waiting for its new order
    for item in events:
        if held is not None:
            if isinstance(item, Event) and item.replace == 2:
                after = item.quotes_after
                take(held, held.quotes_before, after, item.last, False)
                take(item, after, after, item.last, True)
                held = None
                continue
            plain(held)
            held = None
        if isinstance(item, Break):
            counts["breaks"] += 1
            if item.reason in reset_on:
                t = clock(item)
                studies.reset(int(t), t)
                counts["resets"] += 1
            continue
        if replaces_as_one and item.replace == 1:
            held = item
            continue
        plain(item)
    if held is not None:
        plain(held)
    if hasattr(events, "counts"):
        result.stream_counts = dict(events.counts)
    return result
