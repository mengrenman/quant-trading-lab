"""The ladder: what a price-level feed loses, measured on a real order-level stream whose truth is known.

``lob.feeds`` reads a day of orders (frames in LOBSTER's format) as the stream of events the study classes take. A
vendor's feed is a degradation of that stream: levels, not orders, in batches, with trades on a clock of their own.
This module makes those degradations from the same frames, as streams ``lob.feeds.run`` takes, so that a study run
on a rung and on the reference sees the same day through a poorer window and a figure on a rung is like for like with
the reference's (same units, same samples, same session). Nothing is imported by ``lob/__init__.py``.

A ``Day`` reads the frames once; each rung is a function of it, and ``run_rung`` runs studies over one:

- ``reference(day)``: the displayed tape (``hidden_mode`` "none": displayed executions only) with a replace as one
  change (``run``'s ``replaces_as_one``), the stream's own events. The truth the other rungs are read against.
- ``levels(day)``: one event a changed price level, best price first, trades fused: what a per-update feed gives.
- ``batches(day, ms)``: the net change of each level over each batch of real time, and the trades as a stream of
  their own with their own times: what a batched feed gives.
- ``shifted(day, ms, shift_ms)``: batches with the trade stream moved against the book by a constant, a bound on what
  an unsynchronized trade stream does and not a model of any venue.

    day = Day(messages, book, origin=34_200.0, end=57_000.0)        # the frames may be dropped once it is built
    for rung in (reference(day), levels(day), batches(day, 100), shifted(day, 100, 50)):
        result = run_rung(rung, [Spread(), TradeDelta(side="quote", quote_lag=0)])

A collector that counts changes of the book skips the new order of a replace on the reference (``own``) and the
trades of a batch rung (an event with fills whose ``quotes_before`` is its ``quotes_after``: the book it holds, not
a change of it).

What each rung holds. A rung is an iterable of ``Event`` and ``Break`` (``lob.feeds``'s types) and carries the book
its consumer holds (``rows``, a ``BookRows`` of up to ten levels a state), so a study that reads depth reads the
consumer's book and not the day's. Every event's ``first - 1`` and ``last`` are the consumer's states before and
after it, as ``run`` reads them, and the sample of a unit is the state held at its start.

What every rung leaves out. Hidden orders and hidden executions (the tape is the displayed one, so a figure on it
is not one on the hidden-signed tape), iceberg orders, the order of events inside one nanosecond beyond the file's
order, and the identity of orders. The book a consumer holds has the ten levels the frames show, so a study that
reads depth reads no more; a change deeper than that is an update with an unchanged touch (``levels``) or counts
as a changed level (``batches``, from the messages). A rung's ``kind``, ``side``, ``price`` and ``size`` are
meaningful for the reference and for the updates of ``levels``, and placeholders for a batch update (a limit arrival
of size 0 on the bid side): a study that reads them (``OrderToTrade``, ``Resilience`` on a cancel) measures nothing
on a batch rung. The figures of the ladder read the touch, the levels and the fills.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from lob.feeds import CROSS, DEFAULT_RESET, NO_QUOTE, BookRows, Break, Event, PriceLevelBook, Run, lobster_events, run
from lob.lobster import DAY_START, EXECUTION, LOBSTER_TICK
from lob.order import Side
from lob.studies import ARRIVAL_LIMIT, ARRIVAL_MARKET, CANCEL, StudySet

STATE_UPDATE, STATE_TRADE, STATE_BREAK, STATE_START = 0, 1, 2, 3   # what an item of a derived rung is
_CHUNK = 1 << 16
_SIDE = {1: Side.BID, -1: Side.ASK}


# ---- The day ----

class Day:
    """The frames of one stock-day and what every rung shares: the book as arrays, the reference's events as arrays.

    ``messages`` and ``book`` are LOBSTER's frames (``lob.itch.load_frames``' first two, or ``simulated_lobster``'s);
    ``origin`` is the clock's zero in seconds after midnight (09:30:00 by default), where a grid of batches starts,
    and ``end`` the first second not in the session (None: the whole frame, and then the last window of a batch rung
    ends with its last row). Rows from ``end`` on are left out of every rung. A frame without ``ns`` gets one from
    its ``time``, and the reference then runs on that clock too.
    ``from_empty`` says that the file starts from an empty book, as ``lobster_events`` does.

    A ``Day`` keeps the book as arrays and not the frames, which the caller may drop once it is built (the frames of
    a busy stock are about a gigabyte; a ``Day`` and a rung about two and a half, the finest batch more).
    Building a ``Day`` validates the frames as ``lobster_events`` does (it raises the same errors). The first rung
    that needs the reference's events reads the whole stream once (a few seconds for a million rows) and keeps
    them as arrays; a ``Day`` is therefore worth sharing between rungs. The hidden frame is not read.

    ``level[r]`` is the price of the level row ``r`` changed in the book: the row's own ``price``, but for an execution
    with price (a printable ``C``, the ``priced`` column) and an order executed in a cross, whose price is not the
    level they take shares from (``_level_taken``).
    """

    def __init__(self, messages: pd.DataFrame, book: pd.DataFrame, origin: float = DAY_START,
                 end: Optional[float] = None, from_empty: bool = False, tick: int = LOBSTER_TICK):
        if "ns" not in messages.columns:
            messages = messages.assign(ns=np.rint(messages["time"].to_numpy(np.float64) * 1e9).astype(np.int64))
        self.from_empty, self.tick = from_empty, tick
        self.stream = lobster_events(messages, book, None, tick, from_empty, "none")
        self.rows = self.stream.rows
        self.ns = messages["ns"].to_numpy(np.int64)
        self.kind = messages["type"].to_numpy(np.int64)
        self.order_id = messages["order_id"].to_numpy(np.int64)
        self.size = messages["size"].to_numpy(np.int64)
        self.price = messages["price"].to_numpy(np.int64)
        self.direction = messages["direction"].to_numpy(np.int64)
        priced = messages["priced"].to_numpy(bool) if "priced" in messages.columns else np.zeros(len(self.ns), bool)
        self.level = _level_taken(self.rows, self.kind, self.order_id, self.size, self.price, self.direction, priced,
                                  tick)
        self.origin_ns = round(origin * 1e9)
        self.end_ns = int(self.ns[-1]) + 1 if end is None and len(self.ns) else round((end or 0.0) * 1e9)
        self.n = int(np.searchsorted(self.ns, self.end_ns, side="left"))   # the rows of the session
        self._table: Optional[_Table] = None

    @property
    def table(self) -> "_Table":
        """The reference's events and breaks as arrays, read from the stream on first use."""
        if self._table is None:
            self._table = _Table(self)
        return self._table


def _level_taken(rows: BookRows, kind, order_id, size, price, direction, priced, tick: int) -> np.ndarray:
    """The price of the level each row of the frames changed in the book: the row's own price, but for the rows whose
    price is not the level they take shares from.

    ``lob.itch`` writes an execution with price (a printable ``C``, ``priced``) at its own price, a cent off its
    order's, and a displayed order executed in a cross (a row of type 6 with an order) at the cross's price, while the
    book falls at the order's displayed level. The level is read from the book and not from the order's add: it is the
    level of the row's side whose size fell by the row's size between the book after the row before and the book after
    the row. The book shows what changed whatever the order's history, so this is right when the order was placed before
    the frames begin (its add is not among the messages), when a replace made it (its level is the new order's) and
    when the execution is partial (the level falls by the shares executed and is still there). Where the book does not
    show exactly one such level (the first row, whose book before is not in the frames, or a level deeper than the
    ten the frames show) the row keeps its own price, as every other row does.
    """
    level = price.copy()
    taken = np.flatnonzero(((kind == EXECUTION) & priced) | ((kind == CROSS) & (order_id != 0)))
    taken = taken[taken > 0]
    for ask in (False, True):
        r = taken[direction[taken] == (-1 if ask else 1)]
        prices, sizes = (rows.ask_price, rows.ask_size) if ask else (rows.bid_price, rows.bid_size)
        p0, q0, p1, q1 = prices[r - 1], sizes[r - 1], prices[r], sizes[r]
        now = ((p0[:, :, None] == p1[:, None, :]) * q1[:, None, :]).sum(2)      # the size at each price after the row
        fell = np.isfinite(p0) & (q0 - now == size[r][:, None])
        one = fell.sum(1) == 1
        level[r[one]] = np.rint(p0[fell & one[:, None]] * tick).astype(np.int64)
    return level


class _Table:
    """The events of the reference stream (from the first row to ``end``) as arrays, and its breaks.

    Event ``j`` has the stamp ``ns`` (its first row's), the message rows ``first`` to ``last``, the kind, the side
    (1 bid, -1 ask), price, size and aggressor ``lob.feeds`` gave it, whether it is half of a replace (``replace``
    1 for the deletion, 2 for the new order) and, in ``fills``, its fills if it traded. A break stands before event
    ``brk_pos`` (the count of events before it).
    """

    def __init__(self, day: Day):
        ns, first, last, kind, side, price, size, aggressor, replace = ([] for _ in range(9))
        fills: dict = {}
        breaks: list = []
        end = day.end_ns
        for item in day.stream:
            if item.ns >= end:
                break
            if isinstance(item, Break):
                breaks.append((len(ns), item.ns, item.row, item.reason))
                continue
            if item.fills:
                fills[len(ns)] = item.fills
            ns.append(item.ns)
            first.append(item.first)
            last.append(item.last)
            kind.append(item.kind)
            side.append(1 if item.side is Side.BID else -1)
            price.append(item.price)
            size.append(item.size)
            aggressor.append(item.aggressor)
            replace.append(item.replace)
        self.ns = np.array(ns, np.int64)
        self.first = np.array(first, np.int64)
        self.last = np.array(last, np.int64)
        self.kind = np.array(kind, np.int64)
        self.side = np.array(side, np.int64)
        self.price = np.array(price, np.float64)
        self.size = np.array(size, np.int64)
        self.aggressor = np.array(aggressor, np.int64)
        self.replace = np.array(replace, np.int64)
        self.fills = fills
        self.brk_pos = np.array([b[0] for b in breaks], np.int64)
        self.brk_ns = np.array([b[1] for b in breaks], np.int64)
        self.brk_row = np.array([b[2] for b in breaks], np.int64)
        self.brk_reason = [b[3] for b in breaks]
        self.n = len(self.ns)


# ---- A rung ----

@dataclass
class _Feed:
    """A derived rung as arrays, one entry an item of its stream: ``kind`` says whether it is an update of the
    consumer's book, a trade, a break or the (silent) empty state a file that starts empty begins with.

    Every update, reset break and start is a state of the consumer's book, a row of ``rows`` in item order; an update
    is the change from the state before it to its own. A trade and a non-resetting break hold the latest state at
    their stamp. ``changed`` is the price levels an update changed (all depths for batches, inside the frames' window
    for ``levels``).
    """

    rows: BookRows
    kind: np.ndarray
    stamp: np.ndarray
    ekind: np.ndarray
    eside: np.ndarray
    eprice: np.ndarray
    esize: np.ndarray
    eaggressor: np.ndarray
    changed: np.ndarray
    fills: dict
    breaks: dict
    state: np.ndarray = None      # the latest state at each item (-1 before the first)
    owner: np.ndarray = None      # for ``levels``: the reference event each item came from (-1: none)


class Rung:
    """A stream of what a consumer of one kind of feed sees of a day: iterate it for ``Event`` and ``Break``.

    ``name`` says which rung it is, ``replaces_as_one`` what ``run`` should do with replace flags (the reference's
    stream has them; the derived rungs have none), ``rows`` the book the consumer holds, and ``counts`` what the rung
    holds (events, updates, trades, breaks, and what the construction counted: the events ``levels`` split, the ones
    it could not). A rung is re-iterable, and its events share one ``PriceLevelBook``, which ``run`` points at the
    right state.
    """

    def __init__(self, day: Day, name: str, feed: Optional[_Feed], replaces_as_one: bool,
                 counts: Optional[dict] = None):
        self.day, self.name, self.replaces_as_one = day, name, replaces_as_one
        self.origin_ns, self.end_ns = day.origin_ns, day.end_ns
        self._feed = feed
        self.rows = day.rows if feed is None else feed.rows
        self._counts = counts

    def __repr__(self) -> str:
        return f"Rung({self.name!r})"

    @property
    def counts(self) -> dict:
        """What the rung holds, by name; ``run`` copies it into ``Run.stream_counts``."""
        if self._counts is None:
            table = self.day.table
            self._counts = {"events": table.n, "aggressive": len(table.fills),
                            "fills": sum(len(f) for f in table.fills.values()), "breaks": len(table.brk_ns),
                            "replace_deletions": int((table.replace == 1).sum())}
        return self._counts

    @property
    def owner(self) -> Optional[np.ndarray]:
        """For ``levels``: for each item of the stream, the number of the reference event it was made from (a replace
        counted as one, in the order the reference gives them), -1 for an item that is no update. None for the other
        rungs."""
        return None if self._feed is None else self._feed.owner

    def final_book(self) -> PriceLevelBook:
        """The book the consumer holds at the end of the session (the last state before it ends)."""
        book = PriceLevelBook()
        if self._feed is None:
            row = self.day.n - 1
        else:
            row = int(self._feed.state[-1]) if len(self._feed.state) else -1
        if row >= 0:
            book.read(self.rows, row)
        return book

    def __iter__(self) -> Iterator:
        if self._feed is None:
            return self._reference()
        return self._derived()

    def _reference(self) -> Iterator:
        end = self.end_ns
        for item in self.day.stream:
            if item.ns >= end:
                return
            yield item

    def _derived(self) -> Iterator:
        feed, rows = self._feed, self._feed.rows
        book = PriceLevelBook()
        if rows.n:
            book.read(rows, 0)

        def touch(s: int) -> tuple:
            if s < 0 or rows.crossed(s):
                return NO_QUOTE
            return rows.quotes(s)

        held_state, held = -2, None   # the quotes of the state the last update made, to be the next one's before
        m = len(feed.kind)
        for lo in range(0, m, _CHUNK):
            hi = min(lo + _CHUNK, m)
            cols = [a[lo:hi].tolist() for a in (feed.kind, feed.stamp, feed.state, feed.ekind, feed.eside,
                                                feed.eprice, feed.esize, feed.eaggressor)]
            for k, (kind, ns, state, ekind, eside, eprice, esize, eaggressor) in enumerate(zip(*cols, strict=True)):
                item = lo + k
                if kind == STATE_BREAK:
                    row, reason = feed.breaks[item]
                    yield Break(ns / 1e9, ns, row, reason)
                elif kind == STATE_UPDATE:
                    after = touch(state)
                    before = state - 1
                    if before < 0:
                        quotes_before = after                     # nothing is known of the book before the first
                    elif before == held_state:
                        quotes_before = held
                    else:
                        quotes_before = touch(before)
                    held_state, held = state, after
                    yield Event(ns / 1e9, ekind, _SIDE[eside], _price(eprice), esize, feed.fills.get(item, ()),
                                eaggressor, quotes_before, after, book, before + 1, state, 0, 0, ns)
                elif kind == STATE_TRADE:
                    fills = feed.fills[item]
                    q = touch(state)
                    yield Event(ns / 1e9, ARRIVAL_MARKET, _SIDE[eaggressor], 0, sum(f.size for f in fills), fills,
                                eaggressor, q, q, book, state + 1, state, 0, 0, ns)


def _price(p: float):
    """A price in ticks as ``lob.feeds`` has it: an int where it is whole."""
    return int(p) if p == p and int(p) == p else p


class _Rows(BookRows):
    """A ``BookRows`` made of arrays (prices in ticks, NaN where a level is empty) and not of a frame: the states of a
    consumer's book. It sets what ``BookRows`` sets (``levels``, ``tick``, ``n``, the four arrays), so ``quotes``,
    ``side`` and ``crossed`` are ``BookRows``'s own."""

    def __init__(self, ask_price, ask_size, bid_price, bid_size, tick: int):
        self.levels, self.tick, self.n = ask_price.shape[1], tick, len(ask_price)
        self.ask_price, self.ask_size, self.bid_price, self.bid_size = ask_price, ask_size, bid_price, bid_size
        self._top = None
        self._crossed = None


def _states(day: Day, row: np.ndarray) -> _Rows:
    """The consumer's states from the day's book rows (-1: the empty book)."""
    rows = day.rows
    real = row >= 0
    out = []
    for name, empty in (("ask_price", np.nan), ("ask_size", 0.0), ("bid_price", np.nan), ("bid_size", 0.0)):
        a = np.full((len(row), rows.levels), empty)
        a[real] = getattr(rows, name)[row[real]]
        out.append(a)
    return _Rows(*out, rows.tick)


def reference(day: Day, replaces_as_one: bool = True) -> Rung:
    """The displayed tape with a replace as one change: ``lob.feeds.lobster_events`` (``hidden_mode`` "none") as it is,
    from the first row to the session's end, to be run with ``replaces_as_one``.

    A replace is two events in the stream (a deletion and a new order, flagged 1 and 2); ``run`` with
    ``replaces_as_one`` hands every study the deletion with the touch after the new order, so the state between the
    two rows, which never existed, reaches no study, and the new order as an event that changes nothing and that a
    message count leaves out. ``replaces_as_one=False`` keeps the stream's own two events. The stream has no
    hidden execution, so a figure on it is a figure on the displayed tape; it leaves out what ``lob.feeds`` leaves
    out (iceberg orders, the identity of orders, the order of events inside a nanosecond).
    """
    return Rung(day, "reference", None, replaces_as_one)


# ---- Level changes inside the ten levels ----

def _side_changes(price, size, r0, r1, ask: bool):
    """Per pair of rows, the price levels of one side whose size differs, inside the window both books show."""
    p0, q0, p1, q1 = price[r0], size[r0], price[r1], size[r1]
    inf = math.inf if ask else -math.inf
    last0 = np.where(np.isfinite(p0).all(1), p0[:, -1], inf)
    last1 = np.where(np.isfinite(p1).all(1), p1[:, -1], inf)
    limit = np.minimum(last0, last1) if ask else np.maximum(last0, last1)
    with np.errstate(invalid="ignore"):
        in0 = np.isfinite(p0) & ((p0 <= limit[:, None]) if ask else (p0 >= limit[:, None]))
        in1 = np.isfinite(p1) & ((p1 <= limit[:, None]) if ask else (p1 >= limit[:, None]))
        same_price = p0[:, :, None] == p1[:, None, :]
        exact = (same_price & (q0[:, :, None] == q1[:, None, :])).any(2)
        present = same_price.any(1)
    return (in0 & ~exact).sum(1) + (in1 & ~present).sum(1)


def level_changes(rows: BookRows, r0: np.ndarray, r1: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """The price levels, both sides together, whose size differs between book rows ``r0`` and ``r1`` (arrays of row
    numbers), counted inside the window both rows show: a level that moves into view or out of it because another
    level went is not a change. A level that is new, gone or of another size is one change."""
    out = np.zeros(len(r0), np.int64)
    for s in range(0, len(r0), chunk):
        a, b = r0[s:s + chunk], r1[s:s + chunk]
        out[s:s + chunk] = (_side_changes(rows.bid_price, rows.bid_size, a, b, False)
                            + _side_changes(rows.ask_price, rows.ask_size, a, b, True))
    return out


def _levels(rows: BookRows, r: int, ask: bool) -> dict:
    price = (rows.ask_price if ask else rows.bid_price)[r]
    size = (rows.ask_size if ask else rows.bid_size)[r]
    return {float(p): float(q) for p, q in zip(price.tolist(), size.tolist(), strict=True) if p == p and q > 0}


def _window_limit(rows: BookRows, r: int, ask: bool) -> float:
    price = (rows.ask_price if ask else rows.bid_price)[r]
    if np.isfinite(price).all():
        return float(price[-1])
    return math.inf if ask else -math.inf


def _touch_of(bids: dict, asks: dict) -> tuple:
    bid = max(bids) if bids else math.nan
    ask = min(asks) if asks else math.nan
    return bid, bids[bid] if bids else 0.0, ask, asks[ask] if asks else 0.0


def _same(a: tuple, b: tuple) -> bool:
    return all(x == y or (x != x and y != y) for x, y in zip(a, b, strict=True))


def _arrays(bids: dict, asks: dict, levels: int) -> tuple:
    """A book of dicts as the four arrays of one state: the best ``levels`` a side."""
    out = []
    for side, reverse in ((asks, False), (bids, True)):
        price, size = np.full(levels, np.nan), np.zeros(levels)
        for i, p in enumerate(sorted(side, reverse=reverse)[:levels]):
            price[i], size[i] = p, side[p]
        out += [price, size]
    return tuple(out)


class _Sub:
    """One changed level of an event: its side (1 bid, -1 ask), whether the aggressor hit that side, its price, the
    change of its size, the touch before and after it and the book after it (as arrays)."""

    __slots__ = ("side", "resting", "price", "delta", "after", "book")

    def __init__(self, side, resting, price, delta, after, book):
        self.side, self.resting, self.price, self.delta = side, resting, price, delta
        self.after, self.book = after, book


def _split(rows: BookRows, r0: int, r1: int, aggressor: int) -> Optional[list]:
    """The changed levels of one event, the side the aggressor hit first (an event with no aggressor changes one side
    only), each side best price first, with the book after each change.

    Only levels inside the window both books show change; the book after the last must have the touch the true book
    has after the event, or the split is refused (None) and the caller keeps the event whole.
    """
    bids, asks = _levels(rows, r0, False), _levels(rows, r0, True)
    bids1, asks1 = _levels(rows, r1, False), _levels(rows, r1, True)
    order = [False, True] if aggressor < 0 else [True, False]          # a buyer hits the asks, a seller the bids
    out = []
    for ask in order:
        old, new = (asks, asks1) if ask else (bids, bids1)
        lim0, lim1 = _window_limit(rows, r0, ask), _window_limit(rows, r1, ask)
        limit = min(lim0, lim1) if ask else max(lim0, lim1)
        changed = [p for p in set(old) | set(new)
                   if old.get(p, 0.0) != new.get(p, 0.0) and (p <= limit if ask else p >= limit)]
        changed.sort(reverse=not ask)
        for p in changed:
            before = old.get(p, 0.0)
            if new.get(p, 0.0) > 0:
                old[p] = new[p]
            else:
                old.pop(p, None)
            out.append(_Sub(-1 if ask else 1, bool(aggressor) and ask == (aggressor > 0), p, new.get(p, 0.0) - before,
                            _touch_of(bids, asks), _arrays(bids, asks, rows.levels)))
    if not out or not _same(out[-1].after, _touch_of(bids1, asks1)):
        return None
    return out


# ---- Rung: levels ----

def levels(day: Day, replaces_as_one: bool = True) -> Rung:
    """One event a changed price level, best price first, with the trades fused: what a per-update feed gives.

    Built from the reference's events with a replace as one (``replaces_as_one`` False keeps the stream's two events,
    a deletion and a new order, each one update: the rung the exploration built, and the one a feed that publishes a
    replace as two messages gives). An event that changes one level (inside the ten the frames show) is one update,
    as it is; one that changes several (a sweep, a limit order that sweeps and rests, a replace to another price) is
    split into one update a level, the side the aggressor hit first, each side best price first, each with the touch
    the consumer holds after it. A fall of a level by an execution is the trade, not a cancel: its update is a market
    arrival that carries the fills (each fill goes to the changed level of the hit side it took: the one at its price
    or, if the frames show none there, the nearest; for an execution with price, whose price is a cent off its order's
    level, the one the book fell at, ``Day.level``), and no cancel update stands beside it. An update that is no trade
    is a limit arrival where its level grew and a cancel where it shrank, with the level's price and the change of its
    size. An event that changes no level inside the window (a change deeper than ten levels) is an update with an
    unchanged touch. A split that does not end on the true touch after the event is refused and the event stays whole
    (``counts["fallback_events"]``).

    Lost: the order's identity (a replace is a fall and a rise, a cancel a fall), the grouping of the fills of one
    aggressive order (a sweep is several trades: a per-order figure cannot be read from the stream), the order of the
    updates inside one event (the best price first is a rule of this rung and not a fact about any venue) and the
    levels deeper than ten. The state after an update inside a split event is built from the levels before the event
    and the ones it changed, so it shows fewer than ten levels where a level has gone; the state after the last
    update is the true one.
    """
    t = day.table
    rows = day.rows
    n = t.n
    brk_at = np.zeros(n + 1, bool)
    brk_at[t.brk_pos] = True
    pair = np.zeros(n, bool)
    if n > 1 and replaces_as_one:
        pair[:-1] = (t.replace[:-1] == 1) & (t.replace[1:] == 2) & ~brk_at[1:n]
    keep = np.ones(n, bool)
    keep[1:] = ~pair[:-1]
    idx = np.flatnonzero(keep)                                   # the first event of each change of the book
    last = t.last.copy()
    last[pair] = t.last[np.flatnonzero(pair) + 1]                 # a replace ends with its new order
    ns, first, last, kind, side, price, size, aggressor = (a[idx] for a in (
        t.ns, t.first, last, t.kind, t.side, t.price, t.size, t.aggressor))
    m = len(idx)
    where = np.cumsum(keep) - 1
    fills = {int(where[j]): f for j, f in t.fills.items()}
    start = 1 if day.from_empty else 0
    before_row = first - 1
    changed = np.where(before_row >= 0, level_changes(rows, np.maximum(before_row, 0), last), 0)

    splits, fallbacks = {}, 0
    for i in np.flatnonzero(changed >= 2).tolist():
        subs = _split(rows, int(before_row[i]), int(last[i]), int(aggressor[i]) if i in fills else 0)
        if subs is None:
            fallbacks += 1
        else:
            splits[i] = subs
    count = np.ones(m, np.int64)
    for i, subs in splits.items():
        count[i] = len(subs)
    event_of = np.repeat(np.arange(m), count)
    first_update = np.cumsum(count) - count
    u = len(event_of)
    u_kind, u_side, u_price, u_size, u_aggressor, u_row = (a[event_of].copy() for a in (
        kind, side, price, size, aggressor, last))
    u_changed = changed[event_of].copy()
    u_fills: dict = {}                                            # update -> fills
    inside: dict = {}                                             # update -> the book after it, as arrays
    for i, f in fills.items():
        if i not in splits:
            u_fills[int(first_update[i])] = f
    for i, subs in splits.items():
        f = fills.get(i, ())
        resting = [k for k, s in enumerate(subs) if s.resting] or [0]
        at = [fill.price for fill in f]                       # the level a fill took is the one at its price ...
        if any(fill.priced for fill in f):                    # ... but a priced execution's price is a cent off it
            execs = [r for r in range(int(first[i]), int(last[i]) + 1) if day.kind[r] == EXECUTION]
            if len(execs) == len(f):                          # the fills are the event's executions, in their order
                at = [day.level[r] / day.tick if fill.priced else fill.price for r, fill in zip(execs, f, strict=True)]
        owner = [min(resting, key=lambda k: (abs(subs[k].price - x), k)) for x in at]
        for k, s in enumerate(subs):
            j = int(first_update[i]) + k
            mine = tuple(fill for fill, o in zip(f, owner, strict=True) if o == k)
            u_side[j], u_changed[j] = s.side, 1
            if mine:
                u_kind[j], u_price[j], u_aggressor[j], u_side[j] = ARRIVAL_MARKET, 0, aggressor[i], side[i]
                u_size[j] = sum(x.size for x in mine)
                u_fills[j] = mine
            elif s.delta > 0:
                u_kind[j], u_price[j], u_size[j], u_aggressor[j] = ARRIVAL_LIMIT, s.price, s.delta, s.side
            else:
                u_kind[j], u_price[j], u_size[j], u_aggressor[j] = CANCEL, s.price, -s.delta, 0
            if k < len(subs) - 1:
                u_row[j] = -1
                inside[j] = s.book

    # the stream's items: the updates, with the breaks where they stood
    nb = len(t.brk_pos)
    break_at = np.r_[0, np.cumsum(keep)][t.brk_pos]               # the changes of the book before each break
    upd_item = np.arange(u) + start + np.searchsorted(break_at, event_of, side="right")
    brk_item = np.r_[0, np.cumsum(count)][break_at] + np.arange(nb) + start
    total = u + nb + start
    item_kind = np.full(total, STATE_UPDATE, np.int64)
    stamp, ekind, eside, eprice, esize, eaggressor, changed_item, row = (np.zeros(total, dt) for dt in (
        np.int64, np.int64, np.int64, np.float64, np.int64, np.int64, np.int64, np.int64))
    row[:] = -1
    for dst, src in ((stamp, ns[event_of]), (ekind, u_kind), (eside, u_side), (eprice, u_price), (esize, u_size),
                     (eaggressor, u_aggressor), (changed_item, u_changed), (row, u_row)):
        dst[upd_item] = src
    item_kind[brk_item] = STATE_BREAK
    stamp[brk_item], row[brk_item] = t.brk_ns, t.brk_row
    if start:
        item_kind[0], stamp[0] = STATE_START, -1
    owner = np.full(total, -1, np.int64)
    owner[upd_item] = event_of
    states = _states(day, row)
    for j, book in inside.items():
        item = int(upd_item[j])
        for name, a in zip(("ask_price", "ask_size", "bid_price", "bid_size"), book, strict=True):
            getattr(states, name)[item] = a
    feed = _Feed(
        rows=states, kind=item_kind, stamp=stamp, ekind=ekind, eside=eside, eprice=eprice, esize=esize,
        eaggressor=eaggressor, changed=changed_item, fills={int(upd_item[j]): f for j, f in u_fills.items()},
        breaks={int(i): (int(r), why) for i, r, why in zip(brk_item, t.brk_row, t.brk_reason, strict=True)},
        state=np.arange(total), owner=owner)
    counts = {"events": m, "updates": u, "split_events": len(splits), "fallback_events": fallbacks,
              "trades": len(u_fills), "breaks": nb, "level_changes": int(changed_item.sum())}
    return Rung(day, "levels", feed, False, counts)


# ---- Rungs: batches ----

def batches(day: Day, ms: float) -> Rung:
    """The net change of each level over each batch of ``ms`` milliseconds of real time, and the trades as a stream of
    their own with their own times: what a batched feed gives.

    A batch is a window of the clock, [k ms, (k + 1) ms) from the day's origin (09:30:00); the rows of one window
    are one batch, and a reset break (a cross, a halt) ends one early. A batch that changes the net size of at least
    one price level, at any depth (counted from the messages, so a change deeper than the ten levels the frames
    show is one: the touch and the ten levels change only where it does, as the finest batch's test checks on two
    stocks), is an update stamped one nanosecond before its window's end (so that a whole second never splits a batch,
    for a width that divides a second), carrying the touch and the levels after the last row of the window. The
    trades are one event an aggressive order, with its fills, its true aggressor and the time of its first fill,
    placed in the stream by that time and seeing the book of the last batch stamped at or before it: that is the book
    before the trade's own batch (stale by up to the width), unless the trade is in the last nanosecond of its
    window, and for the finest batch, which is stamped at the nanosecond of its rows, the book that already holds the
    trade. ``ms`` may be below one (1e-6 is a nanosecond, the finest batch: the book after the last row of each
    nanosecond).

    Lost by construction: the identity of orders; the order of events inside a batch (a level that rises and falls
    inside it is never seen, nor is a best price that moves and moves back); arrivals against cancels at a level (only
    the net size change: ten lots added and ten cancelled is no update); states shorter than a batch (a spread of one
    tick that lasts 3 ms inside a batch of 50); a window whose rows net to nothing, which is no update though it is
    events of the reference (so the finest batch has the reference's states, the book after each nanosecond, and not
    its events: 1,610 nanoseconds of CMCSA's day and 4 of NVDA's, 2,983 two-tick readings of the strict imbalance
    design against the reference's 2,991); and the link between a trade and the level change it caused (the trade
    is in its own stream, the fall of the level in the next batch). Kept, which a venue's trade feed may not
    keep: the true aggressor of every trade, the grouping of its fills into one order, and the trade's own time to the
    nanosecond.
    """
    return _batched(day, ms, 0.0, f"batches({ms:g} ms)")


def shifted(day: Day, ms: float, shift_ms: float) -> Rung:
    """``batches(day, ms)`` with the trade stream stamped ``shift_ms`` later than the book (earlier if negative).

    A trade is then placed in the stream, and sees the book, at its own time plus the shift: a negative shift
    shows it a book that is older than the trade's own batch by that much more, a positive one a book that already
    holds the trade's effect. It is a bound on what a trade stream that is not synchronized with its book does, by
    a constant that no venue is known to have, and not a model of any venue. A shift of 0 is ``batches``.
    A trade moved to or past the session's end is not in the stream; one moved before the origin is stamped at it.
    """
    return _batched(day, ms, shift_ms, f"shifted({ms:g} ms, {shift_ms:+g} ms)")


def _batched(day: Day, ms: float, shift_ms: float, name: str) -> Rung:
    width = round(ms * 1e6)
    if width < 1:
        raise ValueError("a batch is at least a nanosecond (ms of 1e-6)")
    shift = round(shift_ms * 1e6)
    t, n = day.table, day.n
    ns, kind = day.ns[:n], day.kind[:n]
    resetting = np.array([why in DEFAULT_RESET for why in t.brk_reason], bool)
    reset_rows = t.brk_row[resetting]
    is_reset = np.zeros(n, bool)
    is_reset[reset_rows] = True
    segment = np.cumsum(is_reset)                                  # the break rows opened so far
    limit = np.r_[ns[reset_rows], day.end_ns - 1]                  # the last stamp a segment's batch may have
    rows_f = np.flatnonzero(~is_reset)
    empty = np.zeros(0, np.int64)
    end_rows = g_segment = g_window = per_group = empty
    changed = np.zeros(0, bool)
    if len(rows_f):
        window = (ns[rows_f] - day.origin_ns) // width
        s = segment[rows_f]
        new = (s[1:] != s[:-1]) | (window[1:] != window[:-1])
        group = np.r_[0, np.cumsum(new)]
        ends = np.r_[np.flatnonzero(new), len(rows_f) - 1]
        end_rows, g_segment, g_window = rows_f[ends], s[ends], window[ends]
        # the net size change of each level over each window, from the messages: adds and what executions, deletions
        # and cancels take (and the displayed orders executed in a cross), each at the level the book changed at
        # (``Day.level``: a row's own price, but for a priced execution and an order executed in a cross, whose price is
        # not the level they take shares from, so that an add and its execution in one window net to nothing)
        k = kind[rows_f]
        book_row = ((k >= 1) & (k <= 4)) | ((k == CROSS) & (day.order_id[rows_f] != 0))
        r = rows_f[book_row]
        net = pd.DataFrame({"g": group[book_row], "d": day.direction[r], "p": day.level[r],
                            "q": np.where(day.kind[r] == 1, day.size[r], -day.size[r])}
                           ).groupby(["g", "d", "p"], sort=False)["q"].sum()
        per_group = np.bincount(net.index.get_level_values(0).to_numpy()[net.to_numpy() != 0], minlength=len(ends))
        changed = per_group > 0
    picked = np.flatnonzero(changed)
    u_stamp = np.minimum(day.origin_ns + (g_window[picked] + 1) * width - 1, limit[g_segment[picked]])
    u_row, u_segment, u_changed = end_rows[picked], g_segment[picked], per_group[picked]

    with_fills = np.array(sorted(t.fills), np.int64)
    moved = np.maximum(t.ns[with_fills] + shift, day.origin_ns)
    in_session = moved < day.end_ns
    with_fills, moved = with_fills[in_session], moved[in_session]
    tr_segment = np.searchsorted(reset_rows, t.first[with_fills], side="left")

    other = np.flatnonzero(~resetting)
    n_start, n_reset, n_other = int(day.from_empty), len(reset_rows), len(other)
    n_update, n_trade = len(picked), len(with_fills)
    # the items, laid out as [the empty start][reset breaks][other breaks][updates][trades], then put in stream order:
    # by stamp, then by the segment they belong to (a reset break opens its own, so it comes before the updates of
    # the stamp that follow it), then books before trades before the breaks that open nothing
    layout = [  # kind, stamp, segment, priority, row of the state it makes (-2: none), changed levels
        (STATE_START, np.full(n_start, -1), np.zeros(n_start), 0, -1, 0),
        (STATE_BREAK, ns[reset_rows], np.arange(1, n_reset + 1), -1, reset_rows, 0),
        (STATE_BREAK, t.brk_ns[other], segment[t.brk_row[other]], 2, -2, 0),
        (STATE_UPDATE, u_stamp, u_segment, 0, u_row, u_changed),
        (STATE_TRADE, moved, tr_segment, 1, -2, 0),
    ]
    sizes = [len(part[1]) for part in layout]
    start_at = np.cumsum([0] + sizes)

    def field(i):
        return np.concatenate([np.broadcast_to(part[i], size) for part, size in zip(layout, sizes, strict=True)]
                              ).astype(np.int64)

    item_kind, stamp_c, segment_c, priority_c, row_c, changed_c = (field(i) for i in range(6))
    aggressor_c = np.concatenate([np.zeros(sum(sizes[:-1]), np.int64), t.aggressor[with_fills]])
    order = np.lexsort((np.arange(len(item_kind)), priority_c, segment_c, stamp_c))
    position = np.empty(len(order), np.int64)
    position[order] = np.arange(len(order))
    item_kind, stamp, row, changed_item, eaggressor = (a[order] for a in (
        item_kind, stamp_c, row_c, changed_c, aggressor_c))
    stateful = row >= -1
    ekind = np.full(len(order), ARRIVAL_LIMIT)                      # a batch update's kind and side are placeholders
    why = [t.brk_reason[i] for i in np.flatnonzero(resetting)] + [t.brk_reason[i] for i in other.tolist()]
    broken = np.r_[reset_rows, t.brk_row[other]]
    fills = {int(position[start_at[4] + j]): t.fills[e] for j, e in enumerate(with_fills.tolist())}
    breaks = {int(position[start_at[1] + j]): (int(broken[j]), why[j]) for j in range(n_reset + n_other)}
    total = len(order)
    feed = _Feed(
        rows=_states(day, row[stateful]), kind=item_kind, stamp=stamp, ekind=ekind, eside=np.ones(total, np.int64),
        eprice=np.zeros(total), esize=np.zeros(total, np.int64), eaggressor=eaggressor, changed=changed_item,
        fills=fills, breaks=breaks, state=np.cumsum(stateful) - 1)
    counts = {"events": t.n, "batches": n_update, "windows": len(end_rows),
              "empty_net_change": len(end_rows) - n_update, "trades": n_trade,
              "trades_after_the_end": int((~in_session).sum()), "breaks": len(t.brk_ns),
              "level_changes": int(u_changed.sum())}
    return Rung(day, name, feed, False, counts)


# ---- Running studies over a rung ----

def run_rung(rung: Rung, studies: "StudySet | Iterable | None" = None, unit: float = 1.0, listeners=(),
             reset_on=DEFAULT_RESET, final_samples: bool = True) -> Run:
    """``lob.feeds.run`` over a rung, with the unit samples every rung has the same of.

    The clock is the rung's day's (``origin``), the replace rule the rung's (the reference's ``replaces_as_one``),
    and a unit is sampled from the book the consumer holds at its start, as ``run`` does. ``run`` samples a unit
    only when an event opens it, so a stream that ends early or is sparse near its end would sample fewer units than
    another; with ``final_samples`` the units from the last one an event opened to the one that begins at the
    session's end are sampled from the book the consumer holds at the end (which no event has changed since), so
    every rung of a day gives the same units, the last being the state at the session's end. ``studies``, ``unit``,
    ``listeners`` and ``reset_on`` are ``run``'s. A study sees the rung's events as ``run`` hands them: a collector
    that must treat a replace as one change reads the events its ``observe`` is given (``own`` marks the new order
    of a replace on the reference), and one that is a listener sees the stream's own two events.
    """
    result = run(rung, studies, unit=unit, origin=rung.origin_ns / 1e9, listeners=listeners, reset_on=reset_on,
                 replaces_as_one=rung.replaces_as_one)
    if final_samples:
        last = int((rung.end_ns - rung.origin_ns) // round(unit * 1e9))
        for index, values in result.studies.sample_to(last, rung.final_book(), rows=True):
            result.samples.append((index, values))
            result.counts["samples"] += 1
    return result


__all__ = ["Day", "Rung", "batches", "level_changes", "levels", "reference", "run_rung", "shifted"]
