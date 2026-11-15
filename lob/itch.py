"""One day of Nasdaq TotalView-ITCH 5.0 as frames in LOBSTER's format.

Nasdaq posts a day's feed for every stock as one gzipped file of length-prefixed binary messages (a
2-byte big-endian length, then the message; its first byte is the type, then the stock locate code,
a tracking number and a 6-byte timestamp in nanoseconds after midnight). LOBSTER is a service that
turns such a feed into the two files ``lob.lobster`` reads. This module does the same for one stock,
from the specification and from regularities of the file, and keeps what LOBSTER's files leave out.

The steps, each a function:

- ``scan`` and ``extract``: one pass over the gzipped day each. The first counts the messages by
  type and reads the stock directory and the system events; the second copies the messages of the
  chosen stocks (and the few that belong to no stock) to a small file each, so that the rest of the
  work does not read 12.9 GB again.
- ``read_messages``: one extract as a frame of the messages that concern an order book.
- ``lobster_frames``: replays them into a book of resting orders by reference number and returns the
  messages, the book after each message and the hidden executions, for a window of the day.
- ``save_frames`` and ``load_frames``: the three frames as parquet files.

Choices that set this apart from the simulator's files (``lob.lobster.simulated_lobster``):

- The touch and the levels count every displayed order, odd lots included.
- A replace (``U``) is a deletion and a new order at the same nanosecond, two rows flagged ``replace``.
- An execution (``E``) is on the resting order's side, at its price. An execution with price (``C``)
  that is printable is one too, at its own price, and reduces the book at the order's price (the ``priced``
  column marks those rows). One that is not printable is, in continuous trading, the other half of a trade
  between two displayed orders whose printed half is an ``E`` (or a ``P``) with the same match number at the
  same nanosecond: it takes shares out of its order and is written as a cancel of part or a deletion at the
  order's displayed price, flagged ``paired``, never as an execution (the trade is counted once, by the
  printed half). One at the nanosecond of a cross trade (``Q``) is a cross (type 6), and so is the ``Q``.
  Any other non-printable ``C`` raises: the review found none on the two exploration symbols, and the
  reported five show none apart from the paired ones.
- A non-displayed trade (``P``) carries 'B' as its side whatever its aggressor: it is never a row of
  the messages frame. It goes to a table of its own, signed by the visible executions around it.
- Executions at one nanosecond with consecutive match numbers are one aggressive order (``group``), also
  across deletions, cancels of part and paired reductions between them (a same-side deletion in the middle of a
  sweep, a self-match-prevention cancel in all likelihood: 85 of CMCSA's 5,926 aggressive orders in the session,
  6 of NVDA's 15,618, were cut in two by one before this). An add, a replace, a cross or a trading action ends
  a run. What a count of aggressive orders is depends on the grouping: ``merge_within`` lowers NVDA's by 11.2
  percent and raises its order-to-trade ratio by 10.5 percent, so a figure per aggressive order is reported
  under both.
- The book is replayed from the first message of the day, so the first row of a window shows a book
  full of orders placed before it.

The data is untrusted: every length, type and side or printable byte is checked, no value read from the
file sizes an allocation, and anything unexpected that the replay depends on raises ``ItchError``. A file
that holds a trading halt (one stock's ``H`` after its start-of-day one, or an operational halt), a broken
trade, a message of an unknown type, an execution or cancel of no shares, a non-printable ``C`` that is
neither paired nor at a cross, or a touch off the cent grid or under a dollar inside the window is refused:
the frames would be wrong. Prices stay in ITCH's units (dollars times 10,000, as LOBSTER's); a cent is 100
of them. What it still accepts without a word: a locate code of 0 on an order message, a time after 24
hours, an unknown cross type, an arbitrary price on a ``C`` or a ``P`` (``lobster_frames`` counts the ones
that break the premises: a ``P`` that says 'S' or names an order, a ``C`` more than a cent from its order's
displayed price). What this module leaves out: the order imbalance, market participant and auction
messages, and the attribution of ``F`` orders.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import os
import queue
import threading
from array import array
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional

import numpy as np
import pandas as pd

from lob.lobster import DELETION, EMPTY_ASK, EMPTY_BID, EXECUTION, PARTIAL_CANCEL, SUBMISSION, book_columns

CROSS = 6  # LOBSTER's type for a cross trade
PRICE_UNIT = 10_000  # ITCH prices are dollars times 10,000
CENT = 100
MESSAGE_LENGTHS = {
    "S": 12, "R": 39, "H": 25, "Y": 20, "L": 26, "V": 35, "W": 12, "K": 28, "J": 35, "h": 21,
    "A": 36, "F": 40, "E": 31, "C": 36, "X": 23, "D": 19, "U": 35, "B": 19, "P": 44, "Q": 40, "I": 50, "N": 20,
}
BOOK_TYPES = "AFECXDUPQHS"  # what read_messages keeps
EXTRA_COLUMNS = ["ns", "match", "replace", "group", "paired", "priced"]
HIDDEN_COLUMNS = ["ns", "time", "size", "price", "match", "group", "aggressor"]
PROBLEMS = (
    "not_live_execute", "not_live_cancel", "not_live_delete", "not_live_replace", "over_execution", "over_cancel",
    "duplicate_reference", "empty_add", "empty_execution", "empty_cancel", "unpaired_non_printable_execution",
    "crossed_or_locked", "touch_under_a_dollar", "touch_off_the_cent_grid", "later_trading_action",
    "live_orders_after_end_of_system_hours",
)
_CHUNK = 8 << 20
_LIMIT = np.uint64(1 << 63)

_LENGTH = np.zeros(256, dtype=np.int64)
for _kind, _length in MESSAGE_LENGTHS.items():
    _LENGTH[ord(_kind)] = _length


class ItchError(ValueError):
    """Bytes that are not what the ITCH 5.0 specification (or this module's reading of it) allows."""


class Frames(NamedTuple):
    """What ``lobster_frames`` returns: LOBSTER's messages and book, and the hidden executions."""

    messages: pd.DataFrame
    book: pd.DataFrame
    hidden: pd.DataFrame


# ---- Walking the length-prefixed stream ----

def _boundaries(block: bytes) -> tuple[array, int]:
    """Offsets of the messages that lie whole in ``block`` (at their length prefix), and where the rest begins."""
    offsets = array("q")
    append = offsets.append
    o, last = 0, len(block) - 1
    while o < last:
        append(o)
        o += 2 + (block[o] << 8 | block[o + 1])
    if o > len(block):  # the last message is cut off by the end of the block
        o = offsets.pop()
    return offsets, o


def _check_lengths(arr: np.ndarray, offsets: np.ndarray, position: int = 0) -> np.ndarray:
    """The type byte of each message, after checking that its length is the specification's for that type."""
    if len(offsets) and offsets[-1] + 2 >= len(arr):
        raise ItchError(f"the message at byte {position + int(offsets[-1])} has no type byte")
    kind = arr[offsets + 2]
    size = (arr[offsets].astype(np.int64) << 8) | arr[offsets + 1]
    expected = _LENGTH[kind]
    bad = (size != expected) | (expected == 0)
    if bad.any():
        j = int(np.flatnonzero(bad)[0])
        raise ItchError(f"the message at byte {position + int(offsets[j])} has type {chr(int(kind[j]))!r} and length "
                        f"{int(size[j])}; the specification says {int(expected[j])} (0: no such type)")
    return kind


def _field(arr: np.ndarray, offsets: np.ndarray, start: int, width: int) -> np.ndarray:
    """The big-endian unsigned integer of ``width`` bytes at ``start`` in each message's body (type byte at 0)."""
    position = offsets + (2 + start)
    value = arr[position].astype(np.uint64)
    for j in range(1, width):
        value = (value << np.uint64(8)) | arr[position + j]
    return value


def _read_chunks(path: Path, size: int) -> Iterator[bytes]:
    """The decompressed bytes of ``path`` in blocks, read ahead by a thread (zlib lets go of the interpreter lock)."""
    opener = gzip.open if path.suffix == ".gz" else open
    buffer: queue.Queue = queue.Queue(maxsize=6)
    stop = threading.Event()

    def put(item: Any) -> None:
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.5)
                return
            except queue.Full:
                pass

    def reader() -> None:
        try:
            with opener(path, "rb") as handle:
                while not stop.is_set():
                    data = handle.read(size)
                    if not data:
                        break
                    put(data)
            put(None)
        except BaseException as exc:  # handed to the consumer, which raises it
            put(exc)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while not buffer.empty():
            buffer.get_nowait()
        thread.join(timeout=5)


def _blocks(path: str | Path, size: Optional[int] = None) -> Iterator[tuple[bytes, np.ndarray, np.ndarray, np.ndarray]]:
    """Whole messages of a file in blocks: the bytes, as an array, the offsets of the messages, and their types."""
    position, carry = 0, b""
    for data in _read_chunks(Path(path), size or _CHUNK):
        block = carry + data if carry else data
        offsets, end = _boundaries(block)
        carry = block[end:]
        if offsets:
            arr = np.frombuffer(block, dtype=np.uint8)
            offs = np.frombuffer(offsets, dtype=np.int64)
            yield block, arr, offs, _check_lengths(arr, offs, position)
        position += end
    if carry:
        raise ItchError(f"the file ends inside a message ({len(carry)} bytes after byte {position})")


def _stock(raw: bytes) -> str:
    return raw.decode("ascii").rstrip()


# ---- One pass over the day ----

def scan(path: str | Path) -> dict[str, Any]:
    """One pass over a day's (gzipped) file: the stock directory, the messages by type, the system events.

    Returns ``directory`` (symbol to stock locate code, from the ``R`` messages), ``counts`` (type to
    number of messages), ``system_events`` (``S`` messages as (event code, nanoseconds after midnight)),
    and ``messages``, ``bytes`` and ``blocks`` read. Every message's length is checked against the
    specification, and the gzip's own checksum is verified by reading to its end. Takes a minute or
    two for a full day (70 s measured, with the unzipping on a thread of its own).
    """
    counts = np.zeros(256, dtype=np.int64)
    directory: dict[str, int] = {}
    locates: set[int] = set()
    events: list[tuple[str, int]] = []
    blocks = 0
    for block, arr, offs, kind in _blocks(path):
        counts += np.bincount(kind, minlength=256)
        for o in offs[kind == ord("R")].tolist():
            symbol, locate = _stock(block[o + 13:o + 21]), int.from_bytes(block[o + 3:o + 5], "big")
            if symbol in directory or locate in locates:
                raise ItchError(f"the directory names {symbol!r} or locate {locate} twice")
            directory[symbol] = locate
            locates.add(locate)
        selected = offs[kind == ord("S")]
        if len(selected):
            for o, ns in zip(selected.tolist(), _field(arr, selected, 5, 6).tolist(), strict=True):
                events.append((chr(block[o + 13]), int(ns)))
        blocks += 1
    by_type = {chr(code): int(n) for code, n in enumerate(counts) if n}
    return {"directory": directory, "counts": by_type, "system_events": events, "messages": int(counts.sum()),
            "bytes": int((counts * (_LENGTH + 2)).sum()), "blocks": blocks}


def extract(path: str | Path, symbols: list[str], directory: str | Path, overwrite: bool = False) -> dict[str, Any]:
    """One pass over a day's file: each symbol's messages, as they came, to ``<directory>/<SYMBOL>.itch``.

    A file holds each message with its 2-byte length, so it can be read as the day is, and also every
    message of locate code 0 (those that belong to no stock: system events, circuit-breaker levels
    and the IPO quoting update), so that a file stands alone. The directory's ``R`` messages come
    before any order message, which is how the locate codes are found. Files are written under a
    temporary name and renamed when the pass has read the day to its end. Returns, for each symbol, its
    locate code, number of messages, bytes and SHA-256.
    """
    wanted = list(dict.fromkeys(symbol.upper() for symbol in symbols))
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {symbol: out_dir / f"{symbol}.itch" for symbol in wanted}
    if not overwrite:
        for target in targets.values():
            if target.exists():
                raise FileExistsError(f"{target} exists (overwrite=True replaces it)")
    parts = {symbol: target.with_name(target.name + ".part") for symbol, target in targets.items()}
    handles = {symbol: open(part, "wb") for symbol, part in parts.items()}
    digests = {symbol: hashlib.sha256() for symbol in wanted}
    stats = {symbol: {"locate": None, "messages": 0, "bytes": 0} for symbol in wanted}
    try:
        for block, arr, offs, kind in _blocks(path):
            for o in offs[kind == ord("R")].tolist():
                symbol = _stock(block[o + 13:o + 21])
                if symbol in stats:
                    if stats[symbol]["locate"] is not None:
                        raise ItchError(f"the directory names {symbol!r} twice")
                    stats[symbol]["locate"] = int.from_bytes(block[o + 3:o + 5], "big")
            locate = (arr[offs + 3].astype(np.int64) << 8) | arr[offs + 4]
            length = (arr[offs].astype(np.int64) << 8) | arr[offs + 1]
            none = locate == 0
            for symbol in wanted:
                code = stats[symbol]["locate"]
                if code is None:
                    selected = np.flatnonzero(none)
                else:
                    selected = np.flatnonzero(none | (locate == code))
                if not len(selected):
                    continue
                first = offs[selected]
                last = first + 2 + length[selected]
                data = b"".join([block[a:b] for a, b in zip(first.tolist(), last.tolist(), strict=True)])
                handles[symbol].write(data)
                digests[symbol].update(data)
                stats[symbol]["messages"] += len(selected)
                stats[symbol]["bytes"] += len(data)
        missing = [symbol for symbol in wanted if stats[symbol]["locate"] is None]
        if missing:
            raise ItchError(f"not in the day's stock directory: {missing}")
    except BaseException:
        for handle in handles.values():
            handle.close()
        for part in parts.values():
            part.unlink(missing_ok=True)
        raise
    for symbol in wanted:
        handles[symbol].close()
        os.replace(parts[symbol], targets[symbol])
        stats[symbol]["sha256"] = digests[symbol].hexdigest()
    return stats


def count_messages(raw_file: str | Path) -> dict[str, int]:
    """The messages of an extract (or a whole day) by type, after checking each length."""
    counts = np.zeros(256, dtype=np.int64)
    for _, _, _, kind in _blocks(raw_file):
        counts += np.bincount(kind, minlength=256)
    return {chr(code): int(n) for code, n in enumerate(counts) if n}


# ---- Reading an extract ----

def read_messages(raw_file: str | Path) -> pd.DataFrame:
    """A frame of the messages of an extract that concern an order book, in file order.

    One row a message of types ``A`` (add), ``F`` (add with participant), ``E`` (executed), ``C``
    (executed with price), ``X`` (cancel of part), ``D`` (delete), ``U`` (replace), ``P`` (trade not
    against a displayed order), ``Q`` (cross), ``H`` (trading action) and ``S`` (system event). Columns:

    - ``ns`` (int64): nanoseconds after midnight; ``type``: the type letter;
    - ``ref`` (int64): the order's reference number (``U``: the original's; 0 for ``Q``, and for ``P``
      whatever the message says, which is 0 on Nasdaq);
    - ``new_ref`` (int64): the new order of a ``U``, else 0;
    - ``side`` (int64): +1 buy, -1 sell, 0 none. Set on ``A``, ``F`` and ``P``; a ``P`` says 'B'
      whatever happened, so it says nothing about the aggressor;
    - ``shares`` (int64), ``price`` (int64, ITCH units), ``match`` (int64, 0 where none);
    - ``printable``: 'Y' or 'N' on ``C``, else '';
    - ``code``: the event code of an ``S``, the trading state of an ``H``, the cross type of a ``Q``
      (O opening, C closing, H halted, I intraday), else ''.

    The file must hold one stock (the nonzero locate codes agree, and so do the stock names of the
    messages that carry one) and its timestamps must not run backwards. ``attrs`` carries the symbol.
    Types that would make the book wrong if they were skipped (``B`` broken trade, ``h`` operational
    halt) raise.
    """
    buf = Path(raw_file).read_bytes()
    offsets, end = _boundaries(buf)
    if end != len(buf):
        raise ItchError(f"{raw_file} ends inside a message")
    arr = np.frombuffer(buf, dtype=np.uint8)
    offs = np.frombuffer(offsets, dtype=np.int64)
    kind = _check_lengths(arr, offs)
    for letter in "Bh":
        if (kind == ord(letter)).any():
            raise ItchError(f"the file holds a message of type {letter!r}, which this module does not handle")
    all_ns = _field(arr, offs, 5, 6)
    if len(all_ns) > 1 and (np.diff(all_ns.astype(np.int64)) < 0).any():
        raise ItchError("timestamps run backwards")
    locate = ((arr[offs + 3].astype(np.int64) << 8) | arr[offs + 4]) if len(offs) else np.zeros(0, np.int64)
    own = np.unique(locate[locate != 0])
    if len(own) > 1:
        raise ItchError(f"the file holds more than one stock locate code: {own.tolist()}")

    keep = np.isin(kind, [ord(letter) for letter in BOOK_TYPES])
    base, kind = offs[keep], kind[keep]
    m = len(base)
    ns = _field(arr, base, 5, 6).astype(np.int64)
    ref, new_ref = np.zeros(m, np.int64), np.zeros(m, np.int64)
    side, shares = np.zeros(m, np.int64), np.zeros(m, np.int64)
    price, match = np.zeros(m, np.int64), np.zeros(m, np.int64)
    printable, code = np.full(m, "", dtype="U1"), np.full(m, "", dtype="U1")
    stock_bytes: list[np.ndarray] = []

    def where(letters: str) -> np.ndarray:
        return np.flatnonzero(np.isin(kind, [ord(letter) for letter in letters]))

    def signed(value: np.ndarray, what: str) -> np.ndarray:
        if (value >= _LIMIT).any():
            raise ItchError(f"a {what} above 2**63")
        return value.astype(np.int64)

    def letters(ix: np.ndarray, start: int, table: dict[int, int], what: str) -> np.ndarray:
        value = arr[base[ix] + (2 + start)].astype(np.int64)
        if not np.isin(value, list(table)).all():
            raise ItchError(f"{what} byte outside {[chr(c) for c in table]}")
        out = np.zeros(len(ix), np.int64)
        for letter, sign in table.items():
            out[value == letter] = sign
        return out

    def stock(ix: np.ndarray, start: int) -> None:
        position = base[ix] + (2 + start)
        stock_bytes.append(arr[position[:, None] + np.arange(8)] if len(ix) else np.zeros((0, 8), np.uint8))

    ix = where("AF")
    ref[ix] = signed(_field(arr, base[ix], 11, 8), "reference")
    side[ix] = letters(ix, 19, {ord("B"): 1, ord("S"): -1}, "side")
    shares[ix], price[ix] = _field(arr, base[ix], 20, 4).astype(np.int64), _field(arr, base[ix], 32, 4).astype(np.int64)
    stock(ix, 24)
    ix = where("E")
    ref[ix] = signed(_field(arr, base[ix], 11, 8), "reference")
    shares[ix], match[ix] = _field(arr, base[ix], 19, 4).astype(np.int64), signed(_field(arr, base[ix], 23, 8), "match")
    ix = where("C")
    ref[ix] = signed(_field(arr, base[ix], 11, 8), "reference")
    shares[ix], match[ix] = _field(arr, base[ix], 19, 4).astype(np.int64), signed(_field(arr, base[ix], 23, 8), "match")
    price[ix] = _field(arr, base[ix], 32, 4).astype(np.int64)
    flag = letters(ix, 31, {ord("Y"): 1, ord("N"): 0}, "printable")
    printable[ix] = np.where(flag == 1, "Y", "N")
    for letter in "XD":
        ix = where(letter)
        ref[ix] = signed(_field(arr, base[ix], 11, 8), "reference")
        if letter == "X":
            shares[ix] = _field(arr, base[ix], 19, 4).astype(np.int64)
    ix = where("U")
    ref[ix] = signed(_field(arr, base[ix], 11, 8), "reference")
    new_ref[ix] = signed(_field(arr, base[ix], 19, 8), "reference")
    shares[ix], price[ix] = _field(arr, base[ix], 27, 4).astype(np.int64), _field(arr, base[ix], 31, 4).astype(np.int64)
    ix = where("P")
    ref[ix] = signed(_field(arr, base[ix], 11, 8), "reference")
    side[ix] = letters(ix, 19, {ord("B"): 1, ord("S"): -1}, "side")
    shares[ix], price[ix] = _field(arr, base[ix], 20, 4).astype(np.int64), _field(arr, base[ix], 32, 4).astype(np.int64)
    match[ix] = signed(_field(arr, base[ix], 36, 8), "match")
    stock(ix, 24)
    ix = where("Q")
    shares[ix] = signed(_field(arr, base[ix], 11, 8), "share count")
    price[ix] = _field(arr, base[ix], 27, 4).astype(np.int64)
    match[ix] = signed(_field(arr, base[ix], 31, 8), "match")
    code[ix] = [chr(c) for c in arr[base[ix] + 2 + 39].tolist()]
    stock(ix, 19)
    ix = where("H")
    code[ix] = [chr(c) for c in arr[base[ix] + 2 + 19].tolist()]
    stock(ix, 11)
    ix = where("S")
    code[ix] = [chr(c) for c in arr[base[ix] + 2 + 11].tolist()]

    names = np.unique(np.concatenate(stock_bytes).view("S8").ravel()) if stock_bytes else np.zeros(0, "S8")
    if len(names) > 1:
        raise ItchError(f"the file holds more than one stock: {[_stock(bytes(n)) for n in names]}")
    frame = pd.DataFrame({
        "ns": ns, "type": kind.astype(np.uint8).view("S1").astype("U1"), "ref": ref, "new_ref": new_ref, "side": side,
        "shares": shares, "price": price, "match": match, "printable": printable, "code": code,
    })
    frame.attrs["symbol"] = _stock(bytes(names[0])) if len(names) else None
    frame.attrs["locate"] = int(own[0]) if len(own) else None
    return frame


# ---- The book ----

class _Side:
    """One side of the book: shares by price, and the occupied prices in ascending order."""

    __slots__ = ("size", "prices", "bid", "levels")

    def __init__(self, bid: bool, levels: int):
        self.size: dict[int, int] = {}
        self.prices: list[int] = []
        self.bid = bid
        self.levels = levels

    def near(self, price: int) -> bool:
        """Is ``price`` among the best ``levels`` occupied levels (so a change there changes the book frame)?"""
        prices, n = self.prices, self.levels
        if len(prices) < n:
            return True
        return price >= prices[-n] if self.bid else price <= prices[n - 1]

    def add(self, price: int, shares: int) -> bool:
        near = self.near(price)
        if price in self.size:
            self.size[price] += shares
        else:
            self.size[price] = shares
            bisect.insort(self.prices, price)
        return near

    def reduce(self, price: int, shares: int) -> bool:
        near = self.near(price)
        left = self.size[price] - shares
        if left:
            self.size[price] = left
        else:
            del self.size[price]
            del self.prices[bisect.bisect_left(self.prices, price)]
        return near

    def best(self) -> Optional[int]:
        return (self.prices[-1] if self.bid else self.prices[0]) if self.prices else None


def _groups(sequence: list[tuple[int, int, int, int]], merge_ns: Optional[int],
            joined: Callable[[int, int], bool]) -> tuple[list[int], list[int]]:
    """Aggressive orders among executions given as (ns, match, resting side (0 for hidden), message index).

    An execution joins the group before it when it follows it with nothing between them but messages that
    ``joined(a, b)`` allows (the indexes of the two messages: deletions, cancels of part and paired reductions
    only), at the same nanosecond, with the next match number, unless its resting side differs from the group's
    (hidden executions have no side and join any). With ``merge_ns``, successive groups of the same known side
    that follow each other directly (the next message, the next match number) and are at most that many
    nanoseconds apart are joined: a sensitivity of the review's, which kept the rule that nothing lies between
    them. Returns each execution's group number and each group's resting side.
    """
    first_ns: list[int] = []
    last_ns: list[int] = []
    first_match: list[int] = []
    last_match: list[int] = []
    first_at: list[int] = []
    last_at: list[int] = []
    sides: list[int] = []
    group_of: list[int] = []
    for ns, match, side, at in sequence:
        g = len(sides) - 1
        if g >= 0 and ns == last_ns[g] and match == last_match[g] + 1 and joined(last_at[g], at) and (
                side == 0 or sides[g] == 0 or sides[g] == side):
            last_match[g] = match
            last_at[g] = at
            if sides[g] == 0:
                sides[g] = side
        else:
            g += 1
            first_ns.append(ns)
            last_ns.append(ns)
            first_match.append(match)
            last_match.append(match)
            first_at.append(at)
            last_at.append(at)
            sides.append(side)
        group_of.append(g)
    if merge_ns is None:
        return group_of, sides
    merged_id: list[int] = []
    merged_side: list[int] = []
    merged_last_ns: list[int] = []
    merged_last_match: list[int] = []
    merged_last_at: list[int] = []
    for g in range(len(sides)):
        m = len(merged_side) - 1
        if (m >= 0 and merged_side[m] != 0 and sides[g] == merged_side[m]
                and first_match[g] == merged_last_match[m] + 1 and first_ns[g] - merged_last_ns[m] <= merge_ns
                and first_at[g] == merged_last_at[m] + 1):
            merged_last_ns[m], merged_last_match[m], merged_last_at[m] = last_ns[g], last_match[g], last_at[g]
        else:
            m += 1
            merged_side.append(sides[g])
            merged_last_ns.append(last_ns[g])
            merged_last_match.append(last_match[g])
            merged_last_at.append(last_at[g])
        merged_id.append(m)
    return [merged_id[g] for g in group_of], merged_side


def lobster_frames(messages: pd.DataFrame, start: float = 34200.0, end: float = 57600.0, levels: int = 10, *,
                   merge_within: Optional[float] = None, strict: bool = True,
                   report: Optional[dict[str, Any]] = None) -> Frames:
    """Replay a day's messages into a book, and return LOBSTER's frames for the rows in [start, end).

    ``messages`` is ``read_messages``' frame. The whole day is replayed (orders placed before ``start``
    are in the book at ``start``); times are seconds after midnight. Returns ``Frames(messages, book,
    hidden)``:

    - ``messages``: LOBSTER's six columns (``time``, ``type``, ``order_id``, ``size``, ``price``,
      ``direction``) and ``ns``, ``match`` (0 where none), ``replace`` (True on the two rows of a
      replace), ``group`` (the aggressive order of an execution, else -1), ``paired`` (True on the
      reduction a paired non-printable ``C`` makes) and ``priced`` (True on a type 4 row that is a printable
      ``C``). All int64 but ``time`` (float64 seconds) and ``replace``, ``paired`` and ``priced`` (bool). Types:
      1 an add (``A``, ``F``, and the new order of a ``U``); 2 a cancel of part (``X``, and a paired ``C`` that
      leaves shares; ``size`` is what was cancelled); 3 a deletion (``D``, the old order of a ``U``, and a
      paired ``C`` that takes the rest; ``size`` is what was left); 4 an execution of a displayed order
      (``E``, and ``C`` printable 'Y', at its own price, which is a cent off its order's, while the book is
      reduced at the order's price; the direction is the resting order's side); 6 a cross (``Q``, order_id 0,
      direction 0; and ``C`` printable 'N' at a cross's nanosecond, which reduces the book). A paired ``C`` is
      the non-printing half of a trade between two displayed orders, written at its order's displayed price
      with its match number kept; ``Q`` and ``C`` 'N' are never executions of type 4.
    - ``book``: ``lob.lobster.book_columns(levels)`` after each row: the best ``levels`` occupied levels,
      odd lots included. An empty level has a NaN price and size 0 (as ``read_lobster`` gives, which
      ``save_lobster`` writes as LOBSTER's placeholders). Prices are float64 and sizes int64.
    - ``hidden``: one row a ``P``: ``ns``, ``time``, ``size``, ``price``, ``match``, ``group`` and
      ``aggressor`` (+1, -1, or 0 when no visible execution is in its group). Never in ``messages``.

    Executions (type 4 rows and ``P``) that follow each other, with nothing between them but deletions,
    cancels of part and paired reductions, at one nanosecond, with consecutive match numbers, are one
    aggressive order; its aggressor is the opposite of the resting side of its visible executions, and a run
    of ``P`` alone has aggressor 0 (a ``P`` paired with a non-printable ``C`` is not signed by it: the
    sheet signs a ``P`` by the executions of its run only). With ``merge_within`` (seconds), successive groups
    of the same known side that follow each other directly (no message between) with consecutive match
    numbers, not more than that far apart in time, are one (a sensitivity; P-only groups are never merged).

    Raises ``ItchError`` on: an execute, cancel, delete or replace that names no live order; an execution
    or cancel of more shares than the order has, or of none; a reference added twice; an add of no shares; a
    non-printable ``C`` inside the window that is neither the other half of a printed trade (the same match
    number and nanosecond on an ``E``, a printable ``C`` or a ``P``) nor at the nanosecond of a ``Q``; a
    crossed or locked book, a touch under a dollar, or a touch off the cent grid, after a message inside the
    window; a trading action other than the first (which must say 'T'); and, when the day's end of system hours
    ('E') has passed, orders still live after the last message (the exchange deletes the last orders within about
    75 ms after 'E', so the book is checked at the end of the file, not at 'E'). With ``strict=False``
    none raises: each is counted, the offending message is skipped (an over-large execution or cancel
    is cut to the shares left), and ``report`` (a dict, if given) is filled with the counts under the
    names in ``PROBLEMS`` and with ``live_orders_at_end``, ``rows`` and ``groups`` (the rows of
    ``messages`` and the aggressive orders, hidden-only ones included), and the counts, which are not
    errors, of ``adds_under_a_dollar`` and ``adds_off_the_cent_grid`` (stub quotes far from the touch:
    NVDA has 47 and 2), ``cancels_to_zero`` (a cancel that leaves an order no shares: the order leaves), and
    of what would break a premise of the reading, over the whole file: ``p_side_not_b`` (a ``P`` that says
    'S': ITCH sets 'B' on every one, so the side of a ``P`` is no aggressor), ``p_with_reference`` (a ``P`` that
    names an order) and ``c_price_over_a_cent`` (a ``C`` whose price is more than a cent from its order's
    displayed price; all of NVDA's and CMCSA's are exactly a cent).
    """
    if levels < 1:
        raise ValueError("levels must be at least 1")
    n = len(messages)
    lo, hi = int(round(start * 1e9)), int(round(end * 1e9))
    kinds = np.frombuffer("".join(messages["type"].astype(str)).encode("ascii"), dtype=np.uint8) if n else \
        np.zeros(0, np.uint8)
    codes = messages["code"].astype(str).tolist() if "code" in messages else [""] * n
    flags = messages["printable"].astype(str).tolist()
    counts = {name: 0 for name in PROBLEMS}
    counts.update(adds_under_a_dollar=0, adds_off_the_cent_grid=0, cancels_to_zero=0, p_side_not_b=0,
                  p_with_reference=0, c_price_over_a_cent=0)
    cross_kind = _non_printable(messages, kinds, flags, n)

    r_ns, r_kind, r_id, r_size = array("q"), array("q"), array("q"), array("q")
    r_price, r_dir, r_match, r_rep = array("q"), array("q"), array("q"), array("q")
    r_paired, r_priced = array("q"), array("q")
    hidden_ns, hidden_size, hidden_price, hidden_match = array("q"), array("q"), array("q"), array("q")
    sequence: list[tuple[int, int, int, int]] = []  # executions in the window: (ns, match, side or 0, at)
    owner: list[int] = []  # for each: its row in messages (>= 0), or -1 - its row in hidden
    snaps, change_rows = array("q"), array("q")
    solid = bytearray(b"\x01" * n)  # 0 for a message that executions of one run may have between them
    asks, bids = _Side(False, levels), _Side(True, levels)
    orders: dict[int, list[int]] = {}  # reference -> [is bid, price, shares left]
    state = {"dirty": True, "seen_action": False, "end_of_hours": False}
    L = levels

    def problem(name: str, ns: int, detail: str) -> None:
        if strict:
            raise ItchError(f"{name} at {ns / 1e9:.9f} s: {detail}")
        counts[name] += 1

    def snapshot() -> None:
        ap, bp = asks.prices[:L], bids.prices[:-L - 1:-1]
        na, nb = L - len(ap), L - len(bp)
        snaps.extend(ap + [EMPTY_ASK] * na + [asks.size[p] for p in ap] + [0] * na
                     + bp + [EMPTY_BID] * nb + [bids.size[p] for p in bp] + [0] * nb)

    def emit(ns: int, kind: int, oid: int, size: int, price: int, direction: int, match: int, replace: int,
             paired: int = 0, priced: int = 0) -> None:
        if state["dirty"]:
            change_rows.append(len(r_ns))
            snapshot()
            state["dirty"] = False
        r_ns.append(ns)
        r_kind.append(kind)
        r_id.append(oid)
        r_size.append(size)
        r_price.append(price)
        r_dir.append(direction)
        r_match.append(match)
        r_rep.append(replace)
        r_paired.append(paired)
        r_priced.append(priced)

    def check_touch(ns: int) -> None:
        bid, ask = bids.best(), asks.best()
        if bid is not None and ask is not None and bid >= ask:
            problem("crossed_or_locked", ns, f"best bid {bid} against best ask {ask}")
        if (bid is not None and bid < PRICE_UNIT) or (ask is not None and ask < PRICE_UNIT):
            problem("touch_under_a_dollar", ns, f"touch {bid} / {ask}: the cent tick is assumed")
        if (bid is not None and bid % CENT) or (ask is not None and ask % CENT):
            problem("touch_off_the_cent_grid", ns, f"touch {bid} / {ask}: the cent tick is assumed")

    def add(ns: int, ref: int, bid: bool, price: int, shares: int, inside: bool, replace: int) -> bool:
        if ref in orders:
            problem("duplicate_reference", ns, f"reference {ref} is already live")
            return False
        if shares <= 0:
            problem("empty_add", ns, f"reference {ref} added with {shares} shares")
            return False
        if price < PRICE_UNIT:
            counts["adds_under_a_dollar"] += 1
        if price % CENT:
            counts["adds_off_the_cent_grid"] += 1
        orders[ref] = [1 if bid else 0, price, shares]
        if (bids if bid else asks).add(price, shares):
            state["dirty"] = True
        if inside:
            emit(ns, SUBMISSION, ref, shares, price, 1 if bid else -1, 0, replace)
        return True

    def remove(ns: int, ref: int, inside: bool, replace: int) -> bool:
        order = orders.pop(ref)
        if (bids if order[0] else asks).reduce(order[1], order[2]):
            state["dirty"] = True
        if inside:
            emit(ns, DELETION, ref, order[2], order[1], 1 if order[0] else -1, 0, replace)
        return True

    def execute(i: int, ns: int, ref: int, shares: int, match: int, kind: int, price: Optional[int],
                inside: bool) -> bool:
        order = orders.get(ref)
        if order is None:
            problem("not_live_execute", ns, f"reference {ref} is not live")
            return False
        if shares <= 0:
            problem("empty_execution", ns, f"{shares} shares executed in {ref}")
            return False
        if shares > order[2]:
            problem("over_execution", ns, f"{shares} shares executed of {order[2]} left in {ref}")
            shares = order[2]
        if (bids if order[0] else asks).reduce(order[1], shares):
            state["dirty"] = True
        order[2] -= shares
        if not order[2]:
            del orders[ref]
        if inside:
            direction = 1 if order[0] else -1
            if kind == EXECUTION:
                sequence.append((ns, match, direction, i))
                owner.append(len(r_ns))
            emit(ns, kind, ref, shares, order[1] if price is None else price, direction, match, 0,
                 priced=1 if kind == EXECUTION and price is not None else 0)
        return True

    def reduce_paired(ns: int, ref: int, shares: int, match: int, inside: bool) -> bool:
        """A non-printable C that is the other half of a printed trade: shares out of its order, at its price."""
        order = orders.get(ref)
        if order is None:
            problem("not_live_execute", ns, f"reference {ref} is not live")
            return False
        if shares <= 0:
            problem("empty_execution", ns, f"{shares} shares executed in {ref}")
            return False
        if shares > order[2]:
            problem("over_execution", ns, f"{shares} shares executed of {order[2]} left in {ref}")
            shares = order[2]
        if (bids if order[0] else asks).reduce(order[1], shares):
            state["dirty"] = True
        order[2] -= shares
        whole = not order[2]
        if whole:
            del orders[ref]
        if inside:
            emit(ns, DELETION if whole else PARTIAL_CANCEL, ref, shares, order[1], 1 if order[0] else -1, match, 0,
                 paired=1)
        return True

    for i, (ns, kind, ref, new_ref, side, shares, price, match) in enumerate(zip(
            messages["ns"].tolist(), kinds.tolist(), messages["ref"].tolist(), messages["new_ref"].tolist(),
            messages["side"].tolist(), messages["shares"].tolist(), messages["price"].tolist(),
            messages["match"].tolist(), strict=True)):
        inside = lo <= ns < hi
        changed = True
        if kind == 65 or kind == 70:  # A, F
            changed = add(ns, ref, side > 0, price, shares, inside, 0)
        elif kind == 68:  # D
            solid[i] = 0
            if ref in orders:
                remove(ns, ref, inside, 0)
            else:
                problem("not_live_delete", ns, f"reference {ref} is not live")
                changed = False
        elif kind == 85:  # U
            if ref not in orders:
                problem("not_live_replace", ns, f"original {ref} is not live")
                changed = False
            elif new_ref in orders:
                problem("duplicate_reference", ns, f"reference {new_ref} is already live")
                changed = False
            else:
                bid = orders[ref][0] == 1
                remove(ns, ref, inside, 1)
                changed = add(ns, new_ref, bid, price, shares, inside, 1)
        elif kind == 69:  # E
            changed = execute(i, ns, ref, shares, match, EXECUTION, None, inside)
        elif kind == 67:  # C
            if ref in orders and abs(price - orders[ref][1]) > CENT:
                counts["c_price_over_a_cent"] += 1
            if flags[i] == "Y":
                changed = execute(i, ns, ref, shares, match, EXECUTION, price, inside)
            else:
                how = cross_kind[i]
                if how == "paired":
                    solid[i] = 0
                    changed = reduce_paired(ns, ref, shares, match, inside)
                else:
                    if how == "unpaired" and inside:
                        problem("unpaired_non_printable_execution", ns,
                                f"match {match} has no printed half at this nanosecond and no cross is here")
                    changed = execute(i, ns, ref, shares, match, CROSS, price, inside)
        elif kind == 88:  # X
            solid[i] = 0
            order = orders.get(ref)
            if order is None:
                problem("not_live_cancel", ns, f"reference {ref} is not live")
                changed = False
            elif shares <= 0:
                problem("empty_cancel", ns, f"{shares} shares cancelled in {ref}")
                changed = False
            else:
                if shares > order[2]:
                    problem("over_cancel", ns, f"{shares} shares cancelled of {order[2]} left in {ref}")
                    shares = order[2]
                if (bids if order[0] else asks).reduce(order[1], shares):
                    state["dirty"] = True
                order[2] -= shares
                if not order[2]:
                    counts["cancels_to_zero"] += 1
                    del orders[ref]
                if inside:
                    emit(ns, PARTIAL_CANCEL, ref, shares, order[1], 1 if order[0] else -1, 0, 0)
        elif kind == 80:  # P
            changed = False
            counts["p_side_not_b"] += side != 1
            counts["p_with_reference"] += ref != 0
            if inside:
                sequence.append((ns, match, 0, i))
                owner.append(-1 - len(hidden_ns))
                hidden_ns.append(ns)
                hidden_size.append(shares)
                hidden_price.append(price)
                hidden_match.append(match)
        elif kind == 81:  # Q
            changed = False
            if inside:
                emit(ns, CROSS, 0, shares, price, 0, match, 0)
        elif kind == 72:  # H
            changed = False
            if state["seen_action"]:
                problem("later_trading_action", ns, f"a trading action ({codes[i]!r}) after the start of day's")
            elif codes[i] not in ("T", ""):
                problem("later_trading_action", ns, f"the first trading action says {codes[i]!r}, not 'T'")
            state["seen_action"] = True
        elif kind == 83:  # S
            changed = False
            if codes[i] == "E":
                state["end_of_hours"] = True
        else:
            raise ItchError(f"message type {chr(kind)!r} is not one read_messages returns")
        if changed and inside:
            check_touch(ns)

    if state["end_of_hours"] and orders:
        problem("live_orders_after_end_of_system_hours", 0, f"{len(orders)} orders live at the end of the day")
        counts["live_orders_after_end_of_system_hours"] = len(orders)
    rows = len(r_ns)
    free = np.concatenate(([0], np.cumsum(np.frombuffer(solid, np.uint8), dtype=np.int64)))

    def joined(a: int, b: int) -> bool:
        return b == a + 1 or free[b] == free[a + 1]  # nothing between them that ends a run

    group_of, group_side = _groups(sequence, None if merge_within is None else int(round(merge_within * 1e9)), joined)
    group = np.full(rows, -1, np.int64)
    hidden_group = np.zeros(len(hidden_ns), np.int64)
    for g, o in zip(group_of, owner, strict=True):
        if o >= 0:
            group[o] = g
        else:
            hidden_group[-1 - o] = g
    aggressor = -np.asarray(group_side, np.int64)[hidden_group] if len(hidden_ns) else np.zeros(0, np.int64)

    ns_array = np.frombuffer(r_ns, np.int64)
    out = pd.DataFrame({
        "time": ns_array / 1e9, "type": np.frombuffer(r_kind, np.int64), "order_id": np.frombuffer(r_id, np.int64),
        "size": np.frombuffer(r_size, np.int64), "price": np.frombuffer(r_price, np.int64),
        "direction": np.frombuffer(r_dir, np.int64), "ns": ns_array, "match": np.frombuffer(r_match, np.int64),
        "replace": np.frombuffer(r_rep, np.int64).astype(bool), "group": group,
        "paired": np.frombuffer(r_paired, np.int64).astype(bool),
        "priced": np.frombuffer(r_priced, np.int64).astype(bool),
    })
    book = _book_frame(np.frombuffer(snaps, np.int64).reshape(-1, 4 * L), np.frombuffer(change_rows, np.int64),
                       rows, levels)
    hidden_ns_array = np.frombuffer(hidden_ns, np.int64)
    hidden = pd.DataFrame({
        "ns": hidden_ns_array, "time": hidden_ns_array / 1e9, "size": np.frombuffer(hidden_size, np.int64),
        "price": np.frombuffer(hidden_price, np.int64), "match": np.frombuffer(hidden_match, np.int64),
        "group": hidden_group, "aggressor": aggressor,
    })
    info = {"start": start, "end": end, "levels": levels, "merge_within": merge_within,
            "symbol": messages.attrs.get("symbol")}
    for frame in (out, book, hidden):
        frame.attrs.update(info)
    if report is not None:
        report.update(counts)
        report["live_orders_at_end"] = len(orders)
        report["rows"] = rows
        report["groups"] = len(group_side)
    return Frames(out, book, hidden)


def _non_printable(messages: pd.DataFrame, kinds: np.ndarray, flags: list[str], n: int) -> dict[int, str]:
    """For each non-printable ``C`` (by message index): ``cross`` at the nanosecond of a ``Q``, ``paired`` where an
    ``E``, a printable ``C`` or a ``P`` carries the same match number at the same nanosecond (the printed half of
    a trade between two displayed orders), else ``unpaired``. Match numbers are nonzero on a real day."""
    out: dict[int, str] = {}
    if not n:
        return out
    printable = np.asarray(flags)
    c_n = np.flatnonzero((kinds == ord("C")) & (printable == "N"))
    if not len(c_n):
        return out
    ns, match = messages["ns"].to_numpy(np.int64), messages["match"].to_numpy(np.int64)
    at_cross = set(ns[kinds == ord("Q")].tolist())
    printed = (kinds == ord("E")) | ((kinds == ord("C")) & (printable == "Y")) | (kinds == ord("P"))
    candidates = printed & np.isin(match, match[c_n])
    pairs = set(zip(match[candidates].tolist(), ns[candidates].tolist(), strict=True))
    for i, m, t in zip(c_n.tolist(), match[c_n].tolist(), ns[c_n].tolist(), strict=True):
        out[i] = "cross" if t in at_cross else "paired" if m != 0 and (m, t) in pairs else "unpaired"
    return out


def _book_frame(snaps: np.ndarray, change_rows: np.ndarray, rows: int, levels: int) -> pd.DataFrame:
    """The book frame from the snapshots taken where the book changed (ask prices, sizes, bid prices, sizes)."""
    if rows:
        at = np.searchsorted(change_rows, np.arange(rows), side="right") - 1
        full = snaps[at]
    else:
        full = np.zeros((0, 4 * levels), np.int64)
    columns: dict[str, np.ndarray] = {}
    names = iter(book_columns(levels))
    for i in range(levels):
        for block, empty in ((0, EMPTY_ASK), (1, None), (2, EMPTY_BID), (3, None)):
            values = full[:, block * levels + i]
            name = next(names)
            if empty is None:
                columns[name] = values.copy()
            else:
                columns[name] = np.where(values == empty, np.nan, values.astype(np.float64))
    return pd.DataFrame(columns)


# ---- Saving and loading ----

def _paths(directory: str | Path, symbol: str) -> dict[str, Path]:
    base = Path(directory)
    return {name: base / f"{symbol}_{name}.parquet" for name in ("messages", "book", "hidden")} | {
        "meta": base / f"{symbol}_frames.json"}


def save_frames(frames: Frames, directory: str | Path, symbol: Optional[str] = None) -> dict[str, Path]:
    """Write the three frames as parquet (needs pyarrow) under ``<directory>/<SYMBOL>_<name>.parquet``.

    Also writes ``<SYMBOL>_frames.json`` with the window, levels and grouping option they were made
    with (the frames' ``attrs``). ``symbol`` defaults to the one in ``attrs``. Returns the paths.
    """
    symbol = (symbol or frames.messages.attrs.get("symbol") or "").upper()
    if not symbol:
        raise ValueError("give the symbol")
    Path(directory).mkdir(parents=True, exist_ok=True)
    paths = _paths(directory, symbol)
    for name, frame in zip(("messages", "book", "hidden"), frames, strict=True):
        frame.to_parquet(paths[name], engine="pyarrow", compression="zstd", index=False)
    paths["meta"].write_text(json.dumps(dict(frames.messages.attrs), indent=1) + "\n")
    return paths


def load_frames(directory: str | Path, symbol: str) -> Frames:
    """The frames ``save_frames`` wrote for ``symbol`` (dtypes as made, and ``attrs`` restored)."""
    paths = _paths(directory, symbol.upper())
    loaded = [pd.read_parquet(paths[name], engine="pyarrow") for name in ("messages", "book", "hidden")]
    info = json.loads(paths["meta"].read_text())
    for frame in loaded:
        frame.attrs.update(info)
    return Frames(*loaded)


__all__ = [
    "BOOK_TYPES", "CROSS", "EXTRA_COLUMNS", "HIDDEN_COLUMNS", "ItchError", "Frames", "MESSAGE_LENGTHS", "PROBLEMS",
    "count_messages", "extract", "load_frames", "lobster_frames", "read_messages", "save_frames", "scan",
]
