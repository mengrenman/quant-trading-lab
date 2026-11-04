"""Recordings of a dashboard session, to play back (``LOBManager`` in app/manager.py runs both).

A recording holds the book as it stood when recording began, with whose each resting order
was, the user's account and the strategies' settings, accounts and state (``start``); and a
tape of everything that happened afterwards: every order and cancellation with whose it was,
changes of matching engine and model, and each strategy's decisions, starts, stops and
removals. The matching engines are deterministic, so feeding the tape into the same book
reproduces the session: the book, the trades, the user's fills and P&L, and the strategies'.

A recording is saved as a NumPy ``.npz`` file: one array per column of the tape, and the rest
as JSON in the ``header`` array. It loads without pickle, so a notebook can read one too::

    data = numpy.load("recording-1.npz", allow_pickle=False)
    header = json.loads(str(data["header"]))
    times, kinds = data["time"], data["kind"]  # kinds: KIND_NAMES

Files from elsewhere are untrusted: ``from_bytes`` checks sizes before reading anything, and
every field before a recording is accepted.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from array import array
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# The kinds of event on a tape.
LIMIT, MARKET, CANCEL, ENGINE, MODEL_CHANGE, DECIDE, STRATEGY_START, STRATEGY_STOP, STRATEGY_RESUME, \
    STRATEGY_REMOVE = range(10)
KIND_NAMES = ("limit", "market", "cancel", "engine", "model", "decide", "strategy_start", "strategy_stop",
              "strategy_resume", "strategy_remove")
BOOK_KINDS = (LIMIT, MARKET, CANCEL)
STRATEGY_KINDS = (DECIDE, STRATEGY_START, STRATEGY_STOP, STRATEGY_RESUME, STRATEGY_REMOVE)
# Whose an order was: the model's, the user's, or a strategy's (its id, from 1).
FROM_MODEL, FROM_YOU = 0, -1
BID, ASK, NO_SIDE = 0, 1, -1

MAX_EVENTS = 500_000  # a recording stops by itself at this many events
MAX_RECORDINGS = 3  # kept in a session at once
MAX_FILE_BYTES = 40 * 2**20  # an upload; a full recording is about 6 MB
FORMAT, VERSION = "lob-simulator-recording", 1

# (name, array type code, NumPy type): about 34 bytes an event.
COLUMNS = (("kind", "b", np.int8), ("time", "d", np.float64), ("oid", "q", np.int64), ("side", "b", np.int8),
           ("price", "i", np.int32), ("size", "i", np.int32), ("display", "i", np.int32), ("origin", "q", np.int64))


class Tape:
    """Events in the order they happened, as typed columns, with the few long ones' details in ``extras``.

    ``oid`` is the order an event is about (for a strategy event, the strategy), ``display`` an
    iceberg's peak (0 for an ordinary order), ``origin`` whose the order was.
    """

    def __init__(self) -> None:
        for name, code, _ in COLUMNS:
            setattr(self, name, array(code))
        self.extras: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.kind)

    def append(self, kind: int, time: float, oid: int = 0, side: int = NO_SIDE, price: int = 0, size: int = 0,
               display: int = 0, origin: int = FROM_MODEL, extra: Optional[dict[str, Any]] = None) -> None:
        if extra is not None:
            self.extras[len(self.kind)] = extra
        self.kind.append(kind)
        self.time.append(time)
        self.oid.append(oid)
        self.side.append(side)
        self.price.append(price)
        self.size.append(size)
        self.display.append(display)
        self.origin.append(origin)

    def index_after(self, time: float) -> int:
        """How many events happened at or before ``time``."""
        return bisect_right(self.time, time)


@dataclass
class Recording:
    id: int
    name: str
    created: str  # when recording began, local time
    start: dict[str, Any]
    tape: Tape
    end_time: float  # the simulation time when recording stopped
    meta: dict[str, Any] = field(default_factory=dict)  # summary(), computed once

    @property
    def start_time(self) -> float:
        return float(self.start["time"])

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "created": self.created, "start": self.start_time,
                "end": self.end_time, "events": len(self.tape), **self.meta}


def summary(start: dict[str, Any], tape: Tape) -> dict[str, Any]:
    """What a recording contains, for its card: its book events, the user's orders, models and strategies."""
    from app.model_specs import MODEL_SPECS

    kinds = np.frombuffer(tape.kind, dtype=np.int8)
    origins = np.frombuffer(tape.origin, dtype=np.int64)
    submissions = (kinds == LIMIT) | (kinds == MARKET)
    models = [start["model"]] + [tape.extras[i]["model"] for i in sorted(tape.extras) if tape.kind[i] == MODEL_CHANGE]
    labels = list(dict.fromkeys(MODEL_SPECS[m].label for m in models))  # in order, once each
    strategies = [s["name"] for s in start["strategies"]] + [
        tape.extras[i]["name"] for i in sorted(tape.extras) if tape.kind[i] == STRATEGY_START]
    return {"book_events": int(np.isin(kinds, BOOK_KINDS).sum()),
            "your_orders": int((submissions & (origins == FROM_YOU)).sum()),
            "models": labels, "strategies": list(dict.fromkeys(strategies))}


class Recorder:
    """A recording in progress: the starting state, and the tape as it grows."""

    def __init__(self, start: dict[str, Any], created: str) -> None:
        self.start = start
        self.created = created
        self.tape = Tape()

    @property
    def full(self) -> bool:
        return len(self.tape) >= MAX_EVENTS

    def order(self, kind: int, time: float, oid: int, side: int, price: int, size: int, display: int,
              origin: int) -> None:
        self.tape.append(kind, time, oid, side, price, size, display, origin)

    def cancel(self, time: float, oid: int, origin: int) -> None:
        self.tape.append(CANCEL, time, oid, origin=origin)

    def note(self, kind: int, time: float, oid: int = 0, extra: Optional[dict[str, Any]] = None) -> None:
        """An event that is not an order: an engine or model change, or a strategy's."""
        self.tape.append(kind, time, oid, extra=extra)

    def finish(self, rec_id: int, name: str, end_time: float) -> Recording:
        return Recording(id=rec_id, name=name, created=self.created, start=self.start, tape=self.tape,
                         end_time=end_time, meta=summary(self.start, self.tape))


# ---- Files ----

def to_bytes(rec: Recording) -> bytes:
    """The recording as an ``.npz`` file."""
    header = {"format": FORMAT, "version": VERSION, "name": rec.name, "created": rec.created,
              "end_time": rec.end_time, "start": rec.start, "kind_names": KIND_NAMES,
              "extras": {str(i): e for i, e in sorted(rec.tape.extras.items())}}
    columns = {name: np.frombuffer(getattr(rec.tape, name), dtype=dtype) if len(rec.tape) else np.zeros(0, dtype)
               for name, _, dtype in COLUMNS}
    buf = io.BytesIO()
    np.savez_compressed(buf, header=np.array(json.dumps(header)), **columns)
    return buf.getvalue()


def from_bytes(data: bytes, rec_id: int) -> Recording:
    """A recording from an ``.npz`` file of ``to_bytes``; raises ValueError, saying why, for anything else."""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"the file is larger than {MAX_FILE_BYTES // 2**20} MB")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        members = {info.filename: info for info in archive.infolist()}
    except (zipfile.BadZipFile, ValueError):
        raise ValueError("this is not a recording from this simulator (not an .npz file)") from None
    wanted = ["header.npy"] + [f"{name}.npy" for name, _, _ in COLUMNS]
    missing = [name for name in wanted if name not in members]
    if missing:
        raise ValueError(f"this is not a recording from this simulator (no {missing[0][:-4]})")
    arrays = {}
    for name in wanted:
        # The most a member may hold: the header's JSON, or the widest column of MAX_EVENTS, with its header.
        limit = 8 * 2**20 if name == "header.npy" else MAX_EVENTS * 8 + 4096
        too_long = "its header is too long" if name == "header.npy" else \
            f"the recording is longer than {MAX_EVENTS:,} events"
        if members[name].file_size > limit:
            raise ValueError(too_long)
        try:
            with archive.open(name) as member:
                raw = member.read(limit + 1)  # never more, whatever the archive says
        except (ValueError, OSError, zipfile.BadZipFile, EOFError) as exc:
            raise ValueError(f"its {name[:-4]} cannot be read ({exc})") from None
        if len(raw) > limit:
            raise ValueError(too_long)
        try:
            arrays[name[:-4]] = np.load(io.BytesIO(raw), allow_pickle=False)
        except (ValueError, OSError, EOFError) as exc:
            raise ValueError(f"its {name[:-4]} cannot be read ({exc})") from None
    try:
        header = json.loads(str(arrays.pop("header")[()]))
    except (ValueError, TypeError, IndexError):
        raise ValueError("its header is not JSON") from None
    if not isinstance(header, dict) or header.get("format") != FORMAT:
        raise ValueError("this is not a recording from this simulator")
    if header.get("version") != VERSION:
        raise ValueError(f"it is a version {header.get('version')!r} recording; this simulator reads version {VERSION}")

    tape = Tape()
    n = None
    for name, _, dtype in COLUMNS:
        column = arrays[name]
        if column.ndim == 1 and len(column) > MAX_EVENTS:
            raise ValueError(f"the recording is longer than {MAX_EVENTS:,} events")
        if column.ndim != 1 or column.dtype.kind not in "iuf" or (n is not None and len(column) != n):
            raise ValueError(f"its {name} column is not a list of numbers as long as the others")
        n = len(column)
        if dtype is not np.float64 and column.dtype.kind == "f":
            if not (np.isfinite(column).all() and (column == np.round(column)).all()):
                raise ValueError(f"its {name} column must hold whole numbers")
        if dtype is not np.float64 and len(column) and (column.min() < np.iinfo(dtype).min
                                                         or column.max() > np.iinfo(dtype).max):
            raise ValueError(f"its {name} column holds numbers out of range")
        getattr(tape, name).frombytes(column.astype(dtype).tobytes())
    extras = header.get("extras")
    if not isinstance(extras, dict):
        raise ValueError("its extras are missing")
    try:
        tape.extras = {int(i): e for i, e in extras.items()}
    except (TypeError, ValueError):
        raise ValueError("its extras are not numbered by event") from None
    start = header.get("start")
    name = header.get("name")
    created = header.get("created")
    end_time = header.get("end_time")
    check(start, tape)
    if not isinstance(end_time, (int, float)) or isinstance(end_time, bool) or not math.isfinite(end_time) \
            or end_time < (tape.time[-1] if len(tape) else start["time"]):
        raise ValueError("its end time is before its last event")
    name = name.strip()[:60] if isinstance(name, str) and name.strip() else f"Uploaded {rec_id}"
    created = created[:40] if isinstance(created, str) else ""
    return Recording(id=rec_id, name=name, created=created, start=start, tape=tape, end_time=float(end_time),
                     meta=summary(start, tape))


# ---- Checks ----

def _number(value: Any, what: str, low: float = -math.inf, high: float = math.inf, whole: bool = False) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) \
        and low <= value <= high and (not whole or float(value).is_integer())
    if not ok:
        kind = "a whole number" if whole else "a number"
        span = f" from {low:g}" if low > -math.inf else ""
        raise ValueError(f"{what} must be {kind}{span}")
    return value


def _strategy(spec_key: Any, values: Any, what: str):
    """The strategy a recording names, built from its settings (ValueError if they are not valid)."""
    from app.strategy_specs import STRATEGY_SPECS

    spec = STRATEGY_SPECS.get(spec_key) if isinstance(spec_key, str) else None
    if spec is None or not isinstance(values, dict):
        raise ValueError(f"{what} is not one of the strategies here")
    parsed, errors = spec.parse(values)
    if errors or set(values) - set(parsed):
        raise ValueError(f"{what}: {next(iter(errors.values()), 'unknown settings')}")
    try:
        return spec, parsed, spec.build(parsed)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from None


def check(start: Any, tape: Tape) -> None:
    """Raise ValueError, saying what is wrong, unless ``start`` and ``tape`` could have been recorded here."""
    from dataclasses import fields

    from app.manager import ENGINE_NAMES, Account
    from app.model_specs import MODEL_SPECS

    if not isinstance(start, dict):
        raise ValueError("its starting state is missing")
    t0 = _number(start.get("time"), "its start time", 0)
    for key in ("event_count", "traded_volume", "cum_volume"):
        _number(start.get(key), key.replace("_", " "), 0, whole=True)
    _number(start.get("cum_turnover"), "its turnover", 0)
    _number(start.get("reference_price"), "its reference price", 0)
    _number(start.get("seed"), "its seed", whole=True)
    if start.get("engine") not in ENGINE_NAMES:
        raise ValueError("its matching engine is not one of the engines here")
    if start.get("model") not in MODEL_SPECS or not isinstance(start.get("model_params"), dict):
        raise ValueError("its model is not one of the models here")

    account_fields = {f.name for f in fields(Account)}

    def account(value: Any, whose: str) -> None:
        if not isinstance(value, dict) or set(value) != account_fields:
            raise ValueError(f"{whose} account is not complete")
        for key, v in value.items():
            if not (key == "avg_cost" and v is None):
                _number(v, f"{whose} {key.replace('_', ' ')}", whole=key in ("position", "fills", "bought", "sold",
                                                                          "maker_lots", "taker_lots", "markout_lots"))

    account(start.get("account"), "your")
    strategy_ids: set[int] = set()
    strategies = start.get("strategies")
    if not isinstance(strategies, list):
        raise ValueError("its strategies are missing")
    for s in strategies:
        if not isinstance(s, dict):
            raise ValueError("one of its strategies is not described")
        sid = int(_number(s.get("id"), "a strategy's id", 1, whole=True))
        name = s.get("name")
        if sid in strategy_ids or not isinstance(name, str) or not 0 < len(name) <= 60:
            raise ValueError("its strategies need different ids and names")
        strategy_ids.add(sid)
        _, _, built = _strategy(s.get("kind"), s.get("values"), name)
        _number(s.get("number"), f"{name}'s number", 1, whole=True)
        _number(s.get("next_time"), f"{name}'s next decision", 0)
        if not isinstance(s.get("running"), bool):
            raise ValueError(f"{name} must be running or not")
        if s.get("signal") is not None:
            _number(s.get("signal"), f"{name}'s signal")
        if s.get("target") is not None:
            _number(s.get("target"), f"{name}'s target", whole=True)
        quotes = s.get("quotes")
        if not isinstance(quotes, dict) or set(quotes) - {"bid", "ask"} or not all(
                isinstance(q, list) and len(q) == 2 for q in quotes.values()):
            raise ValueError(f"{name}'s quotes are not readable")
        for q in quotes.values():
            _number(q[0], f"{name}'s quote", 1, whole=True)
            _number(q[1], f"{name}'s quote size", 1, whole=True)
        account(s.get("account"), f"{name}'s")
        state = s.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"{name}'s state is missing")
        built.restore(state)  # raises ValueError for one it could not have had

    orders = start.get("orders")
    if not isinstance(orders, list):
        raise ValueError("its book is missing")
    seen: set[int] = set()
    for row in orders:
        if not isinstance(row, list) or len(row) != 8:
            raise ValueError("an order in its book is not readable")
        oid, side, price, size, hidden, display, timestamp, origin = row
        _number(oid, "an order's id", whole=True)
        if oid in seen or side not in (BID, ASK):
            raise ValueError("its book has an order twice, or one on neither side")
        seen.add(oid)
        _number(price, "an order's price", 1, 2**31 - 1, whole=True)
        _number(size, "an order's size", 1, 2**31 - 1, whole=True)
        _number(hidden, "an iceberg's reserve", 0, 2**31 - 1, whole=True)
        _number(display, "an iceberg's peak", 0, 2**31 - 1, whole=True)
        if (hidden and not display) or (display and size > display):
            raise ValueError("an iceberg in its book shows more than its peak")
        _number(timestamp, "an order's time", 0)
        if origin not in (FROM_MODEL, FROM_YOU) and origin not in strategy_ids:
            raise ValueError("an order in its book belongs to no one it knows")
    bids = [r[2] for r in orders if r[1] == BID]
    asks = [r[2] for r in orders if r[1] == ASK]
    if bids != sorted(bids, reverse=True) or asks != sorted(asks) or (bids and asks and bids[0] >= asks[0]):
        raise ValueError("its book is not in price order, or its best bid is not below its best ask")
    markouts = start.get("markouts")
    if not isinstance(markouts, list):
        raise ValueError("its markouts are missing")
    for row in markouts:
        if not (isinstance(row, list) and len(row) == 5 and isinstance(row[2], bool)):
            raise ValueError("a markout is not readable")
        _number(row[0], "a markout's due event", 0, whole=True)
        if row[1] != FROM_YOU and row[1] not in strategy_ids:
            raise ValueError("a markout belongs to no one it knows")
        _number(row[3], "a markout's price", 1)
        _number(row[4], "a markout's size", 1, whole=True)

    # The tape, a column at a time.
    n = len(tape)
    if not n:
        return
    kinds = np.frombuffer(tape.kind, dtype=np.int8)
    times = np.frombuffer(tape.time, dtype=np.float64)
    oids = np.frombuffer(tape.oid, dtype=np.int64)
    sides = np.frombuffer(tape.side, dtype=np.int8)
    prices = np.frombuffer(tape.price, dtype=np.int32)
    sizes = np.frombuffer(tape.size, dtype=np.int32)
    displays = np.frombuffer(tape.display, dtype=np.int32)
    origins = np.frombuffer(tape.origin, dtype=np.int64)
    if kinds.min() < 0 or kinds.max() >= len(KIND_NAMES):
        raise ValueError("its tape has events of a kind this simulator does not know")
    if not np.isfinite(times).all() or times[0] < t0 or (np.diff(times) < 0).any():
        raise ValueError("its tape's times go backward, or start before the recording")
    submit = (kinds == LIMIT) | (kinds == MARKET)
    if not np.isin(sides[submit], (BID, ASK)).all() or (sizes[submit] < 1).any():
        raise ValueError("its tape has an order on neither side, or of no size")
    if (prices[kinds == LIMIT] < 1).any() or (displays < 0).any() or (displays[kinds == MARKET] != 0).any():
        raise ValueError("its tape has a limit order without a price, or a market order with a peak")
    for i, extra in tape.extras.items():
        if not 0 <= i < n or kinds[i] not in (ENGINE, MODEL_CHANGE, STRATEGY_START) or not isinstance(extra, dict):
            raise ValueError("its tape's details do not match its events")
    started = set(strategy_ids)
    for i in np.flatnonzero(np.isin(kinds, (ENGINE, MODEL_CHANGE) + STRATEGY_KINDS)):
        kind, extra = kinds[i], tape.extras.get(int(i))
        if kind == ENGINE:
            if extra is None or extra.get("engine") not in ENGINE_NAMES:
                raise ValueError("its tape changes to an engine this simulator does not have")
        elif kind == MODEL_CHANGE:
            if extra is None or extra.get("model") not in MODEL_SPECS or not isinstance(extra.get("params"), dict):
                raise ValueError("its tape changes to a model this simulator does not have")
        elif kind == STRATEGY_START:
            sid = int(oids[i])
            name = (extra or {}).get("name")
            if extra is None or sid < 1 or sid in started or not isinstance(name, str) or not 0 < len(name) <= 60:
                raise ValueError("its tape starts a strategy without an id and a name of its own")
            _strategy(extra.get("kind"), extra.get("values"), name)
            _number(extra.get("number"), f"{name}'s number", 1, whole=True)
            started.add(sid)
        elif int(oids[i]) not in started:
            raise ValueError("its tape has a strategy act before it started")
    known = np.isin(origins, list(started | {FROM_MODEL, FROM_YOU}))
    if not known.all():
        raise ValueError("its tape has orders of a strategy it never started")
