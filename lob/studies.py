"""Microstructure studies on the stream of book events: order flow, trade sides, order-to-trade ratios and depth.

Each study reads the events one at a time and keeps the little state its definition needs, so a study costs a
few microseconds an event, and nothing when it is switched off. They are defined here, apart from any dashboard,
so that a notebook can run them on a ``Simulation`` (its ``observer``) and the dashboard's session can run them
on a live market and a replay: the same events give the same values in both.

``observe`` is called once for each event that changes the book, after the matching engine has run:

    observe(t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book, own=False)

- ``t`` is the simulation clock, not an order's or a fill's own timestamp (those lag it in a replay). It never
  decreases, and several events can share it.
- ``kind`` is ``ARRIVAL_LIMIT``, ``ARRIVAL_MARKET`` or ``CANCEL``. Only the models send ``ARRIVAL_MARKET``: a
  user's market order is a limit order at a protected price, so whether an event traded is ``bool(fills)``.
- ``side`` and ``price`` are the order's; for a cancel, the canceled order's (read from the book before the
  engine ran; a cancel of an order that is not there changes nothing and is not observed). ``price`` is 0 for a
  market order. ``size`` is the order's whole size when it arrived, not the peak an iceberg shows, or the canceled
  size (an iceberg's visible part).
- ``fills`` are the engine's ``Fill`` objects for this event (an empty sequence if none), of which only ``.price``
  and ``.size`` are read. ``aggressor`` is +1 if the arriving order bought, -1 if it sold (0 for a cancel).
- ``quotes_before`` and ``quotes_after`` are ``(best bid, lots there, best ask, lots there)`` around the event,
  a price ``None`` for an empty side. They may be the same object when the event cannot have changed the touch.
- ``book`` is the book after the event, for the studies that read depth beyond the touch. It is valid for the
  call only; nothing here keeps it.
- ``own`` is True when the order is the user's or a strategy's rather than the model's; only the order-to-trade
  ratio looks at it.

Windows are in simulation time and exact: a window of ``W`` units holds the events with time in ``(t - W, t]``,
kept in a deque with its sum, so an update is O(1) amortized. A study that needs a full window answers ``None``
until ``W`` units have passed since its last reset. Reading a window at time ``t`` drops what has left it, so a
study must be read at times that do not decrease. A value that is undefined is ``None``, never NaN or infinity.

Studies that read the book (imbalance, spread, depth) are computed when asked, not on every event, and keep
their history by ``sample``, once a unit of simulation time, which ``StudySet.sample_to`` drives. A
``StudySet`` holds a list of studies, passes events to those that are switched on, and does the unit sampling.

Definitions follow Cont, Kukanov and Stoikov (2014) for order flow imbalance, Lee and Ready (1991) for the quote
rule, Easley, Lopez de Prado and O'Hara (2012) for VPIN, and Large (2007) for the restoration of the quotes. Where
a definition follows the code of VisualHFT (Apache License 2.0, github.com/visualHFT/VisualHFT) the study says so.
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from lob.order import Side

ARRIVAL_LIMIT = 0
ARRIVAL_MARKET = 1
CANCEL = 2

MIN_SAMPLES = 20       # unit samples a baseline, a spread scale or a z-score needs before it is defined
HISTORY = 100          # units of samples behind a baseline, a spread scale or a z-score
CALIBRATION_FILLS = 200  # fills whose median size sets the automatic VPIN bucket
BUCKET_MEDIANS = 50    # automatic VPIN bucket, in median fills (VisualHFT's help text recommends 20 to 50 trades)
BVC_BUCKETS = 100      # buckets behind the standard deviation of price changes in bulk volume classification
RECOVERIES = 20        # recovery times behind the median and the history of the resilience score
MAX_PENDING = 20_000   # items a window may hold before an update drains it (reads always do)
_SQRT2 = math.sqrt(2.0)
_BID = Side.BID


def book_quotes(book) -> tuple:
    """The book's quotes as the studies take them: (best bid, lots there, best ask, lots there)."""
    bid, ask = book.best_bid, book.best_ask
    return (bid, book._bid_depth_at[bid] if bid is not None else 0,
            ask, book._ask_depth_at[ask] if ask is not None else 0)


def ofi_increment(before: tuple, after: tuple) -> float:
    """One event's order flow imbalance e_n, as in ``lob.metrics.order_flow_imbalance``.

    Each state is (best bid, depth there, best ask, depth there), a price None for an empty side.
    """
    pb0, qb0, pa0, qa0 = before
    pb1, qb1, pa1, qa1 = after
    if pb0 is None or pa0 is None or pb1 is None or pa1 is None:
        return 0.0
    return ((qb1 if pb1 >= pb0 else 0) - (qb0 if pb1 <= pb0 else 0)
            - (qa1 if pa1 <= pa0 else 0) + (qa0 if pa1 >= pa0 else 0))


def _mid(quotes: tuple) -> Optional[float]:
    return None if quotes[0] is None or quotes[2] is None else (quotes[0] + quotes[2]) / 2.0


def _median(values) -> Optional[float]:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    return float(ordered[n // 2]) if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


@dataclass(frozen=True)
class Setting:
    """One adjustable setting of a study and its allowed range (``options`` instead of a range for a choice)."""
    name: str
    label: str
    default: Any
    low: float = 0.0
    high: float = 0.0
    integer: bool = False
    options: tuple = ()

    def parse(self, raw: Any) -> Any:
        """The value in range, or a ValueError with a readable message."""
        if self.options:
            if raw in self.options:
                return raw
            raise ValueError(f"{self.label}: choose {' or '.join(self.options)}")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{self.label}: enter a number") from None
        if isinstance(raw, bool) or not math.isfinite(value):
            raise ValueError(f"{self.label}: enter a finite number")
        if self.integer:
            if value != int(value):
                raise ValueError(f"{self.label}: enter a whole number")
            value = int(value)
        if not self.low <= value <= self.high:
            raise ValueError(f"{self.label}: must be between {self.low:g} and {self.high:g}")
        return value


class _Window:
    """The (time, amount) items of the last ``span`` units of simulation time, and their sum."""
    __slots__ = ("span", "items", "total")

    def __init__(self, span: float):
        self.span = span
        self.items: deque = deque()
        self.total = 0

    def add(self, t: float, x) -> None:
        items = self.items
        items.append((t, x))
        self.total += x
        if len(items) > MAX_PENDING:
            self.sum_at(t)

    def sum_at(self, t: float):
        items = self.items
        cutoff = t - self.span
        while items and items[0][0] <= cutoff:
            self.total -= items.popleft()[1]
        return self.total


class _Samples:
    """The last ``size`` numbers, with their sum, sum of squares and median kept as they come and go.

    The numbers the studies keep here are whole or half-integers, so the sums are exact.
    """
    __slots__ = ("size", "items", "ordered", "total", "squares")

    def __init__(self, size: int):
        self.size = size
        self.items: deque = deque()
        self.ordered: list = []
        self.total = 0
        self.squares = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, x) -> None:
        items = self.items
        items.append(x)
        insort(self.ordered, x)
        self.total += x
        self.squares += x * x
        if len(items) > self.size:
            old = items.popleft()
            del self.ordered[bisect_left(self.ordered, old)]
            self.total -= old
            self.squares -= old * old

    def median(self) -> Optional[float]:
        n = len(self.ordered)
        if n == 0:
            return None
        return float(self.ordered[n // 2]) if n % 2 else (self.ordered[n // 2 - 1] + self.ordered[n // 2]) / 2.0

    def mean_sd(self, ddof: int = 0) -> tuple[Optional[float], Optional[float]]:
        """The mean and the standard deviation (dividing by n - ddof); None where undefined."""
        n = len(self.items)
        if n == 0:
            return None, None
        mean = self.total / n
        if n - ddof <= 0:
            return mean, None
        variance = (n * self.squares - self.total * self.total) / (n * (n - ddof))
        return mean, math.sqrt(variance) if variance > 0 else 0.0


class Study:
    """What a study is: settings, data that ``reset`` clears, ``observe``, values, and a sample once a unit.

    A subclass sets ``key`` (a short name), ``label``, ``SETTINGS`` and ``headline`` (the key of its main value
    in ``values``), and says in ``needs_fills`` whether only events that traded matter to it, and in
    ``sees_events`` whether events matter to it at all (the studies that read the book when sampled do not). Settings
    are attributes of the study; ``configure`` changes them and starts the study afresh, since data gathered under
    other settings means something else.
    """

    key = ""
    label = ""
    headline = ""
    SETTINGS: tuple = ()
    needs_fills = False
    sees_events = True

    def __init__(self, **settings):
        self.enabled = True
        for spec in self.SETTINGS:
            setattr(self, spec.name, spec.default)
        values = self._parse(settings)
        self._check({**self.settings, **values})
        for name, value in values.items():
            setattr(self, name, value)
        self.reset()

    @property
    def settings(self) -> dict:
        return {spec.name: getattr(self, spec.name) for spec in self.SETTINGS}

    def _parse(self, changes: dict) -> dict:
        specs = {spec.name: spec for spec in self.SETTINGS}
        unknown = set(changes) - set(specs)
        if unknown:
            raise ValueError(f"{self.label}: no setting {', '.join(sorted(unknown))}")
        return {name: specs[name].parse(raw) for name, raw in changes.items()}

    def _check(self, settings: dict) -> None:
        """Raise ValueError if the settings contradict each other."""

    def configure(self, **changes) -> bool:
        """Change settings and start afresh from the study's last time; False if nothing changed.

        Nothing is changed if any value is out of range.
        """
        values = self._parse(changes)
        values = {name: value for name, value in values.items() if getattr(self, name) != value}
        if not values:
            return False
        self._check({**self.settings, **values})
        for name, value in values.items():
            setattr(self, name, value)
        self.reset(int(self._t), self._t)
        return True

    def reset(self, index: int = 0, t: Optional[float] = None) -> None:
        """Clear the data (not the settings). ``index`` is the unit of simulation time the study starts in."""
        self.unit = index  # the last unit sampled
        self.start = float(index) if t is None else float(t)
        self._t = self.start
        self._clear()

    def _clear(self) -> None:
        raise NotImplementedError

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        raise NotImplementedError

    def values(self, t: Optional[float] = None, book=None) -> dict:
        """Every value of the study as of time ``t`` (the last event seen if omitted); ``None`` where undefined."""
        raise NotImplementedError

    def value(self, t: Optional[float] = None, book=None) -> Optional[float]:
        """The study's main value, as ``values`` gives it."""
        return self.values(t, book)[self.headline]

    def sample(self, index: int, book) -> Optional[float]:
        """Take the sample of unit ``index``, from the book as it stands, and return the main value then."""
        self.unit = index
        return self.value(float(index), book)


# ---- Order flow ----

class OFI(Study):
    """Order flow imbalance over a trailing window (Cont, Kukanov and Stoikov 2014), in lots, signed.

    The sum of each event's increment e_n, computed from the best quotes and the lots at them before and after:
    lots added at the bid or taken from the ask count as buying pressure. An event that touches an empty side
    adds nothing. Positive means buying pressure.
    """
    key = "ofi"
    label = "Order flow imbalance"
    headline = "ofi"
    SETTINGS = (Setting("window", "Window (time units)", 5.0, 0.5, 100.0),)

    def _clear(self) -> None:
        self._window = _Window(self.window)

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        self._t = t
        if quotes_before is quotes_after:
            return
        e = ofi_increment(quotes_before, quotes_after)
        if e:
            self._window.add(t, e)

    def values(self, t=None, book=None) -> dict:
        t = self._t if t is None else t
        total = self._window.sum_at(t)
        return {"ofi": total if t - self.start >= self.window else None}


class CVD(Study):
    """Cumulative volume delta: lots bought by takers less lots sold by takers since the last reset.

    A buyer-initiated fill adds its size and a seller-initiated one subtracts it, by the true side (the
    arriving order's). ``clear`` starts the count again without touching the other studies.
    """
    key = "cvd"
    label = "Cumulative volume delta"
    headline = "cvd"
    needs_fills = True

    def _clear(self) -> None:
        self._total = 0

    def clear(self) -> None:
        self._total = 0

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        self._t = t
        lots = 0
        for fill in fills:
            lots += fill.size
        self._total += aggressor * lots

    def values(self, t=None, book=None) -> dict:
        return {"cvd": self._total}


class _Lagged:
    """The mid a feed would show with a stale quote: the once-a-unit mid samples, ``quote_lag`` of them back.

    With lag 0 the mid is the one just before the event, which every fill is on the right side of (a fill is at
    the maker's price and the spread is a tick or more), so a rule that uses it is exact by construction. With
    lag L >= 1 it is the sample of unit floor(t) - L + 1: lag 1 is the last whole unit's sample, at most one unit
    old (the newest sample if that unit has yet to be sampled). Before that sample exists, or when the book was
    one-sided at it, there is no mid and a fill is unknown.
    """

    def _reference(self, t: float, quotes_before: tuple) -> Optional[float]:
        lag = self.quote_lag
        if lag == 0:
            return _mid(quotes_before)
        offset = self.unit - (int(t) - lag + 1)
        if offset < 0:
            offset = 0
        mids = self._mids
        return mids[-1 - offset] if offset < len(mids) else None

    def _sample_mid(self, book) -> None:
        if self.quote_lag and self.side == self.lagged_rule:
            self._mids.append(book.mid_price)


class TradeDelta(_Lagged, Study):
    """Delta of trades: lots bought by takers less lots sold, over a trailing window.

    ``side`` says how a fill's side is known: ``true`` (the arriving order's), ``tick`` (above the previous
    fill's price a buy, below a sell, at it the previous sign; unknown until the first change of price) or
    ``quote`` (Lee and Ready 1991: above the mid a buy, below a sell, at it the tick rule; the mid is the one
    ``quote_lag`` units back, see ``_Lagged``). Unknown fills add nothing. For tick and quote, ``coverage`` is the
    share of lots classified and ``accuracy`` the share of those classified as the true side says, both since the
    last reset. The accuracy is only available because a simulator knows the true side.
    """
    key = "delta"
    label = "Delta of trades"
    headline = "delta"
    needs_fills = True
    lagged_rule = "quote"
    SETTINGS = (
        Setting("window", "Window (time units)", 10.0, 0.5, 100.0),
        Setting("side", "Trade sides", "true", options=("true", "tick", "quote")),
        Setting("quote_lag", "Quote lag (units; 0 is exact)", 1, 0, 100, integer=True),
    )

    def _clear(self) -> None:
        self._window = _Window(self.window)
        self._mids: deque = deque(maxlen=max(self.quote_lag, 1))
        self._prev: Optional[float] = None
        self._last: Optional[int] = None
        self._lots = 0
        self._classified = 0
        self._correct = 0

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        self._t = t
        rule = self.side
        if rule == "true":
            lots = 0
            for fill in fills:
                lots += fill.size
            self._window.add(t, aggressor * lots)
            return
        ref = self._reference(t, quotes_before) if rule == "quote" else None
        prev, last = self._prev, self._last
        signed = lots = classified = correct = 0
        for fill in fills:
            p, n = fill.price, fill.size
            lots += n
            if prev is None:
                tick = None
            elif p > prev:
                tick = 1
            elif p < prev:
                tick = -1
            else:
                tick = last
            prev, last = p, tick
            if rule == "tick":
                sign = tick
            elif ref is None:
                sign = None
            else:
                sign = 1 if p > ref else -1 if p < ref else tick
            if sign is not None:
                classified += n
                signed += sign * n
                if sign == aggressor:
                    correct += n
        self._prev, self._last = prev, last
        self._lots += lots
        self._classified += classified
        self._correct += correct
        if signed:
            self._window.add(t, signed)

    def sample(self, index, book):
        self._sample_mid(book)
        return super().sample(index, book)

    def values(self, t=None, book=None) -> dict:
        t = self._t if t is None else t
        total = self._window.sum_at(t)
        scored = self.side != "true"
        return {"delta": total if t - self.start >= self.window else None,
                "coverage": self._classified / self._lots if scored and self._lots else None,
                "accuracy": self._correct / self._classified if scored and self._classified else None}


class VPIN(_Lagged, Study):
    """VPIN, the volume-synchronized probability of informed trading (Easley, Lopez de Prado and O'Hara 2012).

    Fills fill equal-volume buckets of V lots in order; a fill that crosses the end of a bucket is split between
    it and the next. VPIN is the sum over the last ``buckets`` full buckets of |V_buy - V_sell| / (N V), and
    ``None`` until that many are full. V is ``volume`` lots, or with ``volume`` 0 it is 50 times the median size
    of the first 200 fills since the last reset (those fills only calibrate it; VisualHFT's help recommends 20 to
    50 trades a bucket); ``set_volume`` sets it directly, as a replay does to use the live market's V, and
    ``volume_in_use`` reads it.

    ``side`` says how volume is signed. ``true``: the arriving order's side. ``tick``: the tick rule, as for
    ``TradeDelta`` (a fill whose side is unknown fills the bucket and enters neither V_buy nor V_sell; VisualHFT's
    corrected form). ``mid``: a fill at or above the mid ``quote_lag`` units back is a buy, below a sell (VisualHFT's
    default form; with lag 0 it equals ``true`` in a simulator). ``bvc``: bulk volume classification at the bucket
    level, the volume-bar form of Easley, Lopez de Prado and O'Hara; their 2012 paper classified one-minute time
    bars inside a bucket and used the whole sample's sigma. The bucket's buy share is Phi(x), with x the change of
    the mid over the bucket in units of the standard deviation of the last 100 buckets' changes (this bucket's
    included), so |V_buy - V_sell| / V = |2 Phi(x) - 1|. That is uniform on (0, 1) when x is normal, with mean 1/2:
    BVC-VPIN sits near 0.5 whatever the flow and rises only when recent price changes outrun the trailing sigma
    (Andersen and Bondarenko 2014).

    The sides are on different scales. With the true side and n equal fills a bucket, independent fair-coin sides
    give about sqrt(2 / (pi n)), 0.11 at n = 50. VisualHFT's website bands (0.60 and 0.75, described for its tick-rule
    form) are not carried over: they are not in its code, whose default classifies against the mid.
    """
    key = "vpin"
    label = "VPIN"
    headline = "vpin"
    needs_fills = True
    lagged_rule = "mid"
    SETTINGS = (
        Setting("buckets", "Buckets (N)", 50, 1, 1000, integer=True),
        Setting("volume", "Lots a bucket (0: 50 median fills)", 0, 0, 1_000_000),
        Setting("side", "Trade sides", "true", options=("true", "tick", "mid", "bvc")),
        Setting("quote_lag", "Quote lag (units; mid rule)", 1, 0, 100, integer=True),
    )

    def __init__(self, log_buckets: bool = False, **settings):
        self.log: Optional[list] = [] if log_buckets else None
        super().__init__(**settings)

    def _check(self, settings: dict) -> None:
        if 0 < settings["volume"] < 1:
            raise ValueError("Lots a bucket: at least 1, or 0 to set it from the fills")

    def _clear(self) -> None:
        self._v: Optional[float] = float(self.volume) if self.volume else None
        self._sizes: list = []
        self._mids = deque(maxlen=max(self.quote_lag, 1))
        self._prev: Optional[float] = None
        self._last: Optional[int] = None
        self._filled = 0.0
        self._buy = 0.0
        self._sell = 0.0
        self._true_buy = 0.0
        self._imbalances: deque = deque(maxlen=self.buckets)
        self._closed = 0
        self._open_mid: Optional[float] = None
        self._last_mid: Optional[float] = None
        self._changes = _Samples(BVC_BUCKETS)
        if self.log is not None:
            self.log.clear()

    @property
    def volume_in_use(self) -> Optional[float]:
        """V now: the setting, or the calibrated value; None while 200 fills have yet to set it."""
        return self._v

    def _restart_buckets(self) -> None:
        """Drop the buckets, keeping the once-a-unit mid samples: the lagged mid is the last whole unit's, which
        a new bucket does not change."""
        mids = self._mids
        self.reset(self.unit, self._t)
        self._mids = mids

    def recalibrate(self) -> None:
        """Forget the buckets and the calibrated V, as a change of engine or model should: V is set again from
        the next 200 fills (a fixed ``volume`` stays). The mid samples behind the ``mid`` rule are kept."""
        self._restart_buckets()

    def set_volume(self, v: float) -> None:
        """Use ``v`` lots a bucket instead of calibrating (it must be at least 1); buckets so far are dropped."""
        if not (math.isfinite(v) and v >= 1):
            raise ValueError("A bucket holds at least one lot")
        self._restart_buckets()
        self._v = float(v)

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        if not fills:
            return
        self._t = t
        rule = self.side
        ref = self._reference(t, quotes_before) if rule == "mid" else None
        if rule == "bvc":
            mid_after = _mid(quotes_after)
            if mid_after is not None:
                self._last_mid = mid_after
            if self._open_mid is None and self._v is not None:
                before = _mid(quotes_before)
                self._open_mid = self._last_mid if before is None else before
        else:
            mid_after = None
        prev, last = self._prev, self._last
        for fill in fills:
            p, left = fill.price, fill.size
            if rule == "true":
                sign = aggressor
            elif rule == "mid":
                sign = None if ref is None else 1 if p >= ref else -1
            elif rule == "tick":
                sign = None if prev is None else 1 if p > prev else -1 if p < prev else last
                prev = p
                last = sign
            else:
                sign = 0
            v = self._v
            if v is None:
                self._sizes.append(left)
                if len(self._sizes) >= CALIBRATION_FILLS:
                    self._v = max(1.0, BUCKET_MEDIANS * _median(self._sizes))
                    self._sizes = []
                    if rule == "bvc":
                        self._open_mid = mid_after if mid_after is not None else self._last_mid
                continue
            while left > 0:
                room = v - self._filled
                take = room if left >= room else left
                if sign == 1:
                    self._buy += take
                elif sign == -1:
                    self._sell += take
                if aggressor > 0:
                    self._true_buy += take
                left -= take
                if take == room:
                    self._close(t, mid_after)
                else:
                    self._filled += take
        if rule == "tick":
            self._prev, self._last = prev, last

    def _close(self, t: float, mid_end: Optional[float]) -> None:
        v = self._v
        true_share = self._true_buy / v
        if self.side == "bvc":
            if mid_end is None:
                mid_end = self._last_mid
            start = self._open_mid
            change = 0.0 if mid_end is None or start is None else mid_end - start
            if mid_end is not None:
                self._open_mid = mid_end
            changes = self._changes
            changes.add(change)
            _, sd = changes.mean_sd(ddof=1)
            x = change / sd if sd else 0.0
            imbalance = abs(math.erf(x / _SQRT2))
            share = 0.5 * (1.0 + math.erf(x / _SQRT2))
        else:
            decided = self._buy + self._sell
            imbalance = abs(self._buy - self._sell) / v
            share = self._buy / decided if decided else None
        self._imbalances.append(imbalance)
        self._closed += 1
        if self.log is not None:
            self.log.append((t, imbalance, share, true_share))
        self._filled = self._buy = self._sell = self._true_buy = 0.0

    def values(self, t=None, book=None) -> dict:
        full = len(self._imbalances) == self.buckets
        return {"vpin": sum(self._imbalances) / self.buckets if full else None,
                "buckets": self._closed, "volume": self._v}

    def sample(self, index, book):
        self._sample_mid(book)
        return super().sample(index, book)


class OrderToTrade(Study):
    """Order-to-trade ratio and trade-to-order volume over a trailing window, market-wide.

    The ratio is messages (limit and market arrivals, and cancels) per trade, a trade being an arriving order
    that filled, counted once however many fills it took: fills an order makes depend on the engine. The volume
    form (TTO) is the lots traded over the lots of the orders that arrived. Neither has the regulator's minus
    one: MiFID II RTS 9 (Delegated Regulation 2017/566) defines the ratio per member and instrument over a session
    as (orders / transactions) - 1 by number and (volume of orders / volume of transactions) - 1 by volume,
    counting every submission, modification and cancel as an order. A strategy's market order is an arrival and
    the cancel of its remainder, two messages for one trade. VisualHFT's OTR is a count of book changes per
    print, in which its own code never counts a print, so it is comparable to neither.

    With ``scope`` ``model`` only the model's orders count (the events with ``own`` false), because the quoting
    strategies cancel and replace at every decision and would otherwise set the ratio; ``everyone`` counts all.
    """
    key = "otr"
    label = "Order-to-trade ratio"
    headline = "otr"
    SETTINGS = (
        Setting("window", "Window (time units)", 10.0, 1.0, 100.0),
        Setting("scope", "Orders counted", "model", options=("model", "everyone")),
    )

    def _clear(self) -> None:
        self._items: deque = deque()
        self._trades = 0
        self._arrived = 0
        self._traded = 0
        self._model_only = self.scope == "model"

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        self._t = t
        if own and self._model_only:
            return
        arrived = 0 if kind == CANCEL else size
        items = self._items
        if fills:
            traded = 0
            for fill in fills:
                traded += fill.size
            self._trades += 1
            self._traded += traded
            items.append((t, 1, arrived, traded))
        else:
            items.append((t, 0, arrived, 0))
        self._arrived += arrived
        if len(items) > MAX_PENDING:
            self._drain(t)

    def _drain(self, t: float) -> None:
        items = self._items
        cutoff = t - self.window
        while items and items[0][0] <= cutoff:
            _, trades, arrived, traded = items.popleft()
            self._trades -= trades
            self._arrived -= arrived
            self._traded -= traded

    def values(self, t=None, book=None) -> dict:
        t = self._t if t is None else t
        self._drain(t)
        warm = t - self.start >= self.window
        messages = len(self._items)
        return {"otr": messages / self._trades if warm and self._trades else None,
                "tto": self._traded / self._arrived if warm and self._arrived else None,
                "messages": messages if warm else None,
                "trades": self._trades if warm else None}


# ---- Book state ----

def _spread_ticks(book) -> Optional[int]:
    spread = book.spread
    return None if spread is None else spread // book.tick_size


class BookImbalance(Study):
    """Imbalance of the visible depth over the top ``levels`` price levels of each side, (B - A) / (B + A).

    Positive means bid-heavy. ``equal`` weights every level alike (with one level it is the queue imbalance at
    the touch). ``touch`` weights a level by (1 + d / S)^-2, d its distance in ticks from its own side's best
    quote and S the spread scale, as VisualHFT does (its ImbalanceCalculator): the touch has weight 1 and the
    spread is charged to neither side. S differs from VisualHFT's in how it is estimated: it is the median of
    the last 100 once-a-unit samples of the quoted spread in ticks, once there are 20, where VisualHFT runs a
    P-squared median over every book update; until then S is the current spread (at least one tick), as in
    its warm-up. ``None`` when either side is empty. It is computed from the book it is asked about, not event
    by event.
    """
    key = "imbalance"
    label = "Book imbalance"
    headline = "imbalance"
    sees_events = False
    SETTINGS = (
        Setting("levels", "Levels a side", 10, 1, 20, integer=True),
        Setting("weighting", "Weighting", "equal", options=("equal", "touch")),
    )

    def _clear(self) -> None:
        self._spreads = _Samples(HISTORY)
        self._last: Optional[float] = None
        self._last_scale: Optional[float] = None

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        self._t = t

    def scale(self, book) -> Optional[float]:
        """The spread scale S of the touch weighting, in ticks."""
        if len(self._spreads) >= MIN_SAMPLES:
            return max(self._spreads.median(), 1.0)
        spread = _spread_ticks(book)
        return None if spread is None else max(float(spread), 1.0)

    def compute(self, book) -> Optional[float]:
        """The imbalance of this book now."""
        bids = book.depth_profile(Side.BID, self.levels)
        asks = book.depth_profile(Side.ASK, self.levels)
        if not bids or not asks:
            return None
        if self.weighting == "equal":
            b = sum(depth for _, depth in bids)
            a = sum(depth for _, depth in asks)
        else:
            scale = self.scale(book) * book.tick_size
            top = bids[0][0]
            b = sum(depth / (1.0 + (top - price) / scale) ** 2 for price, depth in bids)
            top = asks[0][0]
            a = sum(depth / (1.0 + (price - top) / scale) ** 2 for price, depth in asks)
        return (b - a) / (b + a) if b + a > 0 else None

    def values(self, t=None, book=None) -> dict:
        if book is None:
            return {"imbalance": self._last, "scale": self._last_scale}
        return {"imbalance": self.compute(book), "scale": self._scale_used(book)}

    def _scale_used(self, book) -> Optional[float]:
        return self.scale(book) if self.weighting == "touch" else None

    def sample(self, index, book):
        self.unit = index
        self._last = self.compute(book)
        self._last_scale = self._scale_used(book)
        spread = _spread_ticks(book)
        if spread is not None:
            self._spreads.add(spread)
        return self._last


class Spread(Study):
    """The quoted spread in ticks, and its z-score against the last 100 units' mean and standard deviation.

    The z-score is of the spread now against the once-a-unit samples before it (population standard deviation,
    as VisualHFT's spread shock does), ``None`` until there are 20 samples or while the spread has not varied.
    A *spread shock* is a z-score of ``shock`` or more (VisualHFT: 2). ``None`` when either side is empty. Without
    a book, ``values`` gives the last sample's spread and its z-score.
    """
    key = "spread"
    label = "Spread"
    headline = "spread"
    sees_events = False
    SETTINGS = (Setting("shock", "Spread shock at z of", 2.0, 0.5, 10.0),)

    def _clear(self) -> None:
        self._samples = _Samples(HISTORY)
        self._last: Optional[int] = None
        self._last_z: Optional[float] = None

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        self._t = t

    def _z(self, spread: Optional[int]) -> Optional[float]:
        if spread is None or len(self._samples) < MIN_SAMPLES:
            return None
        mean, sd = self._samples.mean_sd()
        return (spread - mean) / sd if sd else None

    def values(self, t=None, book=None) -> dict:
        spread, z = (self._last, self._last_z) if book is None else self._read(book)
        return {"spread": spread, "z": z, "shock": None if z is None else z >= self.shock}

    def _read(self, book) -> tuple:
        spread = _spread_ticks(book)
        return spread, self._z(spread)

    def sample(self, index, book):
        self.unit = index
        spread, self._last_z = self._read(book)
        self._last = spread
        if spread is not None:
            self._samples.add(spread)
        return spread


# ---- Resilience ----

class Resilience(Study):
    """Depth depletion and recovery of the book near the touch, and Large's restoration of the quote.

    A side's *near-touch depth* is the visible lots at the K prices from its best quote (``ticks``); its
    *baseline* is the median of its once-a-unit samples over the last 100 units, defined from 20 on. An
    *episode* starts when a take or a cancel on that side leaves the depth at the K prices from the best quote
    *before that event* below ``depleted`` times the baseline. The episode keeps those K prices as its anchor: it
    ends *recovered* when the depth at them is back at ``recovered`` times the baseline or more (checked after
    every event on that side and at each unit sample), the recovery time being the simulation time between; it
    ends *abandoned* when the side's best quote is more than K ticks from the anchor's, or the side is empty
    (counted, not timed). A limit order that improves the quote never starts an episode: measuring depth from the
    current best quote would read the new thin level as a loss, and in the zero-intelligence model that is nearly
    every start. Only one episode a side at a time, and only events that traded or lie inside the anchor's K
    prices are examined, since no other event can change the depth there.

    With the onset ``below``, the design's rule, a take or cancel need only leave the depth below the line. A
    touch that is thin because a limit order has just improved the quote then starts an episode at the next take
    or cancel inside its K prices, though the arrival never does: in 150,000 zero-intelligence events (seed 7)
    ``below`` starts 141 episodes and ``crossing`` 75, and seeds 11 and 1 give 78 against 127 and 72 against 110,
    so about half of the ``below`` starts are a thin touch that an event only took or canceled from. With
    ``crossing`` the depth must also have been at the line or above before the event, found by putting back the
    lots the event took or canceled (the refill of an iceberg makes a take look a little larger), which leaves
    the episodes an event began.

    The resilience score is T_hist / (T_hist + T_now) for the side depleted now (the lower of the two if both
    are), VisualHFT's recovery sub-score (MarketResilienceCalculator), with T_hist the median of the last 20
    recovery times and T_now the time depleted so far; 1 if no side is depleted, and ``None`` while one is and
    there is no history yet.

    After Large (2007) it also counts *quote-moving takes*, takes that leave a side's best quote worse (or the
    side empty), and for each whether a limit order puts the quote back at the pre-take price or better within
    ``restore_window`` units, and after how long: ``restored_share`` of the takes resolved so far, and the median
    of the last 20 times. A take still waiting is in neither. A book whose touch queues are deep never depletes
    (the mike_farmer model's hold 660 to 1,090 lots), and then ``started`` stays 0.
    """
    key = "resilience"
    label = "Depth depletion and recovery"
    headline = "score"
    SETTINGS = (
        Setting("ticks", "Prices from the touch (K)", 3, 1, 20, integer=True),
        Setting("depleted", "Depleted below (share of baseline)", 0.5, 0.05, 0.95),
        Setting("recovered", "Recovered at (share of baseline)", 0.9, 0.1, 1.0),
        Setting("restore_window", "Quote restored within (units)", 30.0, 1.0, 600.0),
        Setting("onset", "Episode starts when the depth is", "below", options=("below", "crossing")),
    )

    def _check(self, settings: dict) -> None:
        if settings["recovered"] <= settings["depleted"]:
            raise ValueError("Recovered at must be above Depleted below")

    def _clear(self) -> None:
        self._samples = (_Samples(HISTORY), _Samples(HISTORY))
        self._baseline: list = [None, None]
        self._episode: list = [None, None]  # (start time, anchor price) a side
        self._open = 0  # episodes now
        self._reach = self.ticks  # the K prices from the touch, in price units (the book's tick size sets it)
        self._crossing = self.onset == "crossing"
        self._pending = (deque(), deque())  # (time, price to restore) of quote-moving takes waiting
        self._recoveries: deque = deque(maxlen=RECOVERIES)
        self._restorations: deque = deque(maxlen=RECOVERIES)
        self.started = self.recovered_n = self.abandoned = 0
        self.quote_moves = self.restored = self.unrestored = 0

    def _near(self, book, s: int, price: int) -> int:
        """Visible lots at the K prices from ``price`` away from the spread's far side."""
        lots = (book._ask_depth_at if s else book._bid_depth_at).get   # the depth tables: this runs for every take
        step = book.tick_size if s else -book.tick_size
        depth = 0
        for _ in range(self.ticks):
            depth += lots(price, 0)
            price += step
        return depth

    def _maybe_start(self, t: float, s: int, best: int, after: Optional[int], book, fills=(), canceled=0) -> None:
        """Start an episode if the depth at the K prices from ``best`` is now low; ``after`` is the new best quote.

        With the onset ``crossing`` the depth must also have been at the line or above before the event: its lots
        are put back from what the event took (``fills``) or canceled (``canceled``).
        """
        base = self._baseline[s]
        if base is None:
            return
        depth = self._near(book, s, best)
        line = self.depleted * base
        if depth >= line:
            return
        if self._crossing:
            tick = book.tick_size
            low, high = (best - (self.ticks - 1) * tick, best) if s == 0 else (best, best + (self.ticks - 1) * tick)
            removed = canceled
            for fill in fills:
                if low <= fill.price <= high:
                    removed += fill.size
            if depth + removed < line:
                return
        self.started += 1
        if after is None or abs(after - best) > self._reach:
            self.abandoned += 1
        else:
            self._episode[s] = (t, best)
            self._open += 1

    def _follow(self, t: float, s: int, best: Optional[int], book, near: bool = True) -> None:
        """End the side's episode if it has recovered (``near``: its prices' depth may have changed) or is lost."""
        t0, anchor = self._episode[s]
        if near:
            base = self._baseline[s]
            if base is not None and self._near(book, s, anchor) >= self.recovered * base:
                self._recoveries.append(t - t0)
                self._episode[s] = None
                self._open -= 1
                self.recovered_n += 1
                return
        if best is None or abs(best - anchor) > self._reach:
            self._episode[s] = None
            self._open -= 1
            self.abandoned += 1

    def _restore(self, t: float, quotes_after: Optional[tuple]) -> None:
        """Expire the quote-moving takes that waited too long; with ``quotes_after``, settle those it restores."""
        limit = self.restore_window
        for s, waiting in enumerate(self._pending):
            if not waiting:
                continue
            while waiting and t - waiting[0][0] > limit:
                waiting.popleft()
                self.unrestored += 1
            if waiting and quotes_after is not None:
                best = quotes_after[2 * s]
                if best is not None and (best <= waiting[-1][1] if s else best >= waiting[-1][1]):
                    done = []
                    while waiting and (best <= waiting[-1][1] if s else best >= waiting[-1][1]):
                        done.append(waiting.pop())
                    for t0, _ in reversed(done):  # the oldest take first
                        self._restorations.append(t - t0)
                    self.restored += len(done)

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        self._t = t
        if quotes_before is not quotes_after:
            pending = self._pending
            if pending[0] or pending[1]:
                self._restore(t, quotes_after)
        if fills:
            maker = 1 if aggressor > 0 else 0
            best = quotes_before[2 * maker]
            after = quotes_after[2 * maker]
            if best is not None:
                if after is None or (after > best if maker else after < best):
                    self.quote_moves += 1
                    self._pending[maker].append((t, best))
                if self._episode[maker] is None:
                    self._maybe_start(t, maker, best, after, book, fills)
                else:
                    self._follow(t, maker, after, book)
            if kind == ARRIVAL_LIMIT:
                rests = 1 - maker
                if self._episode[rests] is not None:
                    self._follow(t, rests, quotes_after[2 * rests], book)
        elif kind == CANCEL:
            if side is _BID:
                s, best = 0, quotes_before[0]
            else:
                s, best = 1, quotes_before[2]
            if best is not None:
                episode = self._episode[s]
                if episode is None:
                    if (price - best if s else best - price) < self._reach:
                        self._maybe_start(t, s, best, quotes_after[2 * s], book, (), size)
                else:
                    near = 0 <= (price - episode[1] if s else episode[1] - price) < self._reach
                    if near or quotes_before is not quotes_after:
                        self._follow(t, s, quotes_after[2 * s], book, near)
        elif self._open and kind == ARRIVAL_LIMIT:
            s = 0 if side is _BID else 1
            episode = self._episode[s]
            if episode is not None:
                near = 0 <= (price - episode[1] if s else episode[1] - price) < self._reach
                if near or quotes_before is not quotes_after:
                    self._follow(t, s, quotes_after[2 * s], book, near)

    def sample(self, index, book):
        self.unit = index
        t = float(index)
        self._reach = self.ticks * book.tick_size
        bests = (book.best_bid, book.best_ask)
        for s in (0, 1):
            if bests[s] is not None:
                samples = self._samples[s]
                samples.add(self._near(book, s, bests[s]))
                if len(samples) >= MIN_SAMPLES:
                    self._baseline[s] = samples.median()
            if self._episode[s] is not None:
                self._follow(t, s, bests[s], book)
        self._restore(t, None)
        return self._score(t, _median(self._recoveries))

    def _score(self, t: float, history: Optional[float]) -> Optional[float]:
        """The resilience score as of ``t``: 1 with no side depleted, else the lower side's, None with no history."""
        scores = []
        depleted = False
        for episode in self._episode:
            if episode is not None:
                depleted = True
                if history is not None:
                    now = t - episode[0]
                    scores.append(1.0 if now <= 0 else history / (history + now))
        if not depleted:
            return 1.0
        return min(scores) if scores else None

    def values(self, t=None, book=None) -> dict:
        t = self._t if t is None else t
        self._restore(t, None)
        history = _median(self._recoveries)
        bid, ask = self._episode
        ended = self.recovered_n + self.abandoned
        resolved = self.restored + self.unrestored
        return {"score": self._score(t, history),
                "baseline_bid": self._baseline[0], "baseline_ask": self._baseline[1],
                "depleted_bid": None if bid is None else t - bid[0],
                "depleted_ask": None if ask is None else t - ask[0],
                "median_recovery": history,
                "abandoned_share": self.abandoned / ended if ended else None,
                "started": self.started, "recovered": self.recovered_n, "abandoned": self.abandoned,
                "quote_moves": self.quote_moves,
                "restored_share": self.restored / resolved if resolved else None,
                "median_restoration": _median(self._restorations)}


# ---- Sets of studies ----

def default_studies() -> list:
    """One of each study, with its default settings."""
    return [OFI(), CVD(), TradeDelta(), VPIN(), OrderToTrade(), BookImbalance(), Spread(), Resilience()]


class StudySet:
    """A list of studies, the events they are given, and the unit sampling.

    Only the studies that are switched on see events, the studies that only matter when a fill happened
    (``needs_fills``) only see events that traded, and those that read the book when sampled see none, so an off
    study costs nothing and a quiet event costs little.
    ``observe`` is a study's ``observe``. With ``auto_sample``, ``observe`` also takes the sample of each unit
    of simulation time the event begins, from the book it is given (after the event); a session that samples
    on its own, from the book as it stood before the event, calls ``sample_to`` itself and sets it False.
    """

    def __init__(self, studies: Optional[Sequence[Study]] = None, auto_sample: bool = True):
        self.studies = list(default_studies() if studies is None else studies)
        keys = [study.key for study in self.studies]
        if len(set(keys)) != len(keys):
            raise ValueError("Two studies with the same key")
        self.auto_sample = auto_sample
        self.unit = 0
        self._t = 0.0
        self._rebuild()

    def _rebuild(self) -> None:
        on = [study for study in self.studies if study.enabled]
        self._always = [study.observe for study in on if study.sees_events and not study.needs_fills]
        self._traded = [study.observe for study in on if study.needs_fills]
        self._sampled = on
        self._next = self.unit + 1 if self.auto_sample else math.inf

    def __getitem__(self, key: str) -> Study:
        for study in self.studies:
            if study.key == key:
                return study
        raise KeyError(key)

    def __iter__(self):
        return iter(self.studies)

    def enable(self, key: str, on: bool = True) -> None:
        """Switch a study on (afresh, from now) or off (it then sees nothing and costs nothing)."""
        study = self[key]
        if study.enabled != on:
            study.enabled = on
            if on:
                study.reset(self.unit, self._t)
            self._rebuild()

    def configure(self, key: str, **changes) -> bool:
        """Change one study's settings; True if anything changed (the study then starts afresh, from now)."""
        study = self[key]
        changed = study.configure(**changes)
        if changed:
            study.reset(self.unit, self._t)
        return changed

    def reset(self, index: int = 0, t: Optional[float] = None) -> None:
        """Start every study afresh in unit ``index`` (settings and switches stay)."""
        self.unit = index
        self._t = float(index) if t is None else float(t)
        for study in self.studies:
            study.reset(index, t)
        self._rebuild()

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book,
                own=False) -> None:
        self._t = t
        if t >= self._next:
            self.sample_to(int(t), book)
        for call in self._always:
            call(t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book, own)
        if fills:
            for call in self._traded:
                call(t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book, own)

    def sample_to(self, index: int, book, limit: int = HISTORY, rows: bool = False) -> list:
        """Sample every unit after the last one up to ``index``, at most the last ``limit``, from ``book``.

        A unit the clock jumped over is sampled with the same book, as the heatmap fills its gaps, so the studies
        see the book as it stood throughout. Returns, with ``rows``, a list of (unit, {key: main value}).
        """
        out = []
        for unit in range(max(self.unit + 1, index - limit + 1), index + 1):
            row = {study.key: study.sample(unit, book) for study in self._sampled}
            if rows:
                out.append((unit, row))
        self.unit = max(self.unit, index)
        self._next = self.unit + 1 if self.auto_sample else math.inf
        return out

    def values(self, t: Optional[float] = None, book=None) -> dict:
        """Every enabled study's ``values``, by key, as of time ``t`` (the last event if omitted)."""
        t = self._t if t is None else t
        return {study.key: study.values(t, book) for study in self.studies if study.enabled}
