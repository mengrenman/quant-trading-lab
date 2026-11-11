"""State and simulation management for the interactive LOB web application."""

from __future__ import annotations

import asyncio
import math
import secrets
import time
from array import array
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from lob.book import OrderBook
from lob.order import Order, OrderType, Fill, Side, cancel_order, limit_order, market_order
from lob.matching import MatchingEngine, PriceTimeEngine, ProRataEngine
from lob.models.informed import InformedZIModel
from app import recording as rec
from app.facts import FactsRecorder
from app.model_specs import MODEL_SPECS, SEED_RANGE
from app.fitting import FIT_OBJECTIVES, MAX_FIT_CANDIDATES
from app.strategy_specs import MAX_STRATEGIES, StrategySpec
from app.studies import StudyBoard
from app.triggers import Firing, TriggerBoard
from lob.strategies import Strategy, View
from lob.studies import ARRIVAL_LIMIT, ARRIVAL_MARKET, CANCEL, book_quotes

EngineName = Literal["price_time", "pro_rata"]
ENGINE_NAMES = ("price_time", "pro_rata")
DEFAULT_SEED = 42

# Simulation time that passes with each manual action (order or cancellation), so
# a session driven only by hand still moves forward on the chart.
MANUAL_ACTION_DT = 0.5
MAX_TRADE_LOG = 100_000  # most recent fills kept for export
MAX_EXPORT_BARS = 200_000
# How far (in ticks) a manual market order may walk the book past the best opposite price.
DEFAULT_PROTECTION_TICKS = 10
MAX_PROTECTION_TICKS = 1_000
# Trades are stored as bars one unit of simulation time wide; every chart
# interval is a whole number of units, so any interval can be rebuilt from them.
MAX_BASE_BARS = 100_000
MAX_CHART_BARS = 1_000
# A fill's markout: its edge against the mid this many events later (in events, not time, so that it
# compares like with like across models that run at different rates).
MARKOUT_EVENTS = 100
# The regime the order flow sets (LOBManager.regime): averages that weigh the last few thousand trades and fills
# most (exponentially, so they follow a change of model and fit in a recording's starting state).
REGIME_TRADES = 2_000
FLOW_FILLS = 2_000
# The order-book heatmap: one column per unit of simulation time, this many ticks either side of the mid.
HEAT_HALF = 30
HEAT_COLUMNS = 600
# A seek that moves the playback further than this many events keeps the studies off until it is this close to its
# target (the longest default study window, VPIN's 50 buckets, is 25,000 events in the zero-intelligence models).
SEEK_STUDY_LOOKBACK = 40_000
# The tape of fills the page shows (the last TAPE_FILLS, each with a number that is never reused).
TAPE_FILLS = 2_000
# The footprint: lots at each price kept for the last FOOTPRINT_UNITS bars of one unit of time (about 6 MB), and
# sent for the last FOOTPRINT_BARS bars of the chart.
FOOTPRINT_UNITS = 12_000
FOOTPRINT_BARS = 300
# The informed-traders model's hidden value, sampled once a unit for the chart's Reveal line (the last
# VALUE_SAMPLES units: 0.8 MB), as many units as there are bars kept.
VALUE_SAMPLES = MAX_BASE_BARS


@dataclass
class TradeRecord:
    timestamp: float
    price: int
    size: int
    maker_order_id: int
    taker_order_id: int
    side: str  # aggressive side: buy or sell


@dataclass
class Notice:
    """A message for the user, shown once by each open tab."""
    seq: int  # increases across the session (never reset), so a tab knows what it has shown
    text: str


@dataclass
class OwnFill:
    """A fill of one of the user's resting orders (someone else's order traded against it)."""
    order_id: int
    side: str         # the user's side: "bid" or "ask"
    price: int
    size: int
    left: int         # lots of the order still resting afterwards (shown and hidden)
    time: float

    def describe(self) -> str:
        kind = "buy" if self.side == "bid" else "sell"
        lots = "lot" if self.size == 1 else "lots"
        rest = f", {self.left:,} left" if self.left else ", now filled in full"
        return f"Your {kind} order #{self.order_id} filled {self.size:,} {lots} at {self.price}{rest}."


@dataclass
class SplitOrder:
    """A large market order sent as equal slices over simulation time (a TWAP schedule)."""
    split_id: int
    side: Side
    total: int
    slices: int
    interval: float
    protection: int
    start_time: float
    turnover0: float           # session totals at the start, for the market's VWAP over the period
    volume0: int
    sweep_filled: int          # what one market order for the whole size would have filled at the start
    sweep_avg: Optional[float]  # ... and at what average price
    sent: int = 0
    filled: int = 0
    cost: float = 0.0
    next_time: float = 0.0
    finished: Optional[str] = None  # "done" or "canceled"

    @property
    def average(self) -> Optional[float]:
        return self.cost / self.filled if self.filled else None


def sweep_estimate(book: OrderBook, side: Side, size: int, limit: int) -> tuple[int, Optional[float]]:
    """What one market order of ``size`` would fill now, walking the visible book up to ``limit``.

    Returns (lots filled, average price).
    """
    levels = book.depth_profile(Side.ASK if side == Side.BID else Side.BID, n_levels=100_000)
    filled, cost = 0, 0.0
    for price, depth in levels:
        if (side == Side.BID and price > limit) or (side == Side.ASK and price < limit) or filled >= size:
            break
        take = min(depth, size - filled)
        filled += take
        cost += take * price
    return filled, (cost / filled if filled else None)


@dataclass
class HeatColumn:
    """The book in one unit of simulation time, [index, index + 1), for the heatmap.

    The visible depth at each price near the mid as the unit began (just before the
    event that reached it), and the lots traded at each price during the unit.
    """
    index: int
    lo: int  # the lowest price: the column spans lo to lo + 2 * HEAT_HALF
    mid: Optional[float]
    depth: list[int]  # visible lots at each price from lo up, bids positive and asks negative
    hidden: dict[int, int]  # iceberg reserves by price (not part of the visible depth)
    traded: dict[int, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"t": self.index, "lo": self.lo, "mid": self.mid, "depth": self.depth,
                "hidden": [[p, n] for p, n in sorted(self.hidden.items())],
                "traded": [[p, n] for p, n in sorted(self.traded.items())]}


@dataclass
class OwnBar:
    """The user's fills in one unit of simulation time, [index, index + 1), as maker or taker."""
    index: int
    bought: int = 0
    buy_cost: float = 0.0
    sold: int = 0
    sell_cost: float = 0.0
    # The account's position and cash after the unit's last fill, for the P&L line on the chart.
    position: int = 0
    cash: float = 0.0


@dataclass
class Account:
    """A trader's position and P&L. Money is in ticks times lots: one lot gaining one tick is 1.

    Average-cost accounting: buying adds to a long position at its average price, and selling
    out of it realizes the difference (and the reverse for a short one); what is still open is
    marked to a price. Each fill is also scored against the mid just before the order that
    caused it (its edge): a resting order usually trades at a better price than the mid, an
    aggressive one at a worse.
    """
    position: int = 0  # lots, positive when long
    cash: float = 0.0  # from trading: a sale adds price times lots, a purchase subtracts it
    avg_cost: Optional[float] = None  # the open position's average price
    realized: float = 0.0
    fills: int = 0
    bought: int = 0
    sold: int = 0
    maker_lots: int = 0
    maker_edge: float = 0.0  # lots times how much better than the mid each traded
    taker_lots: int = 0
    taker_edge: float = 0.0
    markout_lots: int = 0
    markout_edge: float = 0.0  # lots times how much better than the mid MARKOUT_EVENTS events later
    # Against a strategy with no skill on the same side, at the same time (LOBManager._null_rate): the lots that had
    # one, their markout, and the null's markout for as many lots.
    null_lots: int = 0
    null_markout: float = 0.0
    null_edge: float = 0.0

    def markout(self, buying: bool, price: float, size: int, mid_later: float, null: Optional[float] = None) -> float:
        """Book a fill's markout (and, with ``null``, the null's per lot beside it); returns it per lot."""
        per_lot = mid_later - price if buying else price - mid_later
        self.markout_lots += size
        self.markout_edge += per_lot * size
        if null is not None:
            self.null_lots += size
            self.null_markout += per_lot * size
            self.null_edge += null * size
        return per_lot

    def book(self, buying: bool, price: float, size: int, maker: bool, mid_before: Optional[float]) -> None:
        sign = 1 if buying else -1
        self.cash -= sign * price * size
        self.fills += 1
        if buying:
            self.bought += size
        else:
            self.sold += size
        pos = self.position
        if pos == 0 or (pos > 0) == buying:  # opening or adding to a position
            self.avg_cost = price if pos == 0 else (self.avg_cost * abs(pos) + price * size) / (abs(pos) + size)
        else:  # reducing, closing or reversing it
            closed = min(abs(pos), size)
            self.realized += (price - self.avg_cost) * closed * (1 if pos > 0 else -1)
            if size > closed:
                self.avg_cost = price  # reversed: the rest opens a new position here
            elif closed == abs(pos):
                self.avg_cost = None
        self.position = pos + sign * size
        if mid_before is not None:
            edge = (mid_before - price if buying else price - mid_before) * size
            if maker:
                self.maker_lots += size
                self.maker_edge += edge
            else:
                self.taker_lots += size
                self.taker_edge += edge

    def summary(self, mark: float) -> dict[str, Any]:
        unrealized = (mark - self.avg_cost) * self.position if self.position else 0.0
        return {
            "position": self.position, "avg_cost": self.avg_cost, "mark": mark,
            "realized": self.realized, "unrealized": unrealized, "total": self.realized + unrealized,
            "fills": self.fills, "bought": self.bought, "sold": self.sold,
            "maker_lots": self.maker_lots, "taker_lots": self.taker_lots,
            "maker_edge": self.maker_edge / self.maker_lots if self.maker_lots else None,
            "taker_edge": self.taker_edge / self.taker_lots if self.taker_lots else None,
            "edge": (self.maker_edge + self.taker_edge) / (self.maker_lots + self.taker_lots)
                    if self.maker_lots + self.taker_lots else None,
            "markout": self.markout_edge / self.markout_lots if self.markout_lots else None,
            "markout_lots": self.markout_lots,
            # Its markout a lot less the null's, over the lots that had one (LOBManager._null_rate).
            "vs_null": (self.null_markout - self.null_edge) / self.null_lots if self.null_lots else None,
            "null_lots": self.null_lots,
        }


STRATEGY_SAMPLES = 300  # P&L samples kept per strategy (one per unit of simulation time), for its sparkline


@dataclass
class RunningStrategy:
    """A strategy of the user's in the market (lob/strategies.py): its account, its orders, its record."""
    id: int
    spec: StrategySpec
    values: dict[str, Any]  # its settings, to build it afresh on a reset
    strategy: Strategy
    number: int  # its number among the strategies of its kind on the board
    name: str  # e.g. "Momentum 2"
    account: Account = field(default_factory=Account)
    order_ids: set[int] = field(default_factory=set)
    next_time: float = 0.0
    running: bool = True
    signal: Optional[float] = None
    target: Optional[int] = None  # the position it last wanted (signal strategies)
    quotes: dict[str, tuple[int, int]] = field(default_factory=dict)  # side -> (price, lots) resting now
    pnl: deque = field(default_factory=lambda: deque(maxlen=STRATEGY_SAMPLES))  # (time unit, total P&L)
    replayed: bool = False  # follows a recording's tape (in playback) instead of deciding for itself
    fit: Optional[dict[str, Any]] = None  # how it was fitted, if it was (LOBManager.fit_strategy)


@dataclass
class Fit:
    """A strategy fitted over a training period: every candidate's result there, best first."""
    id: int
    kind: str  # the strategy's spec key
    base: dict[str, Any]  # the settings not searched
    grid: dict[str, list[Any]]  # the values tried, by setting
    source: str  # the training period, in words
    events: int  # orders and cancellations in it
    span: tuple[float, float]  # its simulation times
    objective: str  # "pnl" or "sharpe"
    candidates: list[dict[str, Any]]  # {"values", "score", "account", "lots"}, best first
    valid: Optional[int] = None  # how many valid combinations the grid has (None: too many to count)
    size: int = 0  # how many combinations it has, valid or not
    seed: int = 1  # what a sample of them was drawn from


MAX_FIT_WORK = 1_500_000  # candidates times tape events in one fit (under 10 s): fewer candidates over a long period


@dataclass
class Playback:
    """A recording being played back: where on its tape, and what its orders and strategies are now."""
    recording: rec.Recording
    cursor: int = 0  # the next event on the tape
    ids: dict[int, int] = field(default_factory=dict)  # a recorded order's id -> its replay's
    strategies: dict[int, RunningStrategy] = field(default_factory=dict)  # a recorded strategy's id -> its replay
    model: str = ""  # the model the recording was made with, at the cursor
    ended: bool = False


def _risk_adjusted(samples: Iterable[tuple[int, float]]) -> float:
    """The mean change of a P&L per unit of time over the standard deviation of the changes (0 if it never moves)."""
    values = [v for _, v in samples]
    changes = [b - a for a, b in zip(values[:-1], values[1:], strict=True)]
    if len(changes) < 2:
        return 0.0
    mean = sum(changes) / len(changes)
    std = math.sqrt(sum((c - mean) ** 2 for c in changes) / len(changes))
    return mean / std if std > 0 else 0.0


def sparkline(samples: Iterable[tuple[int, float]], width: float = 100, height: float = 22,
              points: int = 60) -> Optional[dict[str, Any]]:
    """An SVG path through the last P&L samples (at most ``points`` of them), and where zero is."""
    values = [v for _, v in samples]
    if len(values) < 2:
        return None
    step = max(1, math.ceil(len(values) / points))
    values = values[::-step][::-1]  # every step-th, ending at the latest
    lo, hi = min(values + [0.0]), max(values + [0.0])
    if hi == lo:  # all zero: a flat line through the middle
        lo, hi = -1.0, 1.0
    span = hi - lo
    y = lambda v: height - 1 - (v - lo) / span * (height - 2)  # noqa: E731
    x = lambda i: i * width / (len(values) - 1)  # noqa: E731
    path = "M" + " L".join(f"{x(i):.1f} {y(v):.1f}" for i, v in enumerate(values))
    return {"path": path, "zero": round(y(0.0), 1), "width": width, "height": height}


@dataclass
class BaseBar:
    """Trades in one unit of simulation time, [index, index + 1)."""
    index: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    buy_volume: int
    # Session totals up to this bar's last trade, for the VWAP (absolute, so they survive truncation).
    cum_volume: int = 0
    cum_turnover: float = 0.0
    # Lots at each price, {price: [lots sold into the bid, lots bought from the ask]}, by the aggressor's side: for
    # the footprint. None once the bar is more than FOOTPRINT_UNITS bars old.
    cells: Optional[dict[int, list[int]]] = None


class LOBManager:
    """Central manager coordinating the LOB, matching engine, and simulation runner.

    There is a single clock, ``sim_time``. Model events advance it by the
    model's own Poisson inter-arrival times (models continue from it when they
    take over a book), and each manual action advances it by
    ``MANUAL_ACTION_DT``. Chart bars are built from executed trades on this
    clock.
    """

    def __init__(self):
        self.engine_type: EngineName = "price_time"
        self.engine: MatchingEngine = PriceTimeEngine()
        self.model_type: str = "smith_zi"
        # Parameters per model, kept when switching back and forth.
        self.model_params: dict[str, dict[str, float | int]] = {
            key: spec.defaults() for key, spec in MODEL_SPECS.items()}
        self.seed: int = DEFAULT_SEED
        self.speed: int = 50  # events per second (1 to 2000)
        self.bar_interval: int = 2  # bar duration in units of simulation time
        self.is_running: bool = False
        # Incremented whenever existing chart bars change (reset, new interval),
        # so every open chart knows to reload.
        self.chart_version: int = 0
        # Incremented whenever a control shown on the page changes (model,
        # parameters, seed, engine, speed), so every open tab can update.
        self.config_version: int = 0
        # Public identifier of this simulation (unlike the secret session ID), so a
        # page can tell when its event stream now belongs to a different simulation.
        self.instance_id = secrets.token_hex(4)

        self._subscribers: set[asyncio.Queue] = set()
        self._bg_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self.pairs: Optional[Any] = None  # the Pairs view's two books (app/pairs.py), made when first needed
        self.statarb: Optional[Any] = None  # the Stat-arb view's many books (app/statarb.py), the same way

        # The stylized-facts sample of the current model and settings (app/facts.py), a cached
        # rendering of it, and the progress of a fast collection: (events done, events asked).
        self.facts = FactsRecorder()
        self.facts_cache: Optional[tuple[Any, str]] = None
        self.collecting: Optional[tuple[int, int]] = None
        self.strategies: list[RunningStrategy] = []  # the user's strategies, kept across resets
        self._strategy_seq = 0
        # Recording and playback (app/recording.py). Recordings are kept across resets.
        self.recorder: Optional[rec.Recorder] = None
        self.recordings: list[rec.Recording] = []
        self._recording_seq = 0
        self.recordings_version = 0  # incremented when the list, the recording or the playback changes
        self.playback: Optional[Playback] = None
        self.seeking: Optional[tuple[int, int]] = None  # (events done, events to go) while a playback seeks
        self.fitting: Optional[tuple[int, int]] = None  # (candidates done, candidates) while a strategy is fitted
        self.last_fit: Optional[Fit] = None
        self._fit_seq = 0
        self._quiet = False  # no notices (while seeking)
        self.notice_seq = 0  # never reset, so a tab can tell which notices it has shown
        self.notices: deque[Notice] = deque(maxlen=20)
        self.own_trade_seq = 0  # counts the user's fills (never reset), so a tab knows when its markers are stale
        self.heat_version = 0  # incremented when the heatmap starts over (reset, after a collection)
        self.heat_changes = 0  # incremented with each new column and each trade in the latest one
        self._split_seq = 0
        # The microstructure studies (app/studies.py). A session's manager is given a board (app/sessions.py); the
        # scratch managers of a fit and a training tape have none, and pay nothing for it. The tape of fills has a
        # number that is never reused. The version changes when what a page holds of either is out of date.
        self.studies: Optional[StudyBoard] = None
        self.studies_version = 0
        self.fill_seq = 0
        self.footprint_on = False  # the chart carries the footprint's cells (a session setting, as the interval is)
        # The informed-traders model's ground truth (see informed_status). Reveal draws its hidden value on the chart:
        # a session flag, not a Param, since a Param change rebuilds the model and reseeds it. The value is sampled
        # once a unit (an array of floats, the last unit sampled at the end) while the model and a live run have it.
        self.reveal_on = False
        self._informed_model = False  # whether the model has informed traders and a hidden value
        self._value_samples = array("d")
        self._value_last = -1
        self._informed_fill_lots = 0  # lots of the informed traders' orders that filled, and what makers lost on them
        self._informed_fill_loss = 0.0  # against the hidden value: lots times ticks
        # Triggers (app/triggers.py): rules on the studies' values, checked by _after_event. _halt ends the batch of
        # steps a rule paused in; _acting is set while a rule's actions run, which are manual actions that would
        # otherwise check the rules again.
        self.triggers: Optional[TriggerBoard] = None
        self._halt = False
        self._acting = False

        self.reset(mid=100)

    def reset(self, mid: int = 100) -> None:
        """Start a fresh session: new book, model, clock, trade history and chart.

        The random generator restarts from ``seed``, so the same seed and
        parameters reproduce a run exactly. A recording in progress is saved; a
        playback ends (the endpoint rewinds one instead: ``rewind_playback``).
        """
        if self.playback is not None:
            self.playback = None
            self.recordings_version += 1
        if self.recorder is not None:
            self._finish_recording("the book was reset")
        self._clear(time=0.0, event_count=0, reference=float(mid))
        self._reset_model(mid)
        self.seed_liquidity(mid=mid)
        self.facts.reset()
        self.chart_version += 1
        self._restart_heatmap()
        self._restart_strategies()
        self._resync_studies()

    def _clear(self, time: float, event_count: int, reference: float) -> None:
        """An empty book, the clock at ``time``, and no trades, orders or fills yet (to reset, or start a playback)."""
        self.rng = np.random.default_rng(self.seed)
        self.book = OrderBook(tick_size=1, lot_size=1)
        self.sim_time: float = time
        self.event_count: int = event_count
        self.traded_volume: int = 0
        self.recent_trades: deque[TradeRecord] = deque(maxlen=40)
        self.own_fills: deque[OwnFill] = deque(maxlen=20)  # fills of the user's resting orders
        self.split_orders: list[SplitOrder] = []          # working
        self.split_history: deque[str] = deque(maxlen=3)  # summaries of finished ones
        # Every fill for export (the most recent MAX_TRADE_LOG), in typed arrays.
        self._trade_log: dict[str, array] = {
            "time": array("d"), "price": array("q"), "size": array("q"), "aggressor": array("b"),
            "maker_order_id": array("q"), "taker_order_id": array("q"), "yours": array("b"),
        }
        self.manual_order_ids: set[int] = set()
        self._base_bars: deque[BaseBar] = deque(maxlen=MAX_BASE_BARS)
        self._own_bars: deque[OwnBar] = deque(maxlen=MAX_BASE_BARS)  # the user's fills, for the chart's markers
        self.account = Account()  # the user's position and P&L
        self._strategy_orders: dict[int, RunningStrategy] = {}  # each strategy order's owner
        self._pnl_index = -1  # the last unit of time the strategies' P&L (and the user's) was sampled at
        self._own_pnl: deque[tuple[int, float]] = deque(maxlen=STRATEGY_SAMPLES)
        self._markouts: deque[tuple[int, Account, bool, float, int, bool]] = deque()  # (due event, account, ...)
        # The market's fills by the model's own orders, for what the flow pays: (due event, maker buying, price,
        # lots, maker the model's, taker the model's); and what they earned, resting and taking ([lots x ticks, lots]).
        self._market_markouts: deque[tuple[int, bool, int, int, bool, bool]] = deque()
        # "join": what the quote joiners on the board earn resting, the null for resting fills.
        self._flow: dict[str, list[float]] = {"make": [0.0, 0.0], "take": [0.0, 0.0], "join": [0.0, 0.0]}
        # The spread just before each trade, and the mid's change from one trade to the next (weighted averages).
        self._regime: dict[str, Any] = {"trades": 0, "w": 0.0, "spread": 0.0, "dw": 0.0, "d2": 0.0, "last_mid": None}
        self._cum_volume = 0  # every lot traded this session, for the VWAP
        self._cum_turnover = 0.0
        self._bars_truncated = False
        # Price used for bars before the first trade.
        self._reference_price = reference
        self.tape_ring: deque[tuple[int, float, int, int, int]] = deque(maxlen=TAPE_FILLS)  # see _record_trade
        self._value_samples = array("d")  # the hidden value's samples start over with the clock (see _restart_values)
        self._value_last = math.floor(time) - 1
        if self.studies is not None:
            self.studies.reset(math.floor(time), time)
        if self.triggers is not None:
            self.triggers.clear_runtime()  # what the rules saw is of the market that ended; their definitions stay
        self._bump_studies()

    def _reset_model(self, mid: int = 100) -> None:
        spec = MODEL_SPECS[self.model_type]
        self.model = spec.build(self.model_params[spec.key], mid) if spec.factory else None
        self._restart_values()

    def _restart_values(self) -> None:
        """A model was built: its hidden value (if it has one) starts at the mid, so its samples and the ground truth
        counted from its orders start over, and a chart that draws them reloads."""
        self._informed_model = isinstance(self.model, InformedZIModel)
        self._informed_fill_lots, self._informed_fill_loss = 0, 0.0
        self._value_samples = array("d")
        first = math.floor(self.sim_time)
        self._value_last = first - 1
        if self._informed_model:
            self._value_samples.append(self.model.value())
            self._value_last = first
        if self.reveal_on:
            self.chart_version += 1

    def seed_liquidity(self, mid: int = 100, half_spread: int = 1, levels: int = 16,
                       volume_per_level: int = 25) -> None:
        """Seed the order book with deep symmetric bids and asks around mid."""
        for i in range(levels):
            bid_p = mid - half_spread - i
            ask_p = mid + half_spread + i
            vol = volume_per_level + (i * 5)
            if bid_p > 0:
                self.book.add_order(limit_order(bid_p, vol, self.sim_time, Side.BID))
            self.book.add_order(limit_order(ask_p, vol, self.sim_time, Side.ASK))

    # ---- The studies, the footprint and the tape ----

    def attach_studies(self, board: StudyBoard) -> None:
        """Give the session a board of studies (its manager's own; the scratch managers of a fit are left without)."""
        self.studies = board
        board.reset(math.floor(self.sim_time), self.sim_time)
        board.version = self.studies_version
        self._resync_studies()

    def _bump_studies(self) -> None:
        """What a page holds of the studies and the tape is out of date: its samples, and its buffer of fills."""
        self.studies_version += 1
        if self.studies is not None:
            self.studies.version = self.studies_version

    def _resync_studies(self) -> None:
        """Measure the next event from the book as it stands.

        The studies take each event's quotes before from the quotes after the last event they saw, which is exact
        while every change of the book goes through ``_process`` or ``_cancel_resting`` and the board is on. It is
        needed after what changes the book outside them (a reset, the start of a playback, which add orders
        directly) and after a time the board was off (a collection, a seek). A new engine or model, a change of
        parameters and ``exit_playback`` change no book, and need none.
        """
        if self.studies is not None:
            self.studies.prev = book_quotes(self.book)

    def _restart_studies(self) -> None:
        """The studies start over from now, after a time they were off (a collection, a seek), and pages reload."""
        if self.studies is not None:
            self.studies.restart(math.floor(self.sim_time), self.sim_time)
            self._resync_studies()
        if self.triggers is not None:
            self.triggers.clear_runtime()  # the values they remembered are of before the gap
        self._bump_studies()

    def _recalibrate_studies(self) -> None:
        """A new engine or model sizes the fills differently: VPIN sets the lots of its bucket again."""
        if self.studies is not None:
            self.studies.recalibrate()

    def _study_failed(self, exc: Exception) -> None:
        """A study raised: the board stays as it was, takes no more events, and says so once; the run goes on."""
        self.studies.fail(exc)
        self._notify(f"The studies switched themselves off after an error ({exc!r}). The simulation goes on.")

    def _observe(self, kind: int, side: Side, price: int, size: int, fills: Iterable[Fill], aggressor: int,
                 own: bool) -> None:
        """Give the studies an event that has changed the book (see ``lob.studies`` for the arguments).

        The quotes before are the board's, kept from the last event. When nothing traded and the order is strictly
        behind its side's best quote (an arrival that rests there, or a cancel of such an order) the touch is as it
        was, and the quotes after are the same object, which saves looking at the book for most events.
        """
        studies = self.studies
        try:
            before = studies.prev
            best = before[0] if side is Side.BID else before[2]
            if not fills and best is not None and (price < best if side is Side.BID else price > best):
                after = before
            else:
                after = studies.prev = book_quotes(self.book)
            studies.observe(self.sim_time, kind, side, price, size, fills, aggressor, before, after, self.book, own)
        except Exception as exc:
            self._study_failed(exc)

    def configure_study(self, key: str, values: dict[str, Any], enabled: Optional[bool] = None) -> bool:
        """Change a study's settings (already parsed against ``STUDY_SPECS``) or switch it on or off.

        Returns whether anything changed; the study starts afresh and pages reload. Raises ValueError for settings
        that contradict each other, and LookupError when this session has no studies or none of that name.
        """
        if self.studies is None:
            raise LookupError("this simulation has no studies")
        try:
            changed = self.studies.configure(key, values, enabled)
        except KeyError:
            raise LookupError(f"no study {key!r}") from None
        if changed:
            self._bump_studies()
        return changed

    def studies_snapshot(self, since: Optional[int] = None) -> dict[str, Any]:
        """The studies as a page shows them: copies of plain values (StudyBoard.snapshot); a session without a board
        gives the same shape, empty. ``since`` is the last unit the page has samples of."""
        studies = self.studies
        if studies is None:
            return {"version": self.studies_version, "failed": None, "suspended": False, "settings": {},
                    "enabled": {}, "values": {}, "keys": [], "unit": None, "samples": []}
        try:
            return studies.snapshot(since, self.sim_time, self.book)
        except Exception as exc:
            self._study_failed(exc)
            return studies.snapshot(since, self.sim_time, self.book)

    def tape_since(self, since: Optional[int] = None) -> dict[str, Any]:
        """The fills after number ``since`` from the tape of the last TAPE_FILLS, with how to use them.

        Each fill is [number, simulation time, price, size, side], side 1 when the buyer was the aggressor and -1
        when the seller was. ``reload`` says the fills replace what the page holds instead of adding to it: it
        had none, or fell behind the ring, or is ahead of it. ``version`` is ``studies_version``: a page that
        holds fills of another version drops them.
        """
        ring = self.tape_ring
        last = self.fill_seq
        first = ring[0][0] if ring else last + 1
        reload = since is None or since < first - 1 or since > last
        if reload:
            fills = list(ring)
        else:
            fills = []
            for row in reversed(ring):
                if row[0] <= since:
                    break
                fills.append(row)
            fills.reverse()
        return {"version": self.studies_version, "seq": last, "reload": reload, "fills": fills}

    def set_footprint(self, on: bool) -> None:
        """Carry the footprint's cells in the chart's candles (or stop); charts reload."""
        if on != self.footprint_on:
            self.footprint_on = on
            self.chart_version += 1

    # ---- The informed traders' hidden value and ground truth ----

    @property
    def reveal_active(self) -> bool:
        """Whether the chart carries the hidden value: asked for, the model has one, and the run is live (a recording
        holds orders, not the value, so a playback has none to draw)."""
        return self.reveal_on and self._informed_model and self.playback is None

    def set_reveal(self, on: bool) -> None:
        """Carry the hidden value in the chart's candles (or stop); charts reload.

        Raises ValueError to turn it on when the model has no hidden value.
        """
        if on and not self._informed_model:
            raise ValueError(f"only the {MODEL_SPECS['informed_zi'].label} model has a hidden value to reveal")
        if on != self.reveal_on:
            self.reveal_on = on
            self.chart_version += 1

    def informed_status(self) -> dict[str, Any]:
        """The informed traders' ground truth and the Reveal switch, for the page (plain values; ``line`` is its text).

        ``share`` is the realized share of the model's market-order lots that the informed traders sent (the model's
        own counters), and ``loss`` what makers lost on their filled lots, in ticks a lot against the hidden value (V
        less the price for a buy, the price less V for a sell). A model without informed traders says none. A
        playback has neither: the tape holds orders and who they came from (the model, you, a strategy), not whether
        the model's were informed nor the value. ``reveal`` says whether the switch is on, whether it can be used
        (``disabled``), whether the chart is drawing the value (``active``) and why not (``note``).
        """
        model = self.model
        playback = self.playback
        reveal = {"on": self.reveal_on, "disabled": not self._informed_model, "active": self.reveal_active,
                  "note": "not recorded" if playback is not None else None}
        if not self._informed_model:
            reveal["note"] = reveal["note"] or f"needs the {MODEL_SPECS['informed_zi'].label} model"
        status: dict[str, Any] = {"model": self._informed_model, "playback": playback is not None, "reveal": reveal,
                                  "share": None, "informed_lots": 0, "taker_lots": 0, "loss": None, "filled_lots": 0}
        if playback is not None:
            recorded = playback.model == "informed_zi"
            status["line"] = ("Informed traders: not recorded (the tape holds orders, not who sent them)"
                              if recorded else "Informed traders: none in the recorded model")
        elif not self._informed_model:
            status["line"] = "Informed traders: none in this model"
        else:
            status["informed_lots"], status["taker_lots"] = model.informed_lots, model.taker_lots
            status["share"] = model.informed_share
            status["filled_lots"] = self._informed_fill_lots
            if self._informed_fill_lots:
                status["loss"] = self._informed_fill_loss / self._informed_fill_lots
            if status["share"] is None:
                status["line"] = "Informed traders: no market orders yet"
            else:
                status["line"] = (f"Informed traders: {100 * status['share']:.1f}% of the model's taker lots "
                                  f"({model.informed_lots:,} of {model.taker_lots:,})")
                if status["loss"] is not None:
                    status["line"] += (f"; makers lose {status['loss']:.2f} ticks a lot to them, "
                                       "against the hidden value")
        return status

    # ---- Triggers (app/triggers.py) ----

    def attach_triggers(self, board: TriggerBoard) -> None:
        """Give the session a board of trigger rules (its manager's own; the scratch managers of a fit have none)."""
        self.triggers = board
        board.clear_runtime()

    def _read_study(self, key: str, book: bool) -> Optional[dict[str, Any]]:
        study = self.studies.set[key]
        return study.values(self.sim_time, self.book if book else None) if study.enabled else None

    def _read_flow(self, key: str) -> Optional[dict[str, Any]]:
        """A study's values as of now, from the events it has seen."""
        return self._read_study(key, False)

    def _read_book(self, key: str) -> Optional[dict[str, Any]]:
        """A study's values as of now, from the book as it stands (the studies that read it)."""
        return self._read_study(key, True)

    def _after_event(self) -> None:
        """Check the trigger rules, and run those that fire.

        Called after each step of the model (an idle one too, so that the studies' windows drain), after each book
        event of a playback, and after each action of the user's: where split orders and strategies already run, and
        never inside ``_process``, since an action is itself an order. It does nothing without a board of rules and
        one of studies, while the studies are off, during a collection or a seek (a rule that acted then would change
        what the seek replays), and while a rule's own actions run (they are manual actions, which come back here).
        """
        board = self.triggers
        if board is None or self._acting or not (board.armed or board.pending):
            return
        studies = self.studies
        if studies is None or studies.suspended or self.collecting is not None or self._quiet:
            return
        if board.armed:
            try:
                firings = board.evaluate(self.sim_time, self._read_flow, self._read_book)
            except Exception as exc:  # a study that cannot be read: the same as one that cannot take an event
                self._study_failed(exc)
                return
            if firings:
                self._acting = True
                try:
                    for firing in firings:
                        self._fire(firing)
                finally:
                    self._acting = False
        for text in board.flush():  # firings a notice held back, when the rule has been quiet for a second
            self._notify(text)

    def _fire(self, firing: Firing) -> None:
        """Run what a rule that fired does: cancel and flatten (your orders; not in a playback, whose tape holds the
        orders of the session it was recorded from), the pause (at once: the batch of steps it came in stops), and a
        notice that says what was done, shown at most once a second for the rule (a pause is always said)."""
        rule = firing.rule
        live = self.playback is None
        done: list[str] = []
        if "cancel" in rule.actions:
            if live:
                n = self.cancel_all_orders()
                done.append(f"canceled {n} order{'s' if n != 1 else ''}")
            else:
                done.append("cancel is off in a playback")
        if "flatten" in rule.actions:
            done.append(self._flatten() if live else "flatten is off in a playback")
        if "pause" in rule.actions:
            self.is_running = False  # the run loop ends after this batch; _halt ends the batch itself
            self._halt = True
            done.append("paused")
        if "notice" in rule.actions:
            text = f"Trigger {rule.name!r} at time {firing.t:.1f}: {firing.describe()}."
            if done:
                text += f" {'; '.join(done).capitalize()}."
            shown = self.triggers.notice(rule, text, urgent="pause" in rule.actions)
            if shown is not None:
                self._notify(shown)

    def _flatten(self) -> str:
        """Close the user's position with a protected market order, canceling at once what rests; says how it went."""
        position = self.account.position
        if position == 0:
            return "already flat"
        side = Side.ASK if position > 0 else Side.BID
        size = abs(position)
        order, fills = self.submit_market_order(size, side)
        if order is None:  # nothing on the far side: no price to start from, nothing was placed
            return f"nothing to flatten against: the {'bid' if side == Side.ASK else 'ask'} side is empty"
        self._withdraw(order.order_id)  # what did not fill within the protection must not stand as an order of yours
        filled = sum(f.size for f in fills)
        return f"flatten {'sold' if side == Side.ASK else 'bought'} {filled:,} of {size:,} lots"

    def triggers_status(self) -> dict[str, Any]:
        """The rules as a page shows them (TriggerBoard.status), with what the session says about running them."""
        board, studies = self.triggers, self.studies
        off = [study.key for study in studies.set if not study.enabled] if studies is not None else []
        status = board.status(off) if board is not None else {"rules": [], "marks": [], "limit": 0}
        return status | {"needs_model": self.model_type == "manual", "playback": self.playback is not None,
                         "seeking": self.seeking is not None, "studies_off": studies is None or studies.suspended}

    # ---- Configuration ----

    def set_engine(self, engine_name: EngineName) -> None:
        if engine_name not in ENGINE_NAMES:
            raise ValueError(f"Unknown engine {engine_name!r}")
        self.engine_type = engine_name
        self.engine = PriceTimeEngine() if engine_name == "price_time" else ProRataEngine()
        self.facts.reset()
        self._recalibrate_studies()
        self.config_version += 1
        if self.recorder is not None:
            self.recorder.note(rec.ENGINE, self.sim_time, extra={"engine": engine_name})

    def toggle_engine(self) -> None:
        self.set_engine("pro_rata" if self.engine_type == "price_time" else "price_time")

    def set_model(self, model_name: str) -> None:
        """Switch models; the new model takes over the current book and clock."""
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model {model_name!r}")
        self.model_type = model_name
        self._reset_model(int(self.book.mid_price or self._reference_price))
        if self.reveal_on and not self._informed_model:  # there is no hidden value to reveal any more
            self.reveal_on = False
        self.facts.reset()
        self._recalibrate_studies()
        self.config_version += 1
        self._note_model()

    def configure(self, model_name: str, raw_params: dict[str, Any], seed: Any) -> dict[str, str]:
        """Validate and apply a model's parameters and the random seed.

        Returns error messages by field; nothing is applied unless every field
        is valid. A valid change rebuilds the model at once, continuing from
        the current book and clock, and restarts the random generator from
        the seed.
        """
        spec = MODEL_SPECS.get(model_name)
        if spec is None:
            return {"model": f"Unknown model {model_name!r}"}
        values, errors = {}, {}
        for param in spec.params:
            try:
                values[param.name] = param.parse(raw_params.get(param.name))
            except ValueError as exc:
                errors[param.name] = str(exc)
        try:
            new_seed = int(seed)
            if not SEED_RANGE[0] <= new_seed <= SEED_RANGE[1]:
                raise ValueError
        except (TypeError, ValueError):
            errors["seed"] = f"Seed: enter a whole number from {SEED_RANGE[0]} to {SEED_RANGE[1]}"
        if errors:
            return errors
        self.model_params[spec.key] = values
        self.seed = new_seed
        self.rng = np.random.default_rng(new_seed)
        if spec.key == self.model_type:
            self._reset_model(int(self.book.mid_price or self._reference_price))
            self.facts.reset()
            self._recalibrate_studies()
            self._note_model()
        self.config_version += 1
        return {}

    def _note_model(self) -> None:
        if self.recorder is not None:
            self.recorder.note(rec.MODEL_CHANGE, self.sim_time, extra={
                "model": self.model_type, "params": dict(self.model_params[self.model_type])})

    def set_speed(self, speed: int) -> None:
        self.speed = max(1, min(2000, int(speed)))
        self.config_version += 1

    def set_bar_interval(self, interval: int) -> None:
        self.bar_interval = max(1, int(interval))
        self.chart_version += 1

    # ---- Order flow ----

    def submit_limit_order(self, price: int, size: int, side: Side,
                           display_size: Optional[int] = None) -> tuple[Order, list[Fill]]:
        """Submit a manual limit order (an iceberg with ``display_size``); returns it and its fills on arrival."""
        self._advance(self.sim_time + MANUAL_ACTION_DT)
        result = self._enter_limit(price, size, side, display_size)
        self._after_event()
        return result

    def _enter_limit(self, price: int, size: int, side: Side,
                     display_size: Optional[int] = None) -> tuple[Order, list[Fill]]:
        """Enter one of the user's limit orders at the current time (the caller moves the clock)."""
        order = limit_order(price=price, size=size, timestamp=self.sim_time, side=side, display_size=display_size)
        self.manual_order_ids.add(order.order_id)
        fills = self._process(order)
        if self.book.get_order(order.order_id) is None:  # filled on arrival
            self.manual_order_ids.discard(order.order_id)
        return order, fills

    def _withdraw(self, order_id: int) -> Optional[Order]:
        """Take one of the user's orders out of the book (the caller moves the clock)."""
        self.manual_order_ids.discard(order_id)
        return self._cancel_resting(order_id, rec.FROM_YOU)

    def _cancel_resting(self, order_id: int, origin: int) -> Optional[Order]:
        """Cancel a resting order of the user's or a strategy's (``origin``) as one event; None if it does not rest."""
        res = self.book.cancel_order(order_id)
        if res is not None:
            if self.recorder is not None:
                self.recorder.cancel(self.sim_time, order_id, origin)
            self.event_count += 1
            self.facts.observe(OrderType.CANCEL, res.side, [], self.book)
            studies = self.studies
            if studies is not None and self.collecting is None and not studies.suspended:
                # Always someone else's order: the model's cancellations go through _process.
                self._observe(CANCEL, res.side, res.price, res.size, (), 0, True)
        return res

    def _origin(self, order_id: int) -> int:
        """Whose an order is, as a recording says: the user's, a strategy's (its id) or the model's."""
        if order_id in self.manual_order_ids:
            return rec.FROM_YOU
        owner = self._strategy_orders.get(order_id)
        return owner.id if owner is not None else rec.FROM_MODEL

    def move_orders(self, side: Side, from_price: int, to_price: int) -> tuple[list[Order], list[Order], list[Fill]]:
        """Move the user's resting orders at one price to another, as one manual action.

        Each order is canceled and entered again at ``to_price`` with what it had left
        (shown and hidden) and the same iceberg peak, in its old queue order. At the new
        price it joins the back of the queue, as when an order's price is changed on an
        exchange, and a price across the spread trades on arrival. Returns the old orders
        (empty when there is nothing to move), the new ones and the new ones' fills.
        """
        old = [o for o in self.book.orders_at(from_price, side) if o.order_id in self.manual_order_ids]
        if not old or from_price == to_price:
            return [], [], []
        self._advance(self.sim_time + MANUAL_ACTION_DT)
        new: list[Order] = []
        fills: list[Fill] = []
        for order in old:
            size, peak = order.size + order.hidden, order.display_size
            self._withdraw(order.order_id)
            replacement, arrival_fills = self._enter_limit(to_price, size, side, display_size=peak)
            new.append(replacement)
            fills.extend(arrival_fills)
        self._after_event()
        return old, new, fills

    def cancel_all_orders(self, side: Optional[Side] = None, price: Optional[int] = None) -> int:
        """Cancel the user's resting orders (all, or those on one side or at one price) as one manual action.

        Returns how many were canceled.
        """
        targets = [o for o in (self.book.get_order(oid) for oid in sorted(self.manual_order_ids))
                   if o is not None and (side is None or o.side == side) and (price is None or o.price == price)]
        if targets:
            self._advance(self.sim_time + MANUAL_ACTION_DT)
            for order in targets:
                self._withdraw(order.order_id)
            self._after_event()
        return len(targets)

    def submit_market_order(self, size: int, side: Side, protection: int = DEFAULT_PROTECTION_TICKS
                            ) -> tuple[Optional[Order], list[Fill]]:
        """Submit a manual market order with price protection, as CME Globex does.

        The order may walk the book at most ``protection`` ticks past the best opposite
        price on arrival: it is entered as a limit order at that price, so whatever is
        not filled rests in the book there as the user's order (in Gould et al. 2013,
        p. 1712, the residue of an order that runs out of matches "becomes active at
        the price p_x"). With protection 0 it fills only at the best price, like a
        market-to-limit order. Returns (None, []) when the opposite side is empty:
        there is no price to start from, so nothing is placed.
        """
        best = self.book.best_ask if side == Side.BID else self.book.best_bid
        if best is None:
            return None, []
        reach = protection * self.book.tick_size
        limit = best + reach if side == Side.BID else max(self.book.tick_size, best - reach)
        return self.submit_limit_order(price=limit, size=size, side=side)

    def cancel_order_by_id(self, order_id: int) -> Optional[Order]:
        """Cancel one of the user's resting orders; None if it no longer rests or is not theirs.

        Only the user's own orders can be canceled: the models' orders stand for other traders.
        """
        if order_id not in self.manual_order_ids:
            return None
        if self.book.get_order(order_id) is None:
            self.manual_order_ids.discard(order_id)
            return None
        self._advance(self.sim_time + MANUAL_ACTION_DT)
        canceled = self._withdraw(order_id)
        self._after_event()
        return canceled

    def step(self, n_events: int = 1) -> None:
        """Perform n simulation steps from the selected model (in playback, n orders or cancellations from the tape).

        After each step the trigger rules are checked (``_after_event``); a rule that pauses ends the batch there.
        """
        self._halt = False  # a pause of an earlier batch is over
        if self.playback is not None:
            self._step_playback(n_events)
            return
        if self.model_type == "manual":
            return

        for _ in range(n_events):
            if self._halt:  # a rule paused the simulation: the rest of the batch does not run
                break
            order = self.model.generate_event(self.book, self.sim_time, self.rng)
            if order is None:
                self.facts.observe_idle()
                self._after_event()
                continue
            informed = self._informed_model and self.model.last_informed
            self._advance(order.timestamp)
            if order.order_type == OrderType.CANCEL and (order.order_id in self.manual_order_ids
                                                         or order.order_id in self._strategy_orders):
                # A model's cancellations stand for other traders withdrawing their own
                # orders, so the user's orders are left alone. The models pick the order
                # to cancel uniformly from the book, so skipping the user's is exact
                # thinning: the model's own orders keep their cancellation rate.
                self.facts.observe_idle()
                self._after_event()
                continue
            fills = self._process(order)
            if informed:
                self._score_informed(order.side, fills)
            if self.split_orders:
                self._run_split_orders()
            for s in self.strategies:
                if s.running and self.sim_time >= s.next_time:
                    self._run_strategy(s)
            self._after_event()

        # Forget manual orders that have been filled or canceled.
        self.manual_order_ids = {oid for oid in self.manual_order_ids if self.book.get_order(oid) is not None}

    def _score_informed(self, side: Side, fills: list[Fill]) -> None:
        """What the makers lost on an informed trader's fills, against the hidden value: V less the price for a buy,
        the price less V for a sell, per lot. Read at once, while V is still the value at the order's own time."""
        value = self.model.value()
        sign = 1 if side == Side.BID else -1
        for fill in fills:
            self._informed_fill_lots += fill.size
            self._informed_fill_loss += fill.size * sign * (value - fill.price)

    def _process(self, order: Order) -> list[Fill]:
        mid_before = self.book.mid_price  # what the user's fills are scored against
        studies = self.studies
        watch = studies is not None and self.collecting is None and not studies.suspended
        if watch:  # what the engine changes or removes: an iceberg's size when it rests, the order a cancel takes out
            orig_size = order.size
            canceled = self.book.get_order(order.order_id) if order.order_type == OrderType.CANCEL else None
        if self.recorder is not None:  # as it arrives, before matching changes it
            origin = self._origin(order.order_id)
            if order.order_type == OrderType.CANCEL:
                self.recorder.cancel(order.timestamp, order.order_id, origin)
            else:
                self.recorder.order(rec.LIMIT if order.order_type == OrderType.LIMIT else rec.MARKET, order.timestamp,
                                    order.order_id, rec.BID if order.side == Side.BID else rec.ASK, order.price,
                                    order.size, order.display_size or 0, origin)
            if self.recorder.full:
                self._finish_recording(f"it reached {rec.MAX_EVENTS:,} events")
        if self._takes(order):
            self._observe_trade()
        fills = self.engine.process(self.book, order)
        self.event_count += 1
        self.facts.observe(order.order_type, order.side, fills, self.book)
        mine, owners = self.manual_order_ids, self._strategy_orders
        maker_side = Side.ASK if order.side == Side.BID else Side.BID
        for fill in fills:
            self._record_trade(fill, order.side)
            # A fill can be the user's on both sides: a market sell against their own bid.
            if fill.taker_order_id in mine:
                self._record_own_trade(fill, order.side, maker=False, mid_before=mid_before)
            if fill.maker_order_id in mine:
                self._record_own_trade(fill, maker_side, maker=True, mid_before=mid_before)
                self._record_own_fill(fill, maker_side=maker_side)
            for owner_id, side, maker in ((fill.maker_order_id, maker_side, True),
                                          (fill.taker_order_id, order.side, False)):
                owner = owners.get(owner_id)
                if owner is not None:
                    self._book_fill(owner.account, fill, side, maker, mid_before)
            maker_model = fill.maker_order_id not in mine and fill.maker_order_id not in owners
            taker_model = fill.taker_order_id not in mine and fill.taker_order_id not in owners
            if maker_model or taker_model:
                self._market_markouts.append((self.event_count + MARKOUT_EVENTS, maker_side == Side.BID, fill.price,
                                              fill.size, maker_model, taker_model))
        mid = self.book.mid_price
        market = self._market_markouts
        while market and market[0][0] <= self.event_count:  # the model's fills first: what the flow pays now
            _, maker_buying, price, size, maker_model, taker_model = market.popleft()
            if mid is not None:
                edge = ((mid - price) if maker_buying else (price - mid)) * size  # the resting side's
                if maker_model:
                    self._flow_add("make", edge, size)
                if taker_model:
                    self._flow_add("take", -edge, size)
        markouts = self._markouts
        if markouts and markouts[0][0] <= self.event_count:
            joiners = {id(s.account): s.running for s in self.strategies if s.spec.key == "quote_joiner"}
            while markouts and markouts[0][0] <= self.event_count:  # fills that are now MARKOUT_EVENTS old
                _, account, buying, price, size, maker = markouts.popleft()
                if mid is None:
                    continue
                joiner = id(account) in joiners
                null = self._null_rate(maker, any(joiners.values()) and not joiner)
                per_lot = account.markout(buying, price, size, mid, null=None if joiner and maker else null)
                if joiner and maker:
                    self._flow_add("join", per_lot * size, size)
        if watch:
            own = order.order_id in mine or order.order_id in owners
            if order.order_type == OrderType.CANCEL:
                if canceled is not None:  # a cancel of an order that is not there changes nothing
                    self._observe(CANCEL, canceled.side, canceled.price, canceled.size, (), 0, own)
            else:
                self._observe(ARRIVAL_LIMIT if order.order_type == OrderType.LIMIT else ARRIVAL_MARKET, order.side,
                              order.price, orig_size, fills, 1 if order.side == Side.BID else -1, own)
        return fills

    def _null_rate(self, maker: bool, joiner_running: bool) -> Optional[float]:
        """What a strategy with no skill earned a lot on the same side lately: the null a fill is held against.

        Resting: the quote joiners', the naive market makers (only while one runs; not for their own fills).
        Taking: the model's own market orders', which know nothing (the random trader comes out about even
        with them). The model's own resting orders are no null for resting: those placed behind the best
        quote, which fill only once the price comes to them, earn much more than a joiner.
        """
        if maker:
            return self._flow_rate("join") if joiner_running and self._flow["join"][1] > 0 else None
        return self._flow_rate("take") if self._flow["take"][1] > 0 else None

    def _takes(self, order: Order) -> bool:
        """Whether an order arriving now trades at once (a trade, as Wyart et al. 2008 count them)."""
        book = self.book
        if order.order_type == OrderType.MARKET:
            return (book.best_ask if order.side == Side.BID else book.best_bid) is not None
        if order.order_type != OrderType.LIMIT:
            return False
        if order.side == Side.BID:
            return book.best_ask is not None and order.price >= book.best_ask
        return book.best_bid is not None and order.price <= book.best_bid

    def _observe_trade(self) -> None:
        """The spread just before a trade, and the mid's change since just before the last one."""
        mid = self.book.mid_price
        if mid is None:
            return
        r = self._regime
        keep = 1 - 1 / REGIME_TRADES
        r["trades"] += 1
        r["w"] = r["w"] * keep + 1
        r["spread"] = r["spread"] * keep + self.book.spread
        if r["last_mid"] is not None:
            change = mid - r["last_mid"]
            r["dw"] = r["dw"] * keep + 1
            r["d2"] = r["d2"] * keep + change * change
        r["last_mid"] = mid

    def _flow_add(self, side: str, edge: float, lots: int) -> None:
        keep = 1 - 1 / FLOW_FILLS
        f = self._flow[side]
        f[0], f[1] = f[0] * keep + edge, f[1] * keep + lots

    def _flow_rate(self, side: str) -> float:
        """What the model's own orders earn a lot, resting ("make") or taking ("take"), MARKOUT_EVENTS events later."""
        edge, lots = self._flow[side]
        return edge / lots if lots > 0 else 0.0

    def regime(self) -> Optional[dict[str, Any]]:
        """Whether the order flow pays liquidity providers here: the spread against the mid's move per trade.

        ``spread``: the average spread just before each trade; ``sigma1``: the root mean square of
        the mid's change from just before one trade to just before the next; their ratio is about
        1.2 to 1.6 in real small-tick markets, where no strategy is more than marginally
        profitable (Wyart et al. 2008). ``make`` and ``take``: what the model's own resting and
        taking orders earn per lot, MARKOUT_EVENTS events later. None until there have been 50
        trades and the model's orders have traded.
        """
        r = self._regime
        if r["trades"] < 50 or r["dw"] <= 0 or self._flow["make"][1] <= 0:
            return None
        spread = r["spread"] / r["w"]
        sigma1 = math.sqrt(r["d2"] / r["dw"])
        joiner = any(s.running and s.spec.key == "quote_joiner" for s in self.strategies)
        return {"trades": r["trades"], "spread": spread, "sigma1": sigma1,
                "ratio": spread / sigma1 if sigma1 > 0 else None,
                "make": self._flow_rate("make"), "take": self._flow_rate("take"),
                "join": self._flow_rate("join") if joiner and self._flow["join"][1] > 0 else None,
                "joiner": joiner}

    def _book_fill(self, account: Account, fill: Fill, side: Side, maker: bool, mid_before: Optional[float]) -> None:
        """Book a fill in an account, and note it for its markout MARKOUT_EVENTS events from now."""
        account.book(side == Side.BID, fill.price, fill.size, maker, mid_before)
        self._markouts.append((self.event_count + MARKOUT_EVENTS, account, side == Side.BID, fill.price, fill.size,
                               maker))

    # ---- The clock and the order-book heatmap ----

    def _advance(self, t: float) -> None:
        """Move the clock forward to ``t`` (never back), starting a heatmap column at each whole unit it reaches.

        A column holds the book as it was just before the event that moves the clock, so
        it shows the book at the start of its unit. When the clock passes several units at
        once, nothing happened in between, so each of them gets a column of the same book:
        the columns run one per unit without gaps. None are taken during a collection: the
        heatmap starts over afterwards.
        """
        if t <= self.sim_time:
            return
        index = math.floor(t)
        if index > self._heat_index and self.collecting is None:
            self._take_heat_columns(max(self._heat_index + 1, index - HEAT_COLUMNS + 1), index)
        if index > self._pnl_index:  # each strategy's P&L and the user's, for the leaderboard's sparklines
            mark = self._mark()
            for s in self.strategies:
                s.pnl.append((index, s.account.summary(mark)["total"]))
            self._own_pnl.append((index, self.account.summary(mark)["total"]))
            self._pnl_index = index
        if self._informed_model and index > self._value_last and self.playback is None:
            self._take_value_samples(index)
        studies = self.studies
        if studies is not None and index > studies.unit and self.collecting is None and not studies.suspended:
            try:  # the studies' samples for each unit the clock reaches, of the book as it is just before the event
                studies.sample_to(index, self.book)
            except Exception as exc:
                self._study_failed(exc)
        self.sim_time = t

    def _take_value_samples(self, index: int) -> None:
        """The hidden value at each unit the clock reaches, as the model has it then (at the event that crosses into
        the unit: the model has advanced it to that event's time); a gap, where nothing happened, repeats it. The last
        VALUE_SAMPLES units are kept."""
        samples = self._value_samples
        first = max(self._value_last + 1, index - VALUE_SAMPLES + 1)
        if first > self._value_last + 1:  # further than the samples reach: they are all older than what is kept
            del samples[:]
        samples.extend([self.model.value()] * (index - first + 1))
        self._value_last = index
        if len(samples) > 2 * VALUE_SAMPLES:
            del samples[:len(samples) - VALUE_SAMPLES]

    def _take_heat_columns(self, first: int, last: int) -> None:
        """Columns for the units ``first`` to ``last``, all of the book as it is now (each with its own trades)."""
        book = self.book
        mid = book.mid_price
        if mid is not None:
            center = round(mid)
        else:
            center = book.best_bid or book.best_ask or round(self._reference_price)
        lo = max(1, center - HEAT_HALF)
        hi = lo + 2 * HEAT_HALF
        depth = [book.depth_at_price(p, Side.BID) - book.depth_at_price(p, Side.ASK) for p in range(lo, hi + 1)]
        hidden: dict[int, int] = {}
        for oid in self.manual_order_ids:  # only the user's orders can be icebergs
            order = book.get_order(oid)
            if order is not None and order.hidden and lo <= order.price <= hi:
                hidden[order.price] = hidden.get(order.price, 0) + order.hidden
        for index in range(first, last + 1):  # the depth and reserves are never changed, so they can be shared
            self._heat_columns.append(HeatColumn(index=index, lo=lo, mid=mid, depth=depth, hidden=hidden))
        self._heat_index = last
        self.heat_changes += 1

    def _restart_heatmap(self) -> None:
        self._heat_columns: deque[HeatColumn] = deque(maxlen=HEAT_COLUMNS)
        now = math.floor(self.sim_time)
        self._take_heat_columns(now, now)
        self.heat_version += 1

    def heat_columns(self, since: Optional[int] = None) -> list[dict[str, Any]]:
        """The heatmap's columns from index ``since`` on (all of them when None)."""
        return [c.to_json() for c in self._heat_columns if since is None or c.index >= since]

    def latest_heat_column(self) -> dict[str, Any]:
        return self._heat_columns[-1].to_json()

    def _record_own_trade(self, fill: Fill, side: Side, maker: bool, mid_before: Optional[float]) -> None:
        """Book a fill of the user's (``side`` is theirs) in their account and in the totals per unit of time."""
        account = self.account
        self._book_fill(account, fill, side, maker, mid_before)
        index = math.floor(fill.timestamp)
        bars = self._own_bars
        if not bars or bars[-1].index < index:
            bars.append(OwnBar(index))
        bar = bars[-1]  # a fill stamped a little before the latest bar joins it, as in _record_trade
        if side == Side.BID:
            bar.bought += fill.size
            bar.buy_cost += fill.price * fill.size
        else:
            bar.sold += fill.size
            bar.sell_cost += fill.price * fill.size
        bar.position, bar.cash = account.position, account.cash
        self.own_trade_seq += 1

    def _mark(self) -> float:
        """The price open positions are marked to: the mid, or the last trade price when a side is empty."""
        mark = self.book.mid_price
        if mark is None:
            mark = self._base_bars[-1].close if self._base_bars else self._reference_price
        return mark

    def account_summary(self) -> dict[str, Any]:
        """The user's position and P&L, the open position marked to the mid (or the last trade price)."""
        return self.account.summary(self._mark())

    # ---- The user's strategies ----

    def start_strategy(self, spec: StrategySpec, values: dict[str, Any]) -> RunningStrategy:
        """Put a strategy of the user's into the market; it decides first at once, then every interval.

        Raises ValueError for settings that contradict each other and RuntimeError when the user
        already has MAX_STRATEGIES. Like the user's own orders, the models never cancel its
        orders; its fills go to its own account.
        """
        if len(self.strategies) >= MAX_STRATEGIES:
            raise RuntimeError(f"at most {MAX_STRATEGIES} strategies at once: remove one first")
        strategy = spec.build(values)
        self._strategy_seq += 1
        number = 1 + max((s.number for s in self.strategies if s.spec.key == spec.key), default=0)
        s = RunningStrategy(id=self._strategy_seq, spec=spec, values=dict(values), strategy=strategy, number=number,
                            name=f"{spec.label} {number}", next_time=self.sim_time)
        self.strategies.append(s)
        self._note_strategy(rec.STRATEGY_START, s)
        self._run_strategy(s)
        return s

    def _note_strategy(self, kind: int, s: RunningStrategy) -> None:
        if self.recorder is not None:
            extra = {"kind": s.spec.key, "values": dict(s.values), "number": s.number, "name": s.name} \
                if kind == rec.STRATEGY_START else None
            self.recorder.note(kind, self.sim_time, s.id, extra)

    def strategy(self, strategy_id: int) -> Optional[RunningStrategy]:
        return next((s for s in self.strategies if s.id == strategy_id), None)

    def stop_strategy(self, s: RunningStrategy) -> None:
        """Stop deciding and withdraw its resting orders (its account and position stay)."""
        s.running = False
        self._note_strategy(rec.STRATEGY_STOP, s)
        self._withdraw_strategy_orders(s)

    def resume_strategy(self, s: RunningStrategy) -> None:
        s.running = True
        s.next_time = self.sim_time
        self._note_strategy(rec.STRATEGY_RESUME, s)
        self._run_strategy(s)

    def remove_strategy(self, s: RunningStrategy) -> None:
        self._withdraw_strategy_orders(s)
        self._note_strategy(rec.STRATEGY_REMOVE, s)
        self.strategies.remove(s)

    def _restart_strategies(self) -> None:
        """A new book: every strategy starts again from scratch (fresh state and account), as it was set."""
        self._strategy_orders = {}
        self._pnl_index = -1
        self.strategies = [self._fresh(s) for s in self.strategies]

    @staticmethod
    def _fresh(s: RunningStrategy) -> RunningStrategy:
        """The strategy as it was set, with a fresh state and account (and deciding for itself)."""
        return RunningStrategy(id=s.id, spec=s.spec, values=s.values, strategy=s.spec.build(s.values),
                               number=s.number, name=s.name, running=s.running)

    def _withdraw_strategy_orders(self, s: RunningStrategy) -> None:
        # In the order they were placed (ids rise), not the set's: a set of ids iterates by their values, which
        # depend on every order made before in the process, and the order of cancellations shapes the book's
        # list of orders, which the models draw from. So a seed replays a run exactly.
        for oid in sorted(s.order_ids):
            self._strategy_orders.pop(oid, None)
            self._cancel_resting(oid, s.id)
        s.order_ids = set()
        s.quotes = {}

    def _strategy_order(self, s: RunningStrategy, price: int, size: int, side: Side) -> Order:
        order = limit_order(price=price, size=size, timestamp=self.sim_time, side=side)
        s.order_ids.add(order.order_id)
        self._strategy_orders[order.order_id] = s
        self._process(order)
        return order

    def _view(self, s: RunningStrategy) -> View:
        """What a strategy sees: the best quotes and their lots, leaving out its own orders, and its position."""
        own: dict[tuple[Side, int], int] = {}
        for oid in s.order_ids:
            o = self.book.get_order(oid)
            if o is not None:
                own[(o.side, o.price)] = own.get((o.side, o.price), 0) + o.size
        best: dict[Side, tuple[Optional[int], int]] = {}
        for side in (Side.BID, Side.ASK):
            best[side] = (None, 0)
            for price, depth in self.book.depth_profile(side, len(own) + 1):  # its orders fill at most len(own) levels
                others = depth - own.get((side, price), 0)
                if others > 0:
                    best[side] = (price, others)
                    break
        (best_bid, bid_depth), (best_ask, ask_depth) = best[Side.BID], best[Side.ASK]
        return View(time=self.sim_time, best_bid=best_bid, best_ask=best_ask, bid_depth=bid_depth,
                    ask_depth=ask_depth, position=s.account.position)

    def _run_strategy(self, s: RunningStrategy) -> None:
        """Ask the strategy what it wants now, and carry it out."""
        strategy, book = s.strategy, self.book
        s.next_time = self.sim_time + strategy.interval
        self._note_strategy(rec.DECIDE, s)
        for oid in [oid for oid in s.order_ids if book.get_order(oid) is None]:  # filled since
            s.order_ids.discard(oid)
            self._strategy_orders.pop(oid, None)
        plan = strategy.decide(self._view(s))
        s.signal = plan.signal
        if plan.target is not None:
            s.target = plan.target
        if plan.quotes is not None:
            if strategy.keeps_queue:  # a quote that is where it is wanted stays, keeping its place in the queue
                kept: dict[Side, Order] = {}
                for oid in sorted(s.order_ids):  # in the order placed (see _withdraw_strategy_orders)
                    o = book.get_order(oid)
                    want = plan.quotes.get(o.side) if o is not None else None
                    if want is not None and o.side not in kept and o.price == want[0] and o.size + o.hidden <= want[1]:
                        kept[o.side] = o
                    else:
                        s.order_ids.discard(oid)
                        self._strategy_orders.pop(oid, None)
                        self._cancel_resting(oid, s.id)
                s.quotes = {side.value: (o.price, o.size) for side, o in kept.items()}
            else:  # new quotes each time: each joins the back of its queue
                self._withdraw_strategy_orders(s)
                kept = {}
            for side, (price, size) in plan.quotes.items():
                if side not in kept and size > 0 and price >= 1:
                    self._strategy_order(s, price, size, side)
                    s.quotes[side.value] = (price, size)
            return
        if plan.target is None:
            return
        need = plan.target - s.account.position
        side = Side.BID if need > 0 else Side.ASK
        if strategy.execution == "aggressive":
            self._withdraw_strategy_orders(s)
            limit = self._protection_limit(side, strategy.protection) if need else None
            if limit is not None:  # a protected market order; what it cannot fill does not wait
                order = self._strategy_order(s, limit, abs(need), side)
                if book.get_order(order.order_id) is not None:
                    self._withdraw_strategy_orders(s)
            return
        # Passive: one order at the best quote on its side (its own included: alone at the best, it stays),
        # left alone while it is still the one wanted, so that it keeps its place in the queue.
        price = book.best_bid if side == Side.BID else book.best_ask
        resting = [o for o in (book.get_order(oid) for oid in s.order_ids) if o is not None]
        if need and price is not None and len(resting) == 1 and resting[0].side == side \
                and resting[0].price == price and resting[0].size + resting[0].hidden == abs(need):
            return
        self._withdraw_strategy_orders(s)
        if need and price is not None:
            self._strategy_order(s, price, abs(need), side)
            s.quotes[side.value] = (price, abs(need))

    def strategies_summary(self) -> list[dict[str, Any]]:
        """The leaderboard: every strategy's settings, state and account, and the user's, best P&L first.

        The user's row (id None) is there only alongside at least one strategy.
        """
        if not self.strategies:
            return []
        mark = self._mark()

        def row(account: Account, samples: deque, **rest: Any) -> dict[str, Any]:
            summary = account.summary(mark)
            edge, markout = summary["edge"], summary["markout"]
            return {"account": summary, "lots": summary["bought"] + summary["sold"], "spark": sparkline(samples),
                    # How much of the edge at the fill the mid took back by MARKOUT_EVENTS events later.
                    "adverse": edge - markout if edge is not None and markout is not None else None, **rest}

        rows = [row(s.account, s.pnl, id=s.id, name=s.name, tag=s.spec.tag, kind=s.spec.key, running=s.running,
                    replayed=s.replayed, quoting=s.spec.family == "quoting", signal_name=s.spec.signal[0],
                    signal_help=s.spec.signal[1], fit=s.fit,
                    settings=s.strategy.describe(), signal=s.signal, target=s.target, quotes=dict(s.quotes),
                    warmup=s.strategy.warming_up())
                for s in self.strategies]
        rows.append(row(self.account, self._own_pnl, id=None, name="You", tag="YOU", kind="you", running=None,
                        replayed=False, quoting=False, signal_name="", signal_help="", fit=None,
                        settings="your own orders, from the order form and the order book", signal=None, target=None,
                        quotes={}, warmup=None))
        return sorted(rows, key=lambda r: r["account"]["total"], reverse=True)

    def _record_own_fill(self, fill: Fill, maker_side: Side) -> None:
        resting = self.book.get_order(fill.maker_order_id)
        own = OwnFill(order_id=fill.maker_order_id, side=maker_side.value, price=fill.price, size=fill.size,
                      left=resting.size + resting.hidden if resting is not None else 0, time=fill.timestamp)
        self.own_fills.append(own)
        self._notify(own.describe())

    def _notify(self, text: str) -> None:
        if self._quiet:
            return
        self.notice_seq += 1
        self.notices.append(Notice(self.notice_seq, text))

    # ---- Recording and playback (app/recording.py) ----

    def start_recording(self) -> None:
        """Record from now on: the book as it stands, then every order, cancellation and change.

        Raises RuntimeError while recording already, or with MAX_RECORDINGS kept.
        """
        if self.recorder is not None:
            raise RuntimeError("already recording")
        if len(self.recordings) >= rec.MAX_RECORDINGS:
            raise RuntimeError(f"at most {rec.MAX_RECORDINGS} recordings are kept: download or delete one first")
        self.recorder = rec.Recorder(self._snapshot(), created=time.strftime("%Y-%m-%d %H:%M"))
        self.recordings_version += 1

    def stop_recording(self) -> rec.Recording:
        if self.recorder is None:
            raise RuntimeError("not recording")
        return self._finish_recording(None)

    def _finish_recording(self, why: Optional[str]) -> rec.Recording:
        self._recording_seq += 1
        recording = self.recorder.finish(self._recording_seq, f"Recording {self._recording_seq}", self.sim_time)
        self.recorder = None
        self.recordings.append(recording)
        self.recordings_version += 1
        self._notify(f"Saved {recording.name}: {len(recording.tape):,} events from {recording.start_time:.2f} to "
                     f"{recording.end_time:.2f}{f', because {why}' if why else ''}. Play it back from Analyze.")
        return recording

    def recording(self, recording_id: int) -> Optional[rec.Recording]:
        return next((r for r in self.recordings if r.id == recording_id), None)

    def delete_recording(self, recording: rec.Recording) -> None:
        if self.playback is not None and self.playback.recording is recording:
            raise RuntimeError("it is playing back: exit playback first")
        self.recordings.remove(recording)
        self.recordings_version += 1

    def add_recording(self, data: bytes) -> rec.Recording:
        """Keep a recording from a file (``app.recording.to_bytes``); ValueError says what is wrong with one."""
        if len(self.recordings) >= rec.MAX_RECORDINGS:
            raise RuntimeError(f"at most {rec.MAX_RECORDINGS} recordings are kept: delete one first")
        recording = rec.from_bytes(data, self._recording_seq + 1)
        self._recording_seq += 1
        names = {r.name for r in self.recordings}
        base, n = recording.name, 1
        while recording.name in names:  # two of one name would be hard to tell apart
            n += 1
            recording.name = f"{base} ({n})"
        self.recordings.append(recording)
        self.recordings_version += 1
        return recording

    def _snapshot(self) -> dict[str, Any]:
        """The session as a recording starts from: the book in queue order, the accounts, the strategies, the clock."""
        accounts = {id(self.account): rec.FROM_YOU} | {id(s.account): s.id for s in self.strategies}
        lots = self.studies.volume if self.studies is not None else None
        orders = [[o.order_id, rec.BID if side == Side.BID else rec.ASK, o.price, o.size, o.hidden,
                   o.display_size or 0, o.timestamp, self._origin(o.order_id)]
                  for side in (Side.BID, Side.ASK) for o in self.book.resting_orders(side)]
        return {
            "time": self.sim_time, "event_count": self.event_count, "traded_volume": self.traded_volume,
            "cum_volume": self._cum_volume, "cum_turnover": self._cum_turnover, "reference_price": self._mark(),
            "engine": self.engine_type, "model": self.model_type,
            "model_params": dict(self.model_params[self.model_type]), "seed": self.seed,
            "orders": orders, "account": asdict(self.account),
            "strategies": [{"id": s.id, "kind": s.spec.key, "values": dict(s.values), "number": s.number,
                            "name": s.name, "running": s.running, "next_time": s.next_time, "signal": s.signal,
                            "target": s.target, "quotes": {k: list(v) for k, v in s.quotes.items()},
                            "account": asdict(s.account), "state": s.strategy.state()} for s in self.strategies],
            "markouts": [[due, accounts[id(account)], buying, price, size, maker]
                         for due, account, buying, price, size, maker in self._markouts if id(account) in accounts],
            "market_markouts": [list(m) for m in self._market_markouts],
            "flow": {side: list(f) for side, f in self._flow.items()},
            "regime": dict(self._regime),
            # The lots of a VPIN bucket, so that a playback measures with the live market's (once it is known).
            **({"studies": {"vpin_v": lots}} if lots is not None else {}),
        }

    def start_playback(self, recording: rec.Recording) -> None:
        """Play a recording back from its start, paused: its tape takes the model's place.

        The session becomes the recording's: its book, your orders and account then, and its
        strategies (they follow the tape). A recording in progress is saved first.
        """
        if self.recorder is not None:
            self._finish_recording("a playback began")
        self.playback = Playback(recording)
        self._load_recording_start(keep_live=False)

    def rewind_playback(self) -> None:
        """Back to the start of the recording; strategies started during the playback start over too."""
        self._load_recording_start(keep_live=True)

    def _load_recording_start(self, keep_live: bool) -> None:
        pb = self.playback
        start = pb.recording.start
        live = [self._fresh(s) for s in self.strategies if not s.replayed] if keep_live else []
        self._clear(time=float(start["time"]), event_count=int(start["event_count"]),
                    reference=float(start["reference_price"]))
        self.traded_volume = int(start["traded_volume"])
        self._cum_volume, self._cum_turnover = int(start["cum_volume"]), float(start["cum_turnover"])
        if start["engine"] != self.engine_type:
            self.engine_type = start["engine"]
            self.engine = PriceTimeEngine() if self.engine_type == "price_time" else ProRataEngine()
            self.config_version += 1
        self.account = Account(**start["account"])
        pb.cursor, pb.ids, pb.strategies, pb.ended, pb.model = 0, {}, {}, False, start["model"]
        self.strategies = []
        for d in start["strategies"]:
            s = self._replayed_strategy(d)
            s.running, s.next_time, s.signal, s.target = d["running"], d["next_time"], d["signal"], d["target"]
            s.quotes = {k: tuple(v) for k, v in d["quotes"].items()}
            s.account = Account(**d["account"])
            s.strategy.restore(d["state"])
        self.strategies.extend(live)
        for oid, side, price, size, hidden, display, timestamp, origin in start["orders"]:
            order = Order(price=price, size=size, timestamp=timestamp, side=Side.BID if side == rec.BID else Side.ASK,
                          display_size=display or None, hidden=hidden)
            self.book.add_order(order)  # it rests as it was: an iceberg's peak and reserve as they were
            pb.ids[oid] = order.order_id
            self._own(order.order_id, origin)
        owners = {rec.FROM_YOU: self.account} | {sid: s.account for sid, s in pb.strategies.items()}
        self._markouts.extend((row[0], owners[row[1]], row[2], row[3], row[4], row[5] if len(row) > 5 else True)
                              for row in start["markouts"] if row[1] in owners)  # a 5-long row: before makers were told
        self._market_markouts.extend(tuple(m) for m in start.get("market_markouts", []))
        for side, f in start.get("flow", {}).items():
            self._flow[side] = [float(f[0]), float(f[1])]
        self._regime.update(start.get("regime", {}))
        self._reset_model(int(self.book.mid_price or self._reference_price))  # the one that takes over afterwards
        self.facts.reset()
        self.chart_version += 1
        self._restart_heatmap()
        self.recordings_version += 1
        if self.studies is not None and "vpin_v" in start.get("studies", {}):
            self.studies.set_volume(start["studies"]["vpin_v"])
        self._resync_studies()

    def _replayed_strategy(self, d: dict[str, Any]) -> RunningStrategy:
        """A recorded strategy (``d``: its id, kind, settings, number and name), on the board and following the tape."""
        from app.strategy_specs import STRATEGY_SPECS

        spec = STRATEGY_SPECS[d["kind"]]
        values, _ = spec.parse(d["values"])
        self._strategy_seq += 1
        s = RunningStrategy(id=self._strategy_seq, spec=spec, values=values, strategy=spec.build(values),
                            number=d["number"], name=d["name"], next_time=self.sim_time, replayed=True)
        self.playback.strategies[d["id"]] = s
        self.strategies.append(s)
        return s

    def _own(self, order_id: int, origin: int) -> None:
        """Make a replayed order the user's or a replayed strategy's, as it was recorded."""
        if origin == rec.FROM_YOU:
            self.manual_order_ids.add(order_id)
        elif origin != rec.FROM_MODEL:
            s = self.playback.strategies.get(origin)
            if s is not None:
                s.order_ids.add(order_id)
                self._strategy_orders[order_id] = s

    def _disown(self, order_id: int) -> None:
        self.manual_order_ids.discard(order_id)
        s = self._strategy_orders.pop(order_id, None)
        if s is not None:
            s.order_ids.discard(order_id)

    def _step_playback(self, n: int, every_event: bool = False) -> None:
        """The next ``n`` orders and cancellations on the tape (with ``every_event``, the next n events of any kind).

        After each, the user's split orders and the strategies started during the playback act,
        as after a model's event, and then the trigger rules (notice and pause only: the tape holds the orders of
        the recorded session). At the end of the tape the simulation stops.
        """
        pb = self.playback
        tape = pb.recording.tape
        done = 0
        self._halt = False
        while done < n:
            if self._halt:  # a rule paused the simulation
                break
            if pb.cursor >= len(tape):
                self._end_of_tape()
                break
            i = pb.cursor
            pb.cursor += 1
            try:
                book_event = self._apply(i)
            except Exception as exc:  # a file from elsewhere that passed the checks but does not play
                pb.ended = True
                self.is_running = False
                self._notify(f"The playback stopped at {tape.time[i]:.2f}: event {i:,} does not play ({exc}).")
                break
            if book_event or every_event:
                done += 1
            if book_event and self.playback is pb:
                if self.split_orders:
                    self._run_split_orders()
                for s in self.strategies:
                    if s.running and not s.replayed and self.sim_time >= s.next_time:
                        self._run_strategy(s)
                self._after_event()
        if self.playback is pb and pb.cursor >= len(tape):  # stop at the last event, not a step later
            self._end_of_tape()
        self.manual_order_ids = {oid for oid in self.manual_order_ids if self.book.get_order(oid) is not None}

    def _end_of_tape(self) -> None:
        pb = self.playback
        if not pb.ended:
            pb.ended = True
            self.is_running = False
            self.recordings_version += 1
            self._notify(f"The end of {pb.recording.name}. Reset plays it again; Exit playback carries on live "
                         "from here.")

    def _apply(self, i: int) -> bool:
        """Apply event ``i`` of the tape; True for an order or a cancellation."""
        pb = self.playback
        tape = pb.recording.tape
        kind, t, oid = tape.kind[i], tape.time[i], tape.oid[i]
        self._advance(t)
        if kind == rec.LIMIT or kind == rec.MARKET:
            side = Side.BID if tape.side[i] == rec.BID else Side.ASK
            if kind == rec.LIMIT:
                order = limit_order(tape.price[i], tape.size[i], t, side, display_size=tape.display[i] or None)
            else:
                order = market_order(tape.size[i], t, side)
            pb.ids[oid] = order.order_id
            self._own(order.order_id, tape.origin[i])
            self._process(order)
            if self.book.get_order(order.order_id) is None:  # it traded on arrival
                self._disown(order.order_id)
            return True
        if kind == rec.CANCEL:
            live = pb.ids.pop(oid, None)
            if live is None or self.book.get_order(live) is None:  # gone already: the market went otherwise
                self.facts.observe_idle()
            elif self._origin(live) == rec.FROM_MODEL:
                self._process(cancel_order(live, t))
            else:
                origin = self._origin(live)
                self._disown(live)
                self._cancel_resting(live, origin)
            return True
        extra = tape.extras.get(i, {})
        if kind == rec.ENGINE:
            self.set_engine(extra["engine"])
        elif kind == rec.MODEL_CHANGE:
            pb.model = extra["model"]
            self._recalibrate_studies()  # as set_model and configure do live
            if self.recorder is not None:
                self.recorder.note(rec.MODEL_CHANGE, t, extra=extra)
        elif kind == rec.STRATEGY_START:
            s = self._replayed_strategy({"id": oid, **extra})
            self._note_strategy(rec.STRATEGY_START, s)
        else:
            s = pb.strategies.get(oid)
            if s is None:
                return False
            if kind == rec.DECIDE:
                self._follow_decision(s)
            elif kind == rec.STRATEGY_STOP:
                s.running = False
            elif kind == rec.STRATEGY_RESUME:
                s.running = True
            elif kind == rec.STRATEGY_REMOVE:
                pb.strategies.pop(oid, None)
                for order_id in list(s.order_ids):
                    self._disown(order_id)
                if s in self.strategies:
                    self.strategies.remove(s)
            self._note_strategy(kind, s)
        return False

    def _follow_decision(self, s: RunningStrategy) -> None:
        """A replayed strategy decides when it did, from the same view, so its signal and state follow along.

        What it then sent is on the tape: the plan itself is not carried out.
        """
        s.next_time = self.sim_time + s.strategy.interval
        for oid in [oid for oid in s.order_ids if self.book.get_order(oid) is None]:
            s.order_ids.discard(oid)
            self._strategy_orders.pop(oid, None)
        plan = s.strategy.decide(self._view(s))
        s.signal = plan.signal
        if plan.target is not None:
            s.target = plan.target
        if plan.quotes is not None:
            s.quotes = {side.value: q for side, q in plan.quotes.items() if q[1] > 0 and q[0] >= 1}

    async def seek_playback(self, to_time: float, chunk: int = 5_000) -> None:
        """Move the playback to ``to_time``: forward through the tape, or from its start again to go back.

        The simulation is paused first. Going back undoes what was done during the playback
        (strategies started during it start over). Runs in chunks, as a collection does. The studies
        are off until the cursor is SEEK_STUDY_LOOKBACK events short of the target (when it is that far
        away), and then start over: they only need the last of what they are to show.
        """
        await self.stop_sim_loop()
        pb = self.playback
        if pb is None:
            return
        # A range input rounds the time it gives (to 15 digits in Chrome): events a hair later count as reached,
        # so the timeline's far end is the end of the tape.
        target = pb.recording.tape.index_after(to_time + 1e-9 * max(1.0, abs(to_time)))
        resume = target - SEEK_STUDY_LOOKBACK  # the cursor at which the studies start again
        studies, held = self.studies, False  # held: the studies are off for this seek
        async with self._lock:
            if target < pb.cursor:
                self.rewind_playback()
            if studies is not None and studies.failed is None and pb.cursor < resume:
                studies.suspended = held = True
        self._quiet = True
        last_notice = time.monotonic()
        try:
            while self.playback is pb and pb.cursor < target and not pb.ended:
                async with self._lock:
                    n = min(chunk, target - pb.cursor)
                    self._step_playback(min(n, resume - pb.cursor) if held and pb.cursor < resume else n,
                                        every_event=True)
                    if held and pb.cursor >= resume:
                        held = False
                        studies.resume()
                        self._restart_studies()
                self.seeking = (pb.cursor, target)
                if time.monotonic() - last_notice > 0.2:
                    await self.broadcast()
                    last_notice = time.monotonic()
                await asyncio.sleep(0)
        finally:
            self._quiet = False
            self.seeking = None
            if held:  # it ended early: the tape did not play, or the session was reset or left the playback
                studies.resume()
                self._restart_studies()
            if self.triggers is not None:
                self.triggers.clear_runtime()  # nothing acted during the seek: the rules start again from the target
            await self.broadcast()

    def exit_playback(self) -> None:
        """Carry on live from here: the selected model takes over the book, and the replayed strategies decide again."""
        if self.playback is None:
            return
        self.playback = None
        for s in self.strategies:
            s.replayed = False
        self.rng = np.random.default_rng(self.seed)
        self._reset_model(int(self.book.mid_price or self._reference_price))
        self.recordings_version += 1
        self.config_version += 1
        self._notify(f"Live from {self.sim_time:.2f}: {MODEL_SPECS[self.model_type].label} takes over the book.")

    def recording_status(self) -> Optional[dict[str, Any]]:
        r = self.recorder
        return None if r is None else {"events": len(r.tape), "since": float(r.start["time"])}

    def playback_status(self) -> Optional[dict[str, Any]]:
        pb = self.playback
        if pb is None:
            return None
        r = pb.recording
        return {"id": r.id, "name": r.name, "start": r.start_time, "end": r.end_time, "time": self.sim_time,
                "cursor": pb.cursor, "events": len(r.tape), "ended": pb.ended,
                "recorded_with": MODEL_SPECS[pb.model].label, "takes_over": MODEL_SPECS[self.model_type].label,
                "seeking": self.seeking}

    # ---- Fitting a strategy over a training period ----

    async def training_tape(self, events: int, seed: int) -> rec.Recording:
        """A training period: ``events`` model steps from the session as it stands, with random numbers of their own.

        A copy of the session (its book, orders, accounts and strategies, which take part) runs on
        with the selected model, seeded from ``seed``, not from the session's own generator: the
        live market afterwards is a different draw, so what a strategy is fitted on is not what it
        then meets. Raises ValueError for the manual model, which makes no orders.
        """
        if self.model_type == "manual":
            raise ValueError("the manual model makes no orders: choose a model, or fit on a recording")
        async with self._lock:
            start = self._snapshot()
        scratch = LOBManager()
        scratch.model_type = self.model_type
        scratch.model_params = {key: dict(values) for key, values in self.model_params.items()}
        scratch.seed = seed
        scratch._quiet = True
        scratch.start_playback(rec.Recording(id=0, name="", created="", start=start, tape=rec.Tape(),
                                             end_time=float(start["time"])))
        scratch.exit_playback()  # the copy carries on live, its model taking over with the seed's numbers
        scratch.start_recording()
        done = 0
        while done < events:
            k = min(5_000, events - done)
            scratch.step(k)
            done += k
            await asyncio.sleep(0)
        return scratch.stop_recording()

    async def fit_strategy(self, spec: StrategySpec, base: dict[str, Any], grid: dict[str, list[Any]],
                           training: rec.Recording, source: str, objective: str, seed: int = 1) -> Fit:
        """Try the combinations of ``grid``'s values (the other settings as in ``base``) over ``training``.

        All of them, or, when there are more than fit (MAX_FIT_CANDIDATES, and fewer over a long
        period, to stay within MAX_FIT_WORK), a random sample of them drawn from ``seed``. Each
        candidate replays the same tape from its start, trading for itself on it (the recorded
        order flow does not react to it), and is scored by its P&L at the end, marked to the mid,
        or by its P&L per unit of risk: the mean change of its P&L per unit of time over the
        standard deviation of those changes. Raises ValueError when no combination is valid, and
        RuntimeError while another fit runs.
        """
        if objective not in FIT_OBJECTIVES:
            raise ValueError(f"the objective must be one of {', '.join(FIT_OBJECTIVES)}")
        budget = max(1, min(MAX_FIT_CANDIDATES, MAX_FIT_WORK // max(1, len(training.tape))))
        candidates, valid = spec.candidates(base, grid, budget, seed)
        if not candidates:
            raise ValueError("none of these combinations of settings is valid")
        if self.fitting is not None:
            raise RuntimeError("another fit is running")
        self.fitting = (0, len(candidates))
        results = []
        last_push = time.monotonic()
        try:
            for i, values in enumerate(candidates):
                run = LOBManager()
                run._quiet = True
                run.start_playback(training)
                s = run.start_strategy(spec, values)
                s.pnl = deque()  # every sample, for the score
                while not run.playback.ended:
                    run._step_playback(5_000, every_event=True)
                    await asyncio.sleep(0)
                account = s.account.summary(run._mark())
                results.append({"values": values, "account": account, "lots": account["bought"] + account["sold"],
                                "score": account["total"] if objective == "pnl" else _risk_adjusted(s.pnl)})
                self.fitting = (i + 1, len(candidates))
                if time.monotonic() - last_push > 0.3:
                    await self.broadcast()
                    last_push = time.monotonic()
        finally:
            self.fitting = None
        results.sort(key=lambda r: (-r["score"], r["lots"]))  # ties: the one that traded less
        self._fit_seq += 1
        tape = training.tape
        self.last_fit = Fit(id=self._fit_seq, kind=spec.key, base=dict(base), grid=grid, source=source,
                            events=training.meta.get("book_events", len(tape)),
                            span=(training.start_time, training.end_time), objective=objective, candidates=results,
                            valid=valid, size=math.prod(len(v) for v in grid.values()), seed=seed)
        await self.broadcast()
        return self.last_fit

    def start_fitted(self, fit_id: int, rank: int) -> RunningStrategy:
        """Start the ``rank``-th candidate (from 1) of the last fit, noting how it did in training.

        Raises LookupError when that fit or candidate is gone, and as start_strategy otherwise.
        """
        fit = self.last_fit
        if fit is None or fit.id != fit_id or not 1 <= rank <= len(fit.candidates):
            raise LookupError("that fit is no longer here: fit it again")
        from app.strategy_specs import STRATEGY_SPECS

        candidate = fit.candidates[rank - 1]
        s = self.start_strategy(STRATEGY_SPECS[fit.kind], candidate["values"])
        s.fit = {"rank": rank, "of": len(fit.candidates), "source": fit.source, "events": fit.events,
                 "objective": FIT_OBJECTIVES[fit.objective], "score": candidate["score"],
                 "in_training": candidate["account"]["total"], "lots": candidate["lots"],
                 "fitted": sorted(fit.grid)}
        return s

    # ---- Split orders (TWAP) ----

    def _protection_limit(self, side: Side, protection: int) -> Optional[int]:
        """The price a protected market order may walk to, or None when the opposite side is empty."""
        best = self.book.best_ask if side == Side.BID else self.book.best_bid
        if best is None:
            return None
        reach = protection * self.book.tick_size
        return best + reach if side == Side.BID else max(self.book.tick_size, best - reach)

    def start_split_order(self, side: Side, total: int, slices: int, interval: float,
                          protection: int = DEFAULT_PROTECTION_TICKS) -> SplitOrder:
        """Send a market order of ``total`` lots as ``slices`` orders, one every ``interval`` of simulation time.

        Each slice is a protected market order for the remaining lots spread over the
        remaining slices; whatever a slice cannot fill is canceled and rolls into the next.
        Slices go out as simulation time passes (so not while the simulation is paused).
        The first goes out now. For comparison, the order also records what one market
        order for the whole size would have done at the start, and the session totals for
        the market's VWAP over the period.
        """
        self._advance(self.sim_time + MANUAL_ACTION_DT)
        limit = self._protection_limit(side, protection)
        sweep_filled, sweep_avg = sweep_estimate(self.book, side, total, limit) if limit is not None else (0, None)
        self._split_seq += 1
        split = SplitOrder(split_id=self._split_seq, side=side, total=total, slices=slices, interval=interval,
                           protection=protection, start_time=self.sim_time, turnover0=self._cum_turnover,
                           volume0=self._cum_volume, sweep_filled=sweep_filled, sweep_avg=sweep_avg,
                           next_time=self.sim_time)
        self.split_orders.append(split)
        self._run_split_orders()
        self._after_event()
        return split

    def cancel_split_order(self, split_id: int) -> bool:
        for split in self.split_orders:
            if split.split_id == split_id:
                self._finish_split(split, "canceled")
                self.split_orders.remove(split)
                return True
        return False

    def _run_split_orders(self) -> None:
        """Send every slice that is due by the current simulation time."""
        for split in list(self.split_orders):
            while split.finished is None and self.sim_time >= split.next_time:
                self._send_slice(split)
        self.split_orders = [s for s in self.split_orders if s.finished is None]

    def _send_slice(self, split: SplitOrder) -> None:
        remaining = split.total - split.filled
        size = -(-remaining // (split.slices - split.sent))  # the rest spread over the slices left
        split.sent += 1
        split.next_time = split.start_time + split.sent * split.interval
        limit = self._protection_limit(split.side, split.protection)
        if limit is not None and size > 0:
            order = limit_order(price=limit, size=size, timestamp=self.sim_time, side=split.side)
            self.manual_order_ids.add(order.order_id)
            fills = self._process(order)
            self._cancel_resting(order.order_id, rec.FROM_YOU)  # a slice does not wait in the book
            self.manual_order_ids.discard(order.order_id)
            split.filled += sum(f.size for f in fills)
            split.cost += sum(f.price * f.size for f in fills)
        if split.sent >= split.slices or split.filled >= split.total:
            self._finish_split(split, "done")

    def _finish_split(self, split: SplitOrder, how: str) -> None:
        split.finished = how
        buying = split.side == Side.BID
        text = (f"Split {'buy' if buying else 'sell'} {how}: {'bought' if buying else 'sold'} "
                f"{split.filled:,} of {split.total:,} lots in {split.sent} order{'s' if split.sent != 1 else ''}")
        average = split.average
        if average is not None:
            text += f", at an average of {average:.2f} over {self.sim_time - split.start_time:.1f} time units."
            if split.sweep_avg is not None:
                gain = (split.sweep_avg - average) if buying else (average - split.sweep_avg)
                pct = gain / split.sweep_avg * 100
                verdict = "the same" if abs(pct) < 0.05 else f"{abs(pct):.1f}% {'better' if pct > 0 else 'worse'}"
                text += (f" One market order at the start would have {'bought' if buying else 'sold'} "
                         f"{split.sweep_filled:,} at an average of {split.sweep_avg:.2f}: the split did {verdict}.")
            volume = self._cum_volume - split.volume0
            if volume:
                text += f" The market's VWAP over the period was {(self._cum_turnover - split.turnover0) / volume:.2f}."
        else:
            text += "; nothing filled."
        self.split_history.append(text)
        self._notify(text)

    def _record_trade(self, fill: Fill, aggressive_side: Side) -> None:
        self.recent_trades.appendleft(
            TradeRecord(
                timestamp=fill.timestamp,
                price=fill.price,
                size=fill.size,
                maker_order_id=fill.maker_order_id,
                taker_order_id=fill.taker_order_id,
                side=aggressive_side.value,
            )
        )
        log = self._trade_log
        log["time"].append(fill.timestamp)
        log["price"].append(fill.price)
        log["size"].append(fill.size)
        log["aggressor"].append(1 if aggressive_side == Side.BID else 0)
        log["maker_order_id"].append(fill.maker_order_id)
        log["taker_order_id"].append(fill.taker_order_id)
        mine = self.manual_order_ids
        log["yours"].append(fill.maker_order_id in mine or fill.taker_order_id in mine)
        if len(log["time"]) > 2 * MAX_TRADE_LOG:
            for column in log.values():
                del column[: len(column) - MAX_TRADE_LOG]

        if self.collecting is None:  # the heatmap's latest column: lots traded at each price
            traded = self._heat_columns[-1].traded
            traded[fill.price] = traded.get(fill.price, 0) + fill.size
            self.heat_changes += 1

        self.traded_volume += fill.size
        self._cum_volume += fill.size
        self._cum_turnover += fill.price * fill.size

        # The tape of fills, stamped with the clock (a fill's own timestamp is its order's, which lags it in a
        # playback after the user acted), and the studies' CVD, which count whatever the studies are doing.
        buying = aggressive_side == Side.BID
        self.fill_seq += 1
        self.tape_ring.append((self.fill_seq, self.sim_time, fill.price, fill.size, 1 if buying else -1))
        studies = self.studies
        if studies is not None:
            studies.cvd.add(fill.size if buying else -fill.size)

        index = math.floor(fill.timestamp)
        price = float(fill.price)
        buy_volume = fill.size if buying else 0
        bars = self._base_bars
        if bars and bars[-1].index >= index:
            bar = bars[-1]
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)
            bar.close = price
            bar.volume += fill.size
            bar.buy_volume += buy_volume
        else:
            if len(bars) == bars.maxlen:
                self._bars_truncated = True
            bar = BaseBar(index, price, price, price, price, fill.size, buy_volume, cells={})
            bars.append(bar)
            if len(bars) > FOOTPRINT_UNITS:  # the bar FOOTPRINT_UNITS back lets go of its cells
                bars[-1 - FOOTPRINT_UNITS].cells = None
        bar.cum_volume, bar.cum_turnover = self._cum_volume, self._cum_turnover
        cell = bar.cells.get(fill.price)
        if cell is None:
            cell = bar.cells[fill.price] = [0, 0]
        cell[buying] += fill.size  # [lots sold into the bid, lots bought from the ask]

    def trades_frame(self) -> pd.DataFrame:
        """The session's fills for export (the most recent MAX_TRADE_LOG), one row per fill.

        ``aggressor`` is the side that initiated the trade; ``yours`` marks fills of the
        user's orders, as maker or taker.
        """
        log = self._trade_log
        column = {name: np.frombuffer(values, dtype=values.typecode)[-MAX_TRADE_LOG:].copy()
                  for name, values in log.items()}
        return pd.DataFrame({
            "time": column["time"],
            "price": column["price"],
            "size": column["size"],
            "aggressor": np.where(column["aggressor"] == 1, "buy", "sell"),
            "maker_order_id": column["maker_order_id"],
            "taker_order_id": column["taker_order_id"],
            "yours": column["yours"].astype(bool),
        })

    def bars_frame(self) -> pd.DataFrame:
        """Every stored bar at the current interval, with its session VWAP, for export."""
        n = self.bar_interval
        last = math.floor(self.sim_time / n)
        first = self._base_bars[0].index // n if self._bars_truncated and self._base_bars else 0
        candles = self.get_candles(limit=min(last - first + 1, MAX_EXPORT_BARS))
        return pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume",
                                              "buy_volume", "sell_volume", "vwap"])

    def session_vwap(self) -> Optional[float]:
        """Volume-weighted average price of every trade this session (None before the first)."""
        return self._cum_turnover / self._cum_volume if self._cum_volume else None

    # ---- Candlesticks ----

    def _chart_buckets(self, limit: int) -> tuple[int, int]:
        """The chart's first and last bar at the current interval.

        At most ``limit`` bars, and none before the oldest stored trade.
        """
        n = self.bar_interval
        last_bucket = math.floor(self.sim_time / n)
        first_bucket = max(0, last_bucket - limit + 1)
        if self._bars_truncated and self._base_bars:
            first_bucket = max(first_bucket, self._base_bars[0].index // n)
        return first_bucket, last_bucket

    def own_trade_marks(self, limit: int = MAX_CHART_BARS) -> list[dict[str, Any]]:
        """The user's fills per chart bar at the current interval, for the markers on the chart.

        One entry per bar in which they traded: lots bought and sold (as maker or taker),
        each with its average price (None when nothing was bought or sold), and the
        account's position and cash after the bar, for the P&L line. If they traded before
        the chart's first bar, an entry at that bar with nothing bought or sold carries the
        position and cash over.
        """
        n = self.bar_interval
        first_bucket, _ = self._chart_buckets(limit)
        totals: dict[int, list[float]] = {}
        carry: Optional[OwnBar] = None
        for bar in reversed(self._own_bars):
            bucket = bar.index // n
            if bucket < first_bucket:
                carry = bar
                break
            entry = totals.setdefault(bucket, [0, 0.0, 0, 0.0, bar.position, bar.cash])  # the bucket's last bar
            entry[0] += bar.bought
            entry[1] += bar.buy_cost
            entry[2] += bar.sold
            entry[3] += bar.sell_cost
        if carry is not None and first_bucket not in totals:
            totals[first_bucket] = [0, 0.0, 0, 0.0, carry.position, carry.cash]
        return [{
            "time": bucket * n,
            "bought": int(bought), "buy_avg": buy_cost / bought if bought else None,
            "sold": int(sold), "sell_avg": sell_cost / sold if sold else None,
            "position": position, "cash": cash,
        } for bucket, (bought, buy_cost, sold, sell_cost, position, cash) in sorted(totals.items())]

    def get_candles(self, since: Optional[int] = None, limit: int = MAX_CHART_BARS,
                    footprint: bool = False, reveal: bool = False) -> list[dict[str, Any]]:
        """OHLCV bars at the current interval, built from executed trades.

        Bars run from the start of the session (or the oldest stored trade) up
        to the interval containing the current simulation time. Intervals
        without trades are drawn flat at the previous close (the mid-price
        before the first trade) with zero volume.

        Args:
            since: Only return bars starting at or after this time.
            limit: Return at most this many of the most recent bars.
            footprint: Add each bar's ``cells``, [price, lots sold into the bid, lots bought from the ask] by
                rising price, which sum to the bar's ``sell_volume`` and ``buy_volume``. Only the last
                FOOTPRINT_BARS bars have them, and ``None`` stands for a bar with trades older than the
                bars that keep theirs (FOOTPRINT_UNITS).
            reveal: Add each bar's ``fair``, the informed-traders model's hidden value at the last unit of the bar
                sampled (None where there is no sample: before the model began, or for another model).
        """
        n = self.bar_interval
        first_bucket, last_bucket = self._chart_buckets(limit)
        if since is not None:
            first_bucket = max(first_bucket, math.floor(since / n))

        # Newest stored bars back to the first requested bucket, and the close and VWAP before them.
        recent: list[BaseBar] = []
        carry = self._reference_price
        carry_vwap: Optional[float] = None
        for bar in reversed(self._base_bars):
            if bar.index // n < first_bucket:
                carry = bar.close
                carry_vwap = bar.cum_turnover / bar.cum_volume
                break
            recent.append(bar)
        recent.reverse()

        candles = []
        i = 0
        for bucket in range(first_bucket, last_bucket + 1):
            o = hi = lo = c = None
            volume = buy_volume = 0
            cells: Optional[dict[int, list[int]]] = {} if footprint and bucket > last_bucket - FOOTPRINT_BARS else None
            while i < len(recent) and recent[i].index // n == bucket:
                bar = recent[i]
                if o is None:
                    o, hi, lo = bar.open, bar.high, bar.low
                else:
                    hi, lo = max(hi, bar.high), min(lo, bar.low)
                c = bar.close
                volume += bar.volume
                buy_volume += bar.buy_volume
                carry_vwap = bar.cum_turnover / bar.cum_volume
                if cells is not None:
                    if bar.cells is None:
                        cells = None
                    else:
                        for price, (sold, bought) in bar.cells.items():
                            cell = cells.setdefault(price, [0, 0])
                            cell[0] += sold
                            cell[1] += bought
                i += 1
            if o is None:
                o = hi = lo = c = carry
            candle = {
                "time": bucket * n,
                "open": o, "high": hi, "low": lo, "close": c,
                "volume": volume,
                "buy_volume": buy_volume,
                "sell_volume": volume - buy_volume,
                "vwap": carry_vwap,  # the session's VWAP at the bar's close (None before the first trade)
            }
            if footprint:
                candle["cells"] = None if cells is None else [[p, sold, bought] for p, (sold, bought) in sorted(
                    cells.items())]
            if reveal:
                j = min(bucket * n + n - 1, self._value_last) - (self._value_last - len(self._value_samples) + 1)
                candle["fair"] = self._value_samples[j] if 0 <= j < len(self._value_samples) else None
            candles.append(candle)
            carry = c
        return candles

    # ---- Subscriber Notification / SSE ----

    def subscribe(self, q: Optional[asyncio.Queue] = None) -> asyncio.Queue:
        """A queue that wakes on news: a new one, or one a stream already waits on (the Pairs view's)."""
        q = q if q is not None else asyncio.Queue(maxsize=10)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    @property
    def has_subscribers(self) -> bool:
        """Whether any event stream is listening."""
        return bool(self._subscribers)

    async def broadcast(self) -> None:
        """Notify all active subscribers to push state."""
        for q in list(self._subscribers):
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(True)
            except Exception:
                pass

    # ---- Background Worker Loop ----

    async def start_sim_loop(self) -> None:
        self.is_running = True
        if self._bg_task is None or self._bg_task.done():
            self._bg_task = asyncio.create_task(self._run_loop())
        await self.broadcast()

    def shutdown(self) -> None:
        """Stop the background loop without waiting for it (used when evicting a session)."""
        self.is_running = False
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
        self._bg_task = None
        self._subscribers.clear()
        if self.pairs is not None:
            self.pairs.shutdown()
        if self.statarb is not None:
            self.statarb.shutdown()

    async def stop_sim_loop(self) -> None:
        self.is_running = False
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
            self._bg_task = None
        await self.broadcast()

    async def collect(self, n_events: int, chunk: int = 5_000) -> bool:
        """Run ``n_events`` model steps as fast as possible, to gather a stylized-facts sample.

        The simulation is paused first. Steps run in chunks, releasing the event loop
        between them so other requests and sessions are served, and subscribers are told
        about progress about five times a second. Stops early if the sample is reset (new
        model, parameters, engine or book). Afterwards every chart reloads, since the run
        moved the clock far ahead. Returns False if a collection was already running.
        """
        if self.collecting is not None:
            return False
        await self.stop_sim_loop()
        generation = self.facts.generation
        self.collecting = (0, n_events)
        last_notice = time.monotonic()
        try:
            done = 0
            while done < n_events and self.facts.generation == generation:
                k = min(chunk, n_events - done)
                async with self._lock:
                    self.step(k)
                done += k
                self.collecting = (done, n_events)
                if time.monotonic() - last_notice > 0.2:
                    await self.broadcast()
                    last_notice = time.monotonic()
                await asyncio.sleep(0)
        finally:
            self.collecting = None
            self.chart_version += 1
            self._restart_heatmap()  # the run moved the clock far past the old columns
            self._restart_studies()
            await self.broadcast()
        return True

    async def _run_loop(self) -> None:
        try:
            while self.is_running:
                if self.speed <= 20:
                    delay = 1.0 / self.speed
                    async with self._lock:
                        self.step(1)
                    await self.broadcast()
                    await asyncio.sleep(delay)
                else:
                    # At high speeds (up to 2000 eps), batch simulation steps into ~20 FPS frames.
                    # This guarantees the server event loop and UI remain 100% responsive to pause and clicks.
                    frame_dt = 0.05
                    steps_per_frame = max(1, int(round(self.speed * frame_dt)))
                    async with self._lock:
                        self.step(steps_per_frame)
                    await self.broadcast()
                    await asyncio.sleep(frame_dt)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in sim loop: {e}")
        finally:
            self.is_running = False
