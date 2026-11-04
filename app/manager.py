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
from app import recording as rec
from app.facts import FactsRecorder
from app.model_specs import MODEL_SPECS, SEED_RANGE
from app.strategy_specs import MAX_STRATEGIES, StrategySpec
from lob.strategies import Strategy, View

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
# The order-book heatmap: one column per unit of simulation time, this many ticks either side of the mid.
HEAT_HALF = 30
HEAT_COLUMNS = 600


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

    def markout(self, buying: bool, price: float, size: int, mid_later: float) -> None:
        self.markout_lots += size
        self.markout_edge += (mid_later - price if buying else price - mid_later) * size

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


@dataclass
class Playback:
    """A recording being played back: where on its tape, and what its orders and strategies are now."""
    recording: rec.Recording
    cursor: int = 0  # the next event on the tape
    ids: dict[int, int] = field(default_factory=dict)  # a recorded order's id -> its replay's
    strategies: dict[int, RunningStrategy] = field(default_factory=dict)  # a recorded strategy's id -> its replay
    model: str = ""  # the model the recording was made with, at the cursor
    ended: bool = False


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
        self._quiet = False  # no notices (while seeking)
        self.notice_seq = 0  # never reset, so a tab can tell which notices it has shown
        self.notices: deque[Notice] = deque(maxlen=20)
        self.own_trade_seq = 0  # counts the user's fills (never reset), so a tab knows when its markers are stale
        self.heat_version = 0  # incremented when the heatmap starts over (reset, after a collection)
        self.heat_changes = 0  # incremented with each new column and each trade in the latest one
        self._split_seq = 0


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
        self._markouts: deque[tuple[int, Account, bool, float, int]] = deque()  # (due event, account, ...)
        self._cum_volume = 0  # every lot traded this session, for the VWAP
        self._cum_turnover = 0.0
        self._bars_truncated = False
        # Price used for bars before the first trade.
        self._reference_price = reference

    def _reset_model(self, mid: int = 100) -> None:
        spec = MODEL_SPECS[self.model_type]
        self.model = spec.build(self.model_params[spec.key], mid) if spec.factory else None

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

    # ---- Configuration ----

    def set_engine(self, engine_name: EngineName) -> None:
        if engine_name not in ENGINE_NAMES:
            raise ValueError(f"Unknown engine {engine_name!r}")
        self.engine_type = engine_name
        self.engine = PriceTimeEngine() if engine_name == "price_time" else ProRataEngine()
        self.facts.reset()
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
        self.facts.reset()
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
        return self._enter_limit(price, size, side, display_size)

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
        return self._withdraw(order_id)

    def step(self, n_events: int = 1) -> None:
        """Perform n simulation steps from the selected model (in playback, n orders or cancellations from the tape)."""
        if self.playback is not None:
            self._step_playback(n_events)
            return
        if self.model_type == "manual":
            return

        for _ in range(n_events):
            order = self.model.generate_event(self.book, self.sim_time, self.rng)
            if order is None:
                self.facts.observe_idle()
                continue
            self._advance(order.timestamp)
            if order.order_type == OrderType.CANCEL and (order.order_id in self.manual_order_ids
                                                         or order.order_id in self._strategy_orders):
                # A model's cancellations stand for other traders withdrawing their own
                # orders, so the user's orders are left alone. The models pick the order
                # to cancel uniformly from the book, so skipping the user's is exact
                # thinning: the model's own orders keep their cancellation rate.
                self.facts.observe_idle()
                continue
            self._process(order)
            if self.split_orders:
                self._run_split_orders()
            for s in self.strategies:
                if s.running and self.sim_time >= s.next_time:
                    self._run_strategy(s)

        # Forget manual orders that have been filled or canceled.
        self.manual_order_ids = {oid for oid in self.manual_order_ids if self.book.get_order(oid) is not None}

    def _process(self, order: Order) -> list[Fill]:
        mid_before = self.book.mid_price  # what the user's fills are scored against
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
        markouts = self._markouts
        while markouts and markouts[0][0] <= self.event_count:  # fills that are now MARKOUT_EVENTS old
            _, account, buying, price, size = markouts.popleft()
            mid = self.book.mid_price
            if mid is not None:
                account.markout(buying, price, size, mid)
        return fills

    def _book_fill(self, account: Account, fill: Fill, side: Side, maker: bool, mid_before: Optional[float]) -> None:
        """Book a fill in an account, and note it for its markout MARKOUT_EVENTS events from now."""
        account.book(side == Side.BID, fill.price, fill.size, maker, mid_before)
        self._markouts.append((self.event_count + MARKOUT_EVENTS, account, side == Side.BID, fill.price, fill.size))

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
        self.sim_time = t

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
        for oid in list(s.order_ids):
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
        if plan.quotes is not None:  # a market maker: new quotes each time (each joins the back of its queue)
            self._withdraw_strategy_orders(s)
            for side, (price, size) in plan.quotes.items():
                if size > 0 and price >= 1:
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
                    replayed=s.replayed,
                    settings=s.strategy.describe(), signal=s.signal, target=s.target, quotes=dict(s.quotes),
                    warmup=s.strategy.warming_up())
                for s in self.strategies]
        rows.append(row(self.account, self._own_pnl, id=None, name="You", tag="YOU", kind="you", running=None,
                        replayed=False,
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
            "markouts": [[due, accounts[id(account)], buying, price, size]
                         for due, account, buying, price, size in self._markouts if id(account) in accounts],
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
        self._markouts.extend((due, owners[owner], buying, price, size)
                              for due, owner, buying, price, size in start["markouts"] if owner in owners)
        self._reset_model(int(self.book.mid_price or self._reference_price))  # the one that takes over afterwards
        self.facts.reset()
        self.chart_version += 1
        self._restart_heatmap()
        self.recordings_version += 1

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
        as after a model's event. At the end of the tape the simulation stops.
        """
        pb = self.playback
        tape = pb.recording.tape
        done = 0
        while done < n:
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
        (strategies started during it start over). Runs in chunks, as a collection does.
        """
        await self.stop_sim_loop()
        pb = self.playback
        if pb is None:
            return
        # A range input rounds the time it gives (to 15 digits in Chrome): events a hair later count as reached,
        # so the timeline's far end is the end of the tape.
        target = pb.recording.tape.index_after(to_time + 1e-9 * max(1.0, abs(to_time)))
        async with self._lock:
            if target < pb.cursor:
                self.rewind_playback()
        self._quiet = True
        last_notice = time.monotonic()
        try:
            while self.playback is pb and pb.cursor < target and not pb.ended:
                async with self._lock:
                    self._step_playback(min(chunk, target - pb.cursor), every_event=True)
                self.seeking = (pb.cursor, target)
                if time.monotonic() - last_notice > 0.2:
                    await self.broadcast()
                    last_notice = time.monotonic()
                await asyncio.sleep(0)
        finally:
            self._quiet = False
            self.seeking = None
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

        index = math.floor(fill.timestamp)
        price = float(fill.price)
        buy_volume = fill.size if aggressive_side == Side.BID else 0
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
            bar = BaseBar(index, price, price, price, price, fill.size, buy_volume)
            bars.append(bar)
        bar.cum_volume, bar.cum_turnover = self._cum_volume, self._cum_turnover

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

    def get_candles(self, since: Optional[int] = None, limit: int = MAX_CHART_BARS) -> list[dict[str, Any]]:
        """OHLCV bars at the current interval, built from executed trades.

        Bars run from the start of the session (or the oldest stored trade) up
        to the interval containing the current simulation time. Intervals
        without trades are drawn flat at the previous close (the mid-price
        before the first trade) with zero volume.

        Args:
            since: Only return bars starting at or after this time.
            limit: Return at most this many of the most recent bars.
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
                i += 1
            if o is None:
                o = hi = lo = c = carry
            candles.append({
                "time": bucket * n,
                "open": o, "high": hi, "low": lo, "close": c,
                "volume": volume,
                "buy_volume": buy_volume,
                "sell_volume": volume - buy_volume,
                "vwap": carry_vwap,  # the session's VWAP at the bar's close (None before the first trade)
            })
            carry = c
        return candles

    # ---- Subscriber Notification / SSE ----

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
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
