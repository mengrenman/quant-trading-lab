"""State and simulation management for the interactive LOB web application."""

from __future__ import annotations

import asyncio
import math
import secrets
import time
from array import array
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from lob.book import OrderBook
from lob.order import Order, OrderType, Fill, Side, limit_order
from lob.matching import MatchingEngine, PriceTimeEngine, ProRataEngine
from app.facts import FactsRecorder
from app.model_specs import MODEL_SPECS, SEED_RANGE

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
class OwnBar:
    """The user's fills in one unit of simulation time, [index, index + 1), as maker or taker."""
    index: int
    bought: int = 0
    buy_cost: float = 0.0
    sold: int = 0
    sell_cost: float = 0.0


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
        self.notice_seq = 0  # never reset, so a tab can tell which notices it has shown
        self.own_trade_seq = 0  # counts the user's fills (never reset), so a tab knows when its markers are stale
        self._split_seq = 0


        self.reset(mid=100)

    def reset(self, mid: int = 100) -> None:
        """Start a fresh session: new book, model, clock, trade history and chart.

        The random generator restarts from ``seed``, so the same seed and
        parameters reproduce a run exactly.
        """
        self.rng = np.random.default_rng(self.seed)
        self.book = OrderBook(tick_size=1, lot_size=1)
        self.sim_time: float = 0.0
        self.event_count: int = 0
        self.traded_volume: int = 0
        self.recent_trades: deque[TradeRecord] = deque(maxlen=40)
        self.own_fills: deque[OwnFill] = deque(maxlen=20)  # fills of the user's resting orders
        self.notices: deque[Notice] = deque(maxlen=20)
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
        self._cum_volume = 0  # every lot traded this session, for the VWAP
        self._cum_turnover = 0.0
        self._bars_truncated = False
        # Price used for bars before the first trade.
        self._reference_price = float(mid)
        self._reset_model(mid)
        self.seed_liquidity(mid=mid)
        self.facts.reset()
        self.chart_version += 1

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
        self.config_version += 1
        return {}

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
        self.sim_time += MANUAL_ACTION_DT
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
        res = self.book.cancel_order(order_id)
        if res is not None:
            self.event_count += 1
            self.facts.observe(OrderType.CANCEL, res.side, [], self.book)
        return res

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
        self.sim_time += MANUAL_ACTION_DT
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
            self.sim_time += MANUAL_ACTION_DT
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
        if self.book.get_order(order_id) is None:
            self.manual_order_ids.discard(order_id)
            return None
        self.sim_time += MANUAL_ACTION_DT
        return self._withdraw(order_id)

    def step(self, n_events: int = 1) -> None:
        """Perform n simulation steps from the selected model."""
        if self.model_type == "manual":
            return

        for _ in range(n_events):
            order = self.model.generate_event(self.book, self.sim_time, self.rng)
            if order is None:
                self.facts.observe_idle()
                continue
            self.sim_time = max(self.sim_time, order.timestamp)
            if order.order_type == OrderType.CANCEL and order.order_id in self.manual_order_ids:
                # A model's cancellations stand for other traders withdrawing their own
                # orders, so the user's orders are left alone. The models pick the order
                # to cancel uniformly from the book, so skipping the user's is exact
                # thinning: the model's own orders keep their cancellation rate.
                self.facts.observe_idle()
                continue
            self._process(order)
            if self.split_orders:
                self._run_split_orders()

        # Forget manual orders that have been filled or canceled.
        self.manual_order_ids = {oid for oid in self.manual_order_ids if self.book.get_order(oid) is not None}

    def _process(self, order: Order) -> list[Fill]:
        fills = self.engine.process(self.book, order)
        self.event_count += 1
        self.facts.observe(order.order_type, order.side, fills, self.book)
        mine = self.manual_order_ids
        maker_side = Side.ASK if order.side == Side.BID else Side.BID
        for fill in fills:
            self._record_trade(fill, order.side)
            # A fill can be the user's on both sides: a market sell against their own bid.
            if fill.taker_order_id in mine:
                self._record_own_trade(fill, order.side)
            if fill.maker_order_id in mine:
                self._record_own_trade(fill, maker_side)
                self._record_own_fill(fill, maker_side=maker_side)
        return fills

    def _record_own_trade(self, fill: Fill, side: Side) -> None:
        """Add a fill of the user's (``side`` is theirs) to the totals per unit of time behind the chart's markers."""
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
        self.own_trade_seq += 1

    def _record_own_fill(self, fill: Fill, maker_side: Side) -> None:
        resting = self.book.get_order(fill.maker_order_id)
        own = OwnFill(order_id=fill.maker_order_id, side=maker_side.value, price=fill.price, size=fill.size,
                      left=resting.size + resting.hidden if resting is not None else 0, time=fill.timestamp)
        self.own_fills.append(own)
        self._notify(own.describe())

    def _notify(self, text: str) -> None:
        self.notice_seq += 1
        self.notices.append(Notice(self.notice_seq, text))

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
        self.sim_time += MANUAL_ACTION_DT
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
            rest = self.book.cancel_order(order.order_id)  # a slice does not wait in the book
            if rest is not None:
                self.event_count += 1
                self.facts.observe(OrderType.CANCEL, rest.side, [], self.book)
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
        each with its average price (None when nothing was bought or sold).
        """
        n = self.bar_interval
        first_bucket, _ = self._chart_buckets(limit)
        totals: dict[int, list[float]] = {}
        for bar in reversed(self._own_bars):
            bucket = bar.index // n
            if bucket < first_bucket:
                break
            entry = totals.setdefault(bucket, [0, 0.0, 0, 0.0])
            entry[0] += bar.bought
            entry[1] += bar.buy_cost
            entry[2] += bar.sold
            entry[3] += bar.sell_cost
        return [{
            "time": bucket * n,
            "bought": int(bought), "buy_avg": buy_cost / bought if bought else None,
            "sold": int(sold), "sell_avg": sell_cost / sold if sold else None,
        } for bucket, (bought, buy_cost, sold, sell_cost) in sorted(totals.items())]

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
