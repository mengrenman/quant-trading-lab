"""State and simulation management for the interactive LOB web application."""

from __future__ import annotations

import asyncio
import math
import secrets
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from lob.book import OrderBook
from lob.order import Order, Fill, Side, limit_order, market_order
from lob.matching import MatchingEngine, PriceTimeEngine, ProRataEngine
from app.model_specs import MODEL_SPECS, SEED_RANGE

EngineName = Literal["price_time", "pro_rata"]
ENGINE_NAMES = ("price_time", "pro_rata")
DEFAULT_SEED = 42

# Simulation time that passes with each manual action (order or cancellation), so
# a session driven only by hand still moves forward on the chart.
MANUAL_ACTION_DT = 0.5
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
class BaseBar:
    """Trades in one unit of simulation time, [index, index + 1)."""
    index: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    buy_volume: int


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
        self.manual_order_ids: set[int] = set()
        self._base_bars: deque[BaseBar] = deque(maxlen=MAX_BASE_BARS)
        self._bars_truncated = False
        # Price used for bars before the first trade.
        self._reference_price = float(mid)
        self._reset_model(mid)
        self.seed_liquidity(mid=mid)
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
        self.config_version += 1

    def toggle_engine(self) -> None:
        self.set_engine("pro_rata" if self.engine_type == "price_time" else "price_time")

    def set_model(self, model_name: str) -> None:
        """Switch models; the new model takes over the current book and clock."""
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model {model_name!r}")
        self.model_type = model_name
        self._reset_model(int(self.book.mid_price or self._reference_price))
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
        self.config_version += 1
        return {}

    def set_speed(self, speed: int) -> None:
        self.speed = max(1, min(2000, int(speed)))
        self.config_version += 1

    def set_bar_interval(self, interval: int) -> None:
        self.bar_interval = max(1, int(interval))
        self.chart_version += 1

    # ---- Order flow ----

    def submit_limit_order(self, price: int, size: int, side: Side) -> tuple[Order, list[Fill]]:
        """Submit a manual limit order; returns the order and its fills on arrival."""
        self.sim_time += MANUAL_ACTION_DT
        order = limit_order(price=price, size=size, timestamp=self.sim_time, side=side)
        self.manual_order_ids.add(order.order_id)
        fills = self._process(order)
        if self.book.get_order(order.order_id) is None:  # filled on arrival
            self.manual_order_ids.discard(order.order_id)
        return order, fills

    def submit_market_order(self, size: int, side: Side) -> tuple[Order, list[Fill]]:
        """Submit a manual market order; returns the order and its fills."""
        self.sim_time += MANUAL_ACTION_DT
        order = market_order(size=size, timestamp=self.sim_time, side=side)
        return order, self._process(order)

    def cancel_order_by_id(self, order_id: int) -> Optional[Order]:
        self.manual_order_ids.discard(order_id)
        res = self.book.cancel_order(order_id)
        if res is not None:
            self.sim_time += MANUAL_ACTION_DT
            self.event_count += 1
        return res

    def step(self, n_events: int = 1) -> None:
        """Perform n simulation steps from the selected model."""
        if self.model_type == "manual":
            return

        for _ in range(n_events):
            order = self.model.generate_event(self.book, self.sim_time, self.rng)
            if order is None:
                continue
            self.sim_time = max(self.sim_time, order.timestamp)
            self._process(order)

        # Forget manual orders that have been filled or canceled.
        self.manual_order_ids = {oid for oid in self.manual_order_ids if self.book.get_order(oid) is not None}

    def _process(self, order: Order) -> list[Fill]:
        fills = self.engine.process(self.book, order)
        self.event_count += 1
        for fill in fills:
            self._record_trade(fill, order.side)
        return fills

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
        self.traded_volume += fill.size

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
            bars.append(BaseBar(index, price, price, price, price, fill.size, buy_volume))

    # ---- Candlesticks ----

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
        last_bucket = math.floor(self.sim_time / n)
        first_bucket = max(0, last_bucket - limit + 1)
        if since is not None:
            first_bucket = max(first_bucket, math.floor(since / n))
        if self._bars_truncated and self._base_bars:
            first_bucket = max(first_bucket, self._base_bars[0].index // n)

        # Newest stored bars back to the first requested bucket, and the close before them.
        recent: list[BaseBar] = []
        carry = self._reference_price
        for bar in reversed(self._base_bars):
            if bar.index // n < first_bucket:
                carry = bar.close
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
                i += 1
            if o is None:
                o = hi = lo = c = carry
            candles.append({
                "time": bucket * n,
                "open": o, "high": hi, "low": lo, "close": c,
                "volume": volume,
                "buy_volume": buy_volume,
                "sell_volume": volume - buy_volume,
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
