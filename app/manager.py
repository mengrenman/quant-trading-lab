"""State and simulation management for the interactive LOB web application."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass, field
import time
from typing import Any, Literal, Optional

import numpy as np

from lob.book import OrderBook
from lob.order import Order, Fill, Side, OrderType, limit_order, market_order, cancel_order
from lob.matching import MatchingEngine, PriceTimeEngine, ProRataEngine
from lob.models.zero_intelligence import SmithFarmerModel
from lob.models.mike_farmer import MikeFarmerModel
from lob.models.strategic import ParlourModel


@dataclass
class TradeRecord:
    timestamp: float
    price: int
    size: int
    maker_order_id: int
    taker_order_id: int
    side: str  # aggressive side: buy or sell


@dataclass
class Candle:
    time: int        # epoch timestamp in seconds (strictly increasing)
    open: float
    high: float
    low: float
    close: float
    volume: int


class LOBManager:
    """Central manager coordinating the LOB, matching engine, and simulation runner."""

    def __init__(self):
        self.book = OrderBook(tick_size=1, lot_size=1)
        self.engine_type: Literal["price_time", "pro_rata"] = "price_time"
        self.engine: MatchingEngine = PriceTimeEngine()

        self.model_type: Literal["smith_zi", "mike_farmer", "parlour", "manual"] = "smith_zi"
        self.model = SmithFarmerModel(mu=6.0, lam=1.0, delta=0.03, price_range=15, lot_size=5, initial_mid=100)
        self.rng = np.random.default_rng(42)

        self.sim_time: float = 0.0
        self.event_count: int = 0
        self.is_running: bool = False
        self.speed: int = 5  # events per second (1 to 50)

        self.recent_trades: deque[TradeRecord] = deque(maxlen=40)
        self.manual_order_ids: set[int] = set()

        # Candlestick chart state
        self.bar_interval: int = 2  # bar duration in seconds
        self.candle_clock: float = 0.0
        self.chart_reset_pending: bool = False
        self.start_epoch: int = int(time.time()) - (40 * self.bar_interval)
        self.candles: deque[Candle] = deque(maxlen=500)
        self._init_candles(mid=100.0)

        self._subscribers: set[asyncio.Queue] = set()
        self._bg_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Seed initial liquidity with deep levels
        self.seed_liquidity(mid=100, half_spread=1, levels=16, volume_per_level=25)

    def _init_candles(self, mid: float = 100.0) -> None:
        """Seed pre-computed historical candles leading up to the current sim state."""
        self.candles.clear()
        self.candle_clock = 0.0
        self.start_epoch = int(time.time()) - (40 * self.bar_interval)
        curr = mid
        rng = np.random.default_rng(123)

        # Generate 40 realistic historical candles around mid
        for i in range(40):
            t = self.start_epoch + i * self.bar_interval
            delta = int(rng.integers(-2, 3))
            o = curr
            c = max(10.0, curr + delta)
            h = max(o, c) + int(rng.integers(0, 2))
            l = min(o, c) - int(rng.integers(0, 2))
            v = int(rng.integers(5, 35))
            self.candles.append(Candle(time=t, open=float(o), high=float(h), low=float(l), close=float(c), volume=v))
            curr = c

        # Append initial open candle at start
        cur_t = self.start_epoch + 40 * self.bar_interval
        self.candles.append(Candle(time=cur_t, open=float(mid), high=float(mid), low=float(mid), close=float(mid), volume=0))

    def _update_candles(self, price: Optional[float], size: int = 0, dt: float = 0.0) -> None:
        """Update current active candlestick or create a new one as candle_clock advances."""
        if price is None:
            if self.candles:
                price = self.candles[-1].close
            else:
                price = 100.0
        else:
            price = float(price)

        self.candle_clock += dt
        bucket_idx = int(self.candle_clock // self.bar_interval)
        cur_time = self.start_epoch + (40 * self.bar_interval) + (bucket_idx * self.bar_interval)

        if not self.candles:
            self.candles.append(Candle(
                time=cur_time,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=size,
            ))
            return

        last = self.candles[-1]
        if cur_time > last.time:
            # Fill skipped intervals if any
            step_t = last.time + self.bar_interval
            while step_t < cur_time:
                self.candles.append(Candle(
                    time=step_t,
                    open=last.close,
                    high=last.close,
                    low=last.close,
                    close=last.close,
                    volume=0,
                ))
                step_t += self.bar_interval
            # New bar
            self.candles.append(Candle(
                time=cur_time,
                open=last.close,
                high=max(last.close, price),
                low=min(last.close, price),
                close=price,
                volume=size,
            ))
        else:
            # Update existing bar
            last.high = max(last.high, price)
            last.low = min(last.low, price)
            last.close = price
            last.volume += size

    def get_candlesticks(self) -> list[dict[str, Any]]:
        return [asdict(c) for c in self.candles]

    def get_latest_candle(self) -> Optional[dict[str, Any]]:
        if not self.candles:
            return None
        return asdict(self.candles[-1])

    def set_bar_interval(self, interval: int) -> None:
        self.bar_interval = max(1, interval)
        mid = float(self.book.mid_price or 100.0)
        self.chart_reset_pending = True
        self._init_candles(mid=mid)

    def seed_liquidity(self, mid: int = 100, half_spread: int = 1, levels: int = 16, volume_per_level: int = 25) -> None:
        """Seed the order book with deep symmetric bids and asks around mid."""
        for i in range(levels):
            bid_p = mid - half_spread - i
            ask_p = mid + half_spread + i
            vol = volume_per_level + (i * 5)
            if bid_p > 0:
                self.book.add_order(limit_order(bid_p, vol, self.sim_time, Side.BID))
            self.book.add_order(limit_order(ask_p, vol, self.sim_time, Side.ASK))

    def reset(self, mid: int = 100) -> None:
        """Clear book, cancel all orders, and re-seed."""
        self.book = OrderBook(tick_size=1, lot_size=1)
        self.recent_trades.clear()
        self.manual_order_ids.clear()
        self.sim_time = 0.0
        self.event_count = 0
        self.candle_clock = 0.0
        self.chart_reset_pending = True
        self._reset_model(mid)
        self.seed_liquidity(mid=mid)
        self._init_candles(mid=float(mid))

    def _reset_model(self, mid: int = 100) -> None:
        if self.model_type == "smith_zi":
            self.model = SmithFarmerModel(mu=6.0, lam=1.0, delta=0.03, price_range=15, lot_size=5, initial_mid=mid)
        elif self.model_type == "mike_farmer":
            self.model = MikeFarmerModel(mu=2.5, lam=8.0, delta=0.015, size_mean=2.2, initial_mid=mid)
        elif self.model_type == "parlour":
            self.model = ParlourModel(arrival_rate=12.0, delta=0.015, lot_size=5, initial_mid=mid)

    def set_engine(self, engine_name: Literal["price_time", "pro_rata"]) -> None:
        self.engine_type = engine_name
        if engine_name == "price_time":
            self.engine = PriceTimeEngine()
        else:
            self.engine = ProRataEngine()

    def set_model(self, model_name: Literal["smith_zi", "mike_farmer", "parlour", "manual"]) -> None:
        self.model_type = model_name
        mid = int(self.book.mid_price or 100)
        self._reset_model(mid)

    def set_speed(self, speed: int) -> None:
        self.speed = max(1, min(2000, int(speed)))

    def submit_limit_order(self, price: int, size: int, side: Side, is_manual: bool = True) -> list[Fill]:
        if is_manual:
            self.sim_time += 0.5
        order = limit_order(price=price, size=size, timestamp=self.sim_time, side=side)
        if is_manual:
            self.manual_order_ids.add(order.order_id)
        fills = self.engine.process(self.book, order)
        self._record_fills(fills, side)
        self.event_count += 1

        dt = 0.5 if is_manual else 0.0
        if fills:
            for idx, f in enumerate(fills):
                self._update_candles(f.price, f.size, dt=(dt if idx == 0 else 0.0))
        else:
            self._update_candles(self.book.mid_price, 0, dt=dt)
        return fills

    def submit_market_order(self, size: int, side: Side, is_manual: bool = True) -> list[Fill]:
        if is_manual:
            self.sim_time += 0.5
        order = market_order(size=size, timestamp=self.sim_time, side=side)
        if is_manual:
            self.manual_order_ids.add(order.order_id)
        fills = self.engine.process(self.book, order)
        self._record_fills(fills, side)
        self.event_count += 1

        dt = 0.5 if is_manual else 0.0
        if fills:
            for idx, f in enumerate(fills):
                self._update_candles(f.price, f.size, dt=(dt if idx == 0 else 0.0))
        else:
            self._update_candles(self.book.mid_price, 0, dt=dt)
        return fills

    def cancel_order_by_id(self, order_id: int) -> Optional[Order]:
        self.manual_order_ids.discard(order_id)
        res = self.book.cancel_order(order_id)
        if res is not None:
            self.event_count += 1
            self._update_candles(self.book.mid_price, 0, dt=0.5)
        return res

    def step(self, n_events: int = 1) -> None:
        """Perform n simulation steps from the selected model."""
        if self.model_type == "manual":
            return

        for _ in range(n_events):
            order = self.model.generate_event(self.book, self.sim_time, self.rng)
            if order is None:
                continue

            self.sim_time = order.timestamp
            fills = self.engine.process(self.book, order)
            self._record_fills(fills, order.side)
            self.event_count += 1

            # Advance candle clock by 0.2s per simulation event
            if fills:
                for idx, f in enumerate(fills):
                    self._update_candles(f.price, f.size, dt=(0.2 if idx == 0 else 0.0))
            else:
                self._update_candles(self.book.mid_price, 0, dt=0.2)

            # Clean manual order IDs if filled or cancelled
            if order.order_type == OrderType.CANCEL:
                self.manual_order_ids.discard(order.order_id)

        # Ensure manual order IDs only retain active resting orders
        active_ids = set(self.book._orders.keys())
        self.manual_order_ids.intersection_update(active_ids)

    def _record_fills(self, fills: list[Fill], aggressive_side: Side) -> None:
        for f in fills:
            self.recent_trades.appendleft(
                TradeRecord(
                    timestamp=f.timestamp,
                    price=f.price,
                    size=f.size,
                    maker_order_id=f.maker_order_id,
                    taker_order_id=f.taker_order_id,
                    side=aggressive_side.value,
                )
            )

    # ---- Subscriber Notification / SSE ----

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

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


# Global singleton instance for the app
manager = LOBManager()
