"""Event-driven simulation harness for LOB models.

Provides a generic simulation loop that accepts any model implementing
the `generate_event` interface. Records a full event log that can be
converted to a pandas DataFrame for analysis in notebooks.
"""

from array import array
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm

from lob.book import OrderBook
from lob.order import Order, OrderType, Side
from lob.matching import MatchingEngine, PriceTimeEngine

# Event types and sides are stored as small integer codes and decoded when
# the DataFrame is built.
_EVENT_TYPES = tuple(t.value for t in OrderType)
_EVENT_CODES = {t: i for i, t in enumerate(OrderType)}
_SIDES = tuple(s.value for s in Side)
_SIDE_CODES = {s: i for i, s in enumerate(Side)}
_NO_FILLS = ()


class LOBModel(Protocol):
    """Interface that all simulation models must implement."""

    def generate_event(self, book: OrderBook, t: float,
                       rng: np.random.Generator) -> Optional[Order]:
        """Generate the next order event given current book state and time.

        Returns an Order to be processed, or None to skip (no event).
        """
        ...


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    n_events: int = 100_000
    seed: int = 42
    warmup_events: int = 1_000
    record_book_snapshots: bool = False
    snapshot_interval: int = 100


class Simulation:
    """Run an LOB simulation with a given model and matching engine.

    Usage:
        sim = Simulation(model=my_model, config=SimulationConfig(n_events=50000))
        results = sim.run()
    """

    def __init__(
        self,
        model: LOBModel,
        book: Optional[OrderBook] = None,
        engine: Optional[MatchingEngine] = None,
        config: Optional[SimulationConfig] = None,
    ):
        self.model = model
        self.book = book or OrderBook()
        self.engine = engine or PriceTimeEngine()
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        # Typed arrays rather than lists of Python objects: about a quarter of
        # the memory, which matters for million-event runs and parallel seeds.
        # Missing quotes (an empty side) are stored as NaN.
        self._event_log: dict[str, array] = {
            "time": array("d"),
            "event_type": array("b"),
            "side": array("b"),
            "price": array("q"),
            "size": array("q"),
            "order_id": array("q"),
            "n_fills": array("q"),
            "fill_volume": array("q"),
            "best_bid": array("d"),
            "best_ask": array("d"),
            "mid_price": array("d"),
            "spread": array("d"),
        }
        self._fill_prices: list[tuple] = []
        # One row per fill, so per-trade quantities (sizes, VWAP) are available.
        self._trade_log: dict[str, list] = {
            "time": [],
            "price": [],
            "size": [],
            "aggressor": [],
            "maker_order_id": [],
            "taker_order_id": [],
        }
        self._snapshots: list[dict] = []

    def run(self, progress: bool = True) -> pd.DataFrame:
        """Run the full simulation and return the event log as a DataFrame."""
        total = self.config.n_events + self.config.warmup_events
        iterator = range(total)
        if progress:
            iterator = tqdm(iterator, desc="Simulating", total=total)

        t = 0.0
        for i in iterator:
            order = self.model.generate_event(self.book, t, self.rng)
            if order is None:
                continue

            t = order.timestamp
            orig_size = order.size if order.order_type != OrderType.CANCEL else 0

            # Process through matching engine.
            fills = self.engine.process(self.book, order)

            # Record after warmup.
            if i >= self.config.warmup_events:
                self._record_event(order, orig_size, fills)

                if (self.config.record_book_snapshots and
                        (i - self.config.warmup_events) % self.config.snapshot_interval == 0):
                    snap = self.book.snapshot()
                    snap["event_index"] = i - self.config.warmup_events
                    snap["time"] = t
                    self._snapshots.append(snap)

        return self._build_dataframe()

    def _record_event(self, order: Order, orig_size: int, fills: list) -> None:
        """Record a single event in the columnar log."""
        log = self._event_log
        log["time"].append(order.timestamp)
        log["event_type"].append(_EVENT_CODES[order.order_type])
        log["side"].append(_SIDE_CODES[order.side])
        log["price"].append(order.price)
        log["size"].append(orig_size)
        log["order_id"].append(order.order_id)

        if fills:
            log["n_fills"].append(len(fills))
            log["fill_volume"].append(sum(f.size for f in fills))
            self._fill_prices.append(tuple(f.price for f in fills))
            trades = self._trade_log
            for f in fills:
                trades["time"].append(f.timestamp)
                trades["price"].append(f.price)
                trades["size"].append(f.size)
                trades["aggressor"].append(order.side.value)
                trades["maker_order_id"].append(f.maker_order_id)
                trades["taker_order_id"].append(f.taker_order_id)
        else:
            log["n_fills"].append(0)
            log["fill_volume"].append(0)
            self._fill_prices.append(_NO_FILLS)

        nan = float("nan")
        b = self.book.best_bid
        a = self.book.best_ask
        log["best_bid"].append(nan if b is None else b)
        log["best_ask"].append(nan if a is None else a)
        if b is not None and a is not None:
            log["mid_price"].append((a + b) / 2.0)
            log["spread"].append(a - b)
        else:
            log["mid_price"].append(nan)
            log["spread"].append(nan)

    def _build_dataframe(self) -> pd.DataFrame:
        """Convert the event log to a DataFrame.

        Columns: time, event_type, side, price, size, order_id, n_fills,
        fill_volume, fill_prices (a tuple of fill prices per event), best_bid,
        best_ask, mid_price and spread. Quote columns are float with NaN where
        a side of the book was empty, or integers if it never was.
        """
        log = self._event_log
        if not log["time"]:
            return pd.DataFrame()

        def column(name: str) -> np.ndarray:
            return np.frombuffer(log[name], dtype=log[name].typecode).copy()

        def quotes(name: str) -> np.ndarray:
            values = column(name)
            return values if np.isnan(values).any() else values.astype(np.int64)

        return pd.DataFrame({
            "time": column("time"),
            "event_type": np.array(_EVENT_TYPES, dtype=object)[column("event_type")],
            "side": np.array(_SIDES, dtype=object)[column("side")],
            "price": column("price"),
            "size": column("size"),
            "order_id": column("order_id"),
            "n_fills": column("n_fills"),
            "fill_volume": column("fill_volume"),
            "fill_prices": pd.Series(self._fill_prices, dtype=object),
            "best_bid": quotes("best_bid"),
            "best_ask": quotes("best_ask"),
            "mid_price": column("mid_price"),
            "spread": quotes("spread"),
        })

    @property
    def snapshots(self) -> list[dict]:
        """Book snapshots recorded during simulation."""
        return self._snapshots

    @property
    def trades(self) -> pd.DataFrame:
        """Every fill recorded after warm-up, one row per fill.

        Columns: time, price, size, aggressor ("bid" for buyer-initiated
        trades, "ask" for seller-initiated), maker_order_id, taker_order_id.
        """
        return pd.DataFrame(self._trade_log)
