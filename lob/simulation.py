"""Event-driven simulation harness for LOB models.

Provides a generic simulation loop that accepts any model implementing
the `generate_event` interface. Records a full event log that can be
converted to a pandas DataFrame for analysis in notebooks.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm

from lob.book import OrderBook
from lob.order import Order, Side, OrderType
from lob.matching import MatchingEngine, PriceTimeEngine


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
        self._event_log: dict[str, list] = {
            "time": [],
            "event_type": [],
            "side": [],
            "price": [],
            "size": [],
            "order_id": [],
            "n_fills": [],
            "fill_volume": [],
            "fill_prices": [],
            "best_bid": [],
            "best_ask": [],
            "mid_price": [],
            "spread": [],
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
        log["event_type"].append(order.order_type.value)
        log["side"].append(order.side.value)
        log["price"].append(order.price)
        log["size"].append(orig_size)
        log["order_id"].append(order.order_id)

        if fills:
            log["n_fills"].append(len(fills))
            log["fill_volume"].append(sum(f.size for f in fills))
            log["fill_prices"].append([f.price for f in fills])
        else:
            log["n_fills"].append(0)
            log["fill_volume"].append(0)
            log["fill_prices"].append([])

        b = self.book.best_bid
        a = self.book.best_ask
        log["best_bid"].append(b)
        log["best_ask"].append(a)
        if b is not None and a is not None:
            log["mid_price"].append((a + b) / 2.0)
            log["spread"].append(a - b)
        else:
            log["mid_price"].append(None)
            log["spread"].append(None)

    def _build_dataframe(self) -> pd.DataFrame:
        """Convert event log to DataFrame."""
        if not self._event_log["time"]:
            return pd.DataFrame()
        return pd.DataFrame(self._event_log)

    @property
    def snapshots(self) -> list[dict]:
        """Book snapshots recorded during simulation."""
        return self._snapshots
