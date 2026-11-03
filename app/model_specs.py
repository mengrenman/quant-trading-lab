"""The models the dashboard can run, with their adjustable parameters.

Defaults are tuned for the dashboard's book (mid-price 100, seeded with
25-100 lot levels); the notebooks use their own values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

from lob.models.hawkes import HawkesZIModel
from lob.models.mike_farmer import MikeFarmer2008Model, MikeFarmerModel
from lob.models.strategic import ParlourModel
from lob.models.zero_intelligence import SmithFarmerModel


@dataclass(frozen=True)
class Param:
    """One adjustable model parameter and its allowed range."""
    name: str
    label: str
    default: float
    low: float
    high: float
    step: float
    integer: bool = False

    def parse(self, raw: Any) -> float | int:
        """Convert form input to a value in range, or raise ValueError with a readable message."""
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{self.label}: enter a number") from None
        if not math.isfinite(value):
            raise ValueError(f"{self.label}: enter a finite number")
        if self.integer:
            if value != int(value):
                raise ValueError(f"{self.label}: enter a whole number")
            value = int(value)
        if not self.low <= value <= self.high:
            raise ValueError(f"{self.label}: must be between {self.low:g} and {self.high:g}")
        return value


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    factory: Optional[Callable[..., Any]]  # None for manual placement
    params: tuple[Param, ...] = ()
    summary: str = ""  # one line for the dashboard
    # The values notebook 05 runs this model with (its mid-price is 10,000; the dashboard's is 100).
    # The dashboard's own defaults trade more, so the page stays lively at its default speed.
    notebook: Optional[dict[str, float | int]] = None

    def defaults(self) -> dict[str, float | int]:
        return {p.name: p.default for p in self.params}

    def build(self, values: dict[str, float | int], initial_mid: int) -> Any:
        return self.factory(initial_mid=initial_mid, **values)


_ZI_PARAMS = (
    Param("mu", "Market order rate μ", 6.0, 0.1, 100.0, 0.1),
    Param("lam", "Limit order rate λ (per tick, per side)", 1.0, 0.05, 20.0, 0.05),
    Param("delta", "Cancellation rate δ (per order)", 0.03, 0.001, 1.0, 0.001),
    Param("price_range", "Placement window L (ticks)", 15, 2, 200, 1, integer=True),
    Param("lot_size", "Order size (lots)", 5, 1, 100, 1, integer=True),
)

MODEL_SPECS: dict[str, ModelSpec] = {spec.key: spec for spec in (
    ModelSpec("smith_zi", "Smith et al. (2003) ZI", SmithFarmerModel, _ZI_PARAMS, summary=(
        "Zero intelligence: orders arrive at random; limit orders land uniformly on the ticks next to the "
        "opposite best quote, and each resting order is canceled at a constant rate."),
        notebook=dict(mu=1.0, lam=0.5, delta=0.1, price_range=50, lot_size=1)),
    ModelSpec("hawkes_zi", "ZI + self-exciting market orders", HawkesZIModel, _ZI_PARAMS + (
        Param("branching", "Branching ratio (orders triggered per market order)", 0.8, 0.0, 0.95, 0.05),
        Param("decay", "Excitation decay rate", 1.0, 0.05, 20.0, 0.05),
    ), summary=(
        "The ZI model with market orders arriving as a self-exciting (Hawkes) process: "
        "the same long-run rate, but clustered in time."),
        notebook=dict(mu=1.0, lam=0.5, delta=0.1, price_range=50, lot_size=1, branching=0.8, decay=0.5)),
    ModelSpec("mike_farmer", "Mike & Farmer (simplified)", MikeFarmerModel, (
        Param("mu", "Market order rate", 2.5, 0.1, 100.0, 0.1),
        Param("lam", "Limit order rate", 8.0, 0.1, 100.0, 0.1),
        Param("delta", "Cancellation rate (per order)", 0.015, 0.001, 1.0, 0.001),
        Param("alpha_price", "Relative-price tail exponent", 1.5, 0.5, 4.0, 0.1),
        Param("hurst", "Order-sign Hurst exponent", 0.7, 0.5, 0.95, 0.05),
        Param("size_mean", "Mean log order size", 2.2, 0.0, 5.0, 0.1),
        Param("size_sigma", "Std of log order size", 0.5, 0.0, 2.0, 0.05),
    ), summary=(
        "Power-law prices measured from the mid, long-memory order signs, log-normal sizes "
        "and constant-rate cancellation."),
        notebook=dict(mu=1.0, lam=2.0, delta=0.05, alpha_price=1.5, hurst=0.7, size_mean=0.5, size_sigma=0.3)),
    ModelSpec("mike_farmer_2008", "Mike & Farmer (2008 structure)", MikeFarmer2008Model, (
        Param("lam", "Order arrival rate", 10.0, 0.1, 200.0, 0.1),
        Param("cancel_rate", "Maximum cancellation rate (per order)", 0.05, 0.001, 5.0, 0.001),
        Param("imbalance_offset", "Imbalance offset B", 0.2, 0.0, 5.0, 0.05),
        Param("hurst", "Order-sign Hurst exponent", 0.75, 0.5, 0.95, 0.05),
        Param("nu", "Relative-price tail ν (Student's t)", 1.3, 0.5, 10.0, 0.1),
        Param("loc", "Relative-price location (ticks)", 1.0, -10.0, 20.0, 0.5),
        Param("scale", "Relative-price scale (ticks)", 3.0, 0.5, 50.0, 0.5),
        Param("lot_size", "Order size (lots)", 5, 1, 100, 1, integer=True),
    ), summary=(
        "Student's t prices relative to the same-side quote, so orders can cross the spread; "
        "long-memory order signs and state-dependent cancellation."),
        notebook=dict(lam=1.0, cancel_rate=0.01, imbalance_offset=0.2, hurst=0.75, nu=1.3, loc=1.0, scale=3.0,
                      lot_size=1)),
    ModelSpec("parlour", "Parlour (1998)-inspired", ParlourModel, (
        Param("arrival_rate", "Trader arrival rate", 12.0, 0.1, 200.0, 0.1),
        Param("delta", "Cancellation rate (per order)", 0.015, 0.001, 1.0, 0.001),
        Param("spread_threshold", "Spread threshold (ticks)", 5, 1, 50, 1, integer=True),
        Param("depth_threshold", "Queue depth threshold (lots)", 10, 1, 500, 1, integer=True),
        Param("price_range", "Maximum placement distance (ticks)", 20, 1, 200, 1, integer=True),
        Param("lot_size", "Order size (lots)", 5, 1, 100, 1, integer=True),
    ), summary=(
        "A hand-tuned rule: a market order becomes more likely when the spread is narrow "
        "and the queue at the best quote is long."),
        notebook=dict(arrival_rate=5.0, delta=0.05, spread_threshold=5, depth_threshold=10, price_range=20,
                      lot_size=1)),
    ModelSpec("manual", "Manual placement only", None,
              summary="No model runs: the book changes only through your own orders."),
)}

SEED_RANGE = (0, 2**32 - 1)
