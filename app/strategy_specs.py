"""The strategies the dashboard can run (lob/strategies.py), with their settings and allowed ranges.

The Strategies card's form and the start endpoint are generated from these, as the model menu and
parameter panel are from ``model_specs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.model_specs import Param
from lob.strategies import EXECUTIONS, MarketMaking, MeanReversion, Momentum, QueueImbalance, Strategy


@dataclass(frozen=True)
class Choice:
    """A setting chosen from a few options."""
    name: str
    label: str
    default: str
    options: tuple[tuple[str, str], ...]  # (value, label)

    def parse(self, raw: Any) -> str:
        if raw in {value for value, _ in self.options}:
            return raw
        raise ValueError(f"{self.label}: choose {' or '.join(label for _, label in self.options)}")


EXECUTION_LABELS = {"aggressive": "crossing the spread", "passive": "resting at the best quote"}
EXECUTION = Choice("execution", "Trade by", "aggressive", tuple((v, EXECUTION_LABELS[v]) for v in EXECUTIONS))
PROTECTION = Param("protection", "Walk at most (ticks)", 5, 0, 100, 1, integer=True)
SIZE = Param("size", "Lots", 5, 1, 1_000, 1, integer=True)


def interval(default: float) -> Param:
    return Param("interval", "Decide every (time units)", default, 0.05, 100.0, 0.05)


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    tag: str  # its badge in the order book
    summary: str
    factory: Callable[..., Strategy]
    params: tuple[Param | Choice, ...]

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def parse(self, form: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """The settings from a form, and a message for each field that is not valid."""
        values, errors = {}, {}
        for param in self.params:
            try:
                values[param.name] = param.parse(form.get(param.name, param.default))
            except ValueError as exc:
                errors[param.name] = str(exc)
        return values, errors

    def build(self, values: dict[str, Any]) -> Strategy:
        """The strategy; raises ValueError when settings contradict each other."""
        return self.factory(**values)


STRATEGY_SPECS: dict[str, StrategySpec] = {spec.key: spec for spec in (
    StrategySpec("market_making", "Market making", "MM", (
        "Quotes a bid and an ask around the mid and earns the spread when both trade; leans against its "
        "inventory, and loses when the price moves through its quotes."), MarketMaking, (
        Param("half_spread", "Ticks from the mid", 2, 1, 50, 1, integer=True),
        Param("size", "Lots a side", 5, 1, 1_000, 1, integer=True),
        Param("limit", "Inventory limit (lots)", 50, 1, 100_000, 1, integer=True),
        Param("lean", "Lean against inventory (0 to 1)", 0.5, 0.0, 1.0, 0.05),
        Param("interval", "Requote every (time units)", 1.0, 0.05, 100.0, 0.05),
    )),
    StrategySpec("momentum", "Momentum", "MOM", (
        "Follows the trend: long while a fast moving average of the mid is above a slow one, short while "
        "it is below, flat once they cross back."), Momentum, (
        Param("fast", "Fast average (steps)", 5, 1, 500, 1, integer=True),
        Param("slow", "Slow average (steps)", 20, 2, 2_000, 1, integer=True),
        Param("threshold", "Enter beyond (ticks apart)", 0.5, 0.0, 50.0, 0.1),
        SIZE, interval(0.5), EXECUTION, PROTECTION,
    )),
    StrategySpec("mean_reversion", "Mean reversion", "MR", (
        "Fades large moves: short when the mid is far above its recent average, long when far below, "
        "flat once it is back near it."), MeanReversion, (
        Param("window", "Average over (steps)", 20, 3, 2_000, 1, integer=True),
        Param("entry", "Enter beyond (standard deviations)", 2.0, 0.1, 10.0, 0.1),
        Param("exit", "Exit within (standard deviations)", 0.5, 0.0, 10.0, 0.1),
        SIZE, interval(0.5), EXECUTION, PROTECTION,
    )),
    StrategySpec("queue_imbalance", "Queue imbalance", "QI", (
        "Reads the queues at the best quotes (Gould and Bonart, 2016): the smaller one is likelier to run "
        "out first, so it buys once the bid's queue is much the larger, sells in the opposite case, and "
        "goes flat when the balance tips back."),
        QueueImbalance, (
        Param("threshold", "Imbalance beyond (0 to 1)", 0.6, 0.0, 0.99, 0.05),
        SIZE, interval(0.25), EXECUTION, PROTECTION,
    )),
)}
MAX_STRATEGIES = 8  # running or stopped, per session
