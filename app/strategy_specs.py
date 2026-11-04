"""The strategies the dashboard can run (lob/strategies.py), with their settings and allowed ranges.

The Strategies card's form and the start endpoint are generated from these, as the model menu and
parameter panel are from ``model_specs``. Each spec also proposes values to try for its main
settings when it is fitted over a training period (``fit_grid``, ``fit_default``: see
LOBManager.fit_strategy in app/manager.py); the card lets the user change them.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.model_specs import Param
from lob.strategies import (EXECUTIONS, AvellanedaStoikov, Breakout, MarketMaking, MeanReversion, Momentum,
                            OrderFlowImbalance, PennyJumping, QueueImbalance, RandomTrader, Strategy)


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
QUOTE_SIZE = Param("size", "Lots a side", 5, 1, 1_000, 1, integer=True)
LIMIT = Param("limit", "Inventory limit (lots)", 50, 1, 100_000, 1, integer=True)


def interval(default: float, label: str = "Decide every (time units)") -> Param:
    return Param("interval", label, default, 0.05, 100.0, 0.05)


FAMILIES = (("quoting", "Making markets"), ("signal", "Trading on a signal"), ("benchmark", "A benchmark"))
MAX_FIT_VALUES = 8  # values tried for one setting
MAX_FIT_CANDIDATES = 48  # combinations tried in one fit: with more, a random sample of them
COUNTED_GRID = 20_000  # grids up to this many combinations are counted exactly (beyond, sampled without counting)


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    tag: str  # its badge in the order book
    summary: str
    factory: Callable[..., Strategy]
    params: tuple[Param | Choice, ...]
    family: str = "signal"  # FAMILIES
    signal: tuple[str, str] = ("", "")  # its signal on the leaderboard: a short name, and what it is
    fit_grid: dict[str, tuple[Any, ...]] = field(default_factory=dict)  # values to try, by setting
    fit_default: tuple[str, ...] = ()  # the settings tried unless the user says otherwise

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def param(self, name: str) -> Param | Choice:
        return next(p for p in self.params if p.name == name)

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

    def grid_values(self, name: str) -> tuple[Any, ...]:
        """The values proposed for fitting a setting (every option of a choice)."""
        param = self.param(name)
        if isinstance(param, Choice):
            return tuple(value for value, _ in param.options)
        return self.fit_grid.get(name, (param.default,))

    def valid(self, values: dict[str, Any]) -> bool:
        try:
            self.build(values)
        except ValueError:
            return False
        return True

    def candidates(self, base: dict[str, Any], grid: dict[str, list[Any]], budget: Optional[int] = None,
                   seed: int = 1) -> tuple[list[dict[str, Any]], Optional[int]]:
        """The combinations of ``grid``'s values to try, the other settings as in ``base``, and how many are valid.

        Combinations whose settings contradict each other are left out. With more valid ones than
        ``budget``, a random ``budget`` of them, drawn from ``seed`` (random search, which with
        many settings does about as well as the whole grid: Bergstra and Bengio 2012), in the
        grid's order. A grid of more than COUNTED_GRID combinations is sampled without counting
        its valid ones (the count is then None).
        """
        names = list(grid)
        combine = lambda combo: {**base, **dict(zip(names, combo, strict=True))}  # noqa: E731
        size = math.prod(len(grid[n]) for n in names)
        rng = random.Random(seed)
        if size <= COUNTED_GRID:
            valid = [v for v in map(combine, itertools.product(*(grid[n] for n in names))) if self.valid(v)]
            if budget is None or len(valid) <= budget:
                return valid, len(valid)
            return [valid[i] for i in sorted(rng.sample(range(len(valid)), budget))], len(valid)
        budget = MAX_FIT_CANDIDATES if budget is None else budget
        picked: dict[tuple[int, ...], dict[str, Any]] = {}
        tried: set[tuple[int, ...]] = set()
        while len(picked) < budget and len(tried) < max(1_000, 50 * budget):
            index = tuple(rng.randrange(len(grid[n])) for n in names)
            if index in tried:
                continue
            tried.add(index)
            values = combine(grid[n][i] for n, i in zip(names, index, strict=True))
            if self.valid(values):
                picked[index] = values
        return [picked[i] for i in sorted(picked)], None


STRATEGY_SPECS: dict[str, StrategySpec] = {spec.key: spec for spec in (
    StrategySpec("market_making", "Market making", "MM", (
        "Quotes a bid and an ask around the mid and earns the spread when both trade; leans against its "
        "inventory, and loses when the price moves through its quotes."), MarketMaking, (
        Param("half_spread", "Ticks from the mid", 2, 1, 50, 1, integer=True),
        QUOTE_SIZE, LIMIT,
        Param("lean", "Lean against inventory (0 to 1)", 0.5, 0.0, 1.0, 0.05),
        interval(1.0, "Requote every (time units)"),
    ), family="quoting", fit_grid={"half_spread": (1, 2, 3, 4), "lean": (0.0, 0.5, 1.0), "interval": (0.25, 1.0, 4.0),
                                   "size": (2, 5, 10), "limit": (20, 50, 100)},
        fit_default=("half_spread", "lean")),
    StrategySpec("avellaneda_stoikov", "Avellaneda-Stoikov", "AS", (
        "Market making after Avellaneda and Stoikov (2008): quotes around a reservation price that moves "
        "against its inventory, more when the price is more volatile, with a spread that widens with "
        "volatility and risk aversion."), AvellanedaStoikov, (
        Param("gamma", "Risk aversion", 0.01, 0.0001, 10.0, 0.001),
        Param("k", "Order arrival decay (a tick)", 1.5, 0.01, 100.0, 0.1),
        Param("horizon", "Horizon (time units)", 10.0, 0.1, 10_000.0, 0.1),
        Param("window", "Volatility over (steps)", 20, 2, 2_000, 1, integer=True),
        QUOTE_SIZE, LIMIT, interval(0.5, "Requote every (time units)"),
    ), family="quoting", signal=("skew", "How far its reservation price is from the mid, in ticks"),
        fit_grid={"gamma": (0.003, 0.01, 0.03, 0.1), "k": (0.5, 1.5, 4.0), "horizon": (3.0, 10.0, 30.0),
                  "window": (10, 20, 40), "size": (2, 5, 10), "limit": (20, 50, 100)},
        fit_default=("gamma", "k")),
    StrategySpec("penny_jumping", "Penny jumping", "PJ", (
        "Rests one tick ahead of the best bid and ask when the spread is wide enough, and joins them "
        "otherwise: first in the queue, for a tick less. It keeps its place while its quotes are where it "
        "wants them."), PennyJumping, (
        Param("min_spread", "Step ahead from a spread of (ticks)", 3, 2, 50, 1, integer=True),
        QUOTE_SIZE, LIMIT, interval(0.5, "Look again every (time units)"),
    ), family="quoting", fit_grid={"min_spread": (2, 3, 4, 6), "limit": (10, 20, 50), "size": (2, 5, 10),
                                   "interval": (0.25, 0.5, 1.0)},
        fit_default=("min_spread", "limit")),
    StrategySpec("momentum", "Momentum", "MOM", (
        "Follows the trend: long while a fast moving average of the mid is above a slow one, short while "
        "it is below, flat once they cross back."), Momentum, (
        Param("fast", "Fast average (steps)", 5, 1, 500, 1, integer=True),
        Param("slow", "Slow average (steps)", 20, 2, 2_000, 1, integer=True),
        Param("threshold", "Enter beyond (ticks apart)", 0.5, 0.0, 50.0, 0.1),
        SIZE, interval(0.5), EXECUTION, PROTECTION,
    ), signal=("gap", "The fast average of the mid less the slow one, in ticks"),
        fit_grid={"fast": (3, 5, 10), "slow": (20, 40, 80), "threshold": (0.25, 0.5, 1.0), "size": (2, 5, 10),
                  "interval": (0.25, 0.5, 1.0), "protection": (0, 2, 5)},
        fit_default=("fast", "slow", "threshold")),
    StrategySpec("mean_reversion", "Mean reversion", "MR", (
        "Fades large moves: short when the mid is far above its recent average, long when far below, "
        "flat once it is back near it."), MeanReversion, (
        Param("window", "Average over (steps)", 20, 3, 2_000, 1, integer=True),
        Param("entry", "Enter beyond (standard deviations)", 2.0, 0.1, 10.0, 0.1),
        Param("exit", "Exit within (standard deviations)", 0.5, 0.0, 10.0, 0.1),
        SIZE, interval(0.5), EXECUTION, PROTECTION,
    ), signal=("z", "How many standard deviations the mid is from its average"),
        fit_grid={"window": (10, 20, 40), "entry": (1.5, 2.0, 2.5), "exit": (0.0, 0.5, 1.0), "size": (2, 5, 10),
                  "interval": (0.25, 0.5, 1.0), "protection": (0, 2, 5)},
        fit_default=("window", "entry", "exit")),
    StrategySpec("breakout", "Breakout", "BO", (
        "Follows a move out of the recent range (a Donchian channel): long when the mid rises above its "
        "highest of the window, short below its lowest, flat once it is back past the middle."), Breakout, (
        Param("window", "Range over (steps)", 20, 2, 2_000, 1, integer=True),
        SIZE, interval(0.5), EXECUTION, PROTECTION,
    ), signal=("range", "Where the mid is in its range: 0 at the middle, +1 at the top, -1 at the bottom"),
        fit_grid={"window": (10, 20, 40, 80), "interval": (0.25, 0.5, 1.0), "size": (2, 5, 10),
                  "protection": (0, 2, 5)},
        fit_default=("window", "interval")),
    StrategySpec("queue_imbalance", "Queue imbalance", "QI", (
        "Reads the queues at the best quotes (Gould and Bonart, 2016): the smaller one is likelier to run "
        "out first, so it buys once the bid's queue is much the larger, sells in the opposite case, and "
        "goes flat when the balance tips back."), QueueImbalance, (
        Param("threshold", "Imbalance beyond (0 to 0.99)", 0.6, 0.0, 0.99, 0.05),
        SIZE, interval(0.25), EXECUTION, PROTECTION,
    ), signal=("imbalance", "(bid lots - ask lots) / (bid lots + ask lots) at the best quotes"),
        fit_grid={"threshold": (0.4, 0.6, 0.8), "interval": (0.25, 0.5, 1.0), "size": (2, 5, 10),
                  "protection": (0, 2, 5)},
        fit_default=("threshold", "interval")),
    StrategySpec("order_flow_imbalance", "Order flow imbalance", "OFI", (
        "Adds up the order flow imbalance at the best quotes (Cont, Kukanov and Stoikov, 2014): lots joining "
        "the best bid or leaving the best ask are buying pressure, and the reverse selling. Long when the "
        "pressure over its window is high, short when it is low, flat when it tips back."), OrderFlowImbalance, (
        Param("window", "Flow over (steps)", 20, 2, 2_000, 1, integer=True),
        Param("threshold", "Enter beyond (average queues)", 2.0, 0.0, 100.0, 0.1),
        SIZE, interval(0.25), EXECUTION, PROTECTION,
    ), signal=("flow", "The order flow imbalance over its window, in average queue lengths at the best quotes"),
        fit_grid={"window": (10, 20, 40), "threshold": (1.0, 2.0, 3.0), "interval": (0.25, 0.5, 1.0),
                  "size": (2, 5, 10), "protection": (0, 2, 5)},
        fit_default=("window", "threshold")),
    StrategySpec("random", "Random", "RND", (
        "A benchmark that knows nothing: now and then it takes a position drawn at random, long, flat or "
        "short. What it makes or loses is the cost of trading without information; fitting it shows what "
        "luck looks like in a training period."), RandomTrader, (
        Param("switch", "Chance of a new position each step", 0.1, 0.0, 1.0, 0.05),
        Param("seed", "Seed", 1, 0, 2**31 - 1, 1, integer=True),
        SIZE, interval(0.5), EXECUTION, PROTECTION,
    ), family="benchmark",
        fit_grid={"seed": (1, 2, 3, 4, 5, 6, 7, 8), "switch": (0.05, 0.1, 0.2), "size": (2, 5, 10),
                  "interval": (0.25, 0.5, 1.0), "protection": (0, 2, 5)},
        fit_default=("seed",)),
)}
MAX_STRATEGIES = 12  # running or stopped, per session
