"""Live stylized facts: notebook 05's measurements, taken on the run the dashboard is showing.

``FactsRecorder`` samples the event stream the way notebook 05 samples a ``Simulation``:
after a warm-up of 5,000 steps it keeps the mid-price every 1,000 events (counting
events after which both sides of the book are quoted), the sign of every limit and
market order, the last fill price of every aggressive order, and a book snapshot
every 400 steps. ``summarize`` computes the notebook's statistics from that sample
and grades them with its rule: "Yes" when the estimate is more than 3 standard
errors beyond the bar, "Weak evidence" at 2 to 3.

The notebook takes its standard errors from 32 independent seeds. Here there is one
run, so they come from the run itself: a moving-block bootstrap for the return
statistics and batch means for the long series (order signs, trade prices,
snapshots). Both are approximate, and too small when a series has long memory, so
the dashboard labels every reading "this run". The notebook also uses its own model
parameters, so its verdicts are not a reference for the dashboard's.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats

from lob.analysis.stylized_facts import (
    classify_depth_profile, compute_acf, compute_rank_volatility_acf, fit_loglog_slope,
)
from lob.metrics import virtual_impact
from lob.order import OrderType, Side

# Sampling, as in notebook 05.
WARMUP_STEPS = 5_000
K_EVENTS = 1_000          # events per return
SNAPSHOT_EVERY = 400      # steps between book snapshots
DEPTH_TICKS = 20          # depth profile: levels from the best quote
IMPACT_SIZES = np.arange(1, 61)
SIGN_LAG = 100            # lag of the order-sign autocorrelation

# Readings need at least this much data; the most recent data is kept up to the caps.
MIN_RETURNS = 100
MIN_SIGNS = 50_000
MIN_TRADES = 1_000
MIN_SNAPSHOTS = 100
MAX_LOG_MIDS = 10_000
MAX_SIGNS = 500_000
MAX_TRADES = 100_000
MAX_SNAPSHOTS = 2_500

BOOTSTRAP_SAMPLES = 200
SIGN_BATCHES = 20
TRADE_BATCHES = 20
SNAPSHOT_BATCHES = 10

FACTS = (
    "Heavy-tailed returns",
    "Volatility clustering",
    "Long memory in order signs",
    "Bid-ask bounce",
    "Hump-shaped depth profile",
    "Concave price impact",
    "Power-law relative prices",
)

# Not measured: whether relative prices follow a power law is built into each model.
PLACEMENT_BY_DESIGN = {
    "smith_zi": "No (uniform placement)",
    "hawkes_zi": "No (uniform placement)",
    "mike_farmer": "Yes (built in)",
    "mike_farmer_2008": "Yes (built in: Student's t)",
    "parlour": "No (geometric placement)",
}


def _keep_last(buffer, cap: int) -> None:
    """Trim a buffer to its last ``cap`` items once it has grown well past them."""
    if len(buffer) > 2 * cap:
        del buffer[: len(buffer) - cap]


@dataclass
class Snapshot:
    """What one book snapshot contributes: depth by distance from the best quotes, and virtual impact."""
    bid_depth: Optional[np.ndarray]  # None when a side of the book was empty
    ask_depth: Optional[np.ndarray]
    impact: np.ndarray  # mid-price move of a buy of each size in IMPACT_SIZES (NaN if the book is too thin)


def take_snapshot(book) -> Snapshot:
    best_bid, best_ask = book.best_bid, book.best_ask
    bids = book.depth_profile(Side.BID, n_levels=50)
    asks = book.depth_profile(Side.ASK, n_levels=50)
    impact = virtual_impact({"best_bid": best_bid, "best_ask": best_ask, "bids": bids, "asks": asks},
                            IMPACT_SIZES)
    if best_bid is None or best_ask is None:
        return Snapshot(None, None, impact)
    bid_depth = np.zeros(DEPTH_TICKS)
    ask_depth = np.zeros(DEPTH_TICKS)
    for price, depth in bids:
        if best_bid - price < DEPTH_TICKS:
            bid_depth[best_bid - price] += depth
    for price, depth in asks:
        if price - best_ask < DEPTH_TICKS:
            ask_depth[price - best_ask] += depth
    return Snapshot(bid_depth, ask_depth, impact)


@dataclass
class FactsData:
    """A copy of the recorded sample, safe to summarize on another thread."""
    steps: int
    events: int
    log_mids: np.ndarray
    signs: np.ndarray
    trade_prices: np.ndarray
    snapshots: list[Snapshot] = field(default_factory=list)


class FactsRecorder:
    """Records what the stylized facts need from each event (see the module docstring)."""

    def __init__(self) -> None:
        self.generation = 0
        self.reset()

    def reset(self) -> None:
        """Start a new sample (a new model, parameters, engine or book)."""
        self.generation += 1
        self.version = 0          # changes whenever the sample does
        self.steps = 0            # model steps since the reset, including steps without an event
        self.events = 0           # events recorded after the warm-up
        self._valid = 0           # recorded events after which both sides were quoted
        self._log_mids = array("d")
        self._signs = array("b")
        self._trade_prices = array("d")
        self._snapshots: list[Snapshot] = []

    def observe_idle(self) -> None:
        """A step in which the model produced no event (counted, as ``Simulation`` counts it)."""
        self.steps += 1

    def observe(self, order_type: OrderType, side: Side, fills: list, book) -> None:
        """Record one processed event: an order (limit, market or cancel) and the fills it caused."""
        step = self.steps
        self.steps += 1
        if step < WARMUP_STEPS:
            return
        self.events += 1
        self.version += 1
        mid = book.mid_price
        if mid is not None:
            if self._valid % K_EVENTS == 0:
                self._log_mids.append(math.log(mid))
                _keep_last(self._log_mids, MAX_LOG_MIDS)
            self._valid += 1
        if order_type != OrderType.CANCEL:
            self._signs.append(1 if side == Side.BID else -1)
            _keep_last(self._signs, MAX_SIGNS)
        if fills:
            self._trade_prices.append(float(fills[-1].price))
            _keep_last(self._trade_prices, MAX_TRADES)
        if (step - WARMUP_STEPS) % SNAPSHOT_EVERY == 0:
            self._snapshots.append(take_snapshot(book))
            _keep_last(self._snapshots, MAX_SNAPSHOTS)

    def data(self) -> FactsData:
        return FactsData(
            steps=self.steps,
            events=self.events,
            log_mids=np.array(self._log_mids[-MAX_LOG_MIDS:]),
            signs=np.array(self._signs[-MAX_SIGNS:], dtype=float),
            trade_prices=np.array(self._trade_prices[-MAX_TRADES:]),
            snapshots=self._snapshots[-MAX_SNAPSHOTS:],
        )


# ---- Statistics ----

def grade(margin: float, se: float, otherwise: str) -> str:
    """Notebook 05's rule: 'Yes' beyond 3 SE, 'Weak evidence' at 2-3 SE, else the fallback."""
    if not np.isfinite(se):
        return otherwise
    if margin > 3 * se:
        return "Yes"
    if margin > 2 * se:
        return "Weak evidence"
    return otherwise


def threshold_call(value: float, se: float, bar: float, above: bool = True) -> str:
    margin = value - bar if above else bar - value
    return grade(margin, se, "No" if margin < -2 * se else "Inconclusive")


def effect_call(value: float, se: float, positive: bool = True) -> str:
    return grade(value if positive else -value, se, "Not detected")


def block_bootstrap(series: np.ndarray, rng: np.random.Generator,
                    samples: int = BOOTSTRAP_SAMPLES) -> np.ndarray:
    """Moving-block bootstrap resamples (one per row) of a series, blocks of about n^(1/3)."""
    n = len(series)
    block = max(2, round(n ** (1 / 3)))
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(samples, n_blocks))
    index = (starts[:, :, None] + np.arange(block)).reshape(samples, -1)[:, :n]
    return series[index]


def lag_acf(series: np.ndarray, lag: int) -> float:
    """Autocorrelation at one lag, defined as in ``compute_acf``."""
    centered = series - series.mean()
    var = centered.var()
    if var == 0 or len(series) <= lag:
        return float("nan")
    return float(np.mean(centered[:-lag] * centered[lag:]) / var)


def batch_se(series: np.ndarray, statistic, batches: int) -> float:
    """Standard error from the statistic's spread over contiguous batches (batch means)."""
    values = [statistic(chunk) for chunk in np.array_split(series, batches)]
    values = np.array([v for v in values if np.isfinite(v)])
    if len(values) < 3:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _row(fact: str, statistic: str, call: str, value: Optional[float] = None, se: Optional[float] = None,
         note: str = "", progress: Optional[float] = None) -> dict[str, Any]:
    ci = None
    if value is not None and se is not None and np.isfinite(se):
        ci = (value - 1.96 * se, value + 1.96 * se)
    return {"fact": fact, "statistic": statistic, "call": call, "value": value, "ci": ci,
            "note": note, "progress": progress}


def _collecting(fact: str, statistic: str, have: int, need: int, unit: str) -> dict[str, Any]:
    return _row(fact, statistic, "Collecting", note=f"{have:,} of {need:,} {unit}", progress=have / need)


def summarize(data: FactsData, model_key: str) -> dict[str, Any]:
    """The facts for the template: one row per fact, in notebook 05's order."""
    rng = np.random.default_rng(20131030)  # fixed, so a summary does not jitter between refreshes
    rows = []

    returns = np.diff(data.log_mids)
    n = len(returns)
    if n < MIN_RETURNS:
        rows.append(_collecting(FACTS[0], "Excess kurtosis of returns over 1,000 events; bar > 1",
                                n, MIN_RETURNS, "returns"))
        rows.append(_collecting(FACTS[1], "Rank ACF of |returns| at lag 1; bar > 0.1", n, MIN_RETURNS, "returns"))
    elif returns.var() == 0:
        for fact in FACTS[:2]:
            rows.append(_row(fact, "", "Not measurable", note="the mid-price never moved"))
    else:
        resamples = block_bootstrap(returns, rng)
        kurtosis = float(stats.kurtosis(returns))
        se = float(np.nanstd(stats.kurtosis(resamples, axis=1), ddof=1))
        rows.append(_row(FACTS[0], "Excess kurtosis of returns over 1,000 events; bar > 1",
                         threshold_call(kurtosis, se, 1.0), kurtosis, se, note=f"{n:,} returns"))
        ranks = stats.rankdata(np.abs(resamples), axis=1)
        centered = ranks - ranks.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (centered[:, :-1] * centered[:, 1:]).mean(axis=1) / (centered ** 2).mean(axis=1)
        clustering = float(compute_rank_volatility_acf(returns, 1)[1])
        se = float(np.nanstd(boot, ddof=1))
        rows.append(_row(FACTS[1], "Rank ACF of |returns| at lag 1; bar > 0.1",
                         threshold_call(clustering, se, 0.1), clustering, se, note=f"{n:,} returns"))

    statistic = f"Order-sign ACF at lag {SIGN_LAG}; Yes if above 0"
    if len(data.signs) < MIN_SIGNS:
        rows.append(_collecting(FACTS[2], statistic, len(data.signs), MIN_SIGNS, "orders"))
    else:
        value = lag_acf(data.signs, SIGN_LAG)
        se = batch_se(data.signs, lambda s: lag_acf(s, SIGN_LAG), SIGN_BATCHES)
        rows.append(_row(FACTS[2], statistic, effect_call(value, se), value, se,
                         note=f"{len(data.signs):,} orders"))

    statistic = "Lag-1 ACF of trade-price changes; Yes if below 0"
    changes = np.diff(data.trade_prices)
    if len(changes) < MIN_TRADES:
        rows.append(_collecting(FACTS[3], statistic, len(changes), MIN_TRADES, "trades"))
    elif changes.var() == 0:
        rows.append(_row(FACTS[3], statistic, "Not measurable", note="every trade at one price"))
    else:
        value = float(compute_acf(changes, 1)[1])
        se = batch_se(changes, lambda c: lag_acf(c, 1), TRADE_BATCHES)
        rows.append(_row(FACTS[3], statistic, effect_call(value, se, positive=False), value, se,
                         note=f"{len(changes):,} trades"))

    quoted = [s for s in data.snapshots if s.bid_depth is not None]
    statistic = f"Mean depth over the {DEPTH_TICKS} ticks from the best quotes"
    if len(quoted) < MIN_SNAPSHOTS:
        rows.append(_collecting(FACTS[4], statistic, len(quoted), MIN_SNAPSHOTS, "snapshots"))
    else:
        profile = (np.mean([s.bid_depth for s in quoted], axis=0) + np.mean([s.ask_depth for s in quoted], axis=0)) / 2
        shape, rise, _, peak = classify_depth_profile(profile)
        if shape == "hump":
            call, note = "Yes", f"peak {peak} tick{'s' if peak != 1 else ''} out, {rise:+.0%} above the best quote"
        else:
            call, note = "No", "rises to a plateau" if shape == "plateau" else "deepest at the best quote"
        rows.append(_row(FACTS[4], statistic, call, note=f"{note}; {len(quoted):,} snapshots"))

    statistic = "Log-log slope of virtual impact against size; Yes if below 0.95"
    if len(data.snapshots) < MIN_SNAPSHOTS:
        rows.append(_collecting(FACTS[5], statistic, len(data.snapshots), MIN_SNAPSHOTS, "snapshots"))
    else:
        impacts = np.array([s.impact for s in data.snapshots])
        keep = np.mean(np.isfinite(impacts), axis=0) >= 0.9
        covered = int(keep.sum())
        if covered < 5:
            rows.append(_row(FACTS[5], statistic, "Not measurable",
                             note=f"90% of snapshots absorb only {covered} lot{'s' if covered != 1 else ''}"))
        else:
            sizes = IMPACT_SIZES[keep]
            slope = fit_loglog_slope(sizes, np.nanmean(impacts[:, keep], axis=0))
            se = batch_se(impacts[:, keep], lambda chunk: fit_loglog_slope(sizes, np.nanmean(chunk, axis=0)),
                          SNAPSHOT_BATCHES)
            rows.append(_row(FACTS[5], statistic, threshold_call(slope, se, 0.95, above=False), slope, se,
                             note=f"sizes 1 to {sizes[-1]}; {len(data.snapshots):,} snapshots"))

    placement = PLACEMENT_BY_DESIGN.get(model_key)
    rows.append(_row(FACTS[6], "Built into the model, not measured",
                     placement.split(" (")[0] if placement else "Not applicable",
                     note=placement[placement.find("(") + 1:-1] if placement else "no model is running"))

    return {
        "warming_up": data.steps < WARMUP_STEPS,
        "warmup_done": min(data.steps, WARMUP_STEPS),
        "warmup_total": WARMUP_STEPS,
        "events": data.events,
        "rows": rows,
    }
