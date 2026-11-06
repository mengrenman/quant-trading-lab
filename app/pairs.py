"""The dashboard's Pairs view: two order books whose gap mean-reverts, and pairs traders in them (lob/pairs.py).

One ``PairsSession`` per browser session (``LOBManager.pairs``, made when first needed), apart from the
single-book market: its own clock, speed (time units a second) and run loop, its own event stream
(``/pairs/events``, which a tab holds in place of ``/events`` while the view is shown), the market's
settings (applied with a reset) and up to ``MAX_TRADERS`` pairs traders. The chart reads ``samples``,
one a unit of time: both mids and fair values, the gap as a trader sees it (from the mids) and as it
truly is (from the fair values), each trader's P&L, and each trader's bands (its mean and entry levels,
in the chart's gap terms); and ``marks``, the times a trader's target changed.

``PairsSession.fit_trader`` tries a trader's settings on training runs of their own: fresh markets from the
same settings with seeds the live market can never have, in worker processes so the view stays responsive.
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import os
import random
import time
from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

from app.fitting import FIT_OBJECTIVES, MAX_FIT_CANDIDATES, candidates
from app.model_specs import Param
from lob.order import Side
from lob.pairs import (A, B, Arbitrageur, CopulaTrader, FairValues, KalmanTrader, PairMarket, PairsTrader,
                       score_trader)

LEVEL = 1000.0  # where A starts, and B at beta times that: far above the fair values' floor (lob.pairs.FLOOR)
SAMPLES = 3_000  # chart points kept, one a unit of time
MARKS = 2_000  # position changes kept, all traders together
MAX_TRADERS = 6
SPEED_MAX = 500  # time units a second (about 39 events each)
SPEED_DEFAULT = 20
MAX_STEP = 5_000  # time units one step request may run
FRAME = 0.05  # seconds between the run loop's frames
CHUNK = 2_000  # events between yields while a step request runs
BOOK_LEVELS = 10  # a side, in each book's panel
# Each trader's color on the chart and the board, by its number: theme colors (--c-<name> in css/themes.css),
# none of them the buy and sell colors.
COLORS = ("lime-400", "amber-300", "cyan-300", "violet-300", "sky-300", "slate-300")
FIT_SPAN = Param("span", "Run length", 10_000, 500, 50_000, 500, integer=True)  # units of time, after its warm-up
FIT_RUNS = Param("runs", "Runs", 3, 1, 10, 1, integer=True)  # training seeds, the same for every candidate
MAX_FIT_UNITS = 750_000  # candidates times runs times units of time in one fit: fewer candidates over long runs
FIT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
CLEARLY = 2.0  # standard errors by which a candidate must beat the user's settings to be called clearly better


@dataclass(frozen=True)
class Switch:
    """A setting that is on or off: a checkbox, on when its name is in the form."""
    name: str
    label: str
    default: bool

    def parse(self, raw: Any) -> bool:
        return raw is not None and raw is not False and str(raw).strip().lower() not in ("", "false", "off", "0", "no")


MARKET_PARAMS: tuple[Param | Switch, ...] = (
    Switch("reverting", "The gap mean-reverts", True),
    Param("half_life", "Half-life", 50.0, 2.0, 5_000.0, 1.0),
    Param("own_volatility", "Own volatility", 0.8, 0.0, 5.0, 0.1),
    Param("volatility", "Common volatility", 1.0, 0.0, 5.0, 0.1),
    Param("impact", "Impact a lot", 0.05, 0.0, 1.0, 0.01),
    Param("beta", "B's beta to A", 1.0, 0.1, 5.0, 0.1),
    Param("seed", "Seed", 1, 0, 2**31 - 1, 1, integer=True),
    Switch("drifting", "Beta drifts", False),
    Param("beta_sd", "Its sd", 0.3, 0.01, 2.0, 0.01),
    Param("beta_half_life", "Its half-life", 2000.0, 10.0, 100_000.0, 10.0),
    Switch("stress", "A stressed regime", False),
    Param("stress_share", "Share of the time", 0.1, 0.01, 0.9, 0.01),
    Param("stress_volatility", "Common vol in it", 3.0, 0.0, 20.0, 0.1),
    Param("stress_length", "Mean length", 50.0, 1.0, 5_000.0, 1.0),
    Param("stress_drift", "Drift in it", 0.0, -10.0, 10.0, 0.1),
    Switch("arbitrageur", "An arbitrageur", False),
    Param("threshold", "Trades past", 2.0, 0.5, 50.0, 0.5),
    Param("arb_size", "Lots a trade", 5, 1, 100, 1, integer=True),
)
WARM = Switch("warm", "Learn from the chart first", True)
INTERVAL = ("interval", "Decide every", 0.1, 50.0, 0.1)


def _every(v: dict[str, Any]) -> str:
    return f"every {v['interval']:g} time unit{'' if v['interval'] == 1 else 's'}"


def _flat(v: dict[str, Any]) -> str:
    return "at its mean" if v["exit"] == 0 else f"within {v['exit']:g} of its mean"


@dataclass(frozen=True)
class TraderKind:
    """A kind of pairs trader the view offers: its settings, how to make one from them, and how to say what it does."""
    key: str
    label: str
    summary: str
    params: tuple[Param | Switch, ...]
    fields: tuple[str, ...]
    make: Callable[..., PairsTrader]
    describe: Callable[[dict[str, Any]], str]
    fit_grid: dict[str, tuple[float, ...]] = field(default_factory=dict)  # values proposed for fitting, by setting
    fit_default: tuple[str, ...] = ()  # the settings tried unless the user says otherwise

    @property
    def numbers(self) -> tuple[Param, ...]:
        """The settings a fit can try: all but the on-or-off ones."""
        return tuple(p for p in self.params if isinstance(p, Param))

    def build(self, values: dict[str, Any]) -> PairsTrader:
        """The trader; raises ValueError when settings contradict each other."""
        return self.make(**{k: values[k] for k in self.fields})

    def valid(self, values: dict[str, Any]) -> bool:
        try:
            self.build(values)
        except ValueError:
            return False
        return True


def _describe_zscore(v: dict[str, Any]) -> str:
    a_lots = round(v["hedge"] * v["size"])  # as lob.pairs trades it
    return (f"Gap B \u2212 {v['hedge']:g} \u00d7 A, its z-score over the last {v['window']} decisions, {_every(v)}. "
            f"Above {v['entry']:g}: short {v['size']} B, long {a_lots} A; below \u2212{v['entry']:g} the reverse; "
            f"flat {_flat(v)}. Market orders in both books.")


def _describe_kalman(v: dict[str, Any]) -> str:
    holds = "follows the filter while it holds" if v["rebalance"] else "stays as it was on entry"
    return (f"A Kalman filter of B on A with a memory of {v['memory']:g} decisions, started by least squares over "
            f"{v['window']}, {_every(v)}. Its error's z-score above {v['entry']:g}: short {v['size']} B, long the "
            f"filter's hedge of A; below \u2212{v['entry']:g} the reverse; flat {_flat(v)}. The hedge {holds}.")


def _describe_copula(v: dict[str, Any]) -> str:
    a_lots = round(v["hedge"] * v["size"])
    back = "0" if v["exit"] == 0 else f"within {v['exit']:g} of 0"
    return (f"A copula on the books' returns {_every(v)}, fitted to the last {v['window']} and again every "
            f"{v['trading']} decisions, which also closes what is open. B's flag above {v['entry']:g} and A's below "
            f"\u2212{v['entry']:g}: short {v['size']} B, long {a_lots} A; the reverse the other way; flat once both "
            f"are back to {back}.")


TRADER_KINDS: dict[str, TraderKind] = {kind.key: kind for kind in (
    TraderKind(
        "zscore", "Z-score",
        "The gap at a fixed hedge, against its mean and sd over a window. Beyond the entry level it shorts a dear gap "
        "(sells B, buys A) or buys a cheap one, and goes flat within the flat level of the mean.",
        (Param("window", "Window", 400, 3, 5_000, 1, integer=True),
         Param(*INTERVAL[:2], 1.0, *INTERVAL[2:]),
         Param("entry", "Enter beyond z", 2.0, 0.1, 10.0, 0.1),
         Param("exit", "Flat within z", 0.0, 0.0, 9.9, 0.1),
         Param("size", "Lots of B", 5, 1, 100, 1, integer=True),
         Param("hedge", "Hedge", 1.0, 0.1, 10.0, 0.1),
         WARM),
        ("window", "interval", "entry", "exit", "size", "hedge"), PairsTrader, _describe_zscore,
        {"window": (200, 400, 1000, 2000), "interval": (1, 2, 5), "entry": (1.5, 2, 2.5, 3), "exit": (0, 0.5, 1),
         "size": (2, 5, 10), "hedge": (0.5, 1, 1.5, 2)}, ("window", "entry")),
    TraderKind(
        "kalman", "Kalman",
        "A regression of B on A whose slope and level may drift, updated at each decision (Chan 2013). The slope is "
        "the hedge; the prediction's error, over its usual size, is the z-score it trades. Its memory must be long "
        "against the gap's half-life, or it follows the gap and leaves nothing to trade. Noise in A's own price biases "
        "the slope low, the more so the weaker the common factor: compare it with the true beta on its row.",
        (Param("window", "First fit", 400, 10, 5_000, 1, integer=True),
         Param("memory", "Memory", 1000.0, 10.0, 100_000.0, 10.0),
         Param("entry", "Enter beyond z", 2.0, 0.1, 10.0, 0.1),
         Param("exit", "Flat within z", 0.0, 0.0, 9.9, 0.1),
         Param("size", "Lots of B", 5, 1, 100, 1, integer=True),
         Param(*INTERVAL[:2], 1.0, *INTERVAL[2:]),
         Switch("rebalance", "Follow the hedge while holding", True),
         WARM),
        ("window", "memory", "entry", "exit", "size", "interval", "rebalance"), KalmanTrader, _describe_kalman,
        {"window": (200, 400, 1000), "memory": (300, 1000, 3000), "entry": (1.5, 2, 2.5, 3), "exit": (0, 0.5, 1),
         "size": (2, 5, 10), "interval": (1, 2, 5)}, ("memory", "entry")),
    TraderKind(
        "copula", "Copula",
        "The books' returns through a copula fitted to their ranks, the family chosen by AIC (Rad, Low and Faff 2016). "
        "How high each return was for what the other's did adds up into a flag; it trades when the flags part and goes "
        "flat when both are back. They used an entry level of 0.5; here that trades too often.",
        (Param("window", "Fit on", 400, 20, 5_000, 1, integer=True),
         Param("trading", "Trade for", 200, 1, 5_000, 1, integer=True),
         Param("entry", "Flags beyond", 1.0, 0.1, 20.0, 0.1),
         Param("exit", "Flat within", 0.0, 0.0, 19.9, 0.1),
         Param("size", "Lots of B", 5, 1, 100, 1, integer=True),
         Param("hedge", "Hedge", 1.0, 0.1, 10.0, 0.1),
         Param(*INTERVAL[:2], 5.0, *INTERVAL[2:]),
         WARM),
        ("window", "trading", "entry", "exit", "size", "hedge", "interval"), CopulaTrader, _describe_copula,
        {"window": (200, 400, 800), "trading": (100, 200, 400), "entry": (0.5, 1, 1.5, 2, 3), "exit": (0, 0.5, 1),
         "size": (2, 5, 10), "hedge": (0.5, 1, 1.5, 2), "interval": (2, 5, 10)}, ("entry", "trading")),
)}


def parse(params: tuple[Param | Switch, ...], form: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Each setting from the form (switches: on when present), and a message for each that is not valid."""
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for p in params:
        if isinstance(p, Switch):
            values[p.name] = p.parse(form.get(p.name))
            continue
        try:
            values[p.name] = p.parse(form.get(p.name, p.default))
        except ValueError as exc:
            errors[p.name] = str(exc)
    return values, errors


def _r(x: Optional[float], digits: int = 3) -> Optional[float]:
    return None if x is None else round(x, digits)


def market_from(s: dict[str, Any], seed: int, traders: Iterable[PairsTrader] = ()) -> PairMarket:
    """A fresh pair of books from the view's market settings, with these traders in it."""
    fair = FairValues(level_a=LEVEL, level_b=LEVEL * s["beta"], beta=s["beta"], volatility=s["volatility"],
                      own_volatility=s["own_volatility"], half_life=s["half_life"] if s["reverting"] else None,
                      impact=s["impact"], beta_sd=s["beta_sd"] if s["drifting"] else 0.0,
                      beta_half_life=s["beta_half_life"], stress_share=s["stress_share"] if s["stress"] else 0.0,
                      stress_volatility=s["stress_volatility"], stress_length=s["stress_length"],
                      stress_drift=s["stress_drift"])
    arb = Arbitrageur(threshold=s["threshold"], size=s["arb_size"]) if s["arbitrageur"] else None
    return PairMarket(fair, arbitrageur=arb, seed=seed, traders=tuple(traders))


# ---- Fitting a trader over training runs ----

def training_seed(fit_seed: int, run: int) -> int:
    """A training market's seed: above any the market's seed setting allows, so never the live market's draw."""
    return (fit_seed + 1) * 2**32 + run


def training_run(kind: str, values: dict[str, Any], settings: dict[str, Any], seed: int, burn: float,
                 span: float) -> dict[str, float]:
    """How one candidate did alone in one training market (lob.pairs.score_trader); runs in a worker process."""
    trader = TRADER_KINDS[kind].build(values)
    return score_trader(market_from(settings, seed, (trader,)), trader, burn, span)


def fit_plan(kind: str, base: dict[str, Any], grid: dict[str, list[Any]], span: int, runs: int,
             seed: int) -> tuple[list[dict[str, Any]], Optional[int], float, bool]:
    """What a fit will run: the settings to try, the user's own among them, first unless they are in the grid; how
    many of the grid's combinations are valid; the time every run gets before it is measured; and whether these
    are a sample of the grid's rather than all of them.

    As many as MAX_FIT_CANDIDATES and MAX_FIT_UNITS allow: with more, a random sample drawn from ``seed``. Every
    run gets the longest warm-up among them, so all are measured over the same span of a market of the same age.
    Raises ValueError when none is valid, or when these runs are too long to compare even two settings.
    """
    spec = TRADER_KINDS[kind]
    tried, valid = candidates(base, grid, spec.valid, MAX_FIT_CANDIDATES - 1, seed)
    if not tried:
        raise ValueError("none of these combinations of settings is valid")
    warm_up = lambda values: spec.build(values).warm_up  # noqa: E731
    burn = max(map(warm_up, [base, *tried]))
    room = MAX_FIT_UNITS // (runs * (burn + span)) - 1  # the rest of the budget, after the user's settings
    if room < 1:
        raise ValueError(f"{runs} run{'' if runs == 1 else 's'} of {burn + span:,.0f} units of time ({burn:,.0f} to "
                         "warm up) are too long to compare settings: fewer runs, a shorter span, or a shorter "
                         "longest window")
    if len(tried) > room:
        picked = random.Random(seed).sample(range(len(tried)), int(room))
        tried = [tried[i] for i in sorted(picked)]
        burn = max(map(warm_up, [base, *tried]))
    return ([] if base in tried else [base]) + tried, valid, burn, valid is None or len(tried) < valid


def rank_candidates(tried: list[dict[str, Any]], runs: list[list[dict[str, float]]], yours: dict[str, Any],
                    objective: str) -> list[dict[str, Any]]:
    """One row per candidate, best first (ties: the one that traded less), from its runs on the same seeds.

    ``against`` is its score less the user's settings', seed by seed, and ``against_se`` the standard error of that
    mean difference (None from a single run, which cannot tell). Runs of a seed share a market only until the first
    trade, which sends each its own way, so the pairing takes out little of the luck: the standard error, measured
    from the differences, says how much is left.
    """
    def mean(c: list[dict[str, float]], key: str) -> float:
        return float(np.mean([run[key] for run in c]))

    def stderr(xs: Any) -> Optional[float]:
        return float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else None

    key = "pnl" if objective == "pnl" else "risk_adjusted"
    theirs = np.array([run[key] for run in runs[tried.index(yours)]])
    rows = []
    for values, c in zip(tried, runs, strict=True):
        scores = np.array([run[key] for run in c])
        rows.append({"values": values, "yours": values == yours, "score": float(scores.mean()),
                     "pnl": mean(c, "pnl"), "pnl_se": stderr([run["pnl"] for run in c]),
                     "before": mean(c, "before_costs"), "costs": mean(c, "costs"), "lots": mean(c, "lots"),
                     "entries": mean(c, "entries"), "against": float((scores - theirs).mean()),
                     "against_se": stderr(scores - theirs)})
    rows.sort(key=lambda row: (-row["score"], row["lots"]))
    return rows


@dataclass
class PairsFit:
    """A trader fitted on training runs: every candidate's result there, best first."""
    id: int
    kind: str
    grid: dict[str, list[Any]]
    settings: dict[str, Any]  # the market's, when it was fitted
    objective: str
    seed: int
    span: float  # the units of time each run is measured over
    burn: float  # the units of time before that
    runs: int
    candidates: list[dict[str, Any]]
    valid: Optional[int]  # how many of the grid's combinations are valid (None: too many to count)
    size: int  # how many combinations the grid has
    sampled: bool  # the candidates are a random sample of the grid's
    seconds: float

    @property
    def verdict(self) -> str:
        """``yours`` (the user's settings came out best), ``clearly`` (the best beat them by CLEARLY standard
        errors), ``unclear`` (it did not), or ``single`` (one run, which cannot tell)."""
        best = self.candidates[0]
        if best["yours"]:
            return "yours"
        if best["against_se"] is None:
            return "single"
        return "clearly" if best["against"] > CLEARLY * best["against_se"] else "unclear"

    @property
    def yours(self) -> dict[str, Any]:
        return next(c for c in self.candidates if c["yours"])


class FitCanceled(Exception):
    """A fit stopped at the user's request."""


def _lower_priority() -> None:
    os.nice(10)  # a worker gives way to the server and the market the user is watching


def _fit_pool() -> Executor:
    """Worker processes for one fit, spawned so that they inherit nothing of the running server."""
    return ProcessPoolExecutor(FIT_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_lower_priority)


_fit_running = False  # one fit at a time on this server: it takes all the cores


@dataclass
class PairsRunner:
    """A pairs trader on the board: its kind and settings, whether it is deciding, and when its target changed."""
    id: int
    name: str
    values: dict[str, Any]
    kind: str = "zscore"
    active: bool = True
    trader: PairsTrader = field(init=False)
    last_target: int = 0
    trades_from: float = 0.0  # the time it has a signal from: what is left of its warm-up, from when it started
    fit: Optional[dict[str, Any]] = None  # how it did in training, if it came from a fit (PairsSession.start_fitted)

    def __post_init__(self) -> None:
        self.fresh()

    def fresh(self) -> None:
        """A new trader with the same settings (flat, nothing seen yet)."""
        self.trader = TRADER_KINDS[self.kind].build(self.values)
        self.trader.active = self.active
        self.last_target = 0
        self.trading_from(0.0)

    def trading_from(self, now: float) -> None:
        """Note when, from ``now``, what it has yet to see will have taken its time."""
        warming = self.trader.warming
        self.trades_from = now + ((warming[1] - warming[0]) * self.trader.interval if warming else 0.0)

    @property
    def color(self) -> str:
        return COLORS[(self.id - 1) % len(COLORS)]

    def describe(self) -> str:
        return TRADER_KINDS[self.kind].describe(self.values)

    def detail(self, true_beta: float) -> str:
        """What is worth saying about it now, besides its position."""
        t = self.trader
        if self.kind == "kalman" and t.beta_hat is not None:
            held = "" if self.values["rebalance"] else f", holding {t.hedge:.3f}"
            return f"hedge {t.beta_hat:.3f}{held} (true beta {true_beta:.3f}), usual error {t.error_sd:.2f}"
        if self.kind == "copula" and t.copula is not None:
            lower, upper = t.copula.tail_dependence()
            tails = f", tails {lower:.2f} and {upper:.2f}" if lower or upper else ""
            return f"{t.copula.describe()}{tails}; fit {t.fits}, tau {t.fit.tau:.2f}"
        return ""

    def fit_note(self, pnl: float, now: float, settings: dict[str, Any]) -> str:
        """How it did in training, a unit of time, beside how it has done here since it had a signal."""
        f = self.fit
        if f is None:
            return ""
        se = f" (se {f['pnl_se'] / f['span']:.3f})" if f["pnl_se"] is not None else ""
        units = now - self.trades_from
        here = f"{pnl / units:+.3f} a unit here over {units:,.0f} units" if units >= 1 else "nothing yet here"
        other = "" if same_market(f["settings"], settings) else " Those runs had other market settings."
        return (f"Fitted by {FIT_OBJECTIVES[f['objective']]}, {_ordinal(f['rank'])} of {f['of']}: "
                f"{f['pnl'] / f['span']:+.3f} a unit in {f['runs']} training run{'' if f['runs'] == 1 else 's'}{se}; "
                f"{here}.{other}")


def same_market(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two sets of market settings make the same market, but for its random numbers."""
    return {k: v for k, v in a.items() if k != "seed"} == {k: v for k, v in b.items() if k != "seed"}


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


class PairsSession:
    def __init__(self) -> None:
        self.settings: dict[str, Any] = {p.name: p.default for p in MARKET_PARAMS}
        self.speed = SPEED_DEFAULT
        self.is_running = False
        self.runners: list[PairsRunner] = []
        self._seq = 0
        self.version = 0  # incremented when the chart starts over (a reset)
        self.changes = 0  # incremented by whatever changes what the page shows outside the clock
        self.fitting: Optional[tuple[int, int]] = None  # (runs done, runs) while a trader is fitted
        self.last_fit: Optional[PairsFit] = None
        self.fit_changes = 0  # incremented when what the last fit's panel says changes
        self._fit_seq = 0
        self._fit_cancel = False
        self._subscribers: set[asyncio.Queue] = set()
        self._task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.reset()

    # ---- The market ----

    def reset(self) -> None:
        """A fresh market from the settings; each trader on the board starts over (flat, as it was set)."""
        for r in self.runners:
            r.fresh()
        self.market = market_from(self.settings, self.settings["seed"], (r.trader for r in self.runners))
        self.samples: deque[dict[str, Any]] = deque(maxlen=SAMPLES)
        self.marks: deque[tuple[int, int, int, int]] = deque(maxlen=MARKS)  # (seq, trader id, sample time, target)
        self.mark_seq = 0
        self._next_sample = 0
        self._sample()
        self.version += 1
        self.changes += 1

    def configure(self, values: dict[str, Any]) -> None:
        self.settings = dict(values)
        self.fit_changes += 1  # the last fit's panel says whether it was on these settings
        self.reset()

    def run_for(self, units: float) -> None:
        """Run on by ``units`` of time."""
        until = self.market.time + units
        while self.market.time < until:
            self._event()

    def _event(self) -> None:
        m = self.market
        m.step()
        for r in self.runners:  # a changed target shows on the first sample that reflects it
            if r.trader.target != r.last_target:
                r.last_target = r.trader.target
                self.mark_seq += 1
                self.marks.append((self.mark_seq, r.id, self._next_sample, r.trader.target))
        while m.time >= self._next_sample:
            self._sample()

    def _sample(self) -> None:
        """The chart's point at the current sample time (just past it: taken after the event that crossed it)."""
        m, f = self.market, self.market.fair
        mark_a, mark_b = m.mark(A), m.mark(B)
        mid_a, mid_b, gap = m.mid(A), m.mid(B), m.gap()
        pnl, bands, signals, hedges = {}, {}, {}, {}
        for r in self.runners:
            t = r.trader
            pnl[r.id] = _r(t.value(mark_a, mark_b), 2)
            hedges[r.id] = _r(t.beta_hat if isinstance(t, KalmanTrader) and t.beta_hat is not None else t.hedge)
            if any(x is not None for x in t.signals):
                signals[r.id] = [_r(x) for x in t.signals]
            # Its levels in the chart's terms: where the gap would be with its own gap at its center, and either side.
            deviation = t.deviation(mid_a, mid_b) if gap is not None else None
            if deviation is not None and t.width is not None:
                center, width = gap - deviation, t.width
                bands[r.id] = [_r(center - width), _r(center), _r(center + width)]
        self.samples.append({"t": self._next_sample, "a": mid_a, "b": mid_b, "fa": _r(f.fair(A)), "fb": _r(f.fair(B)),
                             "gap": _r(gap), "true": _r(f.gap()), "beta": _r(f.beta), "pnl": pnl, "bands": bands,
                             "sig": signals, "hedge": hedges})
        self._next_sample += 1

    # ---- Traders ----

    def start_trader(self, values: dict[str, Any], kind: str = "zscore") -> tuple[PairsRunner, int]:
        """Put a pairs trader in the market, deciding from now on; with ``warm``, it first takes in the chart's mids
        (one every interval, rounded to whole units). Returns it and how many of them it took.
        Raises ValueError (settings that cannot work) or RuntimeError (the board is full).
        """
        if len(self.runners) >= MAX_TRADERS:
            raise RuntimeError(f"At most {MAX_TRADERS} pairs traders: remove one first.")
        if not values["exit"] < values["entry"]:
            raise ValueError("The flat level must be below the entry level.")
        self._seq += 1
        runner = PairsRunner(self._seq, f"{TRADER_KINDS[kind].label} {self._seq}", dict(values), kind)
        trader = runner.trader
        learned = 0
        if values.get("warm"):
            stride = max(1, round(trader.interval))
            mids = [(s["a"], s["b"]) for s in list(self.samples)[::-1][::stride]
                    if s["a"] is not None and s["b"] is not None]
            learned = trader.learn(mids[::-1])
        trader.next_time = self.market.time
        runner.trading_from(self.market.time)
        self.runners.append(runner)
        self.market.traders.append(trader)
        self.changes += 1
        return runner, learned

    def runner(self, runner_id: int) -> Optional[PairsRunner]:
        return next((r for r in self.runners if r.id == runner_id), None)

    def set_active(self, runner: PairsRunner, active: bool) -> None:
        """Stopped, a trader keeps what it holds and decides no more; resumed, it decides again from now."""
        runner.active = runner.trader.active = active
        if active:
            runner.trader.next_time = self.market.time
        self.changes += 1

    def remove(self, runner: PairsRunner) -> None:
        """Off the board, with what it holds (nothing is closed out: its orders would move the books)."""
        self.runners.remove(runner)
        self.market.traders.remove(runner.trader)
        self.changes += 1

    # ---- Fitting a trader over training runs ----

    async def fit_trader(self, kind: str, base: dict[str, Any], grid: dict[str, list[Any]], span: int, runs: int,
                         objective: str, seed: int = 1) -> PairsFit:
        """Try the user's settings ``base`` and the combinations of ``grid``'s values (the rest as in ``base``).

        Each candidate runs alone in ``runs`` fresh markets from the view's market settings, on the same seeds for
        all, which the live market can never have (training_seed), and is measured over the same ``span`` of time
        after the same warm-up (fit_plan). The live market runs on meanwhile: this holds no lock. Raises
        ValueError for what fit_plan refuses, RuntimeError while another fit runs or if the workers fail, and
        FitCanceled after cancel_fit.
        """
        global _fit_running
        if objective not in FIT_OBJECTIVES:
            raise ValueError(f"the objective must be one of {', '.join(FIT_OBJECTIVES)}")
        tried, valid, burn, sampled = fit_plan(kind, base, grid, span, runs, seed)
        if _fit_running:
            raise RuntimeError("another fit is running")
        _fit_running = True
        self._fit_cancel = False
        settings = dict(self.settings)
        total = len(tried) * runs
        self.fitting = (0, total)
        self.changes += 1
        self.notify()
        began = time.monotonic()
        loop = asyncio.get_running_loop()
        pool = _fit_pool()
        try:
            where = {loop.run_in_executor(pool, training_run, kind, values, settings, training_seed(seed, r), burn,
                                          span): (c, r)
                     for c, values in enumerate(tried) for r in range(runs)}
            results: list[list[Any]] = [[None] * runs for _ in tried]
            pending = set(where)
            while pending:
                done, pending = await asyncio.wait(pending, timeout=0.2, return_when=asyncio.FIRST_COMPLETED)
                if self._fit_cancel:
                    raise FitCanceled
                for future in done:
                    c, r = where[future]
                    results[c][r] = future.result()
                if done:
                    self.fitting = (total - len(pending), total)
                    self.changes += 1
                    self.notify()
        except BrokenProcessPool:
            raise RuntimeError("the worker processes running the training stopped unexpectedly") from None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # any run still going ends soon, and its process with it
            _fit_running = False
            self.fitting = None
            self.changes += 1
            self.notify()
        self._fit_seq += 1
        self.last_fit = PairsFit(id=self._fit_seq, kind=kind, grid=grid, settings=settings, objective=objective,
                                 seed=seed, span=span, burn=burn, runs=runs,
                                 candidates=rank_candidates(tried, results, base, objective), valid=valid,
                                 size=math.prod(len(v) for v in grid.values()), sampled=sampled,
                                 seconds=time.monotonic() - began)
        self.fit_changes += 1
        self.notify()
        return self.last_fit

    def cancel_fit(self) -> bool:
        """Stop the fit that is running, if one is: whether one was."""
        self._fit_cancel = self.fitting is not None
        return self._fit_cancel

    def start_fitted(self, fit_id: int, rank: int) -> tuple[PairsRunner, int]:
        """Start the ``rank``-th candidate (from 1) of the last fit, noting how it did in training.

        Raises LookupError when that fit or candidate is gone, and as start_trader otherwise.
        """
        fit = self.last_fit
        if fit is None or fit.id != fit_id or not 1 <= rank <= len(fit.candidates):
            raise LookupError("that fit is no longer here: fit it again")
        candidate = fit.candidates[rank - 1]
        runner, learned = self.start_trader(candidate["values"], fit.kind)
        runner.fit = {"rank": rank, "of": len(fit.candidates), "objective": fit.objective, "runs": fit.runs,
                      "span": fit.span, "pnl": candidate["pnl"], "pnl_se": candidate["pnl_se"],
                      "settings": fit.settings}
        return runner, learned

    # ---- What the page shows ----

    def board(self) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        """The traders' rows, best P&L first, and the arbitrageur's (None without one)."""
        m = self.market
        mark_a, mark_b = m.mark(A), m.mark(B)
        rows = []
        for r in self.runners:
            t = r.trader
            pnl = t.value(mark_a, mark_b)
            rows.append({"id": r.id, "name": r.name, "kind": r.kind, "color": r.color, "describe": r.describe(),
                         "active": r.active, "pnl": pnl, "before": pnl + t.spread_paid, "cost": t.spread_paid,
                         "position": list(t.position), "lots": t.lots, "entries": t.entries,
                         "cost_a_lot": t.spread_paid / t.lots if t.lots else None, "signals": list(t.signals),
                         "target": t.target, "size": t.size, "warming": t.warming, "detail": r.detail(m.fair.beta),
                         "fit": r.fit_note(pnl, m.time, self.settings)})
        rows.sort(key=lambda row: (-row["pnl"], row["id"]))
        arb = m.arbitrageur
        arbitrageur = None
        if arb is not None:
            arbitrageur = {"trades": arb.trades, "position": list(arb.position), "threshold": arb.threshold,
                           "size": arb.size, "pnl": arb.cash + arb.position[A] * mark_a + arb.position[B] * mark_b}
        return rows, arbitrageur

    def book_levels(self, book: int) -> dict[str, Any]:
        b = self.market.books[book]
        bids = b.depth_profile(Side.BID, BOOK_LEVELS)
        asks = b.depth_profile(Side.ASK, BOOK_LEVELS)
        return {"name": "AB"[book], "bids": bids, "asks": list(reversed(asks)), "mid": b.mid_price,
                "spread": b.spread, "fair": self.market.fair.fair(book),
                "most": max([d for _, d in bids + asks] or [1])}

    def gap_sds(self) -> dict[str, Optional[float]]:
        """The gap's standard deviation: in theory (own noise alone), and over the chart's samples."""
        def sd(xs: list[float]) -> Optional[float]:
            if len(xs) < 30:
                return None
            mean = sum(xs) / len(xs)
            return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))

        return {"theory": self.market.fair.gap_sd,
                "fair": sd([s["true"] for s in self.samples]),
                "mids": sd([s["gap"] for s in self.samples if s["gap"] is not None]),
                "units": len(self.samples)}

    def context(self) -> dict[str, Any]:
        """Everything the pushed panels show, as plain data (rendered off the event loop)."""
        m = self.market
        rows, arbitrageur = self.board()
        return {"time": m.time, "events": m.events, "running": self.is_running, "speed": self.speed,
                "gap": m.gap(), "true_gap": m.fair.gap(), "sds": self.gap_sds(), "half_life": m.fair.half_life,
                "beta": m.fair.beta, "drifting": bool(m.fair.beta_sd), "stress_on": bool(m.fair.stress_share),
                "stressed": m.fair.stressed, "floor": m.fair.floor, "pushed": m.fair.pushed,
                "books": [self.book_levels(A), self.book_levels(B)], "rows": rows, "arbitrageur": arbitrageur,
                "settings": dict(self.settings), "full": len(self.runners) >= MAX_TRADERS, "fitting": self.fitting}

    def chart_traders(self) -> list[dict[str, Any]]:
        """What the chart needs to know of each trader, in the order they started."""
        return [{"id": r.id, "name": r.name, "color": r.color, "active": r.active, "kind": r.kind,
                 "entry": r.trader.entry, "exit": r.trader.exit} for r in self.runners]

    def history(self) -> dict[str, Any]:
        """The chart's data so far: every sample kept, the marks within them, and the traders."""
        first = self.samples[0]["t"] if self.samples else 0
        return {"version": self.version, "samples": list(self.samples), "traders": self.chart_traders(),
                "marks": [[i, t, target] for _, i, t, target in self.marks if t >= first],
                "mark_seq": self.mark_seq, "settings": dict(self.settings), "speed": self.speed,
                "running": self.is_running}

    def samples_since(self, t: Optional[int]) -> list[dict[str, Any]]:
        out = []
        for s in reversed(self.samples):
            if t is not None and s["t"] <= t:
                break
            out.append(s)
        return out[::-1]

    def marks_since(self, seq: int) -> list[list[int]]:
        out = []
        for mark in reversed(self.marks):
            if mark[0] <= seq:
                break
            out.append([mark[1], mark[2], mark[3]])
        return out[::-1]

    # ---- The run loop and the event stream ----

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def notify(self) -> None:
        """Wake every event stream (a full queue drops its oldest wake-up: one is enough)."""
        for q in list(self._subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(True)

    async def broadcast(self) -> None:
        self.notify()

    async def start(self) -> None:
        self.is_running = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        self.changes += 1
        await self.broadcast()

    async def stop(self) -> None:
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.changes += 1
        await self.broadcast()

    async def advance(self, units: float) -> None:
        """Run on by ``units`` of time now (paused or not), yielding to the event loop every ``CHUNK`` events."""
        async with self.lock:
            until = self.market.time + units
            while self.market.time < until:
                for _ in range(CHUNK):
                    self._event()
                    if self.market.time >= until:
                        break
                await asyncio.sleep(0)
        await self.broadcast()

    def shutdown(self) -> None:
        """Stop the run loop without waiting for it (when the session is evicted)."""
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self.cancel_fit()
        self._subscribers.clear()

    async def _run(self) -> None:
        # The clock follows the wall clock: each frame runs on by the speed times the time since the last (at
        # most five frames' worth, so a stall is not made up in a burst), to a running target, so a frame's
        # overshoot (up to one event) comes off the next one's run rather than adding up. The target is taken
        # afresh after a reset, or when the clock has moved far from it (a step request).
        target, version, last = self.market.time, self.version, time.monotonic()
        try:
            while self.is_running:
                await asyncio.sleep(FRAME)
                now = time.monotonic()
                elapsed, last = min(now - last, 5 * FRAME), now
                async with self.lock:
                    if version != self.version or abs(self.market.time - target) > max(1.0, 5 * self.speed * FRAME):
                        target, version = self.market.time, self.version
                    target += self.speed * elapsed
                    while self.market.time < target:
                        self._event()
                self.notify()
        except asyncio.CancelledError:
            pass
        finally:
            if self._task is asyncio.current_task():  # ended by an error, not by stop(): show it paused
                self.is_running = False
                self.changes += 1
                self.notify()
