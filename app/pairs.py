"""The dashboard's Pairs view: two order books whose gap mean-reverts, and pairs traders in them (lob/pairs.py).

One ``PairsSession`` per browser session (``LOBManager.pairs``, made when first needed), apart from the
single-book market: its own clock, speed (time units a second) and run loop, its own event stream
(``/pairs/events``, which a tab holds in place of ``/events`` while the view is shown), the market's
settings (applied with a reset) and up to ``MAX_TRADERS`` pairs traders. The chart reads ``samples``,
one a unit of time: both mids and fair values, the gap as a trader sees it (from the mids) and as it
truly is (from the fair values), each trader's P&L, and each trader's bands (its mean and entry levels,
in the chart's gap terms); and ``marks``, the times a trader's target changed.

``PairsSession`` fits a trader's settings on training runs of their own (``TrainedMarket.fit_trader`` in
app/training.py): ``training_run`` is one candidate alone in one fresh market from the same settings, on a seed the
live market can never have, and ``fit_plan`` says what a fit tries.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from app.model_specs import Param
from app.training import TrainedMarket, fit_note, plan_fit, trades_from
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
BOOK_LEVELS = 10  # a side, in each book's panel
# Each trader's color on the chart and the board, by its number: theme colors (--c-<name> in css/themes.css),
# none of them the buy and sell colors.
COLORS = ("lime-400", "amber-300", "cyan-300", "violet-300", "sky-300", "slate-300")
MAX_FIT_UNITS = 750_000  # candidates times runs times units of time in one fit: fewer candidates over long runs


@dataclass(frozen=True)
class Switch:
    """A setting that is on or off: a checkbox, on when its name is in the form."""
    name: str
    label: str
    default: bool

    def parse(self, raw: Any) -> bool:
        return raw is not None and raw is not False and str(raw).strip().lower() not in ("", "false", "off", "0", "no")


@dataclass(frozen=True)
class Choice:
    """A setting with a few named values: a drop-down, with the value it has when nothing is chosen."""
    name: str
    label: str
    default: str
    options: tuple[str, ...]

    def parse(self, raw: Any) -> str:
        """The value chosen, or raise ValueError with a readable message."""
        if raw not in self.options:
            raise ValueError(f"{self.label}: one of {', '.join(self.options)}")
        return str(raw)


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
    params: tuple[Param | Switch | Choice, ...]
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


def parse(params: tuple[Param | Switch | Choice, ...], form: Any) -> tuple[dict[str, Any], dict[str, str]]:
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

def training_run(kind: str, values: dict[str, Any], settings: dict[str, Any], seed: int, burn: float,
                 span: float) -> dict[str, float]:
    """How one candidate did alone in one training market (lob.pairs.score_trader); runs in a worker process."""
    trader = TRADER_KINDS[kind].build(values)
    return score_trader(market_from(settings, seed, (trader,)), trader, burn, span)


def fit_plan(kind: str, base: dict[str, Any], grid: dict[str, list[Any]], span: int, runs: int,
             seed: int) -> tuple[list[dict[str, Any]], Optional[int], float, bool]:
    """What a fit of a trader of this kind will run (training.plan_fit), within MAX_FIT_UNITS."""
    return plan_fit(TRADER_KINDS[kind], base, grid, span, runs, seed, MAX_FIT_UNITS)


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
        self.trades_from = trades_from(self.trader, now)

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
        return fit_note(self.fit, pnl, now, self.trades_from, settings)


class PairsSession(TrainedMarket):
    training = staticmethod(training_run)
    plan = staticmethod(fit_plan)

    def __init__(self) -> None:
        super().__init__(SPEED_DEFAULT)
        self.settings: dict[str, Any] = {p.name: p.default for p in MARKET_PARAMS}
        self.runners: list[PairsRunner] = []
        self._seq = 0
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
