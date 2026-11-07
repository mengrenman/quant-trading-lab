"""Fitting a trader over training runs, for each view that has traders in a market of its own (Pairs and Stat-arb).

``plan_fit`` says what a fit runs, ``rank_candidates`` ranks the results, ``TrainingFit`` holds them, and
``TrainedMarket.fit_trader`` runs the candidates in worker processes while the live market runs on. One fit runs at
a time on a server, whatever the view, because it takes all the cores. A view says how one candidate trades in a
training market (its ``training`` function) and what its fit tries (its ``plan``).
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import os
import random
import threading
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from app.fitting import FIT_OBJECTIVES, MAX_FIT_CANDIDATES, candidates
from app.live import LiveMarket
from app.model_specs import Param

if TYPE_CHECKING:
    from app.pairs import TraderKind

FIT_SPAN = Param("span", "Run length", 10_000, 500, 50_000, 500, integer=True)  # units of time, after its warm-up
FIT_RUNS = Param("runs", "Runs", 3, 1, 10, 1, integer=True)  # training seeds, the same for every candidate
FIT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
CLEARLY = 2.0  # standard errors by which a candidate must beat the user's settings to be called clearly better


def training_seed(fit_seed: int, run: int) -> int:
    """A training market's seed: above any the market's seed setting allows, so never the live market's draw."""
    return (fit_seed + 1) * 2**32 + run


def plan_fit(spec: TraderKind, base: dict[str, Any], grid: dict[str, list[Any]], span: int, runs: int, seed: int,
             budget: int, valid: Optional[Callable[[dict[str, Any]], bool]] = None
             ) -> tuple[list[dict[str, Any]], Optional[int], float, bool]:
    """What a fit will run: the settings to try, the user's own among them, first unless they are in the grid; how
    many of the grid's combinations are valid; the time every run gets before it is measured; and whether these
    are a sample of the grid's rather than all of them.

    As many as MAX_FIT_CANDIDATES and ``budget`` (the most units of time, candidates times runs times burn plus
    span, in one fit) allow: with more, a random sample drawn from ``seed``. Every run gets the longest warm-up
    among them, so all are measured over the same span of a market of the same age. ``valid`` says whether settings
    can work (``spec.valid`` unless given). Raises ValueError when none is valid, or when these runs are too long
    to compare even two settings.
    """
    tried, count = candidates(base, grid, spec.valid if valid is None else valid, MAX_FIT_CANDIDATES - 1, seed)
    if not tried:
        raise ValueError("none of these combinations of settings is valid")
    warm_up = lambda values: spec.build(values).warm_up  # noqa: E731
    burn = max(map(warm_up, [base, *tried]))
    room = budget // (runs * (burn + span)) - 1  # the rest of the budget, after the user's settings
    if room < 1:
        raise ValueError(f"{runs} run{'' if runs == 1 else 's'} of {burn + span:,.0f} units of time ({burn:,.0f} to "
                         "warm up) are too long to compare settings: fewer runs, a shorter span, or a shorter "
                         "longest window")
    if len(tried) > room:
        picked = random.Random(seed).sample(range(len(tried)), int(room))
        tried = [tried[i] for i in sorted(picked)]
        burn = max(map(warm_up, [base, *tried]))
    return ([] if base in tried else [base]) + tried, count, burn, count is None or len(tried) < count


def rank_candidates(tried: list[dict[str, Any]], runs: list[list[dict[str, float]]], yours: dict[str, Any],
                    objective: str) -> list[dict[str, Any]]:
    """One row per candidate, best first (ties: the one that traded less, then the user's settings), from its runs on
    the same seeds.

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
    rows.sort(key=lambda row: (-row["score"], row["lots"], not row["yours"]))
    return rows


@dataclass
class TrainingFit:
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


def _end_with_server() -> None:
    """End this worker when the server does, however that is: it waits for work for ever once the server is gone."""
    server = multiprocessing.parent_process()

    def watch() -> None:
        server.join()
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def _start_worker() -> None:
    _lower_priority()
    _end_with_server()


def _fit_pool() -> Executor:
    """Worker processes for one fit, spawned so that they inherit nothing of the running server."""
    return ProcessPoolExecutor(FIT_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_start_worker)


def _stop_workers(pool: Executor) -> None:
    """Let a pool go without waiting, and stop the runs that have begun, with their processes: they would go on to
    their ends, each taking a core, after the fit that wanted them has gone."""
    processes = list((getattr(pool, "_processes", None) or {}).values())  # a thread pool has none; shutdown drops them
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        process.terminate()


_fit_running = False  # one fit at a time on this server: it takes all the cores


def trades_from(trader: Any, now: float) -> float:
    """The time a trader has a signal from: what is left of its warm-up (``warming``), from ``now``."""
    warming = trader.warming
    return now + ((warming[1] - warming[0]) * trader.interval if warming else 0.0)


def fit_note(fit: Optional[dict[str, Any]], pnl: float, now: float, trades_from: float,
             settings: dict[str, Any]) -> str:
    """How a trader did in training, a unit of time, beside how it has done here since it had a signal."""
    f = fit
    if f is None:
        return ""
    se = f" (se {f['pnl_se'] / f['span']:.3f})" if f["pnl_se"] is not None else ""
    units = now - trades_from
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


class TrainedMarket(LiveMarket):
    """A live market whose traders can be fitted over training runs.

    A subclass has ``settings`` (the market's, as it makes a market) and ``start_trader(values, kind)``, which
    returns the runner it put in the market and how many points it learned from; a runner has ``fit``. It provides
    two hooks: ``plan(kind, base, grid, span, runs, seed)``, which says what a fit runs (what plan_fit returns), a
    method, or a static method if it needs nothing of the market; and ``training(kind, values, settings, seed, burn,
    span)``, a static method, which says how one candidate did alone in one training market (a dict of figures), and
    is a module-level function a spawned process can import.
    """

    def __init__(self, speed: int) -> None:
        super().__init__(speed)
        self.fitting: Optional[tuple[int, int]] = None  # (runs done, runs) while a trader is fitted
        self.last_fit: Optional[TrainingFit] = None
        self.fit_changes = 0  # incremented when what the last fit's panel says changes
        self._fit_seq = 0
        self._fit_cancel = False

    async def fit_trader(self, kind: str, base: dict[str, Any], grid: dict[str, list[Any]], span: int, runs: int,
                         objective: str, seed: int = 1) -> TrainingFit:
        """Try the user's settings ``base`` and the combinations of ``grid``'s values (the rest as in ``base``).

        Each candidate runs alone in ``runs`` fresh markets from the view's market settings, on the same seeds for
        all, which the live market can never have (training_seed), and is measured over the same ``span`` of time
        after the same warm-up (``plan``). The live market runs on meanwhile: this holds no lock. Raises
        ValueError for what ``plan`` refuses, RuntimeError while another fit runs or if the workers fail, and
        FitCanceled after cancel_fit.
        """
        global _fit_running
        if objective not in FIT_OBJECTIVES:
            raise ValueError(f"the objective must be one of {', '.join(FIT_OBJECTIVES)}")
        tried, valid, burn, sampled = self.plan(kind, base, grid, span, runs, seed)
        if _fit_running:
            raise RuntimeError("another fit is running")
        pool = _fit_pool()  # before the slot is taken: a pool that cannot be made leaves it free
        _fit_running = True
        self._fit_cancel = False
        settings = dict(self.settings)
        total = len(tried) * runs
        self.fitting = (0, total)
        self.changes += 1
        self.notify()
        began = time.monotonic()
        loop = asyncio.get_running_loop()
        where: dict[Any, tuple[int, int]] = {}
        try:
            where = {loop.run_in_executor(pool, self.training, kind, values, settings, training_seed(seed, r), burn,
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
            _stop_workers(pool)
            for future in where:
                future.cancel()  # also puts one that failed unseen out of reach of asyncio's "never retrieved" log
            _fit_running = False
            self.fitting = None
            self.changes += 1
            self.notify()
        self._fit_seq += 1
        self.last_fit = TrainingFit(id=self._fit_seq, kind=kind, grid=grid, settings=settings, objective=objective,
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

    def start_fitted(self, fit_id: int, rank: int) -> tuple[Any, int]:
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

    def shutdown(self) -> None:
        """Stop the run loop and any fit without waiting for them (when the session is evicted)."""
        self.cancel_fit()
        super().shutdown()
