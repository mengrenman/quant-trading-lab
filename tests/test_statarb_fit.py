"""Fitting a Stat-arb trader over training runs (StatArbSession.fit_trader, in app/statarb.py and app/training.py).

What the two views share is held here too: one fit at a time on a server, and the results panel."""

import asyncio
import gc
import html
import itertools
import json
import logging
import math
import multiprocessing
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from starlette.testclient import TestClient

from app import statarb as statarb_module
from app import training as training_module
from app.fitting import FIT_SEED, MAX_FIT_CANDIDATES, parse_grid
from app.main import app, registry, render_pairs_fit, render_statarb_fit
from app.pairs import MAX_TRADERS, TRADER_KINDS as PAIRS_KINDS, PairsSession
from app.sessions import COOKIE_NAME
from app.statarb import MARKET_PARAMS, MAX_FIT_BOOK_UNITS, TRADER_KINDS, StatArbSession, market_from, training_run
from app.training import (CLEARLY, FIT_RUNS, FIT_SPAN, FitCanceled, TrainingFit, fit_note, rank_candidates,
                          same_market, trades_from, training_seed)
from lob.crosssection import FlowTrader, LeadLagTrader, MomentumTrader, ReversalTrader
from lob.pairs import CopulaTrader
from lob.sparse_baskets import sparse_baskets
from lob.statarb import (WAIT, BookMarket, CopulaPortfolio, FactorValues, RelationTrader, ResidualTrader,
                         best_baskets, johansen_relations, mixed_loadings, pairs_by_distance, pairs_by_pvalue,
                         score_portfolio, sector_loadings)
from tests import workers
from tests.test_statarb_view import read_stream

DEFAULTS = {kind: {p.name: p.default for p in spec.params} for kind, spec in TRADER_KINDS.items()}
BASE = {p.name: p.default for p in MARKET_PARAMS}
MIXED = {**BASE, "shape": "mixed", "books": 3, "factors": 1}  # three books and one factor: two relations among them
SECTORS = {**BASE, "shape": "sectors", "sectors": 2, "books_each": 2}  # four books, a pair in each sector
OTHER = {**MIXED, "half_life": 30.0, "own_volatility": 0.6, "volatility": 1.3, "impact": 0.1, "table_seed": 5}
DRIFTING = {**MIXED, "reverting": False, "drifting": True, "loading_sd": 0.3, "loading_half_life": 200.0}
QUICK = {  # a minute's work in three books: short formations and windows, an interval of 5
    "pairs": {**DEFAULTS["pairs"], "count": 2, "formation": 60, "window": 20},
    "baskets": {**DEFAULTS["baskets"], "legs": 3, "count": 2, "formation": 60, "window": 20},
    "residuals": {**DEFAULTS["residuals"], "factors": 1, "window": 30, "pca_window": 60},
    "copula": {**DEFAULTS["copula"], "pairs": 2, "window": 60, "trading": 20},
    "reversal": {**DEFAULTS["reversal"], "lookback": 5, "hold": 5, "factors": 1, "pca_window": 30},
    "momentum": {**DEFAULTS["momentum"], "lookback": 5, "skip": 1, "hold": 5, "factors": 1, "pca_window": 30},
    "leadlag": {**DEFAULTS["leadlag"], "hold": 5, "factors": 1, "pca_window": 30, "interval": 5.0},
    "flow": {**DEFAULTS["flow"], "hold": 5, "factors": 1, "pca_window": 30},
}
GRID = {"window": [20, 30], "entry": [1.5, 2.0]}  # the pairs trader's, with QUICK's own settings in it


def after(decisions, interval=5.0):
    """The time this many decisions of an interval take, with the wait for the next event that each makes."""
    return decisions * (interval + WAIT)


ZSCORE = {**{p.name: p.default for p in PAIRS_KINDS["zscore"].params}, "window": 20}


@pytest.fixture(autouse=True)
def threads_for_workers(monkeypatch):
    """Training runs on threads, which share the test's process, so these tests need not spawn; one spawns anyway.

    The server's slot for a fit is put back as it was, whatever a failing test left in it."""
    monkeypatch.setattr(training_module, "_fit_pool", lambda: ThreadPoolExecutor(4))
    monkeypatch.setattr(training_module, "_fit_running", False)


def figures(pnl):
    """What a training run made, with the other figures made up from the P&L."""
    return {"pnl": pnl, "before_costs": pnl + 1.0, "costs": 1.0, "lots": 10.0, "entries": 2.0,
            "risk_adjusted": pnl / 1000}


class Gate:
    """Stands in for a training run: it keeps what it was asked, waits until the test opens it, and gives figures
    worked out from the settings and the run's number (``result``, which is told them)."""

    def __init__(self, result=lambda values, run: 0.0, opened=False):
        self.calls = []
        self.result = result
        self.open = threading.Event()
        self.permits = None  # a semaphore, to let the runs through one at a time
        if opened:
            self.open.set()

    def __call__(self, kind, values, settings, seed, burn, span):
        self.calls.append((kind, values, settings, seed, burn, span))
        if self.permits is not None:
            assert self.permits.acquire(timeout=30)
        else:
            self.open.wait(30)  # a test that forgets to open it fails on time, and does not hold up the exit
        return figures(self.result(values, seed % 2**32))  # a training seed is (fit seed + 1) * 2**32 + run


def by_window(values, run):
    """P&L from the settings and the run, to work ranks out by hand: 10 a unit of window, and what the rest adds."""
    return (10.0 * values.get("window", 0) + values.get("entry", 0.0) + 0.01 * values.get("formation", 0)
            + values.get("factors", 0) + 0.5 * run)


@pytest.fixture
def gate(monkeypatch):
    """Training runs that wait, in both views."""
    gate = Gate()
    monkeypatch.setattr(StatArbSession, "training", staticmethod(gate))
    monkeypatch.setattr(PairsSession, "training", staticmethod(gate))
    yield gate
    gate.open.set()


@pytest.fixture
def stand_in(monkeypatch):
    """Training runs that answer at once, in the Stat-arb view, with figures from by_window."""
    gate = Gate(by_window, opened=True)
    monkeypatch.setattr(StatArbSession, "training", staticmethod(gate))
    return gate


async def until(condition, seconds=30):
    """Wait for what a test depends on, rather than for as long as it ought to take."""
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.005)


def session_with(settings=MIXED):
    session = StatArbSession()
    session.configure(settings)
    return session


def fit_in(session, kind, base, grid, span=300, runs=2, objective="pnl", seed=1):
    return asyncio.run(session.fit_trader(kind, base, grid, span, runs, objective, seed))


# ---- What a training run is ----

def test_a_training_seed_is_never_one_the_live_market_can_have():
    seed = next(p for p in MARKET_PARAMS if p.name == "seed")
    assert training_seed(0, 0) > seed.high
    with pytest.raises(ValueError, match="between"):
        seed.parse(training_seed(0, 0))  # the form refuses it, so no view's market has it
    assert training_seed(3, 5) == 4 * 2**32 + 5
    seeds = {training_seed(s, r) for s in (0, 1, 2, FIT_SEED.high) for r in range(FIT_RUNS.high)}
    assert len(seeds) == 4 * FIT_RUNS.high  # and no two runs of any fits share one
    market = market_from(MIXED, training_seed(FIT_SEED.high, FIT_RUNS.high - 1))
    market.step()  # the largest still makes a market


def hand_market(settings, seed, trader):
    """The market the settings make, put together from the library's own parts."""
    if settings["shape"] == "mixed":
        loadings = mixed_loadings(settings["books"], settings["factors"], seed=settings["table_seed"])
    else:
        loadings = sector_loadings(settings["sectors"], settings["books_each"])
    fair = FactorValues(loadings, level=1000.0, volatility=settings["volatility"],
                        own_volatility=settings["own_volatility"],
                        half_life=settings["half_life"] if settings["reverting"] else None, impact=settings["impact"],
                        loading_sd=settings["loading_sd"] if settings["drifting"] else 0.0,
                        loading_half_life=settings["loading_half_life"])
    return BookMarket(fair, seed=seed, traders=[trader])


# What each kind of trader is made of, every setting away from its default: the market, the settings, the trader by
# hand, its warm-up by hand, and a seed it trades on.
BY_HAND = {
    "pairs": ("pairs", OTHER, {**DEFAULTS["pairs"], "count": 3, "formation": 60, "window": 20, "entry": 1.0,
                               "exit": 0.5, "refit": 40, "size": 3.0, "interval": 4.0},
              lambda: RelationTrader(choose=partial(pairs_by_distance, count=3), formation=60, window=20, entry=1.0,
                                     exit=0.5, refit=40, size=3.0, interval=4.0), 244.0, 5),
    "pairs by p-value": ("pairs", SECTORS, {**DEFAULTS["pairs"], "by_pvalue": True, "count": 2, "formation": 400,
                                            "window": 40, "entry": 1.0, "interval": 1.0},
                         lambda: RelationTrader(choose=partial(pairs_by_pvalue, count=2, every=1), formation=400,
                                                window=40, entry=1.0, size=2.5, interval=1.0), 401.0, 1),
    "pairs in drifting books": ("pairs", DRIFTING, {**DEFAULTS["pairs"], "count": 2, "formation": 60, "window": 20,
                                                    "entry": 1.0, "interval": 4.0},
                                lambda: RelationTrader(choose=partial(pairs_by_distance, count=2), formation=60,
                                                       window=20, entry=1.0, size=2.5, interval=4.0), 244.0, 5),
    "baskets of three": ("baskets", SECTORS, {**DEFAULTS["baskets"], "legs": 3, "count": 2, "formation": 2000,
                                              "window": 40, "entry": 1.0, "interval": 1.0},
                         lambda: RelationTrader(choose=partial(best_baskets, count=2, legs=3, every=1),
                                                formation=2000, window=40, entry=1.0, size=5.0, interval=1.0,
                                                rate=0.25),  # their default rate: a quarter of the way at a decision
                         2001.0, 1),
    "baskets on all the books": ("baskets", SECTORS, {**DEFAULTS["baskets"], "legs": 0, "count": 2, "formation": 2000,
                                                      "window": 40, "entry": 1.0, "interval": 1.0},
                                 lambda: RelationTrader(choose=partial(johansen_relations, count=2, every=1),
                                                        formation=2000, window=40, entry=1.0, size=5.0, interval=1.0,
                                                        rate=0.25), 2001.0, 1),
    "baskets of three by Box-Tiao": ("baskets", SECTORS, {**DEFAULTS["baskets"], "method": "box-tiao", "legs": 3,
                                                          "count": 2, "formation": 2000, "window": 40, "entry": 1.0,
                                                          "interval": 1.0},
                                     lambda: RelationTrader(choose=partial(sparse_baskets, count=2, legs=3, every=1),
                                                            formation=2000, window=40, entry=1.0, size=5.0,
                                                            interval=1.0, rate=0.25), 2001.0, 1),
    "baskets on all the books by Box-Tiao": ("baskets", SECTORS, {**DEFAULTS["baskets"], "method": "box-tiao",
                                                                  "legs": 0, "count": 2, "formation": 2000,
                                                                  "window": 40, "entry": 1.0, "interval": 1.0},
                                             lambda: RelationTrader(choose=partial(sparse_baskets, count=2, legs=0,
                                                                                   every=1),
                                                                    formation=2000, window=40, entry=1.0, size=5.0,
                                                                    interval=1.0, rate=0.25), 2001.0, 1),
    "residuals": ("residuals", OTHER, {**DEFAULTS["residuals"], "factors": 1, "window": 30, "pca_window": 60,
                                       "open_at": 1.0, "close_long": 0.25, "close_short": 0.5, "size": 4.0,
                                       "interval": 5.0},
                  lambda: ResidualTrader(factors=1, window=30, pca_window=60, open_at=1.0, close_long=0.25,
                                         close_short=0.5, size=4.0, interval=5.0), 310.0, 5),
    "copula": ("copula", OTHER, {**DEFAULTS["copula"], "pairs": 2, "window": 60, "trading": 20, "entry": 0.5,
                                 "exit": 0.1, "size": 3.0, "interval": 4.0},
               lambda: CopulaPortfolio(pairs=2, window=60, trading=20, entry=0.5, exit=0.1, size=3.0, interval=4.0),
               252.0, 5),
    "reversal": ("reversal", OTHER, {**DEFAULTS["reversal"], "lookback": 6, "hold": 4, "factors": 1, "pca_window": 40,
                                     "size": 4.0, "interval": 4.0},
                 lambda: ReversalTrader(lookback=6, hold=4, factors=1, pca_window=40, size=4.0, interval=4.0,
                                        rate=0.1),  # its default rate: a tenth of the way at each decision
                 168.0, 5),
    "momentum": ("momentum", OTHER, {**DEFAULTS["momentum"], "lookback": 6, "skip": 2, "hold": 4, "factors": 1,
                                     "pca_window": 40, "size": 4.0, "interval": 4.0},
                 lambda: MomentumTrader(lookback=6, skip=2, hold=4, factors=1, pca_window=40, size=4.0, interval=4.0),
                 168.0, 5),
    "lead-lag": ("leadlag", OTHER, {**DEFAULTS["leadlag"], "hold": 4, "ridge": 0.5, "factors": 1, "pca_window": 40,
                                    "size": 4.0, "interval": 4.0},
                 lambda: LeadLagTrader(lookback=1, hold=4, ridge=0.5, factors=1, pca_window=40, size=4.0,
                                       interval=4.0), 168.0, 5),
    "flow in trades": ("flow", OTHER, {**DEFAULTS["flow"], "measure": "trades", "lookback": 3, "hold": 4, "factors": 1,
                                       "pca_window": 40, "size": 4.0, "interval": 4.0},
                       lambda: FlowTrader(measure="trades", lookback=3, hold=4, factors=1, pca_window=40, size=4.0,
                                          interval=4.0), 168.0, 5),
    "flow in queues": ("flow", OTHER, {**DEFAULTS["flow"], "measure": "queues", "hold": 4, "factors": 0,
                                       "pca_window": 40, "size": 4.0, "interval": 4.0},
                       lambda: FlowTrader(measure="queues", lookback=5, hold=4, factors=0, pca_window=40, size=4.0,
                                          interval=4.0), 8.0, 5),
}


@pytest.mark.parametrize("case", BY_HAND)
def test_a_training_run_is_the_trader_alone_in_a_fresh_market(case):
    kind, settings, values, make, burn, seed = BY_HAND[case]
    trader = make()
    by_hand = score_portfolio(hand_market(settings, seed, trader), trader, burn, 300.0)
    assert training_run(kind, values, settings, seed, burn, 300.0) == by_hand  # number for number
    assert by_hand["lots"] > 0  # and it traded, so that says something
    assert by_hand["pnl"] == pytest.approx(by_hand["before_costs"] - by_hand["costs"])
    assert training_run(kind, values, settings, seed + 1, burn, 300.0) != by_hand


def test_a_training_run_has_the_trader_alone_and_runs_it_for_the_burn_and_the_span(monkeypatch):
    seen = []

    def spy(market, trader, burn, span):
        seen.append((market, trader, burn, span))
        return {"pnl": 1.0}

    monkeypatch.setattr(statarb_module, "score_portfolio", spy)
    assert training_run("residuals", QUICK["residuals"], SECTORS, 9, 123.0, 456.0) == {"pnl": 1.0}
    (market, trader, burn, span), = seen
    assert market.traders == [trader] and market.n == 4 and (burn, span) == (123.0, 456.0)
    assert isinstance(trader, ResidualTrader) and trader.pca_window == 60 and not trader.position.any()
    assert market.fair.k == 3 and market.time == 0.0  # sectors 2 by 2: the market's factor and a sector's each


def test_a_training_run_takes_its_markets_from_the_settings_it_is_given():
    values = QUICK["pairs"]
    one = training_run("pairs", values, MIXED, 3, 305.0, 300.0)
    assert training_run("pairs", values, {**MIXED, "half_life": 10.0}, 3, 305.0, 300.0) != one
    assert training_run("pairs", values, {**MIXED, "seed": 77}, 3, 305.0, 300.0) == one  # the seed setting is not it


# ---- When a trader can first trade ----

def decisions_until_first_order(trader, seed=4):
    """The times of a trader's decisions, in a market of its own, up to the one that made its first order."""
    times = []
    decide = trader.decide

    def counted(mids):
        times.append(market.time)
        return decide(mids)

    trader.decide = counted
    market = market_from(MIXED, seed, (trader,))
    while not trader.lots and market.time < 2_000:
        market.step()
    return times


def low_entry(trader):
    trader.entry = 1e-9  # beyond any z-score that is not zero
    return trader


def built(kind, **more):
    return TRADER_KINDS[kind].build({**QUICK[kind], **more})


def with_chosen(trader):
    trader.choose = lambda history: np.array([[1.0, -1.0, 0.0]])  # a relation at once: choosing is not what is timed
    return trader


def with_open_at_zero(trader):
    trader.open_at = 1e-9
    return trader


@pytest.mark.parametrize("make, decisions, intervals", [
    (lambda: low_entry(RelationTrader(weights=[[1.0, -1.0, 0.0]], window=30, interval=5.0)), 30, 31),
    (lambda: low_entry(built("pairs")), 60, 61),
    (lambda: low_entry(with_chosen(built("baskets"))), 60, 61),
    (lambda: with_open_at_zero(built("residuals")), 61, 62),
    (lambda: built("copula"), 62, 63),
], ids=["fixed weights", "pairs", "baskets", "residuals", "copula"])
def test_a_trader_has_made_its_first_order_before_its_warm_up_ends_and_not_long_before(make, decisions, intervals,
                                                                                        monkeypatch):
    monkeypatch.setattr(CopulaTrader, "decide", lambda self, a, b: 1)  # a pair of it always wants one more
    trader = make()
    warm_up = after(intervals)
    assert trader.warm_up == warm_up
    times = decisions_until_first_order(trader)
    # Its first order comes at a decision that is counted by hand: the window or the formation, the PCA window and
    # one, or (the copulas starting at the one after the window) the window and two.
    assert len(times) == decisions
    gaps = np.diff(times)
    assert (gaps >= 5.0).all() and (gaps < 6.0).all()  # a decision waits an interval, and then for the next event
    # It is made by the decision's number less one intervals in, and a little more; the warm-up is the next two
    # intervals later, and each decision's wait for an event, so that the first order, and the decision after it,
    # come before the measuring does.
    assert warm_up - 2 * 5.0 - decisions * WAIT <= times[-1] < warm_up - 5.0


@pytest.mark.parametrize("make", [
    lambda: low_entry(built("pairs", formation=300, window=30, interval=1.0)),
    lambda: with_open_at_zero(built("residuals", pca_window=300, interval=1.0)),
    lambda: built("copula", window=300, interval=1.0),
], ids=["pairs", "residuals", "copula"])
def test_many_short_decisions_still_make_the_first_order_before_the_warm_up_ends(make, monkeypatch):
    monkeypatch.setattr(CopulaTrader, "decide", lambda self, a, b: 1)
    trader = make()
    times = decisions_until_first_order(trader)
    # Each of some 300 decisions waits for the next event after its interval, some 0.02 units in these three books,
    # which adds up to more than the two intervals that the warm-up used to leave.
    assert times[-1] > (len(times) - 1) * 1.0 + 4.0
    assert times[-1] < trader.warm_up - 1.0


# ---- What a fit runs ----

def plan(session, kind, base, grid, span=300, runs=2, seed=1):
    return session.plan(kind, base, grid, span, runs, seed)


def test_a_plan_tries_the_grid_and_the_users_settings_once():
    session = session_with()
    base = QUICK["pairs"]  # window 20, entry 2: in the grid
    tried, valid, burn, sampled = plan(session, "pairs", base, GRID)
    assert tried == [{**base, "window": w, "entry": e} for w in (20, 30) for e in (1.5, 2.0)]  # the grid's order
    assert tried.count(base) == 1 and valid == 4 and burn == after(61) and not sampled
    theirs = {**base, "window": 40}  # not in it
    tried, valid, burn, sampled = plan(session, "pairs", theirs, GRID)
    assert tried[0] == theirs and len(tried) == 5 and valid == 4 and not sampled  # first, and not counted
    assert tried[1:] == [{**base, "window": w, "entry": e} for w in (20, 30) for e in (1.5, 2.0)]
    for alone in ([2.0], [2]):  # theirs is not tried twice, however it is written
        assert plan(session, "pairs", base, {"entry": alone})[0] == [base]


@pytest.mark.parametrize("kind, grid, burn", [
    ("pairs", {"formation": [60, 100], "interval": [2.0, 5.0]}, after(101)),  # 101 decisions of 5
    ("baskets", {"formation": [60, 100], "interval": [2.0, 5.0]}, after(101)),
    ("residuals", {"pca_window": [60, 100], "interval": [2.0, 5.0]}, after(102)),  # 100 and two, of 5
    ("copula", {"window": [60, 100], "interval": [2.0, 5.0]}, after(103)),  # 100 and three, of 5
])
def test_a_plan_gives_every_run_the_longest_warm_up_among_them(kind, grid, burn):
    session = session_with()
    assert plan(session, kind, QUICK[kind], grid)[2] == burn
    longer = {**QUICK[kind], **{name: max(values) for name, values in grid.items()}}
    assert plan(session, kind, longer, {"entry" if kind != "residuals" else "open_at": [1.5, 2.0]})[2] == burn
    # Theirs counts as well: a longer one than any in the grid.
    assert plan(session, "pairs", {**QUICK["pairs"], "formation": 200}, {"window": [20]})[2] == after(201)


@pytest.mark.parametrize("kind, change, grid, burn", [
    ("reversal", {}, {"pca_window": [60, 100], "interval": [2.0, 5.0]}, after(102)),  # 100 decisions, and two
    ("reversal", {}, {"lookback": [150, 200], "pca_window": [250]}, after(252)),  # the longer of the two
    ("momentum", {}, {"lookback": [60, 100], "skip": [0, 20], "pca_window": [200]}, after(202)),
    ("momentum", {"factors": 0}, {"lookback": [60, 100], "skip": [0, 20]}, after(122)),  # lookback and skip, and two
    ("leadlag", {}, {"pca_window": [60, 100], "interval": [2.0, 5.0]}, after(102)),
    ("leadlag", {"factors": 0}, {"pca_window": [60, 100]}, after(102)),  # its regression takes them without factors
    ("flow", {}, {"pca_window": [60, 100], "interval": [2.0, 5.0]}, after(102)),
    ("flow", {"measure": "trades", "factors": 0}, {"lookback": [5, 25]}, after(27)),
    ("flow", {"measure": "queues", "factors": 0}, {"lookback": [5, 25]}, after(2)),  # queues use no past at all
])
def test_a_plan_gives_every_run_the_longest_warm_up_among_them_for_the_kinds_that_score_every_book(
        kind, change, grid, burn):
    base = {**QUICK[kind], **change}
    tried, valid, burned, _ = plan(session_with(MIXED), kind, base, grid)
    assert burned == burn and valid == math.prod(len(values) for values in grid.values())
    assert burned == max(TRADER_KINDS[kind].build(values).warm_up for values in [base, *tried])


@pytest.mark.parametrize("kind", ["reversal", "momentum", "leadlag", "flow"])
def test_a_grid_that_asks_for_too_many_factors_for_the_books_loses_those_combinations(kind):
    session = session_with(MIXED)  # three books: a factor at most
    tried, valid, _, sampled = plan(session, kind, QUICK[kind], {"factors": [0, 1, 2, 3]})
    assert [values["factors"] for values in tried] == [0, 1] and valid == 2 and not sampled
    for base in (QUICK[kind], {**QUICK[kind], "factors": 0}):
        with pytest.raises(ValueError, match="^none of these combinations of settings is valid$"):
            plan(session, kind, base, {"factors": [2, 3]})
    with pytest.raises(ValueError, match="^your settings need fewer factors than the 3 books less one$"):
        fit_in(session, kind, {**QUICK[kind], "factors": 2}, {"hold": [5, 10]})
    assert session.fitting is None and session.last_fit is None and training_module._fit_running is False
    session.configure({**MIXED, "books": 4})  # four books: two factors, not three
    assert [v["factors"] for v in plan(session, kind, QUICK[kind], {"factors": [2, 3]})[0]] == [1, 2]


@pytest.mark.parametrize("kind, shown, grid", [
    ("reversal", ["Look back", "Hold for"], {"lookback": [5, 10], "hold": [5, 10]}),
    ("momentum", ["Look back", "Hold for"], {"lookback": [5, 10], "hold": [5, 10]}),
    ("leadlag", ["Hold for", "Ridge penalty"], {"hold": [5, 10], "ridge": [0.5, 1.0]}),
    ("flow", ["Hold for", "Trades over"], {"hold": [5, 10], "lookback": [3, 5]}),
])
def test_a_trader_that_scores_every_book_is_fitted_through_the_route_and_a_candidate_started(
        client, statarb, stand_in, kind, shown, grid):
    small_books(client)
    assert client.post("/api/statarb/traders/fit", data=fit_form(kind)).status_code == 200
    fit = statarb.last_fit
    assert fit.kind == kind and fit.grid == grid and fit.burn == after(32)
    assert sorted(tuple(c["values"][name] for name in grid) for c in fit.candidates) == sorted(
        itertools.product(*grid.values()))  # the grid's four, one of them the user's
    assert fit.yours["values"] == QUICK[kind]
    assert all(call[0] == kind and call[1]["interval"] == QUICK[kind]["interval"] for call in stand_in.calls)
    assert all(call[2] == MIXED and call[4] == after(32) for call in stand_in.calls)  # the books as they were set
    panel = render_statarb_fit(statarb)
    assert f"{TRADER_KINDS[kind].label} fitted over training runs" in panel
    for label in shown:
        assert f">{label}</th>" in panel
    reply = client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 2})
    assert reply.status_code == 200 and reply.json()["name"] == f"{TRADER_KINDS[kind].label} 1"
    runner = statarb.runners[0]
    chosen = fit.candidates[1]["values"]
    assert runner.values == chosen and runner.kind == kind and runner.fit["rank"] == 2 and runner.fit["of"] == 4
    assert all(getattr(runner.trader, name) == chosen[name] for name in grid)
    assert statarb.board()[0]["fit"].startswith("Fitted by P&L, 2nd of 4: ")


def test_a_flow_trader_keeps_what_it_reads_through_a_fit(client, statarb, stand_in):
    small_books(client)
    data = fit_form("flow", measure="trades")
    assert client.post("/api/statarb/traders/fit", data=data).status_code == 200
    assert {c["values"]["measure"] for c in statarb.last_fit.candidates} == {"trades"}
    assert {call[1]["measure"] for call in stand_in.calls} == {"trades"}
    assert 'data-statarb-fit-use="{&#34;' in render_statarb_fit(statarb) and "measure&#34;: &#34;trades" in (
        render_statarb_fit(statarb))
    assert client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 1}).status_code == 200
    assert statarb.runners[0].trader.measure == "trades"
    assert client.post("/api/statarb/traders/fit", data=fit_form("flow", measure="tea")).json()["errors"] == {
        "measure": "Reads: one of queues, trades"}


def test_a_baskets_trader_keeps_how_it_chooses_through_a_fit(client, statarb, stand_in):
    small_books(client)
    data = {"kind": "baskets", **form(QUICK["baskets"]), "method": "box-tiao", "fit_count": "on", "grid_count": "1, 2",
            "fit_entry": "on", "grid_entry": "1, 1.5", "span": 500, "runs": 2, "objective": "pnl", "fit_seed": 1}
    assert client.post("/api/statarb/traders/fit", data=data).status_code == 200  # the user's entry, 1, is one of them
    assert {c["values"]["method"] for c in statarb.last_fit.candidates} == {"box-tiao"}
    assert len(statarb.last_fit.candidates) == 4 and statarb.last_fit.yours["values"]["method"] == "box-tiao"
    assert statarb.last_fit.yours["values"]["entry"] == 1.0 and {c["values"]["rate"] for c in (
        statarb.last_fit.candidates)} == {0.25}  # what it does not try stays as the user set it: the default's rate
    assert {call[1]["method"] for call in stand_in.calls} == {"box-tiao"}
    assert "method&#34;: &#34;box-tiao" in render_statarb_fit(statarb)
    assert client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 1}).status_code == 200
    chosen = statarb.runners[0].trader.choose
    assert chosen.func is sparse_baskets and chosen.keywords["legs"] == 3 and statarb.runners[0].values["method"] == (
        "box-tiao")
    reply = client.post("/api/statarb/traders/fit", data={**data, "method": "tea"})
    assert reply.json()["errors"] == {"method": "Chosen by: one of johansen, box-tiao"}
    johansen = {k: v for k, v in data.items() if k != "method"}  # left out: the default, as before
    assert client.post("/api/statarb/traders/fit", data=johansen).status_code == 200
    assert {c["values"]["method"] for c in statarb.last_fit.candidates} == {"johansen"}


def test_a_basket_of_one_book_is_refused_wherever_it_is_made(client, statarb, stand_in):
    spec, one = TRADER_KINDS["baskets"], {**QUICK["baskets"], "legs": 1}
    assert not spec.valid(one) and spec.valid({**one, "legs": 0}) and spec.valid({**one, "legs": 2})
    with pytest.raises(ValueError, match="baskets of two books or more, or 0"):
        spec.build(one)
    session = session_with(MIXED)
    base = {**QUICK["baskets"], "legs": 2}
    tried, valid, _, _ = plan(session, "baskets", base, {"legs": [1, 2, 3]})
    assert [t["legs"] for t in tried] == [2, 3] and valid == 2  # one book is not a basket: left out of the grid
    with pytest.raises(ValueError, match="your settings need baskets of two books or more"):
        plan(session, "baskets", one, {"entry": [2.0]})  # and refused as the user's own, with the reason
    with pytest.raises(ValueError, match="baskets of two books or more"):
        session.start_trader(one, "baskets")  # or started by hand, which would stop the market at its next decision
    assert session._seq == 0 and session.runners == [] and session.market.traders == []
    small_books(client)
    data = {"kind": "baskets", **form(QUICK["baskets"]), "fit_legs": "on", "grid_legs": "1, 2, 3", "span": 500,
            "runs": 2, "objective": "pnl", "fit_seed": 1}
    reply = client.post("/api/statarb/traders/fit", data=data)
    assert reply.status_code == 200 and sorted(c["values"]["legs"] for c in statarb.last_fit.candidates) == [2, 3]
    reply = client.post("/api/statarb/traders/start", data={"kind": "baskets", **form({**QUICK["baskets"], "legs": 1})})
    assert reply.status_code == 422 and "baskets of two books or more" in reply.json()["error"]
    assert statarb.runners == []


def test_baskets_of_more_books_than_there_are_cannot_trade_and_are_left_out_or_refused():
    session = session_with(MIXED)  # three books
    tried, valid, _, _ = plan(session, "baskets", QUICK["baskets"], {"legs": [0, 3, 4]})
    assert sorted(t["legs"] for t in tried) == [0, 3] and valid == 2  # four books of three: left out, not counted
    four = {**QUICK["baskets"], "legs": 4}
    assert four["entry"] == 1.0  # the default's: so a grid of that entry alone is the user's own settings
    with pytest.raises(ValueError, match="your settings need baskets of at most the 3 books"):
        plan(session, "baskets", four, {"entry": [1.0]})
    with pytest.raises(ValueError, match="baskets of at most the 3 books"):
        session.start_trader(four, "baskets")
    assert session._seq == 0 and session.runners == [] and session.market.traders == []
    assert plan(session_with({**MIXED, "books": 4}), "baskets", four, {"entry": [1.0]})[0] == [four]  # four will do


@pytest.mark.parametrize("kind, grid, valid", [
    ("pairs", {"formation": [50, 100], "window": [20, 50, 80]}, [(50, 20), (50, 50), (100, 20), (100, 50), (100, 80)]),
    ("baskets", {"formation": [50, 100], "window": [20, 80]}, [(50, 20), (100, 20), (100, 80)]),
])
def test_a_plan_leaves_out_a_window_above_the_formation(kind, grid, valid):
    session = session_with()
    base = {**QUICK[kind], "formation": 100}
    tried, count, _, _ = plan(session, kind, base, grid)
    assert [(v["formation"], v["window"]) for v in tried if v != base] == [x for x in valid if x != (100, 20)]
    assert count == len(valid)


@pytest.mark.parametrize("kind, grid, valid", [
    ("pairs", {"entry": [1.0, 2.0], "exit": [0.0, 1.0]}, [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0)]),
    ("baskets", {"entry": [1.0, 2.0], "exit": [0.0, 1.0]}, [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0)]),
    ("copula", {"entry": [0.5, 1.0], "exit": [0.0, 0.5]}, [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5)]),
])
def test_a_plan_leaves_out_an_exit_level_at_or_above_the_entry_level(kind, grid, valid):
    session = session_with()
    base = {**QUICK[kind], "entry": 3.0}
    tried, count, _, _ = plan(session, kind, base, grid)
    assert [(v["entry"], v["exit"]) for v in tried if v != base] == valid and count == 3
    with pytest.raises(ValueError, match="none of these combinations"):
        plan(session, kind, base, {"entry": [1.0], "exit": [1.0]})  # level with each other: contradicting


def test_a_plan_leaves_out_residual_levels_that_do_not_close_below_the_opening():
    session = session_with()
    grid = {"open_at": [1.0, 2.0], "close_long": [0.5, 1.0], "close_short": [0.5, 2.0]}
    tried, count, _, _ = plan(session, "residuals", QUICK["residuals"], grid)
    chosen = [(v["open_at"], v["close_long"], v["close_short"]) for v in tried[1:]]
    assert chosen == [(1.0, 0.5, 0.5), (2.0, 0.5, 0.5), (2.0, 1.0, 0.5)] and count == 3
    assert tried[0] == QUICK["residuals"]  # theirs is not among them
    grid = {"window": [100, 300], "pca_window": [200, 500]}  # a PCA window at least as long as the window
    tried, count, _, _ = plan(session, "residuals", QUICK["residuals"], grid)
    assert [(v["window"], v["pca_window"]) for v in tried[1:]] == [(100, 200), (100, 500), (300, 500)] and count == 3


def test_a_plan_leaves_out_factors_at_or_above_the_books():
    base = QUICK["residuals"]
    tried, count, _, _ = plan(session_with(MIXED), "residuals", base, {"factors": [1, 2, 3, 4]})
    assert [v["factors"] for v in tried] == [1, 2] and count == 2  # three books: two factors at most
    tried, count, _, _ = plan(session_with(SECTORS), "residuals", base, {"factors": [2, 3, 4]})
    assert [v["factors"] for v in tried] == [1, 2, 3] and count == 2  # four books: theirs first
    with pytest.raises(ValueError, match="none of these combinations"):
        plan(session_with(MIXED), "residuals", base, {"factors": [3, 4]})
    # The other kinds do not mind how many factors there are.
    assert plan(session_with(MIXED), "pairs", QUICK["pairs"], {"count": [1, 12]})[1] == 2


def test_a_plan_draws_what_the_budget_allows_by_the_books_it_is_spread_over(monkeypatch):
    asked = []
    real = statarb_module.plan_fit

    def spy(spec, base, grid, span, runs, seed, budget, valid):
        asked.append((spec.key, base, grid, span, runs, seed, budget))
        return real(spec, base, grid, span, runs, seed, budget, valid)

    monkeypatch.setattr(statarb_module, "plan_fit", spy)
    base = DEFAULTS["pairs"]  # a formation of 2,000 decisions of 5: 2,001 of them to warm up
    grid = {"window": [200, 400, 800], "entry": [1.5, 2.0, 2.5], "count": [4, 8, 12]}
    cost = 3 * (after(2001) + 10_000)  # a candidate's three runs
    combos = [{**base, "window": w, "entry": e, "count": c} for w, e, c in itertools.product(*grid.values())]
    assert base in combos

    twelve = StatArbSession()  # twelve books
    tried, valid, burn, sampled = twelve.plan("pairs", base, grid, 10_000, 3, 1)
    assert asked[-1][-1] == MAX_FIT_BOOK_UNITS // 12 == 1_333_333 and burn == after(2001)
    room = int(1_333_333 // cost) - 1  # 22 candidates fit, and theirs is one
    assert room == 21 and valid == 27 and sampled
    picked = [combos[i] for i in sorted(random.Random(1).sample(range(27), room))]  # drawn from the seed
    assert tried == ([] if base in picked else [base]) + picked
    assert (room + 1) * cost * 12 <= MAX_FIT_BOOK_UNITS < (room + 2) * cost * 12  # as many as the budget has room for
    again = twelve.plan("pairs", base, grid, 10_000, 3, 1)[0]
    other = twelve.plan("pairs", base, grid, 10_000, 3, 2)[0]
    assert again == tried and other != tried and other == (
        [] if base in other else [base]) + [combos[i] for i in sorted(random.Random(2).sample(range(27), room))]

    three = session_with(MIXED)  # the same grid in three books has four times the room
    tried, valid, burn, sampled = three.plan("pairs", base, grid, 10_000, 3, 1)
    assert asked[-1][-1] == MAX_FIT_BOOK_UNITS // 3 == 5_333_333
    assert tried == combos and valid == 27 and not sampled  # all of them


def test_a_plan_keeps_to_the_most_candidates_a_fit_tries():
    grid = {"window": [10, 20, 30, 40, 50, 60, 70, 80], "entry": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]}
    base = {**QUICK["pairs"], "formation": 100}
    tried, valid, _, sampled = plan(session_with(), "pairs", base, grid, 300, 1)
    assert valid == 64 and sampled and base in tried
    assert len(tried) == MAX_FIT_CANDIDATES - 1 + (base not in tried[1:])  # theirs, and the rest of a fit's most


def test_a_plan_of_a_grid_too_large_to_count_says_it_is_a_sample():
    session = StatArbSession()
    session.configure(MIXED)
    values = [20, 21, 22, 23, 24, 25, 26, 27]
    grid = {"window": values, "formation": [60 + v for v in values], "refit": [100 * v for v in values],
            "count": [1, 2, 3, 4, 5, 6, 7, 8], "size": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}
    tried, count, _, sampled = session.plan("pairs", QUICK["pairs"], grid, 500, 2, 1)
    assert count is None and sampled and len(tried) <= MAX_FIT_CANDIDATES  # 32,768 combinations, none of them counted


def test_a_plan_says_when_the_runs_are_too_long_to_compare_anything():
    session = StatArbSession()  # twelve books: 1,333,333 units of a book's time for all the runs
    base = DEFAULTS["pairs"]  # 10,065 to warm up
    grid = {"window": [200, 400]}
    tried, _, _, sampled = plan(session, "pairs", base, grid, 656_601, 1)  # two of these fit: theirs, and one more
    assert sampled and len(tried) == 1 + (base not in tried[:1]) or len(tried) == 2
    with pytest.raises(ValueError, match="^1 run of 666,667 units of time \\(10,065 to warm up\\) are too long to "
                                         "compare settings: fewer runs, a shorter span, or a shorter longest window$"):
        plan(session, "pairs", base, grid, 656_602, 1)  # one more unit, and not even two
    with pytest.raises(ValueError, match="^3 runs of "):
        plan(session, "pairs", {**base, "formation": 4000, "interval": 50.0}, grid, 50_000, 3)


@pytest.mark.parametrize("kind, change, reason", [
    ("pairs", {"exit": 2.0, "entry": 2.0}, "0 <= exit < entry"),
    ("pairs", {"window": 100, "formation": 60}, "a formation period at least as long as the window"),
    ("baskets", {"window": 2}, "a window of 3 or more"),
    ("residuals", {"factors": 3}, "fewer factors than the 3 books"),
    ("residuals", {"open_at": 1.0, "close_long": 1.0}, "closing < opening"),
    ("residuals", {"window": 100, "pca_window": 60}, "a PCA window no shorter"),
    ("copula", {"entry": 1.0, "exit": 1.0}, "0 <= exit < entry"),
])
def test_the_users_own_settings_that_cannot_work_are_refused_with_the_reason(kind, change, reason):
    session = session_with(MIXED)
    with pytest.raises(ValueError, match=f"^your settings need .*{reason}"):
        plan(session, kind, {**QUICK[kind], **change}, {"entry" if kind != "residuals" else "window": [1.5, 2.0]})
    with pytest.raises(ValueError, match="your settings need"):
        fit_in(session, kind, {**QUICK[kind], **change}, {"entry" if kind != "residuals" else "window": [1.5, 2.0]})
    assert session.fitting is None and session.last_fit is None and training_module._fit_running is False


# ---- What each kind proposes to try ----

@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_every_value_a_kind_proposes_is_inside_its_settings_limits(kind):
    spec = TRADER_KINDS[kind]
    numbers = {p.name: p for p in spec.numbers}
    assert set(spec.fit_grid) <= set(numbers)
    for name, values in spec.fit_grid.items():
        param = numbers[name]
        assert len(values) == len(set(values)) and list(values) == sorted(values)
        for value in values:
            assert param.low <= value <= param.high and param.parse(value) == value
            assert not param.integer or float(value).is_integer()


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_what_a_kind_tries_unless_told_otherwise_is_two_settings_that_it_proposes_values_for(kind):
    spec = TRADER_KINDS[kind]
    assert len(spec.fit_default) == 2 and len(set(spec.fit_default)) == 2
    assert set(spec.fit_default) <= set(spec.fit_grid) and set(spec.fit_default) <= {p.name for p in spec.numbers}
    for name in spec.fit_default:  # the kind's own default is among them: the user's settings are a candidate
        assert next(p for p in spec.numbers if p.name == name).default in spec.fit_grid[name]


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_the_default_candidates_are_all_valid_in_twelve_books_and_within_the_budget(kind):
    spec = TRADER_KINDS[kind]
    grid = {name: list(spec.fit_grid[name]) for name in spec.fit_default}
    first, second = spec.fit_default
    combinations = len(grid[first]) * len(grid[second])  # 9, or 6 where the second is the interval: 1 and 5
    assert (second == "interval") == (combinations == 6)
    outside = {**DEFAULTS[kind], first: grid[first][0] + 1}  # the user's settings: in the grid, and not
    for base, tried_count in ((DEFAULTS[kind], combinations), (outside, combinations + 1)):
        tried, valid, burn, sampled = StatArbSession().plan(kind, base, grid, FIT_SPAN.default, FIT_RUNS.default, 1)
        assert not sampled and valid == combinations
        assert base in tried and len(tried) == tried_count and (tried[0] == base) == (base is outside)
        assert len(tried) * FIT_RUNS.default * (burn + FIT_SPAN.default) * 12 <= MAX_FIT_BOOK_UNITS


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_each_value_a_kind_proposes_suits_the_rest_of_its_defaults(kind):
    spec = TRADER_KINDS[kind]
    session = StatArbSession()
    for name, values in spec.fit_grid.items():
        for value in values:
            assert spec.valid({**DEFAULTS[kind], name: value}), (name, value)
            tried, count, _, _ = session.plan(kind, DEFAULTS[kind], {name: [value]}, 500, 1, 1)
            assert count == 1 and {**DEFAULTS[kind], name: value} in tried


def test_the_pages_fit_panel_has_each_kinds_proposed_values_and_ticks_its_defaults(client):
    page = client.get("/").text
    assert 'id="statarb-fit-panel"' in page and 'id="statarb-fit-result"' in page and 'id="statarb-fit-run"' in page
    for kind, spec in TRADER_KINDS.items():
        start = page.index(f'data-statarb-fit-kind="{kind}"')
        fieldset = page[start:page.index("</fieldset>", start)]
        for q in spec.numbers:
            shown = ",".join(str(v) for v in spec.fit_grid.get(q.name, (q.default,)))
            assert f'name="grid_{q.name}" value="{shown}"' in fieldset
            box = fieldset[fieldset.index(f'name="fit_{q.name}"'):].split(">")[0]
            assert ("checked" in box) == (q.name in spec.fit_default)
        # What the page sends by default parses back to the grid it was made from.
        form = {f"fit_{n}": "on" for n in spec.fit_default} | {
            f"grid_{q.name}": ",".join(str(v) for v in spec.fit_grid.get(q.name, (q.default,))) for q in spec.numbers}
        assert parse_grid(spec.numbers, form) == {n: list(spec.fit_grid[n]) for n in spec.fit_default}


# ---- The session's fit ----

def live_session():
    session = session_with(MIXED)
    session.start_trader(QUICK["pairs"], "pairs")
    session.run_for(50)
    return session


def state_of(session):
    """Everything of the live market and its trader that a fit could change."""
    m, t = session.market, session.runners[0].trader
    return (m.time, m.events, m.rng.bit_generator.state, m.fair._rng.bit_generator.state, m.fair.factors.tolist(),
            m.fair.fairs, [b.mid_price for b in m.books], t.position.tolist(), t.cash, t.lots, t.entries, t.next_time,
            t._seen, len(t._mids), len(session.mids), len(session.samples), session.runners[0].trades_from)


def test_a_fit_ranks_every_candidate_and_leaves_the_live_market_running_on_untouched():
    live, twin = live_session(), live_session()
    assert state_of(live) == state_of(twin)
    steps, fitting, queue = 0, [], live.subscribe()
    changes, fit_changes = live.changes, live.fit_changes

    async def main():
        nonlocal steps
        fit = asyncio.create_task(live.fit_trader("pairs", QUICK["pairs"], GRID, 300, 2, "pnl", 1))
        while not fit.done():
            if live.fitting:
                fitting.append(live.fitting)
            if steps < 600:  # the live market runs on meanwhile, to a time before any trader chooses
                for _ in range(20):
                    live._event()
                    steps += 1
            await asyncio.sleep(0.01)
        return fit.result()

    fit = asyncio.run(main())
    assert len(fit.candidates) == 4 and live.last_fit is fit and live.fitting is None
    assert live.fit_changes == fit_changes + 1  # the panel of results has something new to say
    assert fit.id == 1 and fit.kind == "pairs" and fit.grid == GRID and fit.settings == MIXED and fit.objective == "pnl"
    assert (fit.seed, fit.span, fit.burn, fit.runs) == (1, 300, after(61), 2)
    assert (fit.valid, fit.size, fit.sampled) == (4, 4, False)
    assert 0 < fit.seconds < 120
    assert fitting[0] == (0, 8) and fitting[-1][1] == 8  # four candidates of two runs
    assert all(a[0] <= b[0] for a, b in zip(fitting, fitting[1:], strict=False))
    assert live.changes >= changes + 2 and not queue.empty()  # said when it began and when it ended, to every stream
    scores = [c["score"] for c in fit.candidates]
    assert scores == sorted(scores, reverse=True) and sum(c["yours"] for c in fit.candidates) == 1
    assert fit.yours["values"] == QUICK["pairs"]
    for _ in range(steps):  # a twin that never fitted
        twin._event()
    assert steps > 0 and live.market.time > 50
    assert state_of(live) == state_of(twin)  # not one random number of the live market's was used


def test_a_fit_does_not_take_the_traders_on_the_board_into_its_runs():
    session = session_with(MIXED)
    alone = fit_in(session, "pairs", QUICK["pairs"], {"entry": [1.0, 2.0]})
    session.start_trader({**QUICK["pairs"], "entry": 1.0}, "pairs")
    session.run_for(100)
    crowded = fit_in(session, "pairs", QUICK["pairs"], {"entry": [1.0, 2.0]})
    assert [c["pnl"] for c in crowded.candidates] == [c["pnl"] for c in alone.candidates]


def test_a_fit_is_the_same_from_the_same_seed_and_other_from_another():
    def result(seed):
        fit = fit_in(StatArbSession(), "pairs", QUICK["pairs"], {"entry": [1.5, 2.0]}, seed=seed)
        return [(c["values"]["entry"], c["pnl"], c["costs"]) for c in fit.candidates]

    assert result(1) == result(1) and result(1) != result(2)


def test_a_fit_gives_every_candidate_the_same_seeds_settings_warm_up_and_span(stand_in):
    session = session_with(OTHER)
    base = QUICK["pairs"]  # a formation of 60 and a window of 20 and an entry of 2: in the grid
    grid = {"formation": [60, 80], "window": [20, 30], "entry": [1.5, 2.0]}
    fit = fit_in(session, "pairs", base, grid, span=300, runs=3, seed=7)
    calls = stand_in.calls
    assert len(calls) == 8 * 3
    assert {c[3] for c in calls} == {8 * 2**32, 8 * 2**32 + 1, 8 * 2**32 + 2}
    # The longest warm-up among them, 81 decisions of 5, for every run, in the settings the view had.
    assert all(c[0] == "pairs" and c[2] == OTHER and c[4] == after(81) and c[5] == 300 for c in calls)
    assert fit.burn == after(81)
    assert Counter((tuple(c[1].items()), c[3]) for c in calls).most_common(1)[0][1] == 1  # each once on each seed
    combos = [{**base, "formation": f, "window": w, "entry": e} for f in (60, 80) for w in (20, 30) for e in (1.5, 2.0)]
    assert {tuple(c[1].items()) for c in calls} == {tuple(v.items()) for v in combos}
    # Each candidate's runs are its own, in the order of the seeds: 10 a unit of window, the entry, a hundredth of
    # the formation, and half the run's number: so a mean of 0.5 more, and a standard error of 0.5 / sqrt(3).
    mean = {(v["formation"], v["window"], v["entry"]): 10 * v["window"] + v["entry"] + 0.01 * v["formation"] + 0.5
            for v in combos}
    keys = [(c["values"]["formation"], c["values"]["window"], c["values"]["entry"]) for c in fit.candidates]
    assert keys == sorted(mean, key=lambda key: -mean[key]) and keys[0] == (80, 30, 2.0)
    for row, key in zip(fit.candidates, keys, strict=True):
        assert row["pnl"] == pytest.approx(mean[key]) and row["pnl_se"] == pytest.approx(0.5 / 3**.5)
        assert row["against"] == pytest.approx(mean[key] - mean[(60, 20, 2.0)])  # seed by seed, the same run's term
        assert row["against_se"] == pytest.approx(0, abs=1e-9)
    assert fit.verdict == "clearly" and fit.yours["values"] == base


def test_a_candidate_is_measured_in_the_markets_a_run_by_hand_would_give():
    fit = fit_in(StatArbSession(), "pairs", QUICK["pairs"], {"entry": [1.5, 2.0]}, runs=2, seed=5)
    session = StatArbSession()
    for row in fit.candidates:
        by_hand = [training_run("pairs", row["values"], session.settings, training_seed(5, r), fit.burn,
                                fit.span)["pnl"] for r in range(2)]
        assert row["pnl"] == pytest.approx(sum(by_hand) / 2)
        assert row["pnl_se"] == pytest.approx(abs(by_hand[0] - by_hand[1]) / 2)  # two runs: half their difference


@pytest.mark.parametrize("kind, settings, grid, burn", [
    ("pairs", MIXED, {"entry": [1.0, 2.0]}, after(61)),
    ("baskets", SECTORS, {"entry": [1.0, 2.0]}, after(2001, 1.0)),
    ("residuals", MIXED, {"open_at": [1.0, 1.5]}, after(62)),
    ("copula", MIXED, {"entry": [0.5, 1.0]}, after(63)),
    ("reversal", MIXED, {"hold": [5, 10]}, after(32)),  # the 30 moves of the factors, and one, and one more
    ("momentum", MIXED, {"hold": [5, 10]}, after(32)),
    ("leadlag", MIXED, {"ridge": [0.1, 1.0]}, after(32)),
    ("flow", MIXED, {"hold": [5, 10]}, after(32)),
])
def test_every_kind_fits_and_the_users_row_is_a_run_by_hand(kind, settings, grid, burn):
    base = QUICK[kind] if kind != "baskets" else {**QUICK[kind], "formation": 2000, "interval": 1.0, "entry": 1.0}
    fit = fit_in(session_with(settings), kind, base, grid, span=200, runs=1, seed=3)
    assert fit.burn == burn and len(fit.candidates) == 2 and fit.kind == kind
    by_hand = training_run(kind, base, settings, training_seed(3, 0), burn, 200)
    assert fit.yours["pnl"] == by_hand["pnl"] and fit.yours["lots"] == by_hand["lots"]
    assert fit.yours["entries"] == by_hand["entries"] and fit.yours["costs"] == by_hand["costs"]
    assert fit.yours["pnl_se"] is None and fit.verdict in ("yours", "single")  # one run cannot say


def test_the_training_markets_follow_the_views_settings(stand_in):
    session = session_with(MIXED)
    fit = fit_in(session, "pairs", QUICK["pairs"], {"entry": [1.0, 2.0]}, runs=1)
    assert fit.settings == MIXED and all(c[2] == MIXED for c in stand_in.calls)
    session.configure(OTHER)  # new settings: the next fit is in the new markets
    fit = fit_in(session, "pairs", QUICK["pairs"], {"entry": [1.0, 2.0]}, runs=1)
    assert fit.settings == OTHER and all(c[2] == OTHER for c in stand_in.calls[2:]) and len(stand_in.calls) == 4


def test_a_fit_after_new_settings_is_the_run_by_hand_in_the_new_markets():
    session = StatArbSession()  # twelve books in sectors
    session.configure(OTHER)
    fit = fit_in(session, "pairs", QUICK["pairs"], {"entry": [1.0, 2.0]}, runs=1)
    by_hand = training_run("pairs", QUICK["pairs"], OTHER, training_seed(1, 0), fit.burn, fit.span)["pnl"]
    assert fit.yours["pnl"] == pytest.approx(by_hand)
    assert by_hand != training_run("pairs", QUICK["pairs"], MIXED, training_seed(1, 0), fit.burn, fit.span)["pnl"]


def test_a_fit_with_an_objective_it_does_not_know_runs_nothing(gate):
    with pytest.raises(ValueError, match="objective must be one of pnl, sharpe"):
        fit_in(session_with(), "pairs", QUICK["pairs"], GRID, objective="luck")
    assert gate.calls == [] and training_module._fit_running is False


def test_the_ranking_by_risk_adjusted_score_is_the_one_asked_for(stand_in):
    session = session_with()
    fit = fit_in(session, "pairs", QUICK["pairs"], {"window": [20, 30]}, runs=2, objective="sharpe")
    assert fit.objective == "sharpe" and [c["values"]["window"] for c in fit.candidates] == [30, 20]
    assert fit.candidates[0]["score"] == pytest.approx((300 + 2.0 + 0.6 + 0.25) / 1000)  # the score is pnl / 1000


# ---- One fit at a time on a server ----

def fit_args(session, runs=2):
    """What a session is asked to fit: a Stat-arb pairs trader or a z-score one, over a grid of two entries."""
    if isinstance(session, StatArbSession):
        return "pairs", QUICK["pairs"], {"entry": [1.5, 2.0]}, 500, runs, "pnl", 1
    return "zscore", ZSCORE, {"entry": [1.5, 2.0]}, 500, runs, "pnl", 1


@pytest.mark.parametrize("first, second", [(StatArbSession, PairsSession), (PairsSession, StatArbSession)])
def test_only_one_fit_runs_at_a_time_in_either_view(gate, first, second):
    one, two = first(), second()

    async def main():
        running = asyncio.create_task(one.fit_trader(*fit_args(one)))
        await until(lambda: one.fitting is not None)
        with pytest.raises(RuntimeError, match="^another fit is running$"):
            await two.fit_trader(*fit_args(two))  # not even from the other view
        assert two.fitting is None and two.last_fit is None and training_module._fit_running is True
        assert two.cancel_fit() is False and one.fitting is not None  # the other view cannot stop it
        assert one.cancel_fit() is True
        with pytest.raises(FitCanceled):
            await running
        assert training_module._fit_running is False

    asyncio.run(main())
    gate.open.set()
    assert asyncio.run(two.fit_trader(*fit_args(two))).id == 1  # free again, for either view


def test_canceling_keeps_nothing_and_leaves_room_for_the_next(gate):
    session = session_with()
    fit_changes = session.fit_changes
    assert session.cancel_fit() is False  # nothing to stop, and that does not stop the next fit either

    async def main():
        fit = asyncio.create_task(session.fit_trader("pairs", QUICK["pairs"], GRID, 300, 4, "pnl", 1))
        await until(lambda: gate.calls)  # a run has begun
        assert session.cancel_fit() is True
        with pytest.raises(FitCanceled):
            await fit

    asyncio.run(main())
    assert session.last_fit is None and session.fitting is None and session.fit_changes == fit_changes
    assert training_module._fit_running is False
    gate.open.set()
    assert fit_in(session, "pairs", QUICK["pairs"], GRID).id == 1  # the one that was canceled took no number
    assert PairsSession().cancel_fit() is False


def test_every_run_that_ends_tells_the_streams_how_far_the_fit_has_got(gate, monkeypatch):
    session = session_with()
    gate.permits = threading.Semaphore(0)  # the runs end one at a time, when this test says
    told = []
    notify = session.notify
    monkeypatch.setattr(session, "notify", lambda: (told.append(session.fitting), notify()))
    changes, fit_changes = session.changes, session.fit_changes

    async def main():
        fit = asyncio.create_task(session.fit_trader("pairs", QUICK["pairs"], {"entry": [1.5, 2.0]}, 500, 2, "pnl", 1))
        await until(lambda: session.fitting is not None)
        assert session.fitting == (0, 4) and session.changes == changes + 1 and told == [(0, 4)]  # it has begun
        for done in range(1, 4):
            gate.permits.release()
            await until(lambda done=done: session.fitting == (done, 4))
            assert session.changes == changes + 1 + done and told[-1] == (done, 4)
        gate.permits.release()
        return await fit

    fit = asyncio.run(main())
    assert fit.id == 1 and session.fitting is None and session.last_fit is fit
    assert session.changes == changes + 1 + 4 + 1 and session.fit_changes == fit_changes + 1
    assert told == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), None, None]  # and when it ended, and its results came


def test_the_pool_of_a_fit_is_let_go_without_waiting_and_without_the_runs_not_yet_begun(monkeypatch):
    ended = []

    class Pool(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            ended.append((wait, cancel_futures))
            super().shutdown(wait=False, cancel_futures=cancel_futures)

    monkeypatch.setattr(training_module, "_fit_pool", lambda: Pool(2))
    session = session_with()
    fit_in(session, "pairs", QUICK["pairs"], {"entry": [2.0]}, runs=1)
    assert ended == [(False, True)]
    monkeypatch.setattr(training_module, "_fit_pool", lambda: Pool(2))
    gate = Gate()
    monkeypatch.setattr(StatArbSession, "training", staticmethod(gate))

    async def main():
        fit = asyncio.create_task(session.fit_trader(*fit_args(session)))
        await until(lambda: gate.calls)
        session.cancel_fit()
        with pytest.raises(FitCanceled):
            await fit

    try:
        asyncio.run(main())
    finally:
        gate.open.set()
    assert ended == [(False, True)] * 2


def test_a_fit_looks_for_a_cancel_at_least_twice_a_second(gate, monkeypatch):
    waits = []
    real = asyncio.wait

    def spy(*args, **kwargs):
        waits.append(kwargs.get("timeout"))
        return real(*args, **kwargs)

    monkeypatch.setattr(training_module.asyncio, "wait", spy)
    session = session_with()

    async def main():
        fit = asyncio.create_task(session.fit_trader(*fit_args(session)))
        await until(lambda: len(waits) >= 2)  # it has looked, and looked again
        session.cancel_fit()
        with pytest.raises(FitCanceled):
            await fit

    asyncio.run(main())
    assert all(timeout is not None and 0 < timeout <= 0.5 for timeout in waits)


def test_a_worker_that_dies_says_so_and_frees_the_slot_for_either_view(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    class Broken(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            def boom():
                raise BrokenProcessPool("a worker died")
            return super().submit(boom)

    monkeypatch.setattr(training_module, "_fit_pool", lambda: Broken(1))
    session = session_with()
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        fit_in(session, "pairs", QUICK["pairs"], GRID)
    assert session.fitting is None and session.last_fit is None and training_module._fit_running is False
    monkeypatch.setattr(training_module, "_fit_pool", lambda: ThreadPoolExecutor(2))
    fit_in(session, "pairs", QUICK["pairs"], GRID)
    asyncio.run(PairsSession().fit_trader("zscore", ZSCORE, {"entry": [2.0]}, 500, 1, "pnl", 1))


@pytest.mark.parametrize("view", ["statarb", "pairs"])
def test_evicting_the_session_stops_its_fit_and_frees_the_slot(client, gate, view):
    manager = registry.get(client.cookies[COOKIE_NAME])
    session = getattr(manager, view)
    args = ("pairs", QUICK["pairs"], GRID, 300, 4, "pnl", 1) if view == "statarb" else (
        "zscore", ZSCORE, {"entry": [1.5, 2.0]}, 500, 4, "pnl", 1)

    async def main():
        fit = asyncio.create_task(session.fit_trader(*args))
        await until(lambda: gate.calls)
        manager.shutdown()
        with pytest.raises(FitCanceled):
            await fit

    asyncio.run(main())
    assert training_module._fit_running is False and session.fitting is None and session.last_fit is None


def test_the_workers_are_spawned_one_fewer_than_the_cores_and_give_way_to_the_server(monkeypatch):
    monkeypatch.undo()
    cores = os.cpu_count()
    assert training_module.FIT_WORKERS == (cores - 1 if cores > 1 else 1)  # the server keeps one
    made = []

    class Pool:
        def __init__(self, workers, **kwargs):
            made.append((workers, kwargs))

    monkeypatch.setattr(training_module, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(training_module, "FIT_WORKERS", 7)
    training_module._fit_pool()
    (workers, kwargs), = made
    assert workers == 7 and kwargs["mp_context"].get_start_method() == "spawn"  # nothing of the server is inherited
    assert set(kwargs) == {"mp_context", "initializer"} and kwargs["initializer"] is training_module._start_worker
    nices = []
    monkeypatch.setattr(training_module.os, "nice", nices.append)
    training_module._lower_priority()
    assert nices == [10]


def test_the_training_runs_work_in_real_worker_processes(monkeypatch):
    monkeypatch.undo()  # the real pool: its function and arguments must cross to a spawned process, and back
    monkeypatch.setattr(training_module, "FIT_WORKERS", 2)
    fit = fit_in(session_with(MIXED), "residuals", QUICK["residuals"], {"open_at": [1.0, 1.5]}, span=300, runs=2)
    assert len(fit.candidates) == 2 and fit.burn == after(62)
    for row in fit.candidates:
        by_hand = [training_run("residuals", row["values"], MIXED, training_seed(1, r), fit.burn, 300)["pnl"]
                   for r in range(2)]
        assert row["pnl"] == pytest.approx(sum(by_hand) / 2, abs=1e-9) and by_hand[0] != by_hand[1]


def begun(directory):
    """The processes of runs that have begun, from the files ``workers.waiting`` leaves."""
    return sorted(int(path.name) for path in directory.iterdir())


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_canceling_a_fit_stops_the_runs_that_have_begun_and_their_processes(monkeypatch, tmp_path):
    monkeypatch.undo()  # the real pool, with workers that wait two minutes
    monkeypatch.setenv("LOB_TEST_BEGUN", str(tmp_path))
    monkeypatch.setattr(training_module, "FIT_WORKERS", 2)
    monkeypatch.setattr(StatArbSession, "training", staticmethod(workers.waiting))
    session = session_with(MIXED)

    async def main():
        fit = asyncio.create_task(session.fit_trader("pairs", QUICK["pairs"], {"entry": [1.5, 2.0]}, 300, 1, "pnl", 1))
        children = []
        try:
            await until(lambda: len(begun(tmp_path)) == 2, seconds=90)  # a run in each of two workers
            children = multiprocessing.active_children()
            assert len(children) == 2
            session.cancel_fit()
            with pytest.raises(FitCanceled):
                await fit
            await until(lambda: not any(child.is_alive() for child in children), seconds=20)  # not after two minutes
        finally:
            for child in children:
                child.kill()
            fit.cancel()

    asyncio.run(main())
    assert training_module._fit_running is False


def test_workers_end_with_the_server_that_made_them_even_when_it_is_killed(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = ("import time\nfrom app import training\nfrom tests import workers\ntraining.FIT_WORKERS = 2\n"
              "pool = training._fit_pool()\n"
              "for _ in range(2):\n    pool.submit(workers.waiting, 'pairs', {}, {}, 0, 0.0, 0.0)\ntime.sleep(300)\n")
    server = subprocess.Popen([sys.executable, "-c", script], cwd=root,
                              env={**os.environ, "LOB_TEST_BEGUN": str(tmp_path), "PYTHONPATH": str(root)})
    pids = []
    try:
        deadline = time.monotonic() + 90
        while len(pids) < 2:
            assert time.monotonic() < deadline and server.poll() is None
            time.sleep(0.1)
            pids = begun(tmp_path)
        server.send_signal(signal.SIGKILL)  # nothing of the server's own shutdown runs
        server.wait()
        deadline = time.monotonic() + 30
        while any(alive(pid) for pid in pids):
            assert time.monotonic() < deadline, "the workers outlived the server"
            time.sleep(0.1)
    finally:
        server.kill()
        for pid in pids:
            if alive(pid):
                os.kill(pid, signal.SIGKILL)


def test_a_pool_that_cannot_be_made_leaves_the_slot_free(monkeypatch, stand_in):
    def broken():
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(training_module, "_fit_pool", broken)
    session = session_with()
    with pytest.raises(OSError, match="Too many open files"):
        fit_in(session, "pairs", QUICK["pairs"], GRID)
    assert training_module._fit_running is False and session.fitting is None and session.last_fit is None
    monkeypatch.setattr(training_module, "_fit_pool", lambda: ThreadPoolExecutor(2))
    assert fit_in(session, "pairs", QUICK["pairs"], GRID).id == 1  # this view's next fit, and any other view's
    assert asyncio.run(PairsSession().fit_trader("zscore", ZSCORE, {"entry": [2.0]}, 500, 1, "pnl", 1)).id == 1


def test_runs_that_fail_together_are_not_logged_as_errors_nobody_asked_for(monkeypatch, caplog):
    from concurrent.futures.process import BrokenProcessPool

    together = threading.Barrier(4)

    class Broken(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            def boom():
                together.wait(10)
                raise BrokenProcessPool("a worker died")
            return super().submit(boom)

    monkeypatch.setattr(training_module, "_fit_pool", lambda: Broken(4))
    session = session_with()

    async def main():
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            await session.fit_trader(*fit_args(session))
        await asyncio.sleep(0.3)  # the server's loop goes on, and the runs that fail after the first end
        gc.collect()  # a future that nobody looked at says so when it is collected

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        asyncio.run(main())
    assert [r.getMessage() for r in caplog.records if "never retrieved" in r.getMessage()] == []


# ---- The routes ----

@pytest.fixture
def client():
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def statarb(client):
    return registry.get(client.cookies[COOKIE_NAME]).statarb


@pytest.fixture
def pairs(client):
    return registry.get(client.cookies[COOKIE_NAME]).pairs


def form(values):
    """As the page sends them: a switch only when on."""
    return {k: ("on" if v is True else v) for k, v in values.items() if v is not False}


def small_books(client):
    assert client.post("/api/statarb/config", data=form(MIXED)).status_code == 200


def fit_form(kind="pairs", **more):
    grid = {"pairs": ("window", "20, 30", "entry", "1.5, 2"),
            "residuals": ("window", "30, 40", "open_at", "1, 2"),
            "reversal": ("lookback", "5, 10", "hold", "5, 10"),
            "momentum": ("lookback", "5, 10", "hold", "5, 10"),
            "leadlag": ("hold", "5, 10", "ridge", "0.5, 1"),
            "flow": ("hold", "5, 10", "lookback", "3, 5")}[kind]
    return {"kind": kind, **form(QUICK[kind]), f"fit_{grid[0]}": "on", f"grid_{grid[0]}": grid[1],
            f"fit_{grid[2]}": "on", f"grid_{grid[2]}": grid[3], "span": 500, "runs": 2, "objective": "pnl",
            "fit_seed": 1, **more}


def pairs_fit_form(**more):
    return {"kind": "zscore", **form(ZSCORE), "fit_window": "on", "grid_window": "10, 20", "fit_entry": "on",
            "grid_entry": "1.5, 2", "span": 500, "runs": 4, "objective": "pnl", "fit_seed": 1, **more}


def test_the_fit_route_fits_and_the_stream_brings_the_candidates(client, statarb, stand_in):
    seen = read_stream(client, {"statarb_board"})  # the last of a first push: a stream without the fit fails, not hangs
    assert "statarb_fit" not in seen and render_statarb_fit(statarb) == ""  # nothing before the first fit
    small_books(client)
    reply = client.post("/api/statarb/traders/fit", data=fit_form())
    assert reply.status_code == 200 and reply.json()["verdict"] == statarb.last_fit.verdict == "clearly"
    assert reply.json()["id"] == 1 and statarb.last_fit.kind == "pairs"
    assert re.fullmatch(r"Tried 4 settings, 2 runs each, in \d+ s: the results are above the traders\.",
                        reply.json()["message"])
    seen = read_stream(client, {"statarb_board"})
    assert 'id="statarb-fit-summary"' in seen["statarb_fit"] and 'data-fit-id="1"' in seen["statarb_fit"]
    second = client.post("/api/statarb/traders/fit", data=fit_form(runs=1, fit_window="on", grid_window="20")).json()
    assert second["id"] == 2 and second["message"].startswith("Tried 2 settings, 1 run each, in ")
    third = client.post("/api/statarb/traders/fit", data=fit_form(runs=1, grid_window="20", grid_entry="2")).json()
    assert third["message"].startswith("Tried 1 setting, 1 run each, in ")


def events_of(lines):
    def next_event():
        event = None
        for line in lines:
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: ") and event:
                return event, line[len("data: "):]

    return next_event


def test_the_stream_brings_the_fit_again_when_it_changes_and_when_the_settings_do(client, statarb, stand_in):
    small_books(client)
    with client.stream("GET", "/statarb/events") as resp:
        next_event = events_of(resp.iter_lines())
        while next_event()[0] != "statarb_board":  # the panels, on connect: no fit yet
            pass
        client.post("/api/statarb/traders/fit", data=fit_form())
        seen = {}
        while True:  # the fit comes before the panels of the same wake-up, so the board ends the run
            event, data = next_event()
            seen.setdefault(event, []).append(data)
            if event == "statarb_board" and "statarb_fit" in seen:
                break
        assert len(seen["statarb_fit"]) == 1 and 'data-fit-id="1"' in seen["statarb_fit"][0]
        assert "data-statarb-fit-stale" not in seen["statarb_fit"][0]
        assert json.loads(seen["statarb_state"][-1])["fitting"] is None  # and it is over

        client.post("/api/statarb/config", data=form({**MIXED, "half_life": 10.0}))  # other books, since
        seen = {}
        while True:
            event, data = next_event()
            seen.setdefault(event, []).append(data)
            if event == "statarb_board" and "statarb_fit" in seen:
                break
        assert "data-statarb-fit-stale" in seen["statarb_fit"][0] and 'data-fit-id="1"' in seen["statarb_fit"][0]


def test_a_fit_under_way_is_in_the_streams_state_and_status_and_gone_when_it_ends(client, statarb, gate):
    small_books(client)
    thread, out = start_in_thread(client, "/api/statarb/traders/fit", fit_form())
    wait(lambda: statarb.fitting is not None)
    assert statarb.fitting == (0, 8)  # four candidates of two runs: none done, the gate being shut
    seen = read_stream(client, {"statarb_status", "statarb_state"})
    assert json.loads(seen["statarb_state"])["fitting"] == [0, 8]
    assert 'data-fitting="0/8"' in seen["statarb_status"] and "Fitting 0/8" in seen["statarb_status"]
    gate.open.set()
    thread.join(30)
    assert not thread.is_alive() and out["reply"].status_code == 200 and statarb.fitting is None
    seen = read_stream(client, {"statarb_status", "statarb_state"})
    assert json.loads(seen["statarb_state"])["fitting"] is None and 'data-fitting=""' in seen["statarb_status"]


def test_the_status_says_a_fit_is_running(client, statarb):
    statarb.fitting = (3, 8)
    assert statarb.context()["fitting"] == (3, 8)
    seen = read_stream(client, {"statarb_status", "statarb_state"})
    assert 'data-fitting="3/8"' in seen["statarb_status"] and "Fitting 3/8" in seen["statarb_status"]
    assert json.loads(seen["statarb_state"])["fitting"] == [3, 8]
    statarb.fitting = None
    assert statarb.context()["fitting"] is None
    assert 'data-fitting=""' in read_stream(client, {"statarb_status"})["statarb_status"]


@pytest.mark.parametrize("change, name, message", [
    ({"grid_window": "10, nope"}, "grid_window", "enter a number"),
    ({"grid_window": ""}, "grid_window", "give the values"),
    ({"grid_window": "1, 2"}, "grid_window", "between"),
    ({"grid_entry": "1, 2, 3, 4, 5, 6, 7, 8, 9"}, "grid_entry", "at most"),
    ({"span": 100}, "span", "between"),
    ({"runs": 11}, "runs", "between"),
    ({"span": 600.5}, "span", "whole number"),
    ({"runs": 2.5}, "runs", "whole number"),
    ({"fit_seed": -1}, "fit_seed", "between"),
    ({"objective": "luck"}, "objective", "Best by"),
    ({"window": 1}, "window", "between"),
    ({"count": 99}, "count", "between"),
    ({"kind": "tea_leaves"}, "kind", "No such kind"),
])
def test_what_is_wrong_with_a_fit_is_said_against_its_field(client, statarb, stand_in, change, name, message):
    reply = client.post("/api/statarb/traders/fit", data={**fit_form(), **change})
    assert reply.status_code == 422 and message in reply.json()["errors"][name]
    assert statarb.last_fit is None and statarb.fitting is None and stand_in.calls == []


@pytest.mark.parametrize("change, code", [({"span": 499}, 422), ({"span": 500}, 200), ({"span": 50_000}, 200),
                                          ({"span": 50_001}, 422), ({"runs": 0}, 422), ({"runs": 1}, 200),
                                          ({"runs": 10}, 200), ({"runs": 11}, 422), ({"fit_seed": 0}, 200),
                                          ({"fit_seed": 2**31 - 1}, 200), ({"fit_seed": 2**31}, 422)])
def test_a_fit_takes_the_lengths_and_runs_and_seeds_at_their_limits_and_no_others(client, stand_in, change, code):
    small_books(client)
    data = {**fit_form(grid_window="20", grid_entry="2"), **change}  # one candidate: theirs
    assert client.post("/api/statarb/traders/fit", data=data).status_code == code


def test_the_fit_panel_starts_from_a_run_of_10000_units_three_runs_and_seed_1(client):
    page = client.get("/").text
    panel = page[page.index('id="statarb-fit-panel"'):]
    for name, value, low, high in (("span", 10_000, 500, 50_000), ("runs", 3, 1, 10), ("fit_seed", 1, 0, 2**31 - 1)):
        tag = re.search(rf'<input[^>]*name="{name}"[^>]*>', panel).group(0)
        assert f'value="{value}"' in tag and f'min="{low}"' in tag and f'max="{high}"' in tag
    assert (FIT_SPAN.default, FIT_RUNS.default) == (10_000, 3) and 'data-most="47"' in panel


def test_a_fit_with_nothing_ticked_says_so(client, statarb):
    form_ = fit_form()
    del form_["fit_window"], form_["fit_entry"]
    reply = client.post("/api/statarb/traders/fit", data=form_)
    assert reply.status_code == 422 and "Tick at least one" in reply.json()["errors"]["fit"]
    reply = client.post("/api/statarb/traders/fit", data=fit_form(fit_by_pvalue="on", grid_by_pvalue="1, 2"))
    assert reply.status_code == 200  # a switch is not something to try values of: its box is not heard of


@pytest.mark.parametrize("kind, change, words", [
    ("pairs", {"exit": 3.0, "entry": 2.0}, "0 <= exit < entry"),
    ("pairs", {"window": 100}, "formation period at least as long as the window"),
    ("residuals", {"factors": 3}, "fewer factors than the 3 books"),
    ("residuals", {"open_at": 1.0, "close_long": 1.0}, "closing < opening"),
    ("reversal", {"factors": 2}, "fewer factors than the 3 books less one"),
    ("momentum", {"factors": 2}, "fewer factors than the 3 books less one"),
    ("leadlag", {"factors": 2}, "fewer factors than the 3 books less one"),
    ("flow", {"factors": 2}, "fewer factors than the 3 books less one"),
    ("reversal", {"lookback": 30}, "a PCA window of 10 or more, and more than the lookback"),
    ("momentum", {"lookback": 29}, "a PCA window above the lookback and the skip together"),  # 29, 1 and 1: 31
])
def test_settings_that_cannot_work_are_refused_with_the_reason(client, statarb, stand_in, kind, change, words):
    small_books(client)
    reply = client.post("/api/statarb/traders/fit", data=fit_form(kind, **change))
    assert reply.status_code == 422 and reply.json()["error"].startswith("Not fitted: your settings need ")
    assert words in reply.json()["error"] and statarb.last_fit is None and stand_in.calls == []


def test_combinations_that_cannot_work_and_runs_too_long_are_refused(client, statarb, stand_in):
    reply = client.post("/api/statarb/traders/fit", data=fit_form(fit_exit="on", grid_exit="5", grid_entry="1, 2"))
    assert reply.status_code == 422 and "none of these combinations" in reply.json()["error"]
    reply = client.post("/api/statarb/traders/fit", data=fit_form(formation=4000, interval=50, span=50_000, runs=3))
    assert reply.status_code == 422 and "3 runs of 250,170 units of time (200,170 to warm up) are too long" in (
        reply.json()["error"]) and stand_in.calls == []  # twelve books


def test_a_fit_without_a_kind_is_of_pairs(client, statarb, stand_in):
    small_books(client)
    data = fit_form()
    del data["kind"]
    assert client.post("/api/statarb/traders/fit", data=data).status_code == 200 and statarb.last_fit.kind == "pairs"


def test_canceling_when_nothing_runs_says_so(client):
    assert client.post("/api/statarb/traders/fit/cancel").json() == {"canceled": False}


def test_a_second_fit_at_once_is_refused(client, statarb, monkeypatch):
    monkeypatch.setattr(training_module, "_fit_running", True)
    reply = client.post("/api/statarb/traders/fit", data=fit_form())
    assert reply.status_code == 409 and reply.json()["error"] == "Not fitted: another fit is running."
    assert statarb.fitting is None and training_module._fit_running is True  # not the refused one's to free


def start_in_thread(client, url, data):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("reply", client.post(url, data=data)))
    thread.start()
    return thread, out


def wait(condition, seconds=30):
    asyncio.run(until(condition, seconds))


@pytest.mark.parametrize("first, second", [("pairs", "statarb"), ("statarb", "pairs")])
def test_a_fit_running_in_one_view_refuses_one_in_the_other_until_it_is_canceled(client, statarb, pairs, gate,
                                                                                 first, second):
    small_books(client)
    views = {"pairs": ("/api/pairs/traders/fit", pairs_fit_form(), pairs), "statarb": (
        "/api/statarb/traders/fit", fit_form(), statarb)}
    (url_1, form_1, one), (url_2, form_2, two) = views[first], views[second]
    thread, out = start_in_thread(client, url_1, form_1)
    wait(lambda: one.fitting is not None)  # it is under way: its runs wait at the gate
    reply = client.post(url_2, data=form_2)
    assert reply.status_code == 409 and reply.json()["error"] == "Not fitted: another fit is running."
    assert two.fitting is None and two.last_fit is None and one.fitting is not None
    assert client.post(url_2 + "/cancel").json() == {"canceled": False}  # the other view has none to cancel
    assert one.fitting is not None and training_module._fit_running is True
    assert client.post(url_1 + "/cancel").json() == {"canceled": True}
    thread.join(30)
    assert not thread.is_alive() and out["reply"].json() == {"canceled": True, "message": (
        "Canceled: nothing was kept of those runs.")}
    assert one.fitting is None and one.last_fit is None and training_module._fit_running is False
    gate.open.set()  # the slot is free: the other view's fit goes through
    assert client.post(url_2, data=form_2).status_code == 200 and two.last_fit is not None


def test_a_worker_that_dies_is_said_by_the_route_and_the_other_view_can_fit_after(client, statarb, pairs, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    class Broken(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            def boom():
                raise BrokenProcessPool("a worker died")
            return super().submit(boom)

    monkeypatch.setattr(training_module, "_fit_pool", lambda: Broken(1))
    small_books(client)
    reply = client.post("/api/statarb/traders/fit", data=fit_form())
    assert reply.status_code == 409 and reply.json() == {
        "error": "Not fitted: the worker processes running the training stopped unexpectedly."}
    assert statarb.fitting is None and statarb.last_fit is None and training_module._fit_running is False
    monkeypatch.setattr(training_module, "_fit_pool", lambda: ThreadPoolExecutor(2))
    assert client.post("/api/pairs/traders/fit", data=pairs_fit_form(runs=1)).status_code == 200
    assert pairs.last_fit is not None


def test_a_training_that_cannot_work_in_its_worker_frees_the_slot(client, statarb, monkeypatch):
    def boom(*args):
        raise ValueError("no such market")

    monkeypatch.setattr(StatArbSession, "training", staticmethod(boom))
    small_books(client)
    reply = client.post("/api/statarb/traders/fit", data=fit_form())
    assert reply.status_code == 422 and reply.json() == {"error": "Not fitted: no such market."}
    assert training_module._fit_running is False and statarb.fitting is None and statarb.last_fit is None


# ---- Starting a candidate ----

def test_a_candidate_starts_and_its_row_keeps_what_it_made_in_training_through_a_reset(client, statarb, stand_in):
    small_books(client)
    client.post("/api/statarb/traders/fit", data=fit_form())
    fit = statarb.last_fit
    reply = client.post("/api/statarb/traders/start", data={"fit_id": fit.id, "rank": 2})
    assert reply.status_code == 200 and reply.json()["name"] == "Pairs 1"
    runner = statarb.runners[0]
    assert runner.values == fit.candidates[1]["values"] and runner.kind == "pairs"
    assert runner.fit["rank"] == 2 and runner.fit["of"] == 4 and runner.fit["settings"] == MIXED
    # The candidates by hand: 10 a unit of window, the entry, 0.6 for the formation of 60 and a quarter for the runs
    # (0 and a half): 302.85, 302.35, 202.85 and 202.35.
    assert [c["values"]["window"] for c in fit.candidates] == [30, 30, 20, 20]
    assert [c["values"]["entry"] for c in fit.candidates] == [2.0, 1.5, 2.0, 1.5]
    pnl = 302.35  # the second best, its two runs 302.1 and 302.6
    assert fit.candidates[1]["pnl"] == pytest.approx(pnl) and fit.candidates[1]["pnl_se"] == pytest.approx(0.25)
    note = statarb.board()[0]["fit"]
    assert note == (f"Fitted by P&L, 2nd of 4: {pnl / 500:+.3f} a unit in 2 training runs (se {0.25 / 500:.3f}); "
                    "nothing yet here.")  # +0.605, se 0.001: per unit of time, the span being 500 units
    assert runner.trades_from == 59 * 5.0  # it learned the chart's one point: 59 decisions of 5 to go
    client.post("/api/statarb/control", data={"action": "step", "units": 450})
    row = statarb.board()[0]
    units = statarb.market.time - runner.trades_from
    assert units == pytest.approx(450 - 295, abs=1.0)  # the step ends at the first event past 450
    assert f"; {row['pnl'] / units:+.3f} a unit here over {units:,.0f} units." in row["fit"]
    client.post("/api/statarb/control", data={"action": "reset"})
    again = statarb.runners[0]
    assert again is runner and again.fit["rank"] == 2 and again.trades_from == 300.0  # it watches again
    assert statarb.board()[0]["fit"].endswith("nothing yet here.")
    assert "data-statarb-fitted" in read_stream(client, {"statarb_board"})["statarb_board"]


def test_a_trader_started_by_hand_has_no_fit_note(client, statarb):
    client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **QUICK["pairs"]}))
    assert statarb.board()[0]["fit"] == ""
    assert "data-statarb-fitted" not in read_stream(client, {"statarb_board"})["statarb_board"]


def test_the_note_says_when_the_books_have_other_settings_than_the_runs_had(client, statarb, stand_in):
    small_books(client)
    client.post("/api/statarb/traders/fit", data=fit_form())
    client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 1})
    other = "Those runs had other market settings."
    assert other not in statarb.board()[0]["fit"]
    client.post("/api/statarb/config", data=form({**MIXED, "seed": 9}))  # another seed: the same market
    assert other not in statarb.board()[0]["fit"] and "data-statarb-fit-stale" not in render_statarb_fit(statarb)
    client.post("/api/statarb/config", data=form({**MIXED, "own_volatility": 0.7}))
    assert statarb.board()[0]["fit"].endswith(f"nothing yet here. {other}")
    assert "data-statarb-fit-stale" in render_statarb_fit(statarb)


def test_the_fit_note_gives_its_figures_a_unit_of_time_by_hand():
    fit = {"objective": "sharpe", "rank": 3, "of": 12, "runs": 2, "span": 400.0, "pnl": 100.0, "pnl_se": 20.0,
           "settings": MIXED}
    assert fit_note(fit, 60.0, 1100.0, 1000.0, MIXED) == (  # 100 / 400, 20 / 400 and 60 / 100
        "Fitted by P&L per unit of risk, 3rd of 12: +0.250 a unit in 2 training runs (se 0.050); "
        "+0.600 a unit here over 100 units.")
    assert fit_note(None, 60.0, 1100.0, 1000.0, MIXED) == ""
    one = {**fit, "runs": 1, "pnl_se": None, "pnl": -10.0, "objective": "pnl", "rank": 1}
    assert fit_note(one, -30.0, 11.0, 10.0, OTHER) == (  # a unit of time is enough for a rate; one run has no se
        "Fitted by P&L, 1st of 12: -0.025 a unit in 1 training run; -30.000 a unit here over 1 units. "
        "Those runs had other market settings.")
    assert fit_note(one, -30.0, 10.99, 10.0, MIXED).endswith("; nothing yet here.")
    assert fit_note(one, -30.0, 5.0, 10.0, MIXED).endswith("; nothing yet here.")


@pytest.mark.parametrize("n, word", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (10, "10th"), (11, "11th"),
                                     (12, "12th"), (13, "13th"), (14, "14th"), (20, "20th"), (21, "21st"),
                                     (22, "22nd"), (23, "23rd"), (24, "24th"), (101, "101st"), (111, "111th"),
                                     (211, "211th"), (212, "212th"), (213, "213th"), (1021, "1021st")])
def test_a_rank_is_said_in_ordinals(n, word):
    fit = {"objective": "pnl", "rank": n, "of": 200, "runs": 1, "span": 1.0, "pnl": 0.0, "pnl_se": None,
           "settings": MIXED}
    assert fit_note(fit, 0.0, 0.0, 0.0, MIXED).startswith(f"Fitted by P&L, {word} of 200: ")


def test_a_trader_has_a_signal_from_the_time_its_warm_up_leaves_it():
    waiting = SimpleNamespace(warming=(20, 100), interval=5.0)
    assert trades_from(waiting, 1000.0) == 1400.0  # 80 decisions more, of 5
    assert trades_from(SimpleNamespace(warming=None, interval=5.0), 7.0) == 7.0
    session = StatArbSession()
    runner, _ = session.start_trader({**QUICK["residuals"], "factors": 3, "warm": False}, "residuals")
    assert runner.trades_from == 61 * 5.0  # the PCA window and a decision, of five units each
    runner, _ = session.start_trader({**QUICK["copula"], "warm": False}, "copula")
    assert runner.trades_from == 61 * 5.0


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("kind, change", [("reversal", {}), ("momentum", {}), ("leadlag", {}),
                                          ("flow", {"measure": "queues"}), ("flow", {"measure": "trades"})])
def test_a_trader_that_scores_every_book_has_a_signal_from_the_time_of_its_first_order(kind, change, warm):
    session = StatArbSession()
    session.run_for(100)  # 21 mids at an interval of 5: fewer than the 31 its factors need
    runner, _ = session.start_trader({**QUICK[kind], **change, "warm": warm}, kind)
    runner.fit = {"rank": 1, "of": 3, "objective": "pnl", "runs": 3, "span": 10_000, "pnl": 100.0, "pnl_se": 10.0,
                  "settings": dict(session.settings)}
    began = session.market.time
    while not runner.trader.lots:
        session.run_for(1)
        assert session.market.time < began + 400
    left = 10 if warm else 31  # decisions, of 5 units each
    assert runner.trades_from == began + 5.0 * left
    assert abs(session.market.time - runner.trades_from) < 2 * 5.0 + 1  # the first decision is at once
    note = runner.fit_note(runner.trader.value(session.market.marks()), session.market.time, session.settings)
    assert note.endswith("nothing yet here.")


@pytest.mark.parametrize("data, code, message", [
    ({"fit_id": 1, "rank": 99}, 409, "that fit is no longer here"),
    ({"fit_id": 2, "rank": 1}, 409, "that fit is no longer here"),
    ({"fit_id": 1, "rank": 0}, 409, "that fit is no longer here"),
    ({"fit_id": 1, "rank": 5}, 409, "that fit is no longer here"),
    ({"fit_id": "one", "rank": 1}, 422, "a fit and a rank"),
    ({"fit_id": 1}, 422, "a fit and a rank"),
])
def test_a_candidate_that_is_gone_is_refused_and_the_board_stays_as_it_was(client, statarb, stand_in, data, code,
                                                                          message):
    small_books(client)
    client.post("/api/statarb/traders/fit", data=fit_form())
    reply = client.post("/api/statarb/traders/start", data=data)
    assert reply.status_code == code and message in reply.json()["error"] and not statarb.runners
    assert statarb.market.traders == []
    if code == 409:
        assert reply.json()["error"] == "Not started: that fit is no longer here: fit it again."


@pytest.mark.parametrize("view", ["statarb", "pairs"])
def test_a_fits_candidate_cannot_be_started_before_any_fit(client, view):
    reply = client.post(f"/api/{view}/traders/start", data={"fit_id": 1, "rank": 1})
    assert reply.status_code == 409 and "that fit is no longer here" in reply.json()["error"]


def test_a_fit_id_the_next_fit_has_taken_over_is_stale(client, statarb, stand_in):
    small_books(client)
    client.post("/api/statarb/traders/fit", data=fit_form())
    client.post("/api/statarb/traders/fit", data=fit_form(fit_seed=2))
    reply = client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 1})
    assert reply.status_code == 409 and "no longer here" in reply.json()["error"] and not statarb.runners
    assert client.post("/api/statarb/traders/start", data={"fit_id": 2, "rank": 1}).status_code == 200


def test_a_candidate_that_no_longer_suits_new_settings_is_refused_with_the_reason(client, statarb, stand_in):
    assert client.post("/api/statarb/config", data=form(SECTORS)).status_code == 200  # four books
    data = {**fit_form("residuals", factors=2), "fit_factors": "on", "grid_factors": "2, 3"}
    del data["fit_window"], data["fit_open_at"]  # only the factors are tried
    assert client.post("/api/statarb/traders/fit", data=data).status_code == 200
    assert [c["values"]["factors"] for c in statarb.last_fit.candidates] == [3, 2]  # 3 is first, by_window's factors
    assert client.post("/api/statarb/config", data=form(MIXED)).status_code == 200  # three books: two at most
    seq, version = statarb._seq, statarb.version
    reply = client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 1})
    assert reply.status_code == 422 and reply.json() == {"error": "Not started: fewer factors than books."}
    assert statarb.runners == [] and statarb.market.traders == [] and statarb.version == version
    assert statarb._seq == seq  # and it took no number: the next is the first
    reply = client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 2})
    assert reply.status_code == 200 and reply.json()["name"] == "Residuals 1"
    assert statarb.runners[0].fit["rank"] == 2 and len(statarb.market.traders) == 1


def test_a_trader_that_is_refused_takes_no_number():
    session = session_with(MIXED)
    with pytest.raises(ValueError, match="fewer factors than books"):
        session.start_trader({**QUICK["residuals"], "factors": 3}, "residuals")  # more than three books can have
    with pytest.raises(ValueError, match="0 <= exit < entry"):
        session.start_trader({**QUICK["pairs"], "exit": 2.0}, "pairs")
    assert session._seq == 0 and session.runners == [] and session.market.traders == []
    runner, _ = session.start_trader(QUICK["residuals"], "residuals")
    assert (runner.id, runner.name) == (1, "Residuals 1")
    changes = session.changes
    runner, learned = session.start_trader({**QUICK["pairs"], "warm": False}, "pairs")
    assert runner.name == "Pairs 2" and learned == 0 and session.changes == changes + 1  # the page is told


def test_a_started_candidate_learns_the_marks_kept_one_an_interval_oldest_first(client, statarb, stand_in):
    small_books(client)
    client.post("/api/statarb/traders/fit", data=fit_form())
    client.post("/api/statarb/control", data={"action": "step", "units": 300})
    kept = list(statarb.mids)
    assert len(kept) == 301
    reply = client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 1})
    trader = statarb.runners[0].trader
    assert reply.json()["learned"] == 60 == len(trader._mids)  # of the 61 it could take, the formation's 60
    by_hand = kept[::5][-60:]  # every fifth of the marks, the newest last: 300 is a multiple of five
    assert np.array_equal(np.array(trader._mids), np.array(by_hand))
    assert statarb.runners[0].trades_from == statarb.market.time  # it had all it needed


def test_a_full_board_refuses_a_candidate(client, statarb, stand_in):
    small_books(client)
    client.post("/api/statarb/traders/fit", data=fit_form())
    for _ in range(MAX_TRADERS):
        assert client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 1}).status_code == 200
    reply = client.post("/api/statarb/traders/start", data={"fit_id": 1, "rank": 1})
    assert reply.status_code == 409 and reply.json()["error"] == "Not started: At most 6 traders: remove one first."
    assert len(statarb.runners) == MAX_TRADERS


# ---- The results panel ----

def run(*rows):
    """Training results from (pnl, lots) or just pnl, with the other figures made up from them."""
    out = []
    for row in rows:
        pnl, lots = row if isinstance(row, tuple) else (row, 10.0)
        out.append({**figures(pnl), "lots": lots})
    return out


def settings_of(window, entry):
    return {**QUICK["pairs"], "window": window, "entry": entry}


def fit_of(session, rows, kind="pairs", **more):
    session.last_fit = TrainingFit(
        id=1, kind=kind, grid=more.pop("grid", {"window": [20, 30], "entry": [1.5, 2.0]}), settings=session.settings,
        objective=more.pop("objective", "pnl"), seed=1, span=500, burn=305.0, runs=more.pop("runs", 3),
        candidates=rows, valid=4, size=4, sampled=more.pop("sampled", False), seconds=1.0)
    return session


STATARB_SEEDS = ("Runs of one seed share their factors and shocks, whatever trades, though the order flow parts at "
                 "the first trade, so comparing on the same seeds takes out some of the luck. The standard error, "
                 "measured from the differences, says how much is left, but not for having picked the best of {n}, "
                 "so expect less from it afterwards.")
CLEARLY_NOTE = ("Clearly better means by 2 standard errors of the difference, for the best of all those tried: with "
                "nine settings truly equal and three runs each it still says so in about a quarter of fits, so it is "
                "a guard against a fit with nothing to find, not a test at a stated level. ")
PAIRS_SEEDS = ("Runs of one seed start alike, but the first trade sends each its own way, so that difference carries "
               "both runs&#39; luck; it allows for that, not for having picked the best of {n}, so expect less from "
               "it afterwards.")


@pytest.mark.parametrize("against, se, verdict", [(2.0, 1.0, "unclear"), (2.0001, 1.0, "clearly"),
                                                  (4.0, 2.0, "unclear"), (0.5, 0.0, "clearly"), (-1.0, 0.5, "unclear")])
def test_the_verdict_is_clearly_only_above_the_threshold_in_standard_errors(against, se, verdict):
    best = {"yours": False, "against": against, "against_se": se}
    fit = fit_of(StatArbSession(), [best, {"yours": True}]).last_fit
    assert fit.verdict == verdict and CLEARLY == 2.0


def test_a_tie_goes_to_the_users_settings_whatever_order_the_grid_gives():
    a, b, yours = settings_of(20, 1.5), settings_of(30, 1.5), QUICK["pairs"]
    for tried in ([a, yours, b], [yours, a, b], [a, b, yours]):
        rows = rank_candidates(tried, [run(0.0, 0.0) for _ in tried], yours, "pnl")  # none of them made anything
        assert rows[0]["yours"] and rows[0]["against"] == 0.0
    rows = rank_candidates([a, yours], [run((0.0, 5.0), (0.0, 5.0)), run((0.0, 9.0), (0.0, 9.0))], yours, "pnl")
    assert rows[0]["values"] == a  # the one that traded less still comes first
    fit = fit_of(StatArbSession(), rank_candidates([a, yours], [run(0.0, 0.0), run(0.0, 0.0)], yours, "pnl")).last_fit
    assert fit.verdict == "yours"


def test_a_fit_of_only_the_users_settings_says_that_nothing_was_compared():
    yours = QUICK["pairs"]
    session = fit_of(StatArbSession(), rank_candidates([yours], [run(5.0, 6.0, 7.0)], yours, "pnl"))
    text = render_statarb_fit(session)
    assert "Only your own settings were tried, so nothing was compared." in text
    assert "came out best" not in text and "nothing here says to change them" not in text
    assert "1 setting tried" in text and "1 settings" not in text and text.count("data-rank=") == 1
    both = render_statarb_fit(fit_of(StatArbSession(), rank_candidates(
        [settings_of(20, 1.5), yours], [run(1.0, 1.0), run(2.0, 2.0)], yours, "pnl")))
    assert "Your own settings came out best" in both and "2 settings tried" in both
    assert "nothing was compared" not in both


def test_the_panel_says_plainly_what_the_fit_found():
    a, yours = settings_of(20, 1.5), QUICK["pairs"]
    session = StatArbSession()
    text = render_statarb_fit(fit_of(session, rank_candidates(
        [a, yours], [run(30.0, 110.0, 70.0), run(10.0, 90.0, 50.0)], yours, "pnl")))
    assert 'data-verdict="clearly"' in text and "data-statarb-fit-verdict" in text
    assert "window 20, enter beyond z 1.5 beat your settings by +20.0 (standard error 0.0) on the same seeds" in (
        text.lower()) and "clearly better here" in text
    text = render_statarb_fit(fit_of(session, rank_candidates(
        [a, yours], [run(90.0, 10.0, 51.0), run(10.0, 90.0, 50.0)], yours, "pnl")))
    assert 'data-verdict="unclear"' in text and "not clearly better" in text and "stay with yours" in text
    assert "The best, window 20, enter beyond z 1.5, made +0.3 against your settings" in text
    text = render_statarb_fit(fit_of(session, rank_candidates(
        [a, yours], [run(10.0, 90.0, 50.0), run(90.0, 10.0, 51.0)], yours, "pnl")))
    assert 'data-verdict="yours"' in text and "Your own settings came out best" in text
    text = render_statarb_fit(fit_of(session, rank_candidates([a, yours], [run(20.0), run(10.0)], yours, "pnl"),
                                     runs=1))
    assert 'data-verdict="single"' in text and "in a single run, which cannot tell skill from luck" in text
    assert text.count("data-yours") == 1 and text.count("data-rank=") == 2


def test_each_view_says_what_its_runs_of_one_seed_share_and_the_pairs_one_is_unchanged():
    a, yours = settings_of(20, 1.5), QUICK["pairs"]
    rows = rank_candidates([a, yours], [run(1.0, 2.0), run(2.0, 1.0)], yours, "pnl")
    statarb = render_statarb_fit(fit_of(StatArbSession(), rows))
    assert STATARB_SEEDS.format(n=2) in statarb and "start alike" not in statarb  # the best of the two it tried
    assert CLEARLY_NOTE + STATARB_SEEDS.format(n=2) in statarb
    pairs = PairsSession()
    pairs_rows = rank_candidates([{**ZSCORE, "entry": 1.5}, ZSCORE], [run(1.0, 2.0), run(2.0, 1.0)], ZSCORE, "pnl")
    pairs.last_fit = TrainingFit(id=1, kind="zscore", grid={"entry": [1.5, 2.0]}, settings=pairs.settings,
                                 objective="pnl", seed=1, span=500, burn=21.0, runs=2, candidates=pairs_rows, valid=2,
                                 size=2, sampled=False, seconds=1.0)
    text = render_pairs_fit(pairs)
    assert PAIRS_SEEDS.format(n=2) in text and "share their factors" not in text
    assert CLEARLY_NOTE + PAIRS_SEEDS.format(n=2) + (
        " Each trades alone in its runs; started, it trades in the market on the chart, among whatever else is "
        "there, and its row keeps what it made in training, to compare.") in text
    assert CLEARLY == 2.0


def test_each_views_panel_is_named_for_its_own_buttons_and_nothing_of_the_others():
    a, yours = settings_of(20, 1.5), QUICK["pairs"]
    text = render_statarb_fit(fit_of(StatArbSession(), rank_candidates([a, yours], [run(1.0), run(2.0)], yours, "pnl")))
    assert 'id="statarb-fit-summary"' in text and "data-statarb-fit-start=\"2\"" in text
    assert "data-statarb-fit-use=" in text and "pairs-fit" not in text
    pairs = PairsSession()
    pairs.last_fit = TrainingFit(id=4, kind="zscore", grid={"entry": [1.5, 2.0]}, settings=pairs.settings,
                                 objective="pnl", seed=1, span=500, burn=21.0, runs=1, candidates=rank_candidates(
                                     [{**ZSCORE, "entry": 1.5}, ZSCORE], [run(1.0), run(2.0)], ZSCORE, "pnl"),
                                 valid=2, size=2, sampled=False, seconds=1.0)
    text = render_pairs_fit(pairs)
    assert 'id="pairs-fit-summary"' in text and 'data-fit-id="4"' in text and 'data-kind="zscore"' in text
    assert "data-pairs-fit-start=\"1\"" in text and "data-pairs-fit-use=" in text and "statarb-fit" not in text


def test_a_row_has_the_settings_the_form_can_take_and_the_best_ten_and_the_users_are_shown():
    tried = [settings_of(10 + i, 1.5) for i in range(14)] + [QUICK["pairs"]]
    results = [run(float(100 - i), float(101 - i), float(102 - i)) for i in range(14)] + [run(0.0, 1.0, 2.0)]
    session = fit_of(StatArbSession(), rank_candidates(tried, results, QUICK["pairs"], "pnl"), grid={
        "window": [10], "entry": [1.5]})
    text = render_statarb_fit(session)
    assert text.count("data-rank=") == 11 and 'data-rank="15" data-yours' in text
    assert "The best 10 of the 15 tried, and yours." in text
    use = html.unescape(text.split("data-statarb-fit-use=\"")[1].split("\"")[0])
    assert json.loads(use) == session.last_fit.candidates[0]["values"] == settings_of(10, 1.5)
    assert set(json.loads(use)) == {p.name for p in TRADER_KINDS["pairs"].params}  # a field or a switch of the form
    assert "The best 10 of the 15 tried" in text and text.count("Start</button>") == 11


def test_the_panel_can_be_read_by_risk_adjusted_score_and_says_when_candidates_are_a_sample():
    a, yours = settings_of(20, 1.5), QUICK["pairs"]
    rows = rank_candidates([a, yours], [run(30.0, 110.0, 70.0), run(10.0, 90.0, 50.0)], yours, "sharpe")
    text = render_statarb_fit(fit_of(StatArbSession(), rows, objective="sharpe", sampled=True))
    assert "Score" in text and "+0.020" in text and "P&amp;L per unit of risk" in text
    assert "drawn at random from 4 (seed 1)" in text


def test_the_panel_says_when_the_books_have_changed_since():
    a, yours = settings_of(20, 1.5), QUICK["pairs"]
    session = fit_of(StatArbSession(), rank_candidates([a, yours], [run(1.0), run(2.0)], yours, "pnl"))
    assert "data-statarb-fit-stale" not in render_statarb_fit(session)
    session.configure({**BASE, "shape": "sectors", "half_life": 10.0})
    assert same_market(session.last_fit.settings, session.settings) is False
    assert "data-statarb-fit-stale" in render_statarb_fit(session)
    assert "The books' settings have changed since: these results are for the settings as they were." in (
        html.unescape(render_statarb_fit(session)))
