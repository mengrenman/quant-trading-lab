"""Many books with several common factors, and traders of more than one pair (lob/statarb.py)."""

import copy
import dataclasses
import itertools
import math
import shutil
import subprocess
import sys
import warnings
from functools import cache, partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import lob.crosssection as cross_now
import lob.statarb as statarb_now
from lob.order import Side, limit_order
from lob.pairs import BETA_FLOOR, FLOOR, AnchoredZIModel, CopulaTrader
from lob.sparse_baskets import sparse_baskets
from lob.statarb import (BookMarket, BookTrader, CopulaPortfolio, FactorValues, RelationTrader, ResidualTrader,
                         StateTrader, best_baskets, johansen_relations, mixed_loadings, pairs_by_distance,
                         pairs_by_pvalue, score_holdings, score_portfolio, sector_loadings, trace_rank)

SECTORS = sector_loadings(2, 3)
MIXED = np.array([[1.0, 0.2], [0.8, 0.9], [1.2, -0.5], [0.6, 0.4], [1.1, 1.0], [0.9, -0.2]])


def fair_path(fair, steps):
    """The fair values after each of ``steps`` grid steps."""
    out = []
    for i in range(1, steps + 1):
        fair.advance_to(i * fair.step)
        out.append(list(fair.fairs))
    return np.array(out)


# ---- The fair values ----

@pytest.mark.parametrize("loadings, rank", [(SECTORS, 4), (MIXED, 4), (np.ones((5, 1)), 4), (sector_loadings(3, 4), 9)])
def test_the_books_have_as_many_relations_as_the_factors_leave(loadings, rank):
    f = FactorValues(loadings)
    assert f.rank == rank
    assert f.relations().shape == (rank, len(loadings))
    assert np.allclose(f.relations() @ loadings, 0, atol=1e-9)  # they leave every factor out


def test_a_relation_is_only_the_own_parts():
    f = FactorValues(MIXED, level=3000.0)
    w = f.relations()
    for values in fair_path(f, 200)[-1:]:
        assert w @ (values - f.level) == pytest.approx(w @ f.own, abs=1e-9)


def test_a_pair_within_a_sector_is_a_relation_and_one_across_is_not():
    f = FactorValues(SECTORS)
    within = np.array([1.0, -1.0, 0, 0, 0, 0])
    across = np.array([1.0, 0, 0, -1.0, 0, 0])
    assert np.allclose(within @ SECTORS, 0) and not np.allclose(across @ SECTORS, 0)
    moved = fair_path(f, 5_000)
    assert (moved @ within).std() < 15 < (moved @ across).std()


def test_the_factors_and_own_parts_have_the_variances_they_should():
    f = FactorValues(np.ones((4, 1)), volatility=2.0, own_volatility=0.8, half_life=20.0, step=1.0)
    f.reseed(5)
    factors, own = [], []
    for i in range(1, 40_001):
        f.advance_to(float(i))
        factors.append(f.factors[0])
        own.append(f.own.copy())
    assert np.diff(factors).std() == pytest.approx(2.0, rel=0.02)
    own = np.array(own)
    kappa = math.log(2) / 20.0
    assert own.std() == pytest.approx(0.8 / math.sqrt(2 * kappa), rel=0.05)
    assert np.corrcoef(own[:-1, 0], own[1:, 0])[0, 1] == pytest.approx(math.exp(-kappa), abs=0.01)


def test_without_a_half_life_the_own_parts_wander():
    f = FactorValues(np.ones((4, 1)), volatility=0.0, half_life=None, step=1.0)
    moved = fair_path(f, 10_000)
    assert np.diff(moved, axis=0).std() == pytest.approx(0.8, rel=0.05)
    assert moved.std(axis=0).max() > 15


def test_the_path_depends_on_the_seed_and_the_time_alone():
    a, b = FactorValues(MIXED), FactorValues(MIXED)
    a.reseed(9)
    b.reseed(9)
    a.advance_to(1_000.0)  # all at once, or in pieces as events fall
    for t in np.sort(np.random.default_rng(0).uniform(0, 1_000.0, 777)):
        b.advance_to(float(t))
    b.advance_to(1_000.0)
    assert (a.factors == b.factors).all() and (a.own == b.own).all()
    b.reseed(10)
    b._steps = 0
    b.factors, b.common = np.zeros(2), np.zeros(6)
    b.advance_to(1_000.0)
    assert not (a.factors == b.factors).all()


def test_the_grid_moves_only_when_its_time_has_come():
    f = FactorValues(SECTORS)  # a step of 0.2
    assert f.pushed == 0 and f._steps == 0
    f.advance_to(0.19)
    assert f._steps == 0 and (f.factors == 0).all()
    f.advance_to(1.0)
    assert f._steps == 5 and f.factors.any()
    f.advance_to(1.19)
    assert f._steps == 5


def test_the_factors_do_not_notice_a_trader_and_the_own_parts_only_its_impact():
    # The whole point of their stream: a seed is the same factors and shocks whatever trades.
    def market(traders=()):
        m = BookMarket(FactorValues(SECTORS), seed=4, traders=traders)
        while m.time < 300:
            m.step()
        m.fair.advance_to(300.0)
        return m

    alone = market()
    trader = RelationTrader(weights=np.array([[1.0, -1, 0, 0, 0, 0]]), window=20, entry=0.5)
    busy = market((trader,))
    assert trader.lots > 0
    assert (busy.fair.factors == alone.fair.factors).all()
    assert np.abs(busy.fair.own - alone.fair.own).max() < 15  # against an sd of 4.8 each, with its orders moving them


def test_the_factors_and_the_order_flow_have_streams_of_their_own():
    m = BookMarket(FactorValues(SECTORS), seed=4)
    assert m.fair._rng is not m.rng
    flow = copy.deepcopy(m.rng)  # as yet undrawn: seeded alike, it would give just what the factors first drew
    assert not (flow.standard_normal(m.fair._z.shape) == m.fair._z).any()


def test_a_fair_value_stops_at_the_floor_and_no_relation_moves():
    loadings = np.array([[1.0, 0.5], [1.0, -0.5], [0.5, 0.0], [1.5, 0.2]])
    f = FactorValues(loadings, level=30.0, volatility=2.0)
    free = FactorValues(loadings, level=30.0, volatility=2.0, floor=0.0)
    f.reseed(3)
    free.reseed(3)
    w = f.relations()
    lowest, pushed = math.inf, 0.0
    for i in range(1, 20_001):
        f.advance_to(i * f.step)
        free.advance_to(i * f.step)
        lowest = min(lowest, min(f.fairs))
        assert w @ (np.array(f.fairs) - f.level) == pytest.approx(w @ f.own, abs=1e-6)
        assert np.allclose(f.fairs, f.level + loadings @ f.factors + f.own)  # the push is the first factor's
        if f.pushed > pushed:  # just far enough
            assert min(f.fairs) == pytest.approx(FLOOR)
            pushed = f.pushed
    assert lowest >= FLOOR - 1e-9 and f.pushed > 0
    assert (f.own == free.own).all()  # and it takes no random numbers


def test_drifting_loadings_stay_around_their_start_and_break_the_relations():
    f = FactorValues(MIXED, loading_sd=0.2, loading_half_life=100.0, step=1.0)
    f.reseed(2)
    seen, spread = [], []
    w = f.relations()
    for i in range(1, 8_001):
        f.advance_to(float(i))
        seen.append(f.loadings.copy())
        spread.append(w @ (np.array(f.fairs) - f.level - f.own))
    seen = np.array(seen)
    assert seen.mean(axis=0) == pytest.approx(MIXED, abs=0.08)
    assert (seen - MIXED)[:, :, 1].std() == pytest.approx(0.2, rel=0.2)
    assert np.abs(spread).max() > 5  # the factors are back in what were relations


@pytest.mark.parametrize("loadings, level", [
    ([[2.0], [0.5]], [30.0, 30.0]),  # the lowest book moves more than the factor does
    ([[0.5], [2.0]], [11.0, 400.0]),  # and less
])
def test_a_push_is_just_far_enough_for_the_lowest_book_whatever_its_loading(loadings, level):
    f = FactorValues(np.array(loadings), level=np.array(level), volatility=2.0, own_volatility=0.0)
    f.reseed(3)
    i = 0
    while f.pushed == 0:
        i += 1
        f.advance_to(i * f.step)
    assert f.fairs[0] == pytest.approx(FLOOR) and f.fairs[1] > FLOOR


def test_the_markets_defaults_are_those_the_notebook_says():
    f = FactorValues(SECTORS)
    assert (f.half_life, f.own_volatility, f.impact, f.step, f.loading_half_life) == (50.0, 0.8, 0.05, 0.2, 2000.0)
    assert (f.volatility == 1.0).all() and (f.level == 3000.0).all() and f.floor == FLOOR
    book = BookMarket(f, seed=1).books[0]  # eight levels a side, so that orders stay above a tick at the floor
    assert len(book.depth_profile(Side.BID, 20)) == 8 and book.depth_profile(Side.ASK, 20)[0] == (3001, 10)


def test_drifting_loadings_have_the_half_life_they_are_given():
    f = FactorValues(MIXED, loading_sd=0.2, loading_half_life=20.0, step=0.5)
    f.reseed(4)
    seen = []
    for i in range(1, 32_001):
        f.advance_to(i * f.step)
        seen.append(f.loadings[:, 1].copy())
    away = np.array(seen) - MIXED[:, 1]
    lag = 41  # grid steps: odd, so that a sign that flipped each step would show
    assert (away[:-lag] * away[lag:]).sum() / (away ** 2).sum() == pytest.approx(0.5 ** (lag / 40), abs=0.06)


def test_a_drifting_first_loading_stays_above_its_floor():
    # The floor pushes the first factor, so every book must go up with it.
    f = FactorValues(np.ones((3, 1)), loading_sd=2.0, loading_half_life=20.0, step=1.0)
    f.reseed(2)
    lowest = math.inf
    for i in range(1, 2_001):
        f.advance_to(float(i))
        lowest = min(lowest, f.loadings[:, 0].min())
    assert lowest == BETA_FLOOR


@pytest.mark.parametrize("make, message", [
    (lambda: FactorValues(np.ones(3)), "table"),
    (lambda: FactorValues(np.ones((1, 1))), "table"),
    (lambda: FactorValues(np.ones((2, 0))), "table"),
    (lambda: FactorValues(np.array([[1.0, 1.0], [0.0, 1.0]])), "first factor"),
    (lambda: FactorValues(np.ones((2, 1)), half_life=0.0), "half-life"),
    (lambda: FactorValues(np.ones((2, 1)), own_volatility=-1.0), "volatilities"),
    (lambda: FactorValues(np.ones((2, 1)), volatility=-1.0), "volatilities"),
    (lambda: FactorValues(np.ones((2, 1)), level=5.0), "floor"),
    (lambda: FactorValues(np.ones((2, 1)), floor=-1.0), "floor"),
    (lambda: BookMarket(FactorValues(np.ones((2, 1))), models=[]), "one model"),
])
def test_settings_that_cannot_work_are_refused(make, message):
    with pytest.raises(ValueError, match=message):
        make()


@pytest.mark.parametrize("make", [
    lambda: FactorValues(np.ones((2, 1)), half_life=0.5, loading_half_life=0.5),
    lambda: RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.1, exit=0.0, size=0.5),
    lambda: ResidualTrader(factors=1, window=10, pca_window=10, open_at=0.1, close_long=0.0, close_short=0.0, size=0.5),
    lambda: CopulaPortfolio(pairs=1, trading=1, entry=0.1, exit=0.0, size=0.5),
], ids=["fair values", "relations", "residuals", "copula pairs"])
def test_settings_at_their_limits_are_accepted(make):
    make()


# ---- The market ----

def test_a_market_samples_every_so_often_from_its_start():
    m = BookMarket(FactorValues(SECTORS), seed=7)
    assert m.events == 0
    assert m.run(10.0, every=2.0).time.tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_each_book_gets_its_share_of_the_events():
    m = BookMarket(FactorValues(np.ones((4, 1))), seed=2)
    counts = [0, 0, 0, 0]

    def counted(k, event):
        def wrapped(*args, **kwargs):
            counts[k] += 1
            return event(*args, **kwargs)
        return wrapped

    for k, model in enumerate(m.models):
        model.event = counted(k, model.event)
    for _ in range(40_000):
        m.step()
    assert max(counts) - min(counts) < 0.05 * sum(counts) / 4


def test_a_seeded_market_replays_and_another_seed_does_not():
    one = BookMarket(FactorValues(SECTORS), seed=7).run(200)
    two = BookMarket(FactorValues(SECTORS), seed=7).run(200)
    other = BookMarket(FactorValues(SECTORS), seed=8).run(200)
    assert one.equals(two) and not one.equals(other)


def test_each_book_follows_its_fair_value_and_the_gap_of_the_mids_follows_the_true_one():
    m = BookMarket(FactorValues(np.ones((2, 1))), seed=1)
    d = m.run(4_000)
    assert 30 < m.events / m.time < 45  # about 18 events a book a unit of time, as in lob/pairs.py
    assert (d.mid_0 - d.fair_0).std() < 3.5
    gap, true = (d.mid_1 - d.mid_0).dropna(), (d.fair_1 - d.fair_0)[d.mid_1.notna() & d.mid_0.notna()]
    assert np.corrcoef(gap, true)[0, 1] > 0.85
    assert true.std() == pytest.approx(math.sqrt(2) * 0.8 / math.sqrt(math.log(2) / 25), rel=0.2)  # two own parts


def test_a_market_at_the_floor_keeps_every_book_open():
    m = BookMarket(FactorValues(SECTORS, level=30.0), seed=3)
    next_sample = 0.0
    while m.time < 4_000:
        m.step()
        if m.time >= next_sample:
            next_sample += 1.0
            assert all(m.mid(k) is not None for k in range(6))
    assert m.fair.pushed > 0


def test_an_order_takes_the_book_and_moves_its_fair_value_by_its_impact():
    fair = FactorValues(np.ones((2, 1)), volatility=0.0, own_volatility=0.0, impact=0.1)
    m = BookMarket(fair, seed=1)
    trader = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=4)
    m.add(trader)
    before = list(fair.fairs)
    m._trade(trader, 0, 4)
    assert trader.position.tolist() == [4, 0] and trader.lots == 4
    assert fair.fairs[0] == pytest.approx(before[0] + 0.4) and fair.fairs[1] == before[1]
    assert fair.own.tolist() == pytest.approx([0.4, 0.0])  # what stays, and decays, with its own part
    m._trade(trader, 1, -4)
    assert trader.position.tolist() == [4, -4] and fair.fairs[1] == pytest.approx(before[1] - 0.4)


def test_a_fill_costs_its_distance_from_the_mid_and_the_account_pays_its_price():
    fair = FactorValues(np.ones((2, 1)), volatility=0.0, own_volatility=0.0, impact=0.0)
    m = BookMarket(fair, seed=1, depth=3, lots=10)
    trader = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5)
    m.add(trader)
    assert m.mid(0) == m.mid(1) == 3000
    assert m.books[0].depth_profile(Side.ASK, 9) == [(3001, 10), (3002, 10), (3003, 10)]
    assert m.books[0].depth_profile(Side.BID, 9) == [(2999, 10), (2998, 10), (2997, 10)]
    m._trade(trader, 0, 15)  # ten lots a tick above the mid, and five two above it
    assert trader.position.tolist() == [15, 0] and trader.lots == 15
    assert trader.spread_paid == pytest.approx(10 * 1 + 5 * 2)
    assert trader.cash == pytest.approx(-15 * 3000 - 20)
    m._trade(trader, 1, -15)  # and the same below, on the way down
    assert trader.position.tolist() == [15, -15] and trader.lots == 30
    assert trader.spread_paid == pytest.approx(40)
    assert trader.cash == pytest.approx(-40)


@pytest.mark.parametrize("need, side", [(1, 1), (-1, -1)], ids=["a lot bought", "a lot sold"])
def test_a_single_lot_goes_the_way_it_is_asked(need, side):
    m = BookMarket(FactorValues(np.ones((2, 1)), volatility=0.0, own_volatility=0.0, impact=0.0), seed=1)
    trader = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5)
    m.add(trader)
    m._trade(trader, 0, need)
    assert trader.position.tolist() == [side, 0] and trader.cash == pytest.approx(-side * (3000 + side))


def test_a_book_with_a_side_empty_keeps_its_last_mid_and_nobody_decides_on_it():
    m = BookMarket(FactorValues(np.ones((2, 1)), volatility=0.0, own_volatility=0.0, impact=0.0), seed=1, depth=1)
    trader = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5)
    m.add(trader)
    assert m.mark(0) == 3000
    m._trade(trader, 0, 10)  # the whole of the ask
    assert m.mid(0) is None and m.marks().tolist() == [3000, 3000]
    assert trader.value(m.marks()) == pytest.approx(-10)  # ten lots a tick above where they are marked
    asked = []
    trader.decide = lambda mids: asked.append(mids) or []
    m.trade_for(trader)
    assert asked == [] and trader.next_time == m.time + trader.interval
    paid = trader.spread_paid
    m._trade(trader, 0, -5)  # with no mid, no cost can be told
    assert trader.position[0] == 5 and trader.lots == 15 and trader.spread_paid == paid


def accounts_add_up(market, trader):
    """Its P&L at the mids is what it holds at them and all it paid for it."""
    value = trader.value(market.marks())
    assert value == pytest.approx(trader.cash + float(trader.position @ market.marks()))
    assert trader.spread_paid >= 0 and trader.lots >= np.abs(trader.position).sum()


@pytest.mark.parametrize("make", [
    lambda: RelationTrader(weights=np.array([[1.0, -1, 0, 0, 0, 0]]), window=50),
    lambda: RelationTrader(choose=partial(pairs_by_distance, count=2), formation=300, window=50),
    lambda: RelationTrader(choose=partial(pairs_by_distance, count=2), formation=300, window=50, netted=False),
    lambda: RelationTrader(choose=partial(johansen_relations, count=2, every=5), formation=8_000, window=50),
    lambda: ResidualTrader(factors=2, window=30, pca_window=100),
    lambda: CopulaPortfolio(pairs=2, window=60, trading=20, interval=2.0),
], ids=["fixed", "pairs", "pairs apart", "Johansen", "residuals", "copula pairs"])
def test_every_kind_of_trader_trades_and_its_account_adds_up(make):
    trader = make()
    market = BookMarket(FactorValues(SECTORS), seed=2, traders=(trader,))
    score = score_portfolio(market, trader, trader.formation + 1.0 if hasattr(trader, "formation") else 301.0, 400.0)
    accounts_add_up(market, trader)
    assert trader.lots > 0
    assert score["before_costs"] == pytest.approx(score["pnl"] + score["costs"])


@pytest.mark.parametrize("size, entry", [(5.0, 1.0), (1.0, 2.5)])
def test_a_score_counts_only_what_happens_in_the_span(size, entry):
    trader = RelationTrader(weights=np.array([[1.0, -1, 0, 0, 0, 0]]), window=30, entry=entry, size=size)
    trader.netted_lots = 7  # before the span: not its to count
    score = score_portfolio(BookMarket(FactorValues(SECTORS), seed=3, traders=(trader,)), trader, 100.0, 300.0)
    twin = RelationTrader(weights=np.array([[1.0, -1, 0, 0, 0, 0]]), window=30, entry=entry, size=size)
    m = BookMarket(FactorValues(SECTORS), seed=3, traders=(twin,))
    while m.time < 100.0:
        m.step()
    start, paid, lots, entries = twin.value(m.marks()), twin.spread_paid, twin.lots, twin.entries
    assert lots > 0
    until, next_sample, values = m.time + 300.0, m.time + 1.0, [start]
    while m.time < until:
        m.step()
        while m.time >= next_sample:
            values.append(twin.value(m.marks()))
            next_sample += 1.0
    changes = np.diff(values)
    assert score["deviation"] == pytest.approx(changes.std()) and changes.std() > 0
    assert size == 5.0 or changes.std() < 1  # a small position has a figure as well
    assert score["risk_adjusted"] == pytest.approx(changes.mean() / changes.std())
    assert score["netted"] == 0
    assert score["pnl"] == pytest.approx(twin.value(m.marks()) - start)
    assert score["costs"] == pytest.approx(twin.spread_paid - paid)
    assert score["lots"] == twin.lots - lots and score["entries"] == twin.entries - entries


def test_a_span_too_short_to_sample_has_no_risk_to_speak_of():
    trader = RelationTrader(weights=np.array([[1.0, -1, 0, 0, 0, 0]]), window=30, entry=1.0)
    score = score_portfolio(BookMarket(FactorValues(SECTORS), seed=3, traders=(trader,)), trader, 10.0, 0.5)
    assert (score["gross"], score["exposure"], score["deviation"], score["risk_adjusted"]) == (0.0, 0.0, 0.0, 0.0)


def test_a_score_says_what_it_held_and_how_much_of_the_factors():
    # A relation holds none of the factors; one book alone holds them all.
    hedged = RelationTrader(weights=np.array([[1.0, -1, 0, 0, 0, 0]]), window=30, entry=1.0)
    alone = RelationTrader(weights=np.array([[1.0, 0, 0, 0, 0, 0]]), window=30, entry=1.0)
    scores = [score_portfolio(BookMarket(FactorValues(SECTORS), seed=3, traders=(t,)), t, 100.0, 300.0)
              for t in (hedged, alone)]
    assert scores[0]["gross"] > 0 and scores[0]["exposure"] == 0
    assert scores[1]["exposure"] > 0


# ---- Relations and how they are chosen ----

def history_of(loadings, units=20_000, seed=1, every=1):
    f = FactorValues(loadings, step=1.0)
    f.reseed(seed)
    rng = np.random.default_rng(seed + 100)
    rows = []
    for i in range(1, units + 1):
        f.advance_to(float(i))
        if i % every == 0:
            rows.append(np.array(f.fairs) + 1.7 * rng.standard_normal(f.n))
    return np.array(rows)


class Mids(list):
    """A market's worth of mids to hand a trader, one at a time."""

    def __init__(self, trader, n):
        super().__init__()
        self.trader = trader
        trader.attach(n)

    def see(self, *mids):
        moves = self.trader.decide(np.array(mids, dtype=float))
        for move in moves:
            self.trader.position = self.trader.position + move
        self.append(moves)
        return moves


def test_the_rule_shorts_above_entry_longs_below_and_goes_flat_at_exit():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=5, entry=1.5, exit=0.0, size=4)
    feed = Mids(t, 2)
    for _ in range(4):
        assert feed.see(100, 100) == []  # no window yet, and nothing wanted
    assert feed.see(100, 100) == [] and t.z[0] == 0
    move = feed.see(110, 100)[0]  # the gap jumps above: z = 2
    assert t.z[0] == pytest.approx(2.0) and t.states[0] == -1 and move.tolist() == [-4, 4]
    assert feed.see(109, 100) == []  # still above its mean: it holds
    assert t.states[0] == -1
    move = feed.see(100, 100)[0]  # back past the mean
    assert t.states[0] == 0 and move.tolist() == [4, -4] and t.position.tolist() == [0, 0]
    assert t.entries == 1


@pytest.mark.parametrize("held, gap, z, stays", [
    (-1, 0.7, 0.6, True), (-1, 0.3, 0.27, False),  # a short goes flat below the exit
    (1, -0.7, -0.6, True), (1, -0.3, -0.27, False),  # a long above minus the exit
])
def test_a_relation_goes_flat_inside_its_exit_and_holds_outside_it(held, gap, z, stays):
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=5, entry=1.5, exit=0.5, size=1)
    feed = Mids(t, 2)
    t.states = np.array([held])
    t.position = np.array([held, -held])
    for g in (-1.0, 1.0, -1.0, 1.0):
        assert feed.see(100 + g, 100) == []
    feed.see(100 + gap, 100)
    assert t.z[0] == pytest.approx(z, abs=0.01)
    assert t.states[0] == (held if stays else 0)


def test_entries_count_each_relation_that_opens_or_turns_round():
    t = RelationTrader(weights=np.array([[1.0, -1.0], [1.0, 1.0]]), window=3, entry=0.5, size=2)
    t.attach(2)
    t.states = np.array([0, 0])
    t._become(np.array([1, 0]))
    assert t.entries == 1
    t._become(np.array([1, -1]))  # the first holds: not counted again
    assert t.entries == 2
    t._become(np.array([-1, 0]))  # the first turns round, the second closes
    assert t.entries == 3
    t._become(np.array([0, 0]))
    assert t.entries == 3


def test_not_netted_a_reversal_is_one_order_the_whole_way():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=3, netted=False)
    t.attach(2)
    t.states = np.array([-1])
    t.position = np.array([-3, 3])
    moves = t._become(np.array([1]))
    assert len(moves) == 1 and moves[0].tolist() == [6, -6]


def test_it_goes_straight_from_short_to_long():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=5, entry=1.5, size=2)
    feed = Mids(t, 2)
    for _ in range(4):
        feed.see(100, 100)
    feed.see(120, 100)
    assert t.states[0] == -1
    for _ in range(2):
        feed.see(100, 100)
    assert t.states[0] == 0
    move = feed.see(60, 100)[0]
    assert t.states[0] == 1 and move.tolist() == [2, -2]


def test_lots_are_the_sum_of_the_relations_in_whole_numbers_and_their_largest_leg_is_the_size():
    t = RelationTrader(weights=np.array([[2.0, -1.0, 0.0], [0.0, 0.5, -0.25]]), window=3, entry=0.5, size=5)
    assert t.weights.tolist() == [[1.0, -0.5, 0.0], [0.0, 1.0, -0.5]]
    feed = Mids(t, 3)
    feed.see(100, 100, 100)
    feed.see(100, 100, 100)
    move = feed.see(110, 100, 120)[0]  # the first is dear, the second cheap: -1 and +1
    assert t.states.tolist() == [-1, 1]
    assert move.tolist() == [-5, 8, -2]  # -5, 2.5 + 5, -2.5: rounded after adding them up


def test_netting_sends_one_order_a_book_and_not_netting_one_for_each_relation():
    # Two relations with a leg in common and opposite wishes: netted, they save crossing that book twice.
    weights = np.array([[1.0, -1.0, 0.0], [0.0, -1.0, 1.0]])

    def moves(netted):
        t = RelationTrader(weights=weights, window=3, entry=0.5, size=5, netted=netted)
        feed = Mids(t, 3)
        feed.see(100, 100, 100)
        feed.see(100, 100, 100)
        return t, feed.see(110, 100, 90)

    together, sent = moves(True)
    assert len(sent) == 1 and sent[0].tolist() == [-5, 0, 5]  # book 1 nets to nothing
    assert together.netted_lots == 10  # five lots each way that were never crossed
    apart, sent = moves(False)
    assert len(sent) == 2
    assert sum(np.abs(m).sum() for m in sent) == 20  # and it pays for all of it
    assert (apart.position == together.position).all()


def test_an_order_that_did_not_fill_is_tried_again_when_not_netted():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=5, netted=False)
    t.attach(2)
    t.states = np.array([-1])
    t.position = np.array([-3, 5])  # short of what it wanted, by two lots
    moves = t._become(np.array([-1]))
    assert len(moves) == 1 and moves[0].tolist() == [-2, 0]


def test_a_relation_is_chosen_once_the_formation_is_full():
    calls = []

    def choose(history):
        calls.append(history.copy())
        return np.array([[1.0, -1.0]])

    t = RelationTrader(choose=choose, formation=4, window=3, entry=0.5, size=1)
    feed = Mids(t, 2)
    for i in range(3):
        assert feed.see(100 + i, 100) == [] and t.weights is None
    feed.see(103, 100)
    assert len(calls) == 1 and calls[0].shape == (4, 2) and calls[0][-1].tolist() == [103, 100]
    # The window is already full, from the formation period, and the mids it is deciding on count once: gaps of 1, 2, 3.
    assert t.z == pytest.approx([1 / np.std([1, 2, 3])])


def test_choosing_again_keeps_a_relation_on_the_same_books_and_closes_the_rest():
    choices = iter([np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]), np.array([[1.0, -0.9, 0.0], [1.0, 0.0, -1.0]])])
    t = RelationTrader(choose=lambda history: next(choices), formation=3, window=3, entry=0.5, size=4, refit=2)
    feed = Mids(t, 3)
    for _ in range(2):
        feed.see(100, 100, 100)
    feed.see(110, 100, 120)  # chosen here, with the window full
    assert t.fits == 1 and t.states.tolist() == [-1, 1]
    feed.see(110, 100, 120)
    feed.see(110, 100, 120)
    assert t.fits == 2
    assert t.states[0] != 0  # the first pair is still on, re-estimated
    assert t.weights.tolist()[0] == [1.0, -0.9, 0.0]
    assert t.states[1] == 0  # the new one starts flat, its gap at its mean


def test_choosing_can_be_done_elsewhere_while_the_trader_goes_on():
    from concurrent.futures import Future

    asked = []

    def defer(choose, history):
        future = Future()
        asked.append((future, choose, history.shape))
        return future

    t = RelationTrader(choose=lambda history: np.array([[1.0, -1.0]]), formation=3, window=3, entry=0.5, size=4,
                       defer=defer)
    feed = Mids(t, 2)
    for _ in range(3):
        assert feed.see(100, 100) == []
    assert len(asked) == 1 and asked[0][2] == (3, 2) and t.weights is None and t.warming == (3, 3)
    assert feed.see(100, 100) == [] and len(asked) == 1  # asked once, and it goes on meanwhile
    asked[0][0].set_result(asked[0][1](None))
    move = feed.see(110, 100)[0]  # the answer is taken up, with the mids seen since in its window
    assert t.fits == 1 and t.weights.tolist() == [[1.0, -1.0]] and t.states[0] == -1 and move.tolist() == [-4, 4]
    t.refit, t._next_fit = 1, t._seen + 1  # and it can be asked again
    feed.see(110, 100)
    assert len(asked) == 2


def test_a_chooser_that_finds_nothing_leaves_it_flat():
    t = RelationTrader(choose=lambda history: np.zeros((0, 2)), formation=3, window=3)
    feed = Mids(t, 2)
    for _ in range(5):
        assert feed.see(100, 100) == []
    assert t.weights is None and t.fits == 1


def test_a_chooser_that_finds_nothing_later_closes_what_it_held():
    choices = iter([np.array([[1.0, -1.0]]), np.zeros((0, 2))])
    t = RelationTrader(choose=lambda history: next(choices), formation=3, window=3, entry=0.5, size=4, refit=2)
    feed = Mids(t, 2)
    feed.see(100, 100)
    feed.see(100, 100)
    feed.see(110, 100)
    assert t.position.tolist() == [-4, 4]
    feed.see(110, 100)
    move = feed.see(110, 100)[0]
    assert t.fits == 2 and t.weights is None and t.states is None and t.z is None
    assert move.tolist() == [4, -4] and t.position.tolist() == [0, 0]


def test_johansen_is_asked_as_it_says_and_gives_at_most_as_many_as_it_is_asked(monkeypatch):
    seen = []

    def johansen(data, det_order, k_ar_diff):
        seen.append((len(data), det_order, k_ar_diff))
        return SimpleNamespace(lr1=np.array([40.0, 20.0, 1.0]),
                               evec=np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 3.0], [0.5, 0.5, 3.0]]),
                               cvt=np.array([[25.0, 27.0, 30.0], [12.0, 15.0, 20.0], [1.0, 2.0, 3.0]]))

    monkeypatch.setattr("statsmodels.tsa.vector_ar.vecm.coint_johansen", johansen)
    rows = johansen_relations(np.zeros((50, 3)), count=1)
    assert seen == [(10, 0, 2)]  # one row in five, a constant, and two lags
    assert rows.tolist() == [[0.5, -1.0, 0.25]]  # the first vector, with its largest leg 1
    assert len(johansen_relations(np.zeros((50, 3)), count=5)) == 2  # as many as the test finds


def test_baskets_pass_at_95_percent_and_the_furthest_above_it_come_first(monkeypatch):
    results = {  # the statistic, and its value at 90%, 95% and 99%
        (0, 1): (26.0, [25.0, 27.0, 30.0]),  # past 90% only
        (0, 2): (28.0, [25.0, 27.0, 30.0]),  # 1 above
        (0, 3): (34.0, [19.0, 21.0, 25.0]),  # 13 above, though not the largest
        (1, 2): (35.0, [25.0, 27.0, 30.0]),  # 8 above
        (1, 3): (30.0, [25.0, 27.0, 30.0]),  # 3 above
        (2, 3): (10.0, [25.0, 27.0, 30.0]),
    }
    seen = []

    def johansen(data, det_order, k_ar_diff):
        books = (int(data[0, 0]), int(data[0, 1]))
        seen.append((len(data), det_order, k_ar_diff))
        statistic, values = results[books]
        return SimpleNamespace(lr1=np.array([statistic, 99.0]), cvt=np.array([values, [1.0, 1.0, 1.0]]),
                               evec=np.array([[1.0, 7.0], [-0.5, 7.0]]))

    monkeypatch.setattr("statsmodels.tsa.vector_ar.vecm.coint_johansen", johansen)
    rows = best_baskets(np.tile(np.arange(4.0), (50, 1)), count=6, legs=2)
    assert seen[0] == (10, 0, 2)
    # Those sharing no book first, the strongest first among them; then the rest.
    assert [tuple(np.nonzero(r)[0]) for r in rows] == [(0, 3), (1, 2), (1, 3), (0, 2)]
    assert rows[0, [0, 3]].tolist() == [1.0, -0.5]  # its first vector


def test_johansen_finds_the_true_rank_and_vectors_the_factors_leave_out():
    history = history_of(MIXED, 20_000, every=5)
    j = coint_johansen(history, det_order=0, k_ar_diff=2)
    assert trace_rank(j) == 4
    rows = johansen_relations(history, count=4, every=1)
    assert rows.shape == (4, 6) and (np.abs(rows).max(axis=1) == 1).all()
    assert np.abs(rows @ MIXED).max() < 0.1  # next to loadings of about 1


def test_the_trace_test_goes_up_from_none_and_stops_at_its_first_failure():
    values = np.array([[25.0, 27.0, 30.0], [12.0, 15.0, 20.0], [1.0, 2.0, 3.0]])  # at 90%, 95% and 99%
    johansen = SimpleNamespace(lr1=np.array([28.0, 10.0, 9.0]), cvt=values)
    assert trace_rank(johansen) == 1  # the third would pass, had it got that far
    near = SimpleNamespace(lr1=np.array([26.0, 10.0, 9.0]), cvt=values)
    assert trace_rank(near) == 0 and trace_rank(near, level=0) == 1  # 95% unless told otherwise
    assert trace_rank(SimpleNamespace(lr1=np.array([31.0, 21.0, 4.0]), cvt=values), level=2) == 3


def test_johansen_finds_nothing_where_nothing_is_there():
    wander = history_of(np.eye(4) + 0.01, 8_000, every=5)  # as many factors as books
    assert johansen_relations(wander, count=2, every=1).shape == (0, 4)


def test_a_small_basket_has_as_many_legs_as_asked_and_leaves_the_factors_out():
    history = history_of(MIXED, 20_000, every=5)
    rows = best_baskets(history, count=2, legs=3, every=1)
    assert rows.shape == (2, 6) and (np.count_nonzero(rows, axis=1) == 3).all()
    assert not set(np.nonzero(rows[0])[0]) & set(np.nonzero(rows[1])[0])  # sharing no book while it can
    assert np.abs(rows @ MIXED).max() < 0.15


@pytest.mark.parametrize("choose", [pairs_by_distance, partial(pairs_by_pvalue, every=5)],
                         ids=["distance", "p-value"])
def test_the_pairs_chosen_are_those_within_a_sector_hedged_about_one_for_one(choose):
    history = history_of(SECTORS, 15_000)
    rows = choose(history, count=3)
    assert len(rows) == 3
    for row in rows:
        books = np.nonzero(row)[0]
        assert len(books) == 2 and {int(b) // 3 for b in books} in ({0}, {1})
        assert sorted(row[books].tolist()) == pytest.approx([-1.0, 0.97], abs=0.06)


def test_a_pairs_hedge_is_the_lots_of_the_second_book_for_one_of_the_first():
    rng = np.random.default_rng(1)
    second = 100 + np.cumsum(rng.standard_normal(2_000))
    first = 2 * second + rng.standard_normal(2_000)
    row = pairs_by_distance(np.column_stack([first, second]), count=1)[0]
    assert row == pytest.approx([0.5, -1.0], abs=0.01)  # with its largest leg 1


def test_pairs_by_distance_come_nearest_first():
    history = history_of(SECTORS, 500)
    rebased = history / history[0]
    nearest = sorted(itertools.combinations(range(6), 2),
                     key=lambda pair: ((rebased[:, pair[0]] - rebased[:, pair[1]]) ** 2).sum())
    assert [tuple(np.nonzero(r)[0]) for r in pairs_by_distance(history, count=15)] == nearest


def test_pairs_by_pvalue_pass_at_five_percent_and_the_lowest_come_first(monkeypatch):
    history = np.random.default_rng(1).normal(100.0, 1.0, (50, 3))
    p_values = {(0, 1): 0.04, (0, 2): 0.01, (1, 2): 0.07}
    seen = []

    def coint(x, y, maxlag, autolag):
        seen.append((len(x), maxlag, autolag))
        books = tuple(int(np.flatnonzero((history[::5].T == side).all(axis=1))[0]) for side in (x, y))
        return 0.0, p_values[books], None

    monkeypatch.setattr("statsmodels.tsa.stattools.coint", coint)
    rows = pairs_by_pvalue(history)
    assert seen[0] == (10, 20, "AIC")  # one row in five, and the lags chosen up to 20
    assert [tuple(np.nonzero(r)[0]) for r in rows] == [(0, 2), (0, 1)]


def test_distance_is_on_prices_rebased_to_one():
    rng = np.random.default_rng(2)
    one = 1 + 0.001 * np.cumsum(rng.standard_normal(500))
    other = 1 + 0.001 * np.cumsum(rng.standard_normal(500))
    row = pairs_by_distance(np.column_stack([100 * one, 300 * one, 100 * other]), count=1)[0]
    assert row[2] == 0 and row[0] != 0 and row[1] != 0  # the two that move alike, however far apart their prices


def test_pairs_may_share_a_book_unless_told_otherwise():
    history = history_of(SECTORS, 15_000)
    shared = pairs_by_distance(history, count=3)
    apart = pairs_by_distance(history, count=2, overlap=False)
    books = [set(np.nonzero(r)[0]) for r in apart]
    assert not books[0] & books[1] and len(shared) == 3


def test_no_pair_below_five_percent_means_no_pair():
    wander = 3000 + np.cumsum(np.random.default_rng(1).standard_normal((5_000, 3)), axis=0)  # three random walks
    assert len(pairs_by_pvalue(wander, count=3, every=5)) == 0


# ---- Residuals, after Avellaneda and Lee (2010) ----

def by_hand(moves, eigen, center=True):
    """Their appendix, book by book, without any of the array arithmetic."""
    t, n = moves.shape
    factors = moves @ eigen
    design = np.column_stack([np.ones(t), factors])
    ms, sds, xs, betas, bs = [], [], [], [], []
    for i in range(n):
        coefficients = np.linalg.lstsq(design, moves[:, i], rcond=None)[0]
        residuals = moves[:, i] - design @ coefficients
        x = np.cumsum(residuals)
        lagged = np.column_stack([np.ones(t - 1), x[:-1]])
        a, b = np.linalg.lstsq(lagged, x[1:], rcond=None)[0]
        shocks = x[1:] - lagged @ [a, b]
        ms.append(a / (1 - b))
        sds.append(math.sqrt(np.mean(shocks ** 2) / (1 - b ** 2)))
        xs.append(x[-1])
        betas.append(coefficients[1:])
        bs.append(b)
    ms = np.array(ms)
    if center:
        ms = ms - ms.mean()
    return (np.array(xs) - ms) / np.array(sds), np.array(bs), np.array(betas)


@pytest.mark.parametrize("center", [True, False])
def test_the_s_scores_are_those_of_the_papers_appendix(center):
    moves = np.diff(history_of(SECTORS, 700, every=5), axis=0)
    t = ResidualTrader(factors=2, window=100, pca_window=120, center=center)
    t.attach(6)
    t._pca(moves)
    s, good, betas = t.s_scores(moves[-100:])
    want_s, want_b, want_betas = by_hand(moves[-100:], t._eigen, center)
    assert s == pytest.approx(want_s, rel=1e-6)
    assert betas.T == pytest.approx(want_betas, rel=1e-6)
    assert good.tolist() == ((want_b > 0) & (want_b < math.exp(-2 / 100))).tolist()


def test_s_scores_do_not_depend_on_the_scale_of_the_moves():
    moves = np.diff(history_of(SECTORS, 700, every=5), axis=0)

    def scored(scale):
        t = ResidualTrader(factors=2, window=100, pca_window=120)
        t.attach(6)
        t._pca(moves * scale)
        return t.s_scores(moves[-100:] * scale)

    s, good, _ = scored(1.0)
    small_s, small_good, _ = scored(0.01)
    assert small_s == pytest.approx(s, rel=1e-6) and small_good.tolist() == good.tolist()


def test_a_book_that_has_not_moved_adds_nothing_to_the_eigenportfolios():
    moves = np.diff(history_of(SECTORS, 700, every=5), axis=0)
    moves[:, 0] = 0.0
    t = ResidualTrader(factors=2, window=100, pca_window=120)
    t.attach(6)
    t._pca(moves)
    assert np.isfinite(t._eigen).all() and (t._eigen[0] == 0).all()


def test_a_fit_is_accepted_only_if_it_reverts_within_half_the_window():
    assert ResidualTrader(window=60).max_b == pytest.approx(0.9672, abs=1e-4)  # the paper's number for its 60 days
    rng = np.random.default_rng(5)
    steps = 100
    reverting = np.zeros(steps + 1)
    for i in range(steps):
        reverting[i + 1] = 0.5 * reverting[i] + rng.standard_normal()
    moves = np.column_stack([
        rng.standard_normal(steps),  # the factor, in a book of its own
        np.diff(reverting),
        np.where(np.arange(steps) < steps // 2, 1.0, -1.0) + 0.01 * rng.standard_normal(steps),  # up, then down: slow
        2.0 * (-1.0) ** np.arange(steps) + 0.01 * rng.standard_normal(steps),  # back and forth: b below zero
    ])
    t = ResidualTrader(factors=1, window=steps, pca_window=steps)
    t.attach(4)
    t._eigen = np.array([[1.0], [0.0], [0.0], [0.0]])
    _, good, _ = t.s_scores(moves)
    _, b, _ = by_hand(moves, t._eigen)
    assert 0 < b[1] < t.max_b and b[2] >= t.max_b and b[3] <= 0  # what each book is meant to show
    assert good[1:].tolist() == [True, False, False]


def test_eigenportfolios_are_eigenvectors_over_volatilities_and_their_moves_uncorrelated():
    moves = np.diff(history_of(SECTORS, 3_000, every=5), axis=0)
    t = ResidualTrader(factors=3, window=100, pca_window=500)
    t.attach(6)
    t._pca(moves)
    sd = moves.std(axis=0)
    values, vectors = np.linalg.eigh(np.corrcoef(moves.T))
    top = vectors[:, np.argsort(values)[::-1][:3]]
    assert np.abs(t._eigen * sd[:, None]) == pytest.approx(np.abs(top), abs=1e-9)
    together = np.corrcoef((moves @ t._eigen).T)
    assert together - np.eye(3) == pytest.approx(np.zeros((3, 3)), abs=1e-9)


class Scripted(ResidualTrader):
    """A residual trader told its s-scores, so that the rule alone is tested."""

    def _pca(self, moves):
        pass  # the test sets the eigenportfolios

    def s_scores(self, moves):
        return self.script.pop(0), self.fine.copy(), np.array(self.betas)


@pytest.mark.parametrize("s, start, end", [
    (-1.3, 0, 1), (-1.2, 0, 0), (1.3, 0, -1), (1.2, 0, 0),  # open beyond 1.25
    (-0.6, 1, 1), (-0.4, 1, 0),  # a long closes above -0.5
    (0.8, -1, -1), (0.7, -1, 0),  # a short below 0.75
    (1.3, 1, -1), (-1.3, -1, 1),  # and it can turn round at once
])
def test_the_rule_opens_and_closes_at_the_papers_levels(s, start, end):
    t = Scripted(factors=1, window=10, pca_window=10, pca_every=10**6, size=1)
    t.attach(2)
    t._eigen = np.array([[1.0], [1.0]])
    t.fine = np.array([True, True])
    t.betas = [[0.0, 0.0]]
    t.states = np.array([start, 0])
    t.weights = np.eye(2)
    t.script = [np.array([s, 0.0])]
    t.position = np.array([start, 0])
    for _ in range(11):
        t._mids.append(np.array([100.0, 100.0]))
        t._recent.append(np.array([100.0, 100.0]))
    t.decide(np.array([100.0, 100.0]))
    assert t.states[0] == end


@pytest.mark.parametrize("beyond", [-5.0, 5.0])
def test_a_fit_that_reverts_too_slowly_opens_nothing_and_closes_what_is_open(beyond):
    t = Scripted(factors=1, window=10, pca_window=10, pca_every=10**6, size=1)
    t.attach(2)
    t._eigen = np.array([[1.0], [1.0]])
    t.fine = np.array([False, False])
    t.betas = [[0.0, 0.0]]
    t.states = np.array([1, 0])
    t.weights = np.eye(2)
    t.script = [np.array([beyond, beyond])]
    t.position = np.array([1, 0])
    for _ in range(11):
        t._mids.append(np.array([100.0, 100.0]))
        t._recent.append(np.array([100.0, 100.0]))
    assert t.decide(np.array([100.0, 100.0]))[0].tolist() == [-1, 0]
    assert t.states.tolist() == [0, 0]


def test_the_eigenportfolios_are_found_again_only_so_often():
    class Counting(Scripted):
        times = 0

        def _pca(self, moves):
            self.times += 1
            assert len(moves) == 12  # the moves of its PCA window

        def s_scores(self, moves):
            assert len(moves) == 10  # and of its fit
            return super().s_scores(moves)

    t = Counting(factors=1, window=10, pca_window=12, pca_every=3, size=1)
    t.attach(2)
    t._eigen = np.array([[1.0], [1.0]])
    t.fine = np.array([True, True])
    t.betas = [[0.0, 0.0]]
    t.script = [np.zeros(2) for _ in range(7)]
    for _ in range(12):
        t._mids.append(np.array([100.0, 100.0]))
        t._recent.append(np.array([100.0, 100.0]))
    for _ in range(7):
        t.decide(np.array([100.0, 100.0]))
    assert t.times == 3  # at its first signal, then every third decision


def test_a_position_holds_its_hedge_as_it_was_on_entry_and_the_lots_net():
    t = Scripted(factors=1, window=10, pca_window=10, pca_every=10**6, size=10)
    t.attach(2)
    t._eigen = np.array([[0.5], [0.5]])
    t.fine = np.array([True, True])
    t.betas = [[1.0, 1.0]]  # a factor unit is half a lot of each: so the hedge is half a lot of each
    t.script = [np.array([-2.0, 0.0]), np.array([-1.0, 0.0]), np.array([-1.0, -2.0])]
    for _ in range(11):
        t._mids.append(np.array([100.0, 100.0]))
        t._recent.append(np.array([100.0, 100.0]))
    move = t.decide(np.array([100.0, 100.0]))[0]
    assert move.tolist() == [5, -5] and t.weights[0].tolist() == [0.5, -0.5]
    t.position = t.position + move
    t.betas = [[3.0, 3.0]]  # new loadings: it holds on as it was
    assert t.decide(np.array([100.0, 100.0])) == [] and t.states[0] == 1
    t.position = t.position.copy()
    t.betas = [[1.0, 1.0]]
    moves = t.decide(np.array([100.0, 100.0]))  # the other book opens: long it, short half a lot of each
    assert t.states.tolist() == [1, 1] and moves[0].tolist() == [-5, 5]  # together they net to nothing held


@pytest.mark.parametrize("make, message", [
    (lambda: RelationTrader(weights=np.array([[1.0, -1.0]]), window=2), "window of 3"),
    (lambda: RelationTrader(weights=np.array([[1.0, -1.0]]), entry=1.0, exit=1.0), "exit < entry"),
    (lambda: RelationTrader(weights=np.array([[1.0, -1.0]]), size=0.0), "size above 0"),
    (lambda: RelationTrader(), "either relations"),
    (lambda: RelationTrader(choose=pairs_by_distance, formation=10, window=20), "formation period"),
    (lambda: ResidualTrader(factors=0), "factor or more"),
    (lambda: ResidualTrader(window=5), "window of 10"),
    (lambda: ResidualTrader(window=100, pca_window=50), "PCA window"),
    (lambda: ResidualTrader(open_at=0.5, close_long=0.5), "closing < opening"),
    (lambda: ResidualTrader(open_at=0.5, close_short=0.75), "closing < opening"),
    (lambda: CopulaPortfolio(pairs=0), "a pair or more"),
    (lambda: CopulaPortfolio(entry=1.0, exit=1.0), "exit < entry"),
    (lambda: CopulaPortfolio(trading=0), "trading period"),
])
def test_settings_a_trader_cannot_work_with_are_refused_at_once(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_too_many_factors_are_refused():
    with pytest.raises(ValueError, match="fewer factors"):
        ResidualTrader(factors=3).attach(3)


def test_its_book_leaves_the_factors_out_better_than_its_gross_alone_would():
    t = ResidualTrader(factors=2, window=60, pca_window=200, interval=2.0)
    market = BookMarket(FactorValues(SECTORS), seed=5, traders=(t,))
    score = score_portfolio(market, t, 450.0, 1_500.0)
    assert score["gross"] > 0
    assert score["exposure"] < 0.5 * score["gross"]


# ---- Many copula pairs ----

def test_a_copula_portfolio_has_no_signal_before_it_has_pairs():
    history = history_of(SECTORS, 201, every=1)
    p = CopulaPortfolio(pairs=2, window=200, trading=50, entry=0.5, interval=1.0)
    p.attach(6)
    assert p.signal() is None
    for mids in history:
        p.decide(mids)
    assert len(p.signal()) == 2 and np.isfinite(p.signal()).all()


def test_copula_pairs_chosen_elsewhere_learn_the_mids_that_came_meanwhile():
    from concurrent.futures import Future

    history = history_of(SECTORS, 600, every=1)
    asked = []

    def defer(build, mids):
        future = Future()
        asked.append((future, build, mids))
        return future

    p = CopulaPortfolio(pairs=2, window=200, trading=50, entry=0.5, interval=1.0, size=5, defer=defer)
    p.attach(6)
    for mids in history[:211]:  # asked at the 201st, going on for ten more
        assert p.decide(mids) == []
    assert len(asked) == 1 and asked[0][2].tolist() == history[:201].tolist() and not p.traders
    asked[0][0].set_result(asked[0][1](asked[0][2]))
    assert p.decide(history[211]) == []  # taken up here: these mids are among those it has learned
    (a, b), hedge = p._books[0], p.traders[0].hedge
    alone = CopulaTrader(window=200, trading=50, entry=0.5, exit=0.0, size=1, hedge=hedge, interval=1.0)
    alone.learn(list(zip(history[:201, a].tolist(), history[:201, b].tolist(), strict=True)))  # as it was built
    alone.learn(list(zip(history[201:212, a].tolist(), history[201:212, b].tolist(), strict=True)))  # and what came
    states, wants = [], []
    for mids in history[212:]:
        p.decide(mids)
        states.append(p.states[0])
        wants.append(alone.decide(float(mids[a]), float(mids[b])))
    assert states == wants and any(states)


def test_each_copula_pair_decides_as_the_pair_trader_would_alone():
    history = history_of(SECTORS, 600, every=1)
    p = CopulaPortfolio(pairs=2, window=200, trading=50, entry=0.5, interval=1.0, size=5)
    p.attach(6)
    for mids in history[:201]:
        assert p.decide(mids) == []
    p.decide(history[201])
    assert len(p.traders) == 2
    # A trader on the same pair, from the same mids.
    (a, b), hedge = p._books[0], p.traders[0].hedge
    alone = CopulaTrader(window=200, trading=50, entry=0.5, exit=0.0, size=1, hedge=hedge, interval=1.0)
    alone.learn(list(zip(history[:201, a].tolist(), history[:201, b].tolist(), strict=True)))
    wants = [alone.decide(float(history[201, a]), float(history[201, b]))]
    states = [p.states[0]]
    for mids in history[202:]:
        p.decide(mids)
        states.append(p.states[0])
        wants.append(alone.decide(float(mids[a]), float(mids[b])))
    assert states == wants and any(states)


def test_copula_lots_are_the_size_of_the_pair_in_the_right_direction():
    history = history_of(SECTORS, 300, every=1)
    p = CopulaPortfolio(pairs=1, window=200, trading=50, interval=1.0, size=5)
    p.attach(6)
    for mids in history[:201]:
        p.decide(mids)
    p.decide(history[201])
    a, b = p._books[0]
    assert p.weights[0, b] == 1.0 and -1.0 <= p.weights[0, a] < 0
    assert p.traders[0].hedge == pytest.approx(-p.weights[0, a])
    p.states = np.array([0])
    p.position = np.zeros(6, dtype=int)
    move = p._become(np.array([-1]))[0]  # short the gap: sells b, buys the hedge of a
    assert move[b] == -5 and move[a] == round(5 * p.traders[0].hedge) and np.count_nonzero(move) == 2


def test_a_copula_pairs_state_is_that_of_its_larger_leg_whichever_sign_the_chooser_gave_it():
    history = history_of(SECTORS, 300, every=1)
    p = CopulaPortfolio(pairs=1, choose=lambda mids, count: np.array([[0.5, -1.0, 0, 0, 0, 0]]), window=200, trading=50,
                        interval=1.0, size=4)
    p.attach(6)
    for mids in history[:201]:
        p.decide(mids)
    assert p._books == [(0, 1)] and p.traders[0].hedge == 0.5
    assert p.weights[0, :2].tolist() == [-0.5, 1.0]
    move = p._become(np.array([-1]))[0]  # short the gap: sells the larger leg, buys its hedge
    assert move.tolist() == [2, -4, 0, 0, 0, 0]


def test_copula_pairs_can_be_built_elsewhere_while_the_trader_goes_on_and_learn_the_mids_that_came():
    from concurrent.futures import Future

    history = history_of(SECTORS, 400, every=1)
    asked = []

    def defer(build, mids):
        future = Future()
        asked.append((future, build, mids))
        return future

    p = CopulaPortfolio(pairs=2, window=200, trading=50, entry=0.5, interval=1.0, size=5, defer=defer)
    p.attach(6)
    for mids in history[:200]:
        assert p.decide(mids) == [] and not asked
    assert p.decide(history[200]) == [] and len(asked) == 1  # asked once there are window + 1 mids
    assert asked[0][1] == p._build and asked[0][2].shape == (201, 6) and np.array_equal(asked[0][2], history[:201])
    assert p.traders == [] and p.signal() is None and p.warming == (201, 201)
    for mids in history[201:204]:  # it decides nothing meanwhile, and does not ask again
        assert p.decide(mids) == [] and len(asked) == 1 and p.traders == []
    asked[0][0].set_result(asked[0][1](asked[0][2]))
    assert p.decide(history[204]) == []  # the answer is taken up here: it decides from the next mids
    assert len(p.traders) == 2 and p.states.tolist() == [0, 0] and p.warming is None and len(asked) == 1
    for trader, (a, b) in zip(p.traders, p._books, strict=True):  # each has seen the mids up to these
        assert trader._last == (history[204, a], history[204, b])
    p.decide(history[205])
    assert p.traders[0]._last == (history[205, p._books[0][0]], history[205, p._books[0][1]])


def test_copula_pairs_built_at_once_are_as_they_were_without_a_place_to_build_them_elsewhere():
    history = history_of(SECTORS, 300, every=1)
    p = CopulaPortfolio(pairs=2, window=200, trading=50, entry=0.5, interval=1.0, size=5)
    assert p.defer is None
    p.attach(6)
    for mids in history[:201]:
        p.decide(mids)
    assert len(p.traders) == 2 and p.traders[0]._last == (history[200, p._books[0][0]], history[200, p._books[0][1]])


# ---- Learning from the past ----

@pytest.mark.parametrize("make, needed", [
    (lambda: RelationTrader(choose=partial(pairs_by_distance, count=1), formation=200, window=50), 200),
    (lambda: ResidualTrader(factors=2, window=50, pca_window=199), 200),
    (lambda: CopulaPortfolio(pairs=1, window=199, trading=20, interval=1.0), 200),
], ids=["relations", "residuals", "copula pairs"])
def test_a_trader_learns_from_the_past_instead_of_watching_for_it(make, needed):
    history = history_of(SECTORS, 400)
    watcher = make()
    watcher.attach(6)
    assert watcher.warming == (0, needed)
    for row in history[:needed - 1]:
        watcher.decide(row)
    assert watcher.warming == (needed - 1, needed)
    taught = make()
    taught.attach(6)
    assert taught.learn(list(history[:needed - 1])) == needed - 1
    assert taught.warming == (needed - 1, needed)
    # From here on the two decide alike: one saw those mids as they came, the other all at once.
    for row in history[needed - 1:]:
        assert [m.tolist() for m in taught.decide(row)] == [m.tolist() for m in watcher.decide(row)]
    assert taught.warming is None and watcher.warming is None


def test_a_residual_trader_keeps_the_latest_of_the_past_for_each_estimate():
    t = ResidualTrader(factors=1, window=20, pca_window=30)
    t.attach(2)
    t.learn([np.array([i, i]) for i in range(5)])
    assert [m[0] for m in t._recent] == [0, 1, 2, 3, 4]
    t = ResidualTrader(factors=1, window=20, pca_window=30)
    t.attach(2)
    assert t.learn([np.array([i, i]) for i in range(100)]) == 31
    assert [m[0] for m in t._recent] == list(range(79, 100)) and [m[0] for m in t._mids] == list(range(69, 100))


def test_it_takes_only_as_much_of_the_past_as_it_can_use():
    trader = RelationTrader(choose=partial(pairs_by_distance, count=1), formation=50, window=20)
    trader.attach(6)
    assert trader.learn(list(history_of(SECTORS, 80))) == 50 and trader.warming == (50, 50)
    trader.decide(history_of(SECTORS, 81)[-1])
    assert trader.fits == 1


def test_the_published_rule_is_what_a_residual_trader_does_unless_told_otherwise():
    t = ResidualTrader()
    assert (t.open_at, t.close_long, t.close_short, t.center) == (1.25, 0.5, 0.75, True)
    assert (t.factors, t.window, t.pca_window, t.pca_every, t.record) == (2, 120, 500, 20, False)


def test_a_trader_says_its_levels():
    assert RelationTrader(weights=np.array([[1.0, -1.0]]), entry=1.5, exit=0.5).levels == (1.5, 0.5)
    assert ResidualTrader(open_at=2.0, close_long=0.5, close_short=0.75).levels == (2.0, 0.5)
    assert CopulaPortfolio(entry=1.0, exit=0.25).levels == (1.0, 0.25)


# ---- The market designs ----

def test_mixed_loadings_center_on_one_and_spread_the_rest():
    b = mixed_loadings(2_000, 3, seed=4, spread=0.5)
    assert b[:, 0].mean() == pytest.approx(1.0, abs=0.03) and b[:, 0].std() == pytest.approx(0.2, rel=0.1)
    assert 0.3 <= b[:, 0].min() < 0.5
    assert b[:, 1:].mean() == pytest.approx(0.0, abs=0.03) and b[:, 1:].std() == pytest.approx(0.5, rel=0.05)


def test_sectors_and_mixed_loadings_have_the_shapes_they_say():
    b = sector_loadings(3, 4)
    assert b.shape == (12, 4) and (b[:, 0] == 1).all() and (b[:, 1:].sum(axis=1) == 1).all()
    m = mixed_loadings(12, 3, seed=2)
    assert m.shape == (12, 3) and (m[:, 0] > 0).all() and np.linalg.matrix_rank(m) == 3
    for x in range(12):
        for y in range(x + 1, 12):
            assert np.linalg.matrix_rank(m[[x, y]], tol=1e-6) == 2  # no two parallel
    assert (mixed_loadings(12, 3, seed=2) == m).all()


# ---- Cost-aware trading ----
# A rate below 1 takes a trader's aim only part of the way to its target at a decision, in lots that need not be whole
# (partial adjustment, Garleanu and Pedersen 2013), and a band above 0 keeps it from trading within that many lots of
# the aim, except in a book whose target is 0 (Davis and Norman 1990). Each rule is held to numbers worked out here by
# hand and to a plain-Python reading of the spec, and the defaults to the code as it was before the two settings
# existed (BASELINE, read out of git).

BASELINE = "86bea8e"  # the commit before cost-aware trading: what the defaults must still do, number for number
ROOT = Path(__file__).resolve().parents[1]


def toward_by_hand(aim, position, target, band):
    """The lots to trade in each book, from the spec as a plain loop: the gap from the position to the aim, all of it
    where the target is 0 and what lies beyond the band elsewhere, rounded (half to even, as Python's round does)."""
    moves = []
    for goal, have, want in zip(aim, position, target, strict=True):
        gap = float(goal) - int(have)
        beyond = gap if want == 0 else math.copysign(max(abs(gap) - band, 0), gap)
        moves.append(round(beyond))
    return moves


def aim_by_hand(aim, target, rate):
    """The aim after a decision, from the spec as a plain loop: ``rate`` of the way from where it was to the target."""
    return [float(goal) + rate * (int(want) - float(goal)) for goal, want in zip(aim, target, strict=True)]


def book_trader(position, rate=1.0, band=0.0, aim=None):
    """A trader holding ``position``, and aiming there unless told otherwise: one that has been there a while."""
    trader = BookTrader(rate=rate, band=band)
    trader.attach(len(position))
    trader.position = np.array(position)
    trader._aim = np.array(position if aim is None else aim, dtype=float)
    return trader


# Rate, band, the gaps to the target from books held and aimed at around 20 lots, and the lots to trade: worked out by
# hand. The aim goes rate of the gap, and what is traded is the gap to it less the band, rounded (halves go to even).
TOWARD = [
    (1.0, 0, [-2, 6, 0, -10, 1], [-2, 6, 0, -10, 1]),  # the defaults: the whole gap
    (1.0, 2, [-2, -1, 0, 1, 2], [0, 0, 0, 0, 0]),  # inside the band, its edge too: nothing
    (1.0, 2, [3, 4, 7, -3, -4, -7], [1, 2, 5, -1, -2, -5]),  # beyond it: to its edge, and not to the target
    (0.5, 0, [1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 2, 2, 3, 4]),  # half is .5 1 1.5 2 2.5 3 3.5: halves go to even
    (0.5, 0, [-1, -2, -3, -4, -5, -6, -7], [0, -1, -2, -2, -2, -3, -4]),  # and the same the other way
    (0.75, 0, [1, 2, 3, 4, 5, 6], [1, 2, 2, 3, 4, 4]),  # .75 1.5 2.25 3 3.75 4.5
    (0.7, 0, [2, 3, 4], [1, 2, 3]),  # 1.4 2.1 2.8: the nearest lot, neither down nor up
    (0.1, 0, [1, 2, 3, 4, 5, -1, -4, 14, 17, 20], [0, 0, 0, 0, 0, 0, 0, 1, 2, 2]),  # a tenth: no lot for under half
    (0.5, 2, [2, 3, 4, 5, 6, 7, 8, -3, -6, -8, 0], [0, 0, 0, 0, 1, 2, 2, 0, -1, -2, 0]),  # both: 1 1.5 2 2.5 3 3.5 4
    (1.0, 1.5, [1, 2, 3, 4], [0, 0, 2, 2]),  # a band that is not whole: 0 .5 1.5 2.5 beyond it
]


@pytest.mark.parametrize("rate, band, gaps, moves", TOWARD)
def test_the_move_toward_a_target_is_what_the_spec_works_out_by_hand(rate, band, gaps, moves):
    position = 20 + 3 * np.arange(len(gaps))  # not all alike, and not 0
    target = position + gaps
    trader = book_trader(position, rate, band)
    move = trader._toward(target)
    assert move.tolist() == moves and move.dtype.kind == "i"
    aim = aim_by_hand(position, target, rate)  # the plain loop says the same
    assert trader._aim.tolist() == aim and toward_by_hand(aim, position, target, band) == moves
    assert trader.position.tolist() == (20 + 3 * np.arange(len(gaps))).tolist()
    assert target.tolist() == (position + gaps).tolist()


def test_with_the_defaults_the_aim_is_the_target_and_the_move_is_the_gap_to_it():
    rng = np.random.default_rng(5)
    for _ in range(20):
        for trader in (BookTrader(), BookTrader(rate=1, band=0), BookTrader(rate=1.0)):
            trader.attach(8)
            for _ in range(5):  # wherever it held and aimed before
                trader.position, target = rng.integers(-12, 13, 8), rng.integers(-12, 13, 8)
                move = trader._toward(target)
                assert move.tolist() == (target - trader.position).tolist() and move.dtype.kind == "i"
                assert trader._aim.tolist() == target.tolist()


def test_with_a_rate_of_1_the_aim_is_the_target_whatever_the_band():
    for band in (0, 1, 2.5):
        trader = BookTrader(band=band)
        trader.attach(3)
        for target in ([4, -2, 0], [-3, 5, 1], [0, 0, 0]):
            trader._toward(np.array(target))
            assert trader._aim.tolist() == target


@pytest.mark.parametrize("rate", [1.0, 0.75, 0.5, 0.2, 0.1])
def test_the_aim_after_k_decisions_with_the_same_target_is_the_target_times_one_less_what_is_left(rate):
    trader = BookTrader(rate=rate)
    trader.attach(3)
    target = np.array([10, -4, 0])
    assert trader._aim.tolist() == [0, 0, 0] and trader._aim.dtype.kind == "f"
    for k in range(1, 21):
        trader.position = trader.position + trader._toward(target)  # what it trades has no bearing on the aim
        assert trader._aim == pytest.approx(target * (1 - (1 - rate) ** k), abs=1e-12)


def test_the_aim_is_worked_out_by_hand_for_a_target_that_stays_and_one_that_turns():
    trader = BookTrader(rate=0.5)
    trader.attach(1)
    aims = []
    for target in (8, 8, 8, 8, -8, -8, 0, 0):
        trader.position = trader.position + trader._toward(np.array([target]))
        aims.append(float(trader._aim[0]))
    # half of the way to 8 from 0: 4, 6, 7, 7.5; then half of the way from 7.5 to -8: -0.25; and so on, from the aim
    assert aims == [4.0, 6.0, 7.0, 7.5, -0.25, -4.125, -2.0625, -1.03125]
    trader = BookTrader(rate=0.2)
    trader.attach(1)
    aims = []
    for _ in range(3):
        trader._toward(np.array([10]))
        aims.append(float(trader._aim[0]))
    assert aims == pytest.approx([2.0, 3.6, 4.88])  # 10 * (1 - .8^k)


def test_a_large_gap_is_closed_at_a_decision_by_the_rate_and_the_nearest_lot():
    for rate in (0.75, 0.5, 0.2, 0.1):
        trader = BookTrader(rate=rate)
        trader.attach(2)
        target = np.array([1000, -1000])
        for _ in range(6):
            gap = target - trader.position
            move = trader._toward(target)
            assert (np.abs(move / gap - rate) <= 1 / np.abs(gap)).all()  # the share of the gap is the rate, to a lot
            trader.position = trader.position + move
        assert trader.position[0] == pytest.approx(1000 * (1 - (1 - rate) ** 6), abs=1)


@pytest.mark.parametrize("lots", [3, 500])
@pytest.mark.parametrize("rate", [0.75, 0.5, 0.2, 0.1])
def test_the_share_of_the_gap_closed_per_decision_is_the_rate_on_average_however_few_lots_there_are(rate, lots):
    rng = np.random.default_rng(int(rate * 100))
    trader = BookTrader(rate=rate)
    trader.attach(4)
    closed, gaps = 0.0, 0.0
    for _ in range(2000):  # a new target each time, of 3 lots either way or of 500
        target = rng.integers(-lots, lots + 1, 4)
        gap = target - trader.position
        move = trader._toward(target)
        closed += float((move * np.sign(gap)).sum())
        gaps += float(np.abs(gap).sum())
        trader.position = trader.position + move
    assert closed / gaps == pytest.approx(rate, abs=0.01 if lots > 100 else 0.04)  # to the rounding of a few lots


@pytest.mark.parametrize("rate, target, held, decisions", [
    # A lot to go, by hand: the aim is 1 - .8^k from 0, .2 .36 .488 .5904, and the first above a half lot is the fourth.
    (0.2, 1, 0, 4),
    (0.2, -1, 0, 4),  # the same the other way
    (0.2, 0, 1, 4),  # and closing a lot: the aim .8^k is .8 .64 .512 .4096, and over a half off at the fourth
    (0.5, 1, 0, 2),  # .5 is a half, which goes to even, 0; .75 is a lot
    (0.1, 1, 0, 7),  # 1 - .9^k: .1 .19 .271 .344 .41 .469 .522
    (0.75, 1, 0, 1),
    (0.2, 3, 2, 4),  # from 2 lots to 3: the same lot to go
])
def test_a_lot_to_go_is_closed_by_rounding_not_left_for_ever_and_takes_as_many_decisions_as_worked_out_by_hand(
        rate, target, held, decisions):
    trader = book_trader([held], rate)
    taken = 0
    while trader.position[0] != target and taken < 40:
        trader.position = trader.position + trader._toward(np.array([target]))
        taken += 1
    assert taken == decisions and trader.position[0] == target
    assert not trader._toward(np.array([target])).any()  # and it stays


def test_two_lots_to_go_at_a_rate_of_a_fifth_take_two_decisions_for_the_first_lot_and_seven_for_the_second():
    trader = book_trader([0], 0.2)
    held = []
    for _ in range(8):
        trader.position = trader.position + trader._toward(np.array([2]))
        held.append(int(trader.position[0]))
    # the aim is 2 * (1 - .8^k): .4 .72 .976 1.18 1.34 1.47 1.58 1.66; the position is that rounded
    assert held == [0, 1, 1, 1, 1, 1, 2, 2]


def test_a_book_whose_target_is_0_is_closed_to_the_aim_whatever_the_band_and_one_that_is_not_keeps_it():
    # Held and aimed at: 2, 2, -3 and 4 lots. The targets are 0, 3, 0 and 8, and the band is 3.
    trader = book_trader([2, 2, -3, 4], band=3)
    move = trader._toward(np.array([0, 3, 0, 8]))
    # books 0 and 2 are closed, though they are inside the band or at its edge; book 1 is 1 lot off, inside it; book 3
    # is 4 off, 1 beyond it
    assert move.tolist() == [-2, 0, 3, 1]
    assert trader.position.tolist() == [2, 2, -3, 4]
    trader = book_trader([6, 6, -6], rate=0.5, band=2)  # and with a rate: the aim is half way to the target
    move = trader._toward(np.array([0, 2, 0]))
    assert trader._aim.tolist() == [3.0, 4.0, -3.0]
    assert move.tolist() == [-3, 0, 3]  # the closed books all the way to the aim; the other 2 from the aim, inside it


def test_a_closed_relation_is_closed_all_the_way_to_its_aim_whatever_the_band():
    t = RelationTrader(weights=np.array([[0.5, -1.0]]), window=3, entry=0.5, size=20, band=2)
    t.attach(2)
    move = t._become(np.array([1]))[0]  # target [10, -20]: 8 and 18 to the band's edge
    assert move.tolist() == [8, -18]
    t.position = t.position + move
    assert t._become(np.array([1])) == []  # 2 from it in each book: inside
    move = t._become(np.array([0]))[0]  # closed: both books go to 0, and not to 2 of it
    assert move.tolist() == [-8, 18]
    t.position = t.position + move
    assert t.position.tolist() == [0, 0] and t._become(np.array([0])) == []


def test_asked_again_and_again_it_goes_to_the_target_and_not_into_the_band_and_then_trades_nothing_at_all():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=10, rate=0.5, band=2)
    t.attach(2)
    for _ in range(40):
        for move in t._become(np.array([-1])):
            t.position = t.position + move
    assert t.position.tolist() == [-8, 8] and t._become(np.array([-1])) == []  # the edge of the band, and it stays
    assert t._aim.tolist() == pytest.approx([-10, 10], abs=1e-9)


@pytest.mark.parametrize("rate", [1.0, 0.9, 0.75, 0.5, 0.3, 0.2, 0.1, 0.01])
@pytest.mark.parametrize("band", [0, 1, 2, 3, 0.5, 1.5, 2.25])
def test_the_move_is_the_specs_for_every_gap_and_never_goes_past_the_band_or_the_target(rate, band):
    gaps = np.arange(-20, 21)
    position = (gaps * 7) % 9 - 4
    target = position + gaps  # some of them 0
    trader = book_trader(position, rate, band)
    aim = aim_by_hand(position, target, rate)
    move = trader._toward(target)
    assert move.tolist() == toward_by_hand(aim, position, target, band)
    to_aim = np.array(aim) - position  # the gap to the aim
    closed = target == 0
    inside = ~closed & (np.abs(to_aim) <= band)
    assert (move[inside] == 0).all()  # no trade inside the band, its edge too, unless the book is closed
    assert (np.abs(to_aim - move)[closed] <= 0.5).all()  # a book it no longer wants goes to its aim, to a half lot
    traded = ~closed & (move != 0)
    assert (np.abs(np.abs(to_aim - move)[traded] - band) <= 0.5).all()  # to the band's edge, to a half lot
    assert (move * gaps >= 0).all() and (np.abs(move) <= np.abs(gaps)).all()  # the right way, and not past the target
    if rate == 1.0:  # the aim is the target
        assert (move[closed] == gaps[closed]).all()
        if float(band).is_integer():
            assert (np.abs(gaps - move)[~closed & (np.abs(gaps) > band)] == band).all()  # the whole way is the edge


@pytest.mark.parametrize("rate, band", [(1.0, 0), (1.0, 2), (0.5, 0), (0.5, 2), (0.2, 1), (0.1, 3), (0.8, 1.5),
                                        (0.3, 0.5)])
def test_asked_again_and_again_it_ends_within_the_band_from_any_start_and_then_trades_nothing(rate, band):
    rng = np.random.default_rng(int(rate * 100 + band * 10))
    for _ in range(25):
        start, target = rng.integers(-15, 16, 6), rng.integers(-15, 16, 6)
        trader = book_trader(start, rate, band)  # held at the start, and aiming there
        for _ in range(500):
            before = target - trader.position
            trader.position = trader.position + trader._toward(target)
            after = target - trader.position
            assert (np.abs(after) <= np.abs(before)).all()  # never farther
            assert (after * before >= 0).all()  # and never over
        assert trader._aim == pytest.approx(target, abs=1e-9)  # the aim has got there
        gap, kept = np.abs(target - trader.position), target != 0
        assert (gap[~kept] == 0).all()  # a book it does not want is closed, from wherever it started
        assert (gap[kept] <= band + 0.5).all()  # and one it wants is within the band, to a half lot
        if float(band).is_integer():  # from outside it stops at the edge, from inside it does not move
            assert (gap[kept] <= band).all()
            assert (gap[kept] == np.minimum(np.abs(target - start), band)[kept]).all()
        assert not trader._toward(target).any()


def test_a_trader_has_a_rate_of_1_and_a_band_of_0_unless_told_otherwise_and_they_come_last():
    trader = BookTrader()
    assert trader.rate == 1.0 and trader.band == 0.0
    assert [f.name for f in dataclasses.fields(BookTrader)][-2:] == ["rate", "band"]
    assert [f.name for f in dataclasses.fields(BookTrader)][:2] == ["size", "interval"]
    for make in (lambda: RelationTrader(weights=np.array([[1.0, -1.0]])), ResidualTrader, CopulaPortfolio):
        trader = make()
        assert trader.rate == 1.0 and trader.band == 0.0 and not trader._cost_aware
    trader = RelationTrader(weights=np.array([[1.0, -1.0]]), rate=0.5, band=2)
    assert (trader.rate, trader.band) == (0.5, 2)


# The settings, and what is built with them: kwargs for the trader.
STATE_TRADERS = {
    "fixed relations": lambda **kw: RelationTrader(weights=np.array([[1.0, -1.0]]), **kw),
    "chosen pairs": lambda **kw: RelationTrader(choose=partial(pairs_by_distance, count=1), formation=50, window=20,
                                                **kw),
    "residuals": lambda **kw: ResidualTrader(**kw),
    "copula pairs": lambda **kw: CopulaPortfolio(**kw),
}


@pytest.mark.parametrize("make", STATE_TRADERS.values(), ids=STATE_TRADERS.keys())
@pytest.mark.parametrize("settings", [
    {"rate": 0}, {"rate": 0.0}, {"rate": -0.1}, {"rate": 1.01}, {"rate": 2}, {"rate": float("nan")},
    {"band": -1}, {"band": -0.001}, {"band": float("nan")},
    {"rate": 0, "band": 1}, {"rate": 1.5, "band": 2}, {"rate": 0.5, "band": -1},
])
def test_a_rate_outside_0_to_1_or_a_band_below_0_is_refused_and_says_what_it_needs(make, settings):
    with pytest.raises(ValueError, match=r"rate.*band"):
        make(**settings)


@pytest.mark.parametrize("make", STATE_TRADERS.values(), ids=STATE_TRADERS.keys())
@pytest.mark.parametrize("settings", [
    {}, {"rate": 1}, {"rate": 1.0, "band": 0}, {"rate": 0.5}, {"rate": 0.999}, {"rate": 1e-9}, {"band": 0.5},
    {"band": 1000}, {"rate": 0.25, "band": 3},
])
def test_a_rate_up_to_1_and_a_band_from_0_are_accepted(make, settings):
    trader = make(**settings)
    assert trader.rate == settings.get("rate", 1.0) and trader.band == settings.get("band", 0.0)


@pytest.mark.parametrize("make", [STATE_TRADERS[k] for k in ("fixed relations", "residuals", "copula pairs")],
                         ids=["fixed relations", "residuals", "copula pairs"])
@pytest.mark.parametrize("settings", [{"rate": 0.5}, {"rate": 0.99}, {"band": 1}, {"band": 0.5},
                                      {"rate": 0.5, "band": 1}])
def test_not_netted_it_refuses_a_rate_below_1_or_a_band_above_0(make, settings):
    with pytest.raises(ValueError, match="cost-aware trading needs netting"):
        make(netted=False, **settings)
    assert make(netted=True, **settings)._cost_aware  # netted, the same settings are accepted
    assert not make(netted=False)._cost_aware and not make(netted=False, rate=1.0, band=0.0)._cost_aware


def test_a_trader_that_chooses_its_relations_is_refused_the_same_way_not_netted():
    with pytest.raises(ValueError, match="needs netting"):
        STATE_TRADERS["chosen pairs"](netted=False, band=1)
    assert STATE_TRADERS["chosen pairs"](netted=False).netted is False


# ---- Cost-aware trading: the netted path ----

def test_netted_at_a_rate_below_1_it_goes_part_of_the_way_and_finishes_at_later_decisions_with_the_same_states():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=8, rate=0.5)
    t.attach(2)
    sent, held = [], []
    for _ in range(6):
        moves = t._become(np.array([-1]))  # the same state each time: short the gap, 8 lots a leg
        sent.append([m.tolist() for m in moves])
        for move in moves:
            t.position = t.position + move
        held.append(t.position.tolist())
    # The aim is 4, 6, 7, 7.5, 7.75 and 7.875 lots. Half a lot to go is rounded to none, and three quarters to a lot.
    assert sent == [[[-4, 4]], [[-2, 2]], [[-1, 1]], [], [[-1, 1]], []]
    assert held == [[-4, 4], [-6, 6], [-7, 7], [-7, 7], [-8, 8], [-8, 8]]
    assert t.entries == 1 and t.states.tolist() == [-1]  # one entry however many orders it took


def test_a_state_that_turns_around_part_of_the_way_goes_toward_its_new_target_from_where_it_is():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=8, rate=0.5)
    t.attach(2)
    t.position = t.position + t._become(np.array([-1]))[0]  # [-4, 4] of [-8, 8]
    move = t._become(np.array([1]))[0]  # now the target is [8, -8]: the aim goes half of the way to it from [-4, 4]
    assert move.tolist() == [6, -6] and t.entries == 2
    t.position = t.position + move  # [2, -2]
    sent = []
    for _ in range(3):  # and flat: the aim is 1, then a half, then a quarter lot
        moves = t._become(np.array([0]))
        sent.append([m.tolist() for m in moves])
        for m in moves:
            t.position = t.position + m
    assert sent == [[[-1, 1]], [], [[-1, 1]]] and t.position.tolist() == [0, 0] and t.entries == 2


def test_a_band_stops_it_at_the_edge_and_not_at_the_target():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=8, band=2)
    t.attach(2)
    moves = t._become(np.array([-1]))  # target [-8, 8]: 6 lots to the band's edge
    assert [m.tolist() for m in moves] == [[-6, 6]]
    t.position = t.position + moves[0]
    for _ in range(3):
        assert t._become(np.array([-1])) == []  # 2 from the target is inside the band: nothing
    t.position = np.array([-9, 9])  # past the target by one: still inside it
    assert t._become(np.array([-1])) == []
    t.position = np.array([-10, 10])  # past it by two: at the edge
    assert t._become(np.array([-1])) == []
    t.position = np.array([-11, 11])  # and past the edge, so it comes back to it
    assert [m.tolist() for m in t._become(np.array([-1]))] == [[1, -1]]
    t.position = np.array([-6, 6])
    move = t._become(np.array([1]))[0]  # it turns around: from 14 away to the edge, 12 lots
    assert move.tolist() == [12, -12]
    t.position = t.position + move
    assert t._become(np.array([0]))[0].tolist() == [-6, 6]  # and flat: all the way, and not to the edge
    assert t.entries == 2


def test_a_rate_and_a_band_together_take_the_rate_of_what_lies_beyond_the_band():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=10, rate=0.5, band=2)
    t.attach(2)
    sent = []
    for _ in range(7):
        moves = t._become(np.array([-1]))  # target [-10, 10]
        sent.append([m.tolist() for m in moves])
        for m in moves:
            t.position = t.position + m
    # The aim is 5, 7.5, 8.75, 9.375, 9.6875, 9.84 and 9.92 lots, and the band is 2 lots either side of it. So 3 beyond
    # the band, then 2.5 (a half goes to even, 2), then 1.75 (2), then a third of a lot (none), then 0.69 (1), then
    # inside it.
    assert sent == [[[-3, 3]], [[-2, 2]], [[-2, 2]], [], [[-1, 1]], [], []]
    assert t.position.tolist() == [-8, 8]


def test_a_band_is_around_the_lots_of_each_book_so_one_leg_of_a_relation_can_be_inside_it_while_another_trades():
    t = RelationTrader(weights=np.array([[0.5, -1.0]]), window=3, entry=0.5, size=6, band=2)
    t.attach(2)
    t.position = np.array([1, -1])  # the relation asks for [3, -6]: 2 lots off in the first book, 5 in the second
    move = t._become(np.array([1]))[0]
    assert move.tolist() == [0, -3]  # the first is at the edge of its band and does not trade; the second goes to it
    t.position = t.position + move
    assert t.position.tolist() == [1, -4] and t._become(np.array([1])) == []  # both inside it now, and held there


def test_entries_count_the_states_entered_and_not_the_trades_made():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=8, band=100)
    t.attach(2)
    sent = [t._become(np.array([state])) for state in (-1, 0, 1)]
    assert sent == [[], [], []] and t.entries == 2  # a band so wide that it never trades: and two states entered


def test_through_its_decisions_a_trader_with_a_rate_finishes_what_it_started_though_its_states_do_not_change():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=5, entry=1.5, size=8, rate=0.5)
    feed = Mids(t, 2)
    for _ in range(5):
        assert feed.see(100, 100) == []
    sent = []
    for _ in range(8):
        sent.append([m.tolist() for m in feed.see(110, 100)])  # the gap jumps and stays: z falls, the state holds
        assert t.states.tolist() == [-1]
    assert sent == [[[-4, 4]], [[-2, 2]], [[-1, 1]], [], [[-1, 1]], [], [], []]
    assert t.position.tolist() == [-8, 8] and t.entries == 1


def test_through_its_decisions_a_trader_with_a_band_trades_to_its_edge_and_no_more():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=5, entry=1.5, size=8, band=2)
    feed = Mids(t, 2)
    for _ in range(5):
        feed.see(100, 100)
    assert [m.tolist() for m in feed.see(110, 100)] == [[-6, 6]]
    for _ in range(4):
        assert feed.see(110, 100) == [] and t.states.tolist() == [-1]
    assert t.position.tolist() == [-6, 6]


def test_with_the_defaults_a_trader_goes_the_whole_way_at_once():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=5, entry=1.5, size=8)
    feed = Mids(t, 2)
    for _ in range(5):
        feed.see(100, 100)
    assert [m.tolist() for m in feed.see(110, 100)] == [[-8, 8]]
    assert feed.see(110, 100) == [] and t.position.tolist() == [-8, 8]


# What netting saved is counted against what the states ask of the books, and not against the gap from where the
# trader is, which at a rate or a band holds lots it has not traded yet.
@pytest.mark.parametrize("rate, band, move", [
    (1.0, 0, [-5, 0, 5]), (0.5, 0, [-2, 0, 2]), (0.2, 0, [-1, 0, 1]), (1.0, 2, [-3, 0, 3]), (0.8, 2, [-2, 0, 2]),
    (0.5, 2, [0, 0, 0]),  # an aim of 2.5 lots is half a lot beyond the band, and rounds to none
])
def test_what_netting_saved_is_counted_against_what_the_states_ask_not_against_the_part_traded(rate, band, move):
    # Two relations with a leg in common and opposite wishes: book 1 would be crossed twice, five lots each way.
    weights = np.array([[1.0, -1.0, 0.0], [0.0, -1.0, 1.0]])
    t = RelationTrader(weights=weights, window=3, entry=0.5, size=5, rate=rate, band=band)
    t.attach(3)
    t.states = np.array([0, 0])
    moves = t._become(np.array([-1, 1]))
    assert [m.tolist() for m in moves] == ([move] if any(move) else [])
    assert t.netted_lots == 10  # wanted 5, 10 and 5 lots in the books; the target asks 5, 0 and 5 of them
    for m in moves:
        t.position = t.position + m
    t._become(np.array([-1, 1]))  # asked again with the same states: nothing more is wanted of the relations
    assert t.netted_lots == 10


@pytest.mark.parametrize("rate, band", [(1.0, 0), (0.5, 0), (0.2, 0), (1.0, 2), (0.5, 2)])
def test_what_netting_saved_when_the_relations_change_part_of_the_way_is_the_same_wherever_the_trader_is(rate, band):
    weights = np.array([[1.0, -1.0, 0.0], [0.0, -1.0, 1.0]])
    t = RelationTrader(weights=weights, window=3, entry=0.5, size=5, rate=rate, band=band)
    t.attach(3)
    for m in t._become(np.array([1, 0])):  # asks [5, -5, 0]: it is at most part of the way
        t.position = t.position + m
    assert t.netted_lots == 0 and abs(t.position).max() <= 5
    t._become(np.array([-1, 1]))
    # It wanted 10, 15 and 5 lots in the books, and what is asked changes by 10, 5 and 5: book 1, crossed once for the
    # relations that would cross it three times, saves 10. At a gap of -7, 2 and 5 from [2, -2, 0] it would be 16.
    assert t.netted_lots == 10
    assert type(t.netted_lots) is int  # a count of lots, and not a float that equals one


@pytest.mark.parametrize("settings, saved", [
    ({}, [0, 4, 8]),  # the defaults: against the gap to the target, from where it is, which is still flat
    ({"rate": 0.5}, [0, 0, 0]),  # at a rate: against what the states asked last time, whatever was traded
    ({"band": 2}, [0, 0, 0]),
])
def test_what_netting_saved_with_orders_that_are_not_filled_is_counted_from_the_position_with_the_defaults(settings,
                                                                                                          saved):
    # Two relations with a leg in common, and orders that never fill: it stays flat, and the moves are never applied.
    t = StateTrader(weights=np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]), size=4, **settings)
    t.attach(3)
    counted = []
    for states in ([1, 0], [1, 1], [0, 1]):
        t._become(np.array(states))
        counted.append(t.netted_lots)
        assert type(t.netted_lots) is int and t.position.tolist() == [0, 0, 0]
    # By hand, with the defaults, the targets are [4, -4, 0], [4, 0, -4] and [0, 4, -4], from flat. The first wanted
    # [4, 4, 0] and is 4, 4 and 0 lots from its target: none saved. The second wanted [0, 4, 4] and is 4, 0 and 4 from
    # its target, so it saves 4, all in book 1, whose target and position are both 0. The third wanted [4, 4, 0] and is
    # 0, 4 and 4 from its target: 4 more saved, in book 0. At a rate or a band the ask is the change in the target,
    # which here is all that the relations wanted of each book: nothing saved.
    assert counted == saved


def test_what_netting_saved_when_a_relation_turns_around_part_of_the_way_is_nothing_with_one_relation():
    t = RelationTrader(weights=np.array([[1.0, -1.0, 0.0]]), window=3, entry=0.5, size=5, rate=0.5)
    t.attach(3)
    t.position = t.position + t._become(np.array([1]))[0]  # asks [5, -5, 0]: 5 lots each, 2 traded (2.5 goes to 2)
    assert t.position.tolist() == [2, -2, 0] and t.netted_lots == 0
    move = t._become(np.array([-1]))[0]
    # It wanted 10 lots each way in books 0 and 1 (from long to short), and what is asked changes by 10: netting
    # saved none. The target [-5, 5, 0] is 7 from [2, -2, 0], but the 3 lots short of 10 were not netted: just not
    # traded yet. The aim goes half of the way from [2.5, -2.5, 0] to the target, to [-1.25, 1.25, 0], and the gap to
    # it from [2, -2, 0] is 3.25, which rounds to 3.
    assert t.netted_lots == 0
    assert move.tolist() == [-3, 3, 0]


def test_lots_not_yet_traded_in_a_book_are_not_counted_as_netted_when_another_relation_opens():
    weights = np.array([[1.0, -1.0, 0.0], [0.0, -1.0, 1.0]])
    t = RelationTrader(weights=weights, window=3, entry=0.5, size=5, rate=0.5)
    t.attach(3)
    t.position = t.position + t._become(np.array([1, 0]))[0]  # asks [5, -5, 0]; it is at [2, -2, 0]
    assert t.netted_lots == 0
    t._become(np.array([1, -1]))  # the second relation opens: wanted 0, 5 and 5 lots; the ask [5, 0, -5] changes so
    assert t.netted_lots == 0  # (the gaps are 3, 2 and 5, which would make three lots of book 1 netted: they are not)


@pytest.mark.parametrize("rate, band", [(1.0, 1), (1.0, 2), (0.5, 0), (0.2, 0), (0.5, 1)])
def test_a_pair_that_opens_and_closes_has_nothing_netted_however_little_of_it_was_traded(rate, band):
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=3, rate=rate, band=band)
    t.attach(2)
    for states in ([1], [0], [0], [0], [-1], [0], [0], [0]):  # opens, closes at once, and the same the other way
        for move in t._become(np.array(states)):
            t.position = t.position + move
    assert t.entries == 2
    assert t.netted_lots == 0  # one relation has nothing to net against: it is only closing what it opened


def test_what_a_trader_asked_for_is_nothing_once_its_relations_are_gone():
    t = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5, size=8, rate=0.5)
    t.attach(2)
    t.position = t.position + t._become(np.array([-1]))[0]  # asks [-8, 8]: it is at [-4, 4]
    weights, t.weights = t.weights, None
    t.position = t.position + t._become(np.zeros(0, dtype=int))[0]  # nothing to hold it against: closed
    assert t.position.tolist() == [0, 0]
    assert t._aim.tolist() == [0, 0] and t._asked.tolist() == [0, 0]  # and it aims at nothing
    t.weights, t.states = weights, np.array([0])
    move = t._become(np.array([-1]))[0]  # asks [-8, 8] from flat: all of it is new, and none is netted
    assert t.netted_lots == 0
    assert move.tolist() == [-4, 4]  # half of the way from an aim of 0, and not from the [-4, 4] it had


def test_a_trader_whose_relations_are_gone_closes_what_it_holds_at_once_whatever_its_settings():
    # The spec changes the netted path alone: with nothing to hold a position against, it is closed whole.
    choices = iter([np.array([[1.0, -1.0]]), np.zeros((0, 2))])
    t = RelationTrader(choose=lambda history: next(choices), formation=3, window=3, entry=0.5, size=8, refit=2,
                       rate=0.5, band=1)
    feed = Mids(t, 2)
    feed.see(100, 100)
    feed.see(100, 100)
    assert feed.see(110, 100)[0].tolist() == [-3, 3]  # the aim is 4, and the band of 1 is beyond: 3
    assert feed.see(110, 100)[0].tolist() == [-2, 2]  # the aim is 6, and it is 3 off, 2 beyond the band: at [-5, 5]
    move = feed.see(110, 100)[0]  # nothing is chosen: all of it, and not what the aim and the band would leave
    assert t.weights is None and move.tolist() == [5, -5] and t.position.tolist() == [0, 0]


class Fixed:
    """A pair trader that says what it is told."""

    z = None

    def __init__(self, states):
        self.states = list(states)

    def decide(self, a, b):
        return self.states.pop(0)


def test_a_copula_portfolio_goes_toward_its_target_a_decision_at_a_time_with_the_pairs_states_unchanged():
    p = CopulaPortfolio(pairs=1, size=8, rate=0.5)
    p.attach(2)
    p.traders, p._books = [Fixed([-1] * 6 + [0] * 3)], [(0, 1)]
    p.weights, p.states = np.array([[-1.0, 1.0]]), np.array([0])
    sent = []
    for _ in range(9):
        moves = p.decide(np.array([100.0, 100.0]))
        sent.append([m.tolist() for m in moves])
        for move in moves:
            p.position = p.position + move
    # short the gap sells the second book and buys the first: the aim is 4, 6, 7, 7.5, 7.75 and 7.875 lots; and then
    # flat, with an aim of 3.94, 1.97 and 0.98, which it goes to all the way
    assert sent == [[[4, -4]], [[2, -2]], [[1, -1]], [], [[1, -1]], [], [[-4, 4]], [[-2, 2]], [[-1, 1]]]
    assert p.entries == 1


def test_a_copula_portfolio_with_a_band_trades_to_its_edge_and_waits_there():
    p = CopulaPortfolio(pairs=1, size=8, band=3)
    p.attach(2)
    p.traders, p._books = [Fixed([-1] * 4)], [(0, 1)]
    p.weights, p.states = np.array([[-1.0, 1.0]]), np.array([0])
    moves = p.decide(np.array([100.0, 100.0]))
    assert [m.tolist() for m in moves] == [[5, -5]]
    p.position = p.position + moves[0]
    for _ in range(3):
        assert p.decide(np.array([100.0, 100.0])) == []


def scripted_residuals(**settings):
    t = Scripted(factors=1, window=10, pca_window=10, pca_every=10**6, size=8, **settings)
    t.attach(2)
    t._eigen = np.array([[1.0], [1.0]])
    t.fine = np.array([True, True])
    t.betas = [[0.0, 0.0]]  # no hedge: a book's relation is the book
    for _ in range(11):
        t._mids.append(np.array([100.0, 100.0]))
        t._recent.append(np.array([100.0, 100.0]))
    return t


def test_a_residual_trader_goes_toward_its_target_at_later_decisions_whose_s_scores_hold_the_state():
    t = scripted_residuals(rate=0.5)
    # opens a long at -1.3; -0.6 is not above minus the long's closing level of 0.5: it holds; -0.4 is: it closes
    t.script = [np.array([s, 0.0]) for s in [-1.3] + [-0.6] * 5 + [-0.4] * 4]
    sent, held = [], []
    for _ in range(10):
        moves = t.decide(np.array([100.0, 100.0]))
        sent.append([m.tolist() for m in moves])
        for move in moves:
            t.position = t.position + move
        held.append(t.position.tolist()[0])
    # the aim is 4, 6, 7, 7.5, 7.75 and 7.875 lots; then, flat, 3.94, 1.97, 0.98 and 0.49
    assert sent == [[[4, 0]], [[2, 0]], [[1, 0]], [], [[1, 0]], [], [[-4, 0]], [[-2, 0]], [[-1, 0]], [[-1, 0]]]
    assert held == [4, 6, 7, 7, 8, 8, 4, 2, 1, 0]
    assert t.entries == 1


def test_a_residual_trader_with_a_band_stays_at_its_edge():
    t = scripted_residuals(band=3)
    t.script = [np.array([s, 0.0]) for s in [-1.3, -0.6, -0.6, -0.4, -0.4]]
    sent = []
    for _ in range(5):
        moves = t.decide(np.array([100.0, 100.0]))
        sent.append([m.tolist() for m in moves])
        for move in moves:
            t.position = t.position + move
    # long: 8 lots less the band of 3 is 5; held. Flat: all the way, and not to the band's edge
    assert sent == [[[5, 0]], [], [], [[-5, 0]], []]
    assert t.position.tolist() == [0, 0]


@pytest.mark.parametrize("make, first", [
    (lambda: RelationTrader(weights=np.array([[1.0, -1, 0, 0, 0, 0], [0, 0, 0, 1.0, -1, 0]]), window=30, entry=1.0,
                            size=6, rate=0.5, band=1), 29),
    (lambda: ResidualTrader(factors=2, window=30, pca_window=100, size=6, rate=0.5, band=1), 100),
    (lambda: CopulaPortfolio(pairs=2, window=100, trading=40, entry=0.5, size=6, rate=0.5, band=1, interval=1.0), 101),
], ids=["relations", "residuals", "copula pairs"])
def test_every_order_comes_from_become_at_every_decision_it_can_trade_and_from_nowhere_else(make, first):
    history = history_of(SECTORS, 260)
    trader = make()
    trader.attach(6)
    sent = []
    become = trader._become

    def spy(states):
        sent.append(become(states))
        return sent[-1]

    trader._become = spy
    traded = 0
    for i, mids in enumerate(history):
        before = len(sent)
        moves = trader.decide(mids)
        if i < first:
            assert len(sent) == before and moves == []  # nothing to hold a position against yet
            continue
        assert len(sent) == before + 1 and moves is sent[-1], i  # asked at every decision, and nothing else sent
        for move in moves:
            trader.position = trader.position + move
            traded += 1
    assert traded > 5 and len(sent) == len(history) - first


# ---- Cost-aware trading: every kind, and the defaults against the code before it ----

# Each builds a trader from a statarb module and a crosssection module (the current ones, or the baseline's).
STATE_KINDS = {
    "fixed pairs": lambda s, x, **kw: s.RelationTrader(
        weights=np.array([[1.0, -1, 0, 0, 0, 0], [0, 0, 0, 1.0, -1, 0]]), window=40, entry=1.0, size=6, **kw),
    "a basket of three": lambda s, x, **kw: s.RelationTrader(
        weights=np.array([[1.0, -2, 1, 0, 0, 0]]), window=40, entry=1.0, size=6, **kw),
    "pairs by distance": lambda s, x, **kw: s.RelationTrader(
        choose=partial(s.pairs_by_distance, count=2), formation=200, window=40, entry=1.0, size=6, **kw),
    "pairs chosen again": lambda s, x, **kw: s.RelationTrader(
        choose=partial(s.pairs_by_distance, count=2), formation=150, window=40, entry=1.0, size=6, refit=100, **kw),
    "residuals": lambda s, x, **kw: s.ResidualTrader(factors=2, window=30, pca_window=100, interval=2.0, **kw),
    "copula pairs": lambda s, x, **kw: s.CopulaPortfolio(pairs=2, window=100, trading=40, interval=2.0, entry=0.5,
                                                         **kw),
}
CURRENT = (statarb_now, cross_now)
SETTINGS = [{"rate": 0.3}, {"band": 2.0}, {"rate": 0.5, "band": 1.0}]
SETTING_IDS = ["rate 0.3", "band 2", "rate 0.5 and band 1"]
_BASELINE = {}


def _load(name, source):
    module = ModuleType(name)
    sys.modules[name] = module  # dataclasses look their module up by name
    exec(compile(source, f"{BASELINE}:{name}", "exec"), module.__dict__)
    return module


def baseline_modules():
    """``lob.statarb`` and ``lob.crosssection`` as they were at BASELINE, as modules of their own (the second built on
    the first). Skips the test if git cannot show them."""
    if not _BASELINE:
        if shutil.which("git") is None:
            pytest.skip("git is not available")
        sources = []
        for path in ("lob/statarb.py", "lob/crosssection.py"):
            try:
                shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE}:{path}"], capture_output=True,
                                       text=True)
            except OSError:
                pytest.skip("git cannot be run")
            if shown.returncode:
                pytest.skip(f"git cannot show {path} at {BASELINE}")
            sources.append(shown.stdout)
        statarb = _load("_baseline_statarb", sources[0])
        current = sys.modules["lob.statarb"]
        sys.modules["lob.statarb"] = statarb  # so that crosssection's "from lob.statarb import ..." finds the baseline
        try:
            crosssection = _load("_baseline_crosssection", sources[1])
        finally:
            sys.modules["lob.statarb"] = current
        _BASELINE["modules"] = (statarb, crosssection)
    return _BASELINE["modules"]


def played(modules, make, seed=2, span=400.0, **settings):
    """The market, the trader and the score of a trader built from these modules, run from its warm-up."""
    s, x = modules
    trader = make(s, x, **settings)
    market = s.BookMarket(s.FactorValues(s.sector_loadings(2, 3)), seed=seed, traders=(trader,))
    return market, trader, s.score_portfolio(market, trader, trader.warm_up + 1.0, span)


def same(a, b):
    return a is b if a is None or b is None else np.array_equal(a, b, equal_nan=True)


def assert_as_before(make):
    """A trader with the defaults does number for number what the baseline's did: its score, account and path."""
    (old_market, old, old_score), (new_market, new, new_score) = (played(baseline_modules(), make),
                                                                  played(CURRENT, make))
    assert type(old).__module__ != type(new).__module__ and old_market.__class__ is not new_market.__class__
    assert new.lots > 20 and new.entries > 2 and new_score["costs"] > 0  # it traded, a good deal
    assert old_score.keys() == new_score.keys()
    for key in old_score:
        assert old_score[key] == new_score[key], key
    for name in ("cash", "lots", "spread_paid", "entries", "netted_lots"):
        assert getattr(old, name) == getattr(new, name), name
    assert same(old.position, new.position) and same(old.states, new.states) and same(old.signal(), new.signal())
    assert old_market.time == new_market.time and old_market.events == new_market.events


@pytest.mark.parametrize("name", STATE_KINDS)
def test_with_the_defaults_each_kind_of_trader_does_number_for_number_what_it_did_before_cost_aware_trading(name):
    assert_as_before(STATE_KINDS[name])


def test_a_trader_not_netted_with_the_defaults_does_number_for_number_what_it_did_before_cost_aware_trading():
    assert_as_before(lambda s, x: s.RelationTrader(choose=partial(s.pairs_by_distance, count=2), formation=200,
                                                   window=40, entry=1.0, size=6, netted=False))


@cache
def scored(name, rate=1.0, band=0.0):
    return played(CURRENT, STATE_KINDS[name], rate=rate, band=band)[2]


@pytest.mark.parametrize("settings", SETTINGS, ids=SETTING_IDS)
@pytest.mark.parametrize("name", STATE_KINDS)
def test_with_a_rate_or_a_band_each_kind_trades_fewer_lots_than_with_the_defaults_on_the_same_seed(name, settings):
    plain, careful = scored(name), scored(name, **settings)
    assert plain["lots"] > 100 or name == "copula pairs"
    assert 0 < careful["lots"] < plain["lots"]
    assert careful["entries"] > 0 and careful["costs"] < plain["costs"]


# ---- Stage 4, prep: what the books and the traders have ready for an index and for deals ----
# A trader's cash book by book, a trader's say over the market it joins, a book of another scale, the move of a book's
# orders at a jump of its fair value, what a trader holds of the factors over the long run, and Johansen's twelve books.
# With every option off the market is held to 42ae383 to the bit in test_stage4_identity.py.

CASH_KINDS = {
    **STATE_KINDS,
    "pairs not netted": lambda s, x: s.RelationTrader(
        weights=np.array([[1.0, -1, 0, 0, 0, 0], [0, 0, 0, 1.0, -1, 0]]), window=40, entry=1.0, size=6, netted=False),
    "reversal": lambda s, x: x.ReversalTrader(lookback=3, factors=1, pca_window=40, pca_every=10, hold=3,
                                              interval=2.0),
    "momentum": lambda s, x: x.MomentumTrader(lookback=5, skip=2, factors=1, pca_window=40, pca_every=10, hold=3,
                                              interval=2.0),
    "lead-lag": lambda s, x: x.LeadLagTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3,
                                             interval=2.0),
    "flow of trades": lambda s, x: x.FlowTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3,
                                                interval=2.0),
    "flow of queues": lambda s, x: x.FlowTrader(lookback=1, factors=0, measure="queues", hold=1, interval=2.0),
}


def watch_what_traders_pay(market):
    """What each trader's orders took from its cash in each book, from the fills they got, kept apart from the
    market's own: a dict from the trader's id to its book-by-book total, filled as the market runs."""
    paid, inside = {}, []
    trade, submit = market._trade, market.submit

    def watched_trade(trader, book, need):
        inside.append(trader)
        try:
            trade(trader, book, need)
        finally:
            inside.pop()

    def watched_submit(book, order, impact=True):
        fills = submit(book, order, impact)
        if inside:
            sign = 1 if order.side == Side.BID else -1
            row = paid.setdefault(id(inside[-1]), np.zeros(market.n))
            row[book] -= sign * sum(f.price * f.size for f in fills)
        return fills

    market._trade, market.submit = watched_trade, watched_submit
    return paid


@pytest.mark.parametrize("name", CASH_KINDS)
def test_book_cash_is_kept_a_book_each_and_adds_up_to_cash_for_every_kind_of_trader(name):
    trader = CASH_KINDS[name](statarb_now, cross_now)
    market = BookMarket(FactorValues(SECTORS), seed=2, traders=(trader,))
    paid = watch_what_traders_pay(market)
    score_portfolio(market, trader, trader.warm_up + 1.0, 150.0)
    assert trader.lots > 20 and np.count_nonzero(trader.book_cash) >= 2  # it traded, in more than one book
    assert trader.book_cash.shape == (6,)
    assert np.array_equal(trader.book_cash, paid[id(trader)])  # each book's own, as its fills say
    assert trader.book_cash.sum() == trader.cash  # whole ticks and lots: the sums are exact


def test_a_fill_pays_into_the_book_it_was_made_in_and_a_sale_takes_its_cash_from_that_book_alone():
    fair = FactorValues(np.ones((2, 1)), volatility=0.0, own_volatility=0.0, impact=0.0)
    m = BookMarket(fair, seed=1, depth=3, lots=10)
    trader = RelationTrader(weights=np.array([[1.0, -1.0]]), window=3, entry=0.5)
    m.add(trader)
    assert trader.book_cash.tolist() == [0.0, 0.0]
    m._trade(trader, 0, 15)  # ten lots at 3001 and five at 3002
    assert trader.book_cash.tolist() == [-(10 * 3001 + 5 * 3002), 0.0]
    m._trade(trader, 1, -15)  # ten lots at 2999 and five at 2998
    assert trader.book_cash.tolist() == [-(10 * 3001 + 5 * 3002), 10 * 2999 + 5 * 2998]
    assert trader.cash == -40.0 and trader.book_cash.sum() == -40.0
    trader.attach(2)  # joining a market starts the cash of each book over, as it does the position
    assert trader.book_cash.tolist() == [0.0, 0.0] and trader.position.tolist() == [0, 0]


class Picky(RelationTrader):
    """A trader that wants at least three books, and notes what it saw of the market it was asked to check."""

    def check(self, market):
        self.asked = (market, self.position)
        if market.n < 3:
            raise ValueError("this market has too few books for me")


def test_a_market_asks_a_trader_to_check_it_before_the_trader_attaches_and_a_refused_trader_is_not_added():
    weights = np.array([[1.0, -1.0, 0.0]])
    market = BookMarket(FactorValues(SECTORS), seed=1)
    trader = Picky(weights=weights, window=3)
    market.add(trader)
    assert trader.asked[0] is market and trader.asked[1] is None  # asked before it had a position
    assert trader.position is not None and market.traders[0] is trader
    two = BookMarket(FactorValues(np.ones((2, 1))), seed=1)
    refused = Picky(weights=weights[:, :2], window=3)
    with pytest.raises(ValueError, match="too few books for me"):
        two.add(refused)
    assert two.traders == [] and refused.position is None and not hasattr(refused, "book_cash")
    with pytest.raises(ValueError, match="too few books for me"):  # made with the market, the same
        BookMarket(FactorValues(np.ones((2, 1))), seed=1, traders=(Picky(weights=weights[:, :2], window=3),))


@pytest.mark.parametrize("name", CASH_KINDS)
def test_no_kind_of_trader_asks_anything_of_a_market_by_default(name):
    trader = CASH_KINDS[name](statarb_now, cross_now)
    market = BookMarket(FactorValues(SECTORS), seed=1)
    assert trader.check(market) is None and BookTrader.check(trader, market) is None
    assert getattr(trader, "position", None) is None  # it only looked
    market.add(trader)
    assert market.traders[0] is trader


class Scaled(FactorValues):
    """Fair values that give each book a scale, as an index will: the market reads ``scales``."""

    def __init__(self, loadings, scales, **kwargs):
        super().__init__(loadings, **kwargs)
        self.scales = np.asarray(scales, dtype=float)


def books_as_made(market):
    """Each book's resting orders in the order its id list keeps them: price, size and side."""
    return [[(o.price, o.size, o.side) for o in map(book.get_order, book.all_order_ids())] for book in market.books]


def test_a_book_with_a_scale_has_a_model_that_reaches_8_ticks_a_scale_and_a_starting_book_spaced_by_it():
    fair = Scaled(np.ones((4, 1)), [1.0, 1.0, 4.0, 3.0], level=3000.0)
    m = BookMarket(fair, seed=1, depth=3, lots=10)
    assert [model.reach for model in m.models] == [8, 8, 32, 24]
    assert all(type(model) is AnchoredZIModel for model in m.models)
    assert all((model.limit_rate, model.market_rate, model.cancel_rate, model.max_size) == (10.0, 1.5, 0.05, 5)
               for model in m.models)
    spaced = {0: 1, 1: 1, 2: 4, 3: 3}
    for k, s in spaced.items():
        assert m.books[k].depth_profile(Side.BID, 9) == [(3000 - d * s, 10) for d in (1, 2, 3)]
        assert m.books[k].depth_profile(Side.ASK, 9) == [(3000 + d * s, 10) for d in (1, 2, 3)]
    m.run(20.0)  # and it runs
    assert m.events > 100


def test_a_scale_below_a_sixteenth_still_gives_a_model_that_reaches_a_tick():
    fair = Scaled(np.ones((3, 1)), [1.0, 1.0, 0.02], level=3000.0)  # 8 * 0.02 rounds to 0 ticks
    assert [model.reach for model in BookMarket(fair, seed=1).models] == [8, 8, 1]


def two_books(scales, level):
    """Fair values of two books at a level, with these scales given to them (None: as made with no scales at all)."""
    if scales is None:
        return FactorValues(np.ones((2, 1)), level=level)
    return Scaled(np.ones((2, 1)), scales, level=level)


@pytest.mark.parametrize("scales", [None, [1.0, 2.5]])
def test_the_starting_orders_of_every_book_are_made_at_time_0(scales):
    m = BookMarket(two_books(scales, 100.0), seed=1, depth=3, lots=4)
    for book in m.books:
        orders = book.resting_orders(Side.BID) + book.resting_orders(Side.ASK)
        assert len(orders) == 6 and {order.timestamp for order in orders} == {0.0}


@pytest.mark.parametrize("scales", [None, [1.0, 1.0]])
def test_a_starting_bid_is_never_below_a_tick_in_a_book_whose_fair_value_is_near_it(scales):
    # 10 less 1 to 15 reaches 9, 8, ... 1 and then 0, -1, ... -5
    m = BookMarket(two_books(scales, 10.0), seed=1, depth=15, lots=2)
    for book in m.books:
        assert book.depth_profile(Side.BID, 20) == [(price, 2) for price in range(9, 1, -1)] + [(1, 14)]  # 7 at a tick
        assert book.depth_profile(Side.ASK, 20) == [(price, 2) for price in range(11, 26)]


def test_a_book_of_scale_1_is_made_as_a_book_with_no_scales_and_given_models_are_kept():
    plain = BookMarket(FactorValues(SECTORS), seed=3)
    ones = BookMarket(Scaled(SECTORS, np.ones(6)), seed=3)
    assert plain.fair.scales is None and plain.models[0].reach == 8
    assert books_as_made(plain) == books_as_made(ones)  # the same orders, in the same order
    assert [model.reach for model in ones.models] == [8] * 6
    assert plain.run(30.0).equals(ones.run(30.0)) and plain.events == ones.events
    mine = [AnchoredZIModel(reach=5) for _ in range(3)]
    kept = BookMarket(Scaled(np.ones((3, 1)), [1.0, 2.0, 5.0]), models=mine, seed=1)
    assert [model.reach for model in kept.models] == [5, 5, 5]
    assert all(a is b for a, b in zip(kept.models, mine, strict=True))


def queues_of(book):
    """Every resting order as (price, id, size), per side, best price first and in priority order."""
    return {side: [(o.price, o.order_id, o.size) for o in book.resting_orders(side)] for side in (Side.BID, Side.ASK)}


def a_book_with_queues():
    """Level 100, a book 3 lots a price and 4 prices a side, and two more orders at a price on each side."""
    fair = FactorValues(np.ones((3, 1)), level=100.0, volatility=0.0, own_volatility=0.0, impact=0.1)
    m = BookMarket(fair, seed=3, depth=4, lots=3)
    book = m.books[0]
    for price, size, side in ((99, 2, Side.BID), (99, 5, Side.BID), (97, 1, Side.BID), (101, 4, Side.ASK),
                              (103, 2, Side.ASK), (103, 1, Side.ASK)):
        book.add_order(limit_order(price, size, 1.0, side))
    return m


@pytest.mark.parametrize("ticks", [7, -9, 1, -1, 40, 3.0])
def test_a_shift_moves_every_resting_order_by_the_ticks_and_keeps_every_queue_in_its_order(ticks):
    m = a_book_with_queues()
    book = m.books[0]
    before, depth = queues_of(book), {s: book.depth_profile(s, 50) for s in (Side.BID, Side.ASK)}
    assert [len(book.orders_at(99, Side.BID)), len(book.orders_at(103, Side.ASK))] == [3, 3]
    m._shift(0, ticks)
    after = queues_of(book)
    for side in (Side.BID, Side.ASK):  # the same orders, in the same order, each the ticks higher
        assert after[side] == [(p + int(ticks), oid, size) for p, oid, size in before[side]]
        assert book.depth_profile(side, 50) == [(p + int(ticks), d) for p, d in depth[side]]
    assert [o.order_id for o in book.orders_at(99 + int(ticks), Side.BID)] == [
        oid for p, oid, _ in before[Side.BID] if p == 99]  # the queue at a price keeps its order
    assert book.num_orders() == 14 and sorted(book.all_order_ids()) == sorted(
        oid for side in before.values() for _, oid, _ in side)
    assert book.mid_price == 100.0 + int(ticks)
    assert book.best_bid < book.best_ask


def test_a_shift_by_nothing_changes_nothing_and_a_part_of_a_tick_is_refused():
    m = a_book_with_queues()
    book, before, ids = m.books[0], queues_of(m.books[0]), m.books[0].all_order_ids()
    m._shift(0, 0)
    m._shift(0, 0.0)
    assert queues_of(book) == before and book.all_order_ids() == ids
    with pytest.raises(ValueError, match="whole number"):
        m._shift(0, 2.5)
    assert queues_of(book) == before


@pytest.mark.parametrize("ticks", [-8, -30])
def test_a_shift_down_leaves_no_price_below_a_tick_and_no_bid_at_or_above_an_ask(ticks):
    fair = FactorValues(np.ones((2, 1)), level=12.0, volatility=0.0, own_volatility=0.0)
    m = BookMarket(fair, seed=1, depth=8, lots=2)  # bids 11 down to 4, asks 13 up to 20
    book = m.books[0]
    old = queues_of(book)
    m._shift(0, ticks)
    new = queues_of(book)
    prices = [p for side in new.values() for p, _, _ in side]
    assert min(prices) >= 1 and book.best_bid < book.best_ask
    assert [oid for _, oid, _ in new[Side.BID]] == [oid for _, oid, _ in old[Side.BID]]  # best priority first still
    assert [oid for _, oid, _ in new[Side.ASK]] == [oid for _, oid, _ in old[Side.ASK]]
    assert [p for p, _, _ in new[Side.BID]] == sorted((p for p, _, _ in new[Side.BID]), reverse=True)
    assert [p for p, _, _ in new[Side.ASK]] == sorted(p for p, _, _ in new[Side.ASK])
    low = [oid for p, oid, _ in new[Side.BID] if p == 1]  # those a fall took to the lowest price: in their old order
    assert low == [oid for _, oid, _ in old[Side.BID]][-len(low):] and len(low) >= 5
    assert book.orders_at(1, Side.BID) == [book.get_order(oid) for oid in low]
    assert min(p for p, _, _ in new[Side.ASK]) >= 2
    if ticks == -30:  # every ask came down to the lowest it may
        assert book.depth_profile(Side.ASK, 5) == [(2, 16)] and book.depth_profile(Side.BID, 5) == [(1, 16)]


def test_a_shift_counts_no_event_flow_volume_or_impact_and_leaves_the_fair_values_the_other_books_and_the_draws():
    m = BookMarket(FactorValues(SECTORS, level=1000.0), seed=2)
    m.run(30.0)  # books with a past: orders of every size, flow and volume
    assert m.events > 500 and np.abs(m.flow).sum() > 0 and m.volume.sum() > 0

    def state():
        return (m.events, m.flow.copy(), m.volume.copy(), list(m.fair.fairs), m.fair.own.copy(), m.fair.pushed,
                m.rng.bit_generator.state, m.fair._rng.bit_generator.state, m.time)

    def book_of(k):
        return queues_of(m.books[k]), list(m._rates)[k]

    before, books = state(), [book_of(k) for k in range(6)]
    m.fair.trade = lambda *args: pytest.fail("a shift is not a trade: it has no impact")
    m._shift(2, -13)
    after = state()
    assert before[0] == after[0] and before[3] == after[3] and before[5:] == after[5:]
    assert np.array_equal(before[1], after[1]) and np.array_equal(before[2], after[2])  # flow and volume
    assert np.array_equal(before[4], after[4])  # the own parts
    assert [book_of(k) for k in range(6) if k != 2] == [b for k, b in enumerate(books) if k != 2]
    shifted = queues_of(m.books[2])
    assert shifted[Side.BID] == [(p - 13, oid, size) for p, oid, size in books[2][0][Side.BID]]
    assert shifted[Side.ASK] == [(p - 13, oid, size) for p, oid, size in books[2][0][Side.ASK]]


def test_a_shift_takes_the_books_rate_again_and_moves_its_mark_at_once():
    m = a_book_with_queues()
    m._rates[0], m._last_mid[0] = 99.0, 0.0  # as if stale
    m._shift(0, 6)
    assert m._rates[0] == m.models[0].rate(m.books[0]) and m._rates[0] != 99.0
    assert m._last_mid[0] == m.books[0].mid_price == 106.0 and m.marks()[0] == 106.0
    assert m._rates[1] == m.models[1].rate(m.books[1])  # the others as they were


class Jumps(FactorValues):
    """Fair values that put jumps in ``gapped`` at chosen times, as deals will: ``{time: [(book, ticks), ...]}``, each
    by the first call of ``advance_to`` at or after its time. They move no fair value: only the quotes follow."""

    def __init__(self, loadings, jumps, **kwargs):
        super().__init__(loadings, **kwargs)
        self.jumps = sorted(jumps.items())
        self.filled = []  # the times it was filled at

    def advance_to(self, t):
        super().advance_to(t)
        while self.jumps and self.jumps[0][0] <= t:
            self.gapped.extend(self.jumps.pop(0)[1])
            self.filled.append(t)


class Watcher(BookTrader):
    """A trader that (with an interval of 0) decides at every event, trades nothing and keeps the mids it saw."""

    def decide(self, mids):
        self.seen.append((self.time_of(), mids.copy()))
        return []


def silenced(market, log):
    """Every model's events are none (nothing changes a book but what is done to it); each notes the book it fell to
    and what every book's best bid was then, and what was left in ``gapped``."""
    for k, model in enumerate(market.models):
        model.event = lambda book, t, fair, rng, k=k: log.append(
            (k, [b.best_bid for b in market.books], list(market.fair.gapped))) or None


def test_a_step_with_a_jump_shifts_the_book_before_the_event_and_before_any_trader_decides():
    fair = Jumps(np.ones((3, 1)), {2.0: [(0, 6.4)], 5.0: [(1, -3.6), (0, -6.4)]}, volatility=0.0, own_volatility=0.0,
                 level=1000.0)
    watcher = Watcher(interval=0.0)
    watcher.seen = []
    m = BookMarket(fair, seed=4, traders=(watcher,))
    watcher.time_of = lambda: m.time
    log = []
    silenced(m, log)
    while not fair.filled:
        m.step()
    assert fair.filled[0] == m.time >= 2.0 and fair.gapped == []  # read and emptied by that step
    assert log[-1][1] == [1005, 999, 999]  # the bid of book 0 is 999 + round(6.4) when the event is drawn
    assert log[-1][2] == []  # and nothing is left waiting
    assert all(entry[1] == [999, 999, 999] for entry in log[:-1])
    when, mids = watcher.seen[-1]
    assert when == m.time and mids.tolist() == [1006.0, 1000.0, 1000.0]  # the decision of that step saw the shift
    assert all(mids[0] == 1000.0 for t, mids in watcher.seen[:-1])
    assert m._last_mid[0] == m.books[0].mid_price == m.marks()[0] == 1006.0  # marked at the new mid at once
    assert m.events == 0 and m.flow.tolist() == [0.0] * 3
    while len(fair.filled) < 2:
        m.step()
    assert [b.mid_price for b in m.books] == [1000.0, 996.0, 1000.0]  # book 1 by -4 and book 0 back by -6


def test_a_step_without_a_jump_shifts_nothing_and_a_jump_of_less_than_half_a_tick_nothing_either(monkeypatch):
    plain = BookMarket(FactorValues(np.ones((3, 1))), seed=4)
    shifted = []
    original = plain._shift
    monkeypatch.setattr(plain, "_shift", lambda book, ticks: shifted.append((book, ticks)) or original(book, ticks))
    for _ in range(300):
        plain.step()
    assert shifted == [] and plain.fair.gapped == []
    small = Jumps(np.ones((3, 1)), {0.5: [(2, 0.3), (1, -0.4)]})
    m = BookMarket(small, seed=4)
    before = [queues_of(b) for b in m.books]
    silenced(m, [])
    while not small.filled:
        m.step()
    assert small.gapped == [] and [queues_of(b) for b in m.books] == before  # read, rounded to 0 and emptied


def test_a_jump_costs_no_draw_and_the_clock_and_the_generators_go_on_as_they_would_have():
    def market(jumps):
        m = BookMarket(Jumps(np.ones((3, 1)), jumps, volatility=0.0, own_volatility=0.0), seed=5)
        silenced(m, [])  # no model draws anything
        return m

    with_jumps, without = market({0.3: [(0, 7.0)], 0.9: [(2, -5.0), (1, 2.0)]}), market({})
    for _ in range(60):
        with_jumps.step()
        without.step()
    assert with_jumps.fair.filled and not without.fair.filled
    assert with_jumps.time == without.time and with_jumps.rng.bit_generator.state == without.rng.bit_generator.state
    assert with_jumps.fair._rng.bit_generator.state == without.fair._rng.bit_generator.state
    assert queues_of(with_jumps.books[0]) != queues_of(without.books[0])  # but the book moved


class Rated(AnchoredZIModel):
    """A model with a rate of its own choosing, whose events are none: it notes the books they fall to."""

    def __init__(self, name, log, rate_of):
        super().__init__()
        self.name, self.log, self.rate_of = name, log, rate_of

    def rate(self, book):
        return self.rate_of(book)

    def event(self, book, t, fair, rng):
        self.log.append(self.name)


class Drawn:
    """A generator that draws the same two numbers every time: a step takes 0.01 and a uniform of one half."""

    def __init__(self, real):
        self.real = real

    def exponential(self, scale):
        return 0.01

    def random(self):
        return 0.5

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_the_event_after_a_jump_is_drawn_with_the_rates_after_it_and_never_falls_through_to_the_last_book():
    # Book 0 is busy (rate 1000) while its bid is at 990 or more and quiet (1) below, and the others are quiet. A jump
    # down makes book 0 quiet: a draw of one half of the total is then in book 1, but one half of the old total is
    # far past them all, and would be left on the last book.
    log = []
    busy = lambda book: 1000.0 if book.best_bid >= 990 else 1.0  # noqa: E731
    models = [Rated(0, log, busy), Rated(1, log, lambda book: 1.0), Rated(2, log, lambda book: 1.0)]
    fair = Jumps(np.ones((3, 1)), {0.05: [(0, -20.0)], 0.08: [(0, 20.0)]}, volatility=0.0, own_volatility=0.0,
                 level=1000.0)
    m = BookMarket(fair, models=models, seed=1)
    m.rng = Drawn(m.rng)
    assert m._rates == [1000.0, 1.0, 1.0]
    for _ in range(4):
        m.step()
    assert log == [0, 0, 0, 0] and fair.filled == []  # half of 1002 is in book 0
    m.step()  # the fifth, at 0.05: book 0 goes down 20 ticks and is quiet: half of 3 is in the middle book
    assert fair.filled == [pytest.approx(0.05)] and m._rates == [1.0, 1.0, 1.0] and log[-1] == 1
    for _ in range(2):
        m.step()
    assert log[5:] == [1, 1]  # still quiet
    m.step()  # at 0.08 it goes back up: busy, and half of 1002 is in book 0 (a stale 3 would have said book 1)
    assert fair.filled[1] == pytest.approx(0.08) and m._rates == [1000.0, 1.0, 1.0] and log[-1] == 0
    assert m.books[0].best_bid == 999


# ---- Stage 4, prep: the risk and the factor part of what a trader holds ----

SCORE_KEYS = {"pnl", "before_costs", "costs", "lots", "entries", "gross", "exposure", "deviation", "netted",
              "risk_adjusted"}
HOLDING_KEYS = SCORE_KEYS | {"deviation_long", "risk_adjusted_long", "factor_part"}
HELD = {
    "a hedged pair": lambda: RelationTrader(weights=np.array([[1.0, -1, 0, 0, 0, 0]]), window=30, entry=1.0),
    "one book alone": lambda: RelationTrader(weights=np.array([[1.0, 0, 0, 0, 0, 0]]), window=30, entry=1.0),
}


def sampled_by_hand(trader, seed, burn, span, **kwargs):
    """A trader's P&L at each unit of time from the start of the span, what it held of each factor and the factors at
    each: as a plain loop over the market's steps, apart from the library's own."""
    m = BookMarket(FactorValues(SECTORS, **kwargs), seed=seed, traders=(trader,))
    while m.time < burn:
        m.step()
    values = [trader.value(m.marks())]
    held = [[float(sum(trader.position[b] * SECTORS[b, j] for b in range(6))) for j in range(3)]]
    factors = [m.fair.factors.tolist()]
    next_sample, until = m.time + 1.0, m.time + span
    while m.time < until:
        m.step()
        while m.time >= next_sample:
            values.append(trader.value(m.marks()))
            held.append([float(sum(trader.position[b] * SECTORS[b, j] for b in range(6))) for j in range(3)])
            factors.append(m.fair.factors.tolist())
            next_sample += 1.0
    return values, held, factors, m, trader


@pytest.mark.parametrize("name", HELD)
@pytest.mark.parametrize("horizon, span", [(50, 400.0), (100.0, 600.0)])
def test_score_holdings_gives_the_long_risk_and_the_factor_part_a_plain_loop_gives(name, horizon, span):
    values, held, factors, _, _ = sampled_by_hand(HELD[name](), 3, 100.0, span)
    long = [values[i] - values[i - int(horizon)] for i in range(int(horizon), len(values), int(horizon))]
    part = sum(held[i - 1][j] * (factors[i][j] - factors[i - 1][j]) for i in range(1, len(values)) for j in range(3))
    trader = HELD[name]()
    score = score_holdings(BookMarket(FactorValues(SECTORS), seed=3, traders=(trader,)), trader, 100.0, span,
                           horizon=horizon)
    assert len(long) >= 4 and np.std(long) > 0
    assert score["deviation_long"] == pytest.approx(np.std(long) / math.sqrt(horizon), rel=1e-9)
    mean_change = (values[-1] - values[0]) / (len(values) - 1)
    assert score["risk_adjusted_long"] == pytest.approx(mean_change / (np.std(long) / math.sqrt(horizon)), rel=1e-9)
    assert score["factor_part"] == pytest.approx(part, rel=1e-9, abs=1e-9)
    assert (name == "a hedged pair") == (score["factor_part"] == 0)  # a relation holds none of the factors
    assert score["deviation_long"] != score["deviation"]  # and they are not the unit figure


def test_score_holdings_is_what_score_portfolio_gives_for_the_rest_and_three_keys_more():
    def market():
        trader = HELD["one book alone"]()
        return BookMarket(FactorValues(SECTORS), seed=3, traders=(trader,)), trader

    (m1, t1), (m2, t2) = market(), market()
    plain, holdings = score_portfolio(m1, t1, 100.0, 300.0), score_holdings(m2, t2, 100.0, 300.0)
    assert set(plain) == SCORE_KEYS and set(holdings) == HOLDING_KEYS
    assert all(holdings[key] == plain[key] for key in plain)  # the same keys, to the bit
    assert (t1.cash, t1.lots, m1.events, m1.time) == (t2.cash, t2.lots, m2.events, m2.time)


def test_score_holdings_takes_stretches_of_100_units_unless_told_and_of_1_unit_is_the_risk_a_unit():
    def scored(**kwargs):
        trader = HELD["one book alone"]()
        return score_holdings(BookMarket(FactorValues(SECTORS), seed=3, traders=(trader,)), trader, 100.0, 300.0,
                              **kwargs)

    default, hundred, unit = scored(), scored(horizon=100.0), scored(horizon=1)
    assert default == hundred and default["deviation_long"] > 0  # 3 stretches of 100 in 300 units: not 1 of 200
    assert unit["deviation_long"] == unit["deviation"] > 0 and unit["risk_adjusted_long"] == unit["risk_adjusted"]


def test_score_holdings_gives_the_risk_adjusted_figure_also_when_the_long_risk_is_under_a_unit():
    def one_lot():
        return RelationTrader(weights=np.array([[1.0, 0, 0, 0, 0, 0]]), window=30, entry=1.0, size=1)

    values, _, _, _, _ = sampled_by_hand(one_lot(), 3, 100.0, 600.0)
    long = [values[i] - values[i - 100] for i in range(100, len(values), 100)]
    risk = np.std(long) / 10.0
    trader = one_lot()
    score = score_holdings(BookMarket(FactorValues(SECTORS), seed=3, traders=(trader,)), trader, 100.0, 600.0)
    assert 0.0 < risk < 1.0 and score["deviation_long"] == pytest.approx(risk, rel=1e-9)
    mean_change = (values[-1] - values[0]) / (len(values) - 1)
    assert mean_change != 0 and score["risk_adjusted_long"] == pytest.approx(mean_change / risk, rel=1e-9)


def test_a_span_with_no_whole_unit_in_it_scores_with_no_warning_and_no_long_risk():
    trader = HELD["one book alone"]()
    m = BookMarket(FactorValues(SECTORS), seed=3, traders=(trader,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a mean or a deviation of nothing says so by a warning: there is none to take
        score = score_holdings(m, trader, 100.0, 0.5)
    assert set(score) == HOLDING_KEYS
    assert (score["deviation"], score["deviation_long"], score["risk_adjusted"], score["risk_adjusted_long"]) == (
        0.0, 0.0, 0.0, 0.0)


def test_the_long_risk_is_0_without_two_stretches_of_the_horizon_and_a_horizon_below_a_unit_is_refused():
    trader = HELD["one book alone"]()
    m = BookMarket(FactorValues(SECTORS), seed=3, traders=(trader,))
    score = score_holdings(m, trader, 100.0, 150.0)  # one stretch of 100 and 50 over
    assert score["deviation"] > 0 and score["deviation_long"] == 0.0 and score["risk_adjusted_long"] == 0.0
    score = score_holdings(m, trader, m.time, 150.0, horizon=60)  # two
    assert score["deviation_long"] > 0
    for horizon in (0.5, 0, 12.5):
        with pytest.raises(ValueError, match="horizon"):
            score_holdings(m, trader, m.time, 10.0, horizon=horizon)


class Doubled(FactorValues):
    """Fair values whose exposures are twice their loadings: nothing else about them differs."""

    def exposures(self):
        return 2.0 * self.loadings


def test_score_portfolio_measures_exposure_through_exposures_and_gives_the_same_keys():
    def scored(fair_class, score):
        trader = HELD["one book alone"]()
        return score(BookMarket(fair_class(SECTORS), seed=3, traders=(trader,)), trader, 100.0, 300.0)

    plain, doubled = scored(FactorValues, score_portfolio), scored(Doubled, score_portfolio)
    assert set(plain) == set(doubled) == SCORE_KEYS and plain["exposure"] > 0
    assert doubled["exposure"] == pytest.approx(2 * plain["exposure"], rel=1e-12)
    assert all(doubled[key] == plain[key] for key in SCORE_KEYS - {"exposure"})  # the market is the same market
    held_plain, held_doubled = scored(FactorValues, score_holdings), scored(Doubled, score_holdings)
    assert set(held_doubled) == HOLDING_KEYS
    assert held_doubled["factor_part"] == pytest.approx(2 * held_plain["factor_part"], rel=1e-12)
    assert held_doubled["factor_part"] != 0


# ---- Stage 4, prep: Johansen's test takes twelve books at most ----

def books_with_one_walk(books, units=700, seed=1):
    """Mids of this many books, each a multiple of one random walk and a noise of its own: every basket whose weights
    cancel the walk is a relation."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(units))
    return 100.0 + walk[:, None] * (1.0 + 0.1 * np.arange(books)) + 2.0 * rng.standard_normal((units, books))


def test_johansen_relations_refuses_thirteen_books_and_takes_twelve(monkeypatch):
    twelve = books_with_one_walk(12)
    assert johansen_relations(twelve, count=2, every=5).shape[1] == 12  # a table to read the rank from: fine
    monkeypatch.setattr("statsmodels.tsa.vector_ar.vecm.coint_johansen",
                        lambda *args, **kwargs: pytest.fail("it has no critical values to test against"))
    for books in (13, 16):
        with pytest.raises(ValueError, match="critical values for 12 books at most"):
            johansen_relations(books_with_one_walk(books), count=1)


def test_baskets_by_the_trace_test_and_by_box_tiao_work_with_thirteen_books_as_before():
    history = books_with_one_walk(13)
    rows = best_baskets(history, count=2, legs=3, every=5)
    assert rows.shape == (2, 13) and (np.count_nonzero(rows, axis=1) == 3).all()
    assert (np.abs(rows).max(axis=1) == 1).all()
    sparse = sparse_baskets(history, count=2, legs=3, every=5)
    assert sparse.shape == (2, 13) and (np.count_nonzero(sparse, axis=1) == 3).all()
