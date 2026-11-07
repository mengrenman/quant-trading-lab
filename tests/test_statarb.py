"""Many books with several common factors, and traders of more than one pair (lob/statarb.py)."""

import copy
import itertools
import math
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from lob.order import Side
from lob.pairs import BETA_FLOOR, FLOOR, CopulaTrader
from lob.statarb import (BookMarket, CopulaPortfolio, FactorValues, RelationTrader, ResidualTrader, best_baskets,
                         johansen_relations, mixed_loadings, pairs_by_distance, pairs_by_pvalue, score_portfolio,
                         sector_loadings, trace_rank)

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
