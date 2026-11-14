"""With every option of stage 4 off, the books, the fair values and the traders are those of commit 42ae383, to the bit.

Stage 4 adds an index book, deals, a trend and a few things around them (``lob/statarb.py``). Each sits behind a branch
the default never enters, and nothing that was already drawn may move: so a market run with the new code and none of
the new options is held, number for number, to ``lob/statarb.py`` and ``lob/crosssection.py`` as git shows them at
BASELINE (``test_statarb.py`` does the same for the commit before cost-aware trading, and keeps its own constant).
What is compared is every number a run leaves: the frame ``run`` gives, the events and the time, the state of both
random generators (a draw taken or moved shows there at once), the order flow's ``flow`` and ``volume``, the fair
values and their parts, the factors, the loadings, each book's rates, depth and resting orders in the order the cancel
draw indexes them, and for each trader its account, lots, spread paid, entries, netted lots, position, states, signal
and ``score_portfolio``. ``lob/book.py``, ``lob/pairs.py``, ``lob/order.py`` and ``lob/matching.py`` are not read out
of git: the rule is that they do not change.
"""

import shutil
import subprocess
import sys
from functools import cache, partial
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import lob.crosssection as cross_now
import lob.statarb as statarb_now
from lob.order import Side
from lob.statarb import BookTrader, FactorValues, sector_loadings

BASELINE = "42ae383"  # the commit before stage 4: what the defaults must still do, to the bit
ROOT = Path(__file__).resolve().parents[1]
CURRENT = (statarb_now, cross_now)
SECTORS = sector_loadings(2, 3)
MIXED = np.array([[1.0, 0.2], [0.8, 0.9], [1.2, -0.5], [0.6, 0.4], [1.1, 1.0], [0.9, -0.2]])
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
        statarb = _load("_stage4_baseline_statarb", sources[0])
        current = sys.modules["lob.statarb"]
        sys.modules["lob.statarb"] = statarb  # so that crosssection's "from lob.statarb import ..." finds the baseline
        try:
            crosssection = _load("_stage4_baseline_crosssection", sources[1])
        finally:
            sys.modules["lob.statarb"] = current
        _BASELINE["modules"] = (statarb, crosssection)
    return _BASELINE["modules"]


# ---- What is played ----

# Markets: the table, the fair values' settings, the seed and the books' own (depth and lots of the starting book).
MARKETS = {
    "sectors at level 1000": dict(table=SECTORS, fair={"level": 1000.0}, seed=1),
    "sectors at level 3000": dict(table=SECTORS, fair={}, seed=2),
    "at the floor": dict(table=SECTORS, fair={"level": 30.0, "volatility": 2.0}, seed=3),
    "drifting loadings in a mixed table": dict(
        table=MIXED, fair={"level": 1000.0, "loading_sd": 0.3, "loading_half_life": 200.0}, seed=4),
    "no half-life, a large impact and a shallow starting book": dict(
        table=SECTORS, fair={"level": 1000.0, "half_life": None, "impact": 0.3}, seed=5,
        market={"depth": 5, "lots": 7}),
}

# Traders, each built from the modules it runs in: the state kinds and the cross-sectional ones, with the settings the
# other identity tests use (short windows, so that each trades inside a few hundred units).
KINDS = {
    "pairs": lambda s, x: s.RelationTrader(
        weights=np.array([[1.0, -1, 0, 0, 0, 0], [0, 0, 0, 1.0, -1, 0]]), window=40, entry=1.0, size=6),
    "pairs not netted": lambda s, x: s.RelationTrader(
        weights=np.array([[1.0, -1, 0, 0, 0, 0], [0, 0, 0, 1.0, -1, 0]]), window=40, entry=1.0, size=6, netted=False),
    "pairs with a rate and a band": lambda s, x: s.RelationTrader(
        weights=np.array([[1.0, -1, 0, 0, 0, 0], [0, 0, 0, 1.0, -1, 0]]), window=40, entry=1.0, size=6, rate=0.5,
        band=1.0),
    "pairs chosen again": lambda s, x: s.RelationTrader(
        choose=partial(s.pairs_by_distance, count=2), formation=150, window=40, entry=1.0, size=6, refit=100),
    "residuals": lambda s, x: s.ResidualTrader(factors=2, window=30, pca_window=100, interval=2.0),
    "copula pairs": lambda s, x: s.CopulaPortfolio(pairs=2, window=100, trading=40, interval=2.0, entry=0.5),
    "reversal": lambda s, x: x.ReversalTrader(lookback=3, factors=1, pca_window=40, pca_every=10, hold=3,
                                              interval=2.0, rate=0.5),
    "momentum": lambda s, x: x.MomentumTrader(lookback=5, skip=2, factors=1, pca_window=40, pca_every=10, hold=3,
                                              interval=2.0),
    "lead-lag": lambda s, x: x.LeadLagTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3,
                                             interval=2.0),
    "flow of trades": lambda s, x: x.FlowTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3,
                                                interval=2.0),
    "flow of queues": lambda s, x: x.FlowTrader(lookback=1, factors=0, measure="queues", hold=1, interval=2.0),
}
TOGETHER = {  # several in one market, in this order: the first is the one scored
    "the sectors market": ("sectors at level 1000", ["pairs", "pairs chosen again", "residuals", "reversal",
                                                     "flow of trades"], 1),
    "the floor": ("at the floor", ["pairs", "momentum", "flow of queues"], 3),
    "drifting loadings in a mixed table": ("drifting loadings in a mixed table", ["pairs", "residuals", "lead-lag"],
                                           4),
}
BARE_UNITS = 500.0
SPAN = 150.0  # units a trader is scored over, after the run


def build(modules, market, kinds, seed=None):
    """A market of these settings from these modules, with a trader of each of ``kinds`` in it."""
    s, x = modules
    spec = MARKETS[market]
    fair = s.FactorValues(spec["table"], **spec["fair"])
    traders = [KINDS[kind](s, x) for kind in kinds]
    return s.BookMarket(fair, seed=spec["seed"] if seed is None else seed, traders=traders, **spec.get("market", {})), \
        traders


def books_of(market):
    """Everything each book holds: depth either side, order counts, and its resting orders in the order its id list
    keeps them (the cancel draw indexes that list)."""
    out = []
    for book in market.books:
        resting = [(o.price, o.size, o.side, o.timestamp) for o in map(book._orders.__getitem__, book._id_list)]
        out.append((book.depth_profile(Side.BID, 500), book.depth_profile(Side.ASK, 500), book.num_orders(),
                    book.num_orders(Side.BID), book.num_orders(Side.ASK), book.total_depth(Side.BID),
                    book.total_depth(Side.ASK), resting))
    return out


def observed(market, traders, frame, score=None):
    """Every number a run leaves, as a dict of names to values."""
    fair = market.fair
    out = {"frame": frame.to_numpy(float), "columns": list(frame.columns), "events": market.events,
           "time": market.time, "rng": market.rng.bit_generator.state, "fair rng": fair._rng.bit_generator.state,
           "flow": market.flow.copy(), "volume": market.volume.copy(), "pushed": fair.pushed,
           "factors": fair.factors.copy(), "own": fair.own.copy(), "common": fair.common.copy(),
           "loadings": fair.loadings.copy(), "mean loadings": fair.mean_loadings.copy(), "level": fair.level.copy(),
           "fairs": list(fair.fairs), "marks": market.marks(), "rates": list(market._rates), "books": books_of(market),
           "steps": fair._steps, "block at": fair._at, "block": fair._z.shape, "draws": fair._draws,
           "n": (market.n, fair.n, fair.k), "last mid": list(market._last_mid), "traders": len(market.traders)}
    for i, trader in enumerate(traders):
        for name in ("cash", "lots", "spread_paid", "entries", "netted_lots", "active", "next_time"):
            out[f"trader {i} {name}"] = getattr(trader, name)
        for name in ("position", "states", "_aim", "_asked"):
            value = getattr(trader, name, None)
            out[f"trader {i} {name}"] = None if value is None else np.array(value)
        signal = trader.signal()
        out[f"trader {i} signal"] = None if signal is None else np.array(signal)
        out[f"trader {i} value"] = trader.value(market.marks())
    if score is not None:
        out["score"] = score
    return out


def same(a, b):
    """Equal to the bit: arrays with ``array_equal`` (a NaN equal to a NaN), floats with ``==`` or both NaN."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


@cache
def played(which, market, kinds=(), seed=None):
    """The numbers of one run: ``which`` is 'old' (BASELINE) or 'new' (the working tree). Bare, it runs ``BARE_UNITS``;
    with traders it runs through the warm-up of the slowest and 250 units beyond it, and scores the first of them."""
    modules = baseline_modules() if which == "old" else CURRENT
    m, traders = build(modules, market, kinds, seed)
    assert (m.__class__ is statarb_now.BookMarket) == (which == "new")
    if not traders:
        return observed(m, traders, m.run(BARE_UNITS)), m, traders
    until = max(t.warm_up for t in traders) + 251.0
    frame = m.run(until)
    score = modules[0].score_portfolio(m, traders[0], m.time, SPAN)
    return observed(m, traders, frame, score), m, traders


def assert_as_at_baseline(market, kinds=(), seed=None):
    """The run from the working tree is that of BASELINE in every number, and the run was not a dull one."""
    (old, old_market, _), (new, new_market, new_traders) = played("old", market, kinds, seed), played(
        "new", market, kinds, seed)
    assert old_market.__class__ is not new_market.__class__
    assert old.keys() == new.keys()
    for key in old:
        assert same(old[key], new[key]), key
    assert new["events"] > 10_000 and new["frame"].shape[0] > 100
    for trader in new_traders:
        assert trader.lots > 20 and trader.entries > 0, type(trader).__name__  # each traded: a dull run proves little
    if new_traders:
        assert new["score"]["lots"] > 0 and new["score"].keys() == old["score"].keys()
    return new_market


# ---- The proof ----

@pytest.mark.parametrize("market", MARKETS)
def test_with_the_new_options_off_a_bare_market_is_that_of_42ae383_to_the_tick(market):
    new_market = assert_as_at_baseline(market)
    if market == "at the floor":
        assert new_market.fair.pushed > 0  # the floor was met: the run held something
    if market.startswith("drifting"):
        assert not np.array_equal(new_market.fair.loadings, new_market.fair.mean_loadings)
    if market.startswith("no half-life"):
        assert new_market.fair.half_life is None and np.abs(new_market.fair.own).max() > 5


@pytest.mark.parametrize("kind", KINDS)
def test_with_the_new_options_off_each_kind_of_trader_alone_does_what_it_did_at_42ae383_to_the_tick(kind):
    assert_as_at_baseline("sectors at level 3000", (kind,))


@pytest.mark.parametrize("name", TOGETHER)
def test_with_the_new_options_off_several_traders_together_do_what_they_did_at_42ae383_to_the_tick(name):
    market, kinds, seed = TOGETHER[name]
    new_market = assert_as_at_baseline(market, tuple(kinds), seed)
    assert len(new_market.traders) == len(kinds)


def test_the_view_with_its_defaults_makes_the_market_the_library_makes_at_42ae383():
    from app import statarb as view

    values = {p.name: p.default for p in view.MARKET_PARAMS} | {"shape": "sectors"}  # a new session's settings
    assert values["reverting"] and not values["drifting"]
    seed = values["seed"]
    old_s, old_x = baseline_modules()
    fair = old_s.FactorValues(view.loadings_of(values), level=view.LEVEL, volatility=values["volatility"],
                              own_volatility=values["own_volatility"], half_life=values["half_life"],
                              impact=values["impact"], loading_sd=0.0, loading_half_life=values["loading_half_life"])
    pairs = old_s.RelationTrader(weights=np.eye(12)[:2] - np.eye(12)[2:4], window=40, entry=1.0, size=6)
    old = old_s.BookMarket(fair, seed=seed, traders=(pairs,))
    trader = statarb_now.RelationTrader(weights=np.eye(12)[:2] - np.eye(12)[2:4], window=40, entry=1.0, size=6)
    new = view.market_from(values, seed, (trader,))
    old_frame, new_frame = old.run(300.0), new.run(300.0)
    a, b = observed(old, [pairs], old_frame), observed(new, [trader], new_frame)
    assert type(new.fair) is FactorValues and new.fair.n == 12
    assert a.keys() == b.keys() and all(same(a[key], b[key]) for key in a), [k for k in a if not same(a[k], b[k])]
    assert new.events > 10_000 and trader.lots > 20


def test_the_files_the_proof_does_not_read_out_of_git_are_those_of_42ae383():
    # The two modules above are read out of git; the books, the orders, the matching and the models are imported from
    # the working tree for both runs, so a change to them would be seen in neither. They are not to change.
    baseline_modules()  # skips when git cannot show them
    for name in ("book.py", "order.py", "matching.py", "pairs.py"):
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE}:lob/{name}"], capture_output=True,
                               text=True)
        if shown.returncode:
            pytest.skip(f"git cannot show lob/{name} at {BASELINE}")
        assert (ROOT / "lob" / name).read_text() == shown.stdout, f"lob/{name} is not as it was at {BASELINE}"


# ---- What the defaults add, and where ----

def test_a_trader_gets_book_cash_from_attach_and_it_is_not_a_field_of_any_trader_class():
    import dataclasses

    traders = [KINDS[kind](statarb_now, cross_now) for kind in KINDS]
    for trader in traders:
        assert not hasattr(trader, "book_cash")  # made when it joins a market, not when it is built
        assert "book_cash" not in {f.name for f in dataclasses.fields(trader)}
    market = statarb_now.BookMarket(FactorValues(SECTORS), seed=1, traders=traders)
    for trader in market.traders:
        assert trader.book_cash.shape == (6,) and not trader.book_cash.any()
        assert "book_cash" not in {f.name for f in dataclasses.fields(trader)}
    for cls in (BookTrader, statarb_now.StateTrader, statarb_now.RelationTrader, statarb_now.ResidualTrader,
                statarb_now.CopulaPortfolio, cross_now.CrossSectionTrader, cross_now.FlowTrader):
        assert "book_cash" not in {f.name for f in dataclasses.fields(cls)}
        assert "book_cash" not in cls.__dict__.get("__dataclass_fields__", {})


def test_exposures_with_no_deal_are_the_loadings_object_itself():
    for table, settings in ((SECTORS, {}), (MIXED, {"loading_sd": 0.3, "loading_half_life": 200.0})):
        fair = FactorValues(table, **settings)
        assert fair.exposures() is fair.loadings
        fair.advance_to(100.0)  # drifting loadings are a new array at each step: the same object, whichever it is
        assert fair.exposures() is fair.loadings
    fair = FactorValues(SECTORS)
    assert fair.deals == [] and fair.ended == [] and fair.gapped == [] and fair.premium == 0.0
    assert fair.index_weights is None and fair.scales is None and fair.deal_settings is None and fair.m == 6
