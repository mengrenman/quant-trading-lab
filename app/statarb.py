"""The dashboard's Stat-arb view: many order books with several common factors, and traders of many relations.

One ``StatArbSession`` per browser session (``LOBManager.statarb``, made when first needed), apart from the other
markets, with its own clock, speed and event stream (``/statarb/events``, which a tab holds in place of ``/events``
while the view is shown). The market's settings say how many books and factors there are (sectors of books, or
loadings in no order), and ``lob/statarb.py`` runs it. The chart reads ``samples``, one a unit of time: every book's
mid, and each trader's P&L, what it holds and its signals. A new trader can learn from the mids kept (``MIDS_KEPT``),
since what it chooses from is far longer than the chart.

``StatArbSession`` fits a trader's settings on training runs of their own (``TrainedMarket.fit_trader`` in
app/training.py): ``training_run`` is one candidate alone in one fresh market from the same settings, and
``StatArbSession.plan`` says what a fit tries, within ``MAX_FIT_BOOK_UNITS``.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Optional

import numpy as np

from app.model_specs import Param
from app.pairs import COLORS, MAX_TRADERS, Choice, Switch, TraderKind, _r, parse
from app.training import TrainedMarket, fit_note, plan_fit, trades_from
from lob.crosssection import FlowTrader, LeadLagTrader, MomentumTrader, ReversalTrader
from lob.sparse_baskets import sparse_baskets
from lob.statarb import (BookMarket, BookTrader, CopulaPortfolio, FactorValues, RelationTrader, ResidualTrader,
                         best_baskets, johansen_relations, mixed_loadings, pairs_by_distance, pairs_by_pvalue,
                         score_portfolio, sector_loadings, trace_rank)

LEVEL = 1000.0  # where every book starts, far above the fair values' floor (lob.pairs.FLOOR)
SAMPLES = 3_000  # chart points kept, one a unit of time
MIDS_KEPT = 20_000  # mids kept for a new trader to learn from, one a unit of time
SPEED_MAX = 500  # time units a second (about 18 events a book each)
SPEED_DEFAULT = 50
MAX_STEP = 5_000  # time units one step request may run
MAX_SIGNALS = 12  # signals on the chart, for the trader shown
MIN_TEST_UNITS = 1_000  # the least the tests are run on: over less, they say little (notebook 12)
# The most books times candidates times runs times units of time in one fit: about two minutes on 15 workers at any
# size of market, since a worker runs about 9,000 book-units a second.
MAX_FIT_BOOK_UNITS = 16_000_000
SHAPES = {"sectors": "Sectors", "mixed": "Mixed"}
CROSS_SECTIONAL = ("reversal", "momentum", "leadlag", "flow")  # kinds that score every book (lob/crosssection.py)

MARKET_PARAMS: tuple[Param | Switch, ...] = (
    Param("sectors", "Sectors", 3, 1, 4, 1, integer=True),
    Param("books_each", "Books each", 4, 2, 4, 1, integer=True),
    Param("books", "Books", 12, 3, 12, 1, integer=True),
    Param("factors", "Factors", 3, 1, 4, 1, integer=True),
    Param("table_seed", "Loadings' seed", 3, 0, 10_000, 1, integer=True),
    Switch("reverting", "The own parts mean-revert", True),
    Param("half_life", "Half-life", 50.0, 2.0, 5_000.0, 1.0),
    Param("own_volatility", "Own volatility", 0.8, 0.0, 5.0, 0.1),
    Param("volatility", "Factor volatility", 1.0, 0.0, 5.0, 0.1),
    Param("impact", "Impact a lot", 0.05, 0.0, 1.0, 0.01),
    Param("seed", "Seed", 1, 0, 2**31 - 1, 1, integer=True),
    Switch("drifting", "Loadings drift", False),
    Param("loading_sd", "Their sd", 0.2, 0.01, 1.0, 0.01),
    Param("loading_half_life", "Their half-life", 2000.0, 10.0, 100_000.0, 10.0),
)
WARM = Switch("warm", "Learn from the past first", True)
INTERVAL = Param("interval", "Decide every", 5.0, 1.0, 50.0, 1.0)
LOTS = ("size", "Lots of the largest leg", 5.0, 0.5, 50.0, 0.5)
HOLD = Param("hold", "Hold for", 100, 1, 2000, 1, integer=True)
NEUTRAL = (Param("factors", "Factors", 3, 0, 8, 1, integer=True),
           Param("pca_window", "PCA over", 500, 10, 2000, 1, integer=True))  # 0 factors: the dollar alone
# Trading less for the same signal (lob.statarb.BookTrader._toward): part of the way to the target at each decision,
# and not within a band of lots of it. A rate of 1 and no band go all the way, as every kind did before.
COSTS = (Param("rate", "Trade rate", 1.0, 0.01, 1.0, 0.01), Param("band", "No-trade band", 0.0, 0.0, 10.0, 0.5))
# What a fit proposes for them: the rates and bands that mattered on seeds 1 to 8 (notebook 13, section 5). A trader of
# relations gains from going a quarter to half of the way, one that scores every book from a tenth or less, rebalancing
# at every decision (a hold of 1: at a long hold the rate changes little).
COST_GRID = {"rate": (0.1, 0.25, 0.5, 1.0), "band": (0.0, 0.5, 1.0, 2.0)}
SCORE_COST_GRID = {"rate": (0.02, 0.05, 0.1, 1.0), "band": (0.0, 1.0, 2.0, 4.0)}
TRADING = ("size", "rate", "band", "interval")  # the last settings every kind is made with
LAST = (*COSTS, INTERVAL, WARM)  # and the last it shows, but for lead-lag's interval
Z_PARAMS = (Param("formation", "Choose from", 2000, 50, 4000, 50, integer=True),
            Param("window", "Window", 400, 3, 2000, 1, integer=True),
            Param("entry", "Enter beyond z", 2.0, 0.1, 10.0, 0.1),
            Param("exit", "Flat within z", 0.0, 0.0, 9.9, 0.1),
            Param("refit", "Choose again every", 0, 0, 4000, 50, integer=True))


def loadings_of(s: dict[str, Any]) -> np.ndarray:
    if s["shape"] == "sectors":
        return sector_loadings(s["sectors"], s["books_each"])
    return mixed_loadings(s["books"], s["factors"], seed=s["table_seed"])


def book_names(s: dict[str, Any]) -> list[str]:
    """Each book's name: its sector's letter and its number there, or just its number."""
    if s["shape"] == "sectors":
        return [f"{'ABCD'[i]}{j + 1}" for i in range(s["sectors"]) for j in range(s["books_each"])]
    return [str(i + 1) for i in range(s["books"])]


def market_from(s: dict[str, Any], seed: int, traders=()) -> BookMarket:
    """A fresh market from the view's settings, with these traders in it."""
    fair = FactorValues(loadings_of(s), level=LEVEL, volatility=s["volatility"], own_volatility=s["own_volatility"],
                        half_life=s["half_life"] if s["reverting"] else None, impact=s["impact"],
                        loading_sd=s["loading_sd"] if s["drifting"] else 0.0, loading_half_life=s["loading_half_life"])
    return BookMarket(fair, seed=seed, traders=traders)


def training_run(kind: str, values: dict[str, Any], settings: dict[str, Any], seed: int, burn: float,
                 span: float) -> dict[str, float]:
    """How one candidate did alone in one training market (lob.statarb.score_portfolio); runs in a worker process."""
    trader = TRADER_KINDS[kind].build(values)
    return score_portfolio(market_from(settings, seed, (trader,)), trader, burn, span)


def _make_pairs(count, by_pvalue, formation, window, entry, exit, refit, size, rate, band, interval):  # noqa: A002
    choose = partial(pairs_by_pvalue, count=count, every=1) if by_pvalue else partial(pairs_by_distance, count=count)
    return RelationTrader(choose=choose, formation=formation, window=window, entry=entry, exit=exit,
                          refit=refit or None, size=size, rate=rate, band=band, interval=interval)


def _make_baskets(method, legs, count, formation, window, entry, exit, refit, size, rate, band,  # noqa: A002
                  interval):
    if legs == 1:
        raise ValueError("baskets of two books or more, or 0 for the vectors on all the books")
    if method == "box-tiao":  # legs 0: the portfolios on all the books
        choose = partial(sparse_baskets, count=count, legs=legs, every=1)
    else:
        choose = (partial(best_baskets, count=count, legs=legs, every=1) if legs
                  else partial(johansen_relations, count=count, every=1))
    return RelationTrader(choose=choose, formation=formation, window=window, entry=entry, exit=exit,
                          refit=refit or None, size=size, rate=rate, band=band, interval=interval)


def _trading_less(v: dict[str, Any], sentence: bool = False) -> str:
    """How it trades less than the whole way to its target, if it does (lob.statarb.BookTrader._toward), else '': as
    a sentence of its own, or to end one."""
    goes, trades = ("It goes", "does not trade") if sentence else ("going", "not trading")
    rate = f"{goes} {v['rate']:g} of the way to its target at each decision" if v["rate"] < 1 else ""
    lots = f"{v['band']:g} lot{'' if v['band'] == 1 else 's'}"
    band = f"{trades} within {lots} of {'that' if rate else 'its target'}, except to close a book" if v["band"] else ""
    if sentence and band and not rate:
        band = f"It {band}"
    less = " and ".join(part for part in (rate, band) if part)
    return f"{less}." if sentence and less else less


def _and_less(v: dict[str, Any], said: str) -> str:
    """What a trader of relations does, and then how it trades less, if it does."""
    less = _trading_less(v, sentence=True)
    return f"{said} {less}" if less else said


def _describe_pairs(v: dict[str, Any]) -> str:
    by = "the lowest Engle-Granger p-values" if v["by_pvalue"] else "the least squared distance"
    again = f", again every {v['refit']:,}" if v["refit"] else ""
    return _and_less(v, f"The {v['count']} pair{'' if v['count'] == 1 else 's'} with {by} over {v['formation']:,} "
                        f"decisions{again}, each hedged by least squares; {v['size']:g} lots of its larger leg beyond "
                        f"z of {v['entry']:g} over {v['window']:,} decisions, every {v['interval']:g} units; orders "
                        f"netted.")


def _describe_baskets(v: dict[str, Any]) -> str:
    again = f", again every {v['refit']:,}" if v["refit"] else ""
    trading = (f"{v['size']:g} lots of the largest leg beyond z of {v['entry']:g} over {v['window']:,} decisions, "
               f"every {v['interval']:g} units")
    if v["method"] == "box-tiao":
        count = v["count"]
        if v["legs"] == 2:  # nothing is grown: the pairs themselves are ranked
            what = (f"Up to {count} pair{'' if count == 1 else 's'} ranked by Box-Tiao predictability (d'Aspremont "
                    f"2011), the least predictable first and none rejected")
        elif v["legs"]:
            what = (f"Up to {count} basket{'' if count == 1 else 's'} of {v['legs']} books grown greedily from every "
                    f"pair by Box-Tiao predictability (d'Aspremont 2011), the least predictable first and none "
                    f"rejected")
        elif count == 1:
            what = "The least predictable portfolio of all the books (Box and Tiao 1977), none rejected"
        else:  # as many as there are books at most
            what = (f"Up to {count} of the least predictable portfolios of all the books (Box and Tiao 1977), none "
                    f"rejected")
        return _and_less(v, f"{what}, from {v['formation']:,} decisions{again}; {trading}; there is no test: it "
                            f"always gives baskets.")
    what = f"baskets of {v['legs']} books, by their trace statistics" if v["legs"] else "vectors on all the books"
    return _and_less(v, f"Up to {v['count']} Johansen {what}, from {v['formation']:,} decisions{again}; {trading}.")


def _describe_residuals(v: dict[str, Any]) -> str:
    return _and_less(v, f"Each book's residual against {v['factors']} PCA factor{'' if v['factors'] == 1 else 's'} "
                        f"from {v['pca_window']:,} decisions, fitted over {v['window']:,}, every {v['interval']:g} "
                        f"units: {v['size']:g} lots and its hedge beyond an s-score of {v['open_at']:g}, closed "
                        f"inside {v['close_long']:g} (long) and {v['close_short']:g} (short). Hedges net across the "
                        f"books.")


def _describe_copula(v: dict[str, Any]) -> str:
    return _and_less(v, f"The {v['pairs']} pair{'' if v['pairs'] == 1 else 's'} with the least distance over "
                        f"{v['window']:,} decisions, each through a copula fitted to them and again every "
                        f"{v['trading']}; {v['size']:g} lots of the larger leg when the flags part beyond "
                        f"{v['entry']:g}, every {v['interval']:g} units.")


def _neutral(v: dict[str, Any], window: bool = True) -> str:
    """What a cross-sectional trader's weights are neutral to: the dollar, and its PCA factors."""
    n = v["factors"]
    if not n:
        return "the dollar alone"
    factors = f"the dollar and {n} PCA factor{'' if n == 1 else 's'}"
    return f"{factors} from {v['pca_window']:,} decisions" if window else factors


def _held(v: dict[str, Any]) -> str:
    less = _trading_less(v)
    every = "at every decision" if v["hold"] == 1 else f"every {v['hold']:,} decisions"
    return (f"{v['size']:g} lots of the largest, rebalanced {every}, deciding every "
            f"{v['interval']:g} unit{'' if v['interval'] == 1 else 's'}{', ' + less if less else ''}")


def _describe_reversal(v: dict[str, Any]) -> str:
    return (f"Buys the books that fell over the last {v['lookback']:,} decisions and sells those that rose, neutral to "
            f"{_neutral(v)}: {_held(v)}.")


def _describe_momentum(v: dict[str, Any]) -> str:
    over = (f"the {v['lookback']:,} decisions before the last {v['skip']:,}" if v["skip"]
            else f"the last {v['lookback']:,} decisions")
    return f"Buys the books that rose over {over} and sells those that fell, neutral to {_neutral(v)}: {_held(v)}."


def _describe_leadlag(v: dict[str, Any]) -> str:
    return (f"Buys the books that the other books' last moves say will rise, by a ridge regression (penalty "
            f"{v['ridge']:g}) over {v['pca_window']:,} decisions, neutral to {_neutral(v, window=False)}: "
            f"{_held(v)}.")


def _describe_flow(v: dict[str, Any]) -> str:
    if v["measure"] == "queues":
        reading = "the lots at the best bid against those at the best ask, now"
    else:
        reading = f"the share of what traded over the last {v['lookback']:,} decisions that was bought"
    return (f"Buys the books with the buying in the order flow, read as {reading}, neutral to {_neutral(v)}: "
            f"{_held(v)}.")


TRADER_KINDS: dict[str, TraderKind] = {kind.key: kind for kind in (
    TraderKind(
        "pairs", "Pairs",
        "Many pairs at once, each a z-score of its gap. It chooses them from the past mids: by how close they stayed "
        "(Gatev et al. 2006) or by Engle-Granger. In these books this is the one to beat: small orders in many pairs "
        "cost little, and a pair need not be exactly cointegrated to pay. Going a quarter of the way to its target at "
        "each decision (a trade rate of 0.25) with an entry of 1 earned +8,639 in notebook 13's sectors market, "
        "against +7,808 at an entry of 1.25 alone and +5,706 at these defaults.",
        (Param("count", "Pairs", 8, 1, 12, 1, integer=True), Switch("by_pvalue", "Choose by p-value", False), *Z_PARAMS,
         Param(*LOTS[:2], 2.5, *LOTS[3:]), *LAST),
        ("count", "by_pvalue", "formation", "window", "entry", "exit", "refit", *TRADING), _make_pairs,
        _describe_pairs,
        {"count": (4, 8, 12), "formation": (1000, 2000, 4000), "window": (200, 400, 800), "entry": (1.0, 1.5, 2.0),
         "exit": (0.0, 0.5), **COST_GRID}, ("window", "entry")),
    TraderKind(
        "baskets", "Baskets",
        "Relations of more than two books, from the Johansen test: on all the books, or on a few at a time. It leaves "
        "the factors out exactly, but each leg costs, and its swings grow only with the square root of the legs. On "
        "all the books the test needs a long past (10,000 units of time), or it finds few. Chosen by Box-Tiao "
        "predictability instead (d'Aspremont 2011), they are the portfolios of a few books whose past predicts least "
        "of them, grown greedily from every pair, with no test of whether any is a relation. In notebook 13's "
        "section 6, at the same settings (an entry of 1, a trade rate of 0.25), Box-Tiao earned about what Johansen "
        "did (Box-Tiao less Johansen, paired by seed, in the sectors and mixed markets: +156, se 180, and +367, se "
        "226, for baskets of 3 books; +172, se 230, and -44, se 237, for baskets of 4); it chose more than ten times "
        "as fast (0.01 seconds against 0.23 for 3 books in the sectors market, on a laptop), and its baskets left "
        "less exposure to the factors in the mixed market (0.055 against 0.087) and more in the sectors market "
        "(0.022 against 0.012). Going a quarter of the way at an entry of 1 (the two changed together) earned +2,044 "
        "(se 132) and +2,012 (se 176) more than these defaults, on more than two and a half times the lots.",
        (Choice("method", "Chosen by", "johansen", ("johansen", "box-tiao")),
         Param("legs", "Books each (0: all)", 3, 0, 4, 1, integer=True),
         Param("count", "Baskets", 4, 1, 8, 1, integer=True), *Z_PARAMS, Param(*LOTS), *LAST),
        ("method", "legs", "count", "formation", "window", "entry", "exit", "refit", *TRADING), _make_baskets,
        _describe_baskets,
        {"legs": (0, 3, 4), "count": (2, 4, 8), "formation": (1000, 2000, 4000), "window": (200, 400, 800),
         "entry": (1.0, 1.5, 2.0), "exit": (0.0, 0.5), **COST_GRID}, ("window", "entry")),
    TraderKind(
        "residuals", "Residuals",
        "Stat-arb after Avellaneda and Lee (2010): each book against PCA factors, its residuals fitted as an "
        "Ornstein-Uhlenbeck process, trading its s-score. Their levels (1.25, closing at 0.5 and 0.75) over a short "
        "window lose heavily here, where a round trip costs some 17 times what theirs did against its swings. Too few "
        "factors leaves a factor in, which is far worse than too many. In notebook 13's sectors market a no-trade band "
        "of 1 lot with an entry of 1 (+6,768), or a trade rate of 0.1 with an entry of 0.5 (+7,223), earned more than "
        "its best entry alone (+4,625, at 1.25).",
        (Param("factors", "Factors", 3, 1, 8, 1, integer=True),
         Param("window", "Fit over", 400, 10, 1000, 1, integer=True),
         Param("pca_window", "PCA over", 500, 10, 2000, 1, integer=True),
         Param("open_at", "Enter beyond s", 1.5, 0.1, 10.0, 0.05),
         Param("close_long", "Close a long inside", 0.0, 0.0, 9.9, 0.05),
         Param("close_short", "Close a short inside", 0.0, 0.0, 9.9, 0.05), Param(*LOTS), *LAST),
        ("factors", "window", "pca_window", "open_at", "close_long", "close_short", *TRADING), ResidualTrader,
        _describe_residuals,
        {"factors": (2, 3, 4), "window": (240, 400, 500), "pca_window": (500, 1000), "open_at": (1.0, 1.5, 2.0),
         "close_long": (0.0, 0.5), "close_short": (0.0, 0.75), **COST_GRID}, ("window", "open_at")),
    TraderKind(
        "copula", "Copula pairs",
        "Many pairs through copulas (Rad, Low and Faff 2016): chosen by distance, as they do, each trading when its "
        "flags part. It trades less than the z-score pairs on the same books, and earns less.",
        (Param("pairs", "Pairs", 8, 1, 12, 1, integer=True),
         Param("window", "Fit on", 2000, 50, 4000, 50, integer=True),
         Param("trading", "Trade for", 200, 10, 2000, 10, integer=True),
         Param("entry", "Flags beyond", 1.0, 0.1, 20.0, 0.1),
         Param("exit", "Flat within", 0.0, 0.0, 19.9, 0.1), Param(*LOTS[:2], 2.5, *LOTS[3:]), *LAST),
        ("pairs", "window", "trading", "entry", "exit", *TRADING), CopulaPortfolio, _describe_copula,
        {"pairs": (4, 8, 12), "window": (1000, 2000, 4000), "trading": (100, 200, 400), "entry": (0.5, 1.0, 2.0),
         "exit": (0.0, 0.5), **COST_GRID}, ("entry", "trading")),
    TraderKind(
        "reversal", "Reversal",
        "Buys the books that fell over a look back and sells those that rose (Lehmann 1990; Lo and MacKinlay 1990), "
        "all of them at once, neutral to the dollar and to PCA factors. In these books the sign depends on the "
        "horizon: it reverses over 1 and over 100 units of time and continues over 25. Traded with market orders, "
        "going the whole way at every decision loses, and trading less pays. Its default is the best of notebook 13's "
        "section 5: a look back of 50 decisions, rebalanced at every decision and going a tenth of the way to the "
        "target each time (a trade rate of 0.1), whose orders are of one lot. In the sectors market that made +4,364, "
        "against +3,485 held for 25 decisions instead, and +968 for a look back of 20 held for 100.",
        (Param("lookback", "Look back", 50, 1, 1000, 1, integer=True), replace(HOLD, default=1), *NEUTRAL,
         Param(*LOTS), replace(COSTS[0], default=0.1), COSTS[1], INTERVAL, WARM),
        ("lookback", "hold", "factors", "pca_window", *TRADING), ReversalTrader, _describe_reversal,
        {"lookback": (20, 50, 100), "hold": (1, 25, 100), "factors": (0, 3, 4), **SCORE_COST_GRID},
        ("lookback", "rate")),
    TraderKind(
        "momentum", "Momentum",
        "Buys the books that rose over a look back that ended some decisions ago, and sells those that fell (Jegadeesh "
        "and Titman 1993), neutral to the dollar and to PCA factors. It is here to show how little there is to find in "
        "these books: its score is reversal's turned over, so it is right where reversal is wrong (over 25 units of "
        "time), but earns only a tenth to a fifth of its costs there, and at its look back of 100 it loses before "
        "costs too.",
        (Param("lookback", "Look back", 100, 1, 1000, 1, integer=True),
         Param("skip", "Skip the last", 0, 0, 500, 1, integer=True), HOLD, *NEUTRAL, Param(*LOTS), *LAST),
        ("lookback", "skip", "hold", "factors", "pca_window", *TRADING), MomentumTrader, _describe_momentum,
        {"lookback": (20, 100, 400), "skip": (0, 5), "hold": (20, 100, 400), **SCORE_COST_GRID}, ("lookback", "hold")),
    TraderKind(
        "leadlag", "Lead-lag",
        "Buys the books that the other books' last moves say will rise, from a ridge regression of each book's move "
        "on all the books' moves before it (Lo and MacKinlay 1990; Hou 2007). It knows something only at an interval "
        "of 1 unit of time, a correlation of 0.05 to 0.07 with the next move; held for 20 decisions, as it is by "
        "default, it earns roughly a tenth of its costs.",
        (replace(HOLD, default=20), Param("ridge", "Ridge penalty", 1.0, 0.0, 100.0, 0.1), NEUTRAL[0],
         replace(NEUTRAL[1], label="Fit over"), Param(*LOTS), *COSTS, replace(INTERVAL, default=1.0), WARM),
        ("hold", "ridge", "factors", "pca_window", *TRADING), partial(LeadLagTrader, lookback=1),
        _describe_leadlag, {"hold": (5, 20, 100), "ridge": (0.1, 1.0, 10.0), "interval": (1.0, 5.0), **SCORE_COST_GRID},
        ("hold", "interval")),
    TraderKind(
        "flow", "Order flow",
        "Buys the books with the buying in their order flow: the lots queued at the best bid against the best ask, or "
        "the share of what traded that was bought. The queues know the most (a correlation of about 0.22 with the "
        "next move at an interval of 1 unit of time) and trades some (0.05 to 0.09), but what is known most is known "
        "shortest, and the turnover costs more than it earns. It sees the flow only from when it starts.",
        (Choice("measure", "Reads", "queues", ("queues", "trades")),
         Param("lookback", "Trades over", 5, 1, 1000, 1, integer=True), replace(HOLD, default=25), *NEUTRAL,
         Param(*LOTS), *LAST),
        ("measure", "lookback", "hold", "factors", "pca_window", *TRADING), FlowTrader, _describe_flow,
        {"lookback": (1, 5, 25), "hold": (5, 25, 100), "interval": (1.0, 5.0), **SCORE_COST_GRID},
        ("hold", "interval")),
)}


def too_few_books(kind: str, values: dict[str, Any], books: int) -> Optional[str]:
    """Why a trader of this kind with these settings cannot trade in this many books, if it cannot."""
    if kind == "residuals" and values["factors"] >= books:
        return f"fewer factors than the {books} books"
    if kind == "baskets" and values["legs"] > books:
        return f"baskets of at most the {books} books"
    if kind in CROSS_SECTIONAL and values["factors"] >= books - 1:
        return f"fewer factors than the {books} books less one"
    return None


def parse_market(form: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """The market's settings from the form, and a message for each that is not valid."""
    values, errors = parse(MARKET_PARAMS, form)
    shape = str(form.get("shape", "sectors"))
    if shape not in SHAPES:
        errors["shape"] = f"No such kind of market: {shape}."
    values["shape"] = shape
    if not errors and shape == "mixed" and values["factors"] >= values["books"]:
        errors["factors"] = "Fewer factors than books, or there is no relation among them."
    return values, errors


@dataclass
class StatArbRunner:
    """A trader on the board: its kind and settings, and whether it is deciding."""
    id: int
    name: str
    values: dict[str, Any]
    kind: str
    active: bool = True
    trader: BookTrader = field(init=False)
    trades_from: float = 0.0  # the time it has a signal from: what is left of its warm-up, from when it started
    fit: Optional[dict[str, Any]] = None  # how it did in training, if it came from a fit (TrainedMarket.start_fitted)

    def __post_init__(self) -> None:
        self.fresh()

    def fresh(self) -> None:
        """A new trader with the same settings (flat, nothing seen yet)."""
        self.trader = TRADER_KINDS[self.kind].build(self.values)
        self.trader.active = self.active
        if isinstance(self.trader, (RelationTrader, CopulaPortfolio)):
            self.trader.defer = _choose_elsewhere
        self.trading_from(0.0)

    def trading_from(self, now: float) -> None:
        """Note when, from ``now``, what it has yet to see will have taken its time."""
        self.trades_from = trades_from(self.trader, now)

    @property
    def color(self) -> str:
        return COLORS[(self.id - 1) % len(COLORS)]

    def fit_note(self, pnl: float, now: float, settings: dict[str, Any]) -> str:
        """How it did in training, a unit of time, beside how it has done here since it had a signal."""
        return fit_note(self.fit, pnl, now, self.trades_from, settings)


_chooser = ThreadPoolExecutor(1, thread_name_prefix="statarb-choose")


def _choose_elsewhere(choose, history):
    """Choosing relations, or fitting copulas to pairs, takes seconds with many books: the market runs on meanwhile."""
    return _chooser.submit(choose, history)


def relation_text(row: np.ndarray, names: list[str]) -> str:
    """A relation in lots of each book, for one of its largest leg: ``A1 - 0.98 A2``."""
    legs = [(names[i], float(row[i])) for i in np.nonzero(row)[0]]
    legs.sort(key=lambda leg: -abs(leg[1]))
    legs = [leg for leg in legs if abs(leg[1]) >= 0.005]
    out = ""
    for k, (name, w) in enumerate(legs[:6]):
        sign = ("-" if w < 0 else "") if k == 0 else (" - " if w < 0 else " + ")
        out += f"{sign}{'' if abs(abs(w) - 1) < 0.005 else f'{abs(w):.2f} '}{name}"
    return out + (f" and {len(legs) - 6} more" if len(legs) > 6 else "")


class StatArbSession(TrainedMarket):
    training = staticmethod(training_run)

    def __init__(self) -> None:
        super().__init__(SPEED_DEFAULT)
        self.settings: dict[str, Any] = {p.name: p.default for p in MARKET_PARAMS} | {"shape": "sectors"}
        self.runners: list[StatArbRunner] = []
        self._seq = 0
        self.mark_seq = 0  # the chart has no marks, with so many relations: for the stream's sake
        self.test: Optional[dict[str, Any]] = None  # what the tests last found
        self.reset()

    # ---- The market ----

    def reset(self) -> None:
        """A fresh market from the settings; each trader on the board starts over (flat, as it was set)."""
        for r in self.runners:
            r.fresh()
        self.names = book_names(self.settings)
        self.market = market_from(self.settings, self.settings["seed"], [r.trader for r in self.runners])
        self.samples: deque[dict[str, Any]] = deque(maxlen=SAMPLES)
        self.mids: deque[np.ndarray] = deque(maxlen=MIDS_KEPT)
        self._next_sample = 0
        self.test = None
        self._sample()
        self.version += 1
        self.changes += 1

    def configure(self, values: dict[str, Any]) -> None:
        """New settings: the market starts over. Raises ValueError if a trader on the board could not trade in it."""
        books = len(book_names(values))
        for r in self.runners:
            if r.kind == "residuals" and r.values["factors"] >= books:
                raise ValueError(f"{r.name} takes {r.values['factors']} factors, and needs more books than that: "
                                 f"remove it, or keep more than {books} books")
            if r.kind in CROSS_SECTIONAL and r.values["factors"] >= books - 1:
                raise ValueError(f"{r.name} takes {r.values['factors']} factors, and needs at least "
                                 f"{r.values['factors'] + 2} books: remove it, or apply that many or more")
        self.settings = dict(values)
        self.fit_changes += 1  # the last fit's panel says whether it was on these settings
        self.reset()

    def plan(self, kind: str, base: dict[str, Any], grid: dict[str, list[Any]], span: int, runs: int,
             seed: int) -> tuple[list[dict[str, Any]], Optional[int], float, bool]:
        """What a fit of a trader of this kind will run in these books (training.plan_fit): within MAX_FIT_BOOK_UNITS,
        and none that cannot trade in this many books. Raises ValueError, saying why, when the user's own settings
        cannot."""
        spec, books = TRADER_KINDS[kind], self.market.n

        def refusal(values: dict[str, Any]) -> Optional[str]:
            try:
                spec.build(values)
            except ValueError as exc:
                return str(exc)
            return too_few_books(kind, values, books)

        reason = refusal(base)
        if reason is not None:
            raise ValueError(f"your settings need {reason}")
        return plan_fit(spec, base, grid, span, runs, seed, MAX_FIT_BOOK_UNITS // books,
                        lambda values: refusal(values) is None)

    def _event(self) -> None:
        m = self.market
        m.step()
        while m.time >= self._next_sample:
            self._sample()

    def _sample(self) -> None:
        """The chart's point at the current sample time (just past it: taken after the event that crossed it)."""
        m = self.market
        marks = m.marks()
        self.mids.append(marks)
        loadings = m.fair.loadings
        pnl, signals, gross, exposure = {}, {}, {}, {}
        for r in self.runners:
            t = r.trader
            pnl[r.id] = _r(t.value(marks), 2)
            gross[r.id] = int(np.abs(t.position).sum())
            exposure[r.id] = _r(float(np.linalg.norm(t.position @ loadings)), 2)
            signal = t.signal()
            if signal is not None:
                signals[r.id] = [None if np.isnan(x) else _r(float(x)) for x in np.asarray(signal)[:MAX_SIGNALS]]
        self.samples.append({"t": self._next_sample, "mids": [_r(float(x), 1) for x in marks], "pnl": pnl,
                             "sig": signals, "gross": gross, "exposure": exposure})
        self._next_sample += 1

    # ---- Traders ----

    def start_trader(self, values: dict[str, Any], kind: str) -> tuple[StatArbRunner, int]:
        """Put a trader in the market, deciding from now on; with ``warm``, it first takes in the mids kept (one every
        interval). Returns it and how many it took. Raises ValueError (settings that cannot work, for this market or
        at all) or RuntimeError (the board is full).
        """
        if len(self.runners) >= MAX_TRADERS:
            raise RuntimeError(f"At most {MAX_TRADERS} traders: remove one first.")
        number = self._seq + 1
        runner = StatArbRunner(number, f"{TRADER_KINDS[kind].label} {number}", dict(values), kind)
        trader = runner.trader
        trader.attach(self.market.n)  # may refuse these settings for this many books
        reason = too_few_books(kind, values, self.market.n)
        if reason is not None:
            raise ValueError(reason)
        self._seq = number  # a trader that was refused takes no number
        learned = 0
        if values.get("warm"):
            stride = max(1, round(trader.interval))
            learned = trader.learn(list(self.mids)[::-1][::stride][::-1])
        trader.next_time = self.market.time
        runner.trading_from(self.market.time)
        self.runners.append(runner)
        self.market.traders.append(trader)
        self.changes += 1
        return runner, learned

    def runner(self, runner_id: int) -> Optional[StatArbRunner]:
        return next((r for r in self.runners if r.id == runner_id), None)

    def set_active(self, runner: StatArbRunner, active: bool) -> None:
        """Stopped, a trader keeps what it holds and decides no more; resumed, it decides again from now."""
        runner.active = runner.trader.active = active
        if active:
            runner.trader.next_time = self.market.time
        self.changes += 1

    def remove(self, runner: StatArbRunner) -> None:
        """Off the board, with what it holds (nothing is closed out: its orders would move the books)."""
        self.runners.remove(runner)
        # By identity: traders started alike are equal as dataclasses, and comparing their positions raises.
        self.market.traders[:] = [t for t in self.market.traders if t is not runner.trader]
        self.changes += 1

    # ---- What the tests see ----

    async def test_relations(self, units: int = 10_000) -> dict[str, Any]:
        """The Johansen test and Engle-Granger on the last ``units`` of mids, off the event loop."""
        mids = np.array(self.mids)[-units:]
        loadings = np.array(self.market.fair.mean_loadings)
        result = await asyncio.to_thread(what_the_tests_see, mids, loadings)
        self.test = result
        self.changes += 1
        self.notify()
        return result

    # ---- What the page shows ----

    def relations_of(self, t: BookTrader) -> list[dict[str, Any]]:
        """What it trades, one a relation, with what it holds of each and its signal: largest signals first."""
        weights = getattr(t, "weights", None)
        states = getattr(t, "states", None)
        if weights is None or states is None:
            return []
        signal = t.signal()
        rows = []
        for i, row in enumerate(weights):
            if not row.any():  # a residual trader's book that has not opened yet
                continue
            value = None if signal is None or i >= len(signal) or np.isnan(signal[i]) else float(signal[i])
            rows.append({"text": relation_text(row, self.names), "state": int(states[i]), "signal": value})
        return rows

    def board(self) -> list[dict[str, Any]]:
        """The traders' rows, best P&L first."""
        m = self.market
        marks = m.marks()
        rows = []
        for r in self.runners:
            t = r.trader
            pnl = t.value(marks)
            relations = self.relations_of(t)
            rows.append({"id": r.id, "name": r.name, "kind": r.kind, "color": r.color,
                         "describe": TRADER_KINDS[r.kind].describe(r.values), "active": r.active, "pnl": pnl,
                         "before": pnl + t.spread_paid, "cost": t.spread_paid, "lots": t.lots, "entries": t.entries,
                         "cost_a_lot": t.spread_paid / t.lots if t.lots else None,
                         "gross": int(np.abs(t.position).sum()),
                         "exposure": float(np.linalg.norm(t.position @ m.fair.loadings)), "warming": t.warming,
                         "netted": t.netted_lots, "relations": relations[:MAX_SIGNALS], "total": len(relations),
                         "holding": sum(1 for x in relations if x["state"]),
                         "fit": r.fit_note(pnl, m.time, self.settings)})
        rows.sort(key=lambda row: (-row["pnl"], row["id"]))
        return rows

    def book_rows(self) -> list[dict[str, Any]]:
        m = self.market
        rows = []
        for k, book in enumerate(m.books):
            rows.append({"name": self.names[k], "mid": book.mid_price, "spread": book.spread, "fair": m.fair.fairs[k],
                         "held": [{"id": r.id, "color": r.color, "lots": int(r.trader.position[k])}
                                  for r in self.runners]})
        return rows

    def context(self) -> dict[str, Any]:
        """Everything the pushed panels show, as plain data (rendered off the event loop)."""
        m = self.market
        return {"time": m.time, "events": m.events, "running": self.is_running, "speed": self.speed,
                "books": self.book_rows(), "rows": self.board(), "settings": dict(self.settings),
                "n": m.n, "factors": int(np.linalg.matrix_rank(m.fair.mean_loadings)), "relations": m.fair.rank,
                "floor": m.fair.floor, "pushed": m.fair.pushed, "half_life": m.fair.half_life,
                "drifting": bool(m.fair.loading_sd), "full": len(self.runners) >= MAX_TRADERS, "test": self.test,
                "kept": len(self.mids), "fitting": self.fitting}

    def chart_traders(self) -> list[dict[str, Any]]:
        """What the chart needs to know of each trader, in the order they started."""
        out = []
        for r in self.runners:
            entry, exit_ = r.trader.levels
            out.append({"id": r.id, "name": r.name, "color": r.color, "active": r.active, "kind": r.kind,
                        "entry": entry, "exit": exit_})
        return out

    def history(self) -> dict[str, Any]:
        """The chart's data so far: every sample kept, and the traders."""
        return {"version": self.version, "samples": list(self.samples), "traders": self.chart_traders(), "marks": [],
                "mark_seq": 0, "settings": dict(self.settings), "speed": self.speed, "running": self.is_running,
                "names": self.names, "groups": self.groups()}

    def groups(self) -> list[int]:
        """Which sector each book is in, for its color on the chart (all one, without sectors)."""
        if self.settings["shape"] == "sectors":
            return [i for i in range(self.settings["sectors"]) for _ in range(self.settings["books_each"])]
        return [0] * self.settings["books"]

    def samples_since(self, t: Optional[int]) -> list[dict[str, Any]]:
        out = []
        for s in reversed(self.samples):
            if t is not None and s["t"] <= t:
                break
            out.append(s)
        return out[::-1]

    def marks_since(self, seq: int) -> list:
        return []


def what_the_tests_see(mids: np.ndarray, loadings: np.ndarray, every: int = 5) -> dict[str, Any]:
    """How many relations the trace test finds in these mids (one a unit of time), and how many of the pairs
    Engle-Granger calls cointegrated, against the truth from the loadings."""
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    n = loadings.shape[0]
    thinned = mids[::every]
    enough = len(mids) >= MIN_TEST_UNITS
    pairs = list(itertools.combinations(range(n), 2))
    truly = sum(1 for x, y in pairs if np.linalg.matrix_rank(loadings[[x, y]], tol=1e-9) == 1)
    if not enough:
        return {"units": len(mids), "enough": False, "needed": MIN_TEST_UNITS, "pairs": len(pairs), "truly": truly,
                "true_rank": n - int(np.linalg.matrix_rank(loadings))}
    rank = trace_rank(coint_johansen(thinned, det_order=0, k_ar_diff=2))
    called = sum(1 for x, y in pairs if coint(thinned[:, x], thinned[:, y], maxlag=5, autolag=None)[1] < 0.05)
    return {"units": len(mids), "enough": True, "rank": rank, "true_rank": n - int(np.linalg.matrix_rank(loadings)),
            "pairs": len(pairs), "called": called, "truly": truly}
