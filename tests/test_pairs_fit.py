"""Fitting a pairs trader over training runs (PairsSession.fit_trader in app/pairs.py, and its routes)."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.testclient import TestClient

from app import pairs as pairs_module
from app.fitting import GridError, MAX_FIT_CANDIDATES, parse_grid
from app.main import app, registry, render_pairs_fit
from app.pairs import (CLEARLY, FIT_RUNS, FIT_SPAN, MARKET_PARAMS, MAX_FIT_UNITS, MAX_TRADERS, TRADER_KINDS,
                       FitCanceled, PairsFit, PairsSession, fit_plan, market_from, rank_candidates, same_market,
                       training_run, training_seed)
from app.sessions import COOKIE_NAME
from lob.pairs import A, B, PairsTrader, score_trader

DEFAULTS = {kind: {p.name: p.default for p in spec.params} for kind, spec in TRADER_KINDS.items()}
MARKET = {p.name: p.default for p in MARKET_PARAMS}
QUICK = {**DEFAULTS["zscore"], "window": 20, "entry": 2.0}
GRID = {"window": [10, 20], "entry": [1.5, 2.0]}


@pytest.fixture(autouse=True)
def threads_for_workers(monkeypatch):
    """Training runs on threads, which share the test's process, so these tests need not spawn; one spawns anyway."""
    monkeypatch.setattr(pairs_module, "_fit_pool", lambda: ThreadPoolExecutor(4))


@pytest.fixture
def client():
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def pairs(client):
    return registry.get(client.cookies[COOKIE_NAME]).pairs


def fit_form(**more):
    return {"kind": "zscore", **QUICK, "fit_window": "on", "grid_window": "10, 20", "fit_entry": "on",
            "grid_entry": "1.5, 2", "span": 500, "runs": 2, "objective": "pnl", "fit_seed": 1, **more}


def run(*rows):
    """Training results from (pnl, lots) or just pnl, with the other figures made up from them."""
    out = []
    for row in rows:
        pnl, lots = row if isinstance(row, tuple) else (row, 10.0)
        out.append({"pnl": pnl, "before_costs": pnl + 1.0, "costs": 1.0, "lots": lots, "entries": 2.0,
                    "risk_adjusted": pnl / 1000})
    return out


# ---- What a training run is ----

def test_a_training_seed_is_never_one_the_live_market_can_have():
    seed = next(p for p in MARKET_PARAMS if p.name == "seed")
    assert training_seed(0, 0) > seed.high
    seen = {training_seed(s, r) for s in (0, 1, 2, seed.high) for r in range(10)}
    assert len(seen) == 40  # and no two runs share one


def test_the_market_from_the_settings_is_the_sessions_own():
    session = PairsSession()
    session.run_for(50)
    market = market_from(session.settings, session.settings["seed"])
    while market.time < 50:
        market.step()
    assert session.market.events == market.events and session.market.fair.gap() == market.fair.gap()


def test_a_training_run_is_the_trader_alone_in_a_fresh_market():
    trader = PairsTrader(window=20, entry=2.0)
    by_hand = score_trader(market_from(MARKET, 7, (trader,)), trader, 21.0, 200.0)
    assert training_run("zscore", QUICK, MARKET, 7, 21.0, 200.0) == by_hand
    assert training_run("zscore", QUICK, MARKET, 8, 21.0, 200.0) != by_hand
    assert by_hand["pnl"] == pytest.approx(by_hand["before_costs"] - by_hand["costs"])


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_every_kind_of_trader_trains(kind):
    values = {**DEFAULTS[kind], "window": 30}
    trader = TRADER_KINDS[kind].build(values)
    result = training_run(kind, values, MARKET, 3, trader.warm_up, 100.0)
    assert set(result) == {"pnl", "before_costs", "costs", "lots", "entries", "risk_adjusted"}


# ---- What a fit runs ----

def test_a_plan_tries_the_grid_and_the_users_settings_once():
    tried, valid, burn, sampled = fit_plan("zscore", QUICK, GRID, 500, 2, 1)
    assert len(tried) == 4 and valid == 4 and QUICK in tried and not sampled  # theirs is in the grid
    assert burn == 21.0
    theirs = {**QUICK, "window": 30}
    tried, valid, burn, sampled = fit_plan("zscore", theirs, GRID, 500, 2, 1)
    assert len(tried) == 5 and tried[0] == theirs and valid == 4 and not sampled  # theirs is not: first, counted apart
    assert burn == 31.0  # their warm-up counts too: all get the same


def test_a_plan_gives_every_run_the_longest_warm_up():
    tried, _, burn, _ = fit_plan("zscore", QUICK, {"window": [10, 100], "interval": [1.0, 5.0]}, 500, 1, 1)
    assert burn == max(TRADER_KINDS["zscore"].build(v).warm_up for v in tried) == 101 * 5.0


def test_a_plan_leaves_out_contradictions():
    tried, valid, _, _ = fit_plan("zscore", QUICK, {"entry": [1.0, 2.0], "exit": [0.0, 1.5]}, 500, 1, 1)
    assert valid == 3 and all(v["exit"] < v["entry"] for v in tried)
    with pytest.raises(ValueError, match="none of these"):
        fit_plan("zscore", QUICK, {"entry": [1.0], "exit": [1.5]}, 500, 1, 1)


def test_a_plan_stays_within_the_budget_by_drawing_a_sample():
    grid = {"window": [10, 20, 30, 40, 50, 60, 70, 80], "entry": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}
    tried, valid, burn, sampled = fit_plan("zscore", QUICK, grid, 500, 2, 1)
    assert valid == 64 and sampled and len(tried) <= MAX_FIT_CANDIDATES and QUICK in tried  # never sampled out
    tried, valid, burn, sampled = fit_plan("zscore", QUICK, grid, 20_000, 5, 1)
    assert sampled and len(tried) * 5 * (burn + 20_000) <= MAX_FIT_UNITS and len(tried) >= 2
    assert QUICK in tried
    again = fit_plan("zscore", QUICK, grid, 20_000, 5, 1)[0]
    other = fit_plan("zscore", QUICK, grid, 20_000, 5, 2)[0]
    assert again == tried and other != tried  # drawn from the seed


def test_a_plan_says_when_the_runs_are_too_long_to_compare_anything():
    with pytest.raises(ValueError, match="too long"):
        fit_plan("zscore", QUICK, {"window": [5_000], "interval": [50.0]}, 50_000, 10, 1)


# ---- How the candidates are compared ----

def test_each_candidate_is_compared_with_the_users_settings_seed_by_seed():
    a, yours, b = {"id": "a"}, {"id": "yours"}, {"id": "b"}
    rows = rank_candidates([a, yours, b], [run(30.0, 110.0, 70.0), run(10.0, 90.0, 50.0), run(90.0, 10.0, 20.0)],
                           yours, "pnl")
    assert [r["values"] for r in rows] == [a, yours, b]  # by their means: 70, 50, 40
    best, theirs, last = rows
    assert theirs["yours"] and not best["yours"] and theirs["against"] == 0 and theirs["against_se"] == 0
    # a's differences are 20, 20, 20: what they share cancels, as a's mean and se across the runs do not.
    assert best["against"] == pytest.approx(20) and best["against_se"] == pytest.approx(0)
    assert best["pnl"] == pytest.approx(70) and best["pnl_se"] == pytest.approx(40 / 3**.5)
    # b's are 80, -80, -30.
    assert last["against"] == pytest.approx(-10)
    assert last["against_se"] == pytest.approx(((90**2 + 70**2 + 20**2) / 2 / 3) ** .5)


def test_ties_go_to_the_one_that_traded_less_and_one_run_has_no_standard_error():
    a, b = {"id": "a"}, {"id": "b"}
    rows = rank_candidates([a, b], [run((5.0, 30.0)), run((5.0, 10.0))], a, "pnl")
    assert rows[0]["values"] == b and rows[0]["against_se"] is None and rows[0]["pnl_se"] is None


def test_the_objective_decides_the_order_and_the_comparison():
    a, b = {"id": "a"}, {"id": "b"}
    results = [run(10.0, 10.0), run(20.0, 20.0)]
    results[0][0]["risk_adjusted"] = results[0][1]["risk_adjusted"] = 0.5
    rows = rank_candidates([a, b], results, a, "sharpe")
    assert rows[0]["values"] == a and rows[0]["score"] == 0.5 and rows[1]["against"] == pytest.approx(0.02 - 0.5)
    assert rows[1]["pnl"] == 20


def fit_of(rows):
    return PairsFit(id=1, kind="zscore", grid={"entry": [1.0]}, settings=MARKET, objective="pnl", seed=1, span=500,
                    burn=21.0, runs=3, candidates=rows, valid=2, size=2, sampled=False, seconds=1.0)


@pytest.mark.parametrize("best, theirs, verdict", [
    ((30.0, 110.0, 70.0), (10.0, 90.0, 50.0), "clearly"),  # always 20 better
    ((90.0, 10.0, 51.0), (10.0, 90.0, 50.0), "unclear"),  # better on average, by luck that shows
    ((10.0, 90.0, 50.0), (90.0, 10.0, 51.0), "yours"),
])
def test_the_verdict_says_when_the_best_is_not_clearly_better(best, theirs, verdict):
    a, yours = {"id": "a"}, {"id": "yours"}
    fit = fit_of(rank_candidates([a, yours], [run(*best), run(*theirs)], yours, "pnl"))
    assert fit.verdict == verdict
    assert fit.yours["values"] == yours


def test_a_margin_of_exactly_the_threshold_is_not_clearly_better():
    a, yours = {"id": "a"}, {"id": "yours"}
    rows = rank_candidates([a, yours], [run(0.0, 2.0, 4.0), run(0.0, 0.0, 0.0)], yours, "pnl")
    fit = fit_of(rows)
    assert rows[0]["against"] == pytest.approx(2.0)
    assert rows[0]["against"] / rows[0]["against_se"] == pytest.approx(3**.5)  # below it: not clearly better
    assert CLEARLY > 3**.5 and fit.verdict == "unclear"


def test_one_run_cannot_say():
    a, yours = {"id": "a"}, {"id": "yours"}
    assert fit_of(rank_candidates([a, yours], [run(20.0), run(10.0)], yours, "pnl")).verdict == "single"


# ---- The session's fit ----

def test_a_fit_ranks_every_candidate_and_leaves_the_live_market_alone():
    session = PairsSession()
    session.run_for(30)
    time, events, state = session.market.time, session.market.events, session.market.rng.bit_generator.state
    seen = []

    async def main():
        fit = asyncio.create_task(session.fit_trader("zscore", QUICK, GRID, 500, 2, "pnl", 1))
        while not fit.done():
            await asyncio.sleep(0)  # it gets to run first: from then on, until it ends, it says how far it has got
            if session.fitting:
                seen.append(session.fitting)
            await asyncio.sleep(0.01)
        return fit.result()

    fit = asyncio.run(main())
    assert len(fit.candidates) == 4 and session.last_fit is fit and session.fitting is None
    assert [c["score"] for c in fit.candidates] == sorted((c["score"] for c in fit.candidates), reverse=True)
    assert sum(c["yours"] for c in fit.candidates) == 1 and fit.burn == 21.0 and fit.span == 500
    assert seen[0] == (0, 8) and seen[-1][1] == 8 and all(a[0] <= b[0] for a, b in zip(seen, seen[1:], strict=False))
    assert session.market.time == time and session.market.events == events
    assert session.market.rng.bit_generator.state == state  # not one random number of the live market's was used


def test_a_fit_is_the_same_from_the_same_seed_and_other_from_another():
    def result(seed):
        fit = asyncio.run(PairsSession().fit_trader("zscore", QUICK, GRID, 500, 2, "pnl", seed))
        return [(c["values"]["window"], c["values"]["entry"], c["pnl"]) for c in fit.candidates]

    assert result(1) == result(1) and result(1) != result(2)


def test_a_candidate_is_measured_in_the_markets_a_run_by_hand_would_give(monkeypatch):
    fit = asyncio.run(PairsSession().fit_trader("zscore", QUICK, GRID, 500, 2, "pnl", 5))
    yours = fit.yours
    by_hand = [training_run("zscore", QUICK, MARKET, training_seed(5, r), fit.burn, fit.span)["pnl"] for r in range(2)]
    assert yours["pnl"] == pytest.approx(sum(by_hand) / 2)


def test_the_training_markets_follow_the_views_settings():
    session = PairsSession()
    session.configure({**MARKET, "beta": 2.0, "half_life": 10.0, "seed": 9})
    fit = asyncio.run(session.fit_trader("zscore", QUICK, GRID, 500, 1, "pnl", 1))
    by_hand = training_run("zscore", QUICK, session.settings, training_seed(1, 0), fit.burn, fit.span)["pnl"]
    assert fit.yours["pnl"] == pytest.approx(by_hand) and fit.settings == session.settings
    assert by_hand != training_run("zscore", QUICK, MARKET, training_seed(1, 0), fit.burn, fit.span)["pnl"]


def test_only_one_fit_runs_at_a_time():
    async def main():
        first = asyncio.create_task(PairsSession().fit_trader("zscore", QUICK, GRID, 500, 2, "pnl", 1))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="another fit"):
            await PairsSession().fit_trader("zscore", QUICK, GRID, 500, 2, "pnl", 1)  # not even from another session
        await first

    asyncio.run(main())
    asyncio.run(PairsSession().fit_trader("zscore", QUICK, {"entry": [2.0]}, 500, 1, "pnl", 1))  # free again


def test_canceling_keeps_nothing_and_leaves_room_for_the_next():
    session = PairsSession()
    assert session.cancel_fit() is False

    async def main():
        fit = asyncio.create_task(session.fit_trader("zscore", QUICK, GRID, 5_000, 4, "pnl", 1))
        await asyncio.sleep(0.05)
        assert session.cancel_fit() is True
        with pytest.raises(FitCanceled):
            await fit

    asyncio.run(main())
    assert session.last_fit is None and session.fitting is None and session.fit_changes == 0
    assert asyncio.run(session.fit_trader("zscore", QUICK, {"entry": [2.0]}, 500, 1, "pnl", 1)).id == 1


def test_a_worker_that_dies_says_so_and_frees_the_slot(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    class Broken(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            def boom():
                raise BrokenProcessPool("a worker died")
            return super().submit(boom)

    monkeypatch.setattr(pairs_module, "_fit_pool", lambda: Broken(1))
    session = PairsSession()
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        asyncio.run(session.fit_trader("zscore", QUICK, GRID, 500, 1, "pnl", 1))
    assert session.fitting is None and session.last_fit is None
    monkeypatch.setattr(pairs_module, "_fit_pool", lambda: ThreadPoolExecutor(2))
    asyncio.run(session.fit_trader("zscore", QUICK, GRID, 500, 1, "pnl", 1))


def test_evicting_the_session_stops_its_fit():
    session = PairsSession()

    async def main():
        fit = asyncio.create_task(session.fit_trader("zscore", QUICK, GRID, 5_000, 4, "pnl", 1))
        await asyncio.sleep(0.05)
        session.shutdown()
        with pytest.raises(FitCanceled):
            await fit

    asyncio.run(main())


def test_the_training_runs_work_in_real_worker_processes(monkeypatch):
    monkeypatch.undo()  # the real pool: its function and arguments must cross to a spawned process, and back
    monkeypatch.setattr(pairs_module, "FIT_WORKERS", 2)
    fit = asyncio.run(PairsSession().fit_trader("copula", {**DEFAULTS["copula"], "window": 30, "interval": 1.0},
                                                {"entry": [1.0, 2.0]}, 500, 2, "pnl", 1))
    assert len(fit.candidates) == 2 and fit.burn == 31.0
    by_hand = training_run("copula", fit.yours["values"], MARKET, training_seed(1, 0), 31.0, 500)["pnl"]
    second = training_run("copula", fit.yours["values"], MARKET, training_seed(1, 1), 31.0, 500)["pnl"]
    assert fit.yours["pnl"] == pytest.approx((by_hand + second) / 2)


# ---- The routes ----

def test_the_fit_route_fits_and_the_stream_brings_the_candidates(client, pairs):
    reply = client.post("/api/pairs/traders/fit", data=fit_form())
    assert reply.status_code == 200 and reply.json()["verdict"] == pairs.last_fit.verdict
    assert "Tried 4 settings, 2 runs each" in reply.json()["message"]
    from tests.test_pairs_view import read_stream
    seen = read_stream(client, {"pairs_board"})  # the last of a first push: a stream without the fit fails, not hangs
    assert 'id="pairs-fit-summary"' in seen["pairs_fit"]


def test_nothing_is_pushed_before_the_first_fit(client, pairs):
    from tests.test_pairs_view import read_stream
    seen = read_stream(client, {"pairs_board"})
    assert "pairs_fit" not in seen and render_pairs_fit(pairs) == ""


@pytest.mark.parametrize("change, name, message", [
    ({"grid_window": "10, nope"}, "grid_window", "enter a number"),
    ({"grid_window": ""}, "grid_window", "give the values"),
    ({"grid_window": "1, 2"}, "grid_window", "between"),
    ({"grid_entry": "1, 2, 3, 4, 5, 6, 7, 8, 9"}, "grid_entry", "at most"),
    ({"span": 100}, "span", "between"),
    ({"runs": 11}, "runs", "between"),
    ({"fit_seed": -1}, "fit_seed", "between"),
    ({"objective": "luck"}, "objective", "Best by"),
    ({"exit": 3.0, "entry": 2.0}, "exit", "flat level"),
    ({"window": 1}, "window", "between"),
    ({"kind": "tea_leaves"}, "kind", "No such kind"),
])
def test_what_is_wrong_with_a_fit_is_said_against_its_field(client, pairs, change, name, message):
    reply = client.post("/api/pairs/traders/fit", data=fit_form(**change))
    assert reply.status_code == 422 and message in reply.json()["errors"][name] and pairs.last_fit is None


def test_a_fit_with_nothing_ticked_says_so(client):
    form = fit_form()
    del form["fit_window"], form["fit_entry"]
    reply = client.post("/api/pairs/traders/fit", data=form)
    assert reply.status_code == 422 and "Tick at least one" in reply.json()["errors"]["fit"]


def test_combinations_that_cannot_work_are_refused(client):
    reply = client.post("/api/pairs/traders/fit", data=fit_form(fit_exit="on", grid_exit="5", grid_entry="1, 2"))
    assert reply.status_code == 422 and "none of these combinations" in reply.json()["error"]


def test_a_second_fit_at_once_is_refused(client, pairs, monkeypatch):
    monkeypatch.setattr(pairs_module, "_fit_running", True)
    reply = client.post("/api/pairs/traders/fit", data=fit_form())
    assert reply.status_code == 409 and "another fit" in reply.json()["error"]


def test_canceling_when_nothing_runs_says_so(client):
    assert client.post("/api/pairs/traders/fit/cancel").json() == {"canceled": False}


def test_a_candidate_starts_and_its_row_keeps_what_it_made_in_training(client, pairs):
    client.post("/api/pairs/traders/fit", data=fit_form())
    fit = pairs.last_fit
    reply = client.post("/api/pairs/traders/start", data={"fit_id": fit.id, "rank": 2})
    assert reply.status_code == 200 and reply.json()["name"] == "Z-score 1"
    runner = pairs.runners[0]
    assert runner.values == fit.candidates[1]["values"] and runner.fit["rank"] == 2 and runner.fit["of"] == 4
    note = pairs.board()[0][0]["fit"]
    assert "Fitted by P&L, 2nd of 4" in note
    assert f"{fit.candidates[1]['pnl'] / 500:+.3f} a unit in 2 training runs" in note
    assert "nothing yet here" in note  # the market has not moved since
    client.post("/api/pairs/control", data={"action": "step", "units": 100})
    note = pairs.board()[0][0]["fit"]
    assert f"{pairs.board()[0][0]['pnl'] / (pairs.market.time - runner.trades_from):+.3f} a unit here" in note
    client.post("/api/pairs/control", data={"action": "reset"})
    runner = pairs.runners[0]
    assert runner.fit["rank"] == 2 and runner.trades_from == runner.values["window"]  # it watches again, keeping that


def test_a_trader_started_by_hand_has_no_fit_note(client, pairs):
    client.post("/api/pairs/traders/start", data={"kind": "zscore", **QUICK})
    assert pairs.board()[0][0]["fit"] == ""


def test_the_time_a_trader_has_traded_counts_from_its_signal():
    session = PairsSession()
    session.run_for(5)
    # Without learning from the chart it has all of its window to watch; with it, what the chart lacks.
    runner, _ = session.start_trader({**QUICK, "warm": False, "interval": 2.0})
    assert runner.trades_from == pytest.approx(session.market.time + 20 * 2.0)
    runner, learned = session.start_trader({**QUICK, "warm": True, "interval": 2.0})
    assert learned > 0 and runner.trades_from == pytest.approx(session.market.time + (20 - learned) * 2.0)
    session.run_for(100)
    runner, _ = session.start_trader({**QUICK, "warm": True, "interval": 2.0})
    assert runner.trades_from == session.market.time


@pytest.mark.parametrize("data, code, message", [
    ({"fit_id": 1, "rank": 99}, 409, "no longer here"),
    ({"fit_id": 2, "rank": 1}, 409, "no longer here"),
    ({"fit_id": 1, "rank": 0}, 409, "no longer here"),
    ({"fit_id": "one", "rank": 1}, 422, "a fit and a rank"),
    ({"fit_id": 1}, 422, "a fit and a rank"),
])
def test_a_candidate_that_is_gone_is_refused(client, pairs, data, code, message):
    client.post("/api/pairs/traders/fit", data=fit_form())
    reply = client.post("/api/pairs/traders/start", data=data)
    assert reply.status_code == code and message in reply.json()["error"] and not pairs.runners


def test_a_full_board_refuses_a_candidate(client, pairs):
    client.post("/api/pairs/traders/fit", data=fit_form())
    for _ in range(MAX_TRADERS):
        client.post("/api/pairs/traders/start", data={"fit_id": 1, "rank": 1})
    reply = client.post("/api/pairs/traders/start", data={"fit_id": 1, "rank": 1})
    assert reply.status_code == 409 and "At most" in reply.json()["error"] and len(pairs.runners) == MAX_TRADERS


def test_the_market_on_the_chart_is_not_what_a_candidate_was_fitted_on(client, pairs):
    client.post("/api/pairs/traders/fit", data=fit_form())
    assert "other market settings" not in (client.post("/api/pairs/traders/start", data={"fit_id": 1, "rank": 1})
                                           and pairs.board()[0][0]["fit"])
    assert "changed since" not in render_pairs_fit(pairs)
    changes = pairs.fit_changes
    client.post("/api/pairs/config", data={**{k: v for k, v in MARKET.items() if v is not False and v is not True},
                                           "reverting": "on", "seed": 5})
    assert same_market(pairs.last_fit.settings, pairs.settings)  # other random numbers only: the same market
    assert "changed since" not in render_pairs_fit(pairs) and pairs.fit_changes == changes + 1
    client.post("/api/pairs/config", data={**{k: v for k, v in MARKET.items() if v is not False and v is not True},
                                           "reverting": "on", "half_life": 10})
    assert not same_market(pairs.last_fit.settings, pairs.settings)
    assert "changed since" in render_pairs_fit(pairs) and "other market settings" in pairs.board()[0][0]["fit"]


def test_the_status_says_a_fit_is_running(client, pairs):
    pairs.fitting = (3, 8)
    context = pairs.context()
    assert context["fitting"] == (3, 8)
    from tests.test_pairs_view import read_stream
    seen = read_stream(client, {"pairs_status", "pairs_state"})
    assert 'data-fitting="3/8"' in seen["pairs_status"] and json.loads(seen["pairs_state"])["fitting"] == [3, 8]


# ---- The panel ----

def rendered(pairs, rows, **more):
    pairs.last_fit = PairsFit(id=1, kind="zscore", grid={"window": [10, 20], "entry": [1.5, 2.0]},
                              settings=pairs.settings, objective=more.pop("objective", "pnl"), seed=1, span=500,
                              burn=21.0, runs=3, candidates=rows, valid=4, size=4,
                              sampled=more.pop("sampled", False), seconds=1.0)
    return render_pairs_fit(pairs)


def settings_of(window, entry):
    return {**QUICK, "window": window, "entry": entry}


def test_the_panel_says_plainly_what_the_fit_found(client, pairs):
    a, yours = settings_of(10, 1.5), QUICK
    html = rendered(pairs, rank_candidates([a, yours], [run(30.0, 110.0, 70.0), run(10.0, 90.0, 50.0)], yours, "pnl"))
    assert 'data-verdict="clearly"' in html
    assert "window 10, enter beyond z 1.5 beat your settings by +20.0" in html.lower()
    html = rendered(pairs, rank_candidates([a, yours], [run(90.0, 10.0, 51.0), run(10.0, 90.0, 50.0)], yours, "pnl"))
    assert 'data-verdict="unclear"' in html and "not clearly better" in html and "stay with yours" in html
    html = rendered(pairs, rank_candidates([a, yours], [run(10.0, 90.0, 50.0), run(90.0, 10.0, 51.0)], yours, "pnl"))
    assert 'data-verdict="yours"' in html and "Your own settings came out best" in html
    assert html.count("data-yours") == 1


def test_the_panel_shows_the_users_row_even_outside_the_best_ten(client, pairs):
    tried = [settings_of(10 + i, 1.5) for i in range(14)] + [QUICK]
    results = [run(float(100 - i), float(101 - i), float(102 - i)) for i in range(14)] + [run(0.0, 1.0, 2.0)]
    html = rendered(pairs, rank_candidates(tried, results, QUICK, "pnl"))
    assert html.count("data-rank=") == 11 and 'data-rank="15" data-yours' in html
    assert "The best 10 of the 15 tried, and yours." in html


def test_the_panel_can_be_read_by_risk_adjusted_score(client, pairs):
    a, yours = settings_of(10, 1.5), QUICK
    rows = rank_candidates([a, yours], [run(30.0, 110.0, 70.0), run(10.0, 90.0, 50.0)], yours, "sharpe")
    html = rendered(pairs, rows, objective="sharpe")
    assert "Score" in html and "+0.020" in html and "P&amp;L per unit of risk" in html


def test_the_panel_says_when_the_candidates_are_a_sample(client, pairs):
    a, yours = settings_of(10, 1.5), QUICK
    html = rendered(pairs, rank_candidates([a, yours], [run(1.0), run(2.0)], yours, "pnl"), sampled=True)
    assert "drawn at random from 4 (seed 1)" in html


def test_the_panel_has_a_row_of_settings_the_form_can_take(client, pairs):
    a, yours = settings_of(10, 1.5), QUICK
    html = rendered(pairs, rank_candidates([a, yours], [run(1.0), run(2.0)], yours, "pnl"))
    assert "data-pairs-fit-use=" in html and "&#34;window&#34;: 10" in html and "data-pairs-fit-start=\"2\"" in html


# ---- The form ----

def test_the_page_has_a_fit_panel_for_each_kind(client):
    page = client.get("/").text
    assert 'id="pairs-fit-panel"' in page and 'id="pairs-fit-result"' in page
    for kind, spec in TRADER_KINDS.items():
        assert f'data-pairs-fit-kind="{kind}"' in page
        assert spec.fit_default and set(spec.fit_default) <= {p.name for p in spec.numbers}
        assert set(spec.fit_grid) <= {p.name for p in spec.numbers}
        for param in spec.numbers:  # every value proposed is one the setting takes
            for value in spec.fit_grid.get(param.name, ()):
                param.parse(value)


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_each_kinds_proposed_values_are_tried_whole_at_the_default_length(kind):
    spec = TRADER_KINDS[kind]
    grid = {name: list(spec.fit_grid[name]) for name in spec.fit_default}
    span, runs = FIT_SPAN.default, FIT_RUNS.default
    for base in (DEFAULTS[kind], {**DEFAULTS[kind], "entry": 2.2}):  # the user's settings in the grid, and not
        tried, valid, burn, sampled = fit_plan(kind, base, grid, span, runs, 1)
        first, second = spec.fit_default
        assert not sampled and base in tried and valid == len(grid[first]) * len(grid[second])
        assert len(tried) * runs * (burn + span) <= MAX_FIT_UNITS


def test_a_grid_is_read_from_the_form_and_its_errors_name_the_setting():
    numbers = TRADER_KINDS["zscore"].numbers
    form = {"fit_window": "on", "grid_window": "10; 20, 20", "grid_entry": "9"}
    assert parse_grid(numbers, form) == {"window": [10, 20]}
    with pytest.raises(GridError) as caught:
        parse_grid(numbers, {"fit_entry": "on", "grid_entry": "x"})
    assert caught.value.name == "entry"
    with pytest.raises(GridError) as caught:
        parse_grid(numbers, {})
    assert caught.value.name == "" and "Tick" in f"{str(caught.value)[0].upper()}{str(caught.value)[1:]}"


def test_a_fit_does_not_disturb_what_each_book_holds():
    session = PairsSession()
    session.start_trader(QUICK)
    session.run_for(40)
    before = [session.market.books[k].mid_price for k in (A, B)]
    asyncio.run(session.fit_trader("zscore", QUICK, {"entry": [2.0]}, 500, 1, "pnl", 1))
    assert [session.market.books[k].mid_price for k in (A, B)] == before and len(session.runners) == 1
