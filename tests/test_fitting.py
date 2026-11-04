"""Fitting a strategy's settings over a training period (LOBManager.training_tape and fit_strategy, app/manager.py)."""

import asyncio
import re

import pytest
from starlette.testclient import TestClient

from app.main import app, get_view_context, registry, render_panels
from app.manager import LOBManager, _risk_adjusted
from app.sessions import COOKIE_NAME
from app.strategy_specs import MAX_FIT_CANDIDATES, STRATEGY_SPECS


def run(coro):
    return asyncio.run(coro)


def session(seed=3, events=2000):
    m = LOBManager()
    m.seed = seed
    m.reset()
    m.step(events)
    return m


def by_hand(training, spec, values):
    """One candidate on the training tape, as the fit runs it: from the start, to the end."""
    m = LOBManager()
    m.start_playback(training)
    s = m.start_strategy(spec, values)
    m.step(10**9)
    return s.account.summary(m._mark())["total"]


def test_a_fit_tries_every_valid_combination_on_the_same_period():
    m = session()
    spec = STRATEGY_SPECS["momentum"]
    training = run(m.training_tape(4000, seed=1))
    grid = {"fast": [3, 10, 30], "slow": [20, 40]}  # fast 30 with slow 20 is not a momentum strategy
    fit = run(m.fit_strategy(spec, spec.defaults(), grid, training, "a test", "pnl"))
    tried = sorted((c["values"]["fast"], c["values"]["slow"]) for c in fit.candidates)
    assert tried == [(3, 20), (3, 40), (10, 20), (10, 40), (30, 40)]
    scores = [c["score"] for c in fit.candidates]
    assert scores == sorted(scores, reverse=True)
    for c in fit.candidates[:2] + fit.candidates[-1:]:  # what the fit says each made, it made
        assert c["account"]["total"] == by_hand(training, spec, c["values"]) == c["score"]
    assert m.last_fit is fit and m.fitting is None


def test_a_fit_by_risk_ranks_by_mean_over_spread_of_the_p_and_l_changes():
    assert _risk_adjusted([(0, 0.0), (1, 1.0), (2, 2.0), (3, 2.0)]) == pytest.approx((2 / 3) / (2 ** 0.5 / 3))
    assert _risk_adjusted([(0, 0.0), (1, 0.0)]) == 0.0
    m = session()
    spec = STRATEGY_SPECS["avellaneda_stoikov"]
    training = run(m.training_tape(3000, seed=2))
    fit = run(m.fit_strategy(spec, spec.defaults(), {"gamma": [0.003, 0.1]}, training, "a test", "sharpe"))
    assert [c["score"] for c in fit.candidates] == sorted((c["score"] for c in fit.candidates), reverse=True)
    assert all(c["score"] != c["account"]["total"] for c in fit.candidates)


def test_a_fresh_training_run_is_not_the_market_that_comes_next():
    m = session()
    before = (m.event_count, m.sim_time, m.book.best_bid, m.book.best_ask)
    training = run(m.training_tape(3000, seed=1))
    assert (m.event_count, m.sim_time, m.book.best_bid, m.book.best_ask) == before  # the live session waits
    assert training.start_time == m.sim_time
    m.start_recording()  # what the live session does next
    m.step(3000)
    future = m.stop_recording()
    assert training.tape.time[0] != future.tape.time[0]  # another draw of the same market, not its future
    assert list(training.tape.time) != list(future.tape.time)
    again = run(m.training_tape(3000, seed=1))
    assert again.start_time == m.sim_time != training.start_time  # from where the session is now


def test_a_fresh_run_has_the_live_strategies_in_it_and_the_same_seed_makes_the_same_period():
    m = session()
    spec = STRATEGY_SPECS["market_making"]
    m.start_strategy(spec, spec.defaults())
    one = run(m.training_tape(2000, seed=5))
    two = run(m.training_tape(2000, seed=5))
    assert one.meta["strategies"] == ["Market making 1"]
    assert list(one.tape.time) == list(two.tape.time) and list(one.tape.price) == list(two.tape.price)


def test_fitting_on_a_recording():
    m = session()
    m.start_recording()
    m.step(3000)
    recording = m.stop_recording()
    spec = STRATEGY_SPECS["queue_imbalance"]
    fit = run(m.fit_strategy(spec, spec.defaults(), {"threshold": [0.4, 0.8]}, recording, recording.name, "pnl"))
    for c in fit.candidates:
        assert c["account"]["total"] == by_hand(recording, spec, c["values"])
    assert fit.span == (recording.start_time, recording.end_time) and fit.source == "Recording 1"


def test_fitting_the_random_trader_picks_a_seed():
    m = session()
    spec = STRATEGY_SPECS["random"]
    training = run(m.training_tape(3000, seed=1))
    fit = run(m.fit_strategy(spec, spec.defaults(), {"seed": list(range(1, 9))}, training, "a test", "pnl"))
    assert len({c["account"]["total"] for c in fit.candidates}) > 1  # luck differs between seeds


def test_starting_the_fitted_candidate_keeps_its_training_record():
    m = session()
    spec = STRATEGY_SPECS["breakout"]
    training = run(m.training_tape(2000, seed=1))
    fit = run(m.fit_strategy(spec, spec.defaults(), {"window": [10, 20]}, training, "a test period", "pnl"))
    s = m.start_fitted(fit.id, 2)
    assert s.values == fit.candidates[1]["values"]
    assert s.fit["rank"] == 2 and s.fit["of"] == 2 and s.fit["in_training"] == fit.candidates[1]["account"]["total"]
    assert "fitted on a test period" in render_panels(get_view_context(m))["strategies"]
    with pytest.raises(LookupError):
        m.start_fitted(fit.id + 1, 1)
    with pytest.raises(LookupError):
        m.start_fitted(fit.id, 3)


def test_with_more_combinations_than_a_fit_tries_it_tries_a_random_sample_of_them():
    m = session()
    spec = STRATEGY_SPECS["momentum"]
    training = run(m.training_tape(1000, seed=1))
    grid = {"fast": list(range(1, 8)), "slow": list(range(20, 28))}  # 56 combinations, all valid

    def tried(seed):
        fit = run(m.fit_strategy(spec, spec.defaults(), grid, training, "", "pnl", seed=seed))
        assert (fit.valid, fit.size, fit.seed) == (56, 56, seed)
        return {(c["values"]["fast"], c["values"]["slow"]) for c in fit.candidates}

    one = tried(1)
    assert len(one) == MAX_FIT_CANDIDATES and one <= {(f, s) for f in grid["fast"] for s in grid["slow"]}
    assert tried(1) == one != tried(2)  # the seed draws the sample


def test_a_long_training_period_tries_fewer_candidates(monkeypatch):
    import app.manager as manager_module

    monkeypatch.setattr(manager_module, "MAX_FIT_WORK", 20_000)
    m = session()
    spec = STRATEGY_SPECS["mean_reversion"]
    training = run(m.training_tape(1000, seed=1))  # 1,000 events: 20 candidates fit in the budget
    fit = run(m.fit_strategy(spec, spec.defaults(), {"window": [5, 10, 20, 40], "entry": [1.5, 2.0, 2.5],
                                                     "exit": [0.0, 0.5, 1.0]}, training, "", "pnl"))
    assert len(fit.candidates) == 20 and fit.valid == 36


def test_a_grid_too_large_to_count_is_sampled_without_counting():
    spec = STRATEGY_SPECS["mean_reversion"]
    grid = {"window": [5, 10, 20, 30, 40, 60, 80, 100], "entry": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            "exit": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], "size": [1, 2, 3, 5, 8, 10, 20, 50],
            "interval": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]}  # 32,768 combinations
    picked, valid = spec.candidates(spec.defaults(), grid, budget=MAX_FIT_CANDIDATES, seed=3)
    assert valid is None and len(picked) == MAX_FIT_CANDIDATES
    assert len({tuple(sorted(v.items())) for v in picked}) == MAX_FIT_CANDIDATES  # all different
    assert all(spec.valid(v) for v in picked)  # exit below entry in every one
    assert spec.candidates(spec.defaults(), grid, budget=MAX_FIT_CANDIDATES, seed=3)[0] == picked


def test_a_fit_says_when_it_cannot_be_done():
    m = session()
    spec = STRATEGY_SPECS["momentum"]
    training = run(m.training_tape(1000, seed=1))
    with pytest.raises(ValueError, match="none of these combinations"):
        run(m.fit_strategy(spec, spec.defaults(), {"fast": [40], "slow": [20]}, training, "", "pnl"))
    with pytest.raises(ValueError, match="objective"):
        run(m.fit_strategy(spec, spec.defaults(), {"fast": [3]}, training, "", "luck"))
    m.fitting = (0, 1)
    with pytest.raises(RuntimeError, match="another fit"):
        run(m.fit_strategy(spec, spec.defaults(), {"fast": [3]}, training, "", "pnl"))
    m.fitting = None
    m.set_model("manual")
    with pytest.raises(ValueError, match="manual model"):
        run(m.training_tape(1000, seed=1))


# ---- From the page ----

@pytest.fixture
def client():
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def manager(client):
    return registry.get(client.cookies[COOKIE_NAME])


FORM = {"kind": "momentum", "fast": 5, "slow": 20, "threshold": 0.5, "size": 5, "interval": 0.5,
        "execution": "aggressive", "protection": 5, "fit_fast": "on", "grid_fast": "3, 5; 10, 5",
        "fit_slow": "on", "grid_slow": "20, 40", "source": "fresh", "events": 2000, "fit_seed": 2, "objective": "pnl"}


def test_fitting_from_the_page(client, manager):
    client.post("/api/sim/control", data={"action": "step", "n": 1000})
    res = client.post("/api/strategies/fit", data=FORM)
    assert res.status_code == 200 and res.text.count("data-rank=") == 6  # 3 fasts (5 once) times 2 slows
    fit_id = int(re.search(r'data-fit-id="(\d+)"', res.text)[1])
    assert "fitted over a fresh run of" in res.text and "seed 2" in res.text
    res = client.post("/api/strategies/start", data={"fit_id": fit_id, "rank": 1})
    assert res.status_code == 200 and "as fitted" in res.text
    assert manager.strategies[0].fit["of"] == 6
    assert client.post("/api/strategies/start", data={"fit_id": fit_id + 5, "rank": 1}).status_code == 409
    assert client.post("/api/strategies/start", data={"fit_id": "x", "rank": 1}).status_code == 422


@pytest.mark.parametrize("change, message", [
    ({"fit_fast": None, "fit_slow": None}, "tick at least one setting"),
    ({"grid_fast": "3, x"}, "enter a number"),
    ({"grid_fast": ""}, "give the values to try"),
    ({"grid_fast": "1, 2, 3, 4, 5, 6, 7, 8, 9"}, "at most 8 values"),
    ({"grid_fast": "30", "grid_slow": "20"}, "none of these combinations of settings is valid"),
    ({"fit_seed": -1}, "Seed: must be between"),
    ({"source": "recording:9"}, "no longer here"),
    ({"source": "elsewhere"}, "no longer here"),
    ({"objective": "luck"}, "best by one of"),
    ({"events": 10}, "between 1000 and 50000"),
    ({"kind": "astrology"}, "Choose one of"),
    ({"slow": 0}, "must be between"),
])
def test_the_page_is_told_why_a_fit_cannot_be_done(client, change, message):
    form = {k: v for k, v in {**FORM, **change}.items() if v is not None}
    res = client.post("/api/strategies/fit", data=form)
    assert res.status_code == 422 and message in res.text


def test_fitting_many_settings_from_the_page_tries_a_sample(client, manager):
    spec = STRATEGY_SPECS["mean_reversion"]
    form = {"kind": "mean_reversion", **spec.defaults(), "source": "fresh", "events": 1000, "fit_seed": 4,
            "objective": "pnl"}
    for p in spec.params:  # every setting: 1,458 combinations
        form[f"fit_{p.name}"] = "on"
        form[f"grid_{p.name}"] = ", ".join(str(v) for v in spec.grid_values(p.name))
    res = client.post("/api/strategies/fit", data=form)
    assert res.status_code == 200 and 'data-sampled="true"' in res.text
    assert f"{MAX_FIT_CANDIDATES} of 1,458 combinations, drawn at random (seed 4)" in res.text
    few = client.post("/api/strategies/fit", data=FORM)
    assert 'data-sampled="false"' in few.text and "all 6 combinations" in few.text


def test_fitting_on_a_recording_from_the_page(client, manager):
    client.post("/api/recording/start")
    client.post("/api/sim/control", data={"action": "step", "n": 1500})
    rid = client.post("/api/recording/stop").json()["id"]
    res = client.post("/api/strategies/fit", data={**FORM, "source": f"recording:{rid}"})
    assert res.status_code == 200 and "fitted over Recording 1" in res.text
    manager.set_model("manual")  # a recording still works with the manual model
    assert client.post("/api/strategies/fit", data={**FORM, "source": f"recording:{rid}"}).status_code == 200
    assert "manual model" in client.post("/api/strategies/fit", data=FORM).text


def test_the_card_offers_the_recordings_to_fit_on(client, manager):
    client.post("/api/recording/start")
    client.post("/api/sim/control", data={"action": "step", "n": 300})
    client.post("/api/recording/stop")
    page = client.get("/").text
    assert '<option value="recording:1">Recording 1 (' in page
    assert page.count("<fieldset data-fit-kind=") == len(STRATEGY_SPECS)  # a panel of values for each strategy
    assert 'id="fit-count"' in page  # where the number of combinations shows as settings are ticked
