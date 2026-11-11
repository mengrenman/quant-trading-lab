"""The informed-traders model in the dashboard: its spec and page, the ground truth it reports, and Reveal.

``lob/models/informed.py`` is tested in tests/test_informed.py. Here the session around it: that the model starts,
records and plays back exactly; that the share of informed lots and what makers lose to them (read live, from
``last_informed`` and ``value()``) are what the trade log and a recorded value path give when they are worked out
here by another route; that the hidden value sampled once a unit reaches the chart's candles bucket by bucket, hand
worked at three intervals, and only when asked for, and only for a live run of the model that has one; and the
switch, its routes and its stream.
"""

import asyncio
import inspect
import json
import re
import signal

import numpy as np
import pytest
from starlette.testclient import TestClient

from app import manager as manager_module
from app import recording as rec
from app.main import app, facts_html, get_view_context, registry, render_panels
from app.manager import LOBManager
from app.model_specs import MODEL_SPECS
from app.sessions import COOKIE_NAME
from lob.models.informed import InformedZIModel
from lob.order import Side
from tests.test_recording import start, state, trades

BID, ASK = Side.BID, Side.ASK
KEY = "informed_zi"
NEW = ("sigma_v", "jump_rate", "jump_size", "kappa", "cap", "persistence")
LABEL = "ZI with informed traders"


@pytest.fixture
def client():
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def manager(client):
    return registry.get(client.cookies[COOKIE_NAME])


@pytest.fixture
def deadline():
    """A test that reads a stream fails after a minute instead of waiting for an event that is never sent."""
    def expired(signum, frame):
        raise TimeoutError("the stream did not send what the test waits for")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    yield
    signal.alarm(0)


def informed_manager(record_path=False, seed=None):
    """A manager running the model, with its value path recorded when asked (the same model, built as the spec builds
    it, so the same run: the path is what lets a test work the ground truth out from the trade log)."""
    m = LOBManager()
    if seed is not None:
        m.seed = seed
        m.reset(100)
    m.set_model(KEY)
    if record_path:
        m.model = InformedZIModel(initial_mid=100, record_path=True, **m.model_params[KEY])
    return m


def form(**overrides):
    values = {"model": KEY, "seed": 42, **MODEL_SPECS[KEY].defaults()}
    values.update(overrides)
    return values


def toggle_tag(html):
    """The Reveal button's opening tag, and whether it has the disabled attribute (not the disabled: classes)."""
    tag = re.search(r'<button[^>]*id="reveal-toggle"[^>]*>', html).group(0)
    return tag, re.search(r"\sdisabled(?=[\s>=])", tag) is not None


def candles(m, interval, **kwargs):
    m.set_bar_interval(interval)
    return m.get_candles(reveal=True, **kwargs)


# ---- The spec ----

def test_the_sixth_model_is_a_spec_before_manual_with_the_ranges_the_builder_gave_and_no_notebook_values():
    keys = list(MODEL_SPECS)
    assert keys == ["smith_zi", "hawkes_zi", "mike_farmer", "mike_farmer_2008", "parlour", KEY, "manual"]
    spec = MODEL_SPECS[KEY]
    assert spec.label == LABEL and spec.factory is InformedZIModel and spec.notebook is None
    smith = MODEL_SPECS["smith_zi"].params
    assert spec.params[:len(smith)] == smith  # the ZI model's own parameters, the same
    assert [(p.name, p.default, p.low, p.high, p.step, p.integer) for p in spec.params[len(smith):]] == [
        ("sigma_v", 0.3, 0.0, 5.0, 0.05, False), ("jump_rate", 0.05, 0.0, 2.0, 0.01, False),
        ("jump_size", 2.0, 0.0, 50.0, 0.5, False), ("kappa", 2.0, 0.0, 20.0, 0.1, False),
        ("cap", 3.0, 0.0, 30.0, 0.5, False), ("persistence", 0.5, 0.5, 0.99, 0.01, False)]
    assert "hidden" in spec.summary and "informed" in spec.summary and "Persistence" in spec.summary
    assert "persistence" in next(p.label for p in spec.params if p.name == "persistence").lower()


def test_its_defaults_are_the_models_own_and_every_parameter_parses_at_its_ends():
    spec = MODEL_SPECS[KEY]
    own = inspect.signature(InformedZIModel.__init__).parameters
    for name in NEW:
        assert spec.defaults()[name] == own[name].default, name
    for param in spec.params:
        for value in (param.default, param.low, param.high):
            assert param.parse(value) == value, (param.name, value)
        with pytest.raises(ValueError):
            param.parse(param.high + param.step)
        with pytest.raises(ValueError):
            param.parse(param.low - param.step)
    built = spec.build(spec.defaults(), 100)
    assert isinstance(built, InformedZIModel) and built.kappa == 2.0 and built.value() == 100.0


def test_every_model_with_a_factory_has_a_placement_note_for_the_facts_panel():
    from app.facts import PLACEMENT_BY_DESIGN
    assert {key for key, spec in MODEL_SPECS.items() if spec.factory} == set(PLACEMENT_BY_DESIGN)


# ---- On the page ----

def test_the_page_offers_the_model_with_its_five_parameters_and_the_line_says_none_for_the_others(client, manager):
    html = client.get("/").text
    assert f'value="{KEY}"' in html and LABEL in html
    assert re.search(r'id="informed-line"[^>]*data-informed="no"', html)
    assert "Informed traders: none in this model" in html
    tag, disabled = toggle_tag(html)
    assert disabled and 'aria-pressed="false"' in tag
    assert f"needs the {LABEL} model" in html
    assert "lob-informed.js" in html and client.get("/static/js/lob-informed.js").status_code == 200

    reply = client.post("/api/sim/model", data={"model": KEY})
    assert reply.status_code == 200 and manager.model_type == KEY
    for name in NEW:
        assert f'name="{name}"' in reply.text, name
    assert "Informed traders: no market orders yet" in reply.text and 'data-informed="yes"' in reply.text
    assert "Use notebook 05" not in reply.text  # it has no notebook values to fill in
    assert toggle_tag(client.get("/").text)[1] is False


def test_the_parameters_are_applied_checked_and_kept_per_model(client, manager):
    client.post("/api/sim/model", data={"model": KEY})
    reply = client.post("/api/sim/params", data=form(sigma_v=0.8, kappa=4, cap=10, jump_rate=0.2, jump_size=5, mu=3,
                                                     persistence=0.9))
    assert reply.status_code == 200 and "Applied" in reply.text
    model = manager.model
    assert (model.sigma_v, model.kappa, model.cap, model.jump_rate, model.jump_size, model.mu,
            model.persistence) == (0.8, 4.0, 10.0, 0.2, 5.0, 3.0, 0.9)
    before = (dict(manager.model_params[KEY]), manager.model)
    for field, value in [("sigma_v", "-0.1"), ("sigma_v", "5.1"), ("kappa", "20.5"), ("cap", "30.5"),
                         ("jump_rate", "2.5"), ("jump_size", "abc"), ("persistence", "0.49"),
                         ("persistence", "1"), ("persistence", "nan")]:
        bad = client.post("/api/sim/params", data=form(**{field: value}))
        assert bad.status_code == 422 and "Not applied" in bad.text, (field, value)
        assert (manager.model_params[KEY], manager.model) == before
    client.post("/api/sim/model", data={"model": "smith_zi"})
    client.post("/api/sim/model", data={"model": KEY})
    assert manager.model.kappa == 4.0 and manager.model.persistence == 0.9  # kept while another model ran


def test_the_facts_panel_says_the_notebook_runs_a_model_only_when_it_does(manager):
    runs, does_not = "Notebook 05 runs each model 32 times", "Notebook 05 does not run this model"
    for model, present in (("smith_zi", runs), (KEY, does_not), ("manual", None)):
        manager.set_model(model)
        html = asyncio.run(facts_html(manager))["facts"]
        assert "No model is running" in html or model != "manual"
        for text, here in ((runs, present == runs), (does_not, present == does_not)):
            assert (text in html) == here, (model, text)


# ---- It runs, records and plays back exactly ----

def test_the_model_starts_and_replays_from_its_seed_with_the_same_counters_and_value():
    m = informed_manager(seed=7)
    m.step(20_000)
    first = (m.model.informed_lots, m.model.taker_lots, m.model.value(), m.informed_status()["loss"], m.event_count,
             m.sim_time, list(m._value_samples))
    assert first[0] > 0 and first[1] > first[0] and np.isfinite(first[2])
    assert m.book.best_bid is not None and m.book.best_ask is not None
    m.reset(100)
    assert m.model.taker_lots == 0 and m.informed_status()["taker_lots"] == 0 and m.informed_status()["loss"] is None
    m.step(20_000)
    second = (m.model.informed_lots, m.model.taker_lots, m.model.value(), m.informed_status()["loss"], m.event_count,
              m.sim_time, list(m._value_samples))
    assert second == first


def informed_session():
    """A session on the model that records while it does what a session does, with Reveal on throughout."""
    m = informed_manager(seed=5)
    m.set_reveal(True)
    m.step(2500)
    m.submit_limit_order(m.book.best_bid - 2, 30, BID, display_size=10)
    start(m, "market_making")
    m.step(400)
    begin = len(m._trade_log["time"])
    m.start_recording()
    m.step(4000)
    m.submit_market_order(20, ASK)
    m.step(3000)
    m.toggle_engine()
    m.step(3000)
    start(m, "momentum")
    m.step(2000)
    m.start_split_order(BID, 40, 4, 3.0)
    m.step(3000)
    m.cancel_all_orders()
    m.step(1000)
    expected, expected_trades = state(m), trades(m, begin)
    return m.stop_recording(), expected, expected_trades, m


def test_a_recording_of_the_model_plays_back_exactly_and_says_the_value_is_not_in_it():
    recording, expected, expected_trades, live = informed_session()
    assert recording.meta["models"] == [LABEL]
    p = LOBManager()
    p.set_model(KEY)  # what takes over afterwards
    p.start_playback(recording)
    p.step(10**9)
    assert p.playback.ended and p.playback.cursor == len(recording.tape)
    got = state(p)
    for key in expected:
        assert got[key] == expected[key], key
    assert trades(p) == expected_trades and len(expected_trades["time"]) > 500 and sum(expected_trades["yours"]) > 0

    # The tape holds orders and who they came from, not whether the model's were informed nor the value.
    status = p.informed_status()
    assert status["playback"] is True and status["share"] is None and status["loss"] is None
    assert status["line"].startswith("Informed traders: not recorded")
    assert status["reveal"] == {"on": False, "disabled": False, "active": False, "note": "not recorded"}
    assert not any("fair" in c for c in p.get_candles(reveal=p.reveal_active))  # nothing to draw, nothing sent
    assert p.model.informed_lots == 0  # the model is not driven in a playback

    # The live session was drawing the value all along, and the file survives its trip.
    assert live.informed_status()["reveal"]["active"] is True
    again = rec.from_bytes(rec.to_bytes(recording), 1)
    q = LOBManager()
    q.start_playback(again)
    q.step(10**9)
    assert state(q) == got


def test_a_recording_of_another_model_is_none_in_the_line_and_the_switch_says_not_recorded_whatever_the_model():
    m = LOBManager()
    m.step(1500)
    m.start_recording()
    m.step(1500)
    recording = m.stop_recording()
    m.set_model(KEY)  # what would take over afterwards
    m.start_playback(recording)
    status = m.informed_status()
    assert status["line"] == "Informed traders: none in the recorded model" and status["model"] is True
    assert status["reveal"]["note"] == "not recorded" and status["reveal"]["disabled"] is False
    m.set_model("smith_zi")
    assert m.informed_status()["reveal"] == {"on": False, "disabled": True, "active": False, "note": "not recorded"}


# ---- The ground truth, live ----

def test_the_share_and_the_makers_loss_are_what_the_trade_log_and_the_recorded_value_give():
    m = informed_manager(record_path=True)
    m.step(30_000)
    frame, model = m.trades_frame(), m.model
    assert not frame["yours"].any()  # every fill is the model's own market order
    lots = frame["size"].to_numpy()
    informed = model.is_informed(frame["taker_order_id"])  # the model's own record of its ids: a different route
    assert 0.05 < lots[informed].sum() / lots.sum() < 0.4 and informed.sum() > 100
    assert model.taker_lots == lots.sum()  # every market order filled in full, so lots sent are lots filled
    status = m.informed_status()
    assert status["model"] is True and status["playback"] is False
    assert status["filled_lots"] == lots[informed].sum() == status["informed_lots"]
    assert status["taker_lots"] == lots.sum() and status["share"] == pytest.approx(lots[informed].sum() / lots.sum())
    # The loss on a lot an informed trader took, against V at the time of the fill, read from the recorded path.
    loss = model.maker_loss(frame["time"], frame["price"], frame["aggressor"] == "buy")
    assert status["loss"] == pytest.approx(np.average(loss[informed], weights=lots[informed]), rel=1e-9, abs=1e-9)
    assert status["loss"] > 0.5  # positive by construction, and of the order of the gap to the quote it hit
    assert status["line"] == (f"Informed traders: {100 * status['share']:.1f}% of the model's taker lots "
                              f"({status['informed_lots']:,} of {status['taker_lots']:,}); makers lose "
                              f"{status['loss']:.2f} ticks a lot to them, against the hidden value")
    # Strictly the first fill of an informed order costs the makers its gap: nothing was lost on the uninformed ones
    # that the half-spread does not explain (they paid it): the loss on all fills is smaller.
    assert np.average(loss, weights=lots) < status["loss"]


def test_the_score_of_an_informed_fill_is_worked_by_hand():
    m = informed_manager()
    model = m.model
    model._value = 103.0  # V, as the model has it right after an order
    from lob.order import Fill
    buy = [Fill(1, 2, 101, 5, 0.0), Fill(1, 3, 102, 3, 0.0)]  # a taker buy at 101 and 102: V - price = 2 and 1
    m._score_informed(BID, buy)
    assert m._informed_fill_lots == 8 and m._informed_fill_loss == 5 * 2 + 3 * 1
    model._value = 96.0
    sell = [Fill(1, 4, 99, 4, 0.0)]  # a taker sell at 99: price - V = 3
    m._score_informed(ASK, sell)
    assert m._informed_fill_lots == 12 and m._informed_fill_loss == 13 + 4 * 3
    model._value = 100.0
    m._score_informed(BID, [Fill(1, 5, 104, 2, 0.0)])  # a buy above V: the maker gained 4 a lot: negative loss
    assert m._informed_fill_loss == 25 - 8 and m.informed_status()["loss"] == pytest.approx(17 / 14)


def test_only_orders_the_model_marks_informed_are_scored_at_the_value_the_model_had_for_them(monkeypatch):
    m = informed_manager()
    flags, scored = [], []
    real_generate, real_score = m.model.generate_event, m._score_informed

    def generate(*args):
        order = real_generate(*args)
        flags.append((order, m.model.last_informed, m.model.value()))
        return order

    def score(side, fills):
        scored.append((side, m.model.value(), sum(f.size for f in fills)))
        real_score(side, fills)
    monkeypatch.setattr(m.model, "generate_event", generate)
    monkeypatch.setattr(m, "_score_informed", score)
    m.step(6000)
    sent = [(order.side, value) for order, informed, value in flags if informed and order is not None]
    assert len(sent) > 20 and len(flags) > 5000
    assert [(side, value) for side, value, _ in scored] == sent  # one for each informed order, of its side, at its V
    assert sum(lots for *_, lots in scored) == m._informed_fill_lots > 0
    assert m.model.informed_lots >= m._informed_fill_lots


def test_other_models_say_none_and_report_no_share():
    for model in ("smith_zi", "hawkes_zi", "mike_farmer", "mike_farmer_2008", "parlour", "manual"):
        m = LOBManager()
        m.set_model(model)
        m.step(500)
        status = m.informed_status()
        assert status["line"] == "Informed traders: none in this model", model
        assert status["model"] is False and status["share"] is None and status["loss"] is None, model
        assert (status["informed_lots"], status["taker_lots"], status["filled_lots"]) == (0, 0, 0), model
        assert status["reveal"] == {"on": False, "disabled": True, "active": False,
                                    "note": f"needs the {LABEL} model"}, model
        assert m._informed_model is False and len(m._value_samples) == 0 and m._informed_fill_lots == 0
        json.dumps(status, allow_nan=False)


def test_the_counters_start_with_each_new_model_and_a_value_in_a_new_model_starts_at_the_mid():
    m = informed_manager(seed=3)
    m.step(8000)
    assert m.informed_status()["taker_lots"] > 0 and m._informed_fill_lots > 0
    mid = int(m.book.mid_price)
    m.configure(KEY, {**m.model_params[KEY], "kappa": 4.0}, 9)  # rebuilds the model on the book as it is
    status = m.informed_status()
    assert (status["taker_lots"], status["informed_lots"], status["filled_lots"], status["loss"]) == (0, 0, 0, None)
    assert status["line"] == "Informed traders: no market orders yet"
    assert m.model.value() == float(mid) and list(m._value_samples) == [float(mid)]
    assert m._value_last == int(m.sim_time)
    m.step(500)
    m.set_model("smith_zi")
    m.set_model(KEY)
    assert m.informed_status()["taker_lots"] == 0 and m._informed_fill_lots == 0


def test_the_line_and_the_whole_status_are_plain_json_at_every_stage():
    m = informed_manager()
    for steps in (0, 1, 2, 3, 50, 5000):
        m.step(steps)
        json.dumps(m.informed_status(), allow_nan=False)
    assert m.informed_status()["line"].startswith("Informed traders: ")


# ---- The hidden value, once a unit ----

def test_the_values_samples_are_the_model_value_at_the_event_that_crossed_into_each_unit():
    """The sample for unit k is V at the first event at or after k (the model has advanced it to that event)."""
    m = informed_manager(record_path=True)
    m.step(9000)
    path_t = np.frombuffer(m.model._path[0], dtype=float)
    path_v = np.frombuffer(m.model._path[1], dtype=float)
    last = m._value_last
    samples = np.array(m._value_samples)
    assert last == int(m.sim_time) and len(samples) == last + 1
    expected = [100.0] + [path_v[np.searchsorted(path_t, k, side="left")] for k in range(1, last + 1)]
    assert np.array_equal(samples, expected)
    assert samples.min() != samples.max() and np.isfinite(samples).all()


def test_the_candles_carry_the_value_by_bucket_as_the_last_sample_in_it_worked_by_hand():
    m = informed_manager()
    m._advance(3.5)  # nothing happened: units 1 to 3 repeat the value the model has (100)
    m.model._value = 107.5
    m._advance(6.2)
    assert m._value_last == 6 and list(m._value_samples) == [100.0] * 4 + [107.5] * 3
    assert [c["fair"] for c in candles(m, 1)] == [100.0] * 4 + [107.5] * 3
    # Interval 2: buckets {0, 1}, {2, 3}, {4, 5}, {6, 7 (not yet)}: the last sample of each.
    assert [(c["time"], c["fair"]) for c in candles(m, 2)] == [(0, 100.0), (2, 100.0), (4, 107.5), (6, 107.5)]
    assert [(c["time"], c["fair"]) for c in candles(m, 4)] == [(0, 100.0), (4, 107.5)]
    assert [(c["time"], c["fair"]) for c in candles(m, 5)] == [(0, 107.5), (5, 107.5)]  # unit 4 is in the first
    assert [(c["time"], c["fair"]) for c in candles(m, 3)] == [(0, 100.0), (3, 107.5), (6, 107.5)]  # {3, 4, 5}: unit 5
    # A reload that asks for the bars since a time has the same values for them.
    assert [(c["time"], c["fair"]) for c in candles(m, 2, since=4)] == [(4, 107.5), (6, 107.5)]


def test_the_candles_carry_the_value_only_when_asked_for_and_the_default_payload_is_as_it_was():
    m = informed_manager()
    m.step(3000)
    plain = m.get_candles()
    assert all("fair" not in c for c in plain)
    asked = m.get_candles(reveal=True)
    assert all("fair" in c for c in asked)
    assert [{k: v for k, v in c.items() if k != "fair"} for c in asked] == plain  # nothing else changed
    assert asked[-1]["fair"] == m._value_samples[-1]  # the newest bar has the newest unit's sample


def test_the_buckets_of_a_real_run_have_the_values_the_recorded_path_gives_at_three_intervals():
    m = informed_manager(record_path=True)
    m.step(12_000)
    path_t = np.frombuffer(m.model._path[0], dtype=float)
    path_v = np.frombuffer(m.model._path[1], dtype=float)

    def sample(unit):
        return 100.0 if unit == 0 else path_v[np.searchsorted(path_t, unit, side="left")]
    for interval in (1, 5, 15):
        for c in candles(m, interval):
            unit = min(c["time"] + interval - 1, m._value_last)
            assert c["fair"] == sample(unit), (interval, c["time"])


def test_a_gap_in_the_clock_repeats_the_value_and_a_clock_that_jumps_further_than_the_samples_reach_keeps_the_last(
        monkeypatch):
    monkeypatch.setattr(manager_module, "VALUE_SAMPLES", 50)
    m = informed_manager()
    m._advance(400.5)  # 400 units at once: only the last 50 are kept, all the same value
    assert m._value_last == 400 and list(m._value_samples) == [100.0] * 50
    fair = [c["fair"] for c in candles(m, 1)]
    assert fair[:351] == [None] * 351 and fair[351:] == [100.0] * 50  # none before the samples, the value after
    # Unit by unit, with a value of its own at each: the trim keeps the units where they are.
    for k in range(401, 520):
        m.model._value = float(k)
        m._advance(k + 0.5)
        assert 50 <= len(m._value_samples) <= 100 and m._value_last == k
    fair = {c["time"]: c["fair"] for c in candles(m, 1)}
    kept = len(m._value_samples)
    assert kept >= 50
    for unit in range(519 - kept + 1, 520):
        assert fair[unit] == float(unit), unit
    assert fair[519 - kept] is None and fair[0] is None


def test_the_samples_start_over_with_a_reset_a_playback_and_a_new_model_and_are_not_taken_in_a_playback():
    m = informed_manager(seed=2)
    m.step(3000)
    assert len(m._value_samples) > 20
    m.start_recording()
    m.step(1500)
    recording = m.stop_recording()
    m.reset(100)
    assert list(m._value_samples) == [100.0] and m._value_last == 0
    m.step(500)
    m.start_playback(recording)
    start_unit = int(recording.start["time"])
    assert m._value_last == start_unit and len(m._value_samples) == 1  # a model for what takes over, at the mid
    m.step(10**9)
    assert len(m._value_samples) == 1  # nothing was sampled while the tape played
    m.exit_playback()
    assert m._value_last == int(m.sim_time) and len(m._value_samples) == 1
    m.step(2000)
    assert len(m._value_samples) > 20


# ---- Reveal ----

def test_reveal_is_a_session_flag_that_reloads_the_chart_and_cannot_be_turned_on_without_a_value():
    m = LOBManager()
    version = m.chart_version
    with pytest.raises(ValueError, match="hidden value"):
        m.set_reveal(True)
    assert m.reveal_on is False and m.chart_version == version
    m.set_reveal(False)  # off is always allowed, and changes nothing
    assert m.chart_version == version
    m.set_model(KEY)
    version = m.chart_version
    m.set_reveal(True)
    assert m.reveal_on and m.reveal_active and m.chart_version == version + 1
    m.set_reveal(True)
    assert m.chart_version == version + 1  # no change, no reload
    m.set_reveal(False)
    assert not m.reveal_on and not m.reveal_active and m.chart_version == version + 2


def test_reveal_is_not_a_parameter_so_changing_parameters_keeps_it_and_does_not_reseed_the_model_for_it():
    m = informed_manager(seed=4)
    m.set_reveal(True)
    m.step(2000)
    state_before = m.rng.bit_generator.state
    m.set_reveal(False)
    m.set_reveal(True)
    assert m.rng.bit_generator.state == state_before and m.model.taker_lots > 0  # no rebuild, no reseed
    assert KEY in m.model_params and not any("reveal" in name for name in m.model_params[KEY])
    version = m.chart_version
    m.configure(KEY, {**m.model_params[KEY], "sigma_v": 0.5}, 4)
    assert m.reveal_on and m.chart_version > version  # a parameter change keeps it; the line restarts with the model


def test_the_flag_is_off_when_the_model_changes_to_one_without_a_value_and_the_chart_reloads():
    m = informed_manager()
    m.set_reveal(True)
    version = m.chart_version
    m.set_model("parlour")
    assert m.reveal_on is False and m.chart_version > version  # a line that was drawn has to go
    assert m.informed_status()["reveal"]["disabled"] is True
    m.set_model(KEY)
    assert m.reveal_on is False  # asked for again: it was a choice about the model that was left
    version = m.chart_version
    m.set_model("smith_zi")
    assert m.chart_version == version  # no line was drawn: nothing to reload for


def test_reveal_on_in_a_live_run_is_active_and_in_a_playback_it_is_not_even_with_the_flag_on():
    recording, _, _, live = informed_session()
    assert live.reveal_on and live.reveal_active
    assert live.informed_status()["reveal"] == {"on": True, "disabled": False, "active": True, "note": None}
    p = live
    p.start_playback(recording)
    status = p.informed_status()["reveal"]
    assert status == {"on": True, "disabled": False, "active": False, "note": "not recorded"}
    assert p.reveal_active is False
    p.exit_playback()
    assert p.informed_status()["reveal"] == {"on": True, "disabled": False, "active": True, "note": None}


# ---- The routes and the stream ----

def test_post_chart_reveal_switches_the_value_in_the_candles_and_reloads_the_chart(client, manager):
    client.post("/api/sim/model", data={"model": KEY})
    manager.step(12_000)
    plain = client.get("/api/candlesticks").json()
    assert plain["reveal"] == {"on": False, "disabled": False, "active": False, "note": None}
    assert all("fair" not in c for c in plain["candles"])
    version = manager.chart_version
    on = client.post("/api/chart/reveal", data={"on": "true"})
    assert on.status_code == 204 and manager.reveal_on and manager.chart_version == version + 1
    body = client.get("/api/candlesticks").json()
    assert body["reveal"]["active"] is True
    fair = [c["fair"] for c in body["candles"]]
    assert len(fair) > 50 and all(isinstance(v, float) for v in fair) and len(set(fair)) > 10
    assert body["candles"][-1]["fair"] == manager._value_samples[-1]  # the newest unit's sample
    assert client.post("/api/chart/reveal", data={"on": "true"}).status_code == 204
    assert manager.chart_version == version + 1
    assert client.post("/api/chart/reveal", data={"on": "false"}).status_code == 204
    assert manager.reveal_on is False and manager.chart_version == version + 2
    assert all("fair" not in c for c in client.get("/api/candlesticks").json()["candles"])


def test_post_chart_reveal_for_a_model_without_a_value_is_refused_with_its_reason_and_changes_nothing(client, manager):
    version = manager.chart_version
    refused = client.post("/api/chart/reveal", data={"on": "true"})
    assert refused.status_code == 422 and "hidden value" in refused.json()["detail"]
    assert LABEL in refused.json()["detail"]
    assert manager.reveal_on is False and manager.chart_version == version
    assert client.post("/api/chart/reveal", data={"on": "false"}).status_code == 204  # off is fine
    assert client.post("/api/chart/reveal", data={}).status_code == 422  # a missing field, as everywhere
    body = client.get("/api/candlesticks").json()
    assert body["reveal"] == {"on": False, "disabled": True, "active": False, "note": f"needs the {LABEL} model"}
    assert all("fair" not in c for c in body["candles"])


def test_in_a_playback_the_switch_can_be_set_but_the_candles_carry_no_value(client, manager):
    client.post("/api/sim/model", data={"model": KEY})
    client.post("/api/recording/start")
    client.post("/api/sim/control", data={"action": "step", "n": 800})
    rid = client.post("/api/recording/stop").json()["id"]
    assert client.post("/api/playback/start", data={"recording_id": rid}).status_code == 204
    assert client.post("/api/chart/reveal", data={"on": "true"}).status_code == 204
    client.post("/api/sim/control", data={"action": "step", "n": 300})
    body = client.get("/api/candlesticks").json()
    assert body["reveal"] == {"on": True, "disabled": False, "active": False, "note": "not recorded"}
    assert body["candles"] and all("fair" not in c for c in body["candles"])
    html = client.get("/").text
    assert "not recorded" in re.search(r'id="reveal-note"[^>]*>([^<]*)<', html).group(1)
    assert "Informed traders: not recorded" in html
    panels = render_panels(get_view_context(manager))
    assert 'data-playback="true"' in panels["sim_status"]
    assert client.post("/api/playback/exit").status_code in (200, 204)
    assert client.get("/api/candlesticks").json()["reveal"]["active"] is True


def events_of(lines):
    """Every (name, data) on a stream, the JSON ones decoded, refusing NaN and Infinity as a browser does."""
    event = None
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: ") and event:
            data = line[len("data: "):]
            if event in ("informed", "candles", "chart_reset"):
                data = json.loads(data, parse_constant=lambda token: pytest.fail(f"{token} in JSON"))
            yield event, data
            event = None


def test_the_stream_sends_the_ground_truth_when_it_changes_and_the_value_in_the_candles_when_asked(client, manager,
                                                                                                  deadline):
    client.post("/api/sim/model", data={"model": KEY})
    with client.stream("GET", "/events") as resp:
        events = events_of(resp.iter_lines())

        def next_of(wanted):
            return next(data for name, data in events if name == wanted)
        first = next_of("informed")  # on connect: the state as it stands
        assert first["line"] == "Informed traders: no market orders yet" and first["model"] is True
        assert first["reveal"] == {"on": False, "disabled": False, "active": False, "note": None}
        # A push that changes nothing it shows sends no more of it: read all of the events between the candles of
        # one push and the candles of the next (a wake-up after the first has begun is a push of its own).
        client.post("/api/chart/footprint", data={"on": "false"})
        next_of("candles")
        client.post("/api/chart/footprint", data={"on": "false"})
        between = []
        for name, _ in events:
            if name == "candles":
                break
            between.append(name)
        assert "studies" in between and "informed" not in between
        client.post("/api/sim/control", data={"action": "step", "n": 3000})
        second = next_of("informed")  # sent when it changed: this has the counters
        assert second["taker_lots"] > 0 and second["share"] == pytest.approx(
            second["informed_lots"] / second["taker_lots"]) and second["loss"] is not None
        assert second["line"].startswith(f"Informed traders: {100 * second['share']:.1f}%")
        assert second["taker_lots"] == manager.model.taker_lots
        client.post("/api/chart/reveal", data={"on": "true"})
        assert next_of("chart_reset") is True
        reveal = next_of("informed")["reveal"]
        assert reveal["on"] is True and reveal["active"] is True
        client.post("/api/sim/control", data={"action": "step", "n": 2000})
        bars = next_of("candles")
        assert bars and all(isinstance(bar["fair"], float) for bar in bars)
        client.post("/api/sim/model", data={"model": "smith_zi"})
        other = next_of("informed")
        assert other["line"] == "Informed traders: none in this model" and other["model"] is False
        assert other["reveal"]["disabled"] is True and other["reveal"]["on"] is False


def test_the_stream_of_a_playback_carries_no_value_whatever_the_switch_says(client, manager, deadline):
    client.post("/api/sim/model", data={"model": KEY})
    client.post("/api/chart/reveal", data={"on": "true"})
    client.post("/api/recording/start")
    client.post("/api/sim/control", data={"action": "step", "n": 1500})
    rid = client.post("/api/recording/stop").json()["id"]
    client.post("/api/playback/start", data={"recording_id": rid})
    assert manager.reveal_on is True  # the switch is still on: nothing is recorded to draw
    with client.stream("GET", "/events") as resp:
        seen = {}
        events = events_of(resp.iter_lines())
        for name, data in events:
            seen.setdefault(name, data)
            if {"informed", "candles"} <= set(seen):
                break
        assert seen["informed"]["reveal"] == {"on": True, "disabled": False, "active": False, "note": "not recorded"}
        assert seen["informed"]["line"].startswith("Informed traders: not recorded")
        client.post("/api/sim/control", data={"action": "step", "n": 800})
        bars = next(data for name, data in events if name == "candles")
        assert bars and all("fair" not in bar for bar in bars)


def test_the_stream_of_another_model_says_none_once_and_carries_no_value(client, manager, deadline):
    with client.stream("GET", "/events") as resp:
        seen = {}
        for name, data in events_of(resp.iter_lines()):
            seen.setdefault(name, data)
            if {"informed", "candles"} <= set(seen):  # both come with the first push
                break
    assert seen["informed"]["line"] == "Informed traders: none in this model" and seen["informed"]["share"] is None
    assert seen["candles"] and all("fair" not in bar for bar in seen["candles"])


def test_a_second_tab_of_the_session_sees_the_switch_as_the_first_set_it(client, manager):
    client.post("/api/sim/model", data={"model": KEY})
    client.post("/api/chart/reveal", data={"on": "true"})
    other = TestClient(app)
    other.cookies.set(COOKIE_NAME, client.cookies[COOKIE_NAME])
    html = other.get("/").text
    assert re.search(r'<button[^>]*id="reveal-toggle"[^>]*aria-pressed="true"', html)
    assert "Informed traders: no market orders yet" in html
