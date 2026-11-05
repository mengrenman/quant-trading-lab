"""Recording a dashboard session and playing it back (app/recording.py, LOBManager in app/manager.py)."""

import asyncio
import io
import json
import zipfile

import numpy as np
import pytest
from starlette.testclient import TestClient

from app import recording as rec
from app.main import app, get_view_context, registry, render_panels
from app.manager import LOBManager
from app.sessions import COOKIE_NAME
from app.strategy_specs import STRATEGY_SPECS
from lob.order import Side


def start(m, kind, **settings):
    spec = STRATEGY_SPECS[kind]
    return m.start_strategy(spec, {**spec.defaults(), **settings})


def state(m):
    """Everything a playback must reproduce: the book (with whose each order is), the accounts, the clock."""
    def owner(oid):
        if oid in m.manual_order_ids:
            return "you"
        return m._strategy_orders[oid].name if oid in m._strategy_orders else "model"

    book = [(side.value, o.price, o.size, o.hidden, o.display_size, owner(o.order_id))
            for side in (Side.BID, Side.ASK) for o in m.book.resting_orders(side)]
    mark = m._mark()
    return {"book": book, "you": m.account.summary(mark), "events": m.event_count, "time": m.sim_time,
            "volume": m.traded_volume, "engine": m.engine_type,
            "strategies": {s.name: (s.account.summary(mark), s.running, s.signal, s.target) for s in m.strategies}}


def trades(m, since=0):
    return {k: list(v[since:]) for k, v in m._trade_log.items() if k not in ("maker_order_id", "taker_order_id")}


@pytest.fixture(scope="module")
def session():
    """A session recorded from the middle of a run, with all that a session can do while it records."""
    m = LOBManager()
    m.seed = 11
    m.reset(100)
    m.step(3000)
    m.submit_limit_order(m.book.best_bid - 2, 40, Side.BID, display_size=10)  # an iceberg, resting as it starts
    m.submit_limit_order(m.book.best_ask + 3, 15, Side.ASK)
    mm = start(m, "market_making")
    start(m, "momentum")
    start(m, "avellaneda_stoikov")
    start(m, "quote_joiner", size=2)  # the null: what resting fills are held against, replayed too
    m.step(400)  # momentum and Avellaneda-Stoikov are still warming up when recording begins
    begin = len(m._trade_log["time"])
    m.start_recording()
    m.step(4000)
    start(m, "queue_imbalance", execution="passive")
    mr = start(m, "mean_reversion")
    start(m, "penny_jumping")
    start(m, "random", switch=0.3, execution="passive")
    m.submit_market_order(30, Side.BID)
    m.step(3000)
    m.toggle_engine()
    start(m, "breakout")
    start(m, "order_flow_imbalance")
    m.step(2000)
    m.stop_strategy(mm)
    m.start_split_order(Side.ASK, 60, 4, 3.0)
    m.step(3000)
    m.resume_strategy(mm)
    mine = [o for o in map(m.book.get_order, m.manual_order_ids) if o is not None]
    m.move_orders(mine[0].side, mine[0].price, mine[0].price + (1 if mine[0].side == Side.ASK else -1))
    m.set_model("hawkes_zi")
    m.step(3000)
    m.remove_strategy(mr)
    m.cancel_all_orders()
    m.step(2000)
    expected, expected_trades = state(m), trades(m, begin)
    recording = m.stop_recording()
    return recording, expected, expected_trades


def played(recording):
    p = LOBManager()
    p.start_playback(recording)
    p.step(10**9)
    return p


def test_a_playback_reproduces_the_recorded_session_exactly(session):
    recording, expected, expected_trades = session
    assert recording.meta["strategies"] == [
        "Market making 1", "Momentum 1", "Avellaneda-Stoikov 1", "Quote joiner 1", "Queue imbalance 1",
        "Mean reversion 1", "Penny jumping 1", "Random 1", "Breakout 1", "Order flow imbalance 1"]
    assert recording.meta["models"] == ["Smith et al. (2003) ZI", "ZI + self-exciting market orders"]
    p = played(recording)
    assert p.playback.ended and p.playback.cursor == len(recording.tape)
    got = state(p)
    for key in expected:
        assert got[key] == expected[key], key
    assert trades(p) == expected_trades and len(expected_trades["time"]) > 1000
    assert sum(expected_trades["yours"]) > 0  # your fills are yours again


def test_a_recording_survives_its_file(session):
    recording, expected, _ = session
    data = rec.to_bytes(recording)
    loaded = rec.from_bytes(data, 9)
    assert (loaded.name, loaded.meta, len(loaded.tape)) == (recording.name, recording.meta, len(recording.tape))
    assert state(played(loaded)) == expected
    # A notebook can read it with numpy alone, without pickle.
    npz = np.load(io.BytesIO(data), allow_pickle=False)
    header = json.loads(str(npz["header"]))
    assert header["format"] == rec.FORMAT and header["kind_names"][0] == "limit"
    assert len(npz["time"]) == len(recording.tape) and (np.diff(npz["time"]) >= 0).all()


def test_seeking_back_and_forth_lands_where_stepping_does(session):
    recording, expected, _ = session
    span = recording.end_time - recording.start_time

    async def seeks():
        m = LOBManager()
        m.start_playback(recording)
        await m.seek_playback(recording.start_time + span / 2)
        middle = m.playback.cursor
        await m.seek_playback(recording.start_time + span / 4)  # back: the tape again from its start
        quarter = m.playback.cursor
        straight = LOBManager()
        straight.start_playback(recording)
        straight._step_playback(quarter, every_event=True)
        assert 0 < quarter < middle and state(m) == state(straight)
        assert recording.tape.time[quarter - 1] <= recording.start_time + span / 4 < recording.tape.time[quarter]
        await m.seek_playback(float(f"{recording.tape.time[-1]:.14g}") - 1e-12)  # the end, as a slider rounds it
        assert m.playback.ended and state(m) == expected
        assert m.seeking is None and not m._quiet

    asyncio.run(seeks())


def test_playback_stops_at_the_end_of_the_tape(session):
    recording, _, _ = session
    m = LOBManager()
    m.start_playback(recording)
    m.is_running = True
    m.step(len(recording.tape) - 1)  # orders and cancellations, so it gets to the end
    assert m.playback.ended and not m.is_running
    assert "The end of Recording 1" in m.notices[-1].text
    events = m.event_count
    m.step(100)
    assert m.event_count == events  # nothing more until a rewind
    m.rewind_playback()
    assert m.playback.cursor == 0 and not m.playback.ended and m.sim_time == recording.start_time


def test_trading_during_a_playback_takes_it_its_own_way_without_breaking_it(session):
    recording, expected, _ = session
    m = LOBManager()
    m.start_playback(recording)
    m.step(2000)
    m.submit_market_order(200, Side.BID, protection=20)  # takes out orders the tape will later cancel
    m.submit_limit_order(m.book.best_bid, 50, Side.BID)
    s = start(m, "queue_imbalance")  # yours, deciding for itself on the recorded market
    m.step(10**9)
    assert m.playback.ended and s.account.fills > 0 and not s.replayed
    assert state(m)["book"] != expected["book"]


def test_exiting_a_playback_carries_on_live(session):
    recording, _, _ = session
    m = LOBManager()
    m.start_playback(recording)
    m.step(6000)
    replayed = [s for s in m.strategies if s.replayed]
    assert {s.name for s in replayed} >= {"Market making 1", "Momentum 1", "Avellaneda-Stoikov 1", "Quote joiner 1",
                                          "Queue imbalance 1", "Mean reversion 1", "Penny jumping 1", "Random 1"}
    fills = {s.name: s.account.fills for s in replayed}
    time, events = m.sim_time, m.event_count
    m.exit_playback()
    assert m.playback is None and not any(s.replayed for s in m.strategies)
    assert m.recordings == [] and "takes over the book" in m.notices[-1].text
    m.step(3000)  # the model's events now, and the strategies decide for themselves
    assert m.sim_time > time and m.event_count >= events + 3000
    assert any(s.account.fills > fills[s.name] for s in m.strategies)


def test_strategies_started_during_a_playback_start_over_when_it_rewinds(session):
    recording, _, _ = session
    m = LOBManager()
    m.start_playback(recording)
    s = start(m, "momentum")
    m.step(8000)
    assert s.account.fills > 0
    m.rewind_playback()
    again = next(x for x in m.strategies if x.id == s.id)
    assert again.account.fills == 0 and again.strategy.warming_up() == (0, 20) and not again.replayed
    assert [x.name for x in m.strategies if x.replayed] == ["Market making 1", "Momentum 1",  # as it started
                                                             "Avellaneda-Stoikov 1", "Quote joiner 1"]


def test_a_recording_in_the_older_shape_still_plays():
    """Recordings saved before markouts said which side rested, and before the nulls were kept."""
    m = LOBManager()
    m.step(500)
    joiner = start(m, "quote_joiner")
    from lob.order import market_order

    m._process(market_order(m.book.depth_at_price(m.book.best_bid, Side.BID) + 1, m.sim_time, Side.ASK))
    m.start_recording()  # with its fill's markout pending
    m.step(600)
    header, columns = parts(m.stop_recording())
    start_state = header["start"]
    assert start_state["markouts"] and len(start_state["markouts"][0]) == 6
    for key in ("market_markouts", "flow", "regime"):
        start_state.pop(key)
    start_state["markouts"] = [row[:5] for row in start_state["markouts"]]
    for account in [start_state["account"]] + [s["account"] for s in start_state["strategies"]]:
        for key in ("null_lots", "null_markout", "null_edge"):
            account.pop(key)
    old = rec.from_bytes(npz(header, columns), 5)
    p = LOBManager()
    p.start_playback(old)
    p.step(10**9)
    assert p.playback.ended and p.strategies[0].account.fills == joiner.account.fills


def test_a_reset_keeps_a_recording_in_progress_and_ends_a_playback():
    m = LOBManager()
    m.start_recording()
    m.step(500)
    m.reset()
    assert m.recorder is None and len(m.recordings) == 1 and "because the book was reset" in m.notices[-1].text
    m.start_playback(m.recordings[0])
    m.reset()
    assert m.playback is None and m.sim_time == 0 and m.event_count == 0


def test_a_recording_stops_by_itself_when_full(monkeypatch):
    monkeypatch.setattr(rec, "MAX_EVENTS", 300)
    m = LOBManager()
    m.start_recording()
    m.step(1000)
    assert m.recorder is None and len(m.recordings[0].tape) == 300
    assert "because it reached 300 events" in m.notices[-1].text


def test_recording_a_playback_records_what_it_replays(session):
    recording, expected, _ = session
    m = LOBManager()
    m.start_playback(recording)
    m.start_recording()  # from the start of the playback
    m.step(10**9)
    again = m.stop_recording()
    assert state(m) == expected and state(played(again)) == expected


def test_at_most_a_few_recordings_are_kept():
    m = LOBManager()
    for _ in range(rec.MAX_RECORDINGS):
        m.start_recording()
        m.step(50)
        m.stop_recording()
    with pytest.raises(RuntimeError, match="download or delete one first"):
        m.start_recording()
    with pytest.raises(RuntimeError, match="delete one first"):
        m.add_recording(rec.to_bytes(m.recordings[0]))
    m.delete_recording(m.recordings[0])
    m.add_recording(rec.to_bytes(m.recordings[0]))
    assert [r.name for r in m.recordings] == ["Recording 2", "Recording 3", "Recording 2 (2)"]


# ---- Files from elsewhere ----

def small_recording():
    m = LOBManager()
    m.start_recording()
    m.step(300)
    start(m, "market_making")
    m.step(300)
    return m.stop_recording()


def npz(header, columns):
    buf = io.BytesIO()
    np.savez_compressed(buf, header=np.array(json.dumps(header)), **columns)
    return buf.getvalue()


def parts(recording):
    data = np.load(io.BytesIO(rec.to_bytes(recording)), allow_pickle=False)
    return json.loads(str(data["header"])), {name: data[name].copy() for name, _, _ in rec.COLUMNS}


def corrupt(change):
    header, columns = parts(small_recording())
    change(header, columns)
    return npz(header, columns)


def drop_column(header, columns):
    del columns["price"]


def set_start(key, value):
    return lambda header, columns: header["start"].__setitem__(key, value)


def first_of(kind):
    return lambda columns: int(np.flatnonzero(columns["kind"] == kind)[0])


@pytest.mark.parametrize("data, message", [
    (b"not a zip file at all", "not an .npz file"),
    (npz({"format": "something else"}, {}), "no kind"),
    (corrupt(lambda h, c: h.__setitem__("format", "spreadsheet")), "not a recording from this simulator"),
    (corrupt(lambda h, c: h.__setitem__("version", 99)), "version 99"),
    (corrupt(drop_column), "no price"),
    (corrupt(lambda h, c: c.__setitem__("time", c["time"][::-1].copy())), "times go backward"),
    (corrupt(lambda h, c: c["kind"].__setitem__(0, 42)), "kind this simulator does not know"),
    (corrupt(lambda h, c: c["size"].__setitem__(first_of(rec.LIMIT)(c), 0)), "of no size"),
    (corrupt(lambda h, c: c["origin"].__setitem__(first_of(rec.LIMIT)(c), 77)), "never started"),
    (corrupt(lambda h, c: c["oid"].__setitem__(first_of(rec.DECIDE)(c), 55)), "before it started"),
    (corrupt(lambda h, c: c["time"].__setitem__(slice(None), c["time"] * np.nan)), "times go backward"),
    (corrupt(lambda h, c: c.__setitem__("size", c["size"][:-1])), "as long as the others"),
    (corrupt(set_start("engine", "auction")), "engine"),
    (corrupt(set_start("model", "oracle")), "model"),
    (corrupt(lambda h, c: h["start"]["orders"].append(list(h["start"]["orders"][0]))), "an order twice"),
    (corrupt(lambda h, c: h["start"]["orders"][0].__setitem__(2, h["start"]["orders"][-1][2] + 5)),
     "price order"),
    (corrupt(lambda h, c: h["extras"][next(iter(h["extras"]))]["values"].__setitem__("size", 10**9)),
     "must be between"),
    (corrupt(lambda h, c: h["start"]["account"].pop("cash")), "account is not complete"),
    (corrupt(lambda h, c: h.__setitem__("end_time", -1.0)), "end time"),
])
def test_files_that_are_not_recordings_are_refused_saying_why(data, message):
    with pytest.raises(ValueError, match=message):
        rec.from_bytes(data, 1)


def test_a_recording_too_long_or_too_large_is_refused(monkeypatch):
    data = rec.to_bytes(small_recording())
    monkeypatch.setattr(rec, "MAX_EVENTS", 100)
    with pytest.raises(ValueError, match="longer than 100 events"):
        rec.from_bytes(data, 1)
    monkeypatch.setattr(rec, "MAX_FILE_BYTES", 1000)
    with pytest.raises(ValueError, match="larger than"):
        rec.from_bytes(data, 1)


def test_a_zip_member_declaring_a_huge_size_is_not_read():
    header, columns = parts(small_recording())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, column in {"header": np.array(json.dumps(header)), **columns}.items():
            payload = io.BytesIO()
            np.save(payload, column if name != "time" else np.zeros(rec.MAX_EVENTS + 10))
            z.writestr(f"{name}.npy", payload.getvalue())
    with pytest.raises(ValueError, match="longer than"):
        rec.from_bytes(buf.getvalue(), 1)


# ---- The endpoints ----

@pytest.fixture
def client():
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def manager(client):
    return registry.get(client.cookies[COOKIE_NAME])


def test_recording_from_the_page(client, manager):
    assert client.post("/api/recording/stop").status_code == 409
    assert client.post("/api/recording/start").status_code == 204
    assert client.post("/api/recording/start").status_code == 409
    client.post("/api/sim/control", data={"action": "step", "n": 400})
    client.post("/api/order/limit", data={"price": 98, "size": 7, "side": "bid"})
    panels = render_panels(get_view_context(manager))
    assert 'data-recording="true"' in panels["sim_status"] and "REC " in panels["sim_status"]
    res = client.post("/api/recording/stop")
    assert res.status_code == 200 and res.json()["name"] == "Recording 1"
    rid = res.json()["id"]
    panels = render_panels(get_view_context(manager))
    assert f'data-recording-id="{rid}"' in panels["recordings"] and "1 of yours" in panels["recordings"]

    file = client.get(f"/api/recordings/{rid}/download")
    assert file.status_code == 200
    assert file.headers["content-disposition"] == 'attachment; filename="lob-recording-1.npz"'
    assert client.get("/api/recordings/99/download").status_code == 404
    res = client.post("/api/recordings/upload", files={"file": ("mine.npz", file.content)})
    assert res.status_code == 200 and res.json()["name"] == "Recording 1 (2)"
    res = client.post("/api/recordings/upload", files={"file": ("x.npz", b"PK nonsense")})
    assert res.status_code == 422 and res.json()["detail"].startswith("Not a recording this simulator can play")

def test_an_upload_too_large_is_refused_before_it_is_read(client, monkeypatch):
    monkeypatch.setattr(rec, "MAX_FILE_BYTES", 1000)
    res = client.post("/api/recordings/upload", files={"file": ("x.npz", b"x" * 2000)})
    assert res.status_code == 413


def test_playing_back_from_the_page(client, manager):
    client.post("/api/recording/start")
    client.post("/api/sim/control", data={"action": "step", "n": 600})
    rid = client.post("/api/recording/stop").json()["id"]
    assert client.post("/api/playback/seek", data={"time": 1}).status_code == 409  # nothing playing back
    assert client.post("/api/playback/start", data={"recording_id": 99}).status_code == 404
    assert client.post("/api/playback/start", data={"recording_id": rid}).status_code == 204
    assert manager.playback is not None and manager.sim_time == 0
    panels = render_panels(get_view_context(manager))
    assert 'data-playback="true"' in panels["sim_status"] and 'data-pb-name="Recording 1"' in panels["sim_status"]
    assert "playing back" in panels["recordings"]
    client.post("/api/sim/control", data={"action": "step", "n": 100})
    assert manager.playback.cursor >= 100
    end = manager.playback.recording.end_time
    assert client.post("/api/playback/seek", data={"time": end}).status_code == 204
    assert manager.playback.ended
    assert client.post("/api/playback/seek", data={"time": "nan"}).status_code == 422
    assert client.post("/api/sim/control", data={"action": "reset"}).status_code == 200  # rewinds
    assert manager.playback.cursor == 0 and not manager.playback.ended
    assert client.post(f"/api/recordings/{rid}/delete").status_code == 409  # playing it back
    assert client.post("/api/playback/exit").status_code == 204 and manager.playback is None
    assert client.post("/api/playback/exit").status_code == 204  # nothing to exit: no harm
    assert client.post(f"/api/recordings/{rid}/delete").status_code == 204 and manager.recordings == []
    assert client.post(f"/api/recordings/{rid}/delete").status_code == 404
