"""The studies in a dashboard session: the board (app/studies.py), their hooks in LOBManager, the footprint, the tape.

Each rule is held to a computation made here from the manager's own record of what happened, not by the code under
test: the quotes before an event are compared with the book itself, the footprint with the trade log, the tape with
the trade log, CVD with the signed lots of the trades, and a playback's board with the live board it was recorded from.
"""

import asyncio
import json
import math
import signal

import pytest
from starlette.testclient import TestClient

from app import manager as manager_module
from app import recording as rec
from app.main import app, registry
from app.manager import LOBManager
from app.model_specs import Param
from app.sessions import COOKIE_NAME, SessionRegistry
from app.strategy_specs import STRATEGY_SPECS
from app.studies import SAMPLES, STUDY_LABELS, STUDY_SPECS, StudyBoard, plain
from lob.order import Side, cancel_order
from lob.studies import ARRIVAL_LIMIT, ARRIVAL_MARKET, CANCEL, VPIN, OFI
from tests.test_recording import run_session, start
from tests.test_studies import make_book

BID, ASK = Side.BID, Side.ASK


def board_manager(**settings):
    m = LOBManager()
    m.attach_studies(StudyBoard())
    for key, values in settings.items():
        m.configure_study(key, values)
    return m


def quotes_of(book):
    """(best bid, lots there, best ask, lots there), read from the book as a trader would."""
    bid, ask = book.best_bid, book.best_ask
    return (bid, book.depth_at_price(bid, BID) if bid is not None else 0,
            ask, book.depth_at_price(ask, ASK) if ask is not None else 0)


class Tap:
    """Watches a manager: every event its board is given, and that its quotes before are the book's before it.

    Wraps ``_process`` and ``_cancel_resting`` (checking the board's quotes against the book as each is entered,
    while the board is on) and the board's ``observe`` (keeping the event, checking the quotes after).
    """

    def __init__(self, m):
        self.m = m
        self.events = []  # (t, kind, side, price, size, ((price, size), ...), aggressor, own)
        self.checked = 0
        self.entry = None
        board = m.studies
        observe = board.observe

        def tapped(t, kind, side, price, size, fills, aggressor, before, after, book, own=False):
            assert before == self.entry, "the quotes before are not the book's before the event"
            assert after == quotes_of(book), "the quotes after are not the book's after the event"
            self.events.append((t, kind, side, price, size, tuple((f.price, f.size) for f in fills), aggressor, own))
            observe(t, kind, side, price, size, fills, aggressor, before, after, book, own)

        board.observe = tapped
        for name in ("_process", "_cancel_resting"):
            setattr(m, name, self._wrap(getattr(m, name)))

    def _wrap(self, call):
        def wrapped(*args, **kwargs):
            m = self.m
            self.entry = quotes_of(m.book)
            if m.studies is not None and not m.studies.suspended and m.collecting is None:
                assert m.studies.prev == self.entry, "the board's quotes are stale"
                self.checked += 1
            return call(*args, **kwargs)
        return wrapped


# ---- The board ----

def test_a_new_board_is_at_unit_zero_with_no_quotes_and_takes_no_sample_of_its_own():
    board = StudyBoard()
    assert (board.unit, board.prev, board.version, board.suspended, board.failed) == (0, None, 0, False, None)
    book = make_book([(100, 10), (99, 5)], [(102, 4), (103, 6)])
    quotes = quotes_of(book)
    board.observe(2.5, ARRIVAL_LIMIT, BID, 99, 1, (), 1, quotes, quotes, book)  # the manager samples, not the studies
    assert board.set.unit == 0 and not board.samples


def test_the_board_keeps_the_last_600_units_of_samples_and_samples_at_most_600_after_a_jump():
    book = make_book([(100, 10), (99, 5)], [(102, 4), (103, 6)])
    board = StudyBoard()
    board.sample_to(1500, book)
    units = [row[0] for row in board.samples]
    assert units == list(range(901, 1501)) and SAMPLES == 600 and board.unit == 1500
    assert all(row[1] == 101.0 for row in board.samples)
    board.sample_to(1503, book)
    assert [row[0] for row in board.samples] == list(range(904, 1504))


def test_configuring_a_study_leaves_it_switched_as_it_was_and_a_reset_without_a_time_starts_at_the_unit():
    board = StudyBoard()
    assert board.configure("ofi", {"window": 3.0}) is True
    assert board["ofi"].enabled and board["ofi"].window == 3.0
    assert board.configure("ofi", {"window": 3.0}, enabled=False) is True and not board["ofi"].enabled
    assert board.configure("ofi", {}) is False and not board["ofi"].enabled
    board.reset(7)
    assert board.unit == 7 and board["ofi"].start == 7.0 and not board["ofi"].enabled


def test_the_number_inputs_step_by_a_whole_number_a_twentieth_a_half_or_one_by_the_range():
    steps = {(key, spec.name): spec.step
             for key, specs in STUDY_SPECS.items() for spec in specs if isinstance(spec, Param)}
    assert steps == {("ofi", "window"): 0.5, ("delta", "window"): 0.5, ("delta", "quote_lag"): 1,
                     ("vpin", "buckets"): 1, ("vpin", "volume"): 1.0, ("vpin", "quote_lag"): 1,
                     ("otr", "window"): 0.5, ("imbalance", "levels"): 1, ("spread", "shock"): 0.5,
                     ("resilience", "ticks"): 1, ("resilience", "depleted"): 0.05, ("resilience", "recovered"): 0.05,
                     ("resilience", "restore_window"): 1.0}


def test_the_seek_lookback_and_the_footprint_limits_are_the_ones_the_manager_documents():
    assert manager_module.SEEK_STUDY_LOOKBACK == 40_000
    assert (manager_module.FOOTPRINT_UNITS, manager_module.FOOTPRINT_BARS) == (12_000, 300)


def test_every_study_has_its_settings_as_specs_the_forms_can_use():
    board = StudyBoard()
    assert list(STUDY_SPECS) == board.keys == list(STUDY_LABELS)
    for key, specs in STUDY_SPECS.items():
        assert [spec.name for spec in specs] == [setting.name for setting in board[key].SETTINGS]
        for spec in specs:  # the default is a valid value of the spec, and is what the study starts with
            assert spec.parse(spec.default) == spec.default == board[key].settings[spec.name]
    assert STUDY_SPECS["cvd"] == ()
    side = next(spec for spec in STUDY_SPECS["vpin"] if spec.name == "side")
    assert [value for value, _ in side.options] == ["true", "tick", "mid", "bvc"]
    window = next(spec for spec in STUDY_SPECS["ofi"] if spec.name == "window")
    assert (window.low, window.high, window.integer) == (0.5, 100.0, False)
    levels = next(spec for spec in STUDY_SPECS["imbalance"] if spec.name == "levels")
    assert (levels.low, levels.high, levels.integer, levels.step) == (1, 20, True, 1)


def test_settings_are_checked_against_the_specs_and_a_message_comes_for_each_bad_field():
    board = StudyBoard()
    values, errors = board.parse("vpin", {"buckets": "20", "side": "tick", "volume": "250", "junk": "x"})
    assert values == {"buckets": 20, "side": "tick", "volume": 250.0} and errors == {}  # unknown fields are left alone
    values, errors = board.parse("vpin", {"buckets": "2.5", "side": "sideways", "quote_lag": "-1"})
    assert values == {} and set(errors) == {"buckets", "side", "quote_lag"}
    assert errors["buckets"] == "Buckets (N): enter a whole number"
    assert errors["side"].startswith("Trade sides: choose")
    assert errors["quote_lag"] == "Quote lag (units; mid rule): must be between 0 and 100"
    assert board.parse("ofi", {"window": "nan"})[1] == {"window": "Window (time units): enter a finite number"}
    assert board.parse("ofi", {})[0] == {}  # only the fields given change


def test_settings_that_contradict_each_other_change_nothing():
    board = StudyBoard()
    with pytest.raises(ValueError, match="above Depleted below"):
        board.configure("resilience", {"depleted": 0.8, "recovered": 0.6})
    assert board["resilience"].settings["depleted"] == 0.5
    with pytest.raises(ValueError, match="at least 1"):
        board.configure("vpin", {"volume": 0.5})
    assert board["vpin"].settings["volume"] == 0


def test_a_value_json_cannot_hold_becomes_none_and_a_wrong_kind_is_an_error():
    assert plain(float("nan")) is None and plain(math.inf) is None and plain(-math.inf) is None
    assert plain(None) is None and plain(True) is True and plain(3) == 3 and plain(0.25) == 0.25
    np = pytest.importorskip("numpy")
    assert plain(np.float64("nan")) is None and plain(np.int64(4)) == 4 and isinstance(plain(np.int64(4)), int)
    assert type(plain(np.float32(0.5))) is float
    with pytest.raises(TypeError):
        plain([1])


def test_a_restart_keeps_what_was_not_learned_from_events_and_a_reset_keeps_nothing():
    m = board_manager()
    m.step(6000)
    board = m.studies
    cvd, lots = board.cvd.value(), board.volume
    assert cvd != 0 and lots is not None and board.samples
    board.restart(math.floor(m.sim_time), m.sim_time)
    assert board.cvd.value() == cvd and board.volume == lots and not board.samples
    assert board.set.values(m.sim_time)["ofi"]["ofi"] is None  # warming up
    board.reset(math.floor(m.sim_time), m.sim_time)
    assert board.cvd.value() == 0 and board.volume is None and board.unit == math.floor(m.sim_time)


def test_a_snapshot_is_made_of_copies_and_changes_nothing_in_the_board():
    m = board_manager()
    m.step(5000)
    snap = m.studies_snapshot()
    assert snap["samples"] and snap["samples"][-1][0] == snap["unit"] == math.floor(m.sim_time)
    row = snap["samples"][0]
    row[2] = "scribble"
    snap["settings"]["ofi"]["window"] = 99
    snap["values"]["ofi"]["ofi"] = 99
    again = m.studies_snapshot()
    assert again["samples"][0][2] != "scribble" and again["settings"]["ofi"]["window"] == 5.0
    assert again["values"]["ofi"]["ofi"] != 99
    json.dumps(again, allow_nan=False)


def test_a_session_without_a_board_says_so_in_the_same_shape():
    m = LOBManager()
    snap = m.studies_snapshot()
    assert snap["settings"] == {} and snap["values"] == {} and snap["samples"] == [] and snap["unit"] is None
    assert snap["version"] == m.studies_version and snap["failed"] is None
    with pytest.raises(LookupError):
        m.configure_study("ofi", {"window": 3})


# ---- What reaches the studies: the hook, the quotes before, the clock ----

def test_a_session_without_a_board_is_not_suspended_and_collects_and_seeks_as_any_other(lived, monkeypatch):
    _, recording, _ = lived
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 3000)
    m = LOBManager()
    assert m.studies_snapshot()["suspended"] is False
    version = m.studies_version
    assert asyncio.run(m.collect(2000, chunk=500)) is True
    assert m.studies is None and m.studies_version == version + 1  # pages reload, as after any collection
    m.start_playback(recording)
    steps, real = [], m._step_playback
    m._step_playback = lambda n, every_event=False: (steps.append(n), real(n, every_event=every_event))[1]
    asyncio.run(m.seek_playback(recording.end_time))
    target, expected = len(recording.tape), []
    for cursor in range(0, target, 5000):
        expected.append(min(5000, target - cursor))
    assert m.playback.cursor == target and steps == expected  # no studies to stop at the lookback: plain chunks


def test_an_arrival_on_an_empty_side_reaches_the_studies_and_does_not_stop_them():
    m = board_manager()
    m.step(500)
    for order in list(m.book.resting_orders(BID)):
        m.book.cancel_order(order.order_id)
    m._resync_studies()
    assert m.book.best_bid is None and m.studies.prev[0] is None
    price = m.book.best_ask - 10
    m.submit_limit_order(price, 5, BID)
    assert m.studies.failed is None and m.book.best_bid == price
    ask = m.book.best_ask
    assert m.studies.prev == quotes_of(m.book) == (price, 5, ask, m.book.depth_at_price(ask, ASK))


def test_the_board_belongs_to_a_session_and_not_to_a_scratch_manager():
    assert LOBManager().studies is None
    reg = SessionRegistry()
    _, m, created = reg.get_or_create(None)
    assert created and isinstance(m.studies, StudyBoard) and m.studies.version == m.studies_version
    assert m.studies.prev == quotes_of(m.book)
    _, other, _ = reg.get_or_create(None)
    assert other.studies is not m.studies


def test_fits_and_training_tapes_never_give_a_study_an_event(monkeypatch):
    m = board_manager()
    m.step(3000)
    start(m, "market_making")
    calls = []
    real = LOBManager._observe
    monkeypatch.setattr(LOBManager, "_observe", lambda self, *a: calls.append(1) or real(self, *a))

    async def run():
        training = await m.training_tape(3000, seed=4)
        spec = STRATEGY_SPECS["market_making"]
        await m.fit_strategy(spec, {**spec.defaults()}, {"half_spread": [1, 2]}, training, "a test", "pnl")
        return training
    training = asyncio.run(run())
    assert len(training.tape) > 2000 and calls == []  # the scratch managers ran thousands of events, unwatched
    m.step(100)
    assert len(calls) > 0  # the session's own run is watched


def test_each_event_is_given_to_the_studies_as_it_was_sent_not_as_the_engine_left_it():
    m = board_manager()
    tap = Tap(m)
    m.step(2000)
    assert m.event_count == len(tap.events) > 1500  # every event once, none lost, none twice
    assert {e[1] for e in tap.events} == {ARRIVAL_LIMIT, ARRIVAL_MARKET, CANCEL} and not any(e[7] for e in tap.events)
    assert all(e[6] == (0 if e[1] == CANCEL else 1 if e[2] is BID else -1) for e in tap.events)  # a cancel has none
    n = len(tap.events)

    iceberg, _ = m.submit_limit_order(m.book.best_bid - 2, 40, BID, display_size=10)
    t, kind, side, price, size, fills, aggressor, own = tap.events[n]
    assert (kind, side, price, size, fills, aggressor, own) == (ARRIVAL_LIMIT, BID, iceberg.price, 40, (), 1, True)
    assert m.book.get_order(iceberg.order_id).size == 10  # the engine has rewritten the order's size

    _, got = m.submit_market_order(30, BID)  # a user's market order is a limit order at a protected price
    t, kind, side, price, size, fills, aggressor, own = tap.events[n + 1]
    assert (kind, side, size, aggressor, own) == (ARRIVAL_LIMIT, BID, 30, 1, True)
    assert fills == tuple((f.price, f.size) for f in got) and sum(s for _, s in fills) > 0

    m.cancel_order_by_id(iceberg.order_id)
    t, kind, side, price, size, fills, aggressor, own = tap.events[-1]
    assert (kind, side, price, size, fills, aggressor, own) == (CANCEL, BID, iceberg.price, 10, (), 0, True)

    before = (m.event_count, len(tap.events))
    m._process(cancel_order(10**9, m.sim_time))  # a cancel of an order that is not there changes nothing
    assert (m.event_count, len(tap.events)) == (before[0] + 1, before[1])
    assert tap.checked > 1500


def test_the_quotes_before_an_event_are_always_the_book_before_it(monkeypatch):
    """Through everything that changes a book or a session: the cache of the board is never stale."""
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 2000)
    m = LOBManager()
    m.attach_studies(StudyBoard())
    tap = Tap(m)
    recording, _, _ = run_session(m)  # an iceberg, strategies, a market order, an engine toggle, a model change,
    assert tap.checked > 22_000       # move_orders, cancel_all, a TWAP, and a recording
    live = tap.checked
    m.reset(100)
    m.step(1500)
    start(m, "market_making")
    m.step(1500)
    m.start_playback(recording)
    m.step(4000)
    m.rewind_playback()
    m.step(1500)
    m.submit_market_order(40, ASK)  # the user acts in a playback
    m.step(1500)
    m.submit_limit_order(m.book.best_bid, 25, BID)
    m.step(500)
    asyncio.run(m.seek_playback(recording.end_time))  # the studies are off for a while, and then on again
    assert m.playback.ended
    m.exit_playback()
    m.set_model("parlour")
    m.step(2500)
    asyncio.run(m.collect(6000, chunk=2000))
    m.step(2500)
    assert tap.checked > live + 15_000


def test_the_unit_samples_have_a_row_for_every_unit_the_clock_passes_even_when_it_jumps():
    m = board_manager()
    m.step(3000)
    board = m.studies
    last = board.unit
    assert last == math.floor(m.sim_time) and [row[0] for row in board.samples] == list(range(1, last + 1))
    m._advance(m.sim_time + 40.5)  # nothing happens in between: every unit gets a row, of the same book
    units = [row[0] for row in board.samples]
    assert units == list(range(1, math.floor(m.sim_time) + 1))
    jumped = [row for row in board.samples if row[0] > last]
    assert len(jumped) >= 40 and len({tuple(row[1:]) for row in jumped}) < len(jumped)  # the same values, repeated
    m._advance(m.sim_time + 5000)  # a jump past 600 units keeps the last 600 only
    assert len(board.samples) == SAMPLES and board.samples[-1][0] == math.floor(m.sim_time)
    assert [row[0] for row in board.samples] == list(range(math.floor(m.sim_time) - SAMPLES + 1,
                                                            math.floor(m.sim_time) + 1))


def test_a_sample_holds_the_mid_and_each_studys_main_value_from_the_book_before_the_event():
    m = board_manager()
    m.step(2500)
    board = m.studies
    seen = []
    real = m._advance

    def advance(t):  # what the book is just as the clock moves to a new unit
        if math.floor(t) > board.unit:
            seen.append((math.floor(t), m.book.mid_price, quotes_of(m.book), m.book.spread))
        real(t)
    m._advance = advance
    m.step(1500)
    rows = {row[0]: row for row in board.samples}
    assert seen
    columns = {key: 2 + i for i, key in enumerate(board.keys)}
    for unit, mid, quotes, spread in seen:
        row = rows[unit]
        assert row[1] == mid
        assert row[columns["spread"]] == (None if spread is None else spread // m.book.tick_size)
        assert row[columns["imbalance"]] is not None or quotes[0] is None or quotes[2] is None


def test_the_clock_that_moves_without_an_event_still_drains_a_window():
    m = board_manager()
    m.step(3000)
    t = m.sim_time
    snap = m.studies_snapshot()
    assert snap["values"]["ofi"]["ofi"] is not None
    m._advance(t + 6.0)  # longer than OFI's window of 5 units, shorter than OTR's of 10
    assert m.studies_snapshot()["values"]["ofi"]["ofi"] == 0
    assert m.studies_snapshot()["values"]["otr"]["messages"] > 0
    m._advance(t + 12.0)  # and now longer than that
    values = m.studies_snapshot()["values"]
    assert values["otr"]["messages"] == 0 and values["otr"]["otr"] is None  # no trades to divide by: undefined


def test_otr_counts_the_models_orders_only_and_everyones_by_the_setting():
    m = board_manager()
    tap = Tap(m)
    m.step(2500)
    start(m, "market_making")
    start(m, "quote_joiner", size=2)
    m.submit_limit_order(m.book.best_bid - 1, 5, BID)
    m.step(3000)
    now = m.sim_time
    assert any(e[7] for e in tap.events) and not all(e[7] for e in tap.events)

    def by_hand(scope):
        window = [e for e in tap.events if now - 10 < e[0] <= now and (scope == "everyone" or not e[7])]
        arrived = sum(e[4] for e in window if e[1] != CANCEL)
        traded = sum(s for e in window for _, s in e[5])
        trades = sum(1 for e in window if e[5])
        return {"messages": len(window), "trades": trades, "otr": len(window) / trades,
                "tto": traded / arrived}
    for scope in ("model", "everyone"):
        m.configure_study("otr", {"scope": scope})
        # The new setting starts afresh: replay the window by stepping on, then compare at the end.
        m.step(4000)
        now = m.sim_time
        values = m.studies.set.values(now)["otr"]
        want = by_hand(scope)
        assert {k: values[k] for k in want} == pytest.approx(want), scope
    assert by_hand("model")["messages"] != by_hand("everyone")["messages"]


# ---- A board in a playback equals the board it was recorded from ----

@pytest.fixture(scope="module")
def lived():
    """The session of tests/test_recording.py lived on a board, with the board's state at two points in it."""
    m = board_manager(vpin={"buckets": 5})
    marks, shadow = {}, {}

    def read(name, m):
        board, t = m.studies, m.sim_time
        marks[name] = {"t": t, "values": board.set.values(t), "cvd": board.cvd.value(), "volume": board.volume,
                       "cursor": len(m.recorder.tape) if m.recorder is not None else None, "unit": board.unit}

    def mark(name, m):
        read(name, m)
        board = m.studies
        if name == "start":  # a VPIN that begins here with the V this board has, and follows what the board follows
            vpin = VPIN(log_buckets=True, buckets=5)
            vpin.reset(board.unit, m.sim_time)
            vpin.set_volume(board.volume)
            observe, recalibrate = board.observe, board.recalibrate

            def observed(*args):
                observe(*args)
                vpin.observe(*args)

            def recalibrated():
                recalibrate()
                vpin.recalibrate()
            board.observe, board.recalibrate = observed, recalibrated
            shadow["vpin"] = vpin
        else:
            marks[name]["vpin"] = (shadow["vpin"].values(), list(shadow["vpin"].log))
    result = run_session(m, mark)
    read("end", m)
    vpin = shadow["vpin"]
    marks["end"]["vpin"] = (vpin.values(), list(vpin.log))
    return m, result[0], marks


def play_board(recording, **settings):
    p = board_manager(vpin={"buckets": 5}, **settings)
    p.studies["vpin"].log = []
    p.start_playback(recording)
    return p


def windows_agree(live, p, t):
    """The windowed studies of a playback's board against the live board's, read at the same time."""
    got = p.studies.set.values(t)
    for key, names in (("ofi", ("ofi",)), ("delta", ("delta",)), ("otr", ("otr", "tto", "messages", "trades"))):
        for name in names:
            assert got[key][name] == live["values"][key][name], (key, name)
    assert got["ofi"]["ofi"] is not None and got["delta"]["delta"] is not None and got["otr"]["otr"] is not None


def test_the_board_in_a_playback_equals_the_board_it_was_recorded_from(lived):
    m, recording, marks = lived
    first, segment, end = marks["start"], marks["segment"], marks["end"]
    # V is in the start state, and the playback measures with it.
    assert first["volume"] is not None and recording.start["studies"] == {"vpin_v": first["volume"]}
    p = play_board(recording)
    assert p.studies.volume == first["volume"] and p.studies.cvd.value() == 0

    p._step_playback(segment["cursor"], every_event=True)  # as far as the live run had gone
    windows_agree(segment, p, segment["t"])
    assert p.studies.cvd.value() == segment["cvd"] - first["cvd"] != 0  # CVD counts from the playback's start
    live_vpin, live_log = segment["vpin"]
    assert live_vpin["vpin"] is not None and len(live_log) >= 5  # the check has something to hold
    assert p.studies["vpin"].log == live_log and p.studies["vpin"].values(segment["t"]) == live_vpin
    assert p.studies.volume == first["volume"]

    p._step_playback(10**9)
    assert p.playback.ended and p.sim_time == end["t"]
    windows_agree(end, p, end["t"])
    assert p.studies.cvd.value() == end["cvd"] - first["cvd"]
    live_vpin, live_log = end["vpin"]
    assert live_vpin["vpin"] is not None and len(live_log) >= 5
    assert p.studies["vpin"].log == live_log and p.studies["vpin"].values(end["t"]) == live_vpin
    # The engine toggle and the model change made both set V again, from the same fills.
    assert p.studies.volume == end["volume"] and end["volume"] is not None

    # Depletion: the baselines (the median of the last 100 units' samples) agree. Its episodes need not: one
    # still open when the playback began (or begun at another baseline while it warmed up) shapes the ones after.
    got, live = p.studies.set.values(end["t"])["resilience"], end["values"]["resilience"]
    for name in ("baseline_bid", "baseline_ask"):
        assert got[name] == live[name] is not None, name

    # The samples of every unit after the windows were full.
    columns = {key: 2 + i for i, key in enumerate(p.studies.keys)}
    live_rows, rows = {r[0]: r for r in m.studies.samples}, {r[0]: r for r in p.studies.samples}
    common = [u for u in rows if u >= first["unit"] + 11 and u in live_rows]
    assert len(common) > 200
    for key in ("ofi", "delta", "otr", "imbalance", "spread"):
        mine = [rows[u][columns[key]] for u in common]
        assert mine == [live_rows[u][columns[key]] for u in common], key
        assert sum(v is not None for v in mine) > 0.9 * len(mine)


@pytest.fixture(scope="module")
def cold():
    """A session recorded from its reset, with strategies, the user's orders, an engine toggle and a model change."""
    m = board_manager(vpin={"buckets": 5})
    marks = {}

    def read(name):
        t = m.sim_time
        marks[name] = {"t": t, "values": m.studies.set.values(t), "samples": [list(r) for r in m.studies.samples],
                       "unit": m.studies.unit, "cursor": len(m.recorder.tape) if m.recorder is not None else None}
    m.seed = 5
    m.reset(100)
    m.start_recording()
    m.step(2500)
    m.submit_limit_order(m.book.best_bid - 2, 40, BID, display_size=10)
    start(m, "market_making")
    start(m, "quote_joiner", size=2)
    m.step(2500)
    m.submit_market_order(30, BID)
    m.toggle_engine()
    start(m, "momentum")
    m.step(2500)
    read("middle")
    m.set_model("hawkes_zi")
    m.cancel_all_orders()
    m.step(2500)
    recording = m.stop_recording()
    read("end")
    return recording, marks


def test_a_playback_from_where_the_board_began_equals_it_in_every_study_and_every_sample(cold):
    """With nothing known at the start, a playback's board starts as the live one did, and ends as it did."""
    recording, marks = cold
    assert "studies" not in recording.start  # V was not known: the playback finds it from the same fills
    p = board_manager(vpin={"buckets": 5})
    p.start_playback(recording)
    p._step_playback(marks["middle"]["cursor"], every_event=True)
    middle, event_driven = marks["middle"], ("ofi", "cvd", "delta", "vpin", "otr", "resilience")
    got = p.studies.set.values(middle["t"])
    assert {k: got[k] for k in event_driven} == {k: middle["values"][k] for k in event_driven}
    common = min(p.studies.unit, middle["unit"])
    assert [list(r) for r in p.studies.samples if r[0] <= common] == [r for r in middle["samples"] if r[0] <= common]

    p._step_playback(10**9)
    end = marks["end"]
    assert p.playback.ended and p.sim_time == end["t"] and p.studies.unit == end["unit"]
    assert p.studies.set.values(end["t"]) == end["values"]  # every value of every study, to the last digit
    assert [list(r) for r in p.studies.samples] == end["samples"] and len(end["samples"]) > 100
    values = end["values"]
    assert values["vpin"]["vpin"] is not None and values["resilience"]["started"] > 3 and values["cvd"]["cvd"] != 0
    assert values["otr"]["otr"] is not None and values["vpin"]["volume"] is not None


def test_a_window_is_warm_a_full_window_after_the_start_of_a_playback_not_before_it(lived):
    _, recording, marks = lived
    start_time = recording.start["time"]
    assert start_time != math.floor(start_time)  # it begins part way into a unit
    p = play_board(recording)
    for key, name, window in (("ofi", "ofi", 5.0), ("delta", "delta", 10.0), ("otr", "otr", 10.0)):
        assert p.studies.set[key].settings["window"] == window
        assert p.studies.set.values(start_time + window - 0.01)[key][name] is None, key
    assert p.studies.set.values(start_time + 5.0)["ofi"]["ofi"] == 0  # nothing happened: a full window of nothing
    assert p.studies.unit == math.floor(start_time) and p.studies.cvd.value() == 0


def test_cvd_is_the_signed_lots_of_every_trade_since_the_start(lived):
    m, recording, marks = lived
    frame = m.trades_frame()
    signed = frame["size"].where(frame["aggressor"] == "buy", -frame["size"])
    assert m.studies.cvd.value() == signed.sum() != 0  # the session's own (it ran from the book's reset)
    p = play_board(recording)
    p.step(10**9)
    frame = p.trades_frame()
    assert p.studies.cvd.value() == frame["size"].where(frame["aggressor"] == "buy", -frame["size"]).sum()


def test_the_delta_of_trades_window_is_the_signed_lots_of_the_trades_in_it(lived):
    m, _, marks = lived
    frame, t = m.trades_frame(), marks["end"]["t"]
    inside = frame[(frame["time"] > t - 10) & (frame["time"] <= t)]
    want = inside["size"].where(inside["aggressor"] == "buy", -inside["size"]).sum()
    assert marks["end"]["values"]["delta"]["delta"] == want != 0


# ---- Seeks ----

def straight_events(recording, upto):
    p = play_board(recording)
    seen = []
    real = p._observe
    p._observe = lambda *a: (seen.append(p.playback.cursor), real(*a))[1]
    p._step_playback(upto, every_event=True)
    return p, seen


def test_a_long_seek_keeps_the_studies_off_until_the_lookback_and_then_starts_them_afresh(lived, monkeypatch):
    _, recording, marks = lived
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 3000)
    target = len(recording.tape)
    straight, expected = straight_events(recording, target)

    p = play_board(recording)
    seen, versions = [], []
    real = p._observe
    p._observe = lambda *a: (seen.append(p.playback.cursor), real(*a))[1]
    real_step = p._step_playback
    p._step_playback = lambda n, every_event=False: (versions.append((p.playback.cursor, p.studies.suspended, n)),
                                                     real_step(n, every_event=every_event))[1]
    before = p.studies_version
    asyncio.run(p.seek_playback(recording.end_time))
    resume = target - 3000
    assert p.playback.cursor == target
    assert seen == [c for c in expected if c > resume] and len(seen) > 100  # what a straight run gives, from there
    assert p.studies_version == before + 1 and not p.studies.suspended
    assert [cursor for cursor, suspended, _ in versions if suspended][-1] < resume
    cut = [(cursor, n) for cursor, suspended, n in versions if suspended][-1]
    assert cut[0] + cut[1] == resume  # the last chunk with the studies off ends exactly at the lookback
    assert p.studies.prev == quotes_of(p.book)
    # They start afresh at the lookback, and what they read of the last 3,000 events is what a full run reads.
    assert p.studies.unit == math.floor(p.sim_time) == straight.studies.unit
    got, want = p.studies.set.values(p.sim_time), straight.studies.set.values(p.sim_time)
    for key, names in (("ofi", ("ofi",)), ("delta", ("delta",)), ("otr", ("otr", "tto", "messages", "trades")),
                       ("imbalance", ("imbalance",)), ("spread", ("spread",))):
        for name in names:
            assert got[key][name] == want[key][name], (key, name)
    assert 0 < got["vpin"]["buckets"] < want["vpin"]["buckets"]  # it counts the buckets since the lookback only
    assert p.studies.samples[0][0] > straight.studies.samples[0][0]
    # CVD does not start at the lookback: it counts every trade from the playback's start, as a straight run does.
    assert p.studies.cvd.value() == straight.studies.cvd.value() != 0
    frame = p.trades_frame()
    assert p.studies.cvd.value() == frame["size"].where(frame["aggressor"] == "buy", -frame["size"]).sum()


def test_the_bucket_of_vpin_is_kept_through_the_seeks_restart(lived, monkeypatch):
    _, recording, marks = lived
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 1000)
    p = play_board(recording)
    assert p.studies.volume == marks["start"]["volume"]
    asyncio.run(p.seek_playback(recording.tape.time[marks["segment"]["cursor"] - 1]))  # before the engine changes
    cursor = p.playback.cursor
    assert marks["segment"]["cursor"] <= cursor < marks["segment"]["cursor"] + 20
    assert not {rec.ENGINE, rec.MODEL_CHANGE} & set(recording.tape.kind[:cursor])
    assert p.studies.volume == marks["start"]["volume"] and not p.studies.suspended


def test_a_short_seek_leaves_the_studies_on_and_changes_no_version(lived, monkeypatch):
    _, recording, _ = lived
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 3000)
    p = play_board(recording)
    seen = []
    real = p._observe
    p._observe = lambda *a: (seen.append(p.playback.cursor), real(*a))[1]
    version = p.studies_version
    asyncio.run(p.seek_playback(recording.tape.time[2500]))
    straight, expected = straight_events(recording, p.playback.cursor)
    assert p.studies_version == version and seen == expected and len(seen) > 1000
    assert [r for r in p.studies.samples] == [r for r in straight.studies.samples]
    t = p.sim_time
    assert p.studies.set.values(t) == straight.studies.set.values(t)


def test_a_seek_back_rewinds_the_studies_and_goes_on_from_the_start_of_the_playback(lived, monkeypatch):
    _, recording, _ = lived
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 3000)
    p = play_board(recording)
    asyncio.run(p.seek_playback(recording.end_time))
    version = p.studies_version
    short = recording.tape.time[2000]
    asyncio.run(p.seek_playback(short))  # back: rewound, then forward less than the lookback
    straight, _ = straight_events(recording, p.playback.cursor)
    assert p.studies_version == version + 1  # the rewind's, and no suspension
    assert p.studies.samples and list(p.studies.samples) == list(straight.studies.samples)
    assert p.studies.set.values(p.sim_time) == straight.studies.set.values(p.sim_time)

    asyncio.run(p.seek_playback(recording.end_time))
    version = p.studies_version
    asyncio.run(p.seek_playback(recording.tape.time[15_000]))  # back, and then further than the lookback
    assert p.studies_version == version + 2 and not p.studies.suspended  # the rewind's and the restart's
    assert p.studies.prev == quotes_of(p.book)


def test_a_seek_that_ends_early_gives_the_studies_back(lived, monkeypatch):
    _, recording, _ = lived
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 3000)
    p = play_board(recording)
    real = p._step_playback
    calls = []

    def step(n, every_event=False):
        calls.append(n)
        if len(calls) == 2:
            assert p.studies.suspended
            p.reset(100)  # another request resets the session between two chunks of the seek
            return
        real(n, every_event=every_event)
    p._step_playback = step
    version = p.studies_version
    asyncio.run(p.seek_playback(recording.end_time))
    assert p.playback is None and not p.studies.suspended and not p._quiet
    assert p.studies_version > version and p.studies.prev == quotes_of(p.book)
    p.step(500)  # and they work


# ---- Collecting ----

def test_a_collection_leaves_the_studies_off_and_starts_them_afresh_after_it():
    m = board_manager()
    m.step(6000)
    start(m, "market_making")  # its cancellations and orders go through the manager as the user's do
    start(m, "quote_joiner", size=2)
    m.step(1000)
    board = m.studies
    volume, version, cvd = board.volume, m.studies_version, board.cvd.value()
    calls, units = [], []
    real, real_sample = m._observe, board.sample_to
    m._observe = lambda *a: (calls.append(1), real(*a))[1]
    board.sample_to = lambda *a: (units.append(1), real_sample(*a))[1]
    sampled = len(board.samples)
    asyncio.run(m.collect(20_000, chunk=5000))
    assert calls == [] and units == []  # nothing is given to the studies, and no unit sampled, during it
    assert len(board.samples) == 0 and sampled > 0  # a clean board after
    assert m.studies_version == version + 1 and board.unit == math.floor(m.sim_time)
    assert board.prev == quotes_of(m.book)  # resynced
    frame = m.trades_frame()
    signed = frame["size"].where(frame["aggressor"] == "buy", -frame["size"]).sum()
    assert board.cvd.value() == signed != cvd  # it counted through the collection
    assert board.volume == volume  # the bucket is not set again
    assert m.studies_snapshot()["values"]["ofi"]["ofi"] is None  # warming up
    m.step(3000)
    assert calls and m.studies_snapshot()["values"]["ofi"]["ofi"] is not None


def test_a_collection_that_is_reset_halfway_still_gives_the_studies_back():
    m = board_manager()

    async def run():
        task = asyncio.create_task(m.collect(400_000, chunk=1000))
        await asyncio.sleep(0.05)
        async with m._lock:
            m.reset(100)  # the sample is reset: the collection stops
        return await task
    asyncio.run(run())
    assert m.collecting is None and not m.studies.suspended and m.studies.prev == quotes_of(m.book)
    m.step(2000)
    assert m.studies_snapshot()["values"]["cvd"]["cvd"] != 0


# ---- Versions, settings and failures ----

def test_the_version_changes_exactly_when_a_page_holding_the_studies_is_out_of_date():
    m = board_manager()
    m.step(2000)
    v = m.studies_version
    assert m.studies.version == v

    m.toggle_engine()
    m.set_model("hawkes_zi")
    assert m.configure("hawkes_zi", {**m.model_params["hawkes_zi"], "mu": 7.0}, m.seed) == {}
    m.step(100)
    assert m.studies_version == v  # a new engine or model changes no book and clears nothing a page holds

    m.reset(100)
    assert m.studies_version == v + 1 and m.studies.version == v + 1
    m.step(1000)
    assert m.configure_study("ofi", {"window": 8}) and m.studies_version == v + 2
    assert not m.configure_study("ofi", {"window": 8}) and m.studies_version == v + 2  # no change, no version
    assert m.configure_study("ofi", {}, enabled=False) and m.studies_version == v + 3

    m.step(2000)
    m.start_recording()
    m.step(500)
    recording = m.stop_recording()
    m.start_playback(recording)
    assert m.studies_version == v + 4
    m.step(100)
    m.rewind_playback()
    assert m.studies_version == v + 5
    m.exit_playback()
    assert m.studies_version == v + 5
    asyncio.run(m.collect(10_000))
    assert m.studies_version == v + 6


def test_a_new_engine_or_model_makes_vpin_set_its_bucket_again_from_the_next_200_fills():
    def v_by_hand(m, since):  # 50 times the median size of the first 200 fills after ``since`` fills
        sizes = sorted(m._trade_log["size"][since:since + 200])
        return 50 * (sizes[99] + sizes[100]) / 2

    m = board_manager()
    m.step(6000)
    assert m.studies.volume == v_by_hand(m, 0) == 250.0  # price-time: fills of 5 lots
    seen = len(m._trade_log["size"])
    m.toggle_engine()
    assert m.studies.volume is None and m.studies["vpin"].values(m.sim_time)["buckets"] == 0
    m.step(6000)
    assert m.studies.volume == v_by_hand(m, seen) == 100.0  # pro-rata: fills of 2 lots

    seen = len(m._trade_log["size"])
    m.set_model("hawkes_zi")
    assert m.studies.volume is None
    m.step(6000)
    assert m.studies.volume == v_by_hand(m, seen) is not None
    lots = m.studies.volume
    assert m.configure("smith_zi", {**m.model_params["smith_zi"], "lot_size": 9}, m.seed) == {}  # not the one running
    assert m.studies.volume == lots
    assert m.configure("hawkes_zi", dict(m.model_params["hawkes_zi"]), m.seed) == {}
    assert m.studies.volume is None  # the one running, rebuilt

    m = board_manager(vpin={"volume": 300})  # a number set by the user stays
    m.step(3000)
    m.toggle_engine()
    m.set_model("hawkes_zi")
    assert m.studies.volume == 300.0


def test_a_playback_sets_vpins_bucket_again_where_the_recording_changed_its_engine_or_model(cold):
    recording, marks = cold
    p = board_manager(vpin={"buckets": 5})
    p.start_playback(recording)
    tape = recording.tape
    first = {rec.ENGINE: next(i for i in range(len(tape)) if tape.kind[i] == rec.ENGINE),
             rec.MODEL_CHANGE: next(i for i in range(len(tape)) if tape.kind[i] == rec.MODEL_CHANGE)}
    for kind in (rec.ENGINE, rec.MODEL_CHANGE):
        p._step_playback(first[kind] + 1 - p.playback.cursor, every_event=True)
        assert p.studies.volume is None and p.studies["vpin"].values(p.sim_time)["buckets"] == 0, kind
        p._step_playback(first[kind] + 1 - p.playback.cursor + 1500, every_event=True)
        assert p.studies.volume is not None, kind


def test_a_change_of_settings_clears_that_study_and_leaves_the_others_and_the_settings_survive_a_reset():
    m = board_manager()
    m.step(6000)
    board = m.studies
    columns = {key: 2 + i for i, key in enumerate(board.keys)}
    assert all(row[columns["otr"]] is not None for row in list(board.samples)[-10:])
    assert board.set.values(m.sim_time)["otr"]["otr"] is not None
    before = [row[columns["ofi"]] for row in board.samples]
    assert m.configure_study("otr", {"window": 20.0, "scope": "everyone"})
    assert all(row[columns["otr"]] is None for row in board.samples)  # that study's samples are gone
    assert [row[columns["ofi"]] for row in board.samples] == before  # the others are not touched
    assert board.set.values(m.sim_time)["otr"]["otr"] is None  # warming up again, for 20 units
    snap = m.studies_snapshot()
    assert snap["settings"]["otr"] == {"window": 20.0, "scope": "everyone"} and snap["version"] == m.studies_version
    m.step(8000)
    assert m.studies_snapshot()["values"]["otr"]["otr"] is not None
    m.configure_study("vpin", {"side": "bvc", "buckets": 7}, enabled=False)
    m.reset(100)
    snap = m.studies_snapshot()
    assert snap["settings"]["otr"] == {"window": 20.0, "scope": "everyone"} and not snap["samples"]
    assert snap["settings"]["vpin"]["side"] == "bvc" and snap["enabled"]["vpin"] is False
    assert "vpin" not in snap["values"] and "otr" in snap["values"]  # an off study is not read
    with pytest.raises(ValueError, match="above Depleted below"):
        m.configure_study("resilience", {"depleted": 0.9, "recovered": 0.5})
    with pytest.raises(LookupError):
        m.configure_study("nothing", {})


def test_a_study_that_is_off_is_given_no_events(monkeypatch):
    m = board_manager()
    m.configure_study("otr", {}, enabled=False)
    calls = []
    real = OFI.observe
    monkeypatch.setattr(OFI, "observe", lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    m.studies.set._rebuild()
    m.configure_study("ofi", {}, enabled=False)
    m.step(2000)
    assert calls == [] and len(m.studies.set["otr"]._items) == 0
    m.configure_study("ofi", {}, enabled=True)
    m.step(500)
    assert calls


def test_a_study_that_fails_switches_the_board_off_with_one_notice_and_the_run_goes_on():
    m = board_manager()
    m.step(1000)
    notices = m.notice_seq
    calls = []

    def boom(*args):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("boom")
    m.studies.observe = boom
    events = m.event_count
    m.step(500)
    assert m.event_count - events > 400  # every step ran; none stopped
    assert len(calls) == 3  # the board took no more events once it failed
    assert m.studies.failed == "RuntimeError('boom')" and m.studies.suspended
    assert m.notice_seq == notices + 1 and "switched themselves off" in m.notices[-1].text
    m.step(500)
    asyncio.run(m.collect(5000))
    m.reset(100)
    m.step(500)
    assert m.notice_seq == notices + 1 and m.studies.failed is not None and len(calls) == 3  # and stays off
    snap = m.studies_snapshot()
    assert snap["failed"] == "RuntimeError('boom')" and snap["values"] == {} and snap["suspended"] is True
    json.dumps(snap, allow_nan=False)


def test_a_sample_that_fails_does_the_same_and_so_does_a_read():
    m = board_manager()
    m.step(500)
    notices = m.notice_seq

    def boom(*args, **kwargs):
        raise ValueError("sampling")
    m.studies.set.sample_to = boom
    m.step(2000)
    assert "sampling" in m.studies.failed and m.notice_seq == notices + 1 and m.event_count >= 2500

    other = board_manager()
    other.step(500)

    def read(*args, **kwargs):
        raise ZeroDivisionError("reading")
    other.studies.set.values = read
    snap = other.studies_snapshot()  # the stream asks for this: it must not raise
    assert "reading" in snap["failed"] and snap["values"] == {}
    other.step(200)
    assert other.event_count >= 700


def test_a_playback_seek_does_not_wake_a_failed_board(lived, monkeypatch):
    _, recording, _ = lived
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 3000)
    p = play_board(recording)
    p.studies.fail(RuntimeError("earlier"))
    version = p.studies_version
    asyncio.run(p.seek_playback(recording.end_time))
    assert p.playback.ended and p.studies.suspended and p.studies.failed == "RuntimeError('earlier')"
    assert p.studies_version == version  # nothing was held off or started again


def test_a_board_that_fails_during_a_seek_is_not_woken_when_the_seek_ends(lived, monkeypatch):
    _, recording, _ = lived
    monkeypatch.setattr(manager_module, "SEEK_STUDY_LOOKBACK", 3000)
    p = play_board(recording)
    real = p._step_playback
    calls = []

    def step(n, every_event=False):
        calls.append(n)
        if len(calls) == 2:
            assert p.studies.suspended
            p._study_failed(RuntimeError("while it was off"))  # a page asking for the studies fails, say
        real(n, every_event=every_event)
    p._step_playback = step
    asyncio.run(p.seek_playback(recording.end_time))
    assert p.playback.ended and p.studies.suspended and p.studies.failed == "RuntimeError('while it was off')"
    observed = []
    p._step_playback = real
    p.rewind_playback()
    p._observe = lambda *a: observed.append(1)
    p.step(1000)
    assert observed == []  # still off


def test_the_json_the_studies_give_is_valid_with_an_empty_or_one_sided_book():
    m = board_manager()
    json.dumps(m.studies_snapshot(), allow_nan=False)  # nothing has happened
    m.step(3000)
    for order in list(m.book.resting_orders(ASK)):
        m.book.cancel_order(order.order_id)
    m._resync_studies()
    m._advance(m.sim_time + 3.0)
    snap = m.studies_snapshot()
    assert snap["values"]["imbalance"]["imbalance"] is None and snap["values"]["spread"]["spread"] is None
    json.dumps(snap, allow_nan=False)
    for order in list(m.book.resting_orders(BID)):
        m.book.cancel_order(order.order_id)
    m._resync_studies()
    m._advance(m.sim_time + 3.0)
    json.dumps(m.studies_snapshot(), allow_nan=False)


# ---- The footprint ----

def per_bucket_by_hand(m, n):
    """{bucket time: {price: [sold, bought]}} from the trade log, joining fills to bars as _record_trade does."""
    log = m.trades_frame()
    out, bar = {}, None
    for time, price, size, aggressor in zip(log["time"], log["price"], log["size"], log["aggressor"], strict=True):
        index = math.floor(time)
        if bar is None or bar < index:
            bar = index
        cell = out.setdefault(bar // n * n, {}).setdefault(int(price), [0, 0])
        cell[aggressor == "buy"] += int(size)
    return out


def check_the_cells(m, n):
    m.set_bar_interval(n)
    candles = m.get_candles(footprint=True)
    hand = per_bucket_by_hand(m, n)
    carried = 0
    for c in candles:
        cells = c["cells"]
        if cells is None:
            continue
        carried += 1
        assert [cell[0] for cell in cells] == sorted({cell[0] for cell in cells})  # rising price, each once
        assert sum(cell[2] for cell in cells) == c["buy_volume"] and sum(cell[1] for cell in cells) == c["sell_volume"]
        assert sum(cell[1] + cell[2] for cell in cells) == c["volume"]
        want = hand.get(c["time"], {})
        assert {p: [s, b] for p, s, b in cells} == want, (n, c["time"])
    return candles, carried


@pytest.mark.parametrize("model, engine", [("smith_zi", "price_time"), ("smith_zi", "pro_rata"),
                                           ("parlour", "price_time"), ("parlour", "pro_rata")])
def test_footprint_cells_sum_to_each_candles_volumes_and_equal_the_trade_logs_at_every_interval(model, engine):
    m = LOBManager()
    m.set_model(model)
    m.set_engine(engine)
    m.step(60_000)
    for n in (1, 2, 5, 15, 30, 60):
        candles, carried = check_the_cells(m, n)
        assert carried == min(manager_module.FOOTPRINT_BARS, len(candles))  # the last 300 buckets, as many as there are
        assert all(c["cells"] is None for c in candles[:-manager_module.FOOTPRINT_BARS])
        assert sum(c["volume"] for c in candles) > 0 and any(c["cells"] for c in candles)
        assert all("cells" not in c for c in m.get_candles())  # only when asked for


def test_footprint_cells_hold_in_a_playback_where_fills_are_stamped_before_the_clock(lived):
    _, recording, _ = lived
    p = LOBManager()
    p.start_playback(recording)
    p.step(6000)
    p.submit_market_order(30, BID)  # the user acts: the clock moves on, and orders on the tape are then behind it
    p.step(10**9)
    assert (p.trades_frame()["time"].diff().dropna() < 0).any()  # fills stamped before the latest bar
    for n in (1, 2, 5, 30):
        _, carried = check_the_cells(p, n)
        assert carried > 3


def test_a_bar_lets_go_of_its_cells_after_footprint_units_bars_and_a_bucket_with_one_has_none(monkeypatch):
    monkeypatch.setattr(manager_module, "FOOTPRINT_UNITS", 40)
    m = LOBManager()
    m.step(12_000)
    bars = list(m._base_bars)
    assert len(bars) > 100
    assert all(bar.cells is None for bar in bars[:-40]) and all(bar.cells for bar in bars[-40:])
    for n in (1, 5, 30):
        m.set_bar_interval(n)
        candles = m.get_candles(footprint=True)
        kept = {bar.index // n for bar in bars[-40:]}
        lost = {bar.index // n for bar in bars[:-40]}
        for c in candles:
            bucket = c["time"] // n
            if bucket in lost:
                assert c["cells"] is None, (n, c["time"])  # a bar of it let go of its cells
            elif bucket in kept:
                assert sum(cell[1] + cell[2] for cell in c["cells"]) == c["volume"] > 0
            else:
                assert c["cells"] == [] and c["volume"] == 0  # no trades: nothing to carry


def test_the_candles_carry_cells_for_the_last_footprint_bars_buckets_only(monkeypatch):
    monkeypatch.setattr(manager_module, "FOOTPRINT_BARS", 20)
    m = LOBManager()
    m.step(30_000)
    candles = m.get_candles(footprint=True)
    assert len(candles) > 100
    assert all(c["cells"] is None for c in candles[:-20]) and all(c["cells"] is not None for c in candles[-20:])
    since = candles[-5]["time"]
    assert [c["time"] for c in m.get_candles(since=since, footprint=True)] == [c["time"] for c in candles[-5:]]
    assert m.get_candles(since=since, footprint=True)[0]["cells"] == candles[-5]["cells"]


def test_a_bars_cells_add_up_to_its_volumes():
    m = LOBManager()
    m.step(20_000)
    bar = m._base_bars[-2]
    assert sum(sold + bought for sold, bought in bar.cells.values()) == bar.volume
    assert sum(bought for _, bought in bar.cells.values()) == bar.buy_volume


# ---- The tape ----

def trade_log_rows(m):
    log = m._trade_log
    return list(zip(log["price"], log["size"], log["aggressor"], log["time"], strict=True))


def test_the_tape_numbers_every_fill_once_and_never_again_and_keeps_the_last_2000():
    m = LOBManager()
    m.step(60_000)
    rows = trade_log_rows(m)
    assert m.fill_seq == len(rows) > 3000 and len(m.tape_ring) == 2000
    ring = list(m.tape_ring)
    assert [r[0] for r in ring] == list(range(m.fill_seq - 1999, m.fill_seq + 1))
    for (_, t, price, size, side), (p, s, aggressor, time) in zip(ring, rows[-2000:], strict=True):
        assert (price, size, side) == (p, s, 1 if aggressor == 1 else -1)
        assert t == time  # live, a fill is stamped with the clock
    assert len(m.recent_trades) == 40  # the trades panel's own list stays
    assert [(r.price, r.size) for r in m.recent_trades] == [(r[2], r[3]) for r in reversed(ring[-40:])]

    seq = m.fill_seq
    m.reset(100)
    assert m.fill_seq == seq and not m.tape_ring  # the number goes on, the ring is empty
    m.step(5000)
    assert 0 < len(m.tape_ring) < 2000 and [r[0] for r in m.tape_ring] == list(range(seq + 1, m.fill_seq + 1))


def test_the_tape_is_stamped_with_the_clock_not_with_the_fills_own_timestamp(lived):
    _, recording, _ = lived
    p = LOBManager()
    p.start_playback(recording)
    p.step(6000)
    p.submit_market_order(30, BID)  # the user acts: orders on the tape are then stamped before the clock
    p.step(400)
    ring = list(p.tape_ring)
    log_times = list(p._trade_log["time"])
    assert len(ring) == len(log_times) < 2000
    times = [r[1] for r in ring]
    assert times == sorted(times)  # the clock never goes back
    assert log_times != sorted(log_times) and any(a != b for a, b in zip(times, log_times, strict=True))
    assert all(a >= b for a, b in zip(times, log_times, strict=True))  # a fill's own stamp can only lag the clock
    assert all(t <= p.sim_time for t in times)


def test_the_tape_gives_what_came_after_a_number_and_says_when_to_reload():
    m = LOBManager()
    m.step(40_000)
    last = m.fill_seq
    ring = list(m.tape_ring)
    assert last > 2500
    got = m.tape_since(last - 5)
    assert got["reload"] is False and [f[0] for f in got["fills"]] == list(range(last - 4, last + 1))
    assert got["fills"] == ring[-5:] and got["seq"] == last and got["version"] == m.studies_version
    up_to_date = m.tape_since(last)
    assert up_to_date["reload"] is False and up_to_date["fills"] == []
    first = ring[0][0]
    assert m.tape_since(first - 1)["reload"] is False and len(m.tape_since(first - 1)["fills"]) == 2000
    behind = m.tape_since(first - 2)  # fill first - 1 is gone: the page is missing one
    assert behind["reload"] is True and behind["fills"] == ring
    assert m.tape_since(None)["reload"] is True and m.tape_since(None)["fills"] == ring
    ahead = m.tape_since(last + 10)  # a page from another simulation
    assert ahead["reload"] is True and ahead["fills"] == ring
    version = m.studies_version
    m.reset(100)
    gone = m.tape_since(last)
    assert gone["version"] == version + 1 and gone["fills"] == [] and gone["seq"] == last
    assert json.loads(json.dumps(got))["fills"][0] == list(ring[-5])


def test_an_empty_tape_is_up_to_date_for_a_page_at_zero_and_reloads_any_other():
    m = LOBManager()
    assert m.fill_seq == 0 and not m.tape_ring
    assert m.tape_since(0) == {"version": m.studies_version, "seq": 0, "reload": False, "fills": []}
    assert [m.tape_since(since)["reload"] for since in (None, -1, 1)] == [True, True, True]


# ---- Routes and the stream ----

@pytest.fixture
def client():
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def deadline():
    """A test that reads a stream fails after a minute instead of waiting for an event that is never sent."""
    def expired(signum, frame):
        raise TimeoutError("the stream did not send what the test waits for")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    yield
    signal.alarm(0)


@pytest.fixture
def manager(client):
    return registry.get(client.cookies[COOKIE_NAME])


def strict(response):
    """The body as JSON, refusing NaN and Infinity as a browser's JSON.parse does."""
    def refuse(token):
        raise AssertionError(f"{token} in JSON")
    return json.loads(response.text, parse_constant=refuse)


def test_get_studies_gives_the_snapshot_with_none_as_null_and_the_samples_after_a_unit(client, manager):
    fresh = client.get("/api/studies")
    assert fresh.status_code == 200
    body = strict(fresh)  # a new session: empty lists, not an error
    assert body["samples"] == [] and body["version"] == manager.studies_version and body["failed"] is None
    assert body["values"]["vpin"]["vpin"] is None and body["values"]["ofi"]["ofi"] is None
    assert set(body["values"]) == set(body["settings"]) == set(STUDY_SPECS) and body["keys"] == list(STUDY_SPECS)
    assert body["settings"]["vpin"] == {"buckets": 50, "volume": 0, "side": "true", "quote_lag": 1}

    manager.step(6000)
    body = strict(client.get("/api/studies"))
    assert len(body["samples"]) == body["unit"] == math.floor(manager.sim_time) > 50
    assert all(len(row) == 2 + len(body["keys"]) for row in body["samples"])
    unit = body["unit"]
    assert body["values"]["ofi"]["ofi"] is not None and body["values"]["vpin"]["volume"] == 250.0
    manager.step(1000)
    later = strict(client.get("/api/studies", params={"since": unit}))
    assert [row[0] for row in later["samples"]] == list(range(unit + 1, later["unit"] + 1)) and later["unit"] > unit
    assert strict(client.get("/api/studies", params={"since": later["unit"]}))["samples"] == []
    client.post("/api/sim/control", data={"action": "reset"})
    after = strict(client.get("/api/studies"))
    assert after["samples"] == [] and after["version"] == later["version"] + 1  # 2xx just after a reset


def test_post_studies_config_applies_validates_and_answers_as_the_other_config_routes_do(client, manager):
    manager.step(5000)
    version = manager.studies_version
    ok = client.post("/api/studies/config", data={"study": "vpin", "buckets": "20", "side": "tick"})
    assert ok.status_code == 200 and "applied" in ok.text and "text-emerald-400" in ok.text
    assert manager.studies["vpin"].settings["buckets"] == 20 and manager.studies["vpin"].settings["side"] == "tick"
    assert manager.studies_version == version + 1
    assert strict(client.get("/api/studies"))["settings"]["vpin"]["buckets"] == 20

    same = client.post("/api/studies/config", data={"study": "vpin", "buckets": "20"})
    assert same.status_code == 200 and "no change" in same.text and manager.studies_version == version + 1

    for form, text in (({"study": "vpin", "buckets": "2000"}, "Buckets (N): must be between 1 and 1000"),
                       ({"study": "ofi", "window": "soon"}, "Window (time units): enter a number"),
                       ({"study": "delta", "side": "psychic"}, "Trade sides: choose"),
                       ({"study": "resilience", "depleted": "0.9", "recovered": "0.6"}, "above Depleted below"),
                       ({"study": "vpin", "volume": "0.5"}, "at least 1"),
                       ({"study": "ofi", "enabled": "maybe"}, "choose on or off"),
                       ({"study": "nothing"}, "Choose one of"), ({}, "Choose one of")):
        bad = client.post("/api/studies/config", data=form)
        assert bad.status_code == 422 and text in bad.text and "text-rose-400" in bad.text, form
    assert manager.studies_version == version + 1 and manager.studies["ofi"].settings["window"] == 5.0

    off = client.post("/api/studies/config", data={"study": "otr", "enabled": "off"})
    assert off.status_code == 200 and manager.studies["otr"].enabled is False
    body = strict(client.get("/api/studies"))
    assert body["enabled"]["otr"] is False and "otr" not in body["values"]
    client.post("/api/studies/config", data={"study": "otr", "enabled": "on", "window": "12"})
    assert manager.studies["otr"].enabled is True and manager.studies["otr"].settings["window"] == 12.0


def test_a_change_of_a_study_or_of_the_footprint_wakes_the_streams_of_the_other_tabs(client, manager):
    queue = manager.subscribe()
    try:
        def woken():
            n = 0
            while not queue.empty():
                queue.get_nowait()
                n += 1
            return n
        assert client.post("/api/studies/config", data={"study": "ofi", "window": "7"}).status_code == 200
        assert woken() >= 1
        assert client.post("/api/studies/config", data={"study": "ofi", "window": "7"}).status_code == 200
        assert woken() == 0  # nothing changed: nobody is woken
        assert client.post("/api/chart/footprint", data={"on": "true"}).status_code == 204
        assert woken() >= 1
    finally:
        manager.unsubscribe(queue)


def test_post_chart_footprint_switches_the_cells_in_the_candles_and_reloads_the_chart(client, manager):
    manager.step(8000)
    assert manager.footprint_on is False
    plain_candles = client.get("/api/candlesticks").json()
    assert plain_candles["footprint"] is False and all("cells" not in c for c in plain_candles["candles"])
    version = manager.chart_version
    on = client.post("/api/chart/footprint", data={"on": "true"})
    assert on.status_code == 204 and manager.footprint_on is True and manager.chart_version == version + 1
    body = client.get("/api/candlesticks").json()
    assert body["footprint"] is True and sum(c["cells"] is not None for c in body["candles"]) > 0
    for c in body["candles"]:
        if c["cells"] is not None:
            assert sum(cell[2] for cell in c["cells"]) == c["buy_volume"]
            assert sum(cell[1] + cell[2] for cell in c["cells"]) == c["volume"]
    assert client.post("/api/chart/footprint", data={"on": "true"}).status_code == 204
    assert manager.chart_version == version + 1  # no change, no reload
    assert client.post("/api/chart/footprint", data={"on": "off"}).status_code == 204
    assert manager.footprint_on is False and manager.chart_version == version + 2
    assert all("cells" not in c for c in client.get("/api/candlesticks").json()["candles"])
    assert client.post("/api/chart/footprint", data={"on": "perhaps"}).status_code == 422


def test_get_tape_gives_the_ring_or_what_came_after_a_number(client, manager):
    empty = strict(client.get("/api/tape"))
    assert empty == {"version": manager.studies_version, "seq": 0, "reload": True, "fills": []}
    manager.step(30_000)
    full = strict(client.get("/api/tape"))
    assert full["reload"] is True and len(full["fills"]) == 2000 and full["seq"] == manager.fill_seq
    assert all(len(f) == 5 and f[4] in (1, -1) for f in full["fills"])
    manager.step(500)
    news = strict(client.get("/api/tape", params={"since": full["seq"]}))
    assert news["reload"] is False and [f[0] for f in news["fills"]] == list(range(full["seq"] + 1, news["seq"] + 1))
    assert strict(client.get("/api/tape", params={"since": 5}))["reload"] is True


def events_of(lines):
    """Every (name, data) on a stream; the data as JSON where it is JSON, refusing NaN and Infinity."""
    event = None
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: ") and event:
            data = line[len("data: "):]
            if event in ("studies", "tape", "candles", "chart_reset"):
                data = json.loads(data, parse_constant=lambda token: pytest.fail(f"{token} in JSON"))
            yield event, data
            event = None


def next_push(events, wanted_tape=False):
    """The studies event of the next push and, if that push sent one, its tape event (which follows at once)."""
    for name, data in events:
        if name == "studies":
            after = next(events)
            if after[0] == "tape":
                return data, after[1]
            if not wanted_tape:
                return data, None


def test_the_stream_pushes_the_studies_and_the_tape_in_pieces_and_whole_after_a_new_version(client, manager, deadline):
    manager.step(4000)
    with client.stream("GET", "/events") as resp:
        events = events_of(resp.iter_lines())
        studies, tape = next_push(events)  # on connect: all of it
        assert studies["version"] == manager.studies_version and len(studies["samples"]) == studies["unit"]
        assert studies["values"]["vpin"]["vpin"] is None and studies["failed"] is None and studies["keys"]
        assert tape["reload"] is True and len(tape["fills"]) == min(2000, manager.fill_seq)
        unit, seq = studies["unit"], tape["seq"]

        client.post("/api/sim/control", data={"action": "step", "n": 3000})
        studies, tape = next_push(events, wanted_tape=True)
        assert [row[0] for row in studies["samples"]] == list(range(unit + 1, studies["unit"] + 1))
        assert tape["reload"] is False and [f[0] for f in tape["fills"]] == list(range(seq + 1, tape["seq"] + 1))
        unit, seq = studies["unit"], tape["seq"]
        assert unit > 50 and seq > 300

        client.post("/api/studies/config", data={"study": "ofi", "window": "9"})  # a new version: all again
        studies, tape = next_push(events, wanted_tape=True)
        assert studies["settings"]["ofi"]["window"] == 9.0 and studies["unit"] == unit
        assert len(studies["samples"]) == unit and tape["reload"] is True and tape["seq"] == seq
        assert len(tape["fills"]) == min(2000, seq)
        assert all(row[2] is None for row in studies["samples"])  # that study's samples are gone
        assert any(row[3] is not None for row in studies["samples"])  # and the others' are not

        client.post("/api/sim/control", data={"action": "reset"})
        studies, _ = next_push(events)
        assert studies["version"] == manager.studies_version and studies["samples"] == [] and studies["unit"] == 0
        client.post("/api/sim/control", data={"action": "step", "n": 3000})
        studies, tape = next_push(events, wanted_tape=True)
        assert studies["unit"] > 50 and len(studies["samples"]) == studies["unit"] and tape["reload"] is False
        assert tape["fills"][0][0] == seq + 1  # the numbers go on after a reset


def test_the_stream_pushes_the_footprint_cells_only_to_a_session_that_asked_for_them(client, manager, deadline):
    manager.step(3000)
    with client.stream("GET", "/events") as resp:
        events = events_of(resp.iter_lines())

        def next_of(wanted):
            return next(data for name, data in events if name == wanted)
        assert all("cells" not in bar for bar in next_of("candles"))
        client.post("/api/chart/footprint", data={"on": "true"})
        assert next_of("chart_reset") is True
        client.post("/api/sim/control", data={"action": "step", "n": 2000})
        bars = next_of("candles")
        assert bars and all("cells" in bar for bar in bars)


def test_the_studies_and_the_tape_survive_a_session_that_keeps_running_in_the_background(client, manager):
    async def run():
        await manager.start_sim_loop()
        await asyncio.sleep(0.4)
        await manager.stop_sim_loop()
    manager.speed = 2000
    asyncio.run(run())
    assert manager.studies.failed is None and manager.event_count > 100
    body = strict(client.get("/api/studies"))
    assert body["failed"] is None and body["unit"] == math.floor(manager.sim_time)


# ---- A recording carries V ----

def good_start(m):
    return json.loads(json.dumps(m._snapshot()))


def test_a_recording_stores_the_lots_of_a_vpin_bucket_and_check_validates_them():
    m = board_manager()
    m.step(200)  # fewer than 200 fills: V is not known yet
    assert "studies" not in m._snapshot()
    m.step(5000)
    start = good_start(m)
    assert start["studies"] == {"vpin_v": m.studies.volume} and m.studies.volume >= 1
    tape = rec.Tape()
    rec.check(start, tape)
    rec.check({k: v for k, v in start.items() if k != "studies"}, tape)  # a file made before: no studies
    rec.check({**start, "studies": {"other": [1]}}, tape)  # what is not known is left alone
    for bad in (0.5, 0, -3, math.nan, math.inf, "250", True, None, [250]):
        with pytest.raises(ValueError, match="VPIN bucket"):
            rec.check({**start, "studies": {"vpin_v": bad}}, tape)
    for bad in ([], "x", 3, None):
        with pytest.raises(ValueError, match="studies"):
            rec.check({**start, "studies": bad}, tape)
    loaded = rec.from_bytes(rec.to_bytes(rec.Recording(
        id=1, name="a", created="", start=start, tape=tape, end_time=start["time"])), 1)
    assert loaded.start["studies"] == start["studies"]


def test_a_recording_made_without_a_board_or_before_v_was_known_plays_back_and_calibrates_itself():
    m = LOBManager()  # no board
    m.step(4000)
    m.start_recording()
    m.step(5000)
    recording = m.stop_recording()
    assert "studies" not in recording.start
    p = board_manager()
    p.start_playback(recording)
    assert p.studies.volume is None
    p.step(10**9)
    assert p.studies.volume is not None and p.studies.cvd.value() != 0
    assert p.studies_snapshot()["values"]["vpin"]["buckets"] > 0


def test_the_lots_of_a_vpin_bucket_a_playback_uses_are_the_recordings_not_its_own():
    m = board_manager(vpin={"volume": 125.0})
    m.step(1000)
    m.start_recording()
    m.step(1000)
    recording = m.stop_recording()
    assert recording.start["studies"] == {"vpin_v": 125.0}
    p = board_manager()  # its own setting is to calibrate
    p.start_playback(recording)
    assert p.studies.volume == 125.0 and p.studies.set["vpin"].settings["volume"] == 0
    p.rewind_playback()
    assert p.studies.volume == 125.0
