"""Triggers (app/triggers.py, LOBManager._after_event, and their routes, stream and chart markers).

Each rule is held to a hand-worked case or to a computation made here from the manager's own record, not by the code
under test: the four operators and the rule for a missing value on sequences worked on paper, the cooldown on times
worked on paper, the crossings of a real run against the crossings of a twin run's series of values, the pause against
the first step at which a twin run's CVD (summed from its trade log) is positive, flatten against the book worked by
hand, a playback with every action armed against the session it was recorded from, and the study values of a run with
rules armed against those of a run without.
"""

import asyncio
import json
import math
import signal
from collections import Counter

import pytest
from starlette.testclient import TestClient

from app.main import app, registry
from app.manager import MANUAL_ACTION_DT, LOBManager
from app.sessions import COOKIE_NAME
from app.studies import STUDY_LABELS, StudyBoard
from app.triggers import (ACTIONS, COOLDOWN_MAX, MAX_CONDITIONS, MAX_MARKS, MAX_RULES, NOTICE_EVERY, OPERATORS,
                          THRESHOLD_MAX, TRIGGER_VALUES, VALUE_GROUPS, Condition, TriggerBoard, _flag, _many, g,
                          parse_rule, validate)
from lob.order import Side, cancel_order, limit_order
from lob.studies import StudySet
from tests.test_recording import state
from tests.test_studies import make_book

BID, ASK = Side.BID, Side.ASK
ALWAYS = Condition("cvd", "cvd", "above", -1e9)  # true at every check: the CVD is a number
NEVER = Condition("cvd", "cvd", "below", -1e9)


class Clock:
    """Wall time for the board's notices, moved by hand."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Source:
    """The studies' values for a board by hand (study key -> {value name: number or None}), and what was read."""

    def __init__(self, **values):
        self.values = values
        self.flow_reads, self.book_reads = Counter(), Counter()

    def flow(self, key):
        self.flow_reads[key] += 1
        return self.values.get(key)

    def book(self, key):
        self.book_reads[key] += 1
        return self.values.get(key)

    def set(self, study, **values):
        self.values[study] = values


def c(study, name, op, h):
    return Condition(study, name, op, h)


def fired(board, source, t):
    return [f.rule.name for f in board.evaluate(t, source.flow, source.book)]


# ---- Conditions: the four operators and a missing value, worked by hand ----

SEQUENCE = [None, 4, 5, 6, 6, 5, 4, 5, 6]  # the value at nine checks; the level is 5
BY_HAND = {  # which checks each operator holds at, worked on paper
    "above": [0, 0, 0, 1, 1, 0, 0, 0, 1],       # 6, 6 and the last 6 are above 5; 5 is not
    "below": [0, 1, 0, 0, 0, 0, 1, 0, 0],       # only the two 4s
    "crosses_above": [0, 0, 0, 1, 0, 0, 0, 0, 1],  # 5 -> 6 twice; 4 -> 5 touches the level and does not pass it
    "crosses_below": [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 5 -> 4 once; 6 -> 5 only reaches it; the first 4 follows a gap
}


@pytest.mark.parametrize("op", list(OPERATORS))
def test_each_operator_holds_where_it_does_on_paper(op):
    condition = c("ofi", "ofi", op, 5.0)
    previous, held = None, []
    for now in SEQUENCE:
        held.append(int(condition.met(previous, now)))
        previous = now
    assert held == BY_HAND[op]


@pytest.mark.parametrize("op", list(OPERATORS))
def test_a_board_fires_a_rule_with_no_cooldown_at_the_checks_the_operator_holds_at(op):
    board = TriggerBoard()
    board.create("r", [c("ofi", "ofi", op, 5.0)], cooldown=0)
    source, got = Source(), []
    for t, now in enumerate(SEQUENCE):
        source.set("ofi", ofi=now)
        got.append(int(bool(fired(board, source, float(t)))))
    assert got == BY_HAND[op]


def test_a_crossing_needs_the_value_before_to_be_on_the_near_side_of_the_level_inclusive():
    """The spread is a whole number of ticks and sits on its level: crosses above 2 is 2 -> 3, not 1 -> 2."""
    up, down = c("spread", "spread", "crosses_above", 2), c("spread", "spread", "crosses_below", 2)
    assert [up.met(a, b) for a, b in [(1, 2), (2, 2), (2, 3), (3, 3), (3, 2), (1, 3)]] == [0, 0, 1, 0, 0, 1]
    assert [down.met(a, b) for a, b in [(3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (3, 1)]] == [0, 0, 1, 0, 0, 1]


def test_a_missing_value_is_never_true_and_a_crossing_is_not_read_across_it():
    up = c("ofi", "ofi", "crosses_above", 5.0)
    assert not up.met(None, 6) and not up.met(4, None) and not up.met(None, None)  # None on either side
    assert not c("ofi", "ofi", "above", 5.0).met(9, None) and not c("ofi", "ofi", "below", 5.0).met(1, None)
    board = TriggerBoard()
    board.create("r", [up], cooldown=0)
    source, got = Source(), []
    for t, now in enumerate([4, None, 6, 4, 6, None, 7]):
        source.set("ofi", ofi=now)
        got.append(int(bool(fired(board, source, float(t)))))
    # 6 after the gap has no value before it (the gap was remembered as None); 6 after 4 crosses; 7 after a gap does not
    assert got == [0, 0, 0, 0, 1, 0, 0]
    board.clear_runtime()
    source.set("ofi", ofi=6)
    assert fired(board, source, 9.0) == []  # nor is one read across a reset: the first value is only a "before"


def test_a_value_that_is_not_finite_counts_as_missing():
    board = TriggerBoard()
    board.create("r", [c("ofi", "ofi", "above", 5.0)], cooldown=0)
    source = Source()
    for bad in (math.nan, math.inf, -math.inf):
        source.set("ofi", ofi=bad)
        assert fired(board, source, 1.0) == []
    status = board.status()
    assert status["rules"][0]["conditions"][0]["now"] is None
    json.dumps(status, allow_nan=False)


# ---- Rules: all of, any of, one crossing, cooldown, one shot ----

def test_all_of_and_any_of_over_the_four_combinations_of_two_conditions():
    board = TriggerBoard()
    both = [c("ofi", "ofi", "above", 0.0), c("imbalance", "imbalance", "below", 0.0)]
    board.create("all", both, "all", cooldown=0)
    board.create("any", both, "any", cooldown=0)
    source = Source()
    table = [  # (ofi, imbalance) -> (all of fires, any of fires), worked by hand
        ((1, -1), (1, 1)), ((1, 1), (0, 1)), ((-1, -1), (0, 1)), ((-1, 1), (0, 0)),
    ]
    for i, ((ofi, imbalance), (all_of, any_of)) in enumerate(table):
        source.set("ofi", ofi=ofi)
        source.set("imbalance", imbalance=imbalance)
        names = fired(board, source, float(i))  # a unit apart: the book study is read afresh each time
        assert ("all" in names, "any" in names) == (bool(all_of), bool(any_of))


def test_the_notice_of_a_firing_names_the_conditions_that_held_and_their_values():
    board = TriggerBoard()
    board.create("r", [c("ofi", "ofi", "above", 100.0), c("imbalance", "imbalance", "below", 0.0)], "any", cooldown=0)
    source = Source(ofi={"ofi": 612.25}, imbalance={"imbalance": 0.3})
    (firing,) = board.evaluate(1.0, source.flow, source.book)
    assert firing.describe() == "OFI 612.25 lots is above 100"  # only the condition that held
    source.set("imbalance", imbalance=-0.25)
    source.set("ofi", ofi=3)
    (firing,) = board.evaluate(2.0, source.flow, source.book)
    assert firing.describe() == "Book imbalance -0.25 is below 0"


def test_an_all_of_rule_holds_at_most_one_crossing_and_an_any_of_rule_any_number():
    up, down = c("ofi", "ofi", "crosses_above", 1.0), c("cvd", "cvd", "crosses_below", 1.0)
    level = c("imbalance", "imbalance", "above", 0.0)
    with pytest.raises(ValueError, match="only one crossing"):
        validate("r", [up, down], "all", 1.0, ["notice"])
    with pytest.raises(ValueError, match="only one crossing"):
        TriggerBoard().create("r", [up, level, down], "all")
    validate("r", [up, level], "all", 1.0, ["notice"])  # one crossing and a level
    validate("r", [up, down], "any", 1.0, ["notice"])  # any of: each may cross
    validate("r", [level, level], "all", 1.0, ["notice"])  # levels, as many as you like


def test_an_all_of_rule_with_a_crossing_and_a_level_fires_only_at_the_crossing_when_the_level_holds():
    board = TriggerBoard()
    board.create("r", [c("ofi", "ofi", "crosses_above", 5.0), c("imbalance", "imbalance", "above", 0.2)], cooldown=0)
    source, got = Source(), []
    for t, (ofi, imbalance) in enumerate([(4, 0.5), (6, 0.1), (4, 0.5), (6, 0.5), (6, 0.5)]):
        source.set("ofi", ofi=ofi)
        source.set("imbalance", imbalance=imbalance)
        got.append(int(bool(fired(board, source, float(t)))))
    # the first crossing (check 1) meets a low imbalance; the second (check 3) meets a high one; check 4 is no crossing
    assert got == [0, 0, 0, 1, 0]


def test_the_cooldown_is_in_simulation_time_and_a_crossing_during_it_is_not_carried_past_it():
    board = TriggerBoard(clock=Clock())  # wall time never moves: only the simulation's clock counts
    board.create("level", [c("ofi", "ofi", "above", 5.0)], cooldown=3.0)
    source = Source(ofi={"ofi": 6})
    times = [0, 1, 2, 3, 4, 5, 6, 6.5, 9]
    assert [t for t in times if fired(board, source, t)] == [0, 3, 6, 9]  # t - last >= 3
    rule = board.rules[0]
    assert rule.count == 4 and rule.last_fired == 9

    board = TriggerBoard()
    board.create("cross", [c("ofi", "ofi", "crosses_above", 5.0)], cooldown=10.0)
    source, hits = Source(), []
    for t, now in [(0, 4), (1, 6), (2, 4), (3, 6), (4, 6), (12, 4), (13, 6)]:
        source.set("ofi", ofi=now)
        if fired(board, source, float(t)):
            hits.append(t)
    # t=1 fires; t=3 crosses again inside the cooldown and is lost; t=4 is no crossing (6 after 6); t=13 fires
    assert hits == [1, 13]


def test_a_rule_whose_clock_went_back_without_being_told_is_not_held_by_a_cooldown_of_the_run_before():
    board = TriggerBoard()
    board.create("r", [c("ofi", "ofi", "above", 5.0)], cooldown=50.0)
    source = Source(ofi={"ofi": 6})
    assert fired(board, source, 100.0) == ["r"] and fired(board, source, 120.0) == []
    assert fired(board, source, 3.0) == ["r"]  # a new run from time 0


def test_a_one_shot_rule_disarms_itself_and_arming_it_again_forgets_what_it_saw():
    board = TriggerBoard()
    rule = board.create("r", [c("ofi", "ofi", "crosses_above", 5.0)], cooldown=0, one_shot=True)
    source = Source(ofi={"ofi": 4})
    assert fired(board, source, 0.0) == []
    source.set("ofi", ofi=6)
    assert fired(board, source, 1.0) == ["r"]
    assert rule.armed is False and board.armed == []
    source.set("ofi", ofi=4)
    assert fired(board, source, 2.0) == []
    board.arm(rule.id, True)
    source.set("ofi", ofi=6)
    assert fired(board, source, 3.0) == []  # 4 -> 6 happened while it was off: no value before the first
    source.set("ofi", ofi=4)
    fired(board, source, 4.0)
    source.set("ofi", ofi=6)
    assert fired(board, source, 5.0) == ["r"] and rule.count == 1  # counted again from the arming


def test_a_rule_that_pauses_is_one_shot_unless_it_says_otherwise():
    board = TriggerBoard()
    by_default = board.create("a", [ALWAYS], actions=["notice", "pause"])
    plain = board.create("b", [ALWAYS], actions=["notice"])
    kept = board.create("c", [ALWAYS], actions=["notice", "pause"], one_shot=False)
    forced = board.create("d", [ALWAYS], actions=["notice"], one_shot=True)
    assert (by_default.one_shot, plain.one_shot, kept.one_shot, forced.one_shot) == (True, False, False, True)


def test_a_level_rule_that_pauses_would_fire_again_after_its_cooldown_so_it_disarms_instead():
    board = TriggerBoard()
    rule = board.create("r", [c("ofi", "ofi", "above", 0.0)], cooldown=1.0, actions=["pause"])
    source = Source(ofi={"ofi": 5})
    assert fired(board, source, 0.0) == ["r"] and fired(board, source, 5.0) == [] and rule.count == 1


# ---- What is read, and when ----

def test_the_studies_that_read_the_book_are_read_at_most_once_in_a_tenth_of_a_unit_and_only_when_a_rule_needs_them():
    board = TriggerBoard()
    board.create("book", [c("imbalance", "imbalance", "crosses_above", 0.5)], cooldown=0)
    board.create("flow", [c("ofi", "ofi", "above", 1e9)], cooldown=0)
    board.create("unused", [c("vpin", "vpin", "above", 0.1)], cooldown=0, armed=False)
    source, hits = Source(ofi={"ofi": 1}, imbalance={"imbalance": 0.2}), []
    for i in range(32):  # 1/16 of a unit apart (exact in binary), so the reads fall every second check
        t = i / 16
        if i == 1:
            source.set("imbalance", imbalance=0.9)  # changes between reads: only seen at the next read, at i = 2
        if fired(board, source, t):
            hits.append(i)
    assert source.book_reads["imbalance"] == 16 and source.flow_reads["ofi"] == 32
    assert "vpin" not in source.flow_reads and "vpin" not in source.book_reads  # an unarmed rule is not read for
    assert source.book_reads["ofi"] == 0 and source.flow_reads["imbalance"] == 0  # each kind through its own reader
    assert hits == [2]  # the crossing 0.2 -> 0.9 is seen when the book is read afresh, and only then


def test_a_book_value_read_before_the_market_started_over_is_not_used_after_it():
    board = TriggerBoard()
    board.create("r", [c("imbalance", "imbalance", "above", 0.5)], cooldown=0)
    read = []
    board.evaluate(0.05, lambda s: None, lambda s: read.append(0.9) or {"imbalance": 0.9})
    board.clear_runtime()  # 0.01 later, well inside the tenth of a unit: it would be the value of the market before
    assert not board.evaluate(0.06, lambda s: None, lambda s: read.append(0.1) or {"imbalance": 0.1})
    assert read == [0.9, 0.1]


def test_a_book_value_is_read_afresh_when_the_clock_goes_back_and_when_a_rule_is_armed():
    board = TriggerBoard()
    board.create("book", [c("imbalance", "imbalance", "above", 0.5)], cooldown=0)
    source = Source(imbalance={"imbalance": 0.9})
    assert fired(board, source, 5.0) == ["book"] and source.book_reads["imbalance"] == 1
    fired(board, source, 5.05)
    assert source.book_reads["imbalance"] == 1  # inside the tenth
    fired(board, source, 1.0)
    assert source.book_reads["imbalance"] == 2  # the clock went back: a new run
    board.create("other", [c("spread", "spread", "above", 1)], cooldown=0)
    fired(board, source, 1.01)
    assert source.book_reads["imbalance"] == 3 and source.book_reads["spread"] == 1  # a new rule: read at once


# ---- Status ----

def test_the_status_says_armed_warming_up_disarmed_or_study_off_and_holds_only_json_numbers():
    board = TriggerBoard()
    warm = board.create("warm", [c("ofi", "ofi", "above", 1)], cooldown=0)
    both = board.create("both", [c("ofi", "ofi", "above", 1), c("vpin", "vpin", "above", 0.1)], "all", cooldown=0)
    either = board.create("either", [c("ofi", "ofi", "above", 1), c("vpin", "vpin", "above", 0.1)], "any", cooldown=0)
    board.create("absent", [c("resilience", "depleted_bid", "above", 1)], cooldown=0)
    board.create("off", [c("delta", "delta", "above", 1)], cooldown=0)
    board.create("idle", [c("cvd", "cvd", "above", 1)], cooldown=0, armed=False)
    assert board.state_of(warm) == "armed"  # nothing evaluated yet
    source = Source(ofi={"ofi": None}, vpin={"vpin": None}, resilience={"depleted_bid": None}, delta=None,
                    cvd={"cvd": 0})
    board.evaluate(1.0, source.flow, source.book)
    got = {r.name: board.state_of(r, off=["delta"]) for r in board.rules}
    assert got == {"warm": "warming up",  # a window not full
                   "both": "warming up",  # an all-of rule waits for any of its values
                   "either": "warming up",  # an any-of rule, when all of them are warming
                   "absent": "armed",  # no depletion is no reason to wait
                   "off": "study off", "idle": "disarmed"}
    source.set("ofi", ofi=0.5)
    board.evaluate(2.0, source.flow, source.book)
    assert board.state_of(either) == "armed" and board.state_of(both) == "warming up"
    status = board.status(off=["delta"])
    assert [r["state"] for r in status["rules"]][:2] == ["armed", "warming up"]
    cond = status["rules"][3]["conditions"][0]
    assert cond["now"] is None and cond["none"] == "the bid is not depleted" and cond["met"] is False
    assert status["rules"][4]["conditions"][0]["off"] is True
    json.dumps(status, allow_nan=False)


def test_the_status_carries_what_a_page_needs_to_show_and_to_edit_a_rule():
    board = TriggerBoard()
    rule = board.create("OFI burst", [c("ofi", "ofi", "crosses_above", 500.0)], cooldown=5.0,
                        actions=["notice", "pause"])
    source = Source(ofi={"ofi": 120.5})
    board.evaluate(7.0, source.flow, source.book)
    (shown,) = board.status()["rules"]
    assert shown == {
        "id": rule.id, "name": "OFI burst", "armed": True, "one_shot": True, "combine": "all", "cooldown": 5.0,
        "actions": ["notice", "pause"], "text": "OFI crosses above 500 lots", "state": "armed", "count": 0,
        "fired_at": None,
        "conditions": [{"value": "ofi:ofi", "op": "crosses_above", "threshold": 500.0,
                        "text": "OFI crosses above 500 lots", "label": "OFI", "unit": "lots", "now": 120.5,
                        "met": False, "none": "", "off": False}]}


def test_whether_a_condition_holds_is_not_known_before_the_first_check_and_a_flag_after_it():
    board = TriggerBoard()
    board.create("r", [ALWAYS, c("cvd", "cvd", "below", -1e9)], "any", cooldown=0)
    assert [x["met"] for x in board.status()["rules"][0]["conditions"]] == [None, None]
    board.evaluate(1.0, Source(cvd={"cvd": 3}).flow, None)
    assert [x["met"] for x in board.status()["rules"][0]["conditions"]] == [True, False]


def test_a_rule_without_a_name_is_named_from_its_conditions():
    board = TriggerBoard()
    a = board.create("", [c("ofi", "ofi", "above", 5.0), c("cvd", "cvd", "below", -2.0)], "any")
    assert a.name == "OFI above 5 lots or CVD below -2 lots"
    long = board.create("", [c("ofi", "ofi", "crosses_above", 5.0), c("cvd", "cvd", "crosses_below", -2.0)], "any")
    assert long.name == "OFI crosses above 5 lots or CVD crosses"  # cut at 40 characters, and not left with a space
    b = board.update(a.id, "", [c("vpin", "vpin", "above", 0.4)])
    assert b.name == "VPIN above 0.4"


# ---- Notices: at most one a second for a rule, carrying a count ----

def test_a_rules_notices_are_at_most_one_a_second_and_the_next_one_carries_the_count():
    clock = Clock()
    board = TriggerBoard(clock=clock)
    rule = board.create("r", [ALWAYS], cooldown=0)
    assert board.notice(rule, "first") == "first"  # nothing sent before: at once
    for _ in range(4):
        clock.t += 0.2
        assert board.notice(rule, "later") is None  # 0.2, 0.4, 0.6, 0.8 seconds after
    assert rule.unsent == 4 and board.pending
    assert board.flush() == []  # 0.8 seconds on: not yet
    clock.t += 0.2
    assert board.flush() == ["later (4 firings since the last notice)"]
    assert rule.unsent == 0 and not board.pending and board.flush() == []
    clock.t += NOTICE_EVERY
    assert board.notice(rule, "alone") == "alone"  # one firing: no count
    clock.t += 0.5
    assert board.notice(rule, "a") is None
    clock.t += 0.5
    assert board.notice(rule, "b") == "b (2 firings since the last notice)"  # a firing after the second carries it


def test_each_rule_has_its_own_second():
    clock = Clock()
    board = TriggerBoard(clock=clock)
    a, b = board.create("a", [ALWAYS]), board.create("b", [ALWAYS])
    assert board.notice(a, "a1") == "a1" and board.notice(b, "b1") == "b1"
    assert board.notice(a, "a2") is None and board.notice(b, "b2") is None


def test_an_urgent_notice_goes_out_at_once_with_the_count_of_the_firings_held_back():
    clock = Clock()
    board = TriggerBoard(clock=clock)
    rule = board.create("r", [ALWAYS], cooldown=0)
    assert board.notice(rule, "first") == "first"
    for _ in range(3):
        clock.t += 0.1
        assert board.notice(rule, "held") is None
    clock.t += 0.1  # 0.4 seconds after the last notice: the gate is closed
    assert board.notice(rule, "gated") is None and rule.unsent == 4
    clock.t += 0.1
    assert board.notice(rule, "urgent", urgent=True) == "urgent (5 firings since the last notice)"
    assert rule.unsent == 0 and not board.pending and rule.notice_wall == clock.t
    clock.t += 0.1
    assert board.notice(rule, "after") is None  # an urgent notice opened the second like any other
    assert board.notice(rule, "alone", urgent=True) == "alone (2 firings since the last notice)"


# ---- Reading a form ----

GOOD = {"name": "OFI burst", "value": ["ofi:ofi"], "op": ["crosses_above"], "threshold": ["500"],
        "action": ["notice", "pause"], "cooldown": "5"}


def test_a_form_becomes_the_arguments_of_a_rule_with_defaults_for_what_it_leaves_out():
    got = parse_rule(GOOD)
    assert got == {"name": "OFI burst", "conditions": (Condition("ofi", "ofi", "crosses_above", 500.0),),
                   "combine": "all", "cooldown": 5.0, "actions": ("notice", "pause"), "one_shot": None, "armed": None}
    got = parse_rule({**GOOD, "one_shot": "off", "armed": "on", "combine": "any",
                      "action": ["pause", "pause", "cancel"]})
    assert (got["one_shot"], got["armed"], got["combine"], got["actions"]) == (False, True, "any", ("pause", "cancel"))
    two = parse_rule({"value": ["ofi:ofi", "spread:z"], "op": ["above", "below"], "threshold": ["1", "-2.5"],
                      "action": ["notice"]})
    assert two["conditions"] == (Condition("ofi", "ofi", "above", 1.0), Condition("spread", "z", "below", -2.5))
    assert two["cooldown"] == 10.0 and two["name"] == ""


@pytest.mark.parametrize("change, words", [
    ({"value": [], "op": [], "threshold": []}, "at least one condition"),
    ({"value": ["ofi:nope"]}, "No value 'ofi:nope'"),
    ({"op": ["sideways"]}, "No operator"),
    ({"threshold": ["abc"]}, "must be a number"),
    ({"threshold": ["nan"]}, "must be a number below"),
    ({"threshold": ["1e30"]}, "must be a number below"),
    ({"cooldown": "-1"}, "cooldown must be between"),
    ({"cooldown": "x"}, "cooldown must be a number"),
    ({"cooldown": "1e9"}, "cooldown must be between"),
    ({"action": []}, "at least one action"),
    ({"action": ["explode"]}, "No action"),
    ({"combine": "most"}, "Combine with"),
    ({"name": "x" * 41}, "at most 40 characters"),
    ({"one_shot": "maybe"}, "choose on or off"),
    ({"value": ["ofi:ofi", "cvd:cvd"], "op": ["crosses_above", "crosses_below"], "threshold": ["1", "1"]},
     "only one crossing"),
    ({"value": ["ofi:ofi"] * (MAX_CONDITIONS + 1), "op": ["above"] * (MAX_CONDITIONS + 1),
      "threshold": ["1"] * (MAX_CONDITIONS + 1)}, "at most 4 conditions"),
    ({"op": ["above", "below"]}, "Each condition needs"),
])
def test_what_is_wrong_with_a_form_is_said(change, words):
    form = {**GOOD, **change}
    if "action" in change and not change["action"]:
        form["action"] = []
    with pytest.raises(ValueError, match=words):
        parse_rule(form)


def test_every_value_a_rule_can_read_is_one_the_studies_give_and_book_studies_are_those_that_read_the_book():
    book = make_book([(100, 10), (99, 20), (98, 5)], [(102, 4), (103, 6), (105, 30)])
    studies = StudySet(auto_sample=False)
    values = studies.values(0.0, book)
    for v in TRIGGER_VALUES:
        assert v.name in values[v.study], (v.study, v.name)
        assert v.book == (not studies[v.study].sees_events), (v.study, v.name)
    assert {v.study for v in TRIGGER_VALUES if v.book} == {"imbalance", "spread"}
    assert len({(v.study, v.name) for v in TRIGGER_VALUES}) == len(TRIGGER_VALUES)


def test_the_board_holds_at_most_a_dozen_rules_and_numbers_them_once():
    board = TriggerBoard()
    ids = [board.create(f"r{i}", [ALWAYS]).id for i in range(MAX_RULES)]
    with pytest.raises(ValueError, match="12 rules already"):
        board.create("one too many", [ALWAYS])
    board.delete(ids[3])
    assert board.create("again", [ALWAYS]).id == ids[-1] + 1  # a number is never reused
    with pytest.raises(LookupError):
        board.get(ids[3])


def test_changing_a_rule_clears_what_it_had_seen_and_keeps_its_place_and_number():
    board = TriggerBoard()
    first = board.create("a", [ALWAYS], cooldown=0)
    board.create("b", [ALWAYS], cooldown=0)
    source = Source(cvd={"cvd": 1})
    board.evaluate(1.0, source.flow, source.book)
    assert first.count == 1
    again = board.update(first.id, "a2", [c("ofi", "ofi", "above", 1)], "all", 2.0, ["pause"], armed=False)
    assert again is first and [r.id for r in board.rules] == [1, 2] and first.count == 0 and first.last_fired is None
    assert (first.name, first.armed, first.one_shot, first.cooldown, first.previous) == ("a2", False, True, 2.0, [None])
    assert [r.id for r in board.armed] == [2]


def test_the_firings_that_mark_the_chart_go_with_the_rule_that_made_them_when_it_is_changed_armed_or_deleted():
    board = TriggerBoard()
    a = board.create("a", [ALWAYS], cooldown=0)
    b = board.create("b", [ALWAYS], cooldown=0)
    source = Source(cvd={"cvd": 1})
    for t in (1.0, 2.0):
        board.evaluate(t, source.flow, source.book)
    assert list(board.fired) == [(1.0, 1, "a"), (1.0, 2, "b"), (2.0, 1, "a"), (2.0, 2, "b")]
    board.update(a.id, "a", [ALWAYS], cooldown=0)  # a different rule now
    assert [m[1] for m in board.fired] == [2, 2]
    board.arm(b.id, False)  # disarming leaves what it did; arming starts it afresh
    assert [m[1] for m in board.fired] == [2, 2]
    board.arm(b.id, True)
    assert list(board.fired) == []
    board.evaluate(3.0, source.flow, source.book)
    board.delete(a.id)
    assert [m[1] for m in board.fired] == [2]


def test_the_chart_keeps_the_last_thirty_firings_and_no_more():
    board = TriggerBoard()
    board.create("r", [c("cvd", "cvd", "above", 0)], cooldown=0)
    source = Source(cvd={"cvd": 1})
    assert MAX_MARKS == 30
    for t in range(MAX_MARKS + 10):
        board.evaluate(float(t), source.flow, source.book)
    marks = board.status()["marks"]
    assert len(marks) == 30 and marks[0][0] == 10.0 and marks[-1][0] == 39.0


def test_the_largest_level_and_cooldown_are_the_documented_ones_and_a_little_more_is_refused():
    assert (THRESHOLD_MAX, COOLDOWN_MAX) == (1e12, 100_000.0)
    at = c("cvd", "cvd", "above", 1e12)
    validate("x", [at], "all", 100_000, ["notice"])  # both limits themselves are allowed
    validate("x", [c("cvd", "cvd", "above", -1e12)], "all", 0, ["notice"])
    with pytest.raises(ValueError, match="below 1e\\+12 in size"):
        validate("x", [c("cvd", "cvd", "above", 1.5e12)], "all", 1, ["notice"])
    with pytest.raises(ValueError, match="between 0 and 100,000 units"):
        validate("x", [at], "all", 150_000, ["notice"])


@pytest.mark.parametrize("text, flag", [("on", True), ("true", True), ("1", True), (" TRUE ", True), ("off", False),
                                        ("false", False), ("0", False), ("maybe", None), ("", None), (None, None)])
def test_a_flag_is_read_from_on_true_1_off_false_and_0(text, flag):
    assert _flag(text) is flag


def test_a_field_a_form_does_not_have_is_no_values_and_one_it_has_is_a_list_of_text():
    assert _many({}, "action") == [] and _many({"other": "x"}, "action") == []
    assert _many({"action": "pause"}, "action") == ["pause"]
    assert _many({"action": ["notice", "pause"]}, "action") == ["notice", "pause"]
    assert _many({"threshold": (1, 2.5)}, "threshold") == ["1", "2.5"]


@pytest.mark.parametrize("x, text", [(0, "0"), (5.0, "5"), (123.456, "123.46"), (1234.5678, "1,234.57"),
                                     (-123.456, "-123.46"), (100, "100"), (1.23456, "1.235"), (1.5, "1.5"),
                                     (0.00012, "0.00012"), (0.123456, "0.12346"), (-0.5, "-0.5")])
def test_numbers_are_shown_as_short_as_they_read_right(x, text):
    assert g(x) == text


def test_the_values_a_menu_offers_are_grouped_under_their_own_study_in_the_studies_order():
    assert [key for key, _, _ in VALUE_GROUPS] == list(dict.fromkeys(v.study for v in TRIGGER_VALUES))
    for key, label, values in VALUE_GROUPS:
        assert label == STUDY_LABELS[key] and values and all(v.study == key for v in values)
    assert [v for _, _, values in VALUE_GROUPS for v in values] == sorted(
        TRIGGER_VALUES, key=lambda v: [k for k, _, _ in VALUE_GROUPS].index(v.study))


# ---- In a session: the manager ----

def session(seed=5, model="smith_zi", **kwargs):
    m = LOBManager()
    m.attach_studies(StudyBoard())
    m.attach_triggers(TriggerBoard(clock=kwargs.get("clock", Clock())))
    m.seed = seed
    if model != "smith_zi":
        m.set_model(model)
    m.reset(100)
    return m


def spy(m):
    """Every call of the board's evaluate, as the simulation time it was made at; and of _fire, as the firing."""
    calls, firings = [], []
    real_evaluate, real_fire = m.triggers.evaluate, m._fire

    def evaluate(t, flow, book):
        calls.append(t)
        return real_evaluate(t, flow, book)

    def fire(firing):
        firings.append((firing.t, firing.rule.name, m.event_count))
        return real_fire(firing)

    m.triggers.evaluate, m._fire = evaluate, fire
    return calls, firings


def test_the_rules_are_checked_after_every_step_an_idle_one_included_at_the_clock_it_left():
    m = session()
    m.triggers.create("never", [NEVER])
    calls, _ = spy(m)
    real = m.model.generate_event
    count = {"n": 0}

    def sometimes_idle(book, t, rng):
        count["n"] += 1
        return None if count["n"] % 3 == 0 else real(book, t, rng)

    m.model.generate_event = sometimes_idle
    m.step(30)
    assert len(calls) == 30  # one a step, and every third was a step with no event
    # an idle step stands where the clock is
    assert calls[2] == calls[1] and calls[5] == calls[4] and calls[1] > calls[0]
    assert m.event_count == 20 and calls == sorted(calls)


def test_a_cancel_the_model_aims_at_the_users_order_is_skipped_and_the_rules_are_checked_after_it():
    m = session(5)
    m.step(300)
    order, _ = m.submit_limit_order(m.book.best_bid - 2, 7, BID)
    m.triggers.create("never", [NEVER])
    calls, _ = spy(m)
    events = m.event_count
    m.model.generate_event = lambda book, t, rng: cancel_order(order.order_id, m.sim_time + 0.01)
    m.step(4)
    assert len(calls) == 4 and m.event_count == events  # four checks; none of the four cancels reached the book
    assert m.book.get_order(order.order_id) is not None and order.order_id in m.manual_order_ids


@pytest.mark.parametrize("how", ["no board", "collecting", "quiet", "studies suspended", "studies failed",
                                 "no studies board", "while acting"])
def test_the_rules_are_not_checked_without_a_board_or_while_collecting_seeking_or_acting(how):
    m = session()
    m.triggers.create("always", [ALWAYS], cooldown=0)
    calls, _ = spy(m)
    if how == "no board":
        m.triggers = None
    elif how == "collecting":
        m.collecting = (0, 10)
    elif how == "quiet":
        m._quiet = True
    elif how == "studies suspended":
        m.studies.suspended = True
    elif how == "studies failed":
        m.studies.fail(RuntimeError("x"))
    elif how == "no studies board":
        m.studies = None
    elif how == "while acting":
        m._acting = True
    m.step(20)
    assert calls == []


def first_positive_cvd(seed, limit=400):
    """The first step (and the events by then) at which the CVD is positive, in a run without rules.

    The CVD is summed here from the manager's trade log: the lots of every fill, plus where the buyer was the
    aggressor (the log's 1) and minus where the seller was, as it is after each step.
    """
    twin = session(seed)
    for k in range(1, limit + 1):
        twin.step(1)
        log = twin._trade_log
        pairs = zip(log["size"], log["aggressor"], strict=True)
        cvd = sum(size if aggressor == 1 else -size for size, aggressor in pairs)
        if cvd > 0:
            return k, twin.event_count
    raise AssertionError("the CVD never became positive")


def test_a_pause_stops_the_batch_it_came_in_at_the_step_the_condition_first_holds():
    k, events = first_positive_cvd(5)
    assert 1 <= k < 100  # inside one frame of 100 steps, the batch size at speed 2000 (frames of round(2000 x 0.05))
    m = session(5)
    m.triggers.create("pause", [c("cvd", "cvd", "above", 0)], cooldown=0, actions=["pause", "notice"])
    m.is_running = True
    m.step(100)
    assert m.event_count == events  # not the 100 events the batch asked for
    assert m.is_running is False and m._halt is True
    assert m.triggers.rules[0].count == 1 and m.triggers.rules[0].armed is False  # one-shot by default
    twin = session(5)
    twin.step(100)
    assert twin.event_count == 100  # without the halt the frame would have run on


def test_the_halt_ends_with_the_batch_and_the_next_step_runs_on():
    m = session(5)
    m.triggers.create("pause", [c("cvd", "cvd", "above", 0)], cooldown=0, actions=["pause"])
    m.step(100)
    stopped = m.event_count
    assert m._halt is True
    m.step(1)  # a click on Step after the pause: the rule is spent, the step runs
    assert m._halt is False and m.event_count == stopped + 1
    m.is_running = True
    m.triggers.arm(m.triggers.rules[0].id, True)
    m.triggers.rules[0].one_shot = False
    m.step(100)  # armed again with the CVD still positive: it pauses again, at the first step
    assert m.event_count == stopped + 2 and m.is_running is False


def test_a_pause_ends_the_run_loop_after_its_first_frame_at_speed_2000():
    _, events = first_positive_cvd(5)
    m = session(5)
    m.speed = 2000
    m.triggers.create("pause", [c("cvd", "cvd", "above", 0)], cooldown=0, actions=["pause"])

    async def run():
        await m.start_sim_loop()
        for _ in range(200):
            if not m.is_running:
                break
            await asyncio.sleep(0.01)
        task = m._bg_task
        if task is not None:
            await asyncio.wait_for(task, 5)  # the loop has ended by itself: nothing stopped it
        return m.is_running

    assert asyncio.run(run()) is False
    assert m.event_count == events


def test_crossings_in_a_real_run_are_those_of_the_series_of_values_a_twin_run_gives():
    """The OFI after each step of a run without rules is the series; the crossings of 0 in it are worked out here."""
    twin = session(7)
    series = []
    for _ in range(12000):
        twin.step(1)
        series.append((twin.event_count, twin.studies["ofi"].values(twin.sim_time)["ofi"]))
    assert any(v is None for _, v in series) and sum(v is not None for _, v in series) > 11000
    assert sum(v == 0 for _, v in series) > 300  # an integer series sits on the level often: the inclusive side matters
    expected = {"above": [], "crosses_above": [], "crosses_below": [], "below": []}
    previous = None
    for events, now in series:
        if now is not None:
            expected["above"] += [events] * (now > 0)
            expected["below"] += [events] * (now < 0)
            if previous is not None:
                expected["crosses_above"] += [events] * (previous <= 0 < now)
                expected["crosses_below"] += [events] * (previous >= 0 > now)
        previous = now
    assert all(len(v) > 50 for v in expected.values())
    m = session(7)
    for op in expected:
        m.triggers.create(op, [c("ofi", "ofi", op, 0.0)], cooldown=0)
    _, firings = spy(m)
    m.step(12000)
    for op, events in expected.items():
        assert [e for _, name, e in firings if name == op] == events, op


def test_rules_that_read_the_studies_do_not_change_what_the_studies_say():
    a, b = session(9), session(9)
    for study, name in [("ofi", "ofi"), ("vpin", "vpin"), ("otr", "otr"), ("imbalance", "imbalance"),
                        ("spread", "z"), ("resilience", "score"), ("delta", "delta"), ("cvd", "cvd")]:
        b.triggers.create(study, [c(study, name, "above", 1e12)], cooldown=0)
    a.step(6000)
    b.step(6000)
    assert b.triggers.rules and all(r.count == 0 for r in b.triggers.rules)
    sa, sb = a.studies_snapshot(), b.studies_snapshot()
    assert sa["samples"] == sb["samples"] and sa["values"] == sb["values"] and sa["samples"]
    assert state(a) == state(b)


def test_the_book_studies_are_read_at_most_once_in_a_tenth_of_a_unit_of_the_clocks_time():
    m = session(11)
    m.triggers.create("book", [c("imbalance", "imbalance", "above", 5.0), c("spread", "z", "above", 99)])
    reads, real = Counter(), m._read_study

    def counted(key, book):
        reads[key, book] += 1
        return real(key, book)

    m._read_study = counted
    calls, _ = spy(m)
    m.step(2000)
    last, expected = None, 0
    for t in calls:  # the independent count: a read when none was made, or a tenth of a unit has passed
        if last is None or t - last >= 0.1:
            last, expected = t, expected + 1
    assert len(calls) == 2000 and reads["imbalance", True] == reads["spread", True] == expected
    assert 100 < expected < 2000  # more than a few, far fewer than one for each step
    # no study that no rule uses, and none through the other reader
    assert set(reads) == {("imbalance", True), ("spread", True)}


def test_a_study_that_cannot_be_read_switches_the_board_off_and_the_run_goes_on():
    m = session(5)
    rule = m.triggers.create("ofi", [c("ofi", "ofi", "above", -1e9)], cooldown=0)
    m.step(100)

    def broken(t=None, book=None):
        raise RuntimeError("a study broke")

    m.studies.set["ofi"].values = broken
    seen = rule.count
    m.step(50)  # the read raises inside _after_event: the board is suspended, the steps are not
    assert m.event_count > 100 and m.studies.failed is not None and m.studies.suspended
    assert "a study broke" in m.studies.failed and rule.count == seen
    assert any("switched themselves off" in n.text for n in m.notices)
    status = m.triggers_status()
    assert status["studies_off"] is True
    m.step(20)
    assert rule.count == seen  # nothing is checked while the studies are off


def test_a_rule_on_a_study_that_is_switched_off_waits_and_says_so():
    m = session(5)
    rule = m.triggers.create("v", [c("vpin", "vpin", "above", -1)], cooldown=0)
    m.configure_study("vpin", {}, enabled=False)
    m.step(500)
    status = m.triggers_status()
    assert rule.count == 0 and status["rules"][0]["state"] == "study off" and status["rules"][0]["conditions"][0]["off"]
    m.configure_study("vpin", {}, enabled=True)
    m.step(200)
    assert m.triggers_status()["rules"][0]["state"] == "warming up"  # N buckets are not full


# ---- Actions ----

def manual_book(asks, bids):
    """A session with the manual model and the book as given: [(price, lots)], the user's account flat."""
    m = session(model="manual")
    for side in (BID, ASK):
        for o in list(m.book.resting_orders(side)):
            m.book.cancel_order(o.order_id)
    for price, lots in bids:
        m.book.add_order(limit_order(price, lots, m.sim_time, BID))
    for price, lots in asks:
        m.book.add_order(limit_order(price, lots, m.sim_time, ASK))
    m._resync_studies()
    return m


def trigger_notices(m):
    return [n.text for n in m.notices if n.text.startswith("Trigger")]


def test_cancel_takes_every_order_of_the_users_and_no_one_elses():
    m = manual_book(asks=[(101, 50), (130, 9)], bids=[(99, 50)])
    mine = [m.submit_limit_order(p, 5, s)[0] for p, s in [(90, BID), (91, BID), (120, ASK)]]
    assert len(m.manual_order_ids) == 3
    others = {(side, o.price, o.size) for side in (BID, ASK) for o in m.book.resting_orders(side)
              if o.order_id not in m.manual_order_ids}
    events, clock = m.event_count, m.sim_time
    m.triggers.create("c", [ALWAYS], actions=["cancel", "notice"], one_shot=True)
    m._after_event()
    assert m.manual_order_ids == set() and all(m.book.get_order(o.order_id) is None for o in mine)
    assert {(side, o.price, o.size) for side in (BID, ASK) for o in m.book.resting_orders(side)} == others
    assert m.event_count == events + 3 and m.sim_time == clock + MANUAL_ACTION_DT  # three cancels, one manual action
    assert trigger_notices(m) == [
        "Trigger 'c' at time %.1f: CVD 0 lots is above -1,000,000,000. Canceled 3 orders." % clock]


def test_cancel_with_nothing_resting_says_so_and_does_no_work():
    m = manual_book(asks=[(101, 50)], bids=[(99, 50)])
    events, clock = m.event_count, m.sim_time
    m.triggers.create("c", [ALWAYS], actions=["cancel", "notice"])
    m._after_event()
    assert m.event_count == events and m.sim_time == clock
    assert trigger_notices(m)[0].endswith("Canceled 0 orders.")


def test_flatten_sells_what_the_protection_reaches_and_cancels_the_residue_at_once():
    m = manual_book(asks=[(101, 50), (102, 10)], bids=[(99, 10), (98, 10), (80, 5)])
    m.submit_market_order(50, BID)  # long 50 at 101
    assert m.account.position == 50
    m.triggers.create("f", [ALWAYS], actions=["flatten", "notice"], one_shot=True)
    m._after_event()
    # a protected market order walks 10 ticks from the best bid, 99: it reaches 89 and takes 99 (10) and 98 (10);
    # the 5 lots at 80 are out of reach; the other 30 lots would rest as an ask at 89, and are canceled
    assert m.account.position == 30
    assert m.manual_order_ids == set() and m.book.depth_at_price(89, ASK) == 0 and m.book.best_ask == 102
    assert m.book.depth_at_price(80, BID) == 5 and m.book.best_bid == 80
    assert trigger_notices(m)[0].endswith("Flatten sold 20 of 50 lots.")


def test_flatten_buys_back_a_short_position_and_reports_a_complete_fill():
    m = manual_book(asks=[(101, 20), (102, 20)], bids=[(99, 50)])
    m.submit_market_order(7, ASK)  # short 7
    assert m.account.position == -7
    m.triggers.create("f", [ALWAYS], actions=["flatten", "notice"], one_shot=True)
    m._after_event()
    assert m.account.position == 0 and m.manual_order_ids == set()
    assert m.book.depth_at_price(101, ASK) == 13
    assert trigger_notices(m)[0].endswith("Flatten bought 7 of 7 lots.")


def test_flatten_at_a_position_of_one_lot_sells_it_and_at_minus_one_buys_it_back():
    m = manual_book(asks=[(101, 50)], bids=[(99, 50)])
    m.submit_market_order(1, BID)
    assert m.account.position == 1
    assert m._flatten() == "flatten sold 1 of 1 lots" and m.account.position == 0
    m.submit_market_order(1, ASK)
    assert m.account.position == -1
    assert m._flatten() == "flatten bought 1 of 1 lots" and m.account.position == 0


def test_flatten_on_an_empty_far_side_places_nothing_and_says_so_and_when_flat_says_that():
    m = manual_book(asks=[(101, 50)], bids=[(99, 5)])
    m.submit_market_order(5, BID)
    for o in list(m.book.resting_orders(BID)):
        m.book.cancel_order(o.order_id)
    m._resync_studies()
    assert m.account.position == 5 and m.book.best_bid is None
    events, clock = m.event_count, m.sim_time
    m.triggers.create("f", [ALWAYS], actions=["flatten", "notice"], cooldown=0)
    m._after_event()
    assert m.account.position == 5 and m.event_count == events and m.sim_time == clock and not m.manual_order_ids
    assert trigger_notices(m)[0].endswith("Nothing to flatten against: the bid side is empty.")
    m2 = manual_book(asks=[(101, 5)], bids=[(99, 5)])
    events = m2.event_count
    m2.triggers.create("f", [ALWAYS], actions=["flatten", "notice"])
    m2._after_event()
    assert m2.event_count == events and trigger_notices(m2)[0].endswith("Already flat.")


def test_the_actions_of_a_rule_are_manual_actions_that_do_not_check_the_rules_again():
    m = manual_book(asks=[(101, 50)], bids=[(99, 50)])
    m.submit_market_order(5, BID)
    m.submit_limit_order(90, 5, BID)
    m.triggers.create("all", [ALWAYS], actions=["cancel", "flatten", "notice"], cooldown=0)
    calls, firings = spy(m)
    m._after_event()
    assert len(calls) == 1 and len(firings) == 1 and m.triggers.rules[0].count == 1  # not once for each action
    assert m._acting is False


def test_each_action_of_the_user_checks_the_rules_once_and_one_that_changes_nothing_does_not():
    m = manual_book(asks=[(101, 50), (140, 5)], bids=[(99, 50)])
    rule = m.triggers.create("watch", [ALWAYS], actions=["notice"], cooldown=0)
    m.submit_limit_order(90, 5, BID)
    assert rule.count == 1
    m.submit_market_order(3, BID)
    assert rule.count == 2
    (mine,) = [o for o in map(m.book.get_order, m.manual_order_ids) if o is not None]
    m.move_orders(mine.side, mine.price, mine.price - 1)
    assert rule.count == 3
    m.move_orders(mine.side, 12345, 5)  # nothing of yours there: nothing happened
    assert rule.count == 3
    m.start_split_order(BID, 4, 2, 5.0)
    assert rule.count == 4
    m.cancel_order_by_id(next(iter(m.manual_order_ids)))
    assert rule.count == 5
    m.submit_limit_order(88, 2, BID)
    assert rule.count == 6
    m.cancel_all_orders()
    assert rule.count == 7
    m.cancel_all_orders()  # none left
    assert rule.count == 7
    assert m.cancel_order_by_id(10**9) is None and rule.count == 7


def test_a_manual_session_says_it_needs_a_running_model_and_a_model_session_does_not():
    assert session(model="manual").triggers_status()["needs_model"] is True
    assert session().triggers_status()["needs_model"] is False


def test_a_side_of_the_book_that_is_empty_and_an_ask_that_is_not_depleted_are_not_warming_up():
    m = manual_book(asks=[(101, 5)], bids=[])
    for v in ("spread", "imbalance"):
        m.triggers.create(v, [c(v, v, "above", 5)])
    m._after_event()
    shown = m.triggers_status()["rules"]
    assert [(r["conditions"][0]["now"], r["state"], r["conditions"][0]["none"]) for r in shown] == [
        (None, "armed", "a side is empty")] * 2
    s = session(5)  # a market that runs: its ask is not depleted, its resilience has no timing of a recovery yet
    s.triggers.create("d", [c("resilience", "depleted_ask", "above", 1)])
    s.step(300)
    (rule,) = s.triggers_status()["rules"]
    assert rule["conditions"][0]["now"] is None and rule["state"] == "armed"
    assert rule["conditions"][0]["none"] == "the ask is not depleted"


def test_the_notices_of_a_rule_that_fires_at_every_step_are_one_a_second_and_the_next_has_the_count():
    clock = Clock()
    m = session(5, clock=clock)
    m.triggers.create("always", [ALWAYS], cooldown=0)
    m.step(50)
    assert len(trigger_notices(m)) == 1 and m.triggers.rules[0].count == 50 and m.triggers.pending
    clock.t += 1.5
    m.step(1)
    notices = trigger_notices(m)
    assert len(notices) == 2 and notices[1].endswith("(50 firings since the last notice)") and not m.triggers.pending
    clock.t += 3
    m.step(1)  # a rule that goes quiet is flushed too: nothing is held back at its next check
    assert len(trigger_notices(m)) == 3 and not trigger_notices(m)[2].endswith("firings since the last notice)")


def test_a_notice_held_back_is_sent_when_the_rule_is_quiet_for_a_second_even_if_it_is_disarmed():
    clock = Clock()
    m = session(5, clock=clock)
    rule = m.triggers.create("always", [ALWAYS], cooldown=0)
    m.step(10)
    assert len(trigger_notices(m)) == 1 and rule.unsent == 9
    m.triggers.arm(rule.id, False)
    m.step(5)
    assert len(trigger_notices(m)) == 1  # not yet a second
    clock.t += 1.1
    m.step(1)
    assert trigger_notices(m)[-1].endswith("(9 firings since the last notice)") and rule.unsent == 0


def test_a_pause_is_always_said_even_when_the_rule_spoke_a_moment_ago():
    """A paused simulation checks nothing, so a notice held back for the second would never come: the pause that the
    rule makes is explained by the notice that comes with it, whenever the rule last spoke."""
    clock = Clock()
    m = session(5, clock=clock)
    rule = m.triggers.create("p", [ALWAYS], cooldown=0, actions=["pause", "notice"], one_shot=False)
    m.is_running = True
    m.step(100)
    assert len(trigger_notices(m)) == 1 and not m.is_running and rule.count == 1
    clock.t += 0.3  # resumed by hand within the same second of wall time
    m.is_running = True
    m.step(100)
    notices = trigger_notices(m)
    assert not m.is_running and rule.count == 2 and len(notices) == 2 and notices[1].endswith("Paused.")
    assert rule.unsent == 0 and not m.triggers.pending
    # The same rule without the pause is held to the second.
    other = session(5, clock=clock)
    quiet = other.triggers.create("q", [ALWAYS], cooldown=0, actions=["notice"])
    other.step(3)
    assert len(trigger_notices(other)) == 1 and quiet.unsent == 2


# ---- Playback ----

def recording_of_a_session():
    m = session(5)
    m.step(1500)
    m.start_recording()
    m.step(500)
    m.submit_limit_order(m.book.best_bid - 3, 10, BID)  # a resting order of the user's
    m.step(300)
    m.submit_market_order(15, BID)  # and a position
    m.step(700)
    expected = state(m)
    return m.stop_recording(), expected


@pytest.fixture(scope="module")
def recorded():
    return recording_of_a_session()


def playing(recording, clock=None):
    p = session(1, clock=clock or Clock())
    p.start_playback(recording)
    return p


def test_in_a_playback_the_notice_and_the_pause_act_and_cancel_and_flatten_do_not(recorded):
    recording, expected = recorded
    assert expected["you"]["position"] != 0 and any(row[-1] == "you" for row in expected["book"])  # something to lose
    p = playing(recording)
    p.triggers.create("all", [ALWAYS], cooldown=100_000, actions=["cancel", "flatten", "pause", "notice"],
                      one_shot=False)
    start = p.event_count
    p.is_running = True
    p.step(100)
    assert p.event_count == start + 1 and p.is_running is False and p._halt  # paused after the first book event
    assert trigger_notices(p)[0].endswith("Cancel is off in a playback; flatten is off in a playback; paused.")
    status = p.triggers_status()
    assert status["playback"] is True and status["rules"][0]["count"] == 1
    p.step(10**9)  # to the end: the rule is in its cooldown, and what it was told to do was never done
    assert p.playback.ended and p.triggers.rules[0].count == 1
    assert state(p) == expected  # your resting order and your position are the recording's


def test_a_playback_with_a_rule_that_fires_at_every_event_still_reproduces_the_session(recorded):
    recording, expected = recorded
    p = playing(recording)
    p.triggers.create("always", [ALWAYS], cooldown=0, actions=["cancel", "flatten", "notice"])
    p.step(10**9)
    assert p.triggers.rules[0].count > 1000 and state(p) == expected


def test_a_rule_made_in_a_playback_acts_from_the_playbacks_start_and_the_count_is_that_of_the_playback(recorded):
    recording, _ = recorded
    p = playing(recording)
    p.triggers.create("cross", [c("cvd", "cvd", "crosses_above", 0)], cooldown=0)
    p.step(10**9)
    live = session(5)
    live.triggers.create("cross", [c("cvd", "cvd", "crosses_above", 0)], cooldown=0)
    live.step(3000)
    assert p.triggers.rules[0].count >= 1 and p.triggers.rules[0].count != live.triggers.rules[0].count


def test_after_leaving_a_playback_the_actions_are_on_again(recorded):
    recording, _ = recorded
    p = playing(recording)
    p.step(400)
    p.exit_playback()
    assert p.triggers_status()["playback"] is False
    order, _ = p.submit_limit_order(p.book.best_bid - 5, 4, BID)
    p.triggers.create("c", [ALWAYS], actions=["cancel", "notice"], one_shot=True)
    p._after_event()
    assert p.book.get_order(order.order_id) is None and p.manual_order_ids == set()


def test_nothing_acts_during_a_seek_and_the_rules_start_again_at_the_target(recorded):
    recording, _ = recorded
    p = playing(recording)
    p.triggers.create("watch", [ALWAYS], cooldown=100_000, actions=["notice"])
    p.triggers.create("stop", [ALWAYS], cooldown=0, actions=["pause", "cancel", "flatten"], one_shot=False)
    calls, firings = spy(p)
    p.is_running = True
    p.step(3)
    # both fire, and one pauses the batch
    assert len(calls) == 1 and len(firings) == 2 and p.is_running is False and p._halt
    assert p.triggers.rules[0].count == 1 and p.triggers.rules[1].count == 1
    seen = p.notice_seq
    target = recording.tape.time[1500]
    asyncio.run(p.seek_playback(target))
    assert len(calls) == 1 and len(firings) == 2  # not one check in the whole seek
    assert p.playback.cursor >= 1500 and p.notice_seq == seen and not p._halt and p.is_running is False
    for rule in p.triggers.rules:
        assert (rule.count, rule.last_fired, rule.previous, rule.unsent) == (0, None, [None], 0)
    assert list(p.triggers.fired) == []
    p.step(1)  # the watcher's cooldown of 100,000 is not still running: it fires at once
    assert p.triggers.rules[0].count == 1


def test_a_rewind_and_a_new_playback_clear_what_the_rules_saw_and_keep_the_rules(recorded):
    recording, _ = recorded
    p = playing(recording)
    rule = p.triggers.create("watch", [ALWAYS], cooldown=5, actions=["notice"])
    p.step(200)
    assert rule.count > 0 and rule.last_fired is not None and list(p.triggers.fired)
    before = (rule.id, rule.name, rule.describe(), rule.armed, rule.cooldown, rule.actions)
    p.rewind_playback()
    assert (rule.count, rule.last_fired, rule.previous, rule.evaluated) == (0, None, [None], False)
    assert list(p.triggers.fired) == [] and before == (rule.id, rule.name, rule.describe(), rule.armed, rule.cooldown,
                                                       rule.actions)
    p.step(200)
    assert rule.count > 0
    p.start_playback(recording)
    assert (rule.count, rule.last_fired) == (0, None) and p.triggers.rules == [rule]


# ---- A new market: what the rules saw is cleared, what they are is not ----

def test_a_reset_and_the_end_of_a_collection_clear_what_the_rules_saw_and_keep_their_definitions():
    m = session(5)
    always = m.triggers.create("watch", [ALWAYS, c("ofi", "ofi", "above", -1e9)], "any", cooldown=7,
                               actions=["notice", "cancel"], armed=True)
    one = m.triggers.create("one", [ALWAYS], cooldown=0, actions=["pause"])
    m.step(100)
    assert one.armed is False and always.count > 0 and always.last_fired is not None and always.previous != [None, None]
    marks = list(m.triggers.fired)
    assert marks
    definition = [(r.id, r.name, r.describe(), r.armed, r.cooldown, r.actions, r.one_shot) for r in m.triggers.rules]
    m.reset(100)
    for rule in m.triggers.rules:
        assert (rule.count, rule.last_fired, rule.previous, rule.values, rule.evaluated, rule.unsent) == (
            0, None, [None] * len(rule.conditions), [None] * len(rule.conditions), False, 0)
    assert list(m.triggers.fired) == []
    assert definition == [(r.id, r.name, r.describe(), r.armed, r.cooldown, r.actions, r.one_shot)
                          for r in m.triggers.rules]  # a one-shot rule that fired stays disarmed
    m.step(50)
    assert always.count > 0
    asyncio.run(m.collect(1000, chunk=250))
    assert always.count == 0 and always.last_fired is None and list(m.triggers.fired) == []
    m.step(5)
    assert always.count > 0


def test_nothing_fires_during_a_collection_and_the_first_check_after_it_has_no_value_before():
    m = session(5)
    rule = m.triggers.create("cross", [c("cvd", "cvd", "crosses_above", 0)], cooldown=0)
    calls, _ = spy(m)
    asyncio.run(m.collect(1500, chunk=500))
    assert calls == [] and rule.count == 0 and rule.previous == [None]


def test_a_session_whose_scratch_copies_have_no_board_is_untouched_by_triggers():
    scratch = LOBManager()
    scratch.step(100)
    scratch.submit_limit_order(50, 5, BID)
    assert scratch.triggers is None and scratch.triggers_status() == {
        "rules": [], "marks": [], "limit": 0, "needs_model": False, "playback": False, "seeking": False,
        "studies_off": True}


# ---- Routes and the stream ----

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


def strict(text):
    def refuse(token):
        raise AssertionError(f"{token} in JSON")
    return json.loads(text, parse_constant=refuse)


def test_a_rule_is_made_listed_changed_disarmed_armed_and_deleted_through_the_routes(client, manager):
    assert client.get("/api/triggers").json()["rules"] == []
    made = client.post("/api/triggers", data=GOOD)
    assert made.status_code == 200 and 'data-trigger-id="1"' in made.text
    assert "Added <b>OFI burst</b>: OFI crosses above 500 lots." in made.text
    listed = strict(client.get("/api/triggers").text)
    (rule,) = listed["rules"]
    assert (rule["id"], rule["name"], rule["armed"], rule["one_shot"], rule["cooldown"], rule["actions"]) == (
        1, "OFI burst", True, True, 5.0, ["notice", "pause"])
    assert listed["limit"] == MAX_RULES and listed["needs_model"] is False and listed["playback"] is False
    assert manager.triggers.rules[0].conditions == (Condition("ofi", "ofi", "crosses_above", 500.0),)

    changed = client.post("/api/triggers/1", data={**GOOD, "name": "OFI burst 2", "threshold": ["800"],
                                                   "action": ["notice"], "one_shot": "off", "cooldown": "20"})
    assert changed.status_code == 200 and "Saved <b>OFI burst 2</b>" in changed.text
    (rule,) = client.get("/api/triggers").json()["rules"]
    assert (rule["name"], rule["one_shot"], rule["cooldown"], rule["conditions"][0]["threshold"]) == (
        "OFI burst 2", False, 20.0, 800.0)
    assert rule["armed"] is True  # an edit leaves it armed

    assert client.post("/api/triggers/1/disarm").status_code == 204 and manager.triggers.rules[0].armed is False
    assert client.post("/api/triggers/1/arm").status_code == 204 and manager.triggers.rules[0].armed is True
    second = client.post("/api/triggers", data={**GOOD, "name": "second", "armed": "off"})
    assert 'data-trigger-id="2"' in second.text and manager.triggers.rules[1].armed is False
    assert client.post("/api/triggers/1/delete").status_code == 204
    assert [r["id"] for r in client.get("/api/triggers").json()["rules"]] == [2]


@pytest.mark.parametrize("change, words", [
    ({"value": [], "op": [], "threshold": []}, "at least one condition"),
    ({"value": ["ofi:nope"]}, "No value"),
    ({"threshold": ["abc"]}, "must be a number"),
    ({"cooldown": "-3"}, "cooldown must be between"),
    ({"action": []}, "at least one action"),
    ({"value": ["ofi:ofi", "cvd:cvd"], "op": ["crosses_above", "crosses_above"], "threshold": ["1", "1"]},
     "only one crossing"),
    ({"name": "x" * 50}, "at most 40"),
])
def test_a_rule_that_cannot_be_made_is_refused_with_422_and_its_reason_in_a_fragment(client, manager, change, words):
    answer = client.post("/api/triggers", data={**GOOD, **change})
    assert answer.status_code == 422 and words in answer.text and answer.text.startswith('<span class="text-rose-400"')
    assert manager.triggers.rules == []
    client.post("/api/triggers", data=GOOD)
    answer = client.post("/api/triggers/1", data={**GOOD, **change})
    assert answer.status_code == 422 and words in answer.text
    assert manager.triggers.rules[0].name == "OFI burst"  # unchanged


def test_the_reason_a_rule_is_refused_is_escaped_in_the_fragment(client):
    answer = client.post("/api/triggers", data={**GOOD, "value": ["<b>:x"]})
    assert answer.status_code == 422 and "&lt;b&gt;:x" in answer.text and "<b>:x" not in answer.text


def test_a_thirteenth_rule_is_refused_and_so_is_one_that_does_not_exist_for_an_action_or_a_change(client):
    for i in range(MAX_RULES):
        assert client.post("/api/triggers", data={**GOOD, "name": f"r{i}"}).status_code == 200
    answer = client.post("/api/triggers", data=GOOD)
    assert answer.status_code == 422 and "12 rules already" in answer.text
    for action in ("arm", "disarm", "delete"):
        assert client.post(f"/api/triggers/99/{action}").status_code == 404
    assert client.post("/api/triggers/99", data=GOOD).status_code == 404
    assert client.post("/api/triggers/1/explode").status_code == 422


def test_a_rule_made_in_the_manual_model_says_that_it_is_checked_only_after_your_orders(client):
    client.post("/api/sim/config", data={"model": "manual"})
    answer = client.post("/api/triggers", data=GOOD)
    assert answer.status_code == 200 and "takes no steps" in answer.text


def test_the_page_has_the_editor_with_every_value_operator_and_action_and_the_rules_so_far(client):
    client.post("/api/triggers", data=GOOD)
    page = client.get("/").text
    assert 'id="triggers-card"' in page and 'id="trigger-form"' in page and "lob-triggers.js" in page
    for v in TRIGGER_VALUES:
        assert f'value="{v.study}:{v.name}"' in page
    for op in OPERATORS:
        assert f'<option value="{op}">' in page
    for action in ACTIONS:
        assert f'name="action" value="{action}"' in page
    start = page.index('id="triggers-card" data-initial=\'') + len('id="triggers-card" data-initial=\'')
    initial = json.loads(page[start:page.index("'", start)])
    assert [r["name"] for r in initial["rules"]] == ["OFI burst"]  # what the page starts from
    assert "true only on the event it happens" in page  # the edge rule of a crossing, stated next to the operator


def events_of(lines):
    """(name, data) of each event of a stream; the JSON events parsed, refusing NaN and Infinity as a browser does."""
    event = None
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: ") and event:
            data = line[len("data: "):]
            yield event, strict(data) if event in ("studies", "triggers", "tape") else data
            event = None


def read_until(events, wanted):
    """The events from here to the first for which ``wanted(name, data)`` holds, that one last."""
    seen = []
    for name, data in events:
        seen.append((name, data))
        if wanted(name, data):
            return seen
    raise AssertionError("the stream ended")


def test_the_stream_sends_the_rules_status_on_connect_and_after_that_only_when_it_changed(client, manager, deadline):
    client.post("/api/triggers", data={"name": "level", "value": ["cvd:cvd"], "op": ["above"],
                                       "threshold": ["1e9"], "action": ["notice"], "cooldown": "1"})
    with client.stream("GET", "/events") as resp:
        events = events_of(resp.iter_lines())
        sent = read_until(events, lambda name, data: name == "triggers")[-1][1]
        assert [r["name"] for r in sent["rules"]] == ["level"] and sent["rules"][0]["state"] == "armed"

        client.post("/api/triggers/1/disarm")  # a change: sent
        sent = read_until(events, lambda name, data: name == "triggers" and not data["rules"][0]["armed"])[-1][1]
        assert sent["rules"][0]["state"] == "disarmed"
        pushes = sent_again = 0
        for _ in range(5):  # five more pushes of the panels, none of which changes what the rules are or have seen
            client.post("/api/sim/control", data={"action": "step", "n": 1})
            seen = read_until(events, lambda name, data: name == "studies")
            pushes += 1
            sent_again += sum(name == "triggers" for name, _ in seen)
        assert pushes == 5 and sent_again == 0

        client.post("/api/triggers/1/arm")
        sent = read_until(events, lambda name, data: name == "triggers")[-1][1]
        assert sent["rules"][0]["armed"] is True and sent["rules"][0]["conditions"][0]["now"] is None
        client.post("/api/sim/control", data={"action": "step", "n": 800})  # armed: it reads the CVD, which moved
        sent = read_until(events, lambda name, data: name == "triggers" and data["rules"][0]["conditions"][0]["now"]
                          is not None)[-1][1]
        assert sent["rules"][0]["conditions"][0]["now"] == manager.studies.cvd.value() != 0


def test_a_firing_reaches_the_page_as_a_mark_for_the_chart_and_the_status_counts_it(client, manager, deadline):
    client.post("/api/triggers", data={"name": "any fill", "value": ["cvd:cvd"], "op": ["crosses_above"],
                                       "threshold": ["0"], "action": ["notice"], "cooldown": "0"})
    with client.stream("GET", "/events") as resp:
        events = events_of(resp.iter_lines())
        read_until(events, lambda name, data: name == "triggers")
        client.post("/api/sim/control", data={"action": "step", "n": 1500})
        sent = read_until(events, lambda name, data: name == "triggers" and data["rules"][0]["count"] > 0)[-1][1]
    (rule,) = sent["rules"]
    board_rule = manager.triggers.rules[0]
    assert rule["count"] == board_rule.count > 0 and rule["fired_at"] == board_rule.last_fired
    assert sent["marks"] and all(m[1] == 1 and m[2] == "any fill" for m in sent["marks"])
    assert [m[0] for m in sent["marks"]] == sorted(m[0] for m in sent["marks"]) and len(sent["marks"]) <= 30
    assert sent["marks"][-1][0] == rule["fired_at"]


def test_the_status_never_holds_a_number_json_cannot(client, manager):
    for study, name in [("ofi", "ofi"), ("vpin", "vpin"), ("otr", "otr"), ("imbalance", "imbalance"),
                        ("spread", "z"), ("resilience", "score")]:
        client.post("/api/triggers", data={"value": [f"{study}:{name}"], "op": ["above"], "threshold": ["0"],
                                           "action": ["notice"], "cooldown": "0"})
    manager.step(3000)
    status = strict(client.get("/api/triggers").text)
    assert len(status["rules"]) == 6 and any(r["conditions"][0]["now"] is not None for r in status["rules"])


def test_the_stream_survives_a_session_that_runs_with_rules_armed(client, manager):
    client.post("/api/triggers", data={**GOOD, "action": ["notice"]})
    client.post("/api/triggers", data={"name": "spread", "value": ["spread:spread"], "op": ["above"],
                                       "threshold": ["1"], "action": ["notice"], "cooldown": "1"})
    manager.speed = 2000

    async def run():
        await manager.start_sim_loop()
        await asyncio.sleep(0.5)
        await manager.stop_sim_loop()

    asyncio.run(run())
    assert manager.event_count > 100 and manager.studies.failed is None
    assert strict(client.get("/api/triggers").text)["rules"][1]["count"] > 0


def test_notice_actions_reach_the_page_through_the_notices_event(client, manager):
    client.post("/api/triggers", data={"name": "n", "value": ["cvd:cvd"], "op": ["above"], "threshold": ["-1e9"],
                                       "action": ["notice"], "cooldown": "5"})
    manager.step(10)
    assert [n.text for n in manager.notices][0].startswith("Trigger 'n' at time ")


def test_the_limits_the_editor_holds_are_the_modules(client):
    page = client.get("/").text
    assert f'data-limit="{MAX_RULES}"' in page and f'data-max-conditions="{MAX_CONDITIONS}"' in page
