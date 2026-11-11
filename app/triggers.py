"""Triggers: rules on the studies' values that notify, pause the simulation, cancel your orders or flatten you.

A rule has conditions, each on one value of one study (``TRIGGER_VALUES``: the keys of the studies' snapshot, app/
studies.py), and fires when all of them hold (``all``) or any does (``any``): no mixed precedence. A condition is

- ``above`` or ``below`` a threshold: true at every evaluation while it holds, or
- ``crosses_above`` (the previous value was <= the threshold and the value now is above it) or ``crosses_below``
  (previous >= threshold, now below): true only at the evaluation where it happens.

A value that is ``None`` (a study warming up, a side of the book empty) on either side is never true, and is
remembered as ``None``, so no crossing is read across it: the first value after it can only be a "previous". A
crossing is true for one evaluation, so an ``all`` rule holds at most one crossing condition (two studies do not
cross at the same event), and ``validate`` refuses more. Comparing with the previous *evaluated* value is what this
does, and VisualHFT, whose trigger engine this follows (Apache License 2.0), compares with the previous published
one; VisualHFT combines with AND only.

After a rule fires it is quiet for its ``cooldown``, in units of simulation time, not of wall time (a cooldown of 5
is a sixth of a second of wall time at the dashboard's top speed). A ``one_shot`` rule disarms itself when it fires
(the default for a rule that pauses: a level condition would otherwise pause again as soon as its cooldown passed,
since the simulation's clock stands still while it is paused). Its actions:

``notice``  a message (at most one a second of wall time for a rule, carrying how many firings it stands for;
            a rule that also pauses always says so),
``pause``   stop the simulation, at once, even in the middle of a batch of steps,
``cancel``  cancel all your resting orders,
``flatten`` a protected market order for minus your position, and what rests of it canceled at once.

In a playback only ``notice`` and ``pause`` act: your orders and cancels of the recorded session are on the tape
already, so a rule that traded again would double them.

``TriggerBoard`` is the pure part: it holds the rules and decides, from the values it is given, which fire; the
manager (app/manager.py ``_after_event``) reads the values, runs the actions and says so. A board is given the
studies' values by two functions, ``flow(study)`` and ``book(study)``, each returning the study's ``values()`` (a
dict) or ``None`` when it is off, so that it can be tried by hand. Flow studies are read at every evaluation; the
studies that read the book (imbalance, spread) at most once in ``BOOK_EVERY`` units of simulation time, and only
when an armed rule uses them.

Rule definitions last as long as the session does; what a rule has seen (its previous values, when it last fired,
its count) is cleared when the market starts over (``clear_runtime``).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

from app.studies import STUDY_LABELS

MAX_RULES = 12
MAX_CONDITIONS = 4
COOLDOWN_MAX = 100_000.0
DEFAULT_COOLDOWN = 10.0
NAME_MAX = 40
THRESHOLD_MAX = 1e12
BOOK_EVERY = 0.1  # units of simulation time between reads of the studies that look at the book
NOTICE_EVERY = 1.0  # seconds of wall time between a rule's notices
MAX_MARKS = 30  # firings kept for the chart's markers

OPERATORS = {"above": "above", "below": "below", "crosses_above": "crosses above", "crosses_below": "crosses below"}
PAST = {"above": "is above", "below": "is below", "crosses_above": "crossed above", "crosses_below": "crossed below"}
CROSSINGS = ("crosses_above", "crosses_below")
COMBINES = {"all": "all of", "any": "any of"}
ACTIONS = {"notice": "show a notice", "pause": "pause the simulation", "cancel": "cancel all my orders",
           "flatten": "flatten my position"}
# What each operator means, for the editor, next to the operator.
EDGE_RULES = {
    "above": "true at every event while it holds",
    "below": "true at every event while it holds",
    "crosses_above": "true only on the event it happens: the value before was at or below the level, now above it",
    "crosses_below": "true only on the event it happens: the value before was at or above the level, now below it",
}


@dataclass(frozen=True)
class TriggerValue:
    """One value of one study a condition can read.

    ``book``: it reads the book as it stands (read at most once in ``BOOK_EVERY`` units), not the flow of events.
    ``warms``: ``None`` means the study has not enough to say yet (a window not full), as against a value that is
    absent for good reason (nothing is depleted). ``none`` says what a ``None`` is, for the status.
    """
    study: str
    name: str
    label: str
    unit: str = ""
    book: bool = False
    warms: bool = True
    none: str = "warming up"


TRIGGER_VALUES: tuple[TriggerValue, ...] = (
    TriggerValue("ofi", "ofi", "OFI", "lots"),
    TriggerValue("cvd", "cvd", "CVD", "lots", warms=False, none="no value"),
    TriggerValue("delta", "delta", "Delta of trades", "lots"),
    TriggerValue("vpin", "vpin", "VPIN"),
    TriggerValue("otr", "otr", "Order-to-trade ratio", "msgs/trade", none="warming up, or no trades in the window"),
    TriggerValue("otr", "tto", "Trade-to-order volume", none="warming up, or no orders in the window"),
    TriggerValue("imbalance", "imbalance", "Book imbalance", book=True, warms=False, none="a side is empty"),
    TriggerValue("spread", "spread", "Spread", "ticks", book=True, warms=False, none="a side is empty"),
    TriggerValue("spread", "z", "Spread z-score", book=True, none="warming up, or the spread has not varied"),
    TriggerValue("resilience", "score", "Resilience score", none="a side is depleted and no recovery is timed yet"),
    TriggerValue("resilience", "depleted_bid", "Bid depleted for", "units", warms=False,
                 none="the bid is not depleted"),
    TriggerValue("resilience", "depleted_ask", "Ask depleted for", "units", warms=False,
                 none="the ask is not depleted"),
)
VALUES: dict[tuple[str, str], TriggerValue] = {(v.study, v.name): v for v in TRIGGER_VALUES}
# The values to choose from, grouped by study for a menu: (study key, the study's name, its values).
VALUE_GROUPS: tuple[tuple[str, str, tuple[TriggerValue, ...]], ...] = tuple(
    (key, STUDY_LABELS[key], tuple(v for v in TRIGGER_VALUES if v.study == key))
    for key in dict.fromkeys(v.study for v in TRIGGER_VALUES))


def g(x: float) -> str:
    """A number as short as it reads right: 5.0 as 5, 1234.5678 as 1,234.57, 0.00012 as 0.00012."""
    if x == 0:
        return "0"
    if abs(x) >= 100:
        text = f"{x:,.2f}"
    elif abs(x) >= 1:
        text = f"{x:.3f}"
    else:
        text = f"{x:.5g}"
    return text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(frozen=True)
class Condition:
    """A study's value, an operator and a threshold."""
    study: str
    name: str
    op: str
    threshold: float

    @property
    def value(self) -> TriggerValue:
        return VALUES[(self.study, self.name)]

    def met(self, previous: Optional[float], now: Optional[float]) -> bool:
        """Whether it holds for ``now``, given the value at the evaluation before (None: there was none)."""
        if now is None:
            return False
        h = self.threshold
        if self.op == "above":
            return now > h
        if self.op == "below":
            return now < h
        if previous is None:
            return False
        if self.op == "crosses_above":
            return previous <= h and now > h
        return previous >= h and now < h  # crosses_below

    def describe(self) -> str:
        v = self.value
        return f"{v.label} {OPERATORS[self.op]} {g(self.threshold)}{' ' + v.unit if v.unit else ''}"


def validate(name: str, conditions: Iterable[Condition], combine: str, cooldown: float, actions: Iterable[str]) -> None:
    """Raise ValueError, saying what is wrong, for a rule that cannot be made."""
    conditions, actions = tuple(conditions), tuple(actions)
    if len(name) > NAME_MAX:
        raise ValueError(f"The name can have at most {NAME_MAX} characters.")
    if not conditions:
        raise ValueError("A rule needs at least one condition.")
    if len(conditions) > MAX_CONDITIONS:
        raise ValueError(f"A rule can have at most {MAX_CONDITIONS} conditions.")
    for c in conditions:
        if (c.study, c.name) not in VALUES:
            raise ValueError(f"No value {c.name!r} of the study {c.study!r}.")
        if c.op not in OPERATORS:
            raise ValueError(f"No operator {c.op!r}: choose one of {', '.join(OPERATORS)}.")
        if not math.isfinite(c.threshold) or abs(c.threshold) > THRESHOLD_MAX:
            raise ValueError(f"The level of {c.value.label} must be a number below {THRESHOLD_MAX:g} in size.")
    if combine not in COMBINES:
        raise ValueError(f"Combine with 'all' or 'any', not {combine!r}.")
    crossing = [c for c in conditions if c.op in CROSSINGS]
    if combine == "all" and len(crossing) > 1:
        raise ValueError("A crossing is true only on the event it happens, and two studies do not cross on one event: "
                         "an all-of rule can hold only one crossing (combine them with any of, or make the others "
                         "above or below).")
    if not math.isfinite(cooldown) or not 0 <= cooldown <= COOLDOWN_MAX:
        raise ValueError(f"The cooldown must be between 0 and {COOLDOWN_MAX:,.0f} units of simulation time.")
    if not actions:
        raise ValueError("Choose at least one action.")
    unknown = [a for a in actions if a not in ACTIONS]
    if unknown:
        raise ValueError(f"No action {unknown[0]!r}: choose from {', '.join(ACTIONS)}.")


@dataclass
class Rule:
    """A rule: what it is (its definition, which lasts) and what it has seen (its runtime, cleared with the market)."""
    id: int
    name: str
    conditions: tuple[Condition, ...]
    combine: str = "all"
    cooldown: float = DEFAULT_COOLDOWN
    armed: bool = True
    one_shot: bool = False
    actions: tuple[str, ...] = ("notice",)
    # What it has seen
    previous: list[Optional[float]] = field(default_factory=list)  # each condition's value at the last evaluation
    values: list[Optional[float]] = field(default_factory=list)  # the same, for the status (None before the first)
    flags: list[bool] = field(default_factory=list)  # which conditions held at the last evaluation
    evaluated: bool = False
    last_fired: Optional[float] = None  # simulation time
    count: int = 0
    unsent: int = 0  # firings the notices have not yet counted
    notice_wall: Optional[float] = None  # wall time of the last notice
    latest: str = ""  # the text of the latest firing that no notice has shown

    def __post_init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        n = len(self.conditions)
        self.previous = [None] * n
        self.values = [None] * n
        self.flags = [False] * n
        self.evaluated = False
        self.last_fired = None
        self.count = 0
        self.unsent = 0
        self.notice_wall = None
        self.latest = ""

    def describe(self) -> str:
        joiner = " and " if self.combine == "all" else " or "
        return joiner.join(c.describe() for c in self.conditions)


@dataclass
class Firing:
    """A rule firing at simulation time ``t``, with the conditions that held and their values."""
    rule: Rule
    t: float
    parts: tuple[tuple[Condition, float], ...]

    def describe(self) -> str:
        return ", ".join(f"{c.value.label} {g(v)}{' ' + c.value.unit if c.value.unit else ''} {PAST[c.op]} "
                         f"{g(c.threshold)}" for c, v in self.parts)


class TriggerBoard:
    """The rules of a session and the decision of which fire."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.rules: list[Rule] = []
        self.clock = clock  # wall time, for the notices' rate (a test gives its own)
        self._seq = 0
        self.fired: deque[tuple[float, int, str]] = deque(maxlen=MAX_MARKS)  # (simulation time, rule id, its name)
        self.armed: list[Rule] = []
        self._flow: tuple[str, ...] = ()
        self._book: tuple[str, ...] = ()
        self._book_t: Optional[float] = None
        self._book_values: dict[str, Optional[Mapping[str, Any]]] = {}

    # ---- Definitions ----

    def get(self, rule_id: int) -> Rule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise LookupError(f"No rule {rule_id}")

    def create(self, name: str, conditions: Iterable[Condition], combine: str = "all",
               cooldown: float = DEFAULT_COOLDOWN, actions: Iterable[str] = ("notice",),
               one_shot: Optional[bool] = None, armed: bool = True) -> Rule:
        """Add a rule (ValueError, with the reason, if it cannot be made). ``one_shot`` None: when it pauses."""
        conditions, actions = tuple(conditions), tuple(dict.fromkeys(actions))
        validate(name, conditions, combine, cooldown, actions)
        if len(self.rules) >= MAX_RULES:
            raise ValueError(f"There are {MAX_RULES} rules already: delete one first.")
        self._seq += 1
        rule = Rule(self._seq, name, conditions, combine, float(cooldown), armed,
                    "pause" in actions if one_shot is None else one_shot, actions)
        rule.name = name or rule.describe()[:NAME_MAX].rstrip()
        self.rules.append(rule)
        self._refit()
        return rule

    def update(self, rule_id: int, name: str, conditions: Iterable[Condition], combine: str = "all",
               cooldown: float = DEFAULT_COOLDOWN, actions: Iterable[str] = ("notice",),
               one_shot: Optional[bool] = None, armed: Optional[bool] = None) -> Rule:
        """Change a rule's definition: what it has seen is cleared, since it saw it under another one."""
        rule = self.get(rule_id)
        conditions, actions = tuple(conditions), tuple(dict.fromkeys(actions))
        validate(name, conditions, combine, cooldown, actions)
        rule.conditions, rule.combine, rule.cooldown, rule.actions = conditions, combine, float(cooldown), actions
        rule.name = name or rule.describe()[:NAME_MAX].rstrip()
        rule.one_shot = "pause" in actions if one_shot is None else one_shot
        if armed is not None:
            rule.armed = armed
        rule.clear()
        self._forget(rule)
        self._refit()
        return rule

    def arm(self, rule_id: int, on: bool = True) -> Rule:
        """Arm or disarm a rule. Arming forgets what it has seen: a crossing is not read across the time it was off."""
        rule = self.get(rule_id)
        if on:
            rule.clear()
            self._forget(rule)
        rule.armed = on
        self._refit()
        return rule

    def delete(self, rule_id: int) -> None:
        rule = self.get(rule_id)
        self.rules.remove(rule)
        self._forget(rule)
        self._refit()

    def _forget(self, rule: Rule) -> None:
        """The chart's markers of a rule's firings go with its count: it is a rule that has not fired, now."""
        if any(mark[1] == rule.id for mark in self.fired):
            kept = [mark for mark in self.fired if mark[1] != rule.id]
            self.fired.clear()
            self.fired.extend(kept)

    def clear_runtime(self) -> None:
        """What every rule has seen is forgotten (the market started over); the definitions stay."""
        for rule in self.rules:
            rule.clear()
        self.fired.clear()
        self._book_t = None

    def _refit(self) -> None:
        """The armed rules, and the studies they read, split into those read at every evaluation and the book's."""
        self.armed = [rule for rule in self.rules if rule.armed]
        used = {(c.study, c.value.book) for rule in self.armed for c in rule.conditions}
        self._flow = tuple(sorted(study for study, book in used if not book))
        self._book = tuple(sorted(study for study, book in used if book))
        self._book_t = None

    # ---- Evaluation ----

    def evaluate(self, t: float, flow: Callable[[str], Optional[Mapping[str, Any]]],
                 book: Callable[[str], Optional[Mapping[str, Any]]]) -> list[Firing]:
        """Evaluate the armed rules at simulation time ``t``; returns the firings, in the rules' order.

        ``flow(study)`` and ``book(study)`` give a study's values (None: it is off). A rule that fires is quiet for
        its cooldown, and a one-shot rule is disarmed.
        """
        armed = self.armed
        if not armed:
            return []
        values: dict[str, Optional[Mapping[str, Any]]] = {study: flow(study) for study in self._flow}
        if self._book:
            if self._book_t is None or not 0 <= t - self._book_t < BOOK_EVERY:  # time went back: read afresh too
                self._book_values = {study: book(study) for study in self._book}
                self._book_t = t
            values.update(self._book_values)
        firings: list[Firing] = []
        spent = False
        for rule in armed:
            conditions = rule.conditions
            previous, now, flags = rule.previous, rule.values, rule.flags
            for i, c in enumerate(conditions):
                row = values.get(c.study)
                v = None if row is None else row.get(c.name)
                if v is not None and not math.isfinite(v):
                    v = None
                flags[i] = c.met(previous[i], v)
                previous[i] = now[i] = v
            rule.evaluated = True
            if not (all(flags) if rule.combine == "all" else any(flags)):
                continue
            last = rule.last_fired
            if last is not None:
                if t < last:  # the clock went back without the rule being told: its cooldown is of another run
                    rule.last_fired = None
                elif t - last < rule.cooldown:
                    continue
            rule.last_fired = t
            rule.count += 1
            self.fired.append((t, rule.id, rule.name))
            firings.append(Firing(rule, t, tuple((c, now[i]) for i, c in enumerate(conditions) if flags[i])))
            if rule.one_shot:
                rule.armed = False
                spent = True
        if spent:
            self._refit()
        return firings

    # ---- Notices ----

    def notice(self, rule: Rule, text: str, urgent: bool = False) -> Optional[str]:
        """The notice to show for a firing of ``rule``, or None: at most one in ``NOTICE_EVERY`` seconds of wall time.

        What is held back is counted, and carried by the next notice (``flush`` sends it when the rule is quiet).
        ``urgent`` goes out at once, with the count of the firings held back: the manager says so for a rule that
        pauses, since a paused simulation evaluates nothing, so a held notice would wait for a run that cannot start
        until the pause is explained, and a pause cannot flood (the simulation is started again by hand).
        """
        now = self.clock()
        rule.unsent += 1
        if urgent or rule.notice_wall is None or now - rule.notice_wall >= NOTICE_EVERY:
            return self._send(rule, text, now)
        rule.latest = text
        return None

    def _send(self, rule: Rule, text: str, now: float) -> str:
        n = rule.unsent
        rule.unsent, rule.notice_wall, rule.latest = 0, now, ""
        return text if n == 1 else f"{text} ({n:,} firings since the last notice)"

    @property
    def pending(self) -> bool:
        """Whether any rule holds firings that no notice has counted."""
        return any(rule.unsent for rule in self.rules)

    def flush(self) -> list[str]:
        """The notices for firings held back, of the rules whose last notice is more than ``NOTICE_EVERY`` old."""
        out: list[str] = []
        now = None
        for rule in self.rules:
            if rule.unsent:
                now = self.clock() if now is None else now
                if now - (rule.notice_wall or 0.0) >= NOTICE_EVERY:
                    out.append(self._send(rule, rule.latest, now))
        return out

    # ---- What a page shows ----

    def state_of(self, rule: Rule, off: Iterable[str] = ()) -> str:
        """``disarmed``, ``study off``, ``warming up`` or ``armed``.

        Warming up: every value of an any-of rule is None while its study warms up, or one of an all-of rule's.
        """
        if not rule.armed:
            return "disarmed"
        off = set(off)
        if any(c.study in off for c in rule.conditions):
            return "study off"
        if rule.evaluated:
            warming = [v is None and c.value.warms for c, v in zip(rule.conditions, rule.values, strict=True)]
            if (all(warming) if rule.combine == "any" else any(warming)):
                return "warming up"
        return "armed"

    def status(self, off: Iterable[str] = ()) -> dict[str, Any]:
        """The rules as a page shows them: definitions and what each has seen, as copies of plain values.

        ``off``: the keys of the studies that are switched off. Every number is finite or None.
        """
        off = tuple(off)
        rules = []
        for rule in self.rules:
            rules.append({
                "id": rule.id, "name": rule.name, "armed": rule.armed, "one_shot": rule.one_shot,
                "combine": rule.combine, "cooldown": rule.cooldown, "actions": list(rule.actions),
                "text": rule.describe(), "state": self.state_of(rule, off), "count": rule.count,
                "fired_at": rule.last_fired,
                "conditions": [{
                    "value": f"{c.study}:{c.name}", "op": c.op, "threshold": c.threshold, "text": c.describe(),
                    "label": c.value.label, "unit": c.value.unit, "now": rule.values[i],
                    "met": rule.flags[i] if rule.evaluated else None,
                    "none": c.value.none if rule.values[i] is None else "", "off": c.study in off,
                } for i, c in enumerate(rule.conditions)],
            })
        return {"rules": rules, "marks": [[t, rule_id, name] for t, rule_id, name in self.fired],
                "limit": MAX_RULES}


# ---- Reading a form ----

def _many(form: Any, key: str) -> list[str]:
    """The values a form field has (a form with a repeated field, or a dict with a value or a list of them)."""
    if hasattr(form, "getlist"):
        return [str(v) for v in form.getlist(key)]
    value = form.get(key)
    return [] if value is None else [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


def _flag(text: Optional[str]) -> Optional[bool]:
    return {"on": True, "true": True, "1": True, "off": False, "false": False, "0": False}.get(
        "" if text is None else text.strip().lower())


def parse_rule(form: Any) -> dict[str, Any]:
    """The arguments of ``TriggerBoard.create`` or ``update`` from a form; ValueError says everything that is wrong.

    Fields: ``name`` (blank: made from the conditions), ``value`` ("study:name"), ``op`` and ``threshold``, repeated
    once for each condition, ``combine`` (all or any), ``cooldown``, ``action`` repeated, ``one_shot`` (on or off;
    left out: on when the rule pauses) and ``armed`` (on or off; left out: unchanged, or on for a new rule).
    """
    errors: list[str] = []
    values, ops, levels = _many(form, "value"), _many(form, "op"), _many(form, "threshold")
    if not (len(values) == len(ops) == len(levels)):
        raise ValueError("Each condition needs a value, an operator and a level.")
    conditions = []
    for text, op, level in zip(values, ops, levels, strict=True):
        study, _, name = text.partition(":")
        if (study, name) not in VALUES:
            errors.append(f"No value {text!r} to read.")
            continue
        try:
            threshold = float(level)
        except ValueError:
            errors.append(f"The level of {VALUES[(study, name)].label} must be a number.")
            continue
        conditions.append(Condition(study, name, op, threshold))
    try:
        cooldown = float(str(form.get("cooldown", DEFAULT_COOLDOWN)))
    except ValueError:
        errors.append("The cooldown must be a number of units of simulation time.")
        cooldown = DEFAULT_COOLDOWN
    combine = str(form.get("combine", "all"))
    actions = _many(form, "action")
    one_shot, armed = _flag(form.get("one_shot")), _flag(form.get("armed"))
    for field_name, flag in (("one_shot", one_shot), ("armed", armed)):
        if field_name in form and flag is None:
            errors.append(f"{field_name.replace('_', ' ').capitalize()}: choose on or off.")
    name = str(form.get("name", "")).strip()
    if not errors:
        try:
            validate(name, conditions, combine, cooldown, actions)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(" ".join(errors))
    return {"name": name, "conditions": tuple(conditions), "combine": combine, "cooldown": cooldown,
            "actions": tuple(dict.fromkeys(actions)), "one_shot": one_shot, "armed": armed}
