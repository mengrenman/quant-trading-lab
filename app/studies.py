"""The microstructure studies of one dashboard session (the library's, in lob/studies.py), and what the page needs.

``StudyBoard`` holds one of each study with its settings, which stay when the session starts over while the data
is cleared, and the studies' samples, one row for each unit of simulation time (the last 600), from which the page
draws its sparklines. ``LOBManager`` (app/manager.py) feeds it the book's events and the clock; ``snapshot`` gives
the page copies: the version, the settings, the latest values and the samples since the last it has, with ``None``
for a value that is not defined (never NaN or infinity, which JSON cannot hold). The board exists only in a
session's manager, never in the scratch managers of a fit, so those pay nothing for it.

The settings of each study are described as the other forms of the dashboard are, by ``Param`` and ``Choice``
(``STUDY_SPECS``), made from the library's own settings, so that the form and the check of its input have one source.
"""

from __future__ import annotations

import math
from collections import deque
from numbers import Integral, Real
from typing import Any, Optional

from app.model_specs import Param
from app.strategy_specs import Choice
from lob.studies import CVD, Setting, Study, StudySet, default_studies

SAMPLES = 600  # units of simulation time kept, and sent when a page connects or a version changes

OPTION_LABELS = {
    "true": "the true side", "tick": "tick rule", "quote": "quote rule (Lee and Ready)", "mid": "against the mid",
    "bvc": "bulk volume classification", "equal": "equal weight", "touch": "weighted to the touch",
    "model": "the model's", "everyone": "everyone's", "below": "below the line", "crossing": "crossing the line",
}


def _step(setting: Setting) -> float:
    """The step of a number input for a setting: whole numbers by 1, ranges up to 1 by 0.05, up to 100 by 0.5."""
    if setting.integer:
        return 1
    return 0.05 if setting.high <= 1 else 0.5 if setting.high <= 100 else 1.0


def spec_of(setting: Setting) -> Param | Choice:
    """A library setting as the form's spec: a choice from its options, or a number in its range."""
    if setting.options:
        return Choice(setting.name, setting.label, setting.default,
                      tuple((value, OPTION_LABELS.get(value, value)) for value in setting.options))
    return Param(setting.name, setting.label, setting.default, setting.low, setting.high, _step(setting),
                 integer=setting.integer)


STUDY_SPECS: dict[str, tuple[Param | Choice, ...]] = {
    study.key: tuple(spec_of(setting) for setting in study.SETTINGS) for study in default_studies()}
STUDY_LABELS: dict[str, str] = {study.key: study.label for study in default_studies()}


# Short names for the tiles, and what each study is, for the page (components/studies.html): what it measures, how it is
# defined in a sentence or two, and the caveats to read it by.
STUDY_SHORT: dict[str, str] = {
    "ofi": "OFI", "cvd": "CVD", "delta": "Delta", "vpin": "VPIN", "otr": "OTR and TTO", "imbalance": "Imbalance",
    "spread": "Spread", "resilience": "Resilience"}

STUDY_HELP: dict[str, dict[str, Any]] = {
    "ofi": {
        "what": "Net buying pressure at the best quotes over a trailing window.",
        "definition": "The sum, over the last W units of simulation time, of Cont, Kukanov and Stoikov's (2014) order "
                      "flow imbalance increments: lots added at the best bid or taken from the best ask count as "
                      "buying, the reverse as selling. In lots.",
        "notes": ("Positive means buying pressure. It reads warming up until a full window has passed.",),
    },
    "cvd": {
        "what": "Lots bought by takers less lots sold by takers since the session started.",
        "definition": "Each fill adds its size when the buyer was the aggressor and subtracts it when the seller was "
                      "(the true side). It counts from the last reset, or from the start of a playback, and goes on "
                      "counting through a seek.",
        "notes": ("It has no settings; its switch turns it off.",),
    },
    "delta": {
        "what": "Buy-taker lots less sell-taker lots over a trailing window.",
        "definition": "Over the last W units, with each fill's side taken from the true aggressor, the tick rule "
                      "(above the previous fill's price a buy, below a sell, at it the previous sign), or the quote "
                      "rule (Lee and Ready 1991: above the mid a buy, below a sell, at it the tick rule).",
        "notes": ("For the tick and quote rules the tile shows coverage, the share of lots classified, and accuracy, "
                  "the share of those classified as the true side says; only a simulator can show the accuracy.",
                  "Quote lag 0 uses the mid just before the event and is exact in a simulator, by construction: a "
                  "check, not a result. A lag of 1 or more uses the mid some units back, as a delayed feed would."),
    },
    "vpin": {
        "what": "The volume-synchronized probability of informed trading (Easley, Lopez de Prado and O'Hara 2012).",
        "definition": "Fills fill equal-volume buckets of V lots in order (a fill that crosses a bucket's end is "
                      "split); VPIN is the mean over the last N full buckets of |buys - sells| / V, from 0 to 1.",
        "notes": ("It reads warming up until N buckets are full. With lots a bucket 0, V is 50 times the median fill "
                  "of the first 200 fills.",
                  "VPIN levels differ by side rule: the true side, the tick rule and the mid rule are on different "
                  "scales. VisualHFT's website bands (0.60 and 0.75, described for its tick-rule version) are not "
                  "carried over: they are not in its code, whose default classifies against the mid.",
                  "Bulk volume classification (BVC) sits near 0.5 by construction (Andersen and Bondarenko 2014): "
                  "it reads each bucket's price change against its own volatility, whatever the flow, so its level "
                  "is not evidence of toxicity.",
                  "The mid rule is VisualHFT's default: a fill against the mid some units back. With quote lag 0 it "
                  "equals the true side in a simulator."),
    },
    "otr": {
        "what": "Messages per trade (OTR) and traded lots per lot ordered (TTO) over a trailing window.",
        "definition": "OTR: book events (arrivals and cancels) per trade, a trade being an arriving order that "
                      "filled, counted once however many fills it took. TTO: the lots traded over the lots of the "
                      "orders that arrived.",
        "notes": ("Both are market-wide and without the regulator's minus one (MiFID II RTS 9 counts "
                  "(orders / transactions) - 1 for each member over a session).",
                  "By default only the model's own orders count (scope model): a quoting strategy cancels and "
                  "replaces at every decision and would otherwise set the ratio. Everyone counts them all.",
                  "VisualHFT's OTR is a count of book changes, in which its code never counts a trade: it is "
                  "comparable to neither."),
    },
    "imbalance": {
        "what": "The visible depth on the bid side against the ask side, over the top levels.",
        "definition": "(B - A) / (B + A) over the best L price levels of each side, from -1 (all ask) to +1 (all "
                      "bid). Every level weighs alike, or levels weigh less the further they are from the touch.",
        "notes": ("Touch weighting follows VisualHFT's code: a level d ticks from its side's best quote weighs "
                  "(1 + d / S)^-2, with S the median quoted spread over the last 100 units.",
                  "With one level it is the queue imbalance at the touch. It is a dash while a side is empty."),
    },
    "spread": {
        "what": "The quoted spread in ticks, and how unusual it is.",
        "definition": "Best ask less best bid in ticks, and its z-score against the mean and standard deviation of "
                      "the once-a-unit samples of the last 100 units.",
        "notes": ("A spread shock is marked when z reaches the setting (VisualHFT: 2). The z-score needs 20 "
                  "samples, and is a dash while the spread has not varied.",),
    },
    "resilience": {
        "what": "How fast the depth near the touch comes back after it is taken.",
        "definition": "A depletion starts when a take or a cancel leaves the lots within K prices of a side's best "
                      "quote below q times their median over the last 100 units, and ends when they are back above "
                      "r times it (after Large 2007). The tile shows the side depleted now and for how long, the "
                      "median of the last 20 recovery times, the share of depletions abandoned (the quote moved "
                      "away first), the share of quote-moving takes that a limit order undid within the window and "
                      "its median time (Large's restoration), and the score T_hist / (T_hist + T_now), VisualHFT's "
                      "recovery sub-score, 1 when nothing is depleted.",
        "notes": ("A depletion is anchored at the K prices where it began: measuring from the current best quote "
                  "would count every improving quote as a loss.",
                  "In the Mike and Farmer (simplified) model the touch queues hold hundreds of lots, so a depletion "
                  "never occurs and the tile says no depletions.",
                  "In the zero-intelligence model a side is depleted most of the time at the defaults: its levels "
                  "refill about as slowly as their orders are canceled (a time of the order of 1/delta) while the "
                  "touch is taken far more often, so a depletion ends and the next take that leaves the depth low "
                  "starts another. The depleted now line is then the usual state, not an alarm; the median "
                  "recovery, the abandoned share and the score are what to read, and the crossing onset (in its "
                  "settings) counts fewer starts."),
    },
}


class SessionCVD(CVD):
    """Cumulative volume delta as the session keeps it: the manager adds each fill (``add``), as it is made.

    The other studies see events only while the board is on, and a seek or a collection switches it off; this one
    counts through both, and starts again only when the session does (the book is reset, or a playback begins).
    """
    sees_events = False
    needs_fills = False

    def add(self, lots: int) -> None:
        """A fill of ``lots``: positive when the buyer was the aggressor, negative when the seller was."""
        self._total += lots


def plain(value: Any) -> Any:
    """A number or flag as JSON can hold it: a float that is not finite, which JSON cannot, becomes ``None``."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else None
    raise TypeError(f"a study gave {value!r}, which is not a number")


class StudyBoard:
    """The studies of a session: their settings, their data, and the samples a page draws.

    ``prev`` is the quotes after the last event given to the studies, the next event's quotes before (the manager
    keeps it exact: see ``LOBManager._resync_studies``). While ``suspended`` the manager gives the studies nothing
    (a playback seeking, or the board failed); ``failed`` says why a study stopped the board.
    """

    def __init__(self):
        studies = [SessionCVD() if study.key == "cvd" else study for study in default_studies()]
        self.set = StudySet(studies, auto_sample=False)
        self.keys: list[str] = [study.key for study in self.set]
        self.cvd: SessionCVD = self.set["cvd"]
        self.observe = self.set.observe  # what the manager calls for each event
        self.samples: deque[list] = deque(maxlen=SAMPLES)  # [unit, mid, one main value a study, in keys' order]
        self.unit = 0  # the last unit sampled
        self.prev: Optional[tuple] = None
        self.suspended = False
        self.failed: Optional[str] = None
        self.version = 0  # the manager's studies_version, which says when a page's samples are out of date

    def __getitem__(self, key: str) -> Study:
        return self.set[key]

    def reset(self, index: int = 0, t: Optional[float] = None) -> None:
        """Start every study afresh in unit ``index`` (the clock is ``t``, default the unit's start); settings stay."""
        self.set.reset(index, t)
        self.samples.clear()
        self.unit = index

    def restart(self, index: int, t: Optional[float] = None) -> None:
        """Start afresh as ``reset`` does, keeping what is not learned from the events seen: the CVD, and VPIN's V.

        For when the board was off while the market went on (a collection, a seek): the CVD went on counting, and
        V, the lots of a VPIN bucket, was set before and is not changed by the events the board missed.
        """
        total = self.cvd.value()
        lots = self["vpin"].volume_in_use
        self.reset(index, t)
        self.cvd.add(total)
        if lots is not None:
            self["vpin"].set_volume(lots)

    def recalibrate(self) -> None:
        """The engine or the model changed: fills are sized differently now, so VPIN sets its bucket again."""
        self["vpin"].recalibrate()

    def set_volume(self, lots: float) -> None:
        """Use ``lots`` a VPIN bucket (as a playback does with the one its recording was made with)."""
        self["vpin"].set_volume(lots)

    @property
    def volume(self) -> Optional[float]:
        """The lots a VPIN bucket holds now, None until 200 fills have set it."""
        return self["vpin"].volume_in_use

    def fail(self, exc: Exception) -> None:
        """A study raised: the board stays as it was and takes no more events."""
        self.failed = repr(exc)
        self.suspended = True

    def resume(self) -> None:
        """Take events again, unless the board failed."""
        if self.failed is None:
            self.suspended = False

    def sample_to(self, index: int, book) -> None:
        """Sample every unit after the last up to ``index`` (at most the last 600), from ``book`` as it stands."""
        rows = self.set.sample_to(index, book, limit=SAMPLES, rows=True)
        self.unit = self.set.unit
        if rows:
            mid = plain(book.mid_price)
            for unit, row in rows:
                self.samples.append([unit, mid, *(plain(row.get(key)) for key in self.keys)])

    def parse(self, key: str, form: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """The settings a form gives for a study (those it names), and a message for each that is not valid."""
        values, errors = {}, {}
        for spec in STUDY_SPECS[key]:
            if spec.name in form:
                try:
                    values[spec.name] = spec.parse(form[spec.name])
                except ValueError as exc:
                    errors[spec.name] = str(exc)
        return values, errors

    def configure(self, key: str, values: dict[str, Any], enabled: Optional[bool] = None) -> bool:
        """Change a study's settings and switch it on or off; True if anything changed.

        The study starts afresh and its samples are cleared, since what it gathered under other settings means
        something else. Raises ValueError (nothing is changed) for settings that contradict each other.
        """
        study = self[key]
        changed = self.set.configure(key, **values)
        if enabled is not None and study.enabled != enabled:
            self.set.enable(key, enabled)
            changed = True
        if changed:
            column = 2 + self.keys.index(key)
            for row in self.samples:
                row[column] = None
        return changed

    def snapshot(self, since: Optional[int], t: float, book) -> dict[str, Any]:
        """Everything a page shows, as copies of plain values; ``None`` where a value is not defined.

        ``settings`` and ``enabled`` by study key; ``values`` the latest values of each study that is on, as of the
        simulation time ``t`` and the ``book`` now (empty while the board is off); ``keys`` name the columns of
        ``samples`` after the unit and the mid, ``unit`` is the last unit sampled, and ``samples`` are the rows with
        a unit after ``since`` (all of them if None).
        """
        values: dict[str, Any] = {}
        if not self.suspended:
            values = {key: {name: plain(v) for name, v in study.items()}
                      for key, study in self.set.values(t, book).items()}
        return {
            "version": self.version, "failed": self.failed, "suspended": self.suspended,
            "settings": {study.key: {name: plain(v) for name, v in study.settings.items()} for study in self.set},
            "enabled": {study.key: study.enabled for study in self.set},
            "values": values, "keys": list(self.keys), "unit": self.unit,
            "samples": [list(row) for row in self.samples if since is None or row[0] > since],
        }
