"""Traders of the stock deals in ``lob/statarb.py`` (see ``lob/deals.py`` for what a deal is): one that trades them, and
one that trades pairs and keeps out of their way.

While a deal is on its target's fair value drifts from where the announcement put it to the offer's worth, if it goes
through, and the market has priced a chance that it breaks. ``DealTrader`` enters a deal when the offer is worth more
than the target by ``entry`` ticks in the books' mids, ``r * mid_A - mid_T``: it buys ``size`` lots of the target and
sells ``rint(size * r)`` of the acquirer, and holds them until the deal is over, when it closes both at once. Priced
for a break that does not come it gains the gap closing, and when one does come it loses the jump (Mitchell and Pulvino
2001: the small gains of a merger arbitrageur and its large losses). Each deal it held is a record, with its profit
from the cash of its two books.

``NewsAwareTrader`` is a ``RelationTrader`` that reads the same news. A deal moves its target's fair value by tens of
ticks for a reason that has nothing to do with the relation a pair was chosen for, and a completed deal never comes
back. So it is flat in every relation that holds the target of a deal, and for a ``quarantine`` after the deal ends,
and trades the rest as a ``RelationTrader`` does. It leaves the acquirer's relations alone unless told
(``acquirers``): a deal moves the target's fair value and not the acquirer's in this simulator, so keeping out of the
acquirer gives up trades that the news does not touch. In a real market acquirers move too, and ``acquirers=True``
is the rule for that.

What they leave out: ``DealTrader`` does not size by the gap, the chance of a break or the costs, and does not leave a
deal early; it never takes the other side of a deal (short the target). A deal that is over before the trader has
decided while it was on is never entered, and is no record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lob.deals import Deal
from lob.statarb import BookMarket, RelationTrader, StateTrader


@dataclass
class DealRecord:
    """A deal a ``DealTrader`` held and has closed, in both books: how it came out and what it made.

    ``announced`` and ``outcome`` (``"completed"`` or ``"broken"``) are the deal's, and ``ended`` the time it was over
    (``Deal.ended_at``). ``entered`` is the time of the decision that entered it, and ``flat`` that of the first
    decision at which both its books were flat again. ``lots`` is the most it held of the target, which is ``size``
    lots if it gets to them (a rate or a band may hold it short of that). ``pnl`` is the change in its cash from the
    target and the acquirer, ticks times lots, from the decision it entered at to the one it was flat at: the cost of
    crossing the spread is in it.
    """
    target: int
    acquirer: int
    announced: float
    entered: float
    ended: float
    outcome: str
    lots: int
    pnl: float
    flat: float


@dataclass
class _Held:
    """A deal the trader entered and has not settled: where its cash was, what it has held at most, and whether it
    cannot be settled because a book was not flat when it entered (or another deal's entered it)."""
    deal: Deal
    entered: float
    cash: float
    lots: int = 0
    tainted: bool = False


def _key(deal: Deal) -> tuple[int, int, float]:
    """What identifies a deal: its target and acquirer, and the time it was announced (a book is in one at a time)."""
    return deal.target, deal.acquirer, deal.announced


@dataclass
class DealTrader(StateTrader):
    """Trades each stock deal that is on: long ``size`` lots of the target and short ``rint(size * r)`` of the acquirer.

    It reads ``fair.deals`` (those on) at each decision. ``weights`` has a row for each deal on, ``+1`` at the target
    and ``-r`` at the acquirer, not rescaled, and ``states`` a state for each (1 in the deal, 0 out of it) in the same
    order: both are made again whenever the deals on change, and a deal that goes on keeps its state. It enters a deal
    whose gap in mids, ``r * mid_A - mid_T``, is at least ``entry`` ticks, and stays in until the deal is over. With a
    ``limit`` above 0 it holds at most that many at once, taking those with the widest gap first, the ones it holds
    counting; 0 is any number. ``rate`` and ``band`` act as for any trader on the lots it enters with, but the books of
    a deal that is over are closed all the way at that decision, whatever they are, and again at each decision until
    they are flat.

    Each deal it entered is a ``DealRecord`` in ``outcomes`` once it is over and both books are flat, with its profit
    from the change in ``book_cash`` of the two. That is the deal's alone if both books were flat when it entered. A
    book is in one deal at a time and out of the next for ``cool`` (``lob.deals.Deals``), so they are, unless closing
    takes longer than that (a ``cool`` of less than a decision, or an order that did not fill in full): then the deal,
    and any deal still being closed in those books, is counted in ``not_flat_at_entry`` and has no record. It stays at
    0 with a ``cool`` of more than a few decisions.

    ``signal()`` is the gap in ticks of each deal on, in the order of ``weights``, and ``levels`` is ``(entry, 0.0)``.
    It has no mids to learn from (``learn`` takes none and returns 0, ``warming`` is None) and starts at its first
    decision (``warm_up``). ``check`` refuses a market with no deals.
    """
    entry: float = 8.0  # ticks of gap, r * mid_A - mid_T, at which it enters a deal
    limit: int = 0  # deals held at once at most; 0 for any number
    outcomes: list = field(default_factory=list)  # a DealRecord for each deal it held

    def __post_init__(self):
        if not (math.isfinite(self.size) and self.size > 0):
            raise ValueError("a size above 0")
        if not (math.isfinite(self.entry) and self.entry >= 0):
            raise ValueError("an entry of 0 or more ticks")
        if isinstance(self.limit, bool) or not isinstance(self.limit, (int, np.integer)) or self.limit < 0:
            raise ValueError("a limit of 0 (any number of deals) or more")
        if self.weights is not None or self.states is not None:
            raise ValueError("a deal trader makes its own relations, a row a deal: give it no weights or states")
        self._check_costs()
        self.not_flat_at_entry = 0  # deals with no record because a book of theirs was not flat when they were entered

    def check(self, market: BookMarket) -> None:
        if market.fair.deal_settings is None:
            raise ValueError("this market has no deals")

    def attach(self, n: int) -> None:
        super().attach(n)
        self.weights = np.zeros((0, n))
        self.states = np.zeros(0, dtype=int)
        self._keys: list[tuple[int, int, float]] = []  # the deal of each row
        self._on: list[Deal] = []  # the deals on as of the last ``observe``
        self._held: list[_Held] = []  # the deals it entered that are not settled
        self._now = 0.0
        self._gaps = np.zeros(0)

    def observe(self, market: BookMarket) -> None:
        """Take the deals on, and settle each that is over with both its books flat."""
        self._on = list(market.fair.deals)
        self._now = market.time
        still = []
        for held in self._held:
            target, acquirer = held.deal.target, held.deal.acquirer
            held.lots = max(held.lots, abs(int(self.position[target])))
            if held.deal.outcome is None or self.position[target] or self.position[acquirer]:
                still.append(held)
            elif not held.tainted:
                cash = float(self.book_cash[target] + self.book_cash[acquirer]) - held.cash
                self.outcomes.append(DealRecord(target, acquirer, held.deal.announced, held.entered,
                                                held.deal.ended_at, held.deal.outcome, held.lots, cash, self._now))
        self._held = still

    def decide(self, mids: np.ndarray) -> list[np.ndarray]:
        on = self._on
        keys = [_key(deal) for deal in on]
        kept = dict(zip(self._keys, self.states.tolist(), strict=True))
        states = np.array([kept.get(key, 0) for key in keys], dtype=int)  # the old states, in the new order
        weights = np.zeros((len(on), len(self.position)))
        gaps = np.zeros(len(on))
        for i, deal in enumerate(on):
            weights[i, deal.target], weights[i, deal.acquirer] = 1.0, -deal.ratio
            gaps[i] = deal.ratio * mids[deal.acquirer] - mids[deal.target]
        new = states.copy()
        room = len(on) if self.limit == 0 else self.limit - int(np.count_nonzero(states))
        ready = sorted((i for i in range(len(on)) if not states[i] and gaps[i] >= self.entry), key=lambda i: -gaps[i])
        new[ready[:max(room, 0)]] = 1
        for held in self._held:  # a deal that is over is closed all the way: its aim, which decays at the rate, is 0
            if held.deal.outcome is not None:
                self._aim[[held.deal.target, held.deal.acquirer]] = 0.0
        self.weights, self.states, self._keys, self._gaps = weights, states, keys, gaps
        for i in np.nonzero(new)[0]:
            if not states[i]:
                self._enter(on[i])
        return self._become(new)

    def _enter(self, deal: Deal) -> None:
        """Note a deal entered at this decision, before its orders are sent: where its two books' cash stands."""
        books = {deal.target, deal.acquirer}
        flat = not (self.position[deal.target] or self.position[deal.acquirer])
        if not flat:
            self.not_flat_at_entry += 1
            for held in self._held:  # a deal still being closed in these books is mixed with this one
                if not held.tainted and books & {held.deal.target, held.deal.acquirer}:
                    held.tainted = True
                    self.not_flat_at_entry += 1
        self._held.append(_Held(deal, self._now, float(self.book_cash[deal.target] + self.book_cash[deal.acquirer]),
                                tainted=not flat))

    def signal(self) -> np.ndarray:
        return self._gaps

    @property
    def levels(self) -> tuple[float, float]:
        return self.entry, 0.0

    @property
    def warming(self) -> None:
        return None

    @property
    def warm_up(self) -> float:
        """The time it needs before its first order: one decision, and the wait for the next event."""
        return self.taking(1)

    def learn(self, mids: list[np.ndarray]) -> int:
        return 0


@dataclass
class NewsAwareTrader(RelationTrader):
    """A ``RelationTrader`` that is flat in every relation holding a book that is the target of a deal, or was one that
    ended less than ``quarantine`` ago, and stays flat in it until then; the others it trades as it would.

    It reads ``fair.deals`` and ``fair.ended`` at each decision. A relation that is quarantined while it holds lots
    closes at that decision, as one whose state goes to 0 does (at its ``rate``, within its ``band``), and enters
    again, if its z-score says so, at the first decision after the quarantine ends. ``quarantine`` is in units of
    time, and ``taking(window)`` unless given: the time its window of decisions takes (each waits its interval and
    then for the next event, so a little more than ``window`` times ``interval``), so that when it ends the window holds
    no decision from before the deal's end, but for the odd one that waited longer than the mean. In a market with no
    deals it does what a ``RelationTrader`` with the same settings does, number for number.

    Only the relations holding a target are quiet by default, because only the target's fair value moves with a deal
    here (the drift to the offer's worth, and the fall back if it breaks: ``lob.deals``), and a pair that holds the
    acquirer alone is not disturbed by it. With ``acquirers=True`` a relation holding the acquirer of a deal is as quiet
    as one holding its target: the rule for a market where an acquirer's price moves with the deal too, as it does in a
    real one.
    """
    quarantine: Optional[float] = None
    acquirers: bool = False  # True: the acquirers of deals are quiet as well as their targets

    def __post_init__(self):
        super().__post_init__()
        if self.quarantine is None:
            self.quarantine = self.taking(self.window)
        elif not (math.isfinite(self.quarantine) and self.quarantine >= 0):
            raise ValueError("a quarantine of 0 or more units of time")
        self._quiet: list[int] = []  # the books a relation may not hold, of a deal on or ended < ``quarantine`` ago

    def _books(self, deal: Deal) -> tuple[int, ...]:
        """The books of a deal that a relation must not hold: its target, and its acquirer if ``acquirers``."""
        return (deal.target, deal.acquirer) if self.acquirers else (deal.target,)

    def observe(self, market: BookMarket) -> None:
        books = {book for deal in market.fair.deals for book in self._books(deal)}
        for deal in reversed(market.fair.ended):  # in the order they ended
            if market.time - deal.ended_at >= self.quarantine:
                break
            books.update(self._books(deal))
        self._quiet = sorted(books)

    def _become(self, states: np.ndarray) -> list[np.ndarray]:
        if self._quiet and self.weights is not None and len(self.weights):
            quiet = (self.weights[:, self._quiet] != 0).any(axis=1)
            if quiet.any():
                states = np.where(quiet, 0, states)
        return super()._become(states)
