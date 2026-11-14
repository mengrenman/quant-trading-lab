"""Stock deals in the books of ``lob/statarb.py``: an acquirer offers shares for a target's, and the deal may break.

A deal is the settings of ``Deals`` and, once announced, a ``Deal``. While it is on, the target's fair value is a mix of
what the offer is worth (the ratio ``r`` of the acquirer's, ``r F_A``) and what the target is worth alone (``S_T``, its
level and factors and own part, which go on moving underneath)::

    F_T = pi * r * F_A + (1 - pi) * S_T,        pi = exp(-q * (time left))

``pi`` is the probability the market gives to the deal going through, from now, under a constant hazard ``q`` of
breaking (set by ``priced``). The deal in fact breaks at a time drawn from the true hazard ``p`` (set by ``breaks``), if
that is before its end: ``F_T`` falls back to ``S_T`` at once. At its end the target's level takes the difference
``r F_A - S_T``, so that it goes on from the deal's value, with its own loadings and own part, and its fair value does
not jump. When ``priced`` is ``breaks`` the target's fair value has no drift on average over deals, from just after
the announcement, so that there is nothing to earn from it on average; ``priced`` above ``breaks`` is a premium for
holding the risk of a break. That is on average over deals and with the own parts aside: given them, which revert, it
is not a martingale (a late entry earns a little from them), and with a trend it has that drift too.

``Deals`` is settings and nothing else, and is not changed once made, so one may be given to any number of markets. All
the state, the deals on and those that ended, when each book can next be in a deal and the stream the draws come from,
is in ``FactorValues`` (``lob/statarb.py``), which makes it afresh in ``reseed``. This module imports nothing from
there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Deals:
    """How deals come: announced at ``rate`` a unit of time over the whole market (never more than one at a grid step),
    each worth ``premium`` ticks more than the target at the announcement, lasting ``length`` units (to a whole number
    of the fair values' steps, at least one) unless it breaks, which it does with probability ``breaks``, while the
    market prices a break at ``priced``. A book is in at most one deal, and out of a new one for ``cool`` units after
    its last, counted from the deal's planned end (``length`` after the announcement) whether it completed there or
    broke before it: so which books are free does not depend on which deals break, and the same seed at two values of
    ``breaks`` (``priced`` held) has the same deals, only more of them break at the higher. With ``within`` the
    acquirer is drawn from the books whose loadings are parallel to the target's when there are any free (the same
    sector, in a table of sectors), else from all the free books.

    ``premium`` is in ticks whatever the price, and a target's fair value takes ``pi * r`` times its acquirer's moves,
    where ``r = (S_T + premium) / F_A``. At a level of thousands of ticks ``r`` is near 1. Near the floor, at prices of
    tens of ticks, it is far from 1 (0.5 to 9 at a level of 40 and a volatility of 3), so that a target moves far from
    its acquirer and deals mean little there."""
    rate: float = 0.002  # announcements a unit of time, over the whole market
    premium: float = 60.0  # ticks the offer is worth above the target at the announcement
    length: float = 500.0  # units of time from announcement to completion
    breaks: float = 0.2  # the probability a deal breaks before its end
    priced: float = 0.2  # the probability the market gives to that
    cool: float = 100.0  # units a book stays out of a new deal after its last one's planned end, broken or not
    within: bool = True  # the acquirer from the books whose loadings are parallel to the target's, if any

    def __post_init__(self) -> None:
        if not (math.isfinite(self.rate) and self.rate >= 0):
            raise ValueError("the rate of deals must be 0 or more, a unit of time")
        if not (math.isfinite(self.premium) and self.premium > 0):
            raise ValueError("the premium of a deal must be above 0, in ticks")
        if not (math.isfinite(self.length) and self.length > 0):
            raise ValueError("the length of a deal must be above 0, in units of time")
        if not (0 <= self.breaks < 1):
            raise ValueError("the probability a deal breaks must be from 0 to below 1")
        if not (0 <= self.priced < 1):
            raise ValueError("the probability the market gives to a break must be from 0 to below 1")
        if not (math.isfinite(self.cool) and self.cool >= 0):
            raise ValueError("the cooling time must be 0 or more, in units of time")

    def hazards(self, length: float) -> tuple[float, float]:
        """The true hazard of a break, ``p``, and the priced one, ``q``, a unit of time, for a deal of ``length`` that
        breaks with probability ``breaks`` and is priced at ``priced``: ``-ln(1 - breaks) / length``, and likewise."""
        return -math.log(1 - self.breaks) / length, -math.log(1 - self.priced) / length


@dataclass
class Deal:
    """A deal that is on, or was: ``acquirer`` offers ``ratio`` of its shares for each of ``target``'s.

    ``announced`` and ``ends`` are times, ``offered`` the premium in ticks the offer was worth above the target when it
    was announced. ``outcome`` is ``"completed"`` or ``"broken"`` once it is over, and ``ended_at`` the time it was
    (``ends`` for a completion). It also has, private and not for a trader, the time it will break if it will.
    """
    target: int
    acquirer: int
    ratio: float
    announced: float
    ends: float
    offered: float
    outcome: Optional[str] = None  # "completed" or "broken", once over
    ended_at: Optional[float] = None

    def __post_init__(self) -> None:
        self._breaks_at: Optional[float] = None  # when it will break, if it will: hidden, as a market's future is
