"""Zero-intelligence order flow with self-exciting (Hawkes) market orders.

Real order flow clusters in time: periods of high arrival rates cluster
together. Gould et al. (2013, Section 5.2.4) describe how Zhao (2010) and
Toke (2011) replaced the Poisson processes of the Cont et al. (2010)
zero-intelligence model with Hawkes processes, in which each event raises
the rate of further events for a while. This model is simpler: only market
orders self-excite (Toke used mutually exciting processes for every event
type), on top of the Smith et al. (2003) ZI model.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from lob.book import OrderBook
from lob.models.zero_intelligence import SmithFarmerModel
from lob.order import Order


class HawkesZIModel(SmithFarmerModel):
    """ZI model whose market orders arrive as a self-exciting (Hawkes) process.

    The market order intensity is ``mu * (1 - branching) + E(t)``. Each
    market order raises the excitation E by ``branching * decay``, and E
    decays at rate ``decay`` in between, so each market order triggers
    ``branching`` further market orders on average (it must be below 1).
    The baseline is scaled so the long-run arrival rate of market orders
    stays ``mu``: compared with the ZI model, only their clustering changes.
    Every arrival excites, including one that finds the opposite side empty
    and so does not trade (as in the ZI model, it is then dropped). Limit
    orders and cancellations are as in the ZI model.

    Simulated exactly by thinning: the intensity only decays between events,
    so its current value bounds it until the next event.

    Parameters:
        branching: Expected number of market orders each one triggers (0 <= n < 1).
        decay: Decay rate of the excitation (per unit time).
        Remaining keyword arguments are passed to ``SmithFarmerModel``.
    """

    def __init__(self, branching: float = 0.8, decay: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= branching < 1:
            raise ValueError(f"branching must be in [0, 1), got {branching}")
        if decay <= 0:
            raise ValueError(f"decay must be positive, got {decay}")
        self.branching = branching
        self.decay = decay
        self._excitation = 0.0
        self.market_arrivals = 0

    def _market_intensity(self) -> float:
        return self.mu * (1 - self.branching) + self._excitation

    def generate_event(self, book: OrderBook, t: float,
                       rng: np.random.Generator) -> Optional[Order]:
        self._observe(book, t)

        limit_rate = 2 * self.lam * self.price_range
        cancel_rate = self.delta * book.num_orders()
        bound = limit_rate + self._market_intensity() + cancel_rate
        if bound <= 0:
            return None

        dt = rng.exponential(1.0 / bound)
        self._t += dt
        self._excitation *= math.exp(-self.decay * dt)
        market_rate = self._market_intensity()

        u = rng.random() * bound
        if u < limit_rate:
            return self._generate_limit_order(book, rng)
        if u < limit_rate + market_rate:
            self.market_arrivals += 1
            self._excitation += self.branching * self.decay
            return self._generate_market_order(book, rng)
        if u < limit_rate + market_rate + cancel_rate:
            return self._generate_cancellation(book, rng)
        return None  # thinned: the part of the bound the decayed intensity no longer covers
