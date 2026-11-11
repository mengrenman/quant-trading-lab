"""Zero-intelligence order flow with informed traders, and a latent value they trade on.

None of the other models has a fair value: order flow does not predict the price, and resting orders earn. This
one adds the simplest informed trader to Smith et al.'s (2003) zero-intelligence model, so that toxicity has a
ground truth to measure a study against.

The latent value ``V_t`` starts at the initial mid and moves as a random walk with standard deviation ``sigma_v``
per unit of time, plus jumps at rate ``jump_rate`` of size ``jump_size`` (up or down, equally likely). It is
advanced over the model's own time steps exactly: a Gaussian increment of variance ``sigma_v**2 * dt`` and a
Poisson count of jumps with mean ``jump_rate * dt``.

A random walk knows no floor, but a price does. At the defaults V has a standard deviation of
``sqrt(sigma_v**2 + jump_rate * jump_size**2)`` = 0.54 ticks per square root of a unit of time, and the mid follows
it, so in a session that runs for hours at top speed V would reach price 0 (about 60% of runs in an hour at 2,000
events a second); and once the placement window of the zero-intelligence orders (``max(1, ask - offset)``) reaches
one tick, the bid side collapses to a single price that the informed sells keep hitting (the collapse that
``FairValues`` had before it was given a floor). So V is held at or above ``value_floor``, which is ``VALUE_FLOOR``
= 20 ticks (above ``price_range + 1`` at the default L of 15, so that every bid placed from an ask near V stays
above one tick), or ``price_range + 5`` when that is more, and never above the initial mid. It is cut short at
each advance, after the diffusion and the jumps and without drawing a random number, so a seeded run that never
reaches the floor is unchanged. Near the floor V is no longer a martingale.

Informed traders send market orders, of the size the model's market orders have, at a rate proportional to what
they would make: ``kappa * max(0, V_t - best ask) / tick`` for buys and ``kappa * max(0, best bid - V_t) / tick``
for sells, each at most ``cap`` per unit of time. This is Glosten and Milgrom's (1985) condition: they trade only
when the value lies beyond the quote they hit (with the mid in place of the quote, a buy with the mid below V and
the ask above it would pay more than the value and not be toxic). There are no informed limit orders (they would
make the book itself informative) and no splitting. The rate is added to the model's total event rate, so the next
event is drawn among all of them with the right probabilities.

The uninformed market orders have fair-coin sides by default, as in Smith et al. Setting ``persistence`` above 0.5
makes their sides a Markov chain instead: with probability ``persistence`` an uninformed market order has the side
of the previous uninformed one, else the opposite (0.5 is the fair coin, event for event and generator state; the
first order has a fair coin). Their lag-1 sign autocorrelation is then ``2 * persistence - 1``. This is the control
for the question "is the flow one-sided because it is toxic, or only one-sided?": the sides come in runs of about
``1 / (1 - persistence)`` orders, and a measure of one-sidedness (the volume imbalance of a window) reads them as
it reads informed flow; yet the side of an uninformed order says nothing about V (V does not depend on the flow), so
what the makers lose against V on these fills has no informed part, and the flow's impact on the mid is not backed
by V. Only market orders persist: limit orders keep their fair-coin sides, informed orders are not drawn from the
chain and leave its state as it was, and the chain steps once for every uninformed market order drawn, including
one that finds no liquidity to take and is not sent.

Because V moves between events, the informed rate is not constant over the wait for the next event. The model
simulates it exactly by thinning: waiting times come from the rate the model's events have plus ``cap`` (an upper
bound, as only one side can be profitable), V is advanced to each candidate time, and a candidate is accepted
as an informed order with probability rate / bound at the V of that time, else it is dropped and the wait goes on.
An informed order is therefore always profitable at the V it is sent at. The price is time: a candidate is
dropped when its draw lands in the part of the bound the informed rate at its V leaves unused, so a fraction
(cap - rate) / (base rate + cap) of the candidates is dropped: about 3% at the defaults on the dashboard's book
(the bound ``cap / (base rate + cap)``, what is dropped when V is inside the touch, is about 5% there: the base rate
is about 58, with some 750 resting orders). A large ``cap`` on a quiet book (low ``mu``, ``lam`` and ``delta``)
multiplies the work per event. A call gives up and returns None (no event) after ``MAX_DROPPED`` dropped
candidates in a row.

Ground truth, kept on the model:

* ``informed_ids``: the order ids of the informed orders (an array copy, for tests and notebooks; ``is_informed``
  marks a table of takers); ``informed_lots`` their lots; ``taker_lots`` the lots of every market order the model
  sent (informed or not). The realized informed share of taker volume is ``informed_lots / taker_lots``
  (``informed_share``). It is an outcome of ``kappa``, ``cap`` and ``sigma_v``, not a setting. The ids are kept in a
  compact array (8 bytes an informed order, 11.5 MB after 1.44 million of them, against 100 MB for a set), as a
  session that runs for hours sends about 35 informed orders a second at top speed. ``last_informed`` says whether
  the order the last call returned was informed, which is all a live observer needs.
* ``value()``: V at the model's clock, readable at any time (``value_time`` says which time that is).
* The maker's loss on a fill, against V at the time of the fill (taker buy: ``V_t - price``; taker sell:
  ``price - V_t``), is computed from the fills and the value at the fill's time: right after the order is returned,
  ``value()`` is V at the order's own time, so an observer that reads it there has what it needs. Afterwards,
  construct the model with ``record_path=True``: it then keeps V at each of its event times, ``maker_loss`` and
  ``is_informed`` turn a fill table (``Simulation.trades``: ``time``, ``price``, ``aggressor``,
  ``taker_order_id``) into the loss per fill and whether its taker was informed, and ``value_at`` reads the path.
  V is simulated at the model's event times only, so ``value_at`` is exact at those times (every fill of the model's
  own orders is at one) and the last known value in between.

Reproducibility: the run is a function of the seed. V's random numbers come from a generator of the model's own,
taken from the run's generator at the model's first event without using any of its numbers (a jump ahead in the
same bit generator, or ``value_seed``), and drawn in a fixed order: for each advance the Gaussian, then the
Poisson count, then one coin per jump (a bit generator that cannot jump ahead gives one integer instead, and then
the run is no longer Smith's). With ``kappa`` or ``cap`` 0 and ``persistence`` 0.5 the run is exactly
``SmithFarmerModel``'s, event for event, from the same seed, whatever V does. (``persistence`` takes no random number
the fair coin did not: a side is one draw either way, compared with 1/2 at 0.5, and otherwise with ``persistence``
after a buy or ``1 - persistence`` after a sell.) The same seed gives the same sequence of standard normals for V,
not the same path once the events differ (they do as soon as ``kappa`` or ``persistence`` changes: V is advanced
over the run's own event and candidate times, and each Gaussian is scaled by the square root of the time it spans),
so a comparison across ``kappa`` or ``persistence`` is like for like in model, seed and number of events, not in
the realized path of V.

What the ground truth says, and what it does not. The loss on an informed fill against V is positive by
construction: an informed order is sent only when V is beyond the quote it hits, at the V of the time it is sent,
so the first fill of every informed order has a loss of ``V - ask`` (or ``bid - V``), above 0. "Makers lose to the
informed, against V" is the model's definition of toxicity, not a finding about it; what has content is its size
(the distribution of the gap between V and the quote) and what the makers go on to earn, the realized markouts. A
study cannot lead a jump of V either: nothing in the book carries V until the first informed order after the jump,
so what can be asked is how long after it a study reacts, against how long the mid takes to move. And the informed
share is not a sufficient statistic for the makers' earnings: raising ``kappa`` raises the share but lowers the
loss per informed fill, as the book tracks V more closely (3.3, 2.6, 2.2 and 1.75 ticks a lot at ``kappa`` 0.5, 1,
2 and 4, with ``cap`` 3 and the share 11, 14, 17 and 16%), while raising ``sigma_v`` raises both (1.7, 2.2, 2.9 and
4.5 ticks a lot at ``sigma_v`` 0.15, 0.3, 0.5 and 0.8, with the share 13, 17, 20 and 23%). A sweep over the share
has to say which dial moves.

Defaults (the dashboard's zero-intelligence settings: mu 6, lam 1, delta 0.03, L 15, lot 5, mid 100): about 15% of
taker lots are informed (10% from the diffusion alone, the rest from jumps), and 10 to 25% over a wide range of
``kappa``, ``cap`` and ``sigma_v``. Against V, informed fills cost the makers about 2 ticks a lot; the
uninformed ones pay them the half-spread, about 1.1 ticks a lot, not 0: V - price = (V - mid) - (price - mid) and
only the first part averages 0 for takers whose side does not depend on V.
"""

from __future__ import annotations

import math
from array import array
from typing import Optional

import numpy as np

from lob.book import OrderBook
from lob.models.zero_intelligence import SmithFarmerModel
from lob.order import Order, Side, market_order


class InformedZIModel(SmithFarmerModel):
    """Smith et al. (2003) zero-intelligence model plus informed traders who know a latent value.

    Parameters (keyword only; the zero-intelligence ones are those of ``SmithFarmerModel``):
        sigma_v: Standard deviation of the value's diffusion, in ticks per square root of a unit of time.
        jump_rate: Rate of jumps in the value, per unit of time.
        jump_size: Size of a jump, in ticks (up or down with equal probability).
        kappa: Informed order rate per tick of profitable gap to the quote they hit, per unit of time. 0 turns
            the informed traders off.
        cap: Most an informed side's rate can be, per unit of time. 0 turns the informed traders off.
        persistence: Probability that an uninformed market order has the side of the previous uninformed market
            order (a Markov chain on sides; 0.5, the default, is a fair coin; between 0 and 1, below 0.5 the sides
            alternate). Informed orders do not move the chain.
        value_seed: Seeds the value's own generator. By default it is taken from the run's generator at the
            model's first event, so the seed of the run seeds it.
        record_path: Keep the value at each of the model's event times (two floats an event; for notebooks, not
            for a session that runs for hours), for ``value_at``, ``maker_loss`` and the value's tests.
    """

    #: Most candidate events one call to ``generate_event`` drops in a row before it returns None (no event).
    MAX_DROPPED = 1000
    #: The lowest V goes, in ticks, at the default placement window (``value_floor`` is higher for a wider one).
    VALUE_FLOOR = 20.0

    def __init__(self, *, sigma_v: float = 0.3, jump_rate: float = 0.05, jump_size: float = 2.0,
                 kappa: float = 2.0, cap: float = 3.0, persistence: float = 0.5,
                 value_seed: Optional[int] = None, record_path: bool = False, **kwargs):
        super().__init__(**kwargs)
        for name, value in (("sigma_v", sigma_v), ("jump_rate", jump_rate), ("jump_size", jump_size),
                            ("kappa", kappa), ("cap", cap)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number of at least 0, got {value}")
        if not math.isfinite(persistence) or not 0 <= persistence <= 1:
            raise ValueError(f"persistence must be a number from 0 to 1, got {persistence}")
        self.persistence = float(persistence)
        self._previous_side: Optional[Side] = None  # the side of the last uninformed market order drawn
        self.sigma_v = float(sigma_v)
        self.jump_rate = float(jump_rate)
        self.jump_size = float(jump_size)
        self.kappa = float(kappa)
        self.cap = float(cap)
        self._informed_ids = array("q")  # order ids, in the order they were sent
        self.last_informed = False  # whether the order the last generate_event call returned was informed
        self.informed_lots = 0
        self.taker_lots = 0
        self.jumps = 0  # jumps of the value so far
        self._value = float(self.initial_mid)
        self.value_floor = min(float(self.initial_mid), max(self.VALUE_FLOOR, self.price_range + 5.0))
        self._value_time: Optional[float] = None  # the clock V has been advanced to; set at the first event
        self._vrng: Optional[np.random.Generator] = None if value_seed is None else np.random.default_rng(value_seed)
        self._path: Optional[tuple[array, array]] = (array("d"), array("d")) if record_path else None

    # ---- The latent value ----

    def value(self) -> float:
        """The latent value V at the model's clock (``value_time``); the initial mid before the first event."""
        return self._value

    @property
    def value_time(self) -> Optional[float]:
        """The time ``value()`` is V at; None before the model's first event."""
        return self._value_time

    @property
    def informed_ids(self) -> np.ndarray:
        """The order ids of the informed orders sent so far, as an array (a copy: the model goes on appending)."""
        return np.frombuffer(self._informed_ids, dtype=np.int64).copy()

    @property
    def informed_share(self) -> Optional[float]:
        """The realized informed share of the model's market-order lots so far; None before it sent any."""
        return self.informed_lots / self.taker_lots if self.taker_lots else None

    def _advance_value(self, dt: float) -> None:
        """Move V on by ``dt``: a Gaussian increment of variance sigma_v^2 dt, then a Poisson count of jumps, then
        the floor (which draws no random number)."""
        if dt <= 0:
            return
        rng = self._vrng
        if rng is None:
            raise RuntimeError("the value has no generator yet: pass value_seed, or run an event first")
        value = self._value + self.sigma_v * math.sqrt(dt) * rng.standard_normal()
        if self.jump_rate > 0:
            n = int(rng.poisson(self.jump_rate * dt))
            if n:
                ups = int(rng.integers(0, 2, size=n).sum())
                value += self.jump_size * (2 * ups - n)
                self.jumps += n
        self._value = value if value >= self.value_floor else self.value_floor

    def _advance_value_to(self, t: float) -> None:
        if t > self._value_time:
            self._advance_value(t - self._value_time)
            self._value_time = t

    def _record(self) -> None:
        if self._path is not None:
            self._path[0].append(self._value_time)
            self._path[1].append(self._value)

    def value_at(self, times) -> np.ndarray:
        """V at ``times``: exact at the model's event times, the last recorded value before any other time.

        Needs ``record_path=True``. Before the first event it is the initial mid.
        """
        if self._path is None:
            raise RuntimeError("the value's path was not recorded: construct the model with record_path=True")
        path_t = np.frombuffer(self._path[0], dtype=float).copy()
        path_v = np.frombuffer(self._path[1], dtype=float).copy()
        times = np.asarray(times, dtype=float)
        index = np.searchsorted(path_t, times, side="right") - 1
        known = index >= 0
        value = np.full(times.shape, float(self.initial_mid))
        value[known] = path_v[index[known]]
        return value

    # ---- The informed traders' rate and the ground truth of their fills ----

    def informed_rates(self, book: OrderBook, value: Optional[float] = None) -> tuple[float, float]:
        """The informed buy and sell rates (per unit of time) for the book's quotes and V (by default the current).

        A buy: ``kappa * max(0, V - best ask) / tick``; a sell: ``kappa * max(0, best bid - V) / tick``; each at most
        ``cap``, and 0 for a side with no quote to hit. Both are 0 when V is inside the touch.
        """
        v = self._value if value is None else value
        ask, bid, tick = book.best_ask, book.best_bid, book.tick_size
        buy = 0.0 if ask is None else min(self.cap, self.kappa * max(0.0, v - ask) / tick)
        sell = 0.0 if bid is None else min(self.cap, self.kappa * max(0.0, bid - v) / tick)
        return buy, sell

    def maker_loss(self, times, prices, taker_buys) -> np.ndarray:
        """What the maker lost on each fill against V at its time: ``V - price`` for a taker buy, ``price - V``
        for a taker sell (negative when the maker gained). Needs ``record_path=True``.

        For ``Simulation.trades``: ``maker_loss(t["time"], t["price"], t["aggressor"] == "bid")``.
        """
        value = self.value_at(times)
        prices = np.asarray(prices, dtype=float)
        return np.where(np.asarray(taker_buys, dtype=bool), value - prices, prices - value)

    def is_informed(self, taker_order_ids) -> np.ndarray:
        """Whether each fill's taker order (``taker_order_id``) was an informed order."""
        # The view of the array is gone when np.isin returns: a live one would stop the model appending.
        return np.isin(np.asarray(taker_order_ids, dtype=np.int64),
                       np.frombuffer(self._informed_ids, dtype=np.int64))

    # ---- Events ----

    def _sample_market_side(self, rng: np.random.Generator) -> Side:
        """The side of an uninformed market order: a Markov chain that repeats the last side with probability
        ``persistence`` (a fair coin for the first, and always at 0.5). One draw either way."""
        previous = self._previous_side
        if previous is None:
            p_bid = 0.5
        else:
            p_bid = self.persistence if previous == Side.BID else 1.0 - self.persistence
        side = Side.BID if rng.random() < p_bid else Side.ASK
        self._previous_side = side
        return side

    def _generate_market_order(self, book: OrderBook, rng: np.random.Generator) -> Optional[Order]:
        order = super()._generate_market_order(book, rng)
        if order is not None:
            self.taker_lots += order.size
        return order

    def _generate_informed_order(self, buy: bool, rng: np.random.Generator) -> Order:
        order = market_order(size=self._sample_size(rng), timestamp=self._t, side=Side.BID if buy else Side.ASK)
        self._informed_ids.append(order.order_id)
        self.last_informed = True
        self.informed_lots += order.size
        self.taker_lots += order.size
        return order

    @staticmethod
    def _own_generator(rng: np.random.Generator) -> np.random.Generator:
        """A generator for V taken from the run's, without using a number of it (so kappa 0 stays Smith's run)."""
        try:
            return np.random.Generator(rng.bit_generator.jumped())
        except AttributeError:  # a bit generator without a jump ahead: take a seed from it instead
            return np.random.default_rng(int(rng.integers(0, 2**63)))

    def generate_event(self, book: OrderBook, t: float,
                       rng: np.random.Generator) -> Optional[Order]:
        """Generate the next event: ``SmithFarmerModel``'s, or an informed market order, by thinning."""
        self._observe(book, t)
        self.last_informed = False
        if self._vrng is None:
            self._vrng = self._own_generator(rng)
        if self._value_time is None:
            self._value_time = self._t  # the value starts where the model does, at the initial mid

        limit_rate = 2 * self.lam * self.price_range
        market_rate = self.mu
        base_rate = self._total_rate(book.num_orders())
        cap = self.cap if self.kappa > 0 else 0.0
        bound = base_rate + cap  # one side at most can be profitable, so the informed rate is at most cap
        if bound <= 0:
            return None

        for _ in range(self.MAX_DROPPED if cap > 0 else 1):
            self._t += rng.exponential(1.0 / bound)
            self._advance_value_to(self._t)
            u = rng.random() * bound
            if u < base_rate or cap <= 0:
                if u < limit_rate:
                    order = self._generate_limit_order(book, rng)
                elif u < limit_rate + market_rate:
                    order = self._generate_market_order(book, rng)
                else:
                    order = self._generate_cancellation(book, rng)
                self._record()
                return order
            buy, sell = self.informed_rates(book)
            if u < base_rate + buy:
                order = self._generate_informed_order(True, rng)
            elif u < base_rate + buy + sell:
                order = self._generate_informed_order(False, rng)
            else:
                continue  # dropped: the informed rate at this V is below the bound
            self._record()
            return order
        self._record()
        return None
