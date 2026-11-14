"""Stock deals: an acquirer offers shares for a target's, which may break (lob/deals.py, run by lob/statarb.py)."""

import dataclasses
import functools
import math
from unittest import mock

import numpy as np
import pytest

from lob.deals import Deal, Deals
from lob.statarb import BookMarket, BookTrader, FactorValues, sector_loadings

SECTORS = sector_loadings(3, 4)  # twelve books in three sectors of four: a market factor and a factor a sector
TWO = sector_loadings(1, 2)  # two books, one sector
FOUR = sector_loadings(2, 2)  # four books, two sectors
MIXED = np.array([[1.0, 0.2], [0.8, 0.9], [1.2, -0.5], [0.6, 0.4], [1.1, 1.0], [0.9, -0.2]])  # no two rows parallel
BUSY = dict(rate=0.4, premium=60.0, length=6.0, cool=0.0)  # a deal about every two steps of 1, each over in six
SEEDS = range(1, 9)


def make(table=SECTORS, seed=1, deals=True, **kwargs):
    """Fair values at ``seed``, with deals (``deals``: settings of ``Deals`` over ``BUSY``) or none, and the rest."""
    if deals is not None and deals is not False:
        kwargs["deals"] = Deals(**(BUSY | (deals if isinstance(deals, dict) else {})))
    fair = FactorValues(table, **kwargs)
    fair.reseed(seed)
    return fair


def length_in_steps(fair):
    return max(1, round(fair.deal_settings.length / fair.step))


def pi_of(fair, deal, t):
    """The probability the market gives to a deal at time ``t``, by hand: ``exp(-q * (ends - t))``, with ``q`` the
    priced hazard of a deal that lasts what this one lasts."""
    settings = fair.deal_settings
    q = -math.log(1 - settings.priced) / (length_in_steps(fair) * fair.step)
    return math.exp(-q * (deal.ends - t))


def advance(fair, steps):
    """Steps of the grid, one at a time, yielding the time of each after it is made."""
    for i in range(1, steps + 1):
        fair.advance_to(i * fair.step)
        yield i * fair.step


def follow(fair, steps):
    """Run ``fair`` for ``steps`` grid steps, emptying ``gapped``. For each deal announced, in order, a dict of the
    deal, the fair values of its target and acquirer just after the announcement, and just after it ended (if it did).
    """
    records, seen = {}, 0
    for _ in advance(fair, steps):
        for deal in fair.deals:
            if id(deal) not in records:
                records[id(deal)] = {"deal": deal, "target": fair.fairs[deal.target],
                                     "acquirer": fair.fairs[deal.acquirer]}
        while seen < len(fair.ended):
            deal = fair.ended[seen]
            records[id(deal)] |= {"target_end": fair.fairs[deal.target], "acquirer_end": fair.fairs[deal.acquirer]}
            seen += 1
        fair.gapped.clear()
    return list(records.values())


def settled(records, fair, steps):
    """The records of deals whose outcome was settled by the end of a run of ``steps``: the ones that began at least a
    whole length before it, so that a deal that breaks is not more often over than one that goes through."""
    horizon = steps * fair.step - length_in_steps(fair) * fair.step
    return [r for r in records if r["deal"].announced <= horizon]


# ---- The settings ----

def test_deals_are_settings_that_do_not_change_and_a_deal_has_the_fields_a_trader_reads_and_a_time_it_does_not():
    settings = Deals()
    assert (settings.rate, settings.premium, settings.length, settings.breaks, settings.priced, settings.cool,
            settings.within) == (0.002, 60.0, 500.0, 0.2, 0.2, 100.0, True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.rate = 1.0
    deal = Deal(3, 5, 1.02, 10.0, 510.0, 60.0)
    assert [f.name for f in dataclasses.fields(Deal)] == ["target", "acquirer", "ratio", "announced", "ends", "offered",
                                                          "outcome", "ended_at"]
    assert deal.outcome is None and deal.ended_at is None and deal._breaks_at is None
    assert "breaks_at" not in repr(deal) and "_breaks_at" not in dataclasses.asdict(deal)


def test_the_deals_module_imports_nothing_from_the_module_that_imports_it():
    import ast
    import inspect

    import lob.deals

    tree = ast.parse(inspect.getsource(lob.deals))
    named = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    named += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert named and not [name for name in named if name.startswith("lob")]


@pytest.mark.parametrize("bad", [
    {"rate": -0.1}, {"rate": float("nan")}, {"rate": float("inf")}, {"premium": 0.0}, {"premium": -5.0},
    {"premium": float("inf")}, {"length": 0.0}, {"length": -1.0}, {"length": float("inf")}, {"breaks": -0.1},
    {"breaks": 1.0}, {"breaks": 1.5}, {"priced": -0.1}, {"priced": 1.0}, {"cool": -1.0}, {"cool": float("inf")},
    {"breaks": float("nan")},
])
def test_a_setting_of_deals_that_cannot_work_is_refused(bad):
    with pytest.raises(ValueError):
        Deals(**bad)


def test_a_setting_of_deals_that_can_work_is_taken_at_its_edges():
    Deals(rate=0.0, breaks=0.0, priced=0.0, cool=0.0)
    Deals(breaks=0.999, priced=0.999, premium=1e-6, length=1e-6)


def test_fair_values_refuse_deals_announced_at_a_rate_of_one_or_more_a_step_and_anything_that_is_not_deals():
    with pytest.raises(ValueError, match="once a step"):
        FactorValues(SECTORS, deals=Deals(rate=5.0), step=0.2)  # exactly 1 a step
    with pytest.raises(ValueError, match="once a step"):
        FactorValues(SECTORS, deals=Deals(rate=6.0), step=0.2)
    FactorValues(SECTORS, deals=Deals(rate=4.9), step=0.2)
    for not_deals in (object(), {"rate": 0.1}, True, 0.1):
        with pytest.raises(ValueError, match="Deals"):
            FactorValues(SECTORS, deals=not_deals)
    assert FactorValues(SECTORS, deals=None).deal_settings is None


def test_the_hazards_are_those_of_the_probabilities_over_the_length():
    settings = Deals(breaks=0.2, priced=0.4)
    p, q = settings.hazards(500.0)
    assert p == pytest.approx(-math.log(0.8) / 500.0) and q == pytest.approx(-math.log(0.6) / 500.0)
    assert math.exp(-p * 500.0) == pytest.approx(0.8) and math.exp(-q * 500.0) == pytest.approx(0.6)
    assert Deals(breaks=0.0, priced=0.0).hazards(10.0) == (0.0, 0.0)


@pytest.mark.parametrize("length, step, steps", [(500.0, 0.2, 2500), (1.0, 0.3, 3), (0.01, 0.2, 1), (0.45, 0.2, 2),
                                                 (0.75, 0.2, 4)])
def test_the_length_is_rounded_to_whole_steps_of_at_least_one_and_the_end_is_on_the_grid(length, step, steps):
    fair = make(deals={"length": length, "rate": 1.0}, step=step)
    records = follow(fair, 4000)
    assert records
    for record in records:
        deal = record["deal"]
        assert deal.ends - deal.announced == pytest.approx(steps * step, abs=1e-9)
        assert round(deal.announced / step) * step == pytest.approx(deal.announced, abs=1e-9)
        assert round(deal.ends / step) * step == pytest.approx(deal.ends, abs=1e-9)
        if deal.outcome == "completed":
            assert deal.ended_at == deal.ends  # at the step it ends on, exactly


def test_a_hazard_is_taken_over_the_length_as_rounded_so_that_the_share_that_breaks_is_what_was_given():
    fair = make(deals={"length": 1.0, "breaks": 0.5, "priced": 0.2, "rate": 2.0}, step=0.3)  # 3 steps, 0.9 units
    assert fair._deal_steps == 3
    assert fair._deal_p == pytest.approx(-math.log(0.5) / 0.9) and fair._deal_q == pytest.approx(-math.log(0.8) / 0.9)


@pytest.mark.parametrize("cool, step, steps", [
    (0.0, 0.2, 0), (100.0, 0.2, 500), (0.6, 0.2, 3), (0.7, 0.1, 7), (1.1, 0.1, 11),  # whole steps, up to round-off
    (5.0, 1.0, 5), (5.0 + 5e-10, 1.0, 5), (5.0 + 1.5e-9, 1.0, 6), (5.0 + 1e-6, 1.0, 6), (0.05, 0.2, 1), (0.25, 0.2, 2),
])
def test_a_cooling_time_is_taken_up_to_the_next_whole_step_but_for_the_round_off_of_a_whole_number_of_them(
        cool, step, steps):
    fair = make(deals={"cool": cool}, step=step)
    assert fair._cool_steps == steps


# ---- The announcements ----

def stream_by_hand(seed, rate, step, steps):
    """The announcements of ``[seed, 5]`` by hand: for each step one uniform, and if it is below ``rate * step`` the
    three more, a uniform for the target, one for the acquirer and an exponential. [(step, picks), ...]."""
    rng = np.random.default_rng([seed, 5])
    found = []
    for i in range(1, steps + 1):
        if rng.random() < rate * step:
            found.append((i, (rng.random(), rng.random(), rng.standard_exponential())))
    return found, rng


@pytest.mark.parametrize("seed", SEEDS)
def test_deals_are_announced_at_the_rate_given_on_the_grid_from_a_stream_of_their_own(seed):
    rate, steps = 0.05, 6000
    fair = make(deals={"rate": rate, "length": 2.0, "cool": 0.0, "breaks": 0.3}, seed=seed)  # never six on at once
    by_hand, rng = stream_by_hand(seed, rate, fair.step, steps)
    records = follow(fair, steps)
    assert [round(r["deal"].announced / fair.step) for r in records] == [i for i, _ in by_hand]
    assert fair._drng.bit_generator.state == rng.bit_generator.state  # and not one draw more or fewer
    first, (pick_target, pick_acquirer, wait) = by_hand[0]
    deal = records[0]["deal"]
    assert deal.target == int(pick_target * 12)  # every book is free at the first
    sector = [b for b in range(12) if b // 4 == deal.target // 4 and b != deal.target]
    assert deal.acquirer == sector[int(pick_acquirer * len(sector))]
    p = -math.log(1 - 0.3) / 2.0
    due = deal.announced + wait / p
    assert deal._breaks_at == (due if due <= deal.ends else None)


def test_the_announcements_of_all_eight_seeds_come_at_the_rate_given_within_their_standard_error():
    rate, steps, found = 0.05, 6000, 0
    for seed in SEEDS:
        fair = make(deals={"rate": rate, "length": 2.0, "cool": 0.0}, seed=seed)
        found += len(follow(fair, steps))
    expected = rate * steps * 0.2 * len(SEEDS)  # 480
    assert abs(found - expected) < 4 * math.sqrt(expected)


class Script:
    """A stream of the draws a test says: ``random`` and ``standard_exponential`` give the next of them in turn (and
    there is no next one after the last, which fails the test that asked for it)."""

    def __init__(self, *draws):
        self.draws = iter(draws)

    def random(self):
        return next(self.draws)

    def standard_exponential(self):
        return next(self.draws)


def test_a_uniform_equal_to_the_rate_times_the_step_is_no_announcement_and_costs_no_further_draw():
    fair = make(deals={"rate": 1.5})
    fair._drng = Script(1.5 * fair.step)  # the one draw of the step, on the edge of [0, rate * step)
    fair.advance_to(fair.step)
    assert fair.deals == [] and fair.gapped == []
    fair._drng = Script(np.nextafter(1.5 * fair.step, 0.0), 0.0, 0.0, 1.0)  # one hair below it: three more draws
    fair.advance_to(2 * fair.step)
    assert len(fair.deals) == 1


def test_a_deal_whose_time_to_break_is_exactly_its_end_is_a_deal_that_breaks():
    fair = make(deals={"breaks": 0.5, "length": 5.0})
    t, p = fair.step, fair._deal_p
    ends = (1 + fair._deal_steps) * fair.step  # announced at the first step
    wait = (ends - t) * p
    for _ in range(100):  # the draw of the exponential that makes it so, to the bit
        if t + wait / p == ends:
            break
        wait = float(np.nextafter(wait, np.inf if t + wait / p < ends else -np.inf))
    assert t + wait / p == ends
    fair._drng = Script(0.0, 0.0, 0.0, wait)
    fair.advance_to(t)
    deal = fair.deals[0]
    assert deal.ends == ends and deal._breaks_at == ends
    fair._drng = Script(*[1.0] * (fair._deal_steps + 5))  # no more announcements
    for _t in advance(fair, 1 + fair._deal_steps):  # the first step is done: the rest of them, to its end
        pass
    assert (deal.outcome, deal.ended_at) == ("broken", ends)  # a break takes it over a completion at the same step


def test_the_stream_is_read_a_step_and_three_draws_an_announcement_when_no_two_books_are_free_too():
    for seed in (1, 2, 3):
        fair = make(TWO, seed=seed, deals={"rate": 1.0, "length": 10.0, "cool": 0.0}, step=0.5)  # one deal and no more
        by_hand, rng = stream_by_hand(seed, 1.0, 0.5, 400)
        for _ in advance(fair, 400):
            pass
        assert len(by_hand) > 5 * (len(fair.deals) + len(fair.ended))  # most announcements found no two books free
        assert fair._drng.bit_generator.state == rng.bit_generator.state


@pytest.mark.parametrize("seed", SEEDS)
def test_the_first_offer_is_worth_the_premium_over_the_target_and_the_target_jumps_by_its_share_of_it(seed):
    fair, twin = make(seed=seed), make(seed=seed, deals=None)
    while not fair.deals:
        t = (fair._steps + 1) * fair.step
        fair.advance_to(t)
        twin.advance_to(t)
    (deal,) = fair.deals
    own_value, acquirer_value = twin.fairs[deal.target], twin.fairs[deal.acquirer]
    assert deal.offered == 60.0
    assert deal.ratio * acquirer_value - own_value == pytest.approx(60.0, abs=1e-9)
    pi_0 = 1 - 0.2
    assert fair.gapped == [(deal.target, pytest.approx(pi_0 * 60.0))]
    assert fair.fairs[deal.target] - own_value == pytest.approx(pi_0 * 60.0, abs=1e-9)
    assert deal.ratio * fair.fairs[deal.acquirer] - fair.fairs[deal.target] == pytest.approx(0.2 * 60.0, abs=1e-9)
    assert [fair.fairs[b] for b in range(12) if b != deal.target] == [twin.fairs[b] for b in range(12)
                                                                      if b != deal.target]


@pytest.mark.parametrize("priced", [0.0, 0.2, 0.4, 0.75])
def test_the_gap_opens_at_one_less_the_probability_times_the_premium(priced):
    fair = make(deals={"priced": priced, "premium": 60.0, "length": 500.0, "rate": 0.2}, volatility=0.0,
                own_volatility=0.0)
    while not fair.deals:
        fair.advance_to((fair._steps + 1) * fair.step)
    (deal,) = fair.deals
    assert fair.gapped == [(deal.target, pytest.approx((1 - priced) * 60.0))]
    assert deal.ratio * fair.fairs[deal.acquirer] - fair.fairs[deal.target] == pytest.approx(priced * 60.0, abs=1e-9)
    if priced == 0.4:
        assert fair.gapped[0][1] == pytest.approx(36.0) and priced * 60.0 == 24.0  # the design's example


@pytest.mark.parametrize("seed", SEEDS)
def test_the_first_deals_target_is_the_mix_of_the_offer_and_its_value_alone_while_it_is_on(seed):
    fair, twin = make(seed=seed, deals={"rate": 0.05, "breaks": 0.5, "priced": 0.35}), make(seed=seed, deals=None)
    first, checked = None, 0
    for t in advance(fair, 3000):
        twin.advance_to(t)
        first = first or (fair.deals[0] if fair.deals else None)
        if first is None:
            continue
        if first.outcome:
            break
        pi = pi_of(fair, first, t)
        alone, acquirer = twin.fairs[first.target], twin.fairs[first.acquirer]  # the books' values without deals
        assert fair.fairs[first.acquirer] == acquirer  # the acquirer is in no other deal: its value is its own
        assert fair.fairs[first.target] == pytest.approx(pi * first.ratio * acquirer + (1 - pi) * alone, abs=1e-9)
        checked += 1
    assert checked >= 3 and first.outcome  # it was on for some steps, and it ended


def test_a_book_that_is_in_none_of_the_deals_has_its_own_fair_value_to_the_bit():
    fair, twin = make(deals={"cool": 2.0, "length": 10.0}), make(deals=None)
    for t in advance(fair, 300):
        twin.advance_to(t)
        in_a_deal = {b for d in fair.deals for b in (d.target, d.acquirer)}
        for b in range(12):
            if b not in in_a_deal and not any(d.target == b for d in fair.ended):  # one that has not been a target
                assert fair.fairs[b] == twin.fairs[b]



# ---- A break, and a completion ----

FIRST = {"rate": 0.05, "breaks": 0.5, "priced": 0.5, "length": 30.0, "cool": 1e6}  # a book is in one deal, ever


def watch_first(seed, steps=4000, deals=FIRST, **kwargs):
    """Run the market of ``seed`` and its twin without deals a step at a time, yielding (time, market, twin, first deal)
    from the step the first deal is announced, with that step's ``gapped`` still there to be read. The target and the
    acquirer of the first deal are in no other deal, whatever else is announced, and no completion before the end of a
    run changes their levels, so the twin has their values without any deal until then."""
    fair, twin = make(seed=seed, deals=deals, **kwargs), make(seed=seed, deals=None, **kwargs)
    first = None
    for i in range(1, steps + 1):
        fair.gapped.clear()
        fair.advance_to(i * fair.step)
        twin.advance_to(i * fair.step)
        first = first or (fair.deals[0] if fair.deals else None)
        if first is not None:
            yield i * fair.step, fair, twin, first


def test_a_break_falls_back_to_the_value_alone_by_pi_times_the_gap_and_gaps_the_target_by_that():
    broke = 0
    for seed in SEEDS:
        ended_at = None
        for t, fair, twin, deal in watch_first(seed):
            alone, acquirer = twin.fairs[deal.target], twin.fairs[deal.acquirer]
            if deal.outcome == "broken" and ended_at is None:
                ended_at, broke = t, broke + 1
                gap = deal.ratio * acquirer - alone  # what the offer is worth over the target alone, now
                pi = pi_of(fair, deal, t)
                assert deal.ended_at == t and deal not in fair.deals and fair.ended[-1] is deal
                assert fair.fairs[deal.target] == alone  # the target is its value alone again
                assert [j for book, j in fair.gapped if book == deal.target] == [pytest.approx(-pi * gap, abs=1e-9)]
                assert t - fair.step < deal._breaks_at <= t <= deal.ends  # the first grid point at or after its time
                assert fair.fairs[deal.acquirer] == acquirer
            elif deal.outcome == "broken":
                assert fair.fairs[deal.target] == alone  # and it goes on as itself from there
                assert not [1 for book, _ in fair.gapped if book == deal.target]
                if t > ended_at + 40 * fair.step:
                    break
            elif deal.outcome:
                break
    assert broke >= 4


@pytest.mark.parametrize("seed", SEEDS)
def test_a_deal_breaks_at_the_first_grid_point_at_or_after_its_time_if_that_is_by_its_end_by_hand(seed):
    breaks, length, rate, steps = 0.6, 4.0, 0.05, 5000
    fair = make(seed=seed, deals={"rate": rate, "breaks": breaks, "length": length, "cool": 0.0})
    by_hand, _ = stream_by_hand(seed, rate, fair.step, steps)
    records = follow(fair, steps)
    assert len(records) == len(by_hand) > 30
    p = -math.log(1 - breaks) / length
    seen = {"broken": 0, "completed": 0}
    for (i, (_, _, wait)), record in zip(by_hand, records, strict=True):
        deal, announced = record["deal"], i * fair.step
        due = announced + wait / p
        if due <= deal.ends:  # a break
            k = i
            while k * fair.step < due:
                k += 1
            assert (deal.outcome, deal.ended_at) == ("broken", k * fair.step) and deal._breaks_at == due
        elif deal.ends <= steps * fair.step:
            assert (deal.outcome, deal.ended_at) == ("completed", deal.ends) and deal._breaks_at is None
        seen[deal.outcome or "on"] = seen.get(deal.outcome or "on", 0) + 1
    assert seen["broken"] > 5 and seen["completed"] > 5


def test_a_deal_whose_time_to_break_is_a_grid_point_breaks_at_that_point_and_not_at_the_next():
    fair = make(deals={"rate": 1.0, "length": 50.0, "breaks": 0.5, "cool": 0.0})
    for _t in advance(fair, 100):
        if fair.deals:
            break
    deal = fair.deals[0]
    fair.gapped.clear()
    due = (fair._steps + 1) * fair.step  # the time of the next step of the grid, to the bit
    deal._breaks_at = due
    fair.advance_to(due)
    assert (deal.outcome, deal.ended_at) == ("broken", due) and deal in fair.ended and deal not in fair.deals


def test_with_no_chance_of_a_break_every_deal_completes_and_none_has_a_time_to_break_at():
    fair = make(deals={"breaks": 0.0, "priced": 0.3, "length": 2.0, "rate": 0.2})
    records = follow(fair, 2000)
    outcomes = {r["deal"].outcome for r in records if r["deal"].ended_at is not None}
    assert outcomes == {"completed"} and all(r["deal"]._breaks_at is None for r in records) and len(records) > 20


@pytest.mark.parametrize("seed", SEEDS)
def test_at_completion_the_gap_is_0_the_level_takes_the_difference_and_the_fair_value_does_not_jump(seed):
    deals = {"rate": 0.05, "breaks": 0.0, "priced": 0.4, "length": 20.0, "cool": 1e6}
    done, held = None, []
    for t, fair, twin, deal in watch_first(seed, deals=deals, volatility=0.0, own_volatility=0.0, level=1000.0):
        if done is None and deal.outcome == "completed":
            done = t
            shift = deal.ratio * twin.fairs[deal.acquirer] - twin.fairs[deal.target]  # what the offer was worth over it
            assert deal.ended_at == deal.ends == t
            assert fair.fairs[deal.target] == pytest.approx(deal.ratio * fair.fairs[deal.acquirer], abs=1e-9)  # no gap
            assert fair.level[deal.target] - 1000.0 == pytest.approx(shift, abs=1e-9)
            assert not [1 for book, _ in fair.gapped if book == deal.target]  # a completion gaps nothing
            assert deal not in fair.deals and fair.ended[-1] is deal
        elif done is not None:
            held.append(fair.fairs[deal.target] - fair.level[deal.target])
            assert fair.fairs[deal.target] == pytest.approx(deal.ratio * fair.fairs[deal.acquirer], abs=1e-9)
            if len(held) == 60:
                break
    assert done is not None and np.ptp(held) < 1e-9  # it stays at the value of the offer, and goes on with no jump


def test_a_completion_changes_no_fair_value_beyond_the_step_it_would_have_made_with_the_mix():
    """With nothing moving, the target's value goes from the last step's mix to the offer's, ``(1 - pi)`` of the gap
    at most, and from then on nothing moves: it does not jump."""
    deals = {"rate": 0.05, "breaks": 0.0, "priced": 0.4, "length": 20.0, "cool": 1e6}
    for seed in (1, 2, 3):
        last, steps = None, []
        for t, fair, twin, deal in watch_first(seed, deals=deals, volatility=0.0, own_volatility=0.0):  # noqa: B007
            now = fair.fairs[deal.target]
            if last is not None:
                steps.append(abs(now - last))
            last = now
            if deal.outcome and len(steps) > 1 and t > deal.ends + 10 * fair.step:
                break
        end = round(deal.ends / fair.step) - round(deal.announced / fair.step)  # steps from the announcement to the end
        q = -math.log(1 - 0.4) / (round(20.0 / fair.step) * fair.step)
        assert max(steps[:end]) <= (1 - math.exp(-q * fair.step)) * 60.0 + 1e-9  # no step is more than the mix makes
        assert max(steps[end:]) < 1e-9  # and after the end none moves it


# ---- What a lot of the target earns: fair values only, where it is exact ----

@functools.lru_cache(maxsize=None)
def deal_profits(priced, breaks, within=True, steps=2500, premium=60.0):
    """What one lot of the target earns from just after the announcement to the deal's end, unhedged and hedged with
    ``r`` lots of the acquirer sold, at fair values alone (own parts 0, no lot taken): over the eight seeds, for each
    deal that was settled. A list of dicts: outcome, unhedged, hedged, and the error of the identity the hedged
    profit has (by hand: ``(1 - pi_0) * premium`` for a completion, ``-pi_0 * G_0 - (G - G_0)`` for a break)."""
    out = []
    for seed in SEEDS:
        # a book is not in a new deal in the step its last ends, or that step's value would be of the new deal
        settings = {"priced": priced, "breaks": breaks, "within": within, "premium": premium, "rate": 0.4,
                    "length": 6.0, "cool": 1.0}
        fair = make(seed=seed, deals=settings, own_volatility=0.0, step=1.0)
        records = follow(fair, steps)
        pi_0 = 1 - priced
        for r in settled(records, fair, steps):
            deal = r["deal"]
            unhedged = r["target_end"] - r["target"]
            hedged = unhedged - deal.ratio * (r["acquirer_end"] - r["acquirer"])
            gap = deal.ratio * r["acquirer_end"] - r["target_end"]  # G when it ended, from the target's value alone
            exact = ((1 - pi_0) * premium if deal.outcome == "completed" else -pi_0 * premium - (gap - premium))
            out.append({"outcome": deal.outcome, "unhedged": unhedged, "hedged": hedged, "error": hedged - exact})
    return out


def mean_and_error(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std() / math.sqrt(len(values))


@pytest.mark.parametrize("within", [True, False])
def test_with_priced_equal_to_breaks_a_lot_of_the_target_earns_nothing_on_average_before_costs(within):
    profits = deal_profits(0.2, 0.2, within)
    assert len(profits) > 5000
    mean, error = mean_and_error([p["unhedged"] for p in profits])
    assert abs(mean) < 3.5 * error and error < 0.6  # the gain of a completion and the loss of a break cancel
    mean, error = mean_and_error([p["hedged"] for p in profits])
    assert abs(mean) < 3.5 * error and error < 0.6
    share = np.mean([p["outcome"] == "broken" for p in profits])
    assert abs(share - 0.2) < 4 * math.sqrt(0.2 * 0.8 / len(profits))


@pytest.mark.parametrize("priced, breaks, expected", [(0.4, 0.2, 12.0), (0.2, 0.4, -12.0), (0.4, 0.4, 0.0)])
def test_a_lot_of_the_target_earns_the_gap_less_the_breaks_that_were_priced_on_average(priced, breaks, expected):
    profits = deal_profits(priced, breaks)
    for kind in ("unhedged", "hedged"):
        mean, error = mean_and_error([p[kind] for p in profits])
        assert abs(mean - expected) < 3.5 * error and error < 0.6
    # (1 - breaks) * (1 - pi_0) * G_0 - breaks * pi_0 * G_0: 12 with the design's 60, 40% priced and 20% broken
    assert expected == pytest.approx((1 - breaks) * priced * 60.0 - breaks * (1 - priced) * 60.0)


def test_a_completed_deal_earns_one_less_pi_0_times_the_premium_hedged_exactly_and_a_broken_one_what_it_works_out_to():
    profits = deal_profits(0.4, 0.2)
    completed = [p for p in profits if p["outcome"] == "completed"]
    broken = [p for p in profits if p["outcome"] == "broken"]
    assert len(completed) > 4000 and len(broken) > 1000
    assert max(abs(p["error"]) for p in profits) < 1e-9  # the first to 1e-9 on every deal, the second likewise
    assert max(abs(p["hedged"] - 24.0) for p in completed) < 1e-9  # +24 of every one: (1 - 0.6) * 60
    mean, error = mean_and_error([p["unhedged"] for p in completed])
    assert abs(mean - 24.0) < 3.5 * error  # unhedged it is 24 on average, the acquirer's move being 0 on average
    mean, error = mean_and_error([p["unhedged"] for p in broken])
    assert abs(mean + 36.0) < 3.5 * error  # and a broken one loses pi_0 * 60 = 36 on average
    mean, error = mean_and_error([p["hedged"] for p in broken])
    assert abs(mean + 36.0) < 3.5 * error  # and hedged the same: where G went in between is 0 on average


def test_across_sectors_what_the_hedge_leaves_is_larger_than_within_one():
    within, across = deal_profits(0.4, 0.2, True), deal_profits(0.4, 0.2, False)
    spread = lambda profits: np.std([p["hedged"] for p in profits if p["outcome"] == "broken"])  # noqa: E731
    assert spread(across) > 1.5 * spread(within)
    assert max(abs(p["error"]) for p in across) < 1e-9  # the identity holds whichever books they are


def test_the_designs_example_of_60_ticks_over_500_units_at_40_percent_priced_makes_24_and_36_exactly_when_hedged():
    deals = {"rate": 0.01, "premium": 60.0, "length": 500.0, "breaks": 0.5, "priced": 0.4, "cool": 1e6}
    outcomes = {"completed": 0, "broken": 0}
    for seed in SEEDS:
        entry = None
        for _t, fair, _twin, deal in watch_first(seed, steps=9000, deals=deals, own_volatility=0.0):
            if entry is None:
                entry = (fair.fairs[deal.target], fair.fairs[deal.acquirer])
                assert deal.ratio * entry[1] - entry[0] == pytest.approx(24.0, abs=1e-9)  # the gap opens at 24
            if deal.outcome:
                hedged = fair.fairs[deal.target] - entry[0] - deal.ratio * (fair.fairs[deal.acquirer] - entry[1])
                gap = deal.ratio * fair.fairs[deal.acquirer] - fair.fairs[deal.target]
                if deal.outcome == "completed":
                    assert hedged == pytest.approx(24.0, abs=1e-9)
                else:
                    assert hedged == pytest.approx(-36.0 - (gap - 60.0), abs=1e-9)
                outcomes[deal.outcome] += 1
                break
    assert min(outcomes.values()) >= 1


# ---- A lot taken in a book ----

def first_deal_on(seed=1, table=FOUR, **kwargs):
    """A market, at the step its first deal is on, and the twin without deals, and the deal."""
    for _t, fair, twin, deal in watch_first(seed, deals=FIRST | {"rate": 0.2}, table=table, **kwargs):
        return fair, twin, deal


def test_a_lot_in_the_target_moves_its_fair_value_by_one_less_pi_times_the_impact_and_a_lot_in_its_acquirer_both():
    fair, twin, deal = first_deal_on(own_volatility=0.8, impact=0.05)
    t = fair._steps * fair.step
    pi = pi_of(fair, deal, t)
    others = [b for b in range(4) if b not in (deal.target, deal.acquirer)]
    before = list(fair.fairs)
    fair.trade(deal.target, 3)
    after = list(fair.fairs)
    assert after[deal.target] - before[deal.target] == pytest.approx((1 - pi) * 0.05 * 3, abs=1e-12)
    assert fair.own[deal.target] - twin.own[deal.target] == pytest.approx(0.05 * 3, abs=1e-12)  # its own part, whole
    assert [after[b] for b in range(4) if b != deal.target] == [before[b] for b in range(4) if b != deal.target]
    fair.trade(deal.acquirer, -2)
    again = list(fair.fairs)
    assert again[deal.acquirer] - after[deal.acquirer] == pytest.approx(-0.05 * 2, abs=1e-12)
    assert again[deal.target] - after[deal.target] == pytest.approx(-pi * deal.ratio * 0.05 * 2, abs=1e-12)
    assert fair.own[deal.acquirer] - twin.own[deal.acquirer] == pytest.approx(-0.05 * 2, abs=1e-12)
    fair.trade(others[0], 5)
    assert fair.fairs[others[0]] - again[others[0]] == pytest.approx(0.05 * 5, abs=1e-12)  # a book in no deal
    assert [fair.fairs[b] for b in (deal.target, deal.acquirer, others[1])] == [
        again[deal.target], again[deal.acquirer], again[others[1]]]


def test_a_lot_in_the_index_with_a_deal_on_moves_its_premium_by_its_scale_times_the_impact_and_nothing_else():
    fair, _twin, deal = first_deal_on(own_volatility=0.8, impact=0.05, index=[1.0, 2.0, 0.5, 0.5], index_volatility=0.4)
    assert fair.deals and fair.scales[4] == 4.0
    before = (fair.premium, list(fair.fairs), fair.own.copy(), fair._standalone.copy())
    for lots in (3, -2):
        fair.trade(4, lots)
    move = 4.0 * 0.05 * (3 - 2)  # the scale of the index, the impact and the lots taken: 0.2
    assert fair.premium == pytest.approx(before[0] + move, abs=1e-12)
    assert fair.fairs[4] == pytest.approx(before[1][4] + move, abs=1e-12)
    assert fair.own[4] == pytest.approx(before[2][4] + move, abs=1e-12)
    assert fair.fairs[:4] == before[1][:4] and fair.own[:4].tolist() == before[2][:4].tolist()  # no book moved
    assert fair._standalone.tolist() == before[3].tolist() and deal in fair.deals


def test_a_lot_taken_in_a_book_in_a_deal_stays_in_its_fair_value_as_the_deal_goes_on():
    """The fair value of a target some steps after lots were taken in it and in its acquirer is the mix, by hand, of
    the values the books have alone, which include what was taken: the same lots taken in the twin without deals."""
    fair, twin, deal = first_deal_on(own_volatility=0.8, impact=0.2)
    for lots in (4, -1):
        fair.trade(deal.target, lots)
        twin.trade(deal.target, lots)
    fair.trade(deal.acquirer, 3)
    twin.trade(deal.acquirer, 3)
    for _ in range(25):
        t = (fair._steps + 1) * fair.step
        fair.advance_to(t)
        twin.advance_to(t)
        if deal.outcome:
            break
        pi = pi_of(fair, deal, t)
        assert fair.fairs[deal.acquirer] == pytest.approx(twin.fairs[deal.acquirer], abs=1e-9)
        assert fair.fairs[deal.target] == pytest.approx(
            pi * deal.ratio * twin.fairs[deal.acquirer] + (1 - pi) * twin.fairs[deal.target], abs=1e-9)
        assert fair.own.tolist() == twin.own.tolist()


def test_a_lot_in_a_book_with_no_deal_on_is_what_it_always_was_to_the_bit():
    fair, twin = make(seed=3, deals={"rate": 0.0}, impact=0.07), make(seed=3, deals=None, impact=0.07)
    for t in advance(fair, 50):
        twin.advance_to(t)
        for book, lots in ((1, 2), (4, -3), (1, 1)):
            fair.trade(book, lots)
            twin.trade(book, lots)
        assert fair.fairs == twin.fairs and fair.own.tolist() == twin.own.tolist()


def test_the_exposure_of_a_target_in_a_deal_is_the_mix_of_its_row_and_the_acquirers_and_it_is_a_copy():
    fair, twin, deal = first_deal_on(seed=2, table=SECTORS)
    assert deal.outcome is None
    t = fair._steps * fair.step
    pi = pi_of(fair, deal, t)
    rows = fair.exposures()
    assert rows is not fair.loadings and rows.shape == fair.loadings.shape
    expected = pi * deal.ratio * fair.loadings[deal.acquirer] + (1 - pi) * fair.loadings[deal.target]
    assert rows[deal.target] == pytest.approx(expected, abs=1e-12)
    others = [b for b in range(12) if b != deal.target]
    assert np.array_equal(rows[others], fair.loadings[others])  # nobody else's row changes
    before = fair.loadings.copy()
    rows[:] = 0.0
    assert np.array_equal(fair.loadings, before)  # changing the copy changes nothing
    assert deal.target // 4 == deal.acquirer // 4  # one sector: the rows are parallel, so the mix is a hair longer or
    assert fair.exposures()[deal.target] == pytest.approx((1 + pi * (deal.ratio - 1)) * fair.loadings[deal.target])


def test_exposures_are_the_loadings_themselves_while_no_deal_is_on_also_in_a_market_that_has_deals():
    fair = make(deals={"rate": 0.0})
    assert fair.exposures() is fair.loadings
    fair = make(deals={"rate": 0.3, "length": 3.0, "cool": 0.0}, loading_sd=0.2, loading_half_life=30.0)
    seen = {"on": 0, "off": 0}
    for _t in advance(fair, 800):
        fair.gapped.clear()
        if fair.deals:
            assert fair.exposures() is not fair.loadings
            seen["on"] += 1
        else:
            assert fair.exposures() is fair.loadings
            seen["off"] += 1
    assert seen["on"] > 50 and seen["off"] > 20


def test_the_exposures_of_an_index_are_its_weights_times_those_of_the_books_with_a_deal_on():
    weights = [0.5, 1.0, 0.5, 1.0]
    fair = make(FOUR, deals=FIRST | {"rate": 0.2}, index=weights)
    assert fair.exposures() is fair.loadings and fair.loadings.shape == (5, 3)
    for _ in advance(fair, 4000):
        if fair.deals:
            break
    (deal,) = fair.deals
    rows, pi = fair.exposures(), pi_of(fair, deal, fair._steps * fair.step)
    assert rows is not fair.loadings and rows.shape == (5, 3)
    assert rows[4] == pytest.approx(np.array(weights) @ rows[:4], abs=1e-12)
    # one sector: the target's row is longer by pi * (r - 1) of itself, and the index's by its weight times that
    assert rows[4] - fair.loadings[4] == pytest.approx(
        weights[deal.target] * pi * (deal.ratio - 1) * fair.loadings[deal.target], abs=1e-12)
    assert deal.ratio != 1.0


# ---- What deals leave alone ----

OPTIONS = {
    "plain": {},
    "an index": {"index": True, "index_volatility": 0.4},
    "loadings that drift": {"loading_sd": 0.2, "loading_half_life": 40.0},
    "a trend and an index with weights": {"trend": 0.05, "index": [0.5] * 12, "index_volatility": 0.2},
}


@pytest.mark.parametrize("option", list(OPTIONS))
def test_a_market_with_deals_and_none_on_yet_is_the_path_of_the_same_market_without_them_in_fair_values_and_all(option):
    for seed in (1, 2, 3):
        fair = make(seed=seed, deals={"rate": 0.01, "cool": 0.0}, **OPTIONS[option])
        twin = make(seed=seed, deals=None, **OPTIONS[option])
        steps = 0
        for t in advance(fair, 2000):
            twin.advance_to(t)
            steps += 1
            if fair.deals:
                break
            assert fair.fairs == twin.fairs and not fair.gapped and not fair.ended  # nothing is on, nothing changes
            for name in ("factors", "common", "own", "loadings", "level"):
                assert np.array_equal(getattr(fair, name), getattr(twin, name))
            assert fair.pushed == twin.pushed and fair.premium == twin.premium
        assert fair.deals and steps > 20  # and then there is one


@pytest.mark.parametrize("option", list(OPTIONS))
def test_the_factors_the_own_parts_and_the_streams_are_the_same_with_deals_as_without_whatever_the_deals_do(option):
    fair = make(seed=4, deals={"rate": 0.3, "length": 4.0, "cool": 0.5, "breaks": 0.4}, **OPTIONS[option])
    twin = make(seed=4, deals=None, **OPTIONS[option])
    for t in advance(fair, 3000):
        twin.advance_to(t)
        fair.gapped.clear()
    assert fair.ended  # a good many deals went by
    for name in ("factors", "common", "own", "loadings", "mean_loadings"):
        if name == "common" and fair.index_weights is not None:  # a completion changes a level, which is not in common
            assert np.array_equal(fair.common, twin.common)
        assert np.array_equal(getattr(fair, name), getattr(twin, name))
    assert fair.pushed == twin.pushed and fair.premium == twin.premium
    assert fair._rng.bit_generator.state == twin._rng.bit_generator.state
    if fair.index_weights is not None:
        assert fair._prng.bit_generator.state == twin._prng.bit_generator.state
    assert fair.level.tolist() != twin.level.tolist()  # only the levels of those that completed differ


def test_deals_with_a_rate_of_0_gap_nothing_end_nothing_and_draw_a_uniform_a_step():
    fair = make(deals={"rate": 0.0})
    twin = make(deals=None)
    rng = np.random.default_rng([1, 5])
    for t in advance(fair, 500):
        twin.advance_to(t)
        rng.random()
        assert fair.fairs == twin.fairs and fair.gapped == [] and fair.deals == [] and fair.ended == []
    assert fair._drng.bit_generator.state == rng.bit_generator.state


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_the_announcements_depend_on_the_seed_and_the_time_alone_and_not_on_what_is_traded(seed):
    settings = {"rate": 0.2, "length": 5.0, "cool": 1.0, "breaks": 0.4}
    quiet, busy = make(seed=seed, deals=settings), make(seed=seed, deals=settings)
    rng = np.random.default_rng(seed + 100)
    for t in advance(quiet, 1500):
        busy.advance_to(t)
        for _ in range(3):
            busy.trade(int(rng.integers(12)), float(rng.integers(-5, 6)))  # lots taken, which move own parts
        quiet.gapped.clear()
        busy.gapped.clear()
    key = lambda d: (d.announced, d.target, d.acquirer, d.ends, d.outcome, d.ended_at)  # noqa: E731
    assert [key(d) for d in quiet.ended] == [key(d) for d in busy.ended] and len(quiet.ended) > 50
    assert [key(d) for d in quiet.deals] == [key(d) for d in busy.deals]
    assert quiet._drng.bit_generator.state == busy._drng.bit_generator.state
    assert [d.ratio for d in quiet.ended] != [d.ratio for d in busy.ended]  # what each offer is worth is not the same


def test_a_deal_is_announced_at_the_same_time_between_the_same_books_in_two_markets_of_one_seed_whatever_their_breaks():
    runs = {}
    for breaks in (0.1, 0.5):
        fair = make(seed=5, deals={"rate": 0.05, "length": 20.0, "cool": Deals().cool, "breaks": breaks, "priced": 0.2})
        records = follow(fair, 3000)
        runs[breaks] = {(r["deal"].announced, r["deal"].target, r["deal"].acquirer): r["deal"] for r in records}
    # a broken deal's books stay out till its planned end: every deal is announced as it was, and only its end differs
    assert runs[0.1].keys() == runs[0.5].keys() and len(runs[0.1]) > 10
    outcomes = {d.outcome for run in runs.values() for d in run.values()}
    assert {"completed", "broken"} <= outcomes  # and the deals are not all alike in how they end


# ---- Breaks across the dial ----

def test_the_deals_that_break_at_a_lower_chance_of_a_break_are_some_of_those_that_break_at_a_higher_one():
    """The same seed at two values of ``breaks``, ``priced`` held, at the default cooling time: the deals are the same
    (a broken deal keeps its books out till its planned end), the break time is one exponential over the hazard, so it
    is later at the lower, and a deal that breaks there breaks at the higher too, no later."""
    found = {}
    for breaks in (0.15, 0.45):
        deals = {}
        for seed in SEEDS:
            fair = make(seed=seed, deals={"rate": 0.05, "length": 20.0, "cool": Deals().cool, "breaks": breaks,
                                          "priced": 0.3})
            for r in settled(follow(fair, 2500), fair, 2500):  # those that are over, whatever their outcome
                deal = r["deal"]
                deals[(seed, deal.announced, deal.target, deal.acquirer)] = deal
        found[breaks] = deals
    low, high = found[0.15], found[0.45]
    assert low.keys() == high.keys() and len(low) > 40  # the same deals, announced as they were
    broke_low = {k for k, d in low.items() if d.outcome == "broken"}
    broke_high = {k for k, d in high.items() if d.outcome == "broken"}
    assert broke_low and broke_low < broke_high  # some of them: strictly fewer
    p_low, p_high = -math.log(1 - 0.15) / 20.0, -math.log(1 - 0.45) / 20.0
    for key in broke_low:
        assert high[key].ended_at <= low[key].ended_at  # and no later
        assert (low[key]._breaks_at - key[1]) * p_low == pytest.approx((high[key]._breaks_at - key[1]) * p_high)
    for key in high.keys() - broke_high:
        assert low[key].outcome == "completed"  # one that goes through at the higher goes through at the lower
    # one that breaks at the higher by the exponential's draw alone: the break time is beyond the end at the lower
    for key in broke_high - broke_low:
        assert low[key]._breaks_at is None and high[key]._breaks_at is not None


DIAL = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6)


@functools.lru_cache(maxsize=None)
def dial(seed):
    """Fair values alone at the default settings of ``Deals`` (a cooling time of 100) but for ``priced`` 0.4, in the
    twelve books of three sectors, 20,000 units of one seed at each of ``DIAL``'s values of ``breaks``: the deals of
    each in the order announced, ended and on."""
    runs = {}
    for breaks in DIAL:
        fair = FactorValues(SECTORS, deals=Deals(priced=0.4, breaks=breaks))
        fair.reseed(seed)
        fair.advance_to(20000.0)
        runs[breaks] = sorted(fair.ended + fair.deals, key=lambda d: d.announced)
    return runs


def announced(deals):
    return [(d.announced, d.target, d.acquirer) for d in deals]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_same_seed_has_the_same_deals_announced_at_every_value_of_breaks_with_priced_held(seed):
    runs = dial(seed)
    assert Deals().cool == 100.0 and len(runs[0.2]) > 25  # the default cooling time, and deals enough to tell
    for breaks in DIAL:
        assert announced(runs[breaks]) == announced(runs[0.2]), breaks  # the time, the target and the acquirer
        assert [d.ends for d in runs[breaks]] == [d.ends for d in runs[0.2]]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_deals_that_break_at_a_smaller_value_of_breaks_are_some_of_those_that_break_at_a_larger(seed):
    runs = dial(seed)
    broke = {b: {(d.announced, d.target, d.acquirer) for d in runs[b] if d.outcome == "broken"} for b in DIAL}
    assert broke[0.0] == set()  # a deal that cannot break does not
    for smaller, larger in zip(DIAL, DIAL[1:], strict=False):
        assert broke[smaller] <= broke[larger], (smaller, larger)


def test_across_the_dial_there_are_deals_enough_and_the_share_that_breaks_rises_with_it():
    runs = {seed: dial(seed) for seed in SEEDS}
    assert sum(len(runs[seed][0.2]) for seed in SEEDS) > 250  # 329 on these eight seeds
    share = {b: sum(d.outcome == "broken" for seed in SEEDS for d in runs[seed][b]) for b in DIAL}
    assert list(share.values()) == sorted(share.values()) and share[0.0] == 0 and share[0.6] > 3 * share[0.1] > 0


def free_gaps(seed):
    """For each broken deal of ``dial(seed)``, at each value of ``breaks`` above 0, how long after its planned end plus
    the cooling time the first announcement that picks a book of it came (when one did)."""
    cool, gaps = Deals().cool, []
    for breaks in DIAL[1:]:
        deals = dial(seed)[breaks]
        for i, deal in enumerate(deals):
            if deal.outcome == "broken":
                later = [d.announced for d in deals[i + 1:] if {d.target, d.acquirer} & {deal.target, deal.acquirer}]
                if later:
                    gaps.append(min(later) - (deal.ends + cool))
    return gaps


@pytest.mark.parametrize("seed", SEEDS)
def test_a_broken_deals_books_are_picked_by_no_announcement_before_its_planned_end_plus_the_cooling_time(seed):
    gaps = free_gaps(seed)
    assert gaps and min(gaps) >= -1e-9  # not before the planned end (not the break, long before) plus the cooling time


def test_a_broken_deals_books_can_be_picked_again_from_its_planned_end_plus_the_cooling_time():
    gaps = np.array([gap for seed in SEEDS for gap in free_gaps(seed)])
    assert len(gaps) > 400 and (gaps < 50).sum() > 20  # 517 of them on these seeds, 35 within 50 units of being free
    broken = [d for seed in SEEDS for d in dial(seed)[0.6] if d.outcome == "broken"]
    assert all(d.ended_at <= d.ends for d in broken) and sum(d.ended_at < d.ends for d in broken) > 40  # early ones


def scripted(breaks, cool, wait=0.001):
    """Two books, a deal announced at every step the books are free (every draw is 0, and the exponential ``wait``, so
    that a deal that can break does so at the next step), step 0.5 and length 5 (10 steps): the steps announced at."""
    fair = make(TWO, deals={"rate": 1.0, "length": 5.0, "cool": cool, "breaks": breaks}, step=0.5)

    class Draws:
        def random(self):
            return 0.0

        def standard_exponential(self):
            return wait

    fair._drng = Draws()
    for _t in advance(fair, 60):
        fair.gapped.clear()
    return fair, [round(d.announced / fair.step) for d in sorted(fair.ended + fair.deals, key=lambda d: d.announced)]


@pytest.mark.parametrize("cool, steps", [(0.0, [1, 11, 21, 31, 41, 51]), (3.0, [1, 17, 33, 49])])
@pytest.mark.parametrize("breaks", [0.0, 0.5])
def test_the_books_of_a_deal_are_free_again_at_its_planned_end_plus_the_cooling_time_broken_or_completed(
        breaks, cool, steps):
    fair, announced_at = scripted(breaks, cool)
    assert announced_at == steps  # a deal of 10 steps and a cooling time of 0 or 6 steps
    outcomes = {d.outcome for d in fair.ended}
    assert outcomes == ({"completed"} if breaks == 0.0 else {"broken"})  # every deal that can break does, at once
    if breaks:
        assert all(d.ended_at == d.announced + fair.step for d in fair.ended)  # the next step: well before its end


def test_after_a_break_the_books_are_free_from_the_planned_end_plus_the_cooling_time_and_not_from_the_break():
    fair = make(TWO, deals={"rate": 1.0, "length": 5.0, "cool": 3.0, "breaks": 0.5}, step=0.5)

    class Once:
        def __init__(self):
            self.draws = iter([0.0, 0.0, 0.0, 0.001] + [1.0] * 100)  # one deal at step 1, which breaks at step 2

        def random(self):
            return next(self.draws)

        def standard_exponential(self):
            return next(self.draws)

    fair._drng = Once()
    for _t in advance(fair, 2):
        fair.gapped.clear()
    deal = fair.ended[0]
    assert (deal.outcome, deal.ended_at, deal.ends) == ("broken", 1.0, 5.5) and fair.deals == []
    assert fair._free_at.tolist() == [17, 17]  # step 11 is its planned end, and 6 steps more: not 2 + 6 = 8


@pytest.mark.parametrize("breaks", [0.1, 0.3, 0.6])
def test_the_share_of_deals_that_break_is_the_chance_of_a_break_given(breaks):
    total = broken = 0
    for seed in SEEDS:
        settings = {"breaks": breaks, "priced": 0.2, "rate": 0.4, "length": 6.0, "cool": 1.0}
        fair = make(seed=seed, deals=settings, step=1.0)
        settled_deals = settled(follow(fair, 2000), fair, 2000)
        total += len(settled_deals)
        broken += sum(r["deal"].outcome == "broken" for r in settled_deals)
    assert total > 4000
    assert abs(broken / total - breaks) < 4 * math.sqrt(breaks * (1 - breaks) / total)


def test_a_deal_breaks_by_its_end_with_the_chance_given_whatever_the_step_and_the_length_rounds_to():
    for step, length in ((0.2, 5.0), (0.5, 2.2), (0.3, 1.0)):
        total = broken = 0
        for seed in SEEDS:
            fair = make(seed=seed, deals={"breaks": 0.35, "rate": 1.0, "length": length, "cool": step}, step=step)
            done = settled(follow(fair, 4000), fair, 4000)
            total += len(done)
            broken += sum(r["deal"].outcome == "broken" for r in done)
        assert abs(broken / total - 0.35) < 4 * math.sqrt(0.35 * 0.65 / total)


# ---- Which books ----

def test_a_target_and_an_acquirer_are_two_books_in_no_deal_past_their_cooling_time_and_never_the_index():
    cool, step = 3.0, 0.5
    fair = make(deals={"rate": 1.0, "length": 4.0, "cool": cool, "breaks": 0.4}, index=True, step=step)
    records = follow(fair, 6000)
    assert len(records) > 300 and fair.n == 13
    by_book = {}
    for record in records:
        deal = record["deal"]
        assert deal.target != deal.acquirer and max(deal.target, deal.acquirer) < 12  # never the index, book 12
        for book in (deal.target, deal.acquirer):
            by_book.setdefault(book, []).append(deal)
    for _book, deals in by_book.items():
        for last, deal in zip(deals, deals[1:], strict=False):
            assert last.ended_at is not None and deal.announced >= last.ended_at + cool - 1e-9  # out of it, and cooled
    assert set(by_book) == set(range(12))  # every book gets a deal


def test_the_deals_on_are_settled_before_an_announcement_so_with_no_cooling_their_books_are_free_for_it_that_step():
    fair = make(TWO, deals={"rate": 1.8, "length": 5.0, "cool": 0.0, "breaks": 0.0}, step=0.5)
    records = follow(fair, 3000)
    pairs = list(zip(records, records[1:], strict=False))
    same_step = [new["deal"].announced == old["deal"].ended_at for old, new in pairs if old["deal"].ended_at]
    assert len(same_step) > 100 and np.mean(same_step) > 0.8  # a deal ends and the next is announced at that same step
    assert all(new["deal"].announced >= old["deal"].ended_at for old, new in pairs if old["deal"].ended_at)


def test_a_book_is_in_one_deal_at_a_time_and_with_fewer_than_two_free_nothing_is_announced():
    fair = make(FOUR, deals={"rate": 1.0, "length": 5.0, "cool": 0.0, "breaks": 0.0})
    for _t in advance(fair, 1500):
        fair.gapped.clear()
        books = [b for d in fair.deals for b in (d.target, d.acquirer)]
        assert len(books) == len(set(books)) and len(fair.deals) <= 2


def test_with_deals_within_a_sector_the_acquirer_is_of_the_targets_sector_when_a_book_of_it_is_free_else_any():
    group = [0, 0, 0, 1, 1]  # sectors of three books and of two: a target with its two sector-mates busy has none free
    table = np.array([[1.0, 1.0, 0.0]] * 3 + [[1.0, 0.0, 1.0]] * 2)
    fair = make(table, deals={"rate": 1.0, "length": 3.0, "cool": 0.0, "breaks": 0.3, "within": True}, step=0.5)
    inside = outside = 0
    for t in advance(fair, 8000):
        fair.gapped.clear()
        for deal in [d for d in fair.deals if d.announced == t]:  # announced at this step: the books of the others on
            busy = {b for d in fair.deals if d is not deal for b in (d.target, d.acquirer)}
            busy |= {b for d in fair.ended[-30:] if d.ends > t for b in (d.target, d.acquirer)}  # kept out till its end
            mates = [b for b in range(5) if group[b] == group[deal.target] and b not in busy and b != deal.target]
            if mates:
                assert deal.acquirer in mates
                inside += 1
            else:
                assert group[deal.acquirer] != group[deal.target] and deal.acquirer not in busy  # from another sector
                outside += 1
    assert inside > 300 and outside > 30  # and the rule is used, both ways


def test_with_deals_not_within_a_sector_the_acquirer_is_from_anywhere_among_the_free_books():
    fair = make(deals={"rate": 1.0, "length": 3.0, "cool": 0.0, "within": False}, step=0.5)
    records = follow(fair, 6000)
    same = sum(r["deal"].target // 4 == r["deal"].acquirer // 4 for r in records)
    share = same / len(records)
    assert len(records) > 400 and 0.15 < share < 0.5  # 3 in 11 if all are free, more as the sector-mates are busy
    assert len({r["deal"].acquirer // 4 for r in records if r["deal"].target // 4 == 0}) == 3


def test_in_a_table_with_no_two_rows_parallel_the_acquirer_is_from_anywhere_with_within_on_too():
    fair = make(MIXED, deals={"rate": 1.0, "length": 3.0, "cool": 0.0, "within": True}, step=0.5)
    assert not (fair._parallel & ~np.eye(6, dtype=bool)).any()  # no two books are parallel
    records = follow(fair, 6000)
    pairs = {(r["deal"].target, r["deal"].acquirer) for r in records}
    assert len(records) > 400 and pairs == {(a, b) for a in range(6) for b in range(6) if a != b}


def test_a_table_of_sectors_says_which_rows_are_parallel_and_loadings_that_drift_use_their_means():
    fair = make(deals={"rate": 0.0}, loading_sd=0.3, loading_half_life=20.0)
    expected = np.array([[a // 4 == b // 4 for b in range(12)] for a in range(12)])
    assert np.array_equal(fair._parallel, expected)
    for _t in advance(fair, 200):
        pass
    assert np.array_equal(fair._parallel, expected)  # the means, not the rows as they drift


def test_rows_are_parallel_when_their_directions_are_the_same_but_for_a_few_parts_in_a_hundred_thousand():
    # The cosine of two rows (1, a) and (1, b) is 1 less (a - b) ** 2 / 2 to the order: 4.5e-10 for 3e-5 apart, which
    # is the same direction, and 1.8e-9 for 6e-5 apart, which is not.
    table = np.array([[1.0, 0.0], [1.0, 3e-5], [1.0, 6e-5], [1.0, 1.0]])
    fair = make(table, deals={"rate": 0.0})
    expected = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=bool)
    assert np.array_equal(fair._parallel, expected)


# ---- One object of settings, many markets ----

def test_one_deals_object_given_to_eight_markets_shares_nothing_between_them():
    settings = Deals(rate=0.3, length=4.0, cool=0.5, breaks=0.4)
    shared = [FactorValues(SECTORS, deals=settings, step=0.5) for _ in SEEDS]
    for seed, fair in zip(SEEDS, shared, strict=True):
        fair.reseed(seed)
    for i in range(1, 801):  # all eight at once, a step each in turn
        for fair in shared:
            fair.advance_to(i * 0.5)
            fair.gapped.clear()
    assert all(fair.deal_settings is settings for fair in shared)
    assert len({id(fair.deals) for fair in shared} | {id(fair.ended) for fair in shared}) == 16
    key = lambda d: (d.target, d.acquirer, d.ratio, d.announced, d.ends, d.outcome, d.ended_at, d._breaks_at)  # noqa: E731
    for seed, fair in zip(SEEDS, shared, strict=True):
        alone = FactorValues(SECTORS, deals=Deals(rate=0.3, length=4.0, cool=0.5, breaks=0.4), step=0.5)
        alone.reseed(seed)
        alone.advance_to(400.0)
        assert [key(d) for d in fair.deals] == [key(d) for d in alone.deals]
        assert [key(d) for d in fair.ended] == [key(d) for d in alone.ended] and len(alone.ended) > 40


def test_a_reseed_starts_the_deals_afresh_with_none_on_none_ended_nothing_gapped_and_every_book_free():
    fair = make(deals={"rate": 0.3, "length": 4.0, "cool": 3.0, "breaks": 0.4}, step=0.5)
    for _t in advance(fair, 400):
        if fair.deals and fair.ended:
            break
    assert fair.deals and fair.ended and fair.gapped and fair._free_at.any()
    kept = fair.ended
    fair.reseed(3)
    assert fair.deals == [] and fair.ended == [] and fair.gapped == [] and not fair._free_at.any()
    assert kept and fair._drng.bit_generator.state == np.random.default_rng([3, 5]).bit_generator.state
    assert fair.deal_settings is not None


# ---- The index, with deals ----

def check_index(fair):
    m, c = fair.m, fair.index_weights
    assert fair.fairs[m] == pytest.approx(float(c @ fair.fairs[:m]) + fair.premium, abs=1e-8)  # the sum, after the mix
    assert fair.own[m] == pytest.approx(float(c @ fair.own[:m]) + fair.premium, abs=1e-8)


def test_with_an_index_its_fair_value_is_the_weighted_sum_of_the_books_after_the_mix_at_every_step_and_every_lot():
    for weights in ([0.5, 1.0, 1.0, 0.5], True):
        fair = make(FOUR, deals=FIRST | {"rate": 0.3, "length": 8.0, "cool": 2.0}, index=weights, index_volatility=0.3)
        rng = np.random.default_rng(7)
        on = 0
        for _t in advance(fair, 3000):
            fair.gapped.clear()
            check_index(fair)
            on += bool(fair.deals)
            for _ in range(2):
                fair.trade(int(rng.integers(5)), float(rng.integers(-4, 5)))  # the books and the index
                check_index(fair)
        assert on > 500 and len(fair.ended) > 20


def test_with_an_index_a_completion_leaves_its_level_the_weighted_sum_of_the_books_levels_and_the_identity_holds():
    fair = make(FOUR, deals={"rate": 0.05, "length": 4.0, "cool": 1.0, "breaks": 0.0}, index=[0.5, 1.0, 1.0, 0.5],
                index_volatility=0.3)
    c, checked = fair.index_weights, 0
    for _t in advance(fair, 4000):
        fair.gapped.clear()
        if fair.ended and not fair.deals:  # a completion has happened and none is on
            checked += 1
            assert fair.level[4] == pytest.approx(float(c @ fair.level[:4]), abs=1e-8)
            for book in range(5):  # every fair value is its level and its common and its own part
                alone = fair.level[book] + fair.common[book] + fair.own[book]
                assert fair.fairs[book] == pytest.approx(alone, abs=1e-8)
    assert checked > 100 and len(fair.ended) > 5


def test_with_an_index_it_gaps_by_its_weight_times_what_a_book_gapped_when_that_is_half_a_tick_or_more():
    weights = np.array([0.01, 0.01, 1.0, 1.0])  # a deal in the first two moves the index by under half a tick
    fair = make(FOUR, deals=FIRST | {"rate": 0.3, "length": 3.0, "cool": 1.0}, index=weights)
    small = large = 0
    for _t in advance(fair, 6000):
        entries, fair.gapped = list(fair.gapped), []
        books = [(b, j) for b, j in entries if b < 4]
        index = [j for b, j in entries if b == 4]
        expected = sum(weights[b] * j for b, j in books)
        if not books:
            assert index == []
        elif abs(expected) >= 0.5:
            assert index == [pytest.approx(expected, abs=1e-9)] and entries[-1][0] == 4  # after the books'
            large += 1
        else:
            assert index == []
            small += 1
    assert large > 100 and small > 30


@pytest.mark.parametrize("premium, gaps", [(1.0, True), (1.0 + 1e-9, True), (1.0 - 1e-9, False), (0.4, False)])
def test_an_index_gap_of_exactly_half_a_tick_is_taken_and_one_a_hair_under_it_is_not(premium, gaps):
    # priced 0 means a probability of 1 that the deal goes through: the target gaps by the whole premium, and an index
    # of two books at a half each by half of that
    fair = make(TWO, deals={"rate": 2.0, "priced": 0.0, "premium": premium, "length": 6.0}, index=[0.5, 0.5])
    for _t in advance(fair, 200):
        if fair.deals:
            break
    deal = fair.deals[0]
    assert fair.gapped[0] == (deal.target, premium)
    assert fair.gapped[1:] == ([(2, 0.5 * premium)] if gaps else [])


# ---- The books' orders follow a jump ----

def a_market(table=FOUR, seed=1, deals=None, **kwargs):
    fair = FactorValues(table, deals=Deals(**(BUSY | (deals or {}))), **kwargs)
    return BookMarket(fair, seed=seed)


def instrument(market):
    """Note every shift the market makes (what it did to the book, and what it left alone) and what ``gapped`` held
    after each advance of the fair values that had something in it."""
    shifts, advances, original, fair = [], [], market._shift, market.fair

    def shift(book, ticks):
        was = (market.books[book].best_bid, market.books[book].num_orders(), market.events, market.flow.copy(),
               market.volume.copy(), list(fair.fairs), market.time)
        original(book, ticks)
        now = (market.books[book].best_bid, market.books[book].num_orders(), market.events, market.flow.copy(),
               market.volume.copy(), list(fair.fairs), market.time)
        shifts.append((book, ticks, was, now))

    advance_to = fair.advance_to

    def noting(t):
        advance_to(t)
        if fair.gapped:
            advances.append(list(fair.gapped))

    market._shift, fair.advance_to = shift, noting
    return shifts, advances


def test_a_book_shifts_at_a_jump_by_the_jump_and_at_no_other_time_and_counts_no_event_flow_or_volume():
    m = a_market(deals={"rate": 0.3, "length": 6.0, "cool": 0.0, "breaks": 0.5, "priced": 0.3})
    shifts, advances = instrument(m)
    m.run(300.0)
    gapped = [entry for entry_list in advances for entry in entry_list]
    assert [(b, round(j)) for b, j in gapped] == [(s[0], s[1]) for s in shifts]  # one for each jump, in order, no more
    announced = len(m.fair.deals) + len(m.fair.ended)
    assert len(gapped) == announced + sum(d.outcome == "broken" for d in m.fair.ended) and announced > 30
    for _book, ticks, was, now in shifts:
        assert now[0] - was[0] == ticks and now[1] == was[1]  # the quotes move by the jump, and keep every order
        assert now[2] == was[2] and np.array_equal(now[3], was[3]) and np.array_equal(now[4], was[4])  # no event, flow
        assert now[5] == was[5] and now[6] == was[6]  # moves no fair value
    assert sum(s[1] != 0 for s in shifts) > 40


def test_the_rate_of_every_book_is_that_of_the_orders_it_has_after_every_step_with_deals_on():
    m = a_market(deals={"rate": 0.3, "length": 6.0, "cool": 0.0}, index=True)
    shifts, _ = instrument(m)
    for _ in range(4000):
        m.step()
        assert m._rates == [model.rate(book) for model, book in zip(m.models, m.books, strict=True)]
    assert len(shifts) > 20


def test_an_announcement_moves_the_target_book_to_the_new_fair_value_before_anyone_can_trade_there():
    """Priced at no chance of a break the target jumps by the whole 60 and then does not drift to the offer, so a mid a
    few ticks from its fair value is a book that moved with it: a book that did not is 60 away for some units."""
    m = a_market(deals={"rate": 0.3, "length": 6.0, "cool": 0.0, "breaks": 0.0, "priced": 0.0}, own_volatility=0.0,
                 volatility=0.0)
    first = {}
    for _ in range(6000):
        m.step()
        for deal in m.fair.deals:
            if deal.announced not in first:  # at the first event after the announcement
                first[deal.announced] = abs(m.books[deal.target].mid_price - m.fair.fairs[deal.target])
    assert len(first) > 10 and max(first.values()) < 15.0


def test_with_an_index_the_index_book_shifts_when_a_target_it_holds_jumps_by_its_weight_times_the_jump():
    for weights, scale in ((True, 4.0), ([0.25] * 4, 1.0)):
        m = a_market(deals={"rate": 0.3, "length": 6.0, "cool": 0.0, "breaks": 0.5}, index=weights)
        shifts, advances = instrument(m)
        m.run(150.0)
        gapped = [entry for entry_list in advances for entry in entry_list]
        assert [(b, round(j)) for b, j in gapped] == [(s[0], s[1]) for s in shifts]
        index = [s for s in shifts if s[0] == 4]
        books = [s for s in shifts if s[0] < 4]
        assert len(index) > 15 and len(books) >= len(index)
        for _book, ticks, was, now in index:
            assert now[0] - was[0] == ticks and now[1] == was[1]
        # the whole jump with the sum, and a quarter of it with weights of 1/4; the book shifted by a little more
        # or less than it, from rounding twice
        assert sum(abs(s[1]) for s in index) == pytest.approx(scale / 4.0 * sum(abs(s[1]) for s in books), rel=0.3)


# ---- A jump is not free money ----

@dataclasses.dataclass
class Chaser(BookTrader):
    """Takes ``size`` lots of the target of each deal at its first decision after the news (an announcement, or a
    break), long after an announcement and short after a break, and undoes it ``hold`` units later. ``rounds`` are what
    each lot earned before costs, the target's mid then against the mid it took it at."""
    news: str = "announced"
    hold: float = 5.0
    size: int = 5
    rounds: list = dataclasses.field(default_factory=list)

    def attach(self, n):
        super().attach(n)
        self._seen, self._open, self._now, self._new = set(), [], 0.0, []

    def observe(self, market):
        self._now = market.time
        broken = [d for d in market.fair.ended if d.outcome == "broken"]
        pool = market.fair.deals if self.news == "announced" else broken
        self._new = [d for d in pool if id(d) not in self._seen]
        self._seen.update(id(d) for d in self._new)

    def decide(self, mids):
        sign = 1 if self.news == "announced" else -1
        move = np.zeros(len(mids), dtype=int)
        for deal in self._new:
            move[deal.target] += sign * self.size
            self._open.append((deal, mids[deal.target], self._now + self.hold))
        for item in [i for i in self._open if i[2] <= self._now]:
            self._open.remove(item)
            deal, entry, _ = item
            move[deal.target] -= sign * self.size
            self.rounds.append(sign * (mids[deal.target] - entry))
        return [move] if move.any() else []


@functools.lru_cache(maxsize=None)
def chased(shifted):
    """What a lot earned before costs for a trader that takes the news and one that takes the breaks, in markets of four
    books on seeds 1 to 8, 1,000 units each (301 announcements, and 86 breaks): with the books' orders following the
    jumps, or (``shifted`` False) left where they were."""
    settings = {"rate": 0.3, "premium": 20.0, "length": 40.0, "cool": 10.0, "breaks": 0.3, "priced": 0.3}
    rounds = {"announced": [], "broken": []}
    with mock.patch.object(BookMarket, "_shift", BookMarket._shift if shifted else lambda self, book, ticks: None):
        for seed in SEEDS:
            announced, broken = Chaser(news="announced", interval=0.25), Chaser(news="broken", interval=0.25)
            m = a_market(seed=seed, deals=settings)
            m.add(announced)
            m.add(broken)
            while m.time < 1000.0:
                m.step()
            rounds["announced"] += announced.rounds
            rounds["broken"] += broken.rounds
    return rounds


def test_a_trader_that_buys_the_target_at_its_first_decision_after_each_announcement_and_sells_5_units_later_earns_0():
    rounds = np.array(chased(True)["announced"])
    assert len(rounds) > 250  # announcements, not all of which ... a book is in one deal at a time
    mean, error = mean_and_error(rounds)
    assert abs(mean) < 3.5 * error and error < 0.5  # nothing from the jump, which is 14 ticks a lot (0.7 of 20)


def test_a_trader_that_sells_the_target_at_its_first_decision_after_each_break_and_buys_it_back_5_units_later_earns_0():
    rounds = np.array(chased(True)["broken"])
    assert len(rounds) > 75
    mean, error = mean_and_error(rounds)
    assert abs(mean) < 3.5 * error and error < 0.5  # nothing from the jump, which is 14 ticks a lot or so


def test_with_the_books_orders_left_where_they_were_the_same_traders_would_earn_the_jump_a_good_part_of_it():
    for news in ("announced", "broken"):
        mean, error = mean_and_error(chased(False)[news])
        assert mean > 3.0 and mean > 8 * error  # whoever trades first takes what the stale quotes give
        stays, _ = mean_and_error(chased(True)[news])
        assert abs(stays) < mean / 5
