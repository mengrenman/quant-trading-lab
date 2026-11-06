"""Bivariate copulas (lob/copulas.py): each formula checked against another way of getting the same number."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from lob.copulas import (Clayton, Frank, Gaussian, Gumbel, Independence, Margin, Rotated, StudentT, clayton_tau,
                         fit_copula, fit_family, frank_tau, gumbel_tau, pseudo_observations)

ALL = [
    Gaussian(0.5), Gaussian(-0.4), StudentT(0.5, 4.0), StudentT(0.2, 8.0), Clayton(2.0), Gumbel(2.0), Frank(5.0),
    Frank(-3.0), Rotated(Clayton(2.0)), Rotated(Gumbel(1.5)), Independence(),
]
GRID = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.3, 0.9), (0.75, 0.8), (0.02, 0.05), (0.97, 0.95)]


@pytest.mark.parametrize("copula", ALL, ids=lambda c: c.describe())
def test_h_is_the_derivative_of_the_cdf(copula):
    d = 2e-2 if copula.family == "student" else 1e-3  # scipy's t CDF is by simulation: too noisy for a small step
    for u, v in GRID:
        v = min(max(v, 2 * d), 1 - 2 * d)
        numeric = (float(np.ravel(copula.cdf(u, v + d))[0]) - float(np.ravel(copula.cdf(u, v - d))[0])) / (2 * d)
        assert float(copula.h(u, v)) == pytest.approx(numeric, abs=3e-2 if copula.family == "student" else 2e-3)


@pytest.mark.parametrize("copula", ALL, ids=lambda c: c.describe())
def test_the_density_is_the_derivative_of_h_and_integrates_to_one(copula):
    d = 1e-4
    for u, v in GRID:
        numeric = (float(copula.h(u + d, v)) - float(copula.h(u - d, v))) / (2 * d)
        assert math.exp(float(copula.logpdf(u, v))) == pytest.approx(numeric, rel=1e-3, abs=1e-5)
    total, _ = integrate.dblquad(lambda a, b: math.exp(float(copula.logpdf(a, b))), 1e-4, 1 - 1e-4, 1e-4, 1 - 1e-4,
                                 epsabs=1e-3, epsrel=1e-3)
    assert total == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("copula", ALL, ids=lambda c: c.describe())
def test_h_runs_from_zero_to_one_and_rises(copula):
    for v in (0.05, 0.5, 0.95):
        values = copula.h(np.linspace(1e-9, 1 - 1e-9, 201), v)
        assert np.all(np.diff(values) >= -1e-9)
        assert values[0] < 1e-3 and values[-1] > 1 - 1e-3


@pytest.mark.parametrize("copula", ALL[:-1], ids=lambda c: c.describe())
def test_the_conditional_is_uniform_under_the_right_copula(copula):
    """h(U | V) is the Rosenblatt transform: uniform, and independent of V. So a cumulated h - 0.5 is a random walk."""
    u, v = copula.sample(40_000, np.random.default_rng(1))
    w = copula.h(u, v)
    assert stats.kstest(w, "uniform").statistic < 0.01
    assert abs(np.corrcoef(w, v)[0, 1]) < 0.02
    assert stats.kstest(u, "uniform").statistic < 0.01  # sampling gives uniform margins, too


@pytest.mark.parametrize("copula, tau", [
    (Clayton(2.0), clayton_tau(2.0)), (Gumbel(2.0), gumbel_tau(2.0)), (Frank(5.0), frank_tau(5.0)),
    (Frank(-3.0), frank_tau(-3.0)), (Gaussian(0.5), 2 / math.pi * math.asin(0.5)),
    (StudentT(0.5, 4.0), 2 / math.pi * math.asin(0.5)), (Rotated(Clayton(2.0)), clayton_tau(2.0)),
], ids=lambda x: x.describe() if hasattr(x, "describe") else f"{x:.3f}")
def test_kendalls_tau_matches_the_sample(copula, tau):
    u, v = copula.sample(8_000, np.random.default_rng(2))
    assert stats.kendalltau(u, v).statistic == pytest.approx(tau, abs=0.02)


@pytest.mark.parametrize("copula, lower, upper", [
    (Clayton(2.0), 2 ** -0.5, 0.0), (Gumbel(2.0), 0.0, 2 - math.sqrt(2)), (Rotated(Clayton(2.0)), 0.0, 2 ** -0.5),
    (Rotated(Gumbel(2.0)), 2 - math.sqrt(2), 0.0),
], ids=lambda x: x.describe() if hasattr(x, "describe") else f"{x:.3f}")
def test_tail_dependence_is_the_limit_in_the_corners(copula, lower, upper):
    assert copula.tail_dependence() == pytest.approx((lower, upper))
    q = 1e-7  # the limits themselves, from the CDF; a Gumbel's lower tail, for one, is still 0.11 at q = 0.005
    assert float(copula.cdf(q, q)) / q == pytest.approx(lower, abs=0.02)
    q = 1e-4
    assert (2 * q - 1 + float(copula.cdf(1 - q, 1 - q))) / q == pytest.approx(upper, abs=0.03)


def test_a_sample_shows_the_tails_at_a_finite_level():
    q = 0.02
    for copula in (Clayton(2.0), Rotated(Gumbel(2.0)), Gumbel(2.0)):
        u, v = copula.sample(200_000, np.random.default_rng(3))
        assert np.mean(v[u < q] < q) == pytest.approx(float(copula.cdf(q, q)) / q, abs=0.02)
    assert Gaussian(0.5).tail_dependence() == Frank(5.0).tail_dependence() == (0.0, 0.0)


def test_the_gaussian_and_t_densities_match_their_joint_distributions():
    """Sklar: the copula's density is the joint density over the product of the margins'."""
    for u, v in GRID:
        rho, nu = 0.5, 4.0
        x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        joint = stats.multivariate_t.pdf([x, y], shape=[[1, rho], [rho, 1]], df=nu)
        assert float(StudentT(rho, nu).logpdf(u, v)) == pytest.approx(
            math.log(joint) - stats.t.logpdf(x, nu) - stats.t.logpdf(y, nu), abs=1e-9)
        x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        joint = stats.multivariate_normal.pdf([x, y], cov=[[1, rho], [rho, 1]])
        assert float(Gaussian(rho).logpdf(u, v)) == pytest.approx(
            math.log(joint) - stats.norm.logpdf(x) - stats.norm.logpdf(y), abs=1e-9)


def test_the_student_t_has_tail_dependence_the_gaussian_has_not():
    lower, upper = StudentT(0.5, 4.0).tail_dependence()
    # McNeil, Frey and Embrechts (2015), table 7.1: 0.25 for nu = 4 and rho = 0.5.
    assert lower == upper == pytest.approx(0.25, abs=0.005)


@pytest.mark.parametrize("truth, family, parameter, value", [
    (Gaussian(0.6), "gaussian", "rho", 0.6), (Clayton(2.5), "clayton", "theta", 2.5),
    (Gumbel(1.8), "gumbel", "theta", 1.8),
    (Frank(6.0), "frank", "theta", 6.0), (Frank(-4.0), "frank", "theta", -4.0),
], ids=lambda x: x.describe() if hasattr(x, "describe") else str(x))
def test_each_family_recovers_its_parameter(truth, family, parameter, value):
    u, v = truth.sample(4_000, np.random.default_rng(4))
    fitted, _ = fit_family(family, pseudo_observations(u), pseudo_observations(v))
    assert getattr(fitted, parameter) == pytest.approx(value, rel=0.08)


@pytest.mark.parametrize("truth, wanted", [
    (Gaussian(0.5), {"Gaussian", "Student t"}),  # nearly the same with a large nu
    (StudentT(0.5, 3.0), {"Student t"}),
    (Clayton(2.0), {"Clayton"}),
    (Gumbel(2.0), {"Gumbel"}),
    (Rotated(Clayton(2.0)), {"Rotated Clayton"}),
    (Rotated(Gumbel(2.0)), {"Rotated Gumbel"}),
    (Frank(6.0), {"Frank"}),
    (Independence(), {"Independence", "Frank", "Gaussian"}),  # nothing to tell them apart
], ids=lambda x: x.describe() if hasattr(x, "describe") else "")
def test_aic_picks_the_family_that_made_the_data(truth, wanted):
    u, v = truth.sample(3_000, np.random.default_rng(5))
    fit = fit_copula(pseudo_observations(u), pseudo_observations(v))
    assert fit.copula.name in wanted, [row[:3] for row in fit.table]
    assert fit.table[0][2] == min(row[2] for row in fit.table) and fit.n == 3_000


def test_negative_dependence_leaves_out_the_families_that_cannot_describe_it():
    u, v = Gaussian(-0.5).sample(1_000, np.random.default_rng(6))
    fit = fit_copula(pseudo_observations(u), pseudo_observations(v))
    assert fit.tau < 0 and fit.copula.family in ("gaussian", "student", "frank")
    assert not {"Clayton", "Gumbel", "Rotated Clayton", "Rotated Gumbel"} & {row[0] for row in fit.table}


def test_ranks_stay_inside_the_unit_interval_and_ties_share_their_mean():
    u = pseudo_observations([3.0, 1.0, 2.0, 2.0])
    assert u.tolist() == [4 / 5, 1 / 5, 2.5 / 5, 2.5 / 5]
    m = Margin([3.0, 1.0, 2.0, 2.0])
    assert [m(x) for x in (3.0, 1.0, 2.0)] == [4 / 5, 1 / 5, 2.5 / 5]  # as the ranks in the sample
    assert m(0.0) == 0.5 / 5 and m(9.0) == 4.5 / 5 and m(1.5) == 1.5 / 5  # new values, between their neighbors
    sample = np.random.default_rng(7).integers(-6, 7, 500) / 2.0  # half ticks: many ties
    assert np.mean([Margin(sample)(x) for x in sample]) == pytest.approx(0.5)


@pytest.mark.parametrize("make, message", [
    (lambda: Gaussian(1.0), "rho"), (lambda: StudentT(0.5, 0.0), "nu"), (lambda: Clayton(0.0), "theta"),
    (lambda: Gumbel(0.5), "theta"), (lambda: Frank(0.0), "theta"),
    (lambda: fit_family("nope", [0.5], [0.5]), "unknown"),
    (lambda: fit_copula([0.5, float("nan")], [0.5, 0.5]), "finite"),
])
def test_impossible_parameters_are_refused(make, message):
    with pytest.raises(ValueError, match=message):
        make()
