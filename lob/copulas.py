"""Bivariate copulas: what pairs trading needs of them.

A copula is the dependence of two variables with their margins taken out: the joint distribution
``C(u, v)`` of ``U = F_1(X_1)`` and ``V = F_2(X_2)``, which are uniform. For pairs trading what matters are
the conditional probabilities, the partial derivatives of ``C``:

    h(u | v) = P(U <= u | V = v) = dC/dv

each close to 0 when the first variable is low for what the second is doing, and close to 1 when it
is high (Liew and Wu 2013). Under the right copula ``h(U | V)`` is uniform and independent of ``V``.

Here are the Gaussian, Student t, Clayton, Gumbel and Frank copulas, the rotations of Clayton and
Gumbel that move their dependent tail to the other corner (Rad, Low and Faff 2016 use all of these but the
Gaussian and Frank), and independence. They are fitted by canonical maximum likelihood (the margins
replaced by ranks, the copula's likelihood maximized from the value Kendall's tau implies) and
compared by AIC (``fit_copula``). All of them are exchangeable: ``C(u, v) = C(v, u)``.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import integrate, optimize, special, stats

EPS = 1e-10  # how close to 0 and 1 the uniforms may come


def _clip(x):
    return np.clip(x, EPS, 1 - EPS)


def _norm_ppf(u):
    return special.ndtri(u)


class Copula:
    """One copula with its parameters."""
    name = ""
    family = ""
    n_params = 1

    def h(self, u, v):
        """P(U <= u | V = v)."""
        raise NotImplementedError

    def cdf(self, u, v):
        """C(u, v)."""
        raise NotImplementedError

    def logpdf(self, u, v):
        """The log of the copula's density."""
        raise NotImplementedError

    def tail_dependence(self) -> tuple[float, float]:
        """The limits of P(V < q | U < q) as q falls to 0, and of P(V > q | U > q) as q rises to 1."""
        return 0.0, 0.0

    def describe(self) -> str:
        return self.name

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``n`` draws of (U, V): V uniform, and U from h(U | V), inverted by bisection."""
        v, w = rng.random(n), rng.random(n)
        lo, hi = np.full(n, EPS), np.full(n, 1 - EPS)
        for _ in range(60):
            mid = (lo + hi) / 2
            below = self.h(mid, v) < w
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return (lo + hi) / 2, v

    def conditionals(self, u: float, v: float) -> tuple[float, float]:
        """h(u | v) and h(v | u), for one pair."""
        return float(self.h(u, v)), float(self.h(v, u))


class Independence(Copula):
    name = "Independence"
    family = "independence"
    n_params = 0

    def h(self, u, v):
        return np.asarray(u, dtype=float) * np.ones_like(np.asarray(v, dtype=float))

    def cdf(self, u, v):
        return np.asarray(u, dtype=float) * np.asarray(v, dtype=float)

    def logpdf(self, u, v):
        return np.zeros(np.broadcast(np.asarray(u), np.asarray(v)).shape)


class Gaussian(Copula):
    name = "Gaussian"
    family = "gaussian"

    def __init__(self, rho: float):
        if not -1 < rho < 1:
            raise ValueError("rho must lie between -1 and 1")
        self.rho = rho

    def h(self, u, v):
        rho = self.rho
        return special.ndtr((_norm_ppf(_clip(u)) - rho * _norm_ppf(_clip(v))) / math.sqrt(1 - rho * rho))

    def cdf(self, u, v):
        x, y = np.atleast_1d(_norm_ppf(_clip(u))), np.atleast_1d(_norm_ppf(_clip(v)))
        points = np.column_stack(np.broadcast_arrays(x, y))
        return stats.multivariate_normal.cdf(points, cov=[[1, self.rho], [self.rho, 1]])

    def logpdf(self, u, v):
        x, y, rho = _norm_ppf(_clip(u)), _norm_ppf(_clip(v)), self.rho
        d = 1 - rho * rho
        return -0.5 * math.log(d) - (rho * rho * (x * x + y * y) - 2 * rho * x * y) / (2 * d)

    def describe(self) -> str:
        return f"Gaussian, rho {self.rho:.2f}"


class StudentT(Copula):
    name = "Student t"
    family = "student"
    n_params = 2

    def __init__(self, rho: float, nu: float):
        if not -1 < rho < 1 or not nu > 0:
            raise ValueError("rho must lie between -1 and 1, and nu above 0")
        self.rho, self.nu = rho, nu

    def h(self, u, v):
        rho, nu = self.rho, self.nu
        x, y = special.stdtrit(nu, _clip(u)), special.stdtrit(nu, _clip(v))
        return special.stdtr(nu + 1, (x - rho * y) / np.sqrt((nu + y * y) * (1 - rho * rho) / (nu + 1)))

    def cdf(self, u, v):
        x = np.atleast_1d(special.stdtrit(self.nu, _clip(u)))
        y = np.atleast_1d(special.stdtrit(self.nu, _clip(v)))
        points = np.column_stack(np.broadcast_arrays(x, y))
        return stats.multivariate_t.cdf(points, shape=[[1, self.rho], [self.rho, 1]], df=self.nu,
                                        maxpts=50_000, random_state=0)

    def logpdf(self, u, v):
        rho, nu = self.rho, self.nu
        x, y = special.stdtrit(nu, _clip(u)), special.stdtrit(nu, _clip(v))
        d = 1 - rho * rho
        joint = (special.gammaln((nu + 2) / 2) - special.gammaln(nu / 2) - math.log(nu * math.pi) - 0.5 * math.log(d)
                 - (nu + 2) / 2 * np.log1p((x * x - 2 * rho * x * y + y * y) / (nu * d)))
        one = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
        return joint - 2 * one + (nu + 1) / 2 * (np.log1p(x * x / nu) + np.log1p(y * y / nu))

    def tail_dependence(self) -> tuple[float, float]:
        tail = 2 * special.stdtr(self.nu + 1, -math.sqrt((self.nu + 1) * (1 - self.rho) / (1 + self.rho)))
        return float(tail), float(tail)

    def describe(self) -> str:
        return f"Student t, rho {self.rho:.2f}, nu {self.nu:g}"


class Clayton(Copula):
    """Dependence in the lower tail: both low together."""
    name = "Clayton"
    family = "clayton"

    def __init__(self, theta: float):
        if not theta > 0:
            raise ValueError("Clayton's theta must be above 0")
        self.theta = theta

    def _base(self, u, v):
        return np.power(_clip(u), -self.theta) + np.power(_clip(v), -self.theta) - 1

    def h(self, u, v):
        t = self.theta
        return np.power(_clip(v), -t - 1) * np.power(self._base(u, v), -1 / t - 1)

    def cdf(self, u, v):
        return np.power(self._base(u, v), -1 / self.theta)

    def logpdf(self, u, v):
        t = self.theta
        return (math.log1p(t) - (t + 1) * (np.log(_clip(u)) + np.log(_clip(v)))
                - (2 + 1 / t) * np.log(self._base(u, v)))

    def tail_dependence(self) -> tuple[float, float]:
        return 2 ** (-1 / self.theta), 0.0

    def describe(self) -> str:
        return f"Clayton, theta {self.theta:.2f}"


class Gumbel(Copula):
    """Dependence in the upper tail: both high together."""
    name = "Gumbel"
    family = "gumbel"

    def __init__(self, theta: float):
        if not theta >= 1:
            raise ValueError("Gumbel's theta must be 1 or more")
        self.theta = theta

    def _a(self, u, v):
        return np.power(-np.log(_clip(u)), self.theta) + np.power(-np.log(_clip(v)), self.theta)

    def cdf(self, u, v):
        return np.exp(-np.power(self._a(u, v), 1 / self.theta))

    def h(self, u, v):
        t = self.theta
        a = self._a(u, v)
        v = _clip(v)
        return (np.exp(-np.power(a, 1 / t)) * np.power(a, 1 / t - 1) * np.power(-np.log(v), t - 1) / v)

    def logpdf(self, u, v):
        t = self.theta
        u, v = _clip(u), _clip(v)
        lu, lv = -np.log(u), -np.log(v)
        a = np.power(lu, t) + np.power(lv, t)
        root = np.power(a, 1 / t)
        return (-root + lu + lv + (2 / t - 2) * np.log(a) + (t - 1) * (np.log(lu) + np.log(lv))
                + np.log1p((t - 1) / root))

    def tail_dependence(self) -> tuple[float, float]:
        return 0.0, 2 - 2 ** (1 / self.theta)

    def describe(self) -> str:
        return f"Gumbel, theta {self.theta:.2f}"


class Frank(Copula):
    """Symmetric, with no tail dependence."""
    name = "Frank"
    family = "frank"

    def __init__(self, theta: float):
        if theta == 0:
            raise ValueError("Frank's theta must not be 0")
        self.theta = theta

    def cdf(self, u, v):
        t = self.theta
        return -np.log1p(np.expm1(-t * _clip(u)) * np.expm1(-t * _clip(v)) / math.expm1(-t)) / t

    def h(self, u, v):
        t = self.theta
        a, b = np.expm1(-t * _clip(u)), np.expm1(-t * _clip(v))
        return (b + 1) * a / (math.expm1(-t) + a * b)

    def logpdf(self, u, v):
        t = self.theta
        u, v = _clip(u), _clip(v)
        a, b = np.expm1(-t * u), np.expm1(-t * v)
        denominator = math.expm1(-t) + a * b
        return np.log(abs(t) * abs(math.expm1(-t))) - t * (u + v) - 2 * np.log(np.abs(denominator))

    def describe(self) -> str:
        return f"Frank, theta {self.theta:.2f}"


class Rotated(Copula):
    """A copula turned by 180 degrees (its survival copula): (U, V) follows it when (1 - U, 1 - V) follows ``base``."""

    def __init__(self, base: Copula):
        self.base = base
        self.name = f"Rotated {base.name}"
        self.family = f"rotated_{base.family}"
        self.n_params = base.n_params

    def h(self, u, v):
        return 1 - self.base.h(1 - np.asarray(u, dtype=float), 1 - np.asarray(v, dtype=float))

    def cdf(self, u, v):
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        return u + v - 1 + self.base.cdf(1 - u, 1 - v)

    def logpdf(self, u, v):
        return self.base.logpdf(1 - np.asarray(u, dtype=float), 1 - np.asarray(v, dtype=float))

    def tail_dependence(self) -> tuple[float, float]:
        lower, upper = self.base.tail_dependence()
        return upper, lower

    def describe(self) -> str:
        return "Rotated " + self.base.describe()


def clayton_tau(theta: float) -> float:
    return theta / (theta + 2)


def gumbel_tau(theta: float) -> float:
    return 1 - 1 / theta


def frank_tau(theta: float) -> float:
    """Kendall's tau of a Frank copula: 1 - 4 (1 - D_1(theta)) / theta, with the Debye function D_1."""
    if abs(theta) < 1e-8:
        return 0.0
    a = abs(theta)
    t = np.linspace(0, a, 2001)[1:]
    # The Debye function by the trapezoid rule: its integrand tends to 1 at 0.
    values = np.concatenate([[1.0], t / np.expm1(t)])
    debye = integrate.trapezoid(values, dx=a / 2000) / a
    return math.copysign(1 - 4 * (1 - debye) / a, theta)


def _frank_theta(tau: float) -> float:
    if abs(tau) < 1e-6:
        return 1e-4
    if abs(tau) >= 0.95:
        return math.copysign(100, tau)
    return optimize.brentq(lambda t: frank_tau(t) - tau, -200.0, 200.0, xtol=1e-6)


# The degrees of freedom tried for a Student t copula (a grid: its likelihood is flat in nu, and each try is cheap).
NU_GRID = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
FAMILIES = ("gaussian", "student", "clayton", "rotated_clayton", "gumbel", "rotated_gumbel", "frank")


def _rho_of(tau: float) -> float:
    return float(np.clip(math.sin(math.pi * tau / 2), -0.999, 0.999))


def _bounded_fit(make: Callable[[float], Copula], u, v, lo: float, hi: float, start: float) -> tuple[Copula, float]:
    """The parameter between ``lo`` and ``hi`` that maximizes the likelihood, searching around ``start``."""
    def loss(p: float) -> float:
        value = -float(np.sum(make(p).logpdf(u, v)))
        return value if math.isfinite(value) else 1e12

    start = float(np.clip(start, lo, hi))
    left, right = max(lo, start - 0.5 * abs(start) - 0.1), min(hi, start + 0.5 * abs(start) + 0.1)
    best = optimize.minimize_scalar(loss, bounds=(left, right), method="bounded",
                                    options={"xatol": 1e-3, "maxiter": 40})
    return make(float(best.x)), -float(best.fun)


@dataclass(frozen=True)
class CopulaFit:
    """A fitted copula and how it compares: ``table`` has (name, log-likelihood, AIC, the copula), best first."""
    copula: Copula
    tau: float
    n: int
    table: tuple[tuple[str, float, float, Copula], ...]

    @property
    def aic(self) -> float:
        return self.table[0][2]


def fit_family(family: str, u, v, tau: Optional[float] = None) -> tuple[Copula, float]:
    """A family's parameters by canonical maximum likelihood: the copula and its log-likelihood."""
    if family not in FAMILIES:
        raise ValueError(f"unknown copula family {family!r}")
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if tau is None:
        tau = float(stats.kendalltau(u, v).statistic)
        tau = 0.0 if not math.isfinite(tau) else tau
    t = float(np.clip(tau, -0.95, 0.95))
    if family == "gaussian":
        return _bounded_fit(Gaussian, u, v, -0.999, 0.999, _rho_of(t))
    if family == "student":
        rho = _rho_of(t)
        best, best_ll = None, -math.inf
        for nu in NU_GRID:
            copula = StudentT(rho, nu)
            ll = float(np.sum(copula.logpdf(u, v)))
            if ll > best_ll:
                best, best_ll = copula, ll
        return _bounded_fit(lambda r: StudentT(r, best.nu), u, v, -0.999, 0.999, rho)
    if family in ("clayton", "rotated_clayton"):
        make = Clayton if family == "clayton" else (lambda p: Rotated(Clayton(p)))
        return _bounded_fit(make, u, v, 1e-3, 50.0, 2 * max(t, 1e-3) / (1 - max(t, 1e-3)))
    if family in ("gumbel", "rotated_gumbel"):
        make = Gumbel if family == "gumbel" else (lambda p: Rotated(Gumbel(p)))
        return _bounded_fit(make, u, v, 1.0, 50.0, 1 / (1 - max(t, 0.0)))
    start = _frank_theta(t)
    if start >= 0:
        return _bounded_fit(lambda p: Frank(max(p, 1e-4)), u, v, 1e-4, 200.0, start)
    return _bounded_fit(lambda p: Frank(min(p, -1e-4)), u, v, -200.0, -1e-4, start)


def fit_copula(u, v, families: Sequence[str] = FAMILIES) -> CopulaFit:
    """The family with the lowest AIC among ``families`` and independence, each by canonical maximum likelihood."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("the observations must be finite")
    tau = float(stats.kendalltau(u, v).statistic)
    tau = 0.0 if not math.isfinite(tau) else tau
    table = [("Independence", 0.0, 0.0, Independence())]
    for family in families:
        if tau <= 0 and family in ("clayton", "rotated_clayton", "gumbel", "rotated_gumbel"):
            continue  # these describe positive dependence only
        copula, ll = fit_family(family, u, v, tau)
        table.append((copula.name, ll, 2 * copula.n_params - 2 * ll, copula))
    table.sort(key=lambda row: row[2])
    return CopulaFit(table[0][3], tau, len(u), tuple(table))


def pseudo_observations(x) -> np.ndarray:
    """Ranks over n + 1 (ties share their mean rank), so they lie strictly between 0 and 1."""
    return stats.rankdata(x) / (len(x) + 1)


class Margin:
    """An empirical distribution function: a value's mean rank among a sample, over n + 1.

    Prices sit on half ticks, so returns tie: a tied value gets the middle of its ties, which keeps
    the transform centered (the sample's own values average 0.5 exactly).
    """

    def __init__(self, sample: Sequence[float]):
        self.values = sorted(float(x) for x in sample)
        self.n = len(self.values)

    def __call__(self, x: float) -> float:
        below = bisect.bisect_left(self.values, x)
        ties = bisect.bisect_right(self.values, x) - below
        return (below + (ties + 1) / 2) / (self.n + 1)
