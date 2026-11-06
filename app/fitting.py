"""What a fit tries, on the Trade view (LOBManager.fit_strategy) and the Pairs view (PairsSession.fit_trader) alike."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Callable, Iterable, Optional

from app.model_specs import Param

MAX_FIT_VALUES = 8  # values tried for one setting
MAX_FIT_CANDIDATES = 48  # combinations tried in one fit: with more, a random sample of them
COUNTED_GRID = 20_000  # grids up to this many combinations are counted exactly (beyond, sampled without counting)
FIT_OBJECTIVES = {"pnl": "P&L", "sharpe": "P&L per unit of risk"}
FIT_SEED = Param("fit_seed", "Seed", 1, 0, 2**31 - 1, 1, integer=True)


class GridError(ValueError):
    """What is wrong with the values given for a setting (``name``; "" when it is about no one setting)."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name


def parse_grid(params: Iterable[Any], form: Any) -> dict[str, list[Any]]:
    """The values to try for each setting ticked in a fit panel (``fit_<name>``, with ``grid_<name>``).

    Raises GridError saying what is wrong.
    """
    grid: dict[str, list[Any]] = {}
    for param in params:
        if form.get(f"fit_{param.name}") is None:
            continue
        values: list[Any] = []
        for piece in str(form.get(f"grid_{param.name}", "")).replace(";", ",").split(","):
            if piece.strip():
                try:
                    value = param.parse(piece.strip())
                except ValueError as exc:
                    raise GridError(param.name, str(exc)) from None
                if value not in values:
                    values.append(value)
        if not values:
            raise GridError(param.name, f"{param.label}: give the values to try, separated by commas")
        if len(values) > MAX_FIT_VALUES:
            raise GridError(param.name, f"{param.label}: at most {MAX_FIT_VALUES} values")
        grid[param.name] = values
    if not grid:
        raise GridError("", "tick at least one setting to fit")
    return grid


def candidates(base: dict[str, Any], grid: dict[str, list[Any]], valid: Callable[[dict[str, Any]], bool],
               budget: Optional[int] = None, seed: int = 1) -> tuple[list[dict[str, Any]], Optional[int]]:
    """The combinations of ``grid``'s values to try, the other settings as in ``base``, and how many are valid.

    Combinations ``valid`` turns down (settings that contradict each other) are left out. With more
    valid ones than ``budget``, a random ``budget`` of them, drawn from ``seed`` (random search, which
    with many settings does about as well as the whole grid: Bergstra and Bengio 2012), in the grid's
    order. A grid of more than COUNTED_GRID combinations is sampled without counting its valid ones
    (the count is then None).
    """
    names = list(grid)
    combine = lambda combo: {**base, **dict(zip(names, combo, strict=True))}  # noqa: E731
    size = math.prod(len(grid[n]) for n in names)
    rng = random.Random(seed)
    if size <= COUNTED_GRID:
        good = [v for v in map(combine, itertools.product(*(grid[n] for n in names))) if valid(v)]
        if budget is None or len(good) <= budget:
            return good, len(good)
        return [good[i] for i in sorted(rng.sample(range(len(good)), budget))], len(good)
    budget = MAX_FIT_CANDIDATES if budget is None else budget
    picked: dict[tuple[int, ...], dict[str, Any]] = {}
    tried: set[tuple[int, ...]] = set()
    while len(picked) < budget and len(tried) < max(1_000, 50 * budget):
        index = tuple(rng.randrange(len(grid[n])) for n in names)
        if index in tried:
            continue
        tried.add(index)
        values = combine(grid[n][i] for n, i in zip(names, index, strict=True))
        if valid(values):
            picked[index] = values
    return [picked[i] for i in sorted(picked)], None
