"""Strategy — funding carry signals and combined portfolio allocation."""

from funding_the_fall.strategy.carry import (
    compute_funding_spreads,
    simulate_carry,
    grid_search_params,
    grid_search_per_coin,
    grid_search_all_pairs,
    select_best_params,
    PooledCarryParams,
    PooledGridResult,
)
from funding_the_fall.strategy.allocation import allocate_positions, PositionTarget

__all__ = [
    "compute_funding_spreads",
    "simulate_carry",
    "grid_search_params",
    "grid_search_per_coin",
    "grid_search_all_pairs",
    "select_best_params",
    "allocate_positions",
    "PositionTarget",
    "PooledCarryParams",
    "PooledGridResult",
]
