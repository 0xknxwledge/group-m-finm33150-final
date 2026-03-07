"""Jump-weighted risk — combines jump tail probabilities with cascade amplification.

Owner: John Beecher

Per-coin expected cascade-amplified loss:
  E[amplified loss] = ∫ f(-δ) · δ · A(δ) dδ

where f is the calibrated Merton density and A(δ) is the cascade amplification.
Integrates "how likely is a shock of size δ?" with "how bad does the cascade get?"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import polars as pl

from funding_the_fall.models.merton import MertonParams, merton_log_density
from funding_the_fall.models.cascade import (
    build_positions_from_oi,
    simulate_cascade,
)


def jump_weighted_risk(
    merton_params: MertonParams,
    positions: list,
    dt: float = 1.0,
    orderbook_depth_usd: float | None = None,
    n_shocks: int = 100,
) -> dict:
    """Combine calibrated jump tail probabilities with cascade amplification.

    Numerically integrates over δ ∈ [0.5%, 30%]:
      baseline_loss  = ∫ f(-δ) · δ dδ
      amplified_loss = ∫ f(-δ) · δ · A(δ) dδ

    Returns dict with baseline_loss, amplified_loss, cascade_excess,
    cascade_multiplier, tail_probability_5pct, amplification_at_5pct.
    """
    delta_grid = np.linspace(0.005, 0.30, n_shocks)

    # Left-tail density: f(-δ) for each shock size
    log_density = merton_log_density(-delta_grid, merton_params, dt)
    density = np.exp(log_density)

    # Cascade amplification A(δ) at each grid point
    amplifications = np.ones(n_shocks)
    for i, delta in enumerate(delta_grid):
        result = simulate_cascade(
            positions,
            current_price=1.0,
            initial_shock_pct=float(delta),
            orderbook_depth_usd=orderbook_depth_usd,
        )
        amplifications[i] = result.amplification

    # Trapezoidal integration
    baseline_loss = float(np.trapz(density * delta_grid, delta_grid))
    amplified_loss = float(np.trapz(density * delta_grid * amplifications, delta_grid))
    cascade_excess = amplified_loss - baseline_loss
    cascade_multiplier = amplified_loss / baseline_loss if baseline_loss > 0 else 1.0

    # P(return ≤ -5%) ≈ ∫_{0.05}^{0.30} f(-δ) dδ
    mask = delta_grid >= 0.05
    tail_prob = float(np.trapz(density[mask], delta_grid[mask]))

    # A(5%)
    idx = int(np.argmin(np.abs(delta_grid - 0.05)))

    return {
        "baseline_loss": baseline_loss,
        "amplified_loss": amplified_loss,
        "cascade_excess": cascade_excess,
        "cascade_multiplier": cascade_multiplier,
        "tail_probability_5pct": tail_prob,
        "amplification_at_5pct": float(amplifications[idx]),
    }


def jump_weighted_risk_all_coins(
    merton_params_dict: dict[str, MertonParams],
    oi_df: pl.DataFrame,
    dt: float = 1.0,
    leverage: float = 5.0,
    orderbook_depth_usd: float | None = None,
) -> dict[str, dict]:
    """Compute jump-weighted risk for all coins.

    Suitable for weighting cascade shorts in allocate_positions().
    """
    results = {}
    for coin, params in merton_params_dict.items():
        coin_oi = oi_df.filter(pl.col("coin") == coin)
        positions = build_positions_from_oi(coin_oi, leverage=leverage)
        if not positions:
            continue
        results[coin] = jump_weighted_risk(
            params, positions, dt=dt,
            orderbook_depth_usd=orderbook_depth_usd,
        )
    return results
