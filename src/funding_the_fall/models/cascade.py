"""Liquidation cascade simulator.

Owner: John Beecher

Models the cross-layer liquidation feedback loop across:
  - Perpetual futures (Hyperliquid, $3.4B+ OI)
  - HyperLend (Aave V3 fork, ~$650M TVL)
  - Morpho Blue isolated lending (~$200M TVL)

Cascade dynamics:
  price drop → perp liquidations → forced spot selling →
  lending collateral devaluation → lending liquidations → repeat

The amplification curve A(δ) maps initial shocks to terminal effective
shocks. A(δ) > 1 indicates cascade amplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Position:
    """A leveraged position that can be liquidated."""

    collateral_usd: float
    debt_usd: float
    liquidation_threshold: float  # collateral/debt ratio triggering liquidation
    layer: str                    # "perp", "hyperlend", or "morpho"


@dataclass
class CascadeResult:
    """Result of a single cascade simulation."""

    initial_shock: float
    effective_shock: float
    amplification: float           # effective / initial
    rounds: int
    total_debt_liquidated: float
    liquidations_by_layer: dict[str, float]


def simulate_cascade(
    positions: list[Position],
    current_price: float,
    initial_shock_pct: float,
    orderbook_depth_usd: float | None = None,
    max_rounds: int = 50,
) -> CascadeResult:
    """Simulate a liquidation cascade from an initial exogenous price shock.

    Iterates until no new liquidations are triggered or max_rounds is reached.
    Price impact of forced selling is modeled using square-root market impact
    (consistent with Almgren-Chriss / Jusselin-Rosenbaum).
    """
    raise NotImplementedError


def compute_amplification_curve(
    positions: list[Position],
    current_price: float,
    shocks: NDArray[np.floating] | None = None,
    orderbook_depth_usd: float | None = None,
) -> list[CascadeResult]:
    """Run cascade simulation across a range of initial shocks.

    Default shocks: 1% to 50% in 0.5% increments.
    Returns list of CascadeResult for plotting the A(δ) curve.
    """
    if shocks is None:
        shocks = np.arange(0.01, 0.505, 0.005)
    return [
        simulate_cascade(positions, current_price, s, orderbook_depth_usd)
        for s in shocks
    ]


def cascade_risk_signal(
    positions: list[Position],
    current_price: float,
    orderbook_depth_usd: float | None = None,
    threshold_amplification: float = 1.5,
) -> dict:
    """Compute the current cascade risk level.

    Returns a dict with:
      - risk_score: float in [0, 1]
      - critical_shock: smallest δ where A(δ) > threshold
      - amplification_at_5pct: A(0.05) — amplification from a 5% shock
      - signal: bool — True if cascade risk is elevated
    """
    raise NotImplementedError
