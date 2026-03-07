"""Liquidation cascade simulator.

Owner: Antonio Braz

Models the cross-layer liquidation feedback loop across:
  - Perpetual futures (Hyperliquid, $3.4B+ OI)
  - HyperLend (Aave V3 fork, ~$650M TVL)
  - Morpho Blue isolated lending (~$200M TVL)

Cascade dynamics:
  price drop → perp liquidations → forced spot selling →
  lending collateral devaluation → lending liquidations → repeat

The amplification curve A(δ) maps initial shocks to terminal effective
shocks. A(δ) > 1 indicates cascade amplification.

Price impact model:
  Square-root law: Δp/p = sqrt(V / D)
  where V = forced selling volume, D = orderbook depth.
  Consistent with Almgren-Chriss (2001) and Jusselin-Rosenbaum (2018).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray

# Approximate 1% orderbook depth for major perps on Hyperliquid (USD).
DEFAULT_DEPTH_USD = 5_000_000.0


@dataclass
class Position:
    """A leveraged position that can be liquidated."""

    collateral_usd: float
    debt_usd: float
    liquidation_threshold: float  # collateral/debt ratio triggering liquidation
    layer: str  # "perp", "hyperlend", or "morpho"


@dataclass
class CascadeResult:
    """Result of a single cascade simulation."""

    initial_shock: float
    effective_shock: float
    amplification: float  # effective / initial
    rounds: int
    total_debt_liquidated: float
    liquidations_by_layer: dict[str, float] = field(default_factory=dict)


def _is_liquidated(pos: Position, price_drop_pct: float) -> bool:
    """Check whether a position is liquidated after a given price drop.

    After a drop of `price_drop_pct` (as a fraction, e.g. 0.10 for 10%),
    collateral value shrinks proportionally. Liquidation triggers when
    the health factor falls below 1:

        HF = (collateral * (1 - drop) * liq_threshold) / debt < 1
    """
    shocked_collateral = pos.collateral_usd * (1.0 - price_drop_pct)
    if pos.debt_usd <= 0:
        return False
    health_factor = (shocked_collateral * pos.liquidation_threshold) / pos.debt_usd
    return health_factor < 1.0


def _price_impact(volume_usd: float, depth_usd: float) -> float:
    """Square-root price impact: Δp/p = sqrt(V / D).

    Bounded at 1.0 (price can't go below zero).
    """
    if depth_usd <= 0 or volume_usd <= 0:
        return 0.0
    return min(np.sqrt(volume_usd / depth_usd), 1.0)


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
    depth = orderbook_depth_usd or DEFAULT_DEPTH_USD
    cumulative_drop = initial_shock_pct
    surviving = list(positions)
    total_liquidated = 0.0
    by_layer: dict[str, float] = {}

    rounds_used = 0
    for round_num in range(1, max_rounds + 1):
        newly_liquidated: list[Position] = []
        still_alive: list[Position] = []

        for pos in surviving:
            if _is_liquidated(pos, cumulative_drop):
                newly_liquidated.append(pos)
            else:
                still_alive.append(pos)

        if not newly_liquidated:
            break

        rounds_used = round_num
        forced_volume = 0.0
        for pos in newly_liquidated:
            total_liquidated += pos.debt_usd
            by_layer[pos.layer] = by_layer.get(pos.layer, 0.0) + pos.debt_usd
            forced_volume += pos.collateral_usd * (1.0 - cumulative_drop)

        additional_drop = _price_impact(forced_volume, depth)
        cumulative_drop = cumulative_drop + (1.0 - cumulative_drop) * additional_drop
        cumulative_drop = min(cumulative_drop, 1.0)
        surviving = still_alive

    amplification = (
        cumulative_drop / initial_shock_pct if initial_shock_pct > 0 else 1.0
    )

    return CascadeResult(
        initial_shock=initial_shock_pct,
        effective_shock=cumulative_drop,
        amplification=amplification,
        rounds=rounds_used,
        total_debt_liquidated=total_liquidated,
        liquidations_by_layer=by_layer,
    )


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
    probe_shocks = np.arange(0.005, 0.505, 0.005)
    results = compute_amplification_curve(
        positions,
        current_price,
        probe_shocks,
        orderbook_depth_usd,
    )

    # A(5%) — the headline amplification number
    amp_5pct = 1.0
    for r in results:
        if abs(r.initial_shock - 0.05) < 1e-6:
            amp_5pct = r.amplification
            break

    # Critical shock: smallest δ where A(δ) exceeds threshold
    critical_shock = float("inf")
    for r in results:
        if r.amplification >= threshold_amplification:
            critical_shock = r.initial_shock
            break

    # Risk score: 0–1 based on how small the critical shock is.
    # If a 5% shock already triggers threshold amplification → high risk.
    # If even a 50% shock doesn't → low risk.
    if critical_shock == float("inf"):
        risk_score = 0.0
    else:
        # Map critical_shock from [0.5, 0.005] → [0, 1]
        risk_score = max(0.0, min(1.0, 1.0 - (critical_shock - 0.005) / 0.495))

    return {
        "risk_score": risk_score,
        "critical_shock": critical_shock if critical_shock != float("inf") else None,
        "amplification_at_5pct": amp_5pct,
        "signal": bool(risk_score > 0.5),
    }


def build_positions_from_oi(
    oi_df: pl.DataFrame,
    leverage: float = 5.0,
    liq_threshold: float = 1.0,
    layer: str = "perp",
) -> list[Position]:
    """Build Position list from an open-interest DataFrame.

    Takes the latest snapshot per (venue, coin) and derives collateral/debt
    from the notional OI assuming uniform leverage.

    Expected schema: (timestamp, venue, coin, oi_usd).
    """
    latest = oi_df.sort("timestamp").group_by(["venue", "coin"]).last()
    positions: list[Position] = []
    for row in latest.iter_rows(named=True):
        notional = row.get("oi_usd", 0)
        if notional is None or notional <= 0:
            continue
        collateral = notional / leverage
        positions.append(
            Position(
                collateral_usd=collateral,
                debt_usd=notional - collateral,
                liquidation_threshold=liq_threshold,
                layer=layer,
            )
        )
    return positions


def sensitivity_to_leverage(
    oi_df: pl.DataFrame,
    leverages: list[float] | None = None,
    shocks: NDArray[np.floating] | None = None,
    **cascade_kwargs,
) -> dict[float, list[CascadeResult]]:
    """Amplification curves across leverage assumptions.

    Default leverages: [3, 5, 10, 20].
    """
    if leverages is None:
        leverages = [3.0, 5.0, 10.0, 20.0]
    return {
        lev: compute_amplification_curve(
            build_positions_from_oi(oi_df, leverage=lev),
            current_price=1.0,
            shocks=shocks,
            **cascade_kwargs,
        )
        for lev in leverages
    }


def per_coin_risk_signals(
    oi_df: pl.DataFrame,
    leverage: float = 5.0,
    orderbook_depth_usd: float | None = None,
    threshold_amplification: float = 1.5,
) -> dict[str, dict]:
    """Compute cascade risk signal for each coin individually.

    Returns dict keyed by coin → cascade_risk_signal() output.
    Coins with no OI data are omitted.
    """
    coins = oi_df["coin"].unique().sort().to_list()
    signals: dict[str, dict] = {}
    for coin in coins:
        coin_oi = oi_df.filter(pl.col("coin") == coin)
        positions = build_positions_from_oi(coin_oi, leverage=leverage)
        if not positions:
            continue
        signals[coin] = cascade_risk_signal(
            positions,
            current_price=1.0,
            orderbook_depth_usd=orderbook_depth_usd,
            threshold_amplification=threshold_amplification,
        )
    return signals


def sensitivity_to_depth(
    positions: list[Position],
    depths_usd: list[float] | None = None,
    shocks: NDArray[np.floating] | None = None,
    current_price: float = 1.0,
) -> dict[float, list[CascadeResult]]:
    """Amplification curves across orderbook depth assumptions.

    Default depths: [1M, 5M, 10M, 50M].
    """
    if depths_usd is None:
        depths_usd = [1e6, 5e6, 10e6, 50e6]
    return {
        d: compute_amplification_curve(
            positions,
            current_price,
            shocks=shocks,
            orderbook_depth_usd=d,
        )
        for d in depths_usd
    }
