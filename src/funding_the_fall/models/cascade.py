"""Liquidation cascade simulator.

Owner: Antonio Braz

Models the feedback loop between forced liquidations and price impact
across perpetual futures venues.

Cascade dynamics:
  price drop → perp liquidations → forced selling →
  orderbook absorption → further price impact → repeat

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

# Fallback depth when live orderbook data is unavailable.
DEFAULT_DEPTH_USD = 5_000_000.0

# Max leverage per (venue, coin) from exchange APIs (March 2026).
# Used to build per-venue leverage tiers in build_positions_tiered.
MAX_LEVERAGE: dict[tuple[str, str], int] = {
    # Hyperliquid
    ("hyperliquid", "BTC"): 40,
    ("hyperliquid", "ETH"): 25,
    ("hyperliquid", "SOL"): 20,
    ("hyperliquid", "HYPE"): 10,
    ("hyperliquid", "DOGE"): 10,
    # Lighter
    ("lighter", "BTC"): 50,
    ("lighter", "ETH"): 50,
    ("lighter", "SOL"): 25,
    ("lighter", "HYPE"): 20,
    ("lighter", "DOGE"): 10,
    # OKX
    ("okx", "BTC"): 100,
    ("okx", "ETH"): 100,
    ("okx", "SOL"): 50,
    ("okx", "DOGE"): 50,
    # Kraken
    ("kraken", "BTC"): 50,
    ("kraken", "ETH"): 50,
    ("kraken", "SOL"): 50,
    ("kraken", "HYPE"): 50,
    ("kraken", "DOGE"): 50,
    # Binance
    ("binance", "BTC"): 125,
    ("binance", "ETH"): 100,
    ("binance", "SOL"): 75,
    ("binance", "HYPE"): 75,
    ("binance", "DOGE"): 75,
    # Bybit
    ("bybit", "BTC"): 100,
    ("bybit", "ETH"): 100,
    ("bybit", "SOL"): 100,
    ("bybit", "HYPE"): 75,
    ("bybit", "DOGE"): 75,
    # dYdX
    ("dydx", "BTC"): 50,
    ("dydx", "ETH"): 50,
    ("dydx", "SOL"): 20,
    ("dydx", "HYPE"): 5,
    ("dydx", "DOGE"): 10,
}


def _venue_tiers(venue: str, coin: str) -> list[tuple[float, float]]:
    """Generate leverage tiers for a (venue, coin) pair.

    Tiers: 75% at low leverage (3x), 20% at moderate, 5% at high.
    The moderate tier is max/3 (floored at 5x). The high tier is capped
    at 20x — the vast majority of perp volume is below 20x leverage even
    on venues that offer higher.
    """
    max_lev = MAX_LEVERAGE.get((venue, coin), 10)
    mid_lev = max(5.0, max_lev / 3)
    high_lev = min(20.0, float(max_lev))
    return [(3.0, 0.75), (mid_lev, 0.20), (high_lev, 0.05)]


@dataclass
class Position:
    """A leveraged position that can be liquidated."""

    collateral_usd: float  # margin posted
    debt_usd: float  # notional minus margin
    liquidation_threshold: float  # maintenance margin rate (fraction of notional)
    layer: str  # "perp", "hyperlend", or "morpho"


@dataclass
class CascadeResult:
    """Result of a single cascade simulation."""

    initial_shock: float
    effective_shock: float
    amplification: float  # effective / initial
    rounds: int
    total_notional_liquidated: float
    liquidations_by_layer: dict[str, float] = field(default_factory=dict)


def _is_liquidated(pos: Position, price_drop_pct: float) -> bool:
    """Check whether a perp position is liquidated after a given price drop.

    For a long perp: margin = collateral, notional = collateral + debt.
    After a drop δ, remaining equity = margin - notional * δ.
    Liquidation triggers when equity falls below maintenance margin:

        margin - notional * δ < maintenance_rate * notional
    """
    notional = pos.collateral_usd + pos.debt_usd
    if notional <= 0:
        return False
    remaining_equity = pos.collateral_usd - notional * price_drop_pct
    maintenance_margin = pos.liquidation_threshold * notional
    return remaining_equity < maintenance_margin


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
            notional = pos.collateral_usd + pos.debt_usd
            total_liquidated += notional
            by_layer[pos.layer] = by_layer.get(pos.layer, 0.0) + notional
            forced_volume += notional * (1.0 - cumulative_drop)

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
        total_notional_liquidated=total_liquidated,
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


# Reference OI-to-depth ratio for risk score mapping.
# Calibrated so OI/depth=200 → risk_score≈0.63 (median crypto market).
RISK_REFERENCE_RATIO = 200.0


def cascade_risk_signal(
    positions: list[Position],
    current_price: float,
    orderbook_depth_usd: float | None = None,
    threshold_amplification: float = 1.5,
) -> dict:
    """Compute the current cascade risk level.

    The risk_score is based on the OI-to-depth ratio rather than the cascade
    critical shock. Cascade amplification in crypto is binary (below critical
    shock: A=1, above: catastrophic) due to extreme OI/depth ratios. The
    critical shock is a structural constant determined by max leverage, not a
    market variable. The OI/depth ratio, by contrast, genuinely varies over
    time as open interest grows and shrinks, making it a better input for
    dynamic allocation scaling.

    Returns a dict with:
      - risk_score: float in [0, 1], based on 1 - exp(-oi_depth_ratio / ref)
      - critical_shock: smallest δ where A(δ) > threshold (structural)
      - amplification_at_5pct: A(0.05) — amplification from a 5% shock
      - signal: bool — True if cascade risk is elevated
      - oi_depth_ratio: total OI / orderbook depth
    """
    depth = orderbook_depth_usd or DEFAULT_DEPTH_USD

    probe_shocks = np.arange(0.005, 0.505, 0.005)
    results = compute_amplification_curve(
        positions,
        current_price,
        probe_shocks,
        depth,
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

    # Risk score: based on OI-to-depth ratio (time-varying)
    total_oi = sum(pos.collateral_usd + pos.debt_usd for pos in positions)
    oi_depth_ratio = total_oi / depth if depth > 0 else 0.0
    risk_score = 1.0 - float(np.exp(-oi_depth_ratio / RISK_REFERENCE_RATIO))

    return {
        "risk_score": risk_score,
        "critical_shock": critical_shock if critical_shock != float("inf") else None,
        "amplification_at_5pct": amp_5pct,
        "signal": bool(risk_score > 0.5),
        "oi_depth_ratio": oi_depth_ratio,
    }


def build_positions_from_oi(
    oi_df: pl.DataFrame,
    leverage: float = 5.0,
    liq_threshold: float = 0.005,
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


def build_positions_tiered(
    oi_df: pl.DataFrame,
    tiers: list[tuple[float, float]] | None = None,
    liq_threshold: float = 0.005,
    layer: str = "perp",
) -> list[Position]:
    """Build positions with heterogeneous leverage tiers per (venue, coin).

    When tiers is None (default), each (venue, coin) pair uses venue-specific
    tiers derived from MAX_LEVERAGE: 50% at 3x, 30% at max/2, 20% at max.
    This reflects that a Binance BTC position can reach 125x while a dYdX
    HYPE position maxes at 5x.

    If tiers is provided explicitly, that single tier schedule is applied
    uniformly to all positions.
    """
    latest = oi_df.sort("timestamp").group_by(["venue", "coin"]).last()
    positions: list[Position] = []
    for row in latest.iter_rows(named=True):
        notional = row.get("oi_usd", 0)
        if notional is None or notional <= 0:
            continue
        venue = row.get("venue", "")
        coin = row.get("coin", "")
        row_tiers = tiers or _venue_tiers(venue, coin)
        for leverage, weight in row_tiers:
            tier_notional = notional * weight
            collateral = tier_notional / leverage
            positions.append(
                Position(
                    collateral_usd=collateral,
                    debt_usd=tier_notional - collateral,
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


def depth_by_coin(depth_df: pl.DataFrame) -> dict[str, float]:
    """Aggregate per-coin bid depth (USD) across all venues.

    Input schema: [coin, venue, bid_depth_usd, ...].
    Returns dict mapping coin → total bid-side depth within 1% of mid.
    """
    if depth_df.is_empty():
        return {}
    agg = depth_df.group_by("coin").agg(pl.col("bid_depth_usd").sum())
    return {row["coin"]: row["bid_depth_usd"] for row in agg.iter_rows(named=True)}


def per_coin_risk_signals(
    oi_df: pl.DataFrame,
    leverage: float = 5.0,
    orderbook_depth_usd: float | None = None,
    depth_per_coin: dict[str, float] | None = None,
    threshold_amplification: float = 1.5,
    tiered: bool = True,
) -> dict[str, dict]:
    """Compute cascade risk signal for each coin individually.

    If depth_per_coin is provided (from fetch_orderbook_depth_all),
    each coin uses its measured depth. Otherwise falls back to
    orderbook_depth_usd or the module default.

    If tiered=True (default), uses per-venue leverage tiers from MAX_LEVERAGE
    for realistic position modeling.

    Returns dict keyed by coin → cascade_risk_signal() output.
    """
    depth_per_coin = depth_per_coin or {}
    coins = oi_df["coin"].unique().sort().to_list()
    signals: dict[str, dict] = {}
    for coin in coins:
        coin_oi = oi_df.filter(pl.col("coin") == coin)
        if tiered:
            positions = build_positions_tiered(coin_oi)
        else:
            positions = build_positions_from_oi(coin_oi, leverage=leverage)
        if not positions:
            continue
        coin_depth = depth_per_coin.get(coin, orderbook_depth_usd)
        signals[coin] = cascade_risk_signal(
            positions,
            current_price=1.0,
            orderbook_depth_usd=coin_depth,
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


def validate_cascade(
    candles: pl.DataFrame,
    liq_vol: pl.DataFrame,
    oi_df: pl.DataFrame,
    depth_per_coin: dict[str, float] | None = None,
    drawdown_threshold: float = 0.05,
    window_hours: int = 24,
) -> pl.DataFrame:
    """Compare predicted vs realized liquidation volume during drawdowns.

    Identifies drawdown events from hourly candle data, then for each event:
    - Predicted: run cascade simulator with the OI snapshot nearest the event
    - Realized: sum historical liquidation volume during the drawdown window

    Returns DataFrame: [timestamp, coin, drawdown_pct, predicted_liq_usd, realized_liq_usd].
    """
    depth_per_coin = depth_per_coin or {}
    rows: list[dict] = []

    for coin in candles["coin"].unique().sort().to_list():
        coin_candles = candles.filter(pl.col("coin") == coin).sort("timestamp")
        if coin_candles.is_empty():
            continue

        # Rolling max close over window_hours to find drawdowns
        closes = coin_candles["c"].to_numpy()
        timestamps = coin_candles["timestamp"].to_list()

        rolling_max = np.maximum.accumulate(closes)
        drawdowns = (rolling_max - closes) / np.where(rolling_max > 0, rolling_max, 1.0)

        # Find local drawdown peaks exceeding threshold
        events: list[tuple[int, float]] = []
        i = 0
        while i < len(drawdowns):
            if drawdowns[i] >= drawdown_threshold:
                # Find the trough of this drawdown episode
                j = i
                while j + 1 < len(drawdowns) and drawdowns[j + 1] >= drawdowns[j] * 0.8:
                    j += 1
                events.append((j, float(drawdowns[j])))
                i = j + window_hours  # skip ahead
            else:
                i += 1

        coin_oi = oi_df.filter(pl.col("coin") == coin)
        coin_liq = (
            liq_vol.filter(pl.col("coin") == coin)
            if "coin" in liq_vol.columns
            else liq_vol
        )

        for idx, dd_pct in events:
            ts = timestamps[idx]

            # Nearest OI snapshot
            oi_snap = coin_oi.filter(pl.col("timestamp") <= ts)
            if oi_snap.is_empty():
                continue
            positions = build_positions_tiered(oi_snap)
            if not positions:
                continue

            depth = depth_per_coin.get(coin, DEFAULT_DEPTH_USD)
            result = simulate_cascade(
                positions,
                current_price=1.0,
                initial_shock_pct=dd_pct,
                orderbook_depth_usd=depth,
            )
            predicted = result.total_notional_liquidated

            # Realized liquidation volume in the window around the event
            window_start = ts
            window_end = timestamps[min(idx + window_hours, len(timestamps) - 1)]
            realized_df = coin_liq.filter(
                (pl.col("timestamp") >= window_start)
                & (pl.col("timestamp") <= window_end)
            )
            realized = (
                realized_df["total_usd"].sum() if not realized_df.is_empty() else 0.0
            )

            rows.append(
                {
                    "timestamp": ts,
                    "coin": coin,
                    "drawdown_pct": dd_pct,
                    "predicted_liq_usd": predicted,
                    "realized_liq_usd": realized,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "coin": pl.Utf8,
                "drawdown_pct": pl.Float64,
                "predicted_liq_usd": pl.Float64,
                "realized_liq_usd": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort("timestamp")
