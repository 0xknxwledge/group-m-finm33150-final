"""Portfolio allocation — combines carry and cascade signals.

Owner: Jean-Luc Choiseul

Allocation logic:
  - Carry baseline: 85% NAV, scaled down by cascade risk
  - Cascade max: 15% NAV, scaled up by cascade risk score
  - Remainder is cash (dry powder / risk buffer)

When cascade risk is elevated (risk_score -> 1):
  - Carry scales to ~60% NAV (reduces exposure to fragile markets)
  - Cascade scales to ~15% NAV (opportunistic short on negatively-skewed returns)
  - Cash rises to ~25% NAV

The cascade leg takes short positions on coins where amplification A(delta)
is highest. High amplification implies negatively skewed return distributions:
any shock is magnified by forced liquidation cascades, making shorts positive EV.
In crypto perps, shorts typically receive funding (funding rates are usually
positive), so the bleed from holding the position is minimal or negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PositionTarget:
    """A target position to be executed."""

    timestamp: pd.Timestamp
    coin: str
    venue: str
    side: str  # "long" or "short"
    notional_usd: float  # target notional exposure
    strategy: str  # "carry" or "cascade"


# ---------------------------------------------------------------------------
# Allocation constants
# ---------------------------------------------------------------------------
CARRY_BASE_WEIGHT = 0.85
CASCADE_MAX_WEIGHT = 0.15
CARRY_RISK_DAMPING = 0.30  # carry_scale = 1 - DAMPING * risk_score


def allocate_positions(
    carry_signals: list,
    cascade_signal: dict,
    nav: float,
    per_coin_signals: dict[str, dict] | None = None,
    deepest_venue: str = "binance",
    max_leverage: float = 5.0,
    max_single_exchange_pct: float = 0.40,
    max_net_delta_pct: float = 0.10,
) -> list[PositionTarget]:
    """Convert signals into sized position targets respecting risk limits.

    Parameters
    ----------
    carry_signals : list of CarrySignal
        Entry/exit signals from the carry strategy.
    cascade_signal : dict
        Output of cascade_risk_signal() (aggregate).
    nav : float
        Current portfolio NAV.
    per_coin_signals : dict[str, dict], optional
        Output of per_coin_risk_signals(). If provided, cascade shorts are
        weighted by per-coin amplification. Otherwise uniform across coins.
    deepest_venue : str
        Venue used for cascade shorts (deepest orderbook).
    max_leverage : float
        Max gross leverage as multiple of NAV.
    max_single_exchange_pct : float
        Max exposure to any single exchange as fraction of NAV.
    max_net_delta_pct : float
        Max net delta across all positions as fraction of NAV.

    Risk controls
    -------------
    - Total gross leverage <= max_leverage * NAV
    - Exposure to any single exchange <= max_single_exchange_pct * NAV
    - Net delta across all positions <= max_net_delta_pct * NAV
    """
    risk_score = cascade_signal.get("risk_score", 0.0)
    targets: list[PositionTarget] = []

    # --- Carry leg: scale down with risk ----------------------------------
    carry_scale = 1.0 - CARRY_RISK_DAMPING * risk_score
    carry_budget = CARRY_BASE_WEIGHT * nav * carry_scale

    # Convert carry signals into position targets.
    # Each carry signal is a delta-neutral pair (long + short on two venues),
    # so it contributes gross leverage but ~zero net delta.
    if carry_signals:
        # Distribute carry budget proportionally to spread attractiveness
        total_spread = sum(abs(s.spread) for s in carry_signals) or 1.0
        for sig in carry_signals:
            if sig.action != "enter":
                continue
            weight = abs(sig.spread) / total_spread
            leg_notional = carry_budget * weight
            targets.append(PositionTarget(
                timestamp=sig.timestamp,
                coin=sig.coin,
                venue=sig.long_venue,
                side="long",
                notional_usd=leg_notional,
                strategy="carry",
            ))
            targets.append(PositionTarget(
                timestamp=sig.timestamp,
                coin=sig.coin,
                venue=sig.short_venue,
                side="short",
                notional_usd=leg_notional,
                strategy="carry",
            ))

    # --- Cascade leg: opportunistic short, scaled by risk_score -----------
    cascade_budget = CASCADE_MAX_WEIGHT * nav * risk_score

    if cascade_budget > 0 and per_coin_signals:
        # Weight by per-coin A(5%); coins with higher amplification get more
        amps = {
            coin: sig.get("amplification_at_5pct", 1.0)
            for coin, sig in per_coin_signals.items()
        }
        # Only short coins with amplification > 1 (actual cascade risk)
        amps = {c: a for c, a in amps.items() if a > 1.0}
        total_amp = sum(amps.values()) or 1.0
        for coin, amp in amps.items():
            weight = amp / total_amp
            targets.append(PositionTarget(
                timestamp=carry_signals[0].timestamp if carry_signals else pd.Timestamp.now(),
                coin=coin,
                venue=deepest_venue,
                side="short",
                notional_usd=cascade_budget * weight,
                strategy="cascade",
            ))

    # --- Risk limit enforcement -------------------------------------------
    targets = _enforce_risk_limits(
        targets, nav, max_leverage, max_single_exchange_pct, max_net_delta_pct
    )
    return targets


def _enforce_risk_limits(
    targets: list[PositionTarget],
    nav: float,
    max_leverage: float,
    max_single_exchange_pct: float,
    max_net_delta_pct: float,
) -> list[PositionTarget]:
    """Scale down position targets to satisfy risk constraints."""
    if not targets or nav <= 0:
        return targets

    # 1. Gross leverage cap
    gross = sum(t.notional_usd for t in targets)
    max_gross = max_leverage * nav
    if gross > max_gross:
        scale = max_gross / gross
        targets = [
            PositionTarget(t.timestamp, t.coin, t.venue, t.side,
                           t.notional_usd * scale, t.strategy)
            for t in targets
        ]

    # 2. Single-exchange concentration cap
    max_venue = max_single_exchange_pct * nav
    venue_exposure: dict[str, float] = {}
    for t in targets:
        venue_exposure[t.venue] = venue_exposure.get(t.venue, 0.0) + t.notional_usd
    worst_venue_ratio = max(
        (exp / max_venue for exp in venue_exposure.values()), default=0.0
    )
    if worst_venue_ratio > 1.0:
        scale = 1.0 / worst_venue_ratio
        targets = [
            PositionTarget(t.timestamp, t.coin, t.venue, t.side,
                           t.notional_usd * scale, t.strategy)
            for t in targets
        ]

    # 3. Net delta cap
    net_delta = sum(
        t.notional_usd * (1 if t.side == "long" else -1) for t in targets
    )
    max_delta = max_net_delta_pct * nav
    if abs(net_delta) > max_delta and abs(net_delta) > 0:
        # Scale only the leg that's causing the imbalance
        excess = abs(net_delta) - max_delta
        dominant_side = "long" if net_delta > 0 else "short"
        side_total = sum(
            t.notional_usd for t in targets if t.side == dominant_side
        )
        if side_total > 0:
            scale = (side_total - excess) / side_total
            scale = max(scale, 0.0)
            targets = [
                PositionTarget(t.timestamp, t.coin, t.venue, t.side,
                               t.notional_usd * scale if t.side == dominant_side
                               else t.notional_usd, t.strategy)
                for t in targets
            ]

    return targets
