from __future__ import annotations

import pandas as pd
import polars as pl

from funding_the_fall.models.cascade import (
    build_positions_tiered,
    cascade_risk_signal,
)
from funding_the_fall.strategy.allocation import (
    allocate_positions,
    CARRY_RISK_DAMPING,
)
from funding_the_fall.strategy.carry import (
    CarryParams,
    CarrySignal,
    PooledCarryParams,
    evaluate_carry,
    grid_search_per_coin,
    simulate_carry,
)

from .config import CARRY_LEV, CASCADE_LEV, COINS, NAV


def run_carry_grid_search(
    spreads: pd.DataFrame,
) -> tuple[dict[str, PooledCarryParams], list[CarrySignal], dict[tuple, float]]:
    carry_spreads = spreads.rename(columns={"spread_ann": "spread_annualized"}).copy()

    best_per_coin: dict[str, PooledCarryParams] = {}
    for coin in COINS:
        results = grid_search_per_coin(carry_spreads, coin, min_trades=3, progress=True)
        if results:
            best = results[0]
            best_per_coin[coin] = best.params
            print(
                f"  {coin:5s}  entry>={best.params.entry_spread:.0%}  "
                f"exit<={best.params.exit_spread:.0%}  "
                f"max_hold={best.params.max_holding_epochs:2d}ep  "
                f"trades={best.total_trades:3d}  "
                f"pooled_Sharpe={best.pooled_sharpe:.2f}  "
                f"win={best.avg_win_rate:.0%}"
            )

    pairs = carry_spreads[["coin", "long_venue", "short_venue"]].drop_duplicates().values.tolist()
    all_carry_signals: list[CarrySignal] = []
    pair_results: dict[tuple, float] = {}
    for coin_name, lv, sv in pairs:
        if coin_name not in best_per_coin:
            continue
        bp = best_per_coin[coin_name]
        p = CarryParams(
            coin=coin_name,
            long_venue=lv,
            short_venue=sv,
            entry_spread=bp.entry_spread,
            exit_spread=bp.exit_spread,
            max_holding_epochs=bp.max_holding_epochs,
        )
        sigs = simulate_carry(carry_spreads, p)
        r = evaluate_carry(carry_spreads, p)
        if r.n_trades > 0:
            all_carry_signals.extend(sigs)
            pair_results[(coin_name, lv, sv)] = r.sharpe

    print(
        f"\nTotal carry signals generated: {len(all_carry_signals)} "
        f"({sum(1 for s in all_carry_signals if s.action == 'enter')} entries, "
        f"{len(pair_results)} active pairs)"
    )
    return best_per_coin, all_carry_signals, pair_results


def allocate_and_print(
    all_carry_signals: list[CarrySignal],
    oi_matched: pl.DataFrame,
    measured_depth: dict[str, float],
    signals: dict,
    nav: float = NAV,
    carry_leverage: float = CARRY_LEV,
    cascade_leverage: float = CASCADE_LEV,
):
    total_depth = sum(measured_depth.values()) / max(len(measured_depth), 1)

    latest_snap = oi_matched.filter(pl.col("timestamp") == oi_matched["timestamp"].max())
    latest_positions = build_positions_tiered(latest_snap)
    agg_signal = cascade_risk_signal(
        latest_positions, current_price=1.0, orderbook_depth_usd=total_depth
    )

    entry_signals = [s for s in all_carry_signals if s.action == "enter"]

    targets = allocate_positions(
        carry_signals=entry_signals,
        cascade_signal=agg_signal,
        nav=nav,
        per_coin_signals=signals,
        deepest_venue="binance",
        carry_leverage=carry_leverage,
        cascade_leverage=cascade_leverage,
    )

    carry_notional = sum(t.notional_usd for t in targets if t.strategy == "carry")
    carry_margin = sum(t.collateral_usd for t in targets if t.strategy == "carry")
    cascade_notional = sum(t.notional_usd for t in targets if t.strategy == "cascade")
    cascade_margin = sum(t.collateral_usd for t in targets if t.strategy == "cascade")
    gross = sum(t.notional_usd for t in targets)
    total_margin = sum(t.collateral_usd for t in targets)
    net = sum(t.notional_usd * (1 if t.side == "long" else -1) for t in targets)

    risk_score = agg_signal["risk_score"]
    carry_scale = 1.0 - CARRY_RISK_DAMPING * risk_score

    print(
        f"Aggregate cascade risk_score: {risk_score:.3f}  ->  carry scaled to {carry_scale:.0%} of baseline"
    )
    print(f"\n{'Strategy':<12} {'Margin':>12} {'Notional':>12} {'Leverage':>9} {'% NAV':>8}")
    print("-" * 55)
    print(
        f"{'Carry':<12} ${carry_margin:>11,.0f} ${carry_notional:>11,.0f} {carry_leverage:>8.1f}x {carry_notional / nav:>7.1%}"
    )
    print(
        f"{'Cascade':<12} ${cascade_margin:>11,.0f} ${cascade_notional:>11,.0f} {cascade_leverage:>8.1f}x {cascade_notional / nav:>7.1%}"
    )
    print(
        f"{'Cash':<12} ${nav - total_margin:>11,.0f} {'':>12} {'':>9} {(nav - total_margin) / nav:>7.1%}"
    )
    print("-" * 55)
    print(f"{'Gross':12} ${total_margin:>11,.0f} ${gross:>11,.0f} {gross / nav:>8.1f}x")
    print(f"{'Net delta':12} {'':>12} ${net:>11,.0f} {'':>9} {net / nav:>7.1%}")
    print(f"\nTotal position targets: {len(targets)}")

    return targets, agg_signal


def print_risk_limits(targets, signals: dict, nav: float = NAV):
    venue_exp: dict[str, float] = {}
    coin_exp: dict[str, float] = {}
    for t in targets:
        venue_exp[t.venue] = venue_exp.get(t.venue, 0) + t.notional_usd
        coin_exp[t.coin] = coin_exp.get(t.coin, 0) + t.notional_usd

    gross = sum(t.notional_usd for t in targets)
    net = sum(t.notional_usd * (1 if t.side == "long" else -1) for t in targets)
    total_margin = sum(t.collateral_usd for t in targets)

    print("Exposure by venue (notional):")
    for v, e in sorted(venue_exp.items(), key=lambda x: -x[1]):
        print(f"  {v:14s} ${e:>11,.0f}  ({e / nav:.1%} of NAV)")

    print("\nExposure by coin (notional):")
    for c, e in sorted(coin_exp.items(), key=lambda x: -x[1]):
        print(f"  {c:5s} ${e:>11,.0f}  ({e / nav:.1%} of NAV)")

    cascade_targets = [t for t in targets if t.strategy == "cascade"]
    if cascade_targets:
        cascade_lev = CASCADE_LEV
        print(f"\nCascade shorts (weighted by per-coin amplification, {cascade_lev:.1f}x leverage):")
        for t in sorted(cascade_targets, key=lambda x: -x.notional_usd):
            amp = signals.get(t.coin, {}).get("amplification_at_5pct", 0)
            print(
                f"  SHORT {t.coin:5s} on {t.venue:12s}  "
                f"margin=${t.collateral_usd:>9,.0f}  notional=${t.notional_usd:>10,.0f}  (A(5%)={amp:.1f}x)"
            )

    print("\n--- Risk Limit Check ---")
    print(
        f"  Gross leverage:  {gross / nav:.2f}x  (limit: 5.0x)  {'PASS' if gross / nav <= 5.0 else 'FAIL'}"
    )
    max_venue_pct = max(venue_exp.values()) / nav if venue_exp else 0
    print(
        f"  Max venue:       {max_venue_pct:.1%}  (limit: 40%)   {'PASS' if max_venue_pct <= 0.40 else 'FAIL'}"
    )
    print(
        f"  Net delta:       {abs(net) / nav:.1%}  (limit: 10%)   {'PASS' if abs(net) / nav <= 0.10 else 'FAIL'}"
    )
    print(
        f"  Total margin:    {total_margin / nav:.1%} of NAV       {'PASS' if total_margin <= nav else 'FAIL'}"
    )
