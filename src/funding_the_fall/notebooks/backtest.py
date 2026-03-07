from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from funding_the_fall.backtest.engine import BacktestResult, run_backtest
from funding_the_fall.backtest.performance import (
    compute_performance_from_result,
    pnl_decomposition,
)
from funding_the_fall.models.cascade import generate_cascade_signals

from .config import CARRY_LEV, CASCADE_LEV, DEPTH_SCALE


def generate_cascade_signals_from_oi(
    oi_matched: pl.DataFrame,
    measured_depth: dict[str, float],
    venue: str = "hyperliquid",
    top_n: int = 2,
    rebalance_hours: int = 24,
    risk_threshold: float = 0.5,
):
    cascade_signals = generate_cascade_signals(
        oi_df=oi_matched,
        depth_per_coin=measured_depth,
        venue=venue,
        top_n=top_n,
        rebalance_hours=rebalance_hours,
        risk_threshold=risk_threshold,
    )

    n_enter = sum(1 for s in cascade_signals if s.action == "enter")
    n_exit = sum(1 for s in cascade_signals if s.action == "exit")
    coins_targeted = set(s.coin for s in cascade_signals if s.action == "enter")
    print(f"Cascade signals: {n_enter} entries, {n_exit} exits")
    print(f"Coins targeted: {sorted(coins_targeted)}")
    return cascade_signals


def run_backtests(
    all_carry_signals,
    cascade_signals,
    funding: pl.DataFrame,
    candles: pl.DataFrame,
    initial_nav: float = 1_000_000.0,
    carry_leverage: float = CARRY_LEV,
    cascade_leverage: float = CASCADE_LEV,
    depth_per_coin: dict[str, float] | None = None,
    impact_model: str = "sqrt",
    depth_scale: float = DEPTH_SCALE,
):
    funding_pd = funding.to_pandas()
    candles_pd = candles.to_pandas()

    bt_kwargs = dict(
        carry_signals=all_carry_signals,
        funding_df=funding_pd,
        candles_df=candles_pd,
        initial_nav=initial_nav,
        carry_leverage=carry_leverage,
        carry_budget_pct=0.85,
        cascade_signals=cascade_signals,
        cascade_leverage=cascade_leverage,
        cascade_budget_pct=0.15,
        depth_per_coin=depth_per_coin,
        impact_model=impact_model,
        depth_scale=depth_scale,
    )

    result = run_backtest(**bt_kwargs)

    nav = result.nav_series()
    trades = result.trades_df()
    print(f"Backtest: {len(result.portfolio_states)} epochs, {result.trade_count} trades")
    print(f"NAV: ${nav.iloc[0]:,.0f} -> ${nav.iloc[-1]:,.0f} ({nav.iloc[-1] / nav.iloc[0] - 1:+.2%})")
    print(f"Carry trades: {len(trades[trades['strategy'] == 'carry']) if not trades.empty else 0}")
    print(f"Cascade trades: {len(trades[trades['strategy'] == 'cascade']) if not trades.empty else 0}")
    if not trades.empty and "strategy" in trades.columns:
        liq = trades[trades["strategy"].str.contains("liquidation")]
        print(f"Liquidations: {len(liq)}")

    # Zero-cost comparison (no fees, but still has impact)
    result_zero = run_backtest(**{**bt_kwargs, "fee_multiplier": 0.0})

    return result, result_zero


def print_fee_comparison(result, result_zero, risk_free_rate: float = 0.03):
    stats = compute_performance_from_result(result, risk_free_rate=risk_free_rate)
    stats_zero = compute_performance_from_result(result_zero, risk_free_rate=risk_free_rate)

    print(f"{'Metric':<25} {'With Fees':>14} {'Zero Fees':>14} {'Fee Drag':>14}")
    print("-" * 70)
    for label, v_real, v_zero in [
        ("Total Return", stats.total_return, stats_zero.total_return),
        ("Ann. Return", stats.annualized_return, stats_zero.annualized_return),
        ("Sharpe", stats.sharpe_ratio, stats_zero.sharpe_ratio),
        ("Max Drawdown", stats.max_drawdown, stats_zero.max_drawdown),
    ]:
        if "Sharpe" in label:
            print(f"{label:<25} {v_real:>14.2f} {v_zero:>14.2f} {v_real - v_zero:>+14.2f}")
        else:
            print(f"{label:<25} {v_real:>14.2%} {v_zero:>14.2%} {v_real - v_zero:>+14.2%}")
    print(
        f"{'Trading Costs':<25} ${stats.total_trading_costs:>13,.0f} ${stats_zero.total_trading_costs:>13,.0f}"
    )


def plot_nav_and_leverage(result, result_zero):
    nav = result.nav_series()
    nav_zero = result_zero.nav_series()

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax = axes[0]
    ax.plot(
        nav_zero.index, nav_zero.values / 1e6, color="gray", lw=1, ls="--", alpha=0.7, label="Zero fees"
    )
    ax.plot(nav.index, nav.values / 1e6, color="steelblue", lw=1.5, label="With fees")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylabel("NAV ($M)")
    ax.set_title("Portfolio NAV")
    ax.legend(fontsize=9)

    states_df = pd.DataFrame([s.__dict__ for s in result.portfolio_states])
    ax2 = axes[1]
    ax2.plot(
        states_df["timestamp"], states_df["gross_leverage"], color="coral", lw=1, label="Gross leverage"
    )
    ax2.set_ylabel("Gross Leverage")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8, loc="upper left")

    ax3 = ax2.twinx()
    ax3.grid(False)
    ax3.plot(
        states_df["timestamp"],
        states_df["n_positions"],
        color="gray",
        lw=0.8,
        alpha=0.6,
        label="# positions",
    )
    ax3.set_ylabel("# Positions", color="gray")
    ax3.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    plt.show()


def print_performance_stats(result, risk_free_rate: float = 0.05):
    stats = compute_performance_from_result(result, risk_free_rate=risk_free_rate)

    perf_rows = [
        ("Total Return", f"{stats.total_return:+.2%}"),
        ("Annualized Return", f"{stats.annualized_return:+.2%}"),
        ("Annualized Volatility", f"{stats.annualized_vol:.2%}"),
        ("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}"),
        ("Calmar Ratio", f"{stats.calmar_ratio:.2f}"),
        ("Max Drawdown", f"{stats.max_drawdown:.2%}"),
        ("Max DD Duration", f"{stats.max_drawdown_duration_days:.0f} days"),
        ("Total Trades", f"{stats.total_trades:,}"),
        ("Cumulative Funding", f"${stats.total_funding_collected:+,.0f}"),
        ("Cumulative Trading Costs", f"${stats.total_trading_costs:,.0f}"),
        ("Net PnL", f"${stats.net_pnl:+,.0f}"),
    ]
    perf_table = pd.DataFrame(perf_rows, columns=["Metric", "Value"])
    print(perf_table.to_string(index=False))


def plot_drawdown_and_rolling_sharpe(result):
    nav = result.nav_series()
    returns = nav.pct_change().dropna()
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    ax = axes[0]
    ax.fill_between(drawdown.index, drawdown.values, 0, color="salmon", alpha=0.6)
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater Curve")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    window = 90
    if len(returns) > window:
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean * 3 * 365 - 0.05) / (rolling_std * np.sqrt(3 * 365))
        rolling_sharpe = rolling_sharpe.clip(-5, 5)
        ax2 = axes[1]
        ax2.plot(rolling_sharpe.dropna().index, rolling_sharpe.dropna().values, color="steelblue", lw=1)
        ax2.axhline(0, color="gray", ls=":", lw=0.8)
        ax2.set_ylabel("Rolling Sharpe (30d)")
        ax2.set_xlabel("Date")
        ax2.set_title("30-Day Rolling Sharpe Ratio")

    fig.tight_layout()
    plt.show()


def print_trade_statistics(result):
    trades = result.trades_df()
    if not trades.empty:
        for strat in ["carry", "cascade"]:
            st = trades[trades["strategy"] == strat]
            if st.empty:
                continue
            total_notional = st["notional_usd"].sum()
            total_fees = st["fee_usd"].sum()
            print(f"--- {strat.upper()} ---")
            print(f"  Trades: {len(st)}")
            print(f"  Coins: {sorted(st['coin'].unique())}")
            print(f"  Total notional: ${total_notional:,.0f}")
            print(f"  Total fees: ${total_fees:,.0f}")
            print(f"  Avg fee/trade: ${total_fees / len(st):,.2f}")
            print()

        liq_trades = trades[trades["strategy"].str.contains("liquidation")]
        if not liq_trades.empty:
            print("--- LIQUIDATIONS ---")
            print(f"  Total: {len(liq_trades)}")
            for coin in sorted(liq_trades["coin"].unique()):
                n = len(liq_trades[liq_trades["coin"] == coin])
                print(f"  {coin}: {n}")
        else:
            print("No liquidations.")
    else:
        print("No trades executed.")


def run_impact_comparison(
    all_carry_signals,
    cascade_signals,
    funding: pl.DataFrame,
    candles: pl.DataFrame,
    initial_nav: float = 1_000_000.0,
    carry_leverage: float = CARRY_LEV,
    cascade_leverage: float = CASCADE_LEV,
    depth_per_coin: dict[str, float] | None = None,
    depth_scale: float = DEPTH_SCALE,
) -> dict[str, BacktestResult]:
    """Run backtest with no impact, sqrt impact, and calibrated linear impact.

    The linear model is calibrated per-coin so that at each coin's median trade
    size, linear_cost == sqrt_cost.  This is achieved by setting
    D_linear[coin] = sqrt(V_median[coin] * D[coin]) for each coin, which
    requires no engine changes — just a modified depth_per_coin dict.
    """
    funding_pd = funding.to_pandas()
    candles_pd = candles.to_pandas()

    bt_kwargs = dict(
        carry_signals=all_carry_signals,
        funding_df=funding_pd,
        candles_df=candles_pd,
        initial_nav=initial_nav,
        carry_leverage=carry_leverage,
        carry_budget_pct=0.85,
        cascade_signals=cascade_signals,
        cascade_leverage=cascade_leverage,
        cascade_budget_pct=0.15,
        depth_per_coin=depth_per_coin,
        depth_scale=depth_scale,
    )

    results = {}
    results["none"] = run_backtest(**{**bt_kwargs, "impact_model": "none"})
    results["sqrt"] = run_backtest(**{**bt_kwargs, "impact_model": "sqrt"})

    # Calibrate linear: per-coin depth so linear matches sqrt at median trade.
    # Engine computes d_eff = D_raw * S.  Setting sqrt_cost == linear_cost at
    # V_med gives D_lin = sqrt(V_med * D_raw / S).
    linear_depth = dict(depth_per_coin) if depth_per_coin else {}
    if depth_per_coin:
        trades = results["none"].trades_df()
        if not trades.empty:
            for coin, d in depth_per_coin.items():
                coin_trades = trades.loc[trades["coin"] == coin, "notional_usd"]
                if not coin_trades.empty and d > 0:
                    v_med = float(coin_trades.abs().median())
                    linear_depth[coin] = (v_med * d / depth_scale) ** 0.5
    results["linear"] = run_backtest(
        **{**bt_kwargs, "impact_model": "linear", "depth_per_coin": linear_depth}
    )
    return results


def print_impact_comparison(
    results: dict[str, BacktestResult],
    risk_free_rate: float = 0.05,
):
    """Print a comparison table across impact model variants."""
    print(f"{'Metric':<25} {'No Impact':>14} {'Sqrt (base)':>14} {'Linear (cal.)':>14}")
    print("-" * 70)

    navs = {k: v.nav_series() for k, v in results.items()}
    final = {k: v.iloc[-1] for k, v in navs.items()}
    initial = navs["none"].iloc[0]
    returns = {k: v / initial - 1 for k, v in final.items()}

    for label, vals in [
        ("Total Return", returns),
        ("Final NAV", final),
    ]:
        if "Return" in label:
            print(f"{label:<25} {vals['none']:>14.2%} {vals['sqrt']:>14.2%} {vals['linear']:>14.2%}")
        else:
            print(f"{label:<25} ${vals['none']:>13,.0f} ${vals['sqrt']:>13,.0f} ${vals['linear']:>13,.0f}")

    impact_vals = {k: results[k].portfolio_states[-1].cumulative_impact_costs for k in results}
    print(f"{'Impact Costs':<25} ${impact_vals['none']:>13,.0f} ${impact_vals['sqrt']:>13,.0f} ${impact_vals['linear']:>13,.0f}")

    total_costs = {k: results[k].portfolio_states[-1].cumulative_trading_costs for k in results}
    print(f"{'Total Costs':<25} ${total_costs['none']:>13,.0f} ${total_costs['sqrt']:>13,.0f} ${total_costs['linear']:>13,.0f}")


def plot_impact_comparison(results: dict[str, BacktestResult]):
    """Plot NAV curves for all three impact model variants."""
    fig, ax = plt.subplots(figsize=(14, 5))

    styles = {
        "none": ("gray", "--", "No Impact"),
        "sqrt": ("steelblue", "-", "Sqrt Impact (base case)"),
        "linear": ("coral", "-", "Linear Impact (calibrated)"),
    }
    for model, (color, ls, label) in styles.items():
        nav = results[model].nav_series()
        ax.plot(nav.index, nav.values / 1e6, color=color, ls=ls, lw=1.5, label=label)

    ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylabel("NAV ($M)")
    ax.set_title("Impact Model Sensitivity")
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_pnl_decomposition(result):
    decomp = pnl_decomposition(result)

    if not decomp.empty:
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        ax = axes[0]
        ax.plot(decomp.index, decomp["nav"] / 1e6, color="steelblue", lw=1.5, label="NAV")
        ax.set_ylabel("$M")
        ax.set_title("NAV and PnL Components")
        ax.legend(fontsize=8, loc="upper left")

        ax_r = ax.twinx()
        ax_r.grid(False)
        ax_r.plot(
            decomp.index,
            decomp["cumulative_funding"] / 1e3,
            color="green",
            lw=1,
            alpha=0.8,
            label="Cum. funding ($K)",
        )
        ax_r.plot(
            decomp.index,
            -decomp["cumulative_costs"] / 1e3,
            color="red",
            lw=1,
            alpha=0.8,
            label="Cum. costs ($K)",
        )
        ax_r.set_ylabel("$K")
        ax_r.legend(fontsize=8, loc="upper right")

        # Align LHS y=1.0 ($1M NAV) with RHS y=0 ($0K costs/funding)
        lmin, lmax = ax.get_ylim()
        frac = (1.0 - lmin) / (lmax - lmin)
        rmin, _ = ax_r.get_ylim()
        ax_r.set_ylim(rmin, rmin * (1 - 1 / frac))

        ax2 = axes[1]
        ax2.plot(decomp.index, decomp["gross_leverage"], color="coral", lw=1, label="Gross leverage")
        ax2.set_ylabel("Leverage")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=8, loc="upper left")

        ax3 = ax2.twinx()
        ax3.grid(False)
        ax3.fill_between(
            decomp.index, decomp["n_positions"], 0, color="gray", alpha=0.2, label="# positions"
        )
        ax3.set_ylabel("# Positions")
        ax3.legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        plt.show()

        final = decomp.iloc[-1]
        initial_nav = decomp["nav"].iloc[0]
        print(f"Initial NAV:    ${initial_nav:>12,.0f}")
        print(f"Final NAV:      ${final['nav']:>12,.0f}")
        print(f"Net PnL:        ${final['nav'] - initial_nav:>+12,.0f}")
        print(f"Cum. Funding:   ${final['cumulative_funding']:>+12,.0f}")
        print(f"Cum. Costs:     ${final['cumulative_costs']:>12,.0f}")
        print(
            f"Price PnL:      ${final['nav'] - initial_nav - final['cumulative_funding'] + final['cumulative_costs']:>+12,.0f}"
        )
    else:
        print("No portfolio states to decompose.")
