"""Performance analytics — Sharpe, drawdown, trade statistics.

Owner: Jean Mauratille

Produces the performance tables required for the pitchbook and
technical notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from funding_the_fall.backtest.engine import BacktestResult


@dataclass
class PerformanceStats:
    """Summary statistics for a backtest."""

    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: float
    total_trades: int
    win_rate: float
    avg_trade_pnl: float
    total_funding_collected: float
    total_trading_costs: float
    total_liquidation_losses: float
    net_pnl: float
    calmar_ratio: float


def compute_performance(
    nav_series: pd.Series,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> PerformanceStats:
    """Compute comprehensive performance statistics.

    Parameters
    ----------
    nav_series : pd.Series
        Time-indexed Series of portfolio NAV.
    trades_df : pd.DataFrame
        DataFrame of executed trades (from BacktestResult.trades_df()).
    risk_free_rate : float
        Annualized risk-free rate for Sharpe calculation.
    """
    if nav_series.empty:
        return PerformanceStats(
            total_return=0, annualized_return=0, annualized_vol=0,
            sharpe_ratio=0, max_drawdown=0, max_drawdown_duration_days=0,
            total_trades=0, win_rate=0, avg_trade_pnl=0,
            total_funding_collected=0, total_trading_costs=0,
            total_liquidation_losses=0, net_pnl=0, calmar_ratio=0,
        )

    # Returns
    returns = nav_series.pct_change().dropna()
    total_return = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1)

    # Annualization: infer frequency from median timedelta
    if len(nav_series) > 1:
        diffs = pd.Series(nav_series.index).diff().dropna()
        median_hours = diffs.dt.total_seconds().median() / 3600
        periods_per_year = 8760.0 / max(median_hours, 0.1)
    else:
        periods_per_year = 8760.0

    n_periods = len(returns)
    ann_return = float((1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year)) if n_periods > 1 else 0.0
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Drawdown
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    max_dd = float(drawdown.min())

    # Drawdown duration
    max_dd_dur = 0.0
    dd_start = None
    for ts, dd in drawdown.items():
        if dd < -1e-10 and dd_start is None:
            dd_start = ts
        elif dd >= -1e-10 and dd_start is not None:
            dur = (ts - dd_start).total_seconds() / 86400
            max_dd_dur = max(max_dd_dur, dur)
            dd_start = None
    if dd_start is not None:
        dur = (nav_series.index[-1] - dd_start).total_seconds() / 86400
        max_dd_dur = max(max_dd_dur, dur)

    # Trade statistics
    total_trades = 0
    win_rate = 0.0
    avg_pnl = 0.0
    total_fees = 0.0

    if trades_df is not None and not trades_df.empty:
        total_trades = len(trades_df)
        total_fees = float(trades_df["fee_usd"].sum())

    # Funding and costs from NAV series endpoints
    net_pnl = float(nav_series.iloc[-1] - nav_series.iloc[0])

    calmar = ann_return / abs(max_dd) if max_dd < -1e-10 else 0.0

    return PerformanceStats(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_trade_pnl=avg_pnl,
        total_funding_collected=0.0,  # filled from BacktestResult
        total_trading_costs=total_fees,
        total_liquidation_losses=0.0,  # filled from BacktestResult
        net_pnl=net_pnl,
        calmar_ratio=calmar,
    )


def compute_performance_from_result(
    result: BacktestResult,
    risk_free_rate: float = 0.05,
) -> PerformanceStats:
    """Convenience wrapper: compute performance directly from BacktestResult."""
    nav = result.nav_series()
    trades = result.trades_df()
    stats = compute_performance(nav, trades, risk_free_rate)

    # Fill in funding/costs from portfolio states
    if result.portfolio_states:
        stats.total_funding_collected = result.portfolio_states[-1].cumulative_funding
        stats.total_trading_costs = result.portfolio_states[-1].cumulative_fees
        stats.total_liquidation_losses = result.portfolio_states[-1].cumulative_liquidation_losses

    return stats


def pnl_decomposition(result: BacktestResult) -> pd.DataFrame:
    """Decompose PnL into carry, cascade, funding, and cost components.

    Returns DataFrame indexed by timestamp with columns:
      [nav, carry_trades, cascade_trades, cumulative_funding, cumulative_costs]
    """
    if not result.portfolio_states:
        return pd.DataFrame()

    rows = []
    for s in result.portfolio_states:
        rows.append({
            "timestamp": s.timestamp,
            "nav": s.nav,
            "cumulative_funding": s.cumulative_funding,
            "cumulative_fees": s.cumulative_fees,
            "cumulative_liquidation_losses": s.cumulative_liquidation_losses,
            "n_positions": s.n_positions,
            "gross_leverage": s.gross_leverage,
        })

    df = pd.DataFrame(rows).set_index("timestamp")

    # Trade PnL by strategy (realized at exit)
    trades = result.trades_df()
    if not trades.empty:
        for strat in ["carry", "cascade"]:
            strat_trades = trades[trades["strategy"] == strat]
            if not strat_trades.empty:
                df[f"{strat}_trade_count"] = 0
                for ts, grp in strat_trades.groupby("timestamp"):
                    if ts in df.index:
                        df.loc[ts, f"{strat}_trade_count"] = len(grp)

    return df
