"""Performance analytics — Sharpe, drawdown, trade statistics.

Owner: Jean Mauratille

Produces the performance tables required for the pitchbook and
technical notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
    net_pnl: float
    calmar_ratio: float


def compute_performance(
    nav_series: pd.Series,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> PerformanceStats:
    """Compute comprehensive performance statistics.

    Parameters:
        nav_series: time-indexed Series of portfolio NAV
        trades_df: DataFrame of executed trades
        risk_free_rate: annualized risk-free rate for Sharpe calculation
    """
    raise NotImplementedError


def pnl_decomposition(
    result: object,
) -> pd.DataFrame:
    """Decompose PnL into components: carry, cascade, and costs.

    Returns DataFrame with columns:
      [timestamp, carry_pnl, cascade_pnl, funding_pnl, trading_costs, net_pnl]
    """
    raise NotImplementedError
