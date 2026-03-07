"""Backtester — portfolio engine, transaction costs, and performance analytics."""

from funding_the_fall.backtest.engine import (
    run_backtest,
    OpenPosition,
    BacktestResult,
    PortfolioState,
    Trade,
)
from funding_the_fall.backtest.costs import TransactionCostModel, VENUE_FEES
from funding_the_fall.backtest.performance import (
    compute_performance,
    compute_performance_from_result,
    pnl_decomposition,
    PerformanceStats,
)

__all__ = [
    "run_backtest",
    "OpenPosition",
    "BacktestResult",
    "PortfolioState",
    "Trade",
    "TransactionCostModel",
    "VENUE_FEES",
    "compute_performance",
    "compute_performance_from_result",
    "pnl_decomposition",
    "PerformanceStats",
]
