"""Backtest engine — event-driven portfolio simulation.

Owner: Jean Mauratille

Processes funding epochs (8h intervals) sequentially:
  1. Apply funding payments to open positions
  2. Mark-to-market all positions
  3. Generate new signals (carry + cascade)
  4. Compute position targets via allocation
  5. Execute trades (with or without transaction costs)
  6. Enforce risk limits

Produces a trade log and a time series of portfolio state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from funding_the_fall.backtest.costs import TransactionCostModel


@dataclass
class Trade:
    """A single executed trade."""

    timestamp: pd.Timestamp
    coin: str
    venue: str
    side: str                 # "long" or "short"
    notional_usd: float
    price: float
    fee_usd: float            # transaction cost (spread + impact)
    strategy: str             # "carry" or "cascade"


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time."""

    timestamp: pd.Timestamp
    nav: float
    gross_leverage: float
    net_delta_pct: float
    n_positions: int
    positions: dict           # {(coin, venue): notional_usd}
    cumulative_funding: float
    cumulative_trading_costs: float


@dataclass
class BacktestResult:
    """Complete backtest output."""

    trades: list[Trade] = field(default_factory=list)
    portfolio_states: list[PortfolioState] = field(default_factory=list)
    funding_payments: pd.DataFrame | None = None  # detailed funding log

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def nav_series(self) -> pd.Series:
        return pd.Series(
            {s.timestamp: s.nav for s in self.portfolio_states}
        )


def run_backtest(
    funding_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    oi_df: pd.DataFrame | None = None,
    initial_nav: float = 1_000_000.0,
    cost_model: TransactionCostModel | None = None,
    carry_weight: float = 0.70,
    cascade_weight: float = 0.30,
) -> BacktestResult:
    """Run the full backtest.

    Parameters:
        funding_df: unified funding rates [timestamp, venue, coin, funding_rate]
        candles_df: mark prices [timestamp, venue, coin, o, h, l, c, v]
        oi_df: open interest (optional, for cascade sizing)
        initial_nav: starting capital in USD
        cost_model: transaction cost model (None = zero costs)
        carry_weight: allocation to carry strategy
        cascade_weight: allocation to cascade overlay

    Returns:
        BacktestResult with trade log, portfolio states, and funding payments.

    The project requires running this twice:
      (A) cost_model=None              → zero transaction costs
      (B) cost_model=TransactionCostModel(...)  → calibrated costs
    """
    raise NotImplementedError
