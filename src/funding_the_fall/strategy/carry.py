"""Funding rate carry strategy — signal generation and entry/exit rules.

Owner: Jean-Luc Choiseul

For each token at each funding epoch (8h), the strategy:
  1. Ranks all venue pairs by funding rate spread
  2. Generates entry signals when spread exceeds threshold
  3. Generates exit signals when spread mean-reverts or holding period expires

Positions are delta-neutral: long the perp on the low-rate venue,
short on the high-rate venue.

Entry/exit thresholds are NOT hardcoded — use grid_search_params() to find
optimal thresholds per (coin, venue_pair) by scanning over a parameter grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CarrySignal:
    """A single carry trade signal."""

    timestamp: pd.Timestamp
    coin: str
    long_venue: str         # venue with lower funding rate (we receive)
    short_venue: str        # venue with higher funding rate (we pay less)
    spread: float           # annualized spread in decimal (e.g. 0.15 = 15%)
    action: str             # "enter" or "exit"


@dataclass
class CarryParams:
    """Parameters for carry entry/exit rules on a single (coin, venue_pair)."""

    coin: str
    long_venue: str
    short_venue: str
    entry_spread: float          # min annualized spread to enter
    exit_spread: float           # spread level to exit
    max_holding_epochs: int      # max 8h epochs before forced exit


@dataclass
class GridSearchResult:
    """Result of a single grid point evaluation."""

    params: CarryParams
    n_trades: int
    total_carry_pnl: float       # cumulative funding collected
    sharpe: float                # annualized Sharpe of carry PnL
    avg_holding_epochs: float
    win_rate: float


def compute_funding_spreads(funding_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise funding rate spreads across venues for each coin/epoch.

    Input: DataFrame with columns [timestamp, venue, coin, funding_rate]
    Output: DataFrame with columns [timestamp, coin, long_venue, short_venue,
            spread, spread_annualized]

    Spread = short_venue_rate - long_venue_rate (positive means carry profit).
    Annualized = per-period spread × 3 × 365 (3 funding periods per day).
    """
    raise NotImplementedError


def simulate_carry(
    spreads_df: pd.DataFrame,
    params: CarryParams,
) -> list[CarrySignal]:
    """Run carry strategy with given params on historical spread data.

    Returns the full list of entry + exit signals in chronological order.
    Used both for live signal generation and for grid search evaluation.
    """
    raise NotImplementedError


def evaluate_carry(
    spreads_df: pd.DataFrame,
    params: CarryParams,
) -> GridSearchResult:
    """Evaluate a single parameter set: run simulate_carry and compute metrics.

    Metrics: trade count, total carry PnL, Sharpe, avg holding period, win rate.
    """
    raise NotImplementedError


def grid_search_params(
    spreads_df: pd.DataFrame,
    coin: str,
    long_venue: str,
    short_venue: str,
    entry_spreads: list[float] | None = None,
    exit_spreads: list[float] | None = None,
    max_holding_epochs_list: list[int] | None = None,
) -> list[GridSearchResult]:
    """Grid search over entry/exit thresholds for a (coin, venue_pair).

    Default grids:
      entry_spreads:  [0.05, 0.08, 0.10, 0.12, 0.15, 0.20] annualized
      exit_spreads:   [0.01, 0.02, 0.03, 0.05] annualized
      max_holding:    [15, 30, 45, 60] epochs (5–20 days)

    Returns list of GridSearchResult sorted by Sharpe descending.
    """
    if entry_spreads is None:
        entry_spreads = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    if exit_spreads is None:
        exit_spreads = [0.01, 0.02, 0.03, 0.05]
    if max_holding_epochs_list is None:
        max_holding_epochs_list = [15, 30, 45, 60]

    results: list[GridSearchResult] = []
    for entry in entry_spreads:
        for exit_ in exit_spreads:
            if exit_ >= entry:
                continue  # exit must be below entry
            for hold in max_holding_epochs_list:
                p = CarryParams(
                    coin=coin,
                    long_venue=long_venue,
                    short_venue=short_venue,
                    entry_spread=entry,
                    exit_spread=exit_,
                    max_holding_epochs=hold,
                )
                results.append(evaluate_carry(spreads_df, p))

    results.sort(key=lambda r: r.sharpe, reverse=True)
    return results


def grid_search_all_pairs(
    spreads_df: pd.DataFrame,
) -> dict[tuple[str, str, str], list[GridSearchResult]]:
    """Run grid search for every (coin, long_venue, short_venue) in the data.

    Returns dict keyed by (coin, long_venue, short_venue) → sorted results.
    """
    pairs = (
        spreads_df[["coin", "long_venue", "short_venue"]]
        .drop_duplicates()
        .values.tolist()
    )
    return {
        (coin, lv, sv): grid_search_params(spreads_df, coin, lv, sv)
        for coin, lv, sv in pairs
    }


def select_best_params(
    grid_results: dict[tuple[str, str, str], list[GridSearchResult]],
    min_trades: int = 5,
) -> dict[tuple[str, str, str], CarryParams]:
    """Select best params per pair, requiring a minimum trade count.

    Filters out results with fewer than min_trades, then picks highest Sharpe.
    """
    best: dict[tuple[str, str, str], CarryParams] = {}
    for key, results in grid_results.items():
        valid = [r for r in results if r.n_trades >= min_trades]
        if valid:
            best[key] = valid[0].params  # already sorted by Sharpe desc
    return best
