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

import pandas as pd


@dataclass
class CarrySignal:
    """A single carry trade signal."""

    timestamp: pd.Timestamp
    coin: str
    long_venue: str  # venue with lower funding rate (we receive)
    short_venue: str  # venue with higher funding rate (we pay less)
    spread: float  # annualized spread in decimal (e.g. 0.15 = 15%)
    action: str  # "enter" or "exit"


@dataclass
class CarryParams:
    """Parameters for carry entry/exit rules on a single (coin, venue_pair)."""

    coin: str
    long_venue: str
    short_venue: str
    entry_spread: float  # min annualized spread to enter
    exit_spread: float  # spread level to exit
    max_holding_epochs: int  # max 8h epochs before forced exit


@dataclass
class GridSearchResult:
    """Result of a single grid point evaluation."""

    params: CarryParams
    n_trades: int
    total_carry_pnl: float  # cumulative funding collected
    sharpe: float  # annualized Sharpe of carry PnL
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
    FUNDING_PERIODS_PER_DAY = 3
    DAYS_PER_YEAR = 365

    # For each (timestamp, coin), build all ordered venue pairs
    grouped = funding_df.groupby(["timestamp", "coin"])
    rows: list[dict] = []

    for (ts, coin), grp in grouped:
        venues = grp.set_index("venue")["funding_rate"]
        venue_list = venues.index.tolist()
        for i, v1 in enumerate(venue_list):
            for v2 in venue_list[i + 1 :]:
                r1, r2 = venues[v1], venues[v2]
                # Convention: long_venue has the lower rate (we receive funding),
                # short_venue has the higher rate (we pay less negative / more positive).
                # spread = short_rate - long_rate; positive => carry profit.
                if r1 <= r2:
                    long_v, short_v = v1, v2
                    spread = r2 - r1
                else:
                    long_v, short_v = v2, v1
                    spread = r1 - r2
                rows.append(
                    {
                        "timestamp": ts,
                        "coin": coin,
                        "long_venue": long_v,
                        "short_venue": short_v,
                        "spread": spread,
                        "spread_annualized": spread * FUNDING_PERIODS_PER_DAY * DAYS_PER_YEAR,
                    }
                )

    return pd.DataFrame(rows)


def simulate_carry(
    spreads_df: pd.DataFrame,
    params: CarryParams,
) -> list[CarrySignal]:
    """Run carry strategy with given params on historical spread data.

    Returns the full list of entry + exit signals in chronological order.
    Used both for live signal generation and for grid search evaluation.
    """
    # Filter to the specific (coin, long_venue, short_venue) pair
    mask = (
        (spreads_df["coin"] == params.coin)
        & (spreads_df["long_venue"] == params.long_venue)
        & (spreads_df["short_venue"] == params.short_venue)
    )
    pair_df = spreads_df.loc[mask].sort_values("timestamp").reset_index(drop=True)

    signals: list[CarrySignal] = []
    in_position = False
    entry_epoch_idx = 0

    for idx, row in pair_df.iterrows():
        ts = row["timestamp"]
        ann_spread = row["spread_annualized"]

        if not in_position:
            # Entry condition: annualized spread exceeds entry threshold
            if ann_spread >= params.entry_spread:
                signals.append(
                    CarrySignal(
                        timestamp=pd.Timestamp(ts),
                        coin=params.coin,
                        long_venue=params.long_venue,
                        short_venue=params.short_venue,
                        spread=ann_spread,
                        action="enter",
                    )
                )
                in_position = True
                entry_epoch_idx = idx
        else:
            # Exit conditions:
            #   1. Spread mean-reverts below exit threshold
            #   2. Holding period exceeds max epochs
            epochs_held = idx - entry_epoch_idx
            if ann_spread <= params.exit_spread or epochs_held >= params.max_holding_epochs:
                signals.append(
                    CarrySignal(
                        timestamp=pd.Timestamp(ts),
                        coin=params.coin,
                        long_venue=params.long_venue,
                        short_venue=params.short_venue,
                        spread=ann_spread,
                        action="exit",
                    )
                )
                in_position = False

    # If still in position at end of data, force exit on last row
    if in_position and len(pair_df) > 0:
        last = pair_df.iloc[-1]
        signals.append(
            CarrySignal(
                timestamp=pd.Timestamp(last["timestamp"]),
                coin=params.coin,
                long_venue=params.long_venue,
                short_venue=params.short_venue,
                spread=last["spread_annualized"],
                action="exit",
            )
        )

    return signals


def evaluate_carry(
    spreads_df: pd.DataFrame,
    params: CarryParams,
) -> GridSearchResult:
    """Evaluate a single parameter set: run simulate_carry and compute metrics.

    Metrics: trade count, total carry PnL, Sharpe, avg holding period, win rate.
    """
    import numpy as np

    signals = simulate_carry(spreads_df, params)

    # Pair up enter/exit signals into round-trip trades
    trades: list[dict] = []
    i = 0
    while i < len(signals):
        if signals[i].action == "enter":
            entry_sig = signals[i]
            # Find matching exit
            if i + 1 < len(signals) and signals[i + 1].action == "exit":
                exit_sig = signals[i + 1]

                # Filter the spread series for the holding window
                mask = (
                    (spreads_df["coin"] == params.coin)
                    & (spreads_df["long_venue"] == params.long_venue)
                    & (spreads_df["short_venue"] == params.short_venue)
                    & (spreads_df["timestamp"] >= entry_sig.timestamp)
                    & (spreads_df["timestamp"] <= exit_sig.timestamp)
                )
                window = spreads_df.loc[mask].sort_values("timestamp")
                holding_epochs = max(len(window) - 1, 1)

                # PnL = sum of per-epoch spreads collected while in the trade
                # Each epoch the carry profit is the raw spread (not annualized)
                per_epoch_pnl = window["spread"].sum()

                trades.append(
                    {
                        "entry_ts": entry_sig.timestamp,
                        "exit_ts": exit_sig.timestamp,
                        "holding_epochs": holding_epochs,
                        "pnl": per_epoch_pnl,
                    }
                )
                i += 2
            else:
                i += 1
        else:
            i += 1

    n_trades = len(trades)
    if n_trades == 0:
        return GridSearchResult(
            params=params,
            n_trades=0,
            total_carry_pnl=0.0,
            sharpe=0.0,
            avg_holding_epochs=0.0,
            win_rate=0.0,
        )

    pnl_list = [t["pnl"] for t in trades]
    total_pnl = sum(pnl_list)
    avg_hold = np.mean([t["holding_epochs"] for t in trades])
    win_rate = np.mean([1.0 if p > 0 else 0.0 for p in pnl_list])

    # Annualized Sharpe: mean per-trade return / std, scaled by sqrt(trades/year)
    # Each epoch is 8h, so ~1095 epochs/year
    pnl_arr = np.array(pnl_list)
    if pnl_arr.std() > 0 and n_trades > 1:
        per_trade_sharpe = pnl_arr.mean() / pnl_arr.std()
        # Scale to annual: assume avg_hold epochs per trade
        trades_per_year = 1095.0 / avg_hold if avg_hold > 0 else 0.0
        sharpe = per_trade_sharpe * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    return GridSearchResult(
        params=params,
        n_trades=n_trades,
        total_carry_pnl=total_pnl,
        sharpe=float(sharpe),
        avg_holding_epochs=float(avg_hold),
        win_rate=float(win_rate),
    )


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
