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

from dataclasses import dataclass, field

import pandas as pd

from funding_the_fall.backtest.costs import VENUE_FEES


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
    trade_pnls: list[float] = field(default_factory=list)


@dataclass
class PooledCarryParams:
    """Carry parameters shared across all venue pairs for a coin."""

    coin: str
    entry_spread: float
    exit_spread: float
    max_holding_epochs: int
    leverage: float = 4.0


@dataclass
class PooledGridResult:
    """Result of pooled grid search for a coin."""

    params: PooledCarryParams
    total_trades: int
    total_pnl: float
    pooled_sharpe: float
    avg_holding_epochs: float
    avg_win_rate: float


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
                        "spread_annualized": spread
                        * FUNDING_PERIODS_PER_DAY
                        * DAYS_PER_YEAR,
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
            if (
                ann_spread <= params.exit_spread
                or epochs_held >= params.max_holding_epochs
            ):
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


MAINTENANCE_RATE = 0.005  # must match engine.py


def evaluate_carry(
    spreads_df: pd.DataFrame,
    params: CarryParams,
    leverage: float = 1.0,
    coin_prices: pd.Series | None = None,
) -> GridSearchResult:
    """Evaluate a single parameter set: run simulate_carry and compute metrics.

    Parameters
    ----------
    leverage : float
        Position leverage. PnL is expressed per unit margin (= per-notional × leverage).
        When coin_prices is provided, trades are truncated at the liquidation threshold.
    coin_prices : pd.Series, optional
        Timestamp-indexed mean price for the coin. Used to detect whether a price
        move during a trade would trigger isolated-margin liquidation of one leg.

    Metrics: trade count, total carry PnL, Sharpe, avg holding period, win rate.
    """
    import numpy as np

    signals = simulate_carry(spreads_df, params)
    liq_threshold = 1.0 / leverage - MAINTENANCE_RATE if leverage > 1.0 else None

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

                # Check for liquidation: if price moves > 1/leverage,
                # one leg of the carry pair gets liquidated (isolated margin)
                if coin_prices is not None and liq_threshold is not None:
                    trade_prices = coin_prices.loc[entry_sig.timestamp:exit_sig.timestamp]
                    if len(trade_prices) > 0:
                        entry_price = trade_prices.iloc[0]
                        abs_returns = (trade_prices / entry_price - 1).abs()
                        breached = abs_returns > liq_threshold
                        if breached.any():
                            liq_ts = abs_returns[breached].index[0]
                            window = window[window["timestamp"] <= liq_ts]

                holding_epochs = max(len(window) - 1, 1)

                # PnL per margin = (carry - fees) × leverage
                carry_income = window["spread"].sum()
                long_fee = VENUE_FEES.get(params.long_venue, 0.0005)
                short_fee = VENUE_FEES.get(params.short_venue, 0.0005)
                round_trip_cost = 2 * (long_fee + short_fee)  # open + close
                trade_pnl = (carry_income - round_trip_cost) * leverage

                trades.append(
                    {
                        "entry_ts": entry_sig.timestamp,
                        "exit_ts": exit_sig.timestamp,
                        "holding_epochs": holding_epochs,
                        "pnl": trade_pnl,
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
        trade_pnls=pnl_list,
    )


def grid_search_params(
    spreads_df: pd.DataFrame,
    coin: str,
    long_venue: str,
    short_venue: str,
    entry_spreads: list[float] | None = None,
    exit_spreads: list[float] | None = None,
    max_holding_epochs_list: list[int] | None = None,
    progress: bool = False,
) -> list[GridSearchResult]:
    """Grid search over entry/exit thresholds for a (coin, venue_pair).

    Default grids:
      entry_spreads:  [0.15, 0.20, 0.25, 0.3, 0.35] annualized
      exit_spreads:   [0.01, 0.02, 0.03, 0.05] annualized
      max_holding:    [15, 30, 45, 60] epochs (5–20 days)

    Returns list of GridSearchResult sorted by Sharpe descending.
    """
    if entry_spreads is None:
        entry_spreads = [0.15, 0.20, 0.25, 0.3, 0.35]
    if exit_spreads is None:
        exit_spreads = [0.01, 0.02, 0.03, 0.05]
    if max_holding_epochs_list is None:
        max_holding_epochs_list = [15, 30, 45, 60]

    # Build parameter combos
    combos = [
        (entry, exit_, hold)
        for entry in entry_spreads
        for exit_ in exit_spreads
        if exit_ < entry
        for hold in max_holding_epochs_list
    ]

    iterator = combos
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(combos, desc=f"{coin} {long_venue}→{short_venue}", leave=False)
        except ImportError:
            pass

    results: list[GridSearchResult] = []
    for entry, exit_, hold in iterator:
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


def _extract_trade_windows(
    spreads_df: pd.DataFrame,
    params: CarryParams,
) -> list[dict]:
    """Extract trade windows (entry/exit times + per-epoch spreads) for a pair.

    Separates signal generation from PnL evaluation so that leverage can be
    swept without re-running simulate_carry.
    """
    signals = simulate_carry(spreads_df, params)
    windows: list[dict] = []
    i = 0
    while i < len(signals):
        if signals[i].action == "enter":
            if i + 1 < len(signals) and signals[i + 1].action == "exit":
                entry_sig, exit_sig = signals[i], signals[i + 1]
                mask = (
                    (spreads_df["coin"] == params.coin)
                    & (spreads_df["long_venue"] == params.long_venue)
                    & (spreads_df["short_venue"] == params.short_venue)
                    & (spreads_df["timestamp"] >= entry_sig.timestamp)
                    & (spreads_df["timestamp"] <= exit_sig.timestamp)
                )
                window = spreads_df.loc[mask].sort_values("timestamp")
                windows.append({
                    "entry_ts": entry_sig.timestamp,
                    "exit_ts": exit_sig.timestamp,
                    "timestamps": window["timestamp"].values,
                    "spreads": window["spread"].values,
                    "long_venue": params.long_venue,
                    "short_venue": params.short_venue,
                })
                i += 2
            else:
                i += 1
        else:
            i += 1
    return windows


def _score_windows_at_leverage(
    windows: list[dict],
    leverage: float,
    coin_prices: pd.Series | None,
) -> tuple[list[float], list[float]]:
    """Score pre-computed trade windows at a given leverage.

    Returns (pnl_list, holding_epochs_list).
    """
    import numpy as np

    liq_threshold = (1.0 / leverage - MAINTENANCE_RATE) if leverage > 1.0 else None
    pnls: list[float] = []
    holds: list[float] = []

    for w in windows:
        spreads_arr = w["spreads"]
        ts_arr = w["timestamps"]

        # Check for liquidation
        n_epochs = len(spreads_arr)
        if coin_prices is not None and liq_threshold is not None and n_epochs > 0:
            trade_prices = coin_prices.loc[w["entry_ts"]:w["exit_ts"]]
            if len(trade_prices) > 0:
                entry_price = trade_prices.iloc[0]
                abs_returns = (trade_prices / entry_price - 1).abs()
                breached = abs_returns > liq_threshold
                if breached.any():
                    liq_ts = abs_returns[breached].index[0]
                    keep = ts_arr <= np.datetime64(liq_ts)
                    spreads_arr = spreads_arr[keep]
                    n_epochs = len(spreads_arr)

        carry_income = float(spreads_arr.sum()) if n_epochs > 0 else 0.0
        long_fee = VENUE_FEES.get(w["long_venue"], 0.0005)
        short_fee = VENUE_FEES.get(w["short_venue"], 0.0005)
        round_trip_cost = 2 * (long_fee + short_fee)
        trade_pnl = (carry_income - round_trip_cost) * leverage

        pnls.append(trade_pnl)
        holds.append(float(max(n_epochs - 1, 1)))

    return pnls, holds


def grid_search_per_coin(
    spreads_df: pd.DataFrame,
    coin: str,
    entry_spreads: list[float] | None = None,
    exit_spreads: list[float] | None = None,
    max_holding_epochs_list: list[int] | None = None,
    leverage_grid: list[float] | None = None,
    candles_df: pd.DataFrame | None = None,
    min_trades: int = 3,
    progress: bool = False,
) -> list[PooledGridResult]:
    """Grid search pooled across all venue pairs for a single coin.

    Instead of fitting separate parameters per (coin, venue-pair), evaluates
    each candidate (entry, exit, max_hold, leverage) across ALL pairs for the
    coin simultaneously and computes a pooled Sharpe from the combined trade PnL
    distribution. This prevents overfitting to pair-specific noise.

    When candles_df is provided, trades are checked for liquidation: if the
    price moves more than 1/leverage from entry, the carry trade is truncated
    at that point (isolated-margin liquidation of one leg).

    Returns list of PooledGridResult sorted by pooled Sharpe descending.
    """
    import numpy as np

    if entry_spreads is None:
        entry_spreads = [0.15, 0.20, 0.25, 0.3, 0.35]
    if exit_spreads is None:
        exit_spreads = [0.01, 0.02, 0.03, 0.05]
    if max_holding_epochs_list is None:
        max_holding_epochs_list = [15, 30, 45, 60]
    if leverage_grid is None:
        leverage_grid = [4.0]

    pairs = (
        spreads_df[spreads_df["coin"] == coin][["long_venue", "short_venue"]]
        .drop_duplicates()
        .values.tolist()
    )
    if not pairs:
        return []

    # Build price series for liquidation checks (once per coin)
    coin_prices: pd.Series | None = None
    if candles_df is not None:
        col = "c" if "c" in candles_df.columns else "close"
        coin_prices = (
            candles_df[candles_df["coin"] == coin]
            .groupby("timestamp")[col].mean()
            .sort_index()
        )
        if coin_prices.empty:
            coin_prices = None

    # Signal combos (leverage-independent)
    signal_combos = [
        (entry, exit_, hold)
        for entry in entry_spreads
        for exit_ in exit_spreads
        if exit_ < entry
        for hold in max_holding_epochs_list
    ]

    # Pre-compute trade windows for each (entry, exit, hold, pair)
    # Key: (entry, exit, hold) -> list of trade windows across all pairs
    windows_cache: dict[tuple, list[dict]] = {}
    for entry, exit_, hold in signal_combos:
        all_windows: list[dict] = []
        for lv, sv in pairs:
            p = CarryParams(
                coin=coin, long_venue=lv, short_venue=sv,
                entry_spread=entry, exit_spread=exit_,
                max_holding_epochs=hold,
            )
            all_windows.extend(_extract_trade_windows(spreads_df, p))
        windows_cache[(entry, exit_, hold)] = all_windows

    # Now sweep leverage: re-score cached windows at each leverage
    full_combos = [
        (entry, exit_, hold, lev)
        for entry, exit_, hold in signal_combos
        for lev in leverage_grid
    ]

    iterator = full_combos
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(full_combos, desc=f"{coin}", leave=False)
        except ImportError:
            pass

    results: list[PooledGridResult] = []
    for entry, exit_, hold, lev in iterator:
        windows = windows_cache[(entry, exit_, hold)]
        if not windows:
            continue

        all_pnls, all_holds = _score_windows_at_leverage(windows, lev, coin_prices)

        n = len(all_pnls)
        if n < min_trades:
            continue

        pnl_arr = np.array(all_pnls)
        avg_hold = float(np.mean(all_holds))

        if pnl_arr.std() > 0 and n > 1:
            per_trade_sharpe = pnl_arr.mean() / pnl_arr.std()
            trades_per_year = 1095.0 / avg_hold if avg_hold > 0 else 0.0
            sharpe = float(per_trade_sharpe * np.sqrt(trades_per_year))
        else:
            sharpe = 0.0

        results.append(PooledGridResult(
            params=PooledCarryParams(
                coin=coin, entry_spread=entry,
                exit_spread=exit_, max_holding_epochs=hold,
                leverage=lev,
            ),
            total_trades=n,
            total_pnl=float(pnl_arr.sum()),
            pooled_sharpe=sharpe,
            avg_holding_epochs=avg_hold,
            avg_win_rate=float(np.mean(pnl_arr > 0)),
        ))

    results.sort(key=lambda r: r.pooled_sharpe, reverse=True)
    return results
