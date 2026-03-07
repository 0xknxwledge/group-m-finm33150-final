"""Backtest engine — event-driven portfolio simulation.

Owner: Jean Mauratille

Processes funding epochs sequentially:
  1. Mark-to-market all positions using current prices
  2. Apply funding payments to open positions
  3. Check liquidations (isolated margin per position)
  4. Process pre-computed carry + cascade signals (open/close positions)
  5. Record portfolio state

Produces a trade log and a time series of portfolio state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from funding_the_fall.backtest.costs import VENUE_FEES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

MAINTENANCE_RATE = 0.005  # 0.5% of notional, consistent with cascade.py


@dataclass
class OpenPosition:
    """A live position tracked by the engine (isolated margin)."""

    coin: str
    venue: str
    side: str  # "long" or "short"
    entry_price: float
    notional: float  # abs notional at entry
    collateral: float  # margin posted
    leverage: float
    strategy: str
    entry_timestamp: pd.Timestamp
    pair_id: int = 0  # links long+short legs of a carry trade
    cumulative_funding: float = 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        ret = (current_price - self.entry_price) / self.entry_price
        if self.side == "short":
            ret = -ret
        return self.notional * ret

    def equity(self, current_price: float) -> float:
        return self.collateral + self.cumulative_funding + self.unrealized_pnl(current_price)

    def is_liquidated(self, current_price: float) -> bool:
        return self.equity(current_price) < MAINTENANCE_RATE * self.notional


@dataclass
class Trade:
    """A single executed trade."""

    timestamp: pd.Timestamp
    coin: str
    venue: str
    side: str
    notional_usd: float
    price: float
    fee_usd: float
    strategy: str


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time."""

    timestamp: pd.Timestamp
    nav: float
    cash: float
    gross_leverage: float
    net_delta_pct: float
    n_positions: int
    cumulative_funding: float
    cumulative_trading_costs: float
    n_liquidations: int = 0


@dataclass
class BacktestResult:
    """Complete backtest output."""

    trades: list[Trade] = field(default_factory=list)
    portfolio_states: list[PortfolioState] = field(default_factory=list)

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def nav_series(self) -> pd.Series:
        return pd.Series({s.timestamp: s.nav for s in self.portfolio_states})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_price_lookup(candles_df: pd.DataFrame) -> dict[tuple, float]:
    """Build {(timestamp, coin): price} from candles.

    Uses mean close across venues as the reference price.
    """
    col = "c" if "c" in candles_df.columns else "close"
    grouped = candles_df.groupby(["timestamp", "coin"])[col].mean()
    return grouped.to_dict()


def _build_funding_lookup(funding_df: pd.DataFrame) -> dict[tuple, float]:
    """Build {(timestamp, venue, coin): rate} from funding data."""
    idx = funding_df.set_index(["timestamp", "venue", "coin"])["funding_rate"]
    return idx.to_dict()


def _compute_nav(
    cash: float, positions: list[OpenPosition], prices: dict[str, float],
) -> float:
    """NAV = cash + sum of equity across all open positions."""
    total_equity = sum(
        pos.equity(prices.get(pos.coin, pos.entry_price))
        for pos in positions
    )
    return cash + total_equity


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_backtest(
    carry_signals: list,
    funding_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    initial_nav: float = 1_000_000.0,
    carry_leverage: float = 4.0,
    carry_budget_pct: float = 0.85,
    cascade_signals: list | None = None,
    cascade_leverage: float = 1.5,
    cascade_budget_pct: float = 0.15,
    fee_multiplier: float = 1.0,
) -> BacktestResult:
    """Run the full backtest with position-level margin tracking.

    Parameters
    ----------
    carry_signals : list of CarrySignal
        Pre-computed entry/exit signals (from simulate_carry with best params).
    funding_df : DataFrame
        Raw funding rates [timestamp, venue, coin, funding_rate].
    candles_df : DataFrame
        Price candles [timestamp, venue, coin, c, ...].
    initial_nav : float
        Starting capital in USD.
    carry_leverage : float
        Leverage per carry leg (each leg is isolated margin).
    carry_budget_pct : float
        Max fraction of NAV allocated as carry margin.
    cascade_signals : list of CascadeSignal, optional
        Pre-computed cascade short entry/exit signals.
    cascade_leverage : float
        Leverage for cascade shorts.
    cascade_budget_pct : float
        Max fraction of NAV allocated as cascade margin.

    Position model
    --------------
    Each carry trade opens two isolated-margin positions (long + short).
    Each cascade trade opens a single short position.
    At each epoch: funding is applied, then liquidation is checked.
    If one leg of a carry pair is liquidated, the other is force-closed.
    """
    # Pre-build lookups
    price_lookup = _build_price_lookup(candles_df)
    funding_lookup = _build_funding_lookup(funding_df)

    # Signal indices: timestamp -> list of signals
    carry_index: dict[pd.Timestamp, list] = defaultdict(list)
    for sig in carry_signals:
        carry_index[sig.timestamp].append(sig)

    cascade_index: dict[pd.Timestamp, list] = defaultdict(list)
    for sig in (cascade_signals or []):
        cascade_index[sig.timestamp].append(sig)

    # Sorted unique epochs from price data
    epochs = sorted(candles_df["timestamp"].unique())

    # State
    cash = initial_nav
    positions: list[OpenPosition] = []
    all_trades: list[Trade] = []
    states: list[PortfolioState] = []
    cumulative_funding = 0.0
    cumulative_costs = 0.0
    total_liquidations = 0
    next_pair_id = 0

    for ts in epochs:
        ts = pd.Timestamp(ts)

        # Current prices for all coins
        prices: dict[str, float] = {}
        for key, price in price_lookup.items():
            if key[0] == ts:
                prices[key[1]] = price
        if not prices:
            continue

        # --- 1. Apply funding payments ------------------------------------
        for pos in positions:
            rate = funding_lookup.get((ts, pos.venue, pos.coin), 0.0)
            payment = pos.notional * rate * (-1 if pos.side == "long" else 1)
            pos.cumulative_funding += payment
            cumulative_funding += payment

        # --- 2. Check liquidations ----------------------------------------
        liquidated_pair_ids: set[int] = set()
        surviving: list[OpenPosition] = []

        for pos in positions:
            price = prices.get(pos.coin, pos.entry_price)
            if pos.is_liquidated(price):
                equity = max(pos.equity(price), 0.0)
                cash += equity
                cumulative_costs += pos.collateral - equity
                total_liquidations += 1
                if pos.strategy == "carry":
                    liquidated_pair_ids.add(pos.pair_id)
                all_trades.append(Trade(
                    timestamp=ts, coin=pos.coin, venue=pos.venue,
                    side="close_" + pos.side, notional_usd=pos.notional,
                    price=price, fee_usd=0.0,
                    strategy=pos.strategy + "_liquidation",
                ))
            else:
                surviving.append(pos)

        # Force-close the other leg of any liquidated carry pair
        if liquidated_pair_ids:
            still_alive: list[OpenPosition] = []
            for pos in surviving:
                if pos.pair_id in liquidated_pair_ids and pos.strategy == "carry":
                    price = prices.get(pos.coin, pos.entry_price)
                    equity = pos.equity(price)
                    exit_fee = VENUE_FEES.get(pos.venue, 0.0005) * pos.notional * fee_multiplier
                    cash += equity - exit_fee
                    cumulative_costs += exit_fee
                    all_trades.append(Trade(
                        timestamp=ts, coin=pos.coin, venue=pos.venue,
                        side="close_" + pos.side, notional_usd=pos.notional,
                        price=price, fee_usd=exit_fee,
                        strategy="carry_pair_unwind",
                    ))
                else:
                    still_alive.append(pos)
            surviving = still_alive

        positions = surviving

        # --- 3. Process carry signals -------------------------------------
        for sig in carry_index.get(ts, []):
            if sig.action == "enter":
                price = prices.get(sig.coin)
                if price is None:
                    continue
                nav = _compute_nav(cash, positions, prices)
                used_carry = sum(p.collateral for p in positions if p.strategy == "carry")
                remaining = max(carry_budget_pct * nav - used_carry, 0)
                pair_margin = min(remaining, cash)
                leg_collateral = pair_margin / 2
                if leg_collateral <= 0:
                    continue
                leg_notional = leg_collateral * carry_leverage

                long_fee = VENUE_FEES.get(sig.long_venue, 0.0005) * leg_notional * fee_multiplier
                short_fee = VENUE_FEES.get(sig.short_venue, 0.0005) * leg_notional * fee_multiplier
                entry_cost = long_fee + short_fee
                if entry_cost >= pair_margin:
                    continue

                pid = next_pair_id
                next_pair_id += 1

                positions.append(OpenPosition(
                    coin=sig.coin, venue=sig.long_venue, side="long",
                    entry_price=price, notional=leg_notional,
                    collateral=leg_collateral, leverage=carry_leverage,
                    strategy="carry", entry_timestamp=ts, pair_id=pid,
                ))
                positions.append(OpenPosition(
                    coin=sig.coin, venue=sig.short_venue, side="short",
                    entry_price=price, notional=leg_notional,
                    collateral=leg_collateral, leverage=carry_leverage,
                    strategy="carry", entry_timestamp=ts, pair_id=pid,
                ))
                cash -= pair_margin + entry_cost
                cumulative_costs += entry_cost

                all_trades.append(Trade(ts, sig.coin, sig.long_venue, "long",
                                        leg_notional, price, long_fee, "carry"))
                all_trades.append(Trade(ts, sig.coin, sig.short_venue, "short",
                                        leg_notional, price, short_fee, "carry"))

            elif sig.action == "exit":
                to_close, to_keep = [], []
                for pos in positions:
                    if (pos.coin == sig.coin and pos.strategy == "carry"
                            and pos.venue in (sig.long_venue, sig.short_venue)):
                        to_close.append(pos)
                    else:
                        to_keep.append(pos)
                for pos in to_close:
                    price = prices.get(pos.coin, pos.entry_price)
                    equity = pos.equity(price)
                    exit_fee = VENUE_FEES.get(pos.venue, 0.0005) * pos.notional * fee_multiplier
                    cash += equity - exit_fee
                    cumulative_costs += exit_fee
                    all_trades.append(Trade(ts, pos.coin, pos.venue,
                                            "close_" + pos.side, pos.notional,
                                            price, exit_fee, "carry"))
                positions = to_keep

        # --- 4. Process cascade signals -----------------------------------
        for sig in cascade_index.get(ts, []):
            if sig.action == "enter":
                price = prices.get(sig.coin)
                if price is None:
                    continue
                nav = _compute_nav(cash, positions, prices)
                used_cascade = sum(p.collateral for p in positions if p.strategy == "cascade")
                remaining = max(cascade_budget_pct * nav - used_cascade, 0)
                collateral = min(remaining, cash)
                if collateral <= 0:
                    continue
                notional = collateral * cascade_leverage

                fee = VENUE_FEES.get(sig.venue, 0.0005) * notional * fee_multiplier
                if fee >= collateral:
                    continue

                positions.append(OpenPosition(
                    coin=sig.coin, venue=sig.venue, side="short",
                    entry_price=price, notional=notional,
                    collateral=collateral, leverage=cascade_leverage,
                    strategy="cascade", entry_timestamp=ts,
                ))
                cash -= collateral + fee
                cumulative_costs += fee
                all_trades.append(Trade(ts, sig.coin, sig.venue, "short",
                                        notional, price, fee, "cascade"))

            elif sig.action == "exit":
                to_close, to_keep = [], []
                for pos in positions:
                    if pos.coin == sig.coin and pos.strategy == "cascade":
                        to_close.append(pos)
                    else:
                        to_keep.append(pos)
                for pos in to_close:
                    price = prices.get(pos.coin, pos.entry_price)
                    equity = pos.equity(price)
                    exit_fee = VENUE_FEES.get(pos.venue, 0.0005) * pos.notional * fee_multiplier
                    cash += equity - exit_fee
                    cumulative_costs += exit_fee
                    all_trades.append(Trade(ts, pos.coin, pos.venue,
                                            "close_short", pos.notional,
                                            price, exit_fee, "cascade"))
                positions = to_keep

        # --- 5. Record portfolio state ------------------------------------
        nav = _compute_nav(cash, positions, prices)
        gross = sum(p.notional for p in positions)
        net_delta = sum(
            p.notional * (1 if p.side == "long" else -1) for p in positions
        )
        states.append(PortfolioState(
            timestamp=ts, nav=nav, cash=cash,
            gross_leverage=gross / nav if nav > 0 else 0.0,
            net_delta_pct=net_delta / nav if nav > 0 else 0.0,
            n_positions=len(positions),
            cumulative_funding=cumulative_funding,
            cumulative_trading_costs=cumulative_costs,
            n_liquidations=total_liquidations,
        ))

    return BacktestResult(trades=all_trades, portfolio_states=states)
