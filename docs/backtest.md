# backtest — Engine, Costs, and Performance

---

## engine.py

Owner: Jean Mauratille

Event-driven portfolio simulation. Processes funding epochs sequentially:
1. Apply funding payments to open positions
2. Mark-to-market all positions
3. Generate new signals (carry + cascade)
4. Compute position targets via allocation
5. Execute trades (with or without transaction costs)
6. Enforce risk limits

### Dataclasses

```python
@dataclass
class Trade:
    timestamp: pd.Timestamp
    coin: str
    venue: str
    side: str           # "long" or "short"
    notional_usd: float
    price: float
    fee_usd: float      # transaction cost (spread + impact)
    strategy: str       # "carry" or "cascade"
```

```python
@dataclass
class PortfolioState:
    timestamp: pd.Timestamp
    nav: float
    gross_leverage: float
    net_delta_pct: float
    n_positions: int
    positions: dict              # {(coin, venue): notional_usd}
    cumulative_funding: float
    cumulative_trading_costs: float
```

```python
@dataclass
class BacktestResult:
    trades: list[Trade]
    portfolio_states: list[PortfolioState]
    funding_payments: pd.DataFrame | None    # detailed funding log

    # Properties / methods
    trade_count: int
    def trades_df(self) -> pd.DataFrame
    def nav_series(self) -> pd.Series        # timestamp -> NAV
```

### Functions

```python
def run_backtest(
    funding_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    oi_df: pd.DataFrame | None = None,
    initial_nav: float = 1_000_000.0,
    cost_model: TransactionCostModel | None = None,
    carry_weight: float = 0.70,
    cascade_weight: float = 0.30,
) -> BacktestResult
```
Run the full backtest. **Status: not implemented (stub).**

Must be run twice:
- (A) `cost_model=None` — zero transaction costs
- (B) `cost_model=TransactionCostModel(...)` — calibrated Almgren-Chriss costs

Inputs:
- `funding_df`: `[timestamp, venue, coin, funding_rate]`
- `candles_df`: mark prices `[timestamp, venue, coin, o, h, l, c, v]`
- `oi_df`: open interest (optional, for cascade position sizing)

---

## costs.py

Owner: Jean Mauratille

Almgren-Chriss linear impact transaction cost model.

```
Permanent impact:  g(v) = γ v
Temporary impact:  h(v) = ε sgn(v) + (η/τ) n

Total cost of trading n units in one period:
  n · h(n/τ) = ε|n| + (η/τ) n²
```

Reference: Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions".

### Dataclass

```python
@dataclass
class TransactionCostModel:
    epsilon: float = 0.0003   # half spread (3 bps default)
    eta: float = 0.0001       # temporary impact coefficient
    gamma: float = 0.00001    # permanent impact coefficient
```

### Methods

```python
def fixed_cost(self, notional_usd: float) -> float
# Fixed cost component (spread crossing): ε × |notional|

def temporary_impact(self, notional_usd: float, tau: float = 1.0) -> float
# Temporary impact cost: (η/τ) × notional²

def permanent_impact(self, notional_usd: float) -> float
# Permanent impact cost: ½ γ × notional²

def total_cost(self, notional_usd: float, tau: float = 1.0) -> float
# Total execution cost: fixed + temporary (always positive)

def implementation_shortfall(
    self,
    trade_notionals: list[float],
    tau: float = 1.0,
) -> float
# E[IS] = ½γX² + ε Σ|nk| + (η̃/τ) Σnk²
# where η̃ = η - ½γτ
```

### Venue Fee Schedules

Per-venue taker fees for perp futures (base tier):

```python
VENUE_FEES: dict[str, float] = {
    "hyperliquid": 0.00045,  # 4.5 bps
    "lighter":     0.0,      # zero-fee DEX
    "binance":     0.0005,   # 5 bps (USDM futures VIP0)
    "bybit":       0.00055,  # 5.5 bps
    "okx":         0.0005,   # 5 bps
    "kraken":      0.0005,   # 5 bps
    "dydx":        0.0005,   # 5 bps
}
```

```python
def make_cost_model(venue: str) -> TransactionCostModel
```
Create a `TransactionCostModel` with venue-specific spread estimates. Falls back to 5 bps for unknown venues.

---

## performance.py

Owner: Jean Mauratille

Performance analytics — Sharpe, drawdown, trade statistics.

### Dataclass

```python
@dataclass
class PerformanceStats:
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
```

### Functions

```python
def compute_performance(
    nav_series: pd.Series,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> PerformanceStats
```
Compute comprehensive performance statistics. **Status: not implemented (stub).**

```python
def pnl_decomposition(result: object) -> pd.DataFrame
```
Decompose PnL into components: carry, cascade, and costs.
Returns: `[timestamp, carry_pnl, cascade_pnl, funding_pnl, trading_costs, net_pnl]`.
**Status: not implemented (stub).**
