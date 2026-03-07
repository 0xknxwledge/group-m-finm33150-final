# strategy — Carry and Allocation

---

## carry.py

Owner: Jean-Luc Choiseul

Funding rate carry strategy — signal generation and entry/exit rules.

For each token at each funding epoch, the strategy:
1. Ranks all venue pairs by funding rate spread
2. Generates entry signals when spread exceeds threshold
3. Generates exit signals when spread mean-reverts or holding period expires

Positions are delta-neutral: long the perp on the low-rate venue, short on the high-rate venue.

Entry/exit thresholds are not hardcoded — use `grid_search_params()` to find optimal thresholds per `(coin, venue_pair)`.

### Dataclasses

```python
@dataclass
class CarrySignal:
    timestamp: pd.Timestamp
    coin: str
    long_venue: str    # venue with lower funding rate (we receive)
    short_venue: str   # venue with higher funding rate (we pay less)
    spread: float      # annualized spread in decimal (e.g. 0.15 = 15%)
    action: str        # "enter" or "exit"
```

```python
@dataclass
class CarryParams:
    coin: str
    long_venue: str
    short_venue: str
    entry_spread: float        # min annualized spread to enter
    exit_spread: float         # spread level to exit
    max_holding_epochs: int    # max epochs before forced exit
```

```python
@dataclass
class GridSearchResult:
    params: CarryParams
    n_trades: int
    total_carry_pnl: float     # cumulative funding collected
    sharpe: float              # annualized Sharpe of carry PnL
    avg_holding_epochs: float
    win_rate: float
```

### Functions

```python
def compute_funding_spreads(funding_df: pd.DataFrame) -> pd.DataFrame
```
Compute pairwise funding rate spreads across venues for each coin/epoch.

- Input: `[timestamp, venue, coin, funding_rate]`
- Output: `[timestamp, coin, long_venue, short_venue, spread, spread_annualized]`

Spread = `short_venue_rate - long_venue_rate` (positive means carry profit).
Annualized = `per-period spread × 3 × 365` (3 funding periods per day).

```python
def simulate_carry(
    spreads_df: pd.DataFrame,
    params: CarryParams,
) -> list[CarrySignal]
```
Run carry strategy with given params on historical spread data. Returns full list of entry + exit signals in chronological order.

Exit conditions:
1. Spread mean-reverts below `exit_spread`
2. Holding period exceeds `max_holding_epochs`

If still in position at end of data, a forced exit is generated on the last row.

```python
def evaluate_carry(
    spreads_df: pd.DataFrame,
    params: CarryParams,
) -> GridSearchResult
```
Evaluate a single parameter set. Runs `simulate_carry`, pairs enter/exit into round-trip trades, and computes metrics.

PnL per trade = sum of raw (non-annualized) spreads collected during the holding window.
Annualized Sharpe assumes ~1095 epochs/year (8h epochs).

```python
def grid_search_params(
    spreads_df: pd.DataFrame,
    coin: str,
    long_venue: str,
    short_venue: str,
    entry_spreads: list[float] | None = None,
    exit_spreads: list[float] | None = None,
    max_holding_epochs_list: list[int] | None = None,
) -> list[GridSearchResult]
```
Grid search over entry/exit thresholds for a `(coin, venue_pair)`.

Default grids:
- `entry_spreads`: `[0.05, 0.08, 0.10, 0.12, 0.15, 0.20]` annualized
- `exit_spreads`: `[0.01, 0.02, 0.03, 0.05]` annualized
- `max_holding_epochs_list`: `[15, 30, 45, 60]` epochs (5–20 days)

Returns list sorted by Sharpe descending. Skips combinations where `exit >= entry`.

```python
def grid_search_all_pairs(
    spreads_df: pd.DataFrame,
) -> dict[tuple[str, str, str], list[GridSearchResult]]
```
Run grid search for every `(coin, long_venue, short_venue)` in the data.

```python
def select_best_params(
    grid_results: dict[tuple[str, str, str], list[GridSearchResult]],
    min_trades: int = 5,
) -> dict[tuple[str, str, str], CarryParams]
```
Select best params per pair, requiring at least `min_trades`. Picks highest Sharpe among valid results.

---

## allocation.py

Owner: Jean-Luc Choiseul

Portfolio allocation — combines carry and cascade signals into sized position targets.

**Allocation logic:**
- Carry baseline: 85% NAV, scaled down by cascade risk
- Cascade max: 15% NAV, scaled up by cascade risk score
- Remainder is cash (dry powder / risk buffer)

When cascade risk is elevated (`risk_score → 1`):
- Carry scales to ~60% NAV (reduces exposure to fragile markets)
- Cascade scales to ~15% NAV (opportunistic short on negatively-skewed returns)
- Cash rises to ~25% NAV

The cascade leg takes short positions on coins where `A(δ)` is highest. High amplification implies negatively skewed return distributions. In crypto perps, shorts typically receive funding, so the bleed cost is minimal.

### Constants

```python
CARRY_BASE_WEIGHT = 0.85
CASCADE_MAX_WEIGHT = 0.15
CARRY_RISK_DAMPING = 0.30   # carry_scale = 1 - DAMPING * risk_score
```

### Dataclass

```python
@dataclass
class PositionTarget:
    timestamp: pd.Timestamp
    coin: str
    venue: str
    side: str           # "long" or "short"
    notional_usd: float # target notional exposure
    strategy: str       # "carry" or "cascade"
```

### Functions

```python
def allocate_positions(
    carry_signals: list,
    cascade_signal: dict,
    nav: float,
    per_coin_signals: dict[str, dict] | None = None,
    deepest_venue: str = "binance",
    max_leverage: float = 5.0,
    max_single_exchange_pct: float = 0.40,
    max_net_delta_pct: float = 0.10,
) -> list[PositionTarget]
```
Convert signals into sized position targets respecting risk limits.

Parameters:
- `carry_signals`: list of `CarrySignal` (entry/exit)
- `cascade_signal`: output of `cascade_risk_signal()` (aggregate)
- `nav`: current portfolio NAV
- `per_coin_signals`: output of `per_coin_risk_signals()`; if provided, cascade shorts are weighted by per-coin `A(5%)`
- `deepest_venue`: venue for cascade shorts (deepest orderbook)

Risk controls enforced:
1. Total gross leverage ≤ `max_leverage × NAV`
2. Exposure to any single exchange ≤ `max_single_exchange_pct × NAV`
3. Net delta ≤ `max_net_delta_pct × NAV`

Carry signals contribute gross leverage but ~zero net delta (delta-neutral pairs). Cascade leg weighted by per-coin amplification; only coins with `A(5%) > 1.0` receive shorts.

```python
def _enforce_risk_limits(
    targets: list[PositionTarget],
    nav: float,
    max_leverage: float,
    max_single_exchange_pct: float,
    max_net_delta_pct: float,
) -> list[PositionTarget]
```
Scale down position targets to satisfy risk constraints. Applied in order: gross leverage cap → single-exchange concentration cap → net delta cap. For the delta cap, only the dominant side is scaled to minimize distortion.
