# Validation Plan: Jump-Diffusion Liquip Score vs Historical Liquidations

## Objective

Test whether the Merton jump-diffusion Liquip Score produces better-calibrated liquidation probability estimates than the original GBM Liquip Score when measured against actual historical liquidation events on Hyperliquid.

---

## Data Sources

All data is available through Allium Analytics (Snowflake).

### Primary: HyperCore Perpetual Liquidations

**Table: `hyperliquid.dex.trades`**

| Column | Use |
|--------|-----|
| `liquidated_user` | Non-null identifies a liquidation event |
| `liquidation_mark_price` | Exact price at liquidation trigger |
| `liquidation_method` | `"market"` (normal) vs `"backstop"` (HLP vault absorption) |
| `coin` | Asset being liquidated |
| `amount`, `price`, `usd_amount` | Position size and execution price |
| `seller_start_position` / `buyer_start_position` | Position size immediately before liquidation |
| `seller_closed_pnl` / `buyer_closed_pnl` | Realized loss |
| `timestamp` | Event time |

**Table: `hyperliquid.raw.perpetual_market_asset_contexts`**

Periodic snapshots of mark price, oracle price, funding rate, open interest, and max leverage per coin. Used to reconstruct the price environment at any historical point.

**Table: `hyperliquid.raw.misc_events`**

Ledger-level liquidation records containing `accountValue` at liquidation time and the full list of `liquidatedPositions` — gives account-level context not available in the trade stream.

### Secondary: HyperEVM Lending Liquidations (Felix / Morpho Blue)

**Table: `hyperevm.lending.liquidations`**

Each row is one on-chain liquidation with borrower address, collateral seized (token + USD), debt repaid (token + USD), market address, and block timestamp.

**Position history tables** (for reconstructing pre-liquidation state):
- `hyperevm.lending.deposits` — collateral supply events
- `hyperevm.lending.loans` — borrow events
- `hyperevm.lending.withdrawals` — collateral withdrawal events
- `hyperevm.lending.repayments` — debt repayment events

**Price data**: `hyperevm.dex.token_prices_hourly` or Allium token price API.

---

## Methodology

### Step 1: Build the Liquidation Event Dataset

```sql
-- HyperCore perp liquidations
SELECT
    liquidated_user,
    coin,
    timestamp AS liquidation_time,
    liquidation_mark_price,
    liquidation_method,
    usd_amount AS liquidation_size_usd,
    COALESCE(seller_closed_pnl, buyer_closed_pnl) AS realized_pnl
FROM hyperliquid.dex.trades
WHERE liquidated_user IS NOT NULL
  AND coin IN ('HYPE', 'BTC', 'ETH')
ORDER BY timestamp
```

```sql
-- HyperEVM lending liquidations (Felix Vanilla Markets)
SELECT
    borrower_address,
    market_address,
    token_symbol AS collateral_token,
    amount AS collateral_seized,
    usd_amount AS collateral_usd,
    repay_token_symbol AS debt_token,
    repay_amount AS debt_repaid,
    repay_usd_amount AS debt_repaid_usd,
    block_timestamp AS liquidation_time
FROM hyperevm.lending.liquidations
ORDER BY block_timestamp
```

### Step 2: Reconstruct Pre-Liquidation Positions

For each liquidated user-coin pair, reconstruct the position state at lookback windows **t - 7d, t - 14d, t - 30d** before the liquidation event.

**HyperCore perps**: Walk the `hyperliquid.dex.trades` stream backward from the liquidation timestamp to find the user's position size, entry price, and effective leverage at each lookback point. Leverage is inferred from `start_position` and the account's margin (from `misc_events` or estimated from trade history).

**HyperEVM lending**: Replay the deposit/loan/withdrawal/repayment event stream for the borrower up to each lookback timestamp to get collateral value and outstanding debt. Compute LTV = debt / collateral at that point.

### Step 3: Compute Predicted Liquip Scores

At each lookback snapshot, compute both:

1. **GBM Liquip Score**: P_GBM(liq in T days) using realized volatility from the preceding 90 days of hourly candles
2. **Merton Liquip Score**: P_Merton(liq in T days) using parameters calibrated on the same 90-day window

Where T = remaining days until the actual liquidation occurred (or a fixed horizon for the control group).

Calibration uses the two-stage approach from our implementation:
- Heuristic: iterative 3-sigma filtering to separate jumps from diffusion
- MLE: L-BFGS-B refinement of (mu, sigma, lambda, mu_J, sigma_J)

### Step 4: Build the Control Group

For each liquidated position, sample 10 non-liquidated positions in the same coin at the same timestamp with similar leverage. This controls for market conditions and leverage distribution while varying the outcome.

Selection criteria:
- Same coin, same week
- Leverage within +/- 20% of the liquidated position
- Position survived at least 30 days beyond the snapshot
- No survivorship bias: include positions that were voluntarily closed

### Step 5: Evaluation Metrics

#### 5a. Discrimination — ROC/AUC

Binary classification: did the position get liquidated within the prediction horizon?

- Compute ROC curves for GBM and Merton scores separately
- Compare AUC: higher AUC = better separation of liquidated vs surviving positions
- Stratify by leverage bucket (low/medium/high) and by coin

#### 5b. Calibration — Reliability Diagram

Bin all predicted probabilities into deciles. For each bin, compute the actual fraction that were liquidated. Plot predicted vs observed.

- A perfectly calibrated model falls on the 45-degree line
- GBM is expected to systematically underpredict at moderate leverage (based on our preliminary results showing 3-109x underestimation)
- Merton should be closer to the diagonal

Quantify calibration error with **Brier Score**:
```
BS = (1/N) * sum((predicted_i - actual_i)^2)
```

#### 5c. Log-Likelihood on Realized Outcomes

For each position with known outcome (liquidated = 1, survived = 0):
```
LL = sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
```

Compare LL_GBM vs LL_Merton. This directly measures which model assigns higher probability to the events that actually occurred.

#### 5d. Temporal Analysis

Plot the ratio P_Merton / P_GBM as a function of days-before-liquidation. We expect:
- At t-30d: large divergence (Merton captures jump risk that GBM misses)
- At t-1d: convergence (both models detect imminent liquidation from proximity to threshold)

This demonstrates when the jump-diffusion extension provides the most signal — the "early warning" regime.

---

## Expected Results

Based on preliminary findings (GBM underestimates by 3-109x at 70-86% LTV):

1. **AUC**: Merton > GBM, with the gap largest for HYPE-denominated positions (highest jump intensity, lambda = 11 jumps/day)
2. **Reliability**: GBM predicted probabilities cluster below the diagonal (systematic underestimation). Merton closer to calibrated.
3. **Temporal**: Merton provides earlier warning signals, particularly valuable for MM spread adjustment (can widen spreads days before a cascade, not hours)
4. **Backstop events**: Positions liquidated via `"backstop"` (HLP vault) likely had the most extreme price moves — expect the GBM/Merton gap to be largest for these events

---

## Known Limitations and Mitigations

| Limitation | Mitigation |
|------------|------------|
| No historical position snapshot table for perps — must reconstruct from trade stream | Start with lending liquidations (position state directly observable from events), add perps as a second phase |
| Merton parameters are calibrated on the full 90-day window, which includes the liquidation event itself | Use strictly backward-looking calibration: at each lookback point, calibrate only on data available at that time |
| Control group selection may introduce subtle bias | Sensitivity analysis: vary matching criteria (leverage tolerance, time window) and confirm results are robust |
| HyperEVM lending history may be short (Felix launched relatively recently) | Supplement with HyperCore perp liquidations, which have deeper history |
| Allium `extra_fields` may be missing for some recent trades (websocket vs API refetch gap) | Filter to trades older than 48 hours for completeness |

---

## Implementation Sequence

1. **Query and cache** — Pull all liquidation events + surrounding trade history into local parquet files
2. **Position reconstruction** — Build position state timeseries for liquidated users and control group
3. **Score computation** — Run GBM and Merton Liquip Scores at each lookback window (parallelize across positions)
4. **Evaluation** — ROC/AUC, reliability diagrams, Brier scores, temporal analysis
5. **Dashboard** — Generate validation results visualization (companion to `jump_diffusion_results.png`)

Estimated effort: 1-2 days for a clean implementation, assuming Allium query access.
