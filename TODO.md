# TODO — Group M Final Project

**Funding the Fall: Cross-Venue Carry and Cascade Alpha in Crypto Perpetuals**

FINM 33150 — Winter 2026

---

## Team & Ownership

| Member | Module | Path |
|--------|--------|------|
| **Antonio Braz** | Data Pipeline | `src/funding_the_fall/data/` |
| **John Beecher** | Models (Merton + Cascade) | `src/funding_the_fall/models/` |
| **Jean-Luc Choiseul** | Strategy (Carry + Allocation) | `src/funding_the_fall/strategy/` |
| **Jean Mauratille** | Backtester + Transaction Costs | `src/funding_the_fall/backtest/` |

**Dependency chain:** Antonio → John + Jean-Luc → Jean

Everyone writes their own section of the technical notebook and pitchbook.

---

## Project Requirements (from QuantTradingProject.pdf)

- [x] Non-standard data source (not equity OHLC/VWAP/volume) — crypto funding rates + on-chain liquidation data
- [ ] Hold 5+ distinct assets simultaneously — BTC, ETH, SOL, HYPE, DOGE
- [ ] 40+ trades over simulation period, not excessively clustered
- [ ] Leverage with documented risk controls
- [ ] Analysis **(A) with zero txn costs** and **(B) with realistic costs**
- [ ] Code in `src/` library, notebook imports from it (minimal inline code)
- [ ] Pitchbook PDF — non-technical, no jargon, self-explanatory
- [ ] Technical notebook (.ipynb) — academic paper style
- [ ] 2+ academic paper PDFs included
- [ ] Zip submission named with last names (e.g. `Beecher_Braz_Choiseul_Mauratille.zip`)

---

## Phase 1 — Data Pipeline (Antonio)

**Goal:** Populate `data/` with parquet files that everyone else consumes.

### 1a. Funding Rates (`data/fetchers.py`)
- [x] **Hyperliquid** — via 0xArchive (fallback: direct API)
- [x] **Lighter** — via 0xArchive
- [x] **OKX** — paginated, 90-day history
- [x] **Kraken** — hourly rates aggregated to 8h
- [x] **Binance** — paginated, VPN required
- [x] **Bybit** — cursor-paginated, VPN required
- [x] **dYdX** — paginated via `effectiveAtBeforeOrAt`
- [x] **Predicted fundings** — `predictedFundings` endpoint

### 1b. Candles / Mark Prices (`data/fetchers.py`)
- [x] All 7 venues, all 5 tokens, configurable interval

### 1c. Open Interest (`data/fetchers.py`)
- [x] **Hyperliquid + Lighter** — historical via 0xArchive
- [x] **OKX, Kraken, Binance, Bybit, dYdX** — snapshots via live APIs

### 1d. Liquidation Events (`data/fetchers.py`)
- [x] **0xArchive** — Hyperliquid liquidations (May 2025+)
- [x] **OKX** — recent liquidation orders
- [x] **Binance** — forced liquidation orders (VPN)

### 1e. Lending Positions (`data/lending.py`) — for cascade model
- [x] **HyperLend event replay** — `scan_hyperlend_events()` + `replay_positions()`
- [x] **HyperLend current snapshot** — `fetch_hyperlend_positions()`
- [x] **Reserve price history** — `fetch_reserve_prices()` via 0xArchive (ETH) + stablecoin pegs
- [ ] **Reserve token discovery** — populate RESERVES dict with full addresses, decimals, LTs from first scan
- [x] **DeFi Llama** — `fetch_tvl_history()` for HyperLend TVL plots

### 1f. Infrastructure
- [x] **Unified schema** — all polars DataFrames follow column specs in `fetchers.py`
- [x] **Storage layer** — polars loaders in `storage.py`, pandas compat kept
- [ ] **Data validation** — timestamps aligned to 8h funding epochs, no gaps, UTC
- [ ] **Initial data pull** — `scripts/pull_data.py`, see instructions below

**Tokens:** BTC, ETH, SOL, HYPE, DOGE (all 5 × all 7 venues)
**Lookback:** 90 days minimum (0xArchive has data from Apr 2023)

### Data Sources

| Source | Data | Auth | Notes |
|--------|------|------|-------|
| 0xArchive | Funding, candles, OI, liquidations, prices (HL + Lighter) | API key | Historical from Apr 2023 |
| OKX | Funding, candles, OI, liquidations | None | Free, works from US |
| Kraken Futures | Funding (hourly→8h), candles, OI | None | Free, works from US |
| Binance | Funding, candles, OI, liquidations | None | VPN required |
| Bybit | Funding, candles, OI | None | VPN required |
| dYdX | Funding, candles, OI | None | No auth needed |
| Hyperliquid | Live meta, orderbook, predicted fundings | None | Direct API fallback |
| HyperLend | Lending positions via event replay | None | HyperEVM RPC |
| DeFi Llama | TVL history | None | Free |

### Antonio's remaining tasks

1. **Set `OXARCHIVE_API_KEY` env var** and verify access.

2. **Quick smoke test** — confirm one venue works:
   ```bash
   PYTHONPATH=src python scripts/pull_data.py --venues hyperliquid --coins BTC --days 7
   ```

3. **Full deep pull** (VPN on for Binance/Bybit):
   ```bash
   PYTHONPATH=src python scripts/pull_data.py
   ```
   This pulls 0xArchive back to Apr 2023 (~1060 days), Kraken full history, OKX 90 days, Binance/Bybit/dYdX max available. Candles are the biggest dataset — use `--skip-candles` for a faster first pass on funding/OI/liquidations only.

4. **Validate output** — check the coverage matrix printed at the end. Every coin×venue cell should have rows. Gaps in VPN venues are OK for now.

5. **Reserve prices** — the script also pulls `reserve_prices.parquet` for the cascade model. Verify ETH prices look reasonable.

6. **Commit parquet files** are gitignored — push the script, not the data.

---

## Phase 2 — Models (John)

**Goal:** Calibrate jump models, compare Merton vs Kou, build cascade simulator.

- [ ] **Merton jump-diffusion** (`models/merton.py`)
  - Heuristic stage: iterative 3σ-filtering
  - MLE stage: L-BFGS-B on full Merton log-likelihood
  - Symmetric log-normal jumps — baseline model
- [ ] **Kou double-exponential jump-diffusion** (`models/kou.py`)
  - Same 2-stage calibration, but with asymmetric double-exponential jumps
  - p = prob jump is up, η₁ = up-jump rate, η₂ = down-jump rate
  - Captures crypto's negative skew (crash-heavy left tail)
  - Memoryless property useful for cascade threshold analysis
- [ ] **Model comparison** (`models/compare.py`)
  - Calibrate both on all 5 tokens
  - Compare via AIC/BIC (BIC penalizes Kou's extra parameter)
  - Show GBM is rejected (Jarque-Bera p ≈ 0)
  - QQ plots and tail probability comparison in notebook
  - Expect Kou preferred for assets with skewed returns (HYPE, SOL)
- [ ] **Cascade simulator** (`models/cascade.py`)
  - Model: perp liquidations → spot selling → lending liquidations → repeat
  - Price impact via square-root law (Jusselin-Rosenbaum, consistent with Almgren-Chriss)
  - Parameterize with real OI + liquidation data from Antonio's pipeline
  - Compute amplification curve A(δ)
- [ ] **Cascade risk signal**
  - Expose `cascade_risk_signal()` returning risk_score, critical_shock, signal bool
  - Jean-Luc's allocation code consumes this

### Why Kou over Merton (notebook justification)
- Merton's log-normal jumps are symmetric — same probability of 10% up vs 10% down
- Kou's double-exponential allows heavier left tail (more frequent/larger crashes)
- Crypto empirically has negative skew — liquidation cascades are downside events
- Kou's memoryless property: P(J > a+b | J > a) = P(J > b), analytically convenient
- CGMY rejected: infinite-activity small jumps aren't relevant; we care about discrete large liquidation events. Also much harder to calibrate (FFT on characteristic function) and explain in defense.

---

## Phase 3 — Strategy (Jean-Luc)

**Goal:** Turn data into actionable trade signals with position sizing.

- [ ] **Funding spread computation** (`strategy/carry.py`)
  - For each coin × epoch: rank all venue pairs by spread
  - Annualize spreads (8h rate × 3 × 365)
- [ ] **Grid search over entry/exit rules** (`strategy/carry.py`)
  - Grid search per (coin, venue_pair)
  - Entry spread grid: [5%, 8%, 10%, 12%, 15%, 20%] annualized
  - Exit spread grid: [1%, 2%, 3%, 5%] annualized
  - Max holding period grid: [15, 30, 45, 60] epochs (5–20 days)
  - Evaluate each combo: trade count, carry PnL, Sharpe, win rate
  - Select best params per pair (require min 5 trades to avoid overfit)
  - Report grid search results as a table in the notebook
- [ ] **Combined allocation** (`strategy/allocation.py`)
  - 70% NAV → carry, 30% NAV → cascade
  - Dynamic: when cascade risk signal fires, shift weight toward cascade
  - Position sizing proportional to spread attractiveness
- [ ] **Risk limits enforcement**
  - Max gross leverage: 5×
  - Max net delta: 10% NAV
  - Max single-exchange exposure: 40% NAV

---

## Phase 4 — Backtester (Jean)

**Goal:** Simulate the strategy end-to-end, produce required performance analysis.

- [ ] **Portfolio engine** (`engine.py`)
  - Event loop over 8h funding epochs
  - Apply funding payments, mark-to-market, execute trades
  - Track positions per (coin, venue), NAV over time
  - Produce trade log (must have 40+ trades)
- [ ] **Transaction cost model** (`costs.py`) — Almgren-Chriss framework
  - Permanent impact: g(v) = γv
  - Temporary impact: h(v) = ε sgn(v) + (η/τ)n
  - Implementation shortfall: E[IS] = ½γX² + ε Σ|nₖ| + (η̃/τ) Σnₖ²
  - Calibrate γ, η, ε from orderbook depth / venue fee schedules
  - Per-venue taker fees already in `costs.py`
- [ ] **Run two scenarios**
  - (A) `cost_model=None` — zero transaction costs
  - (B) `cost_model=TransactionCostModel(...)` — calibrated Almgren-Chriss
- [ ] **Performance analytics** (`performance.py`)
  - Sharpe ratio, max drawdown, Calmar ratio
  - Win rate, avg trade PnL
  - PnL decomposition: carry vs cascade vs costs
  - Trade clustering analysis (prove 40+ trades aren't clustered)

---

## Phase 5 — Deliverables (All)

- [ ] **Technical notebook** (`notebooks/final.ipynb`)
  - Imports from `src/funding_the_fall/` — minimal inline code
  - Structure: Abstract → Motivation → Literature → Data → Models → Strategy → Backtest → Results → Risk → Conclusion
  - Each member writes their section
  - Non-interactive plots only (matplotlib, not plotly)
- [ ] **Pitchbook** (`pitchbook.pdf`)
  - Non-technical, no jargon
  - Performance tables, clear plots
  - Self-explanatory without presenter
  - Jean-Luc leads assembly, all contribute
- [ ] **Academic papers** (2+ PDFs in repo)
  - Almgren & Chriss (2001) — "Optimal Execution of Portfolio Transactions"
  - Aloosh & Li (2024) — "Perpetual Futures and Funding Rate Arbitrage"
  - Optionally: Alexander et al. (2023), Jusselin & Rosenbaum (2018)
- [ ] **Submission zip**: `Beecher_Braz_Choiseul_Mauratille.zip`

---

## Key Design Decisions to Document

These require a sentence or two in the notebook justifying the choice (per professor's instructions):

- Why 5× leverage cap?
- Why 70/30 carry/cascade split?
- Why these 5 tokens?
- Why Kou preferred over Merton? (show AIC/BIC comparison, asymmetric tails argument)
- Why not CGMY? (overkill for discrete large events, hard to calibrate/explain)
- Grid search results: which entry/exit thresholds won per pair, and why?
- Square-root impact vs linear temporary impact assumption
