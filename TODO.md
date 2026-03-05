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
- [ ] **Hyperliquid** — `POST api.hyperliquid.xyz/info` → `fundingHistory` (no auth)
- [ ] **Binance** — `GET fapi.binance.com/fapi/v1/fundingRate` (no auth, VPN to Switzerland)
- [ ] **Bybit** — `GET api.bybit.com/v5/market/funding/history` (no auth, VPN to Switzerland)
- [ ] **dYdX** — `GET indexer.dydx.trade/v4/historicalFunding` (no auth)
- [ ] **Predicted fundings** — `POST api.hyperliquid.xyz/info` → `predictedFundings` (cross-venue predictions in one call)

### 1b. Candles / Mark Prices (`data/fetchers.py`)
- [ ] **Hyperliquid** candles for all 5 tokens, 1h interval, 90+ days

### 1c. Open Interest (`data/fetchers.py`)
- [ ] **Hyperliquid** — `metaAndAssetCtxs` (current OI + mark price + funding for all assets)
- [ ] **Binance** — `GET fapi.binance.com/futures/data/openInterestHist` (VPN)
- [ ] **Bybit** — `GET api.bybit.com/v5/market/open-interest` (VPN)

### 1d. Liquidation Events (`data/fetchers.py`)
- [ ] **Binance** — `GET fapi.binance.com/fapi/v1/forceOrders` (public, no auth, VPN)
- [ ] **Coinalyze** — `GET api.coinalyze.net/v1/liquidation-history` (free, no auth)

### 1e. Lending Positions (`data/lending.py`) — for cascade model
- [ ] **HyperLend** (~$360M TVL) — query via HyperEVM RPC (Aave V3 fork)
  - RPC: `https://rpc.hyperliquid.xyz/evm` (free, 100 req/min, chain ID 999)
  - Pool contract: `0x00A89d7a5A02160f20150EbEA7a2b5E4879A1A8b`
  - Scan `Borrow` events to discover active borrowers
  - Call `getUserAccountData(address)` → health factor, collateral, debt
  - Call `getReserveData(asset)` → total supplied, borrowed, utilization
- [ ] **Morpho Blue** (~$5.4B TVL) — query via The Graph subgraph
  - Free tier sufficient (signup at https://thegraph.com/studio/ for API key)
  - GraphQL query for all borrower positions: market, collateral, debt, LTV
  - LLTV + liquidation penalty available per market
- [ ] **DeFi Llama** — historical TVL for HyperLend + Morpho Blue
  - `GET https://api.llama.fi/protocol/{protocol}` (free, no auth)

### 1f. Infrastructure
- [ ] **Unified schema** — all DataFrames follow column specs in `fetchers.py` and `lending.py`
- [ ] **Storage layer** — fetch once, save to `data/*.parquet`, load with `storage.py`
- [ ] **Data validation** — timestamps aligned to 8h funding epochs, no gaps, UTC

**Tokens:** BTC, ETH, SOL, HYPE, DOGE
**Lookback:** 90 days minimum (more is better for backtest statistical significance)

### Data Sources (all free, no paid APIs)

| Source | Data | Endpoint | Auth |
|--------|------|----------|------|
| Hyperliquid | Funding, candles, OI, predicted fundings | `api.hyperliquid.xyz/info` | None |
| Binance | Funding, OI, liquidation orders | `fapi.binance.com/fapi/v1/*` | None (VPN) |
| Bybit | Funding, OI | `api.bybit.com/v5/market/*` | None (VPN) |
| dYdX | Funding | `indexer.dydx.trade/v4/*` | None |
| Coinalyze | Aggregated liquidation history | `api.coinalyze.net/v1/*` | None |
| HyperLend | Borrower positions, health factors | `rpc.hyperliquid.xyz/evm` (Aave V3 ABI) | None |
| Morpho Blue | Borrower positions, LTV, markets | The Graph subgraph (GraphQL) | Free Graph API key |
| DeFi Llama | Historical TVL (HyperLend, Morpho) | `api.llama.fi/protocol/*` | None |

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

- Why 90-day lookback?
- Why 5× leverage cap?
- Why 70/30 carry/cascade split?
- Why these 5 tokens?
- Why Kou preferred over Merton? (show AIC/BIC comparison, asymmetric tails argument)
- Why not CGMY? (overkill for discrete large events, hard to calibrate/explain)
- Grid search results: which entry/exit thresholds won per pair, and why?
- Square-root impact vs linear temporary impact assumption
