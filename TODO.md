# TODO — Group M Final Project

**Funding the Fall: Cross-Venue Carry and Cascade Alpha in Crypto Perpetuals**

FINM 33150 — Winter 2026

---

## Team & Ownership

| Member | Module | Path |
|--------|--------|------|
| **John Beecher** | Data Pipeline + Models (Merton + Kou) | `src/funding_the_fall/data/`, `src/funding_the_fall/models/` |
| **Antonio Braz** | Cascade Simulator | `src/funding_the_fall/models/cascade.py` |
| **Jean-Luc Choiseul** | Strategy (Carry + Allocation) | `src/funding_the_fall/strategy/` |
| **Jean Mauratille** | Backtester + Transaction Costs | `src/funding_the_fall/backtest/` |

**Dependency chain:** John (data) -> John (Merton/Kou) + Antonio (cascade) + Jean-Luc -> Jean

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

## Status Overview (as of 2026-03-06)

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Data Pipeline | **DONE** | 7 venues x 5 coins, 5 parquet files committed |
| 2a. Jump-Diffusion Models | **NOT STARTED** | Dataclasses defined; all 3 core functions are stubs |
| 2b. Cascade Simulator | **DONE** | All functions implemented (~280 lines) |
| 3. Carry Strategy | **NOT STARTED** | 3 core functions are stubs |
| 3. Allocation | **DONE** | `allocate_positions` + `_enforce_risk_limits` implemented |
| 4. Backtest Engine | **NOT STARTED** | `run_backtest` is a stub |
| 4. Transaction Costs | **DONE** | Almgren-Chriss fully implemented |
| 4. Performance Analytics | **NOT STARTED** | 2 core functions are stubs |
| 5. Notebook (EDA + Cascade) | **DONE** | Sections 1-4 complete with plots and prose |
| 5. Notebook (Strategy-Conclusion) | **NOT STARTED** | Sections 5-9 are placeholders |
| 5. Pitchbook | **NOT STARTED** | Does not exist yet |
| 5. Academic Papers | **PARTIAL** | Only `Accumulation_Algorithms.pdf`; need Almgren-Chriss + Aloosh-Li |

---

## Phase 1 — Data Pipeline (John) -- COMPLETE

All data fetching, storage, and validation is done. See `data/fetchers.py` (~1,600 lines), `data/lending.py` (~750 lines), `data/storage.py`.

### Current dataset (committed in `data/`)

| File | Rows | Coverage |
|------|------|----------|
| `funding_rates.parquet` | 29,217 | 7 venues x 5 coins, 2025-03 to 2026-03 |
| `candles.parquet` | ~170K | 7 venues x 5 coins, hourly |
| `open_interest.parquet` | 10,025 | Historical (HL/Lighter) + snapshots (others) |
| `liquidations.parquet` | 8,059 | 0xArchive (May 2025+) + OKX |
| `reserve_prices.parquet` | 3,000 | ETH + USDC/USDT for lending replay |

### Re-pulling data

To refresh data (e.g. for a newer window), set `OXARCHIVE_API_KEY` and run:
```bash
PYTHONPATH=src python scripts/pull_data.py                    # full pull
PYTHONPATH=src python scripts/backfill_data.py                # fix liquidations + dYdX
PYTHONPATH=src python scripts/pull_data.py --quick --coins BTC  # quick single-coin test
```

---

## Phase 2a — Jump-Diffusion Models (John) -- BLOCKED

**Goal:** Calibrate jump models, compare Merton vs Kou.

Dataclasses (`MertonParams`, `KouParams`, `ModelComparison`) are defined. `compare.py` logic is implemented but blocked by the calibration stubs.

- [ ] **Merton jump-diffusion** (`models/merton.py`)
  - `heuristic_calibration()` — iterative 3-sigma filtering
  - `merton_log_density()` — Poisson-mixture log-likelihood
  - `mle_calibration()` — L-BFGS-B optimization
- [ ] **Kou double-exponential jump-diffusion** (`models/kou.py`)
  - `heuristic_calibration()` — iterative 3-sigma filtering
  - `kou_log_density()` — double-exponential mixture log-likelihood
  - `mle_calibration()` — L-BFGS-B optimization
- [ ] **Model comparison** (`models/compare.py`) — works once above are done
  - AIC/BIC comparison, QQ plots, Jarque-Bera test

## Phase 2b — Cascade Simulator (Antonio) -- COMPLETE

`models/cascade.py` (~280 lines): `simulate_cascade`, `compute_amplification_curve`, `cascade_risk_signal`, `build_positions_from_oi`, sensitivity functions. All implemented and used in notebook section 4.

---

## Phase 3 — Strategy (Jean-Luc) -- NOT STARTED

**Goal:** Turn data into actionable trade signals with position sizing.

- [ ] **`compute_funding_spreads()`** (`strategy/carry.py`) — pairwise cross-venue spreads per epoch
- [ ] **`simulate_carry()`** (`strategy/carry.py`) — run carry trades given params
- [ ] **`evaluate_carry()`** (`strategy/carry.py`) — compute Sharpe, win rate, trade count for a param set
- [ ] **Grid search** — framework exists (`grid_search_params`, `grid_search_all_pairs`, `select_best_params`) but blocked by the 3 stubs above
- [x] **`allocate_positions()`** (`strategy/allocation.py`) — 85/15 carry/cascade, risk limits, risk scaling

---

## Phase 4 — Backtester (Jean) -- NOT STARTED

**Goal:** Simulate the strategy end-to-end, produce required performance analysis.

`backtest/costs.py` is done (Almgren-Chriss). Everything else is stubs:

- [ ] **`run_backtest()`** (`backtest/engine.py`) — event loop over 8h epochs
- [ ] **`compute_performance()`** (`backtest/performance.py`) — Sharpe, drawdown, Calmar, win rate
- [ ] **`pnl_decomposition()`** (`backtest/performance.py`) — carry vs cascade vs costs
- [ ] **Two scenarios**: (A) zero costs, (B) calibrated Almgren-Chriss
- [ ] **Trade clustering analysis** — prove 40+ trades aren't clustered

---

## Phase 5 — Deliverables (All) -- PARTIALLY STARTED

- [x] **Notebook sections 1-4** (Setup, EDA, Jump-Diffusion placeholders, Cascade) — done
- [ ] **Notebook section 3** — fill in once Merton/Kou calibration works (John)
- [ ] **Notebook sections 5-6** — carry strategy + allocation (Jean-Luc)
- [ ] **Notebook sections 7-8** — backtest + performance (Jean)
- [ ] **Notebook section 9** — conclusion with final numbers (All)
- [ ] **Pitchbook** (`pitchbook.pdf`) — non-technical, no jargon, self-explanatory
- [ ] **Academic papers** (2+ PDFs)
  - [ ] Almgren & Chriss (2001) — "Optimal Execution of Portfolio Transactions"
  - [ ] Aloosh & Li (2024) — "Perpetual Futures and Funding Rate Arbitrage"
- [ ] **Submission zip**: `Beecher_Braz_Choiseul_Mauratille.zip`

---

## Key Design Decisions to Document

These require a sentence or two in the notebook justifying the choice (per professor's instructions):

- Why 5x leverage cap?
- Why 85/15 carry/cascade split?
- Why these 5 tokens?
- Why Kou preferred over Merton? (show AIC/BIC comparison, asymmetric tails argument)
- Why not CGMY? (overkill for discrete large events, hard to calibrate/explain)
- Grid search results: which entry/exit thresholds won per pair, and why?
- Square-root impact vs linear temporary impact assumption
