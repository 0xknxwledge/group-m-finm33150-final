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

- [x] Non-standard data source (not equity OHLC/VWAP/volume) — crypto funding rates + OI from on-chain venues
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

## Status Overview (as of 2026-03-07)

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Data Pipeline | **DONE** | 115K funding rows (hourly), 53K OI rows; all unit bugs fixed |
| 2a. Jump-Diffusion Models | **DONE** | Merton + Kou calibration, model comparison, jump-weighted risk |
| 2b. Cascade Simulator | **DONE** | All functions implemented (~280 lines) |
| 3. Carry Strategy | **DONE** | All functions implemented; needs 8h→1h epoch update |
| 3. Allocation | **DONE** | `allocate_positions` + `_enforce_risk_limits` implemented |
| 4. Backtest Engine | **NOT STARTED** | `run_backtest` is a stub |
| 4. Transaction Costs | **DONE** | Almgren-Chriss + all 7 venue perp fees |
| 4. Performance Analytics | **NOT STARTED** | 2 core functions are stubs |
| 5. Notebook (EDA + Cascade) | **DONE** | Sections 1-4 complete with plots and prose |
| 5. Notebook (Strategy-Conclusion) | **IN PROGRESS** | Section 5 spread EDA done; 6-9 are placeholders |
| 5. Pitchbook | **NOT STARTED** | Does not exist yet |
| 5. Academic Papers | **PARTIAL** | Only `Accumulation_Algorithms.pdf`; need Almgren-Chriss + Aloosh-Li |

---

## Phase 1 — Data Pipeline (John) -- DONE

See `data/fetchers.py` (~1,600 lines), `data/storage.py`.

### Bugs fixed (2026-03-07)

- **Pagination**: 0xArchive API returns `next_cursor` (snake_case); code had `nextCursor` (camelCase). Only first page was ever fetched.
- **Lighter funding**: needed `interval=1h` parameter; without it, raw ~10s ticks all showed 0.0012.
- **OI interval**: added `interval=1h` for Hyperliquid/Lighter OI (raw ticks were sub-second frequency).
- **Lighter OI**: was multiplied by mark_price erroneously (already in USD). Fixed.
- **Binance/Bybit/OKX OI**: returned in base asset (contracts), not USD. Now fetches mark price and converts.
- **dYdX/Kraken funding**: removed unnecessary 8h aggregation — kept native 1h rates.
- **Venue fees**: corrected Hyperliquid (3.5→4.5 bps) and Binance (4→5 bps, was spot fee); added OKX, Kraken, Lighter.
- **Dead code removed**: liquidations, reserve prices, `lending.py`.

### Funding rate semantics

| Venue | Rate unit | Settlement | Hourly payment |
|-------|-----------|------------|----------------|
| Hyperliquid | 8h rate | every 1h | rate / 8 |
| Lighter | 8h rate | every 1h | rate / 8 |
| dYdX | 1h rate | every 1h | rate |
| Kraken | 1h rate | every 1h | rate |
| Binance | 8h rate | every 8h | forward-fill on 1h grid |
| Bybit | 8h rate | every 8h | forward-fill on 1h grid |
| OKX | 8h rate | every 8h | forward-fill on 1h grid |

### Current dataset

| File | Rows | Coverage |
|------|------|----------|
| `funding_rates.parquet` | 115,115 | 7 venues × 5 coins, hourly where available |
| `candles.parquet` | ~170K | 7 venues × 5 coins, hourly |
| `open_interest.parquet` | 53,282 | Hyperliquid + Lighter hourly; CEXes snapshot only |

### Re-pulling data

```bash
OXARCHIVE_API_KEY=<key> PYTHONPATH=src python scripts/pull_data.py          # full pull (364 days)
OXARCHIVE_API_KEY=<key> PYTHONPATH=src python scripts/pull_data.py --quick --coins BTC  # quick test
```

---

## Phase 2a — Jump-Diffusion Models (John) -- DONE

**Goal:** Calibrate jump models, compare Merton vs Kou.

- [x] **Merton jump-diffusion** (`models/merton.py`) — heuristic + MLE calibration, Poisson-mixture density
- [x] **Kou double-exponential** (`models/kou.py`) — heuristic + MLE, FFT density recovery from characteristic function
- [x] **Model comparison** (`models/compare.py`) — AIC/BIC: Merton uniformly preferred (Kou's extra parameter not justified)
- [x] **Jump-weighted risk** (`models/risk.py`) — ∫ f(-δ)·δ·A(δ) dδ, combines jump tail probabilities with cascade amplification

## Phase 2b — Cascade Simulator (Antonio) -- COMPLETE

`models/cascade.py` (~280 lines): `simulate_cascade`, `compute_amplification_curve`, `cascade_risk_signal`, `build_positions_from_oi`, sensitivity functions. All implemented and used in notebook section 4.

---

## Phase 3 — Strategy (Jean-Luc) -- DONE

**Goal:** Turn data into actionable trade signals with position sizing.

- [x] **`compute_funding_spreads()`** (`strategy/carry.py`) — pairwise cross-venue spreads per epoch
- [x] **`simulate_carry()`** (`strategy/carry.py`) — entry/exit signals with threshold-based rules
- [x] **`evaluate_carry()`** (`strategy/carry.py`) — Sharpe, win rate, trade count for a param set
- [x] **Grid search** (`grid_search_params`, `grid_search_all_pairs`, `select_best_params`) — scans entry/exit thresholds per pair
- [x] **`allocate_positions()`** (`strategy/allocation.py`) — 85/15 carry/cascade, risk limits, risk scaling
- [ ] **Needs update**: `carry.py` docstring references 8h epochs; should be updated to 1h

---

## Phase 4 — Backtester (Jean) -- NOT STARTED

**Goal:** Simulate the strategy end-to-end, produce required performance analysis.

`backtest/costs.py` is done (Almgren-Chriss). Everything else is stubs:

- [ ] **`run_backtest()`** (`backtest/engine.py`) — event loop over 1h epochs
- [ ] **`compute_performance()`** (`backtest/performance.py`) — Sharpe, drawdown, Calmar, win rate
- [ ] **`pnl_decomposition()`** (`backtest/performance.py`) — carry vs cascade vs costs
- [ ] **Two scenarios**: (A) zero costs, (B) calibrated Almgren-Chriss
- [ ] **Trade clustering analysis** — prove 40+ trades aren't clustered

---

## Phase 5 — Deliverables (All) -- PARTIALLY STARTED

- [x] **Notebook sections 1-4** (Setup, EDA, Jump-Diffusion, Cascade) — done
- [x] **Notebook section 3** — Merton/Kou calibration, QQ plots, tail fit, jump-weighted risk (John)
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
- Why Merton preferred over Kou? (show AIC/BIC comparison, symmetric tails at hourly frequency)
- Why not CGMY? (overkill for discrete large events, hard to calibrate/explain)
- Grid search results: which entry/exit thresholds won per pair, and why?
- Square-root impact vs linear temporary impact assumption
