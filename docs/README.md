# API Reference — Funding the Fall

Two-component crypto perpetual futures strategy.

## Modules

| File | Module | Owner |
|------|--------|-------|
| [data.md](data.md) | `data/storage.py`, `data/fetchers.py` | Antonio Braz / John Beecher |
| [models.md](models.md) | `models/cascade.py`, `models/merton.py`, `models/kou.py`, `models/compare.py`, `models/risk.py` | Antonio Braz / John Beecher |
| [strategy.md](strategy.md) | `strategy/carry.py`, `strategy/allocation.py` | Jean-Luc Choiseul |
| [backtest.md](backtest.md) | `backtest/engine.py`, `backtest/costs.py`, `backtest/performance.py` | Jean Mauratille |

## Dependency Chain

```
data/ (storage + fetchers)
  └── models/ (merton, kou, compare, cascade, risk)
        └── strategy/ (carry, allocation)
              └── backtest/ (engine, costs, performance)
```

## Quick Import

```python
from funding_the_fall.data.storage import load_funding, load_candles, load_oi
from funding_the_fall.models.cascade import simulate_cascade, cascade_risk_signal
from funding_the_fall.models.merton import calibrate_merton
from funding_the_fall.models.kou import calibrate_kou
from funding_the_fall.models.compare import compare_models
from funding_the_fall.models.risk import jump_weighted_risk
from funding_the_fall.strategy.carry import compute_funding_spreads, simulate_carry
from funding_the_fall.strategy.allocation import allocate_positions
from funding_the_fall.backtest.costs import TransactionCostModel, make_cost_model
from funding_the_fall.backtest.engine import run_backtest
from funding_the_fall.backtest.performance import compute_performance, pnl_decomposition
```
