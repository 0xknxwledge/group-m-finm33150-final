# models — Jump-Diffusion and Cascade

---

## cascade.py

Owner: Antonio Braz

Models the feedback loop between forced liquidations and price impact across perpetual futures venues.

**Cascade dynamics:**
```
price drop -> perp liquidations -> forced selling ->
orderbook absorption -> further price impact -> repeat
```

**Price impact model:** Square-root law: `Δp/p = sqrt(V / D)` where V = forced selling volume, D = orderbook depth. Consistent with Almgren-Chriss (2001) and Jusselin-Rosenbaum (2018).

The amplification curve `A(δ)` maps initial shocks to terminal effective shocks. `A(δ) > 1` indicates cascade amplification.

### Constants

```python
DEFAULT_DEPTH_USD = 5_000_000.0  # fallback when live orderbook data unavailable

MAX_LEVERAGE: dict[tuple[str, str], int]
# Max leverage per (venue, coin) from exchange APIs (March 2026).
# Keys: ("hyperliquid", "BTC"), ("binance", "ETH"), etc.
```

### Dataclasses

```python
@dataclass
class Position:
    collateral_usd: float          # margin posted
    debt_usd: float                # notional minus margin
    liquidation_threshold: float   # maintenance margin rate (fraction of notional)
    layer: str                     # "perp", "hyperlend", or "morpho"
```

```python
@dataclass
class CascadeResult:
    initial_shock: float
    effective_shock: float
    amplification: float               # effective / initial
    rounds: int
    total_notional_liquidated: float
    liquidations_by_layer: dict[str, float]
```

### Core Simulation

```python
def simulate_cascade(
    positions: list[Position],
    current_price: float,
    initial_shock_pct: float,
    orderbook_depth_usd: float | None = None,
    max_rounds: int = 50,
) -> CascadeResult
```
Simulate a liquidation cascade from an initial exogenous price shock. Iterates until no new liquidations are triggered or `max_rounds` is reached.

```python
def compute_amplification_curve(
    positions: list[Position],
    current_price: float,
    shocks: NDArray | None = None,        # default: 1% to 50% in 0.5% steps
    orderbook_depth_usd: float | None = None,
) -> list[CascadeResult]
```
Run `simulate_cascade` across a range of initial shocks. Returns list of `CascadeResult` for plotting `A(δ)`.

### Risk Signal

```python
def cascade_risk_signal(
    positions: list[Position],
    current_price: float,
    orderbook_depth_usd: float | None = None,
    threshold_amplification: float = 1.5,
) -> dict
```
Returns:
- `risk_score`: float in [0, 1]
- `critical_shock`: smallest δ where `A(δ) > threshold`
- `amplification_at_5pct`: `A(0.05)`
- `signal`: bool — True if cascade risk is elevated (`risk_score > 0.5`)

Risk score maps `critical_shock` from `[0.5, 0.005] → [0, 1]`: a 5% critical shock = high risk, a 50% critical shock = low risk.

```python
def per_coin_risk_signals(
    oi_df: pl.DataFrame,
    leverage: float = 5.0,
    orderbook_depth_usd: float | None = None,
    depth_per_coin: dict[str, float] | None = None,
    threshold_amplification: float = 1.5,
    tiered: bool = True,
) -> dict[str, dict]
```
Compute `cascade_risk_signal` for each coin individually. If `depth_per_coin` is provided, each coin uses its measured depth. If `tiered=True` (default), uses per-venue leverage tiers from `MAX_LEVERAGE`.

### Position Builders

```python
def build_positions_from_oi(
    oi_df: pl.DataFrame,
    leverage: float = 5.0,
    liq_threshold: float = 0.005,
    layer: str = "perp",
) -> list[Position]
```
Build `Position` list from OI DataFrame assuming uniform leverage. Takes latest snapshot per `(venue, coin)`.
Expected schema: `(timestamp, venue, coin, oi_usd)`.

```python
def build_positions_tiered(
    oi_df: pl.DataFrame,
    tiers: list[tuple[float, float]] | None = None,
    liq_threshold: float = 0.005,
    layer: str = "perp",
) -> list[Position]
```
Build positions with heterogeneous leverage tiers per `(venue, coin)`. Default tiers from `MAX_LEVERAGE`: 50% at 3x, 30% at max/2, 20% at max. This reflects that a Binance BTC position can reach 125x while a dYdX HYPE position maxes at 5x.

### Sensitivity Analysis

```python
def sensitivity_to_leverage(
    oi_df: pl.DataFrame,
    leverages: list[float] | None = None,   # default: [3, 5, 10, 20]
    shocks: NDArray | None = None,
    **cascade_kwargs,
) -> dict[float, list[CascadeResult]]
```
Amplification curves across leverage assumptions.

```python
def sensitivity_to_depth(
    positions: list[Position],
    depths_usd: list[float] | None = None,  # default: [1M, 5M, 10M, 50M]
    shocks: NDArray | None = None,
    current_price: float = 1.0,
) -> dict[float, list[CascadeResult]]
```
Amplification curves across orderbook depth assumptions.

### Utilities

```python
def depth_by_coin(depth_df: pl.DataFrame) -> dict[str, float]
```
Aggregate per-coin bid depth (USD) across all venues.
Input schema: `[coin, venue, bid_depth_usd, ...]`. Returns `coin → total bid-side depth within 1% of mid`.

```python
def validate_cascade(
    candles: pl.DataFrame,
    liq_vol: pl.DataFrame,
    oi_df: pl.DataFrame,
    depth_per_coin: dict[str, float] | None = None,
    drawdown_threshold: float = 0.05,
    window_hours: int = 24,
) -> pl.DataFrame
```
Compare predicted vs realized liquidation volume during drawdowns. Identifies drawdown events from hourly candle data, runs the cascade simulator with the nearest OI snapshot, and sums realized liquidation volume over the window.
Returns: `[timestamp, coin, drawdown_pct, predicted_liq_usd, realized_liq_usd]`.

---

## merton.py

Owner: John Beecher

Merton jump-diffusion calibration and density. Augments GBM with a compound Poisson jump process:

```
dS/S = (μ - λk)dt + σ dW + J dN
```
where `N ~ Poisson(λ)`, `J ~ LogNormal(μ_J, σ_J)`.

Log-return density is a Poisson-weighted mixture of normals:
```
f(r) = Σ_n [e^{-λdt}(λdt)^n/n!] × φ(r; m_n, v_n²)
```
Jump sizes are symmetric (log-normal). For asymmetric tails, see `kou.py`.

### Dataclass

```python
@dataclass
class MertonParams:
    sigma: float          # diffusion volatility (per period)
    lam: float            # jump intensity (jumps per period)
    mu_j: float           # mean jump size (log)
    sigma_j: float        # jump size volatility (log)
    mu: float             # drift (per period)
    log_likelihood: float
    aic: float
    bic: float
    n_params: int = 5
```

### Functions

```python
def calibrate_merton(returns: NDArray, dt: float = 1.0) -> MertonParams
```
Full two-stage calibration: heuristic → MLE.

```python
def heuristic_calibration(
    returns: NDArray,
    dt: float = 1.0,
    threshold_sigma: float = 3.0,
    n_iterations: int = 3,
) -> MertonParams
```
Stage 1: iterative 3σ-filtering separates jump and diffusion components. Estimates jump distribution from outliers.

```python
def mle_calibration(
    returns: NDArray,
    dt: float = 1.0,
    heuristic_params: MertonParams | None = None,
) -> MertonParams
```
Stage 2: L-BFGS-B optimization on the full Merton log-likelihood. Sets `log_likelihood`, `aic`, `bic`.

```python
def merton_log_density(
    x: NDArray,
    params: MertonParams,
    dt: float = 1.0,
    n_terms: int = 20,
) -> NDArray
```
Log-density of the Merton model (Poisson mixture of normals). Vectorized over x; uses logsumexp for numerical stability.

---

## kou.py

Owner: John Beecher

Kou (2002) double-exponential jump-diffusion. Augments GBM with asymmetric jumps:

```
dS/S = (μ - λκ)dt + σ dW + J dN
```
Jump sizes have a double-exponential distribution:
```
f_J(x) = p · η₁ · exp(-η₁ x) · 1_{x≥0}  +  (1-p) · η₂ · exp(η₂ x) · 1_{x<0}
```

Density is recovered via FFT of the closed-form characteristic function:
```
φ(u) = exp{dt·[iuμ̃ - σ²u²/2 + λ(pη₁/(η₁-iu) + (1-p)η₂/(η₂+iu) - 1)]}
```
This avoids the intractable k-fold convolution of double-exponential distributions.

### Dataclass

```python
@dataclass
class KouParams:
    sigma: float    # diffusion volatility (per period)
    lam: float      # jump intensity (jumps per period)
    p: float        # probability jump is positive (0 < p < 1)
    eta1: float     # positive jump rate (η₁ > 1, finite mean requirement)
    eta2: float     # negative jump rate (η₂ > 0)
    mu: float       # drift (per period)
    log_likelihood: float
    aic: float
    bic: float
    n_params: int = 6

    # Properties
    mean_positive_jump: float   # 1/η₁
    mean_negative_jump: float   # -1/η₂
    jump_mean: float            # p/η₁ - (1-p)/η₂
    tail_asymmetry: float       # down-jump intensity / up-jump intensity; > 1 = heavier left tail
```

### Functions

```python
def calibrate_kou(returns: NDArray, dt: float = 1.0) -> KouParams
```
Full two-stage calibration: heuristic → MLE.

```python
def heuristic_calibration(
    returns: NDArray,
    dt: float = 1.0,
    threshold_sigma: float = 3.0,
    n_iterations: int = 3,
) -> KouParams
```
Like Merton heuristic, but fits separate exponentials to positive and negative jumps to estimate `p`, `η₁`, `η₂`.

```python
def mle_calibration(
    returns: NDArray,
    dt: float = 1.0,
    heuristic_params: KouParams | None = None,
) -> KouParams
```
L-BFGS-B with bounds: `σ > 0`, `λ > 0`, `0 < p < 1`, `η₁ > 1`, `η₂ > 0`. Sets `log_likelihood`, `aic`, `bic`.

```python
def kou_log_density(
    x: NDArray,
    params: KouParams,
    dt: float = 1.0,
    n_terms: int = 20,   # unused; kept for API compat with Merton
) -> NDArray
```
Log-density via FFT inversion of the characteristic function. FFT grid: 2^14 points over ±50%.

---

## compare.py

Owner: John Beecher

Model comparison — Merton vs Kou via AIC/BIC.

### Dataclass

```python
@dataclass
class ModelComparison:
    coin: str
    n_obs: int
    merton: MertonParams
    kou: KouParams
    preferred: str   # "merton" or "kou" (chosen by BIC)

    # Properties
    bic_delta: float   # BIC(Merton) - BIC(Kou); positive favors Kou
    aic_delta: float   # AIC(Merton) - AIC(Kou); positive favors Kou
```

### Functions

```python
def compare_models(
    returns: NDArray,
    coin: str = "",
    dt: float = 1.0,
) -> ModelComparison
```
Calibrate both Merton and Kou on the same return series. Preferred model chosen by BIC.

```python
def compare_all_tokens(
    returns_dict: dict[str, NDArray],
    dt: float = 1.0,
) -> dict[str, ModelComparison]
```
Run model comparison for every token in the universe.

**Result:** Merton uniformly preferred across all 5 tokens (Kou's extra parameter is not justified by BIC at hourly frequency — tails are not sufficiently asymmetric).

---

## risk.py

Owner: John Beecher

Jump-weighted risk — combines jump tail probabilities with cascade amplification.

**Formula:**
```
E[amplified loss] = ∫ f(-δ) · δ · A(δ) dδ
```
where `f` is the calibrated Merton density and `A(δ)` is the cascade amplification.

### Functions

```python
def jump_weighted_risk(
    merton_params: MertonParams,
    positions: list,
    dt: float = 1.0,
    orderbook_depth_usd: float | None = None,
    n_shocks: int = 100,
) -> dict
```
Numerically integrates over δ ∈ [0.5%, 50%] using the trapezoidal rule.

Returns:
- `baseline_loss`: `∫ f(-δ) · δ dδ` (no cascade)
- `amplified_loss`: `∫ f(-δ) · δ · A(δ) dδ`
- `cascade_excess`: `amplified_loss - baseline_loss`
- `cascade_multiplier`: `amplified_loss / baseline_loss`
- `tail_probability_5pct`: `P(return ≤ -5%)`
- `amplification_at_5pct`: `A(0.05)`

```python
def jump_weighted_risk_all_coins(
    merton_params_dict: dict[str, MertonParams],
    oi_df: pl.DataFrame,
    dt: float = 1.0,
    leverage: float = 5.0,
    orderbook_depth_usd: float | None = None,
    depth_per_coin: dict[str, float] | None = None,
    tiered: bool = False,
) -> dict[str, dict]
```
Compute `jump_weighted_risk` for all coins. If `tiered=True`, uses `build_positions_tiered`. If `depth_per_coin` is provided, each coin uses its measured orderbook depth.
