# Extending the Liquip Score: A Jump-Diffusion Framework for Liquidation Probability

**Author:** John Beecher
**Date:** February 2026
**Extends:** Anthias Liquip Score: Measuring Liquidation Probability (Anthias Team)

---

## Abstract

The Anthias Liquip Score provides a closed-form liquidation probability metric by modeling position value as geometric Brownian motion (GBM). While elegant and computationally efficient, GBM assumes normally distributed returns — an assumption that systematically underestimates the probability of extreme price moves in crypto assets. We propose replacing GBM with the Merton jump-diffusion process, which augments continuous diffusion with discrete, random jumps. The resulting "Jump-Diffusion Liquip Score" retains analytical tractability (a Poisson-weighted sum of normal CDFs) while capturing the fat tails, excess kurtosis, and sudden dislocations empirically observed in DeFi collateral assets. We describe a two-stage calibration procedure — heuristic initialization followed by maximum likelihood refinement — and outline three backtesting methodologies to validate that the extension improves upon the original score.

---

## 1. Motivation

### 1.1 The GBM Assumption and Its Limitations

The current Liquip Score models the value of a wallet position S(t) as:

```
S(t) = § × exp((μ - σ²/2)t + σW(t))
```

This implies log-returns are i.i.d. normal with mean (μ - σ²/2) per day and variance σ² per day. The liquidation probability over t days is then:

```
P(liquidation) = Φ((ln((§-φ)/§) - (μ - σ²/2)t) / (σ√t))
```

The Anthias paper itself notes (Section 3): "asset returns frequently exhibit characteristics such as skewness and kurtosis, thereby deviating from a perfect normal distribution. Specifically, asset returns often display 'fat tails,' indicating a greater probability of extreme events than what a normal distribution would predict."

### 1.2 Empirical Evidence from Hyperliquid Collateral Assets

Using 30-day hourly candle data from the Hyperliquid API, we can directly measure the departure from normality for Felix's primary collateral assets:

| Asset | Ann. Vol | Excess Kurtosis (hourly) | Skewness (hourly) |
|-------|----------|--------------------------|-------------------|
| HYPE  | ~136%    | To be measured           | To be measured     |
| BTC   | ~62%     | To be measured           | To be measured     |
| ETH   | ~84%     | To be measured           | To be measured     |

*(These will be populated from the `liquidation_analysis.py` candle data. Standard values for crypto assets: excess kurtosis 5-15, skewness -0.5 to -1.5.)*

Under normal returns, we expect excess kurtosis = 0 and skewness = 0. Any significant positive kurtosis directly implies that the GBM Liquip Score underestimates the probability of large moves — precisely the moves that trigger liquidations.

### 1.3 Types of Extreme Moves in DeFi

The normal distribution fails to capture several empirically observed phenomena:

1. **Jump discontinuities:** Exchange delistings, protocol exploits (e.g., the JELLY incident on HyperCore that caused >$10M unrealized loss), regulatory announcements, and oracle failures produce instantaneous price gaps that are not the tail of a continuous diffusion process.

2. **Volatility clustering:** High-volatility periods persist (GARCH effects). A 10% daily move today makes another 10% move tomorrow far more likely than the unconditional distribution suggests.

3. **Negative skewness:** Crypto crashes are faster and larger than rallies. The distribution of returns is asymmetric in exactly the direction that matters for liquidation.

The Merton jump-diffusion model directly addresses (1) and partially addresses (3) through negative jump mean. It does not address (2), which would require a stochastic volatility extension (discussed in Future Work).

---

## 2. The Merton Jump-Diffusion Model

### 2.1 Price Dynamics

We replace the GBM dynamics with:

```
dS/S = (μ - λk)dt + σdW(t) + JdN(t)
```

Where:
- **σdW(t)** is the continuous diffusion component (same as current Liquip Score)
- **N(t)** is a Poisson process with intensity λ (average number of jumps per unit time)
- **J** is the random jump size, with ln(1+J) ~ Normal(μ_J, σ_J²)
- **k = E[J] = exp(μ_J + σ_J²/2) - 1** is the drift compensator ensuring the jump component has zero expected contribution to the drift (so that μ retains its interpretation as the total expected return)

The five parameters are:
- **μ**: drift rate (expected return per day, same as original)
- **σ**: diffusion volatility (continuous-component volatility)
- **λ**: jump intensity (expected number of jumps per day)
- **μ_J**: mean of log-jump size (negative for downward jumps)
- **σ_J**: standard deviation of log-jump size

### 2.2 Solution

The position value at time t is:

```
S(t) = § × exp((μ - σ²/2 - λk)t + σW(t) + Σᵢ₌₁^N(t) ln(1 + Jᵢ))
```

### 2.3 Key Property: Conditional Normality

Conditional on N(t) = n jumps occurring in [0, t]:

- The diffusion term σW(t) ~ Normal(0, σ²t)
- The jump sum Σ ln(1+Jᵢ) ~ Normal(n·μ_J, n·σ_J²)
- These are independent

Therefore:

```
ln(S(t)/§) | N(t)=n  ~  Normal(mₙ, vₙ²)
```

where:

```
mₙ = (μ - σ²/2 - λk)t + n·μ_J
vₙ² = σ²t + n·σ_J²
```

This is the central property that preserves analytical tractability.

---

## 3. The Jump-Diffusion Liquip Score

### 3.1 Derivation

We seek P(S(t) < § - φ), the probability that the position value falls below the liquidation threshold within t days.

**Step 1: Condition on N(t) = n.**

```
P(S(t) < §-φ | N(t)=n) = P(ln(S(t)/§) < ln((§-φ)/§) | N(t)=n)
```

Since ln(S(t)/§) | N(t)=n is Normal(mₙ, vₙ²), this is:

```
P(S(t) < §-φ | N(t)=n) = Φ(zₙ)
```

where:

```
zₙ = (ln((§-φ)/§) - mₙ) / vₙ
   = (ln((§-φ)/§) - (μ - σ²/2 - λk)t - n·μ_J) / √(σ²t + n·σ_J²)
```

**Step 2: Marginalize over N(t) using the law of total probability.**

N(t) ~ Poisson(λt), so P(N(t) = n) = e^{-λt}(λt)ⁿ/n!

```
P(liquidation in t days) = Σₙ₌₀^∞  [e^{-λt}(λt)ⁿ / n!] × Φ(zₙ)
```

This is the **Jump-Diffusion Liquip Score**.

### 3.2 Properties

**Reduces to original Liquip Score when λ = 0.** The sum collapses to the n=0 term (P(N(t)=0) = 1), and z₀ = (ln((§-φ)/§) - (μ - σ²/2)t) / (σ√t), which is exactly the Anthias formula.

**Monotonically increases with jump intensity λ.** Adding jumps (with μ_J < 0, i.e., downward jumps) strictly increases the liquidation probability. This matches intuition: more jump risk means more liquidation risk.

**Rapid convergence.** For typical crypto parameters (λ ≈ 0.01-0.05 jumps/day, t ≈ 30 days), λt ≈ 0.3-1.5. Poisson probabilities for n > 15 are negligible (< 10⁻¹⁵). In practice, truncating the sum at N=20 terms gives machine-precision accuracy.

**Computational cost.** Each term requires one Φ evaluation. With N=20 terms, the jump-diffusion score requires ~20× the computation of the original score — still effectively instantaneous (microseconds per position).

### 3.3 Extension to Wallet Variance Framework

The original Liquip Score computes wallet variance σ² = v'Ωv from the covariance matrix of all position assets. The jump-diffusion extension requires specifying jump parameters at the wallet level.

Two approaches:

**Approach A: Aggregate jumps.** Treat the wallet's combined return process as having its own jump component. Calibrate (λ, μ_J, σ_J) from the historical time series of wallet-level returns (computed as v'r_t where r_t is the vector of asset returns at time t). This is simple but treats all assets as jumping simultaneously.

**Approach B: Per-asset jumps with aggregation.** Calibrate jump parameters per asset, then compute the wallet-level jump distribution as the weighted sum of independent compound Poisson processes. This is more accurate but significantly more complex — the wallet-level process is no longer a simple Merton model, and the Liquip Score becomes a multidimensional integral. For positions dominated by a single collateral asset (common in Felix CDPs), Approach A is a good approximation.

**Recommendation:** Use Approach A for production. It maintains the simplicity of the original framework while capturing the key improvement. Approach B is a worthwhile direction for future research on multi-asset positions.

### 3.4 Days Until Liquidation (Jump-Diffusion)

The "Days Until Liquidation" metric generalizes directly. We seek the time t* such that:

```
Σₙ₌₀^∞  [e^{-λt*}(λt*)ⁿ / n!] × Φ(zₙ(t*)) = α
```

This does not admit a closed-form solution (unlike the original, which yields a quadratic in t). However, the bisection method from Section 3.2.2 of the Anthias paper applies directly: the left-hand side is monotonically increasing in t*, so bisection on [0, t_max] converges reliably.

---

## 4. Calibration

### 4.1 Two-Stage Approach

We recommend a two-stage calibration: heuristic initialization to get parameter estimates in the right ballpark, followed by maximum likelihood estimation (MLE) for statistical rigor.

#### Stage 1: Heuristic Initialization

Given a time series of k daily (or hourly) log-returns {r₁, r₂, ..., rₖ}:

**Step 1 — Estimate diffusion volatility σ̂.** Compute the sample standard deviation of all returns. Then iteratively:
  - Flag returns with |rₜ - r̄| > 3σ̂ as candidate jumps
  - Recompute σ̂ from non-flagged returns only
  - Repeat until the set of flagged returns stabilizes (typically 2-3 iterations)

**Step 2 — Estimate jump intensity λ̂.** Count the number of flagged returns n_jumps:
```
λ̂ = n_jumps / (k × Δt)
```
where Δt is the time interval (1 day for daily returns, 1/24 for hourly).

**Step 3 — Estimate jump size distribution.** From the flagged returns {r_j1, r_j2, ...}:
```
μ̂_J = mean(flagged returns)
σ̂_J = std(flagged returns)
```

**Step 4 — Estimate drift.**
```
μ̂ = mean(all returns) / Δt + λ̂k̂
```
where k̂ = exp(μ̂_J + σ̂_J²/2) - 1.

#### Stage 2: Maximum Likelihood Refinement

The log-likelihood of the observed return series under Merton is:

```
ℓ(μ, σ, λ, μ_J, σ_J) = Σₜ₌₁ᵏ log f(rₜ)
```

where the density of a single return is:

```
f(rₜ) = Σₙ₌₀^∞ [e^{-λΔt}(λΔt)ⁿ / n!] × φ(rₜ; (μ-σ²/2-λk)Δt + nμ_J, σ²Δt + nσ_J²)
```

and φ(x; m, v²) is the normal PDF with mean m and variance v².

**Optimization.** Initialize with (μ̂, σ̂, λ̂, μ̂_J, σ̂_J) from Stage 1. Maximize ℓ using L-BFGS-B (scipy.optimize.minimize with bounds to enforce σ > 0, λ ≥ 0, σ_J > 0). Truncate the infinite sum at N=20 terms.

**Standard errors.** Compute the Hessian of ℓ at the MLE. The inverse Hessian gives the asymptotic covariance matrix; diagonal entries give parameter standard errors.

### 4.2 Why Both Stages

The heuristic provides:
- **Interpretable diagnostics.** You can directly examine which returns were classified as jumps and verify they correspond to real market events (exchange incidents, protocol exploits, etc.)
- **Robust initialization.** MLE optimization landscapes for mixture models can be multimodal. Starting from heuristic estimates dramatically reduces the risk of converging to a spurious local optimum
- **A fallback.** If MLE fails to converge (rare but possible when jumps are very infrequent and λ is near zero), the heuristic estimates remain usable

MLE provides:
- **Joint consistency.** All parameters are estimated simultaneously to maximize the probability of the observed data, avoiding the sequential dependency of the heuristic
- **Statistical efficiency.** MLE makes full use of every observation, rather than hard-classifying returns into jump/no-jump bins based on an arbitrary threshold
- **Standard errors and hypothesis testing.** Enables formal comparison with the GBM model (see Section 5.3)

### 4.3 Practical Considerations

**Data requirements.** The Merton model has 5 parameters vs. 2 for GBM (μ, σ). Reliable estimation of jump parameters requires sufficient jump observations. With λ ≈ 0.02 jumps/day and 30 days of hourly data (720 observations), we expect ~0.6 jumps total — barely enough. Recommendations:
- Use at least 90 days of hourly data (≈2,160 observations, ≈2-5 jump events)
- Alternatively, use daily data over 1+ years if available
- For assets with very short histories (new tokens), consider using a related asset's jump parameters as a prior (e.g., calibrate to HYPE but apply to kHYPE)

**Rolling recalibration.** Parameters should be re-estimated periodically (weekly or monthly) as market regimes change. The heuristic is fast enough for daily recalibration; MLE is fast enough for weekly.

**Parameter stability.** Monitor whether λ̂ and σ̂_J are stable across recalibration windows. If they vary dramatically, this suggests the simple Merton model may be insufficient and a time-varying intensity model (Hawkes process) could be warranted.

---

## 5. Validation: Is the Jump-Diffusion Score Better?

Three complementary approaches, in order of increasing data requirements:

### 5.1 Log-Likelihood Ratio Test (No Liquidation Data Required)

Since GBM is nested within Merton (set λ = 0), we can use a likelihood ratio test:

```
Test statistic: D = -2(ℓ_GBM - ℓ_Merton)
```

Under the null hypothesis (GBM is correct), D is asymptotically χ²(3) distributed (3 additional parameters: λ, μ_J, σ_J). If D exceeds the critical value at the desired significance level, we reject GBM in favor of Merton.

**Data needed:** Only historical returns (already available from `candleSnapshot` API).

**What this tests:** Whether the return-generating process has jumps. This is a necessary condition for the jump-diffusion Liquip Score to differ from the original; it does not directly test liquidation prediction accuracy.

### 5.2 Calibration / Reliability Diagrams (Requires Liquidation History)

A well-calibrated probability model should satisfy: among all position-time observations where the predicted liquidation probability is p, approximately fraction p should actually be liquidated.

**Procedure:**
1. For a historical period, take snapshots of all Felix positions at regular intervals (e.g., daily)
2. For each position-snapshot, compute both the GBM Liquip Score and the Jump-Diffusion Liquip Score for a forward horizon (e.g., 7 days)
3. Record whether the position was actually liquidated within the 7-day window
4. Bin predictions into buckets (0-1%, 1-5%, 5-10%, 10-20%, 20-50%, 50-100%)
5. Compare predicted probability (bucket midpoint) vs. observed liquidation frequency

Plot reliability diagrams for both models. The better-calibrated model has points closer to the 45-degree diagonal.

**Quantitative metric: Brier Score.**

```
BS = (1/N) Σᵢ (pᵢ - oᵢ)²
```

where pᵢ is the predicted probability and oᵢ ∈ {0, 1} is the outcome. Lower is better. Decompose into calibration, resolution, and uncertainty components to understand *why* one model outperforms.

**Data needed:** Historical position snapshots + liquidation events. Available from Felix contract events (`TroveLiquidated` on TroveManagers, `Liquidate` on Morpho Blue) and from Allium Analytics (`hyperliquid.dex.trades`).

### 5.3 Discrimination / ROC Analysis (Requires Liquidation History)

**Procedure:**
1. Same position-snapshot data as 5.2
2. For various threshold τ, classify positions as "at risk" if Liquip Score > τ
3. Compute true positive rate (sensitivity) and false positive rate (1-specificity) at each τ
4. Plot ROC curves for both models

**Quantitative metric: AUC (Area Under the ROC Curve).** Higher AUC means the model better discriminates between positions that will and will not be liquidated. A perfect model has AUC = 1; random guessing has AUC = 0.5.

**Additional discrimination test:** At fixed lookback horizons before liquidation (30, 14, 7 days before), compare the average Liquip Score assigned by each model to positions that were eventually liquidated. The jump-diffusion model should assign higher scores earlier, providing more advance warning.

**Data needed:** Same as 5.2, but the analysis focuses on the ranking of predictions rather than their absolute calibration.

---

## 6. Expected Results

### 6.1 Magnitude of Improvement

For massively overcollateralized positions (Felix CDPs at ~3000% ICR), both models will produce near-zero Liquip Scores. The improvement matters most for:

- **Morpho Blue Vanilla Markets** positions at 80-86% LLTV, where the liquidation buffer φ is small relative to position size
- **Short time horizons** (1-7 days), where jump risk dominates diffusion risk
- **High-volatility assets** (HYPE at 136% annualized), where jumps are more frequent and larger

### 6.2 Illustrative Example

Consider a Morpho Blue position with:
- Collateral: 1000 HYPE at $25 = $25,000
- Debt: $20,000 USDC (80% LTV)
- Liquidation buffer: φ = $5,000 / $45,000 = 11.1% of position value

Under GBM (σ = 136% annualized = 7.1% daily):
```
Liquip Score (7-day) = Φ((ln(0.889) - (-0.0025)×7) / (0.071×√7))
                     ≈ Φ(-0.609) ≈ 27.1%
```

Under Merton (σ_diffusion = 5.5%, λ = 0.03/day, μ_J = -0.08, σ_J = 0.06):
```
Liquip Score (7-day) = Σ Poisson(n; 0.21) × Φ(zₙ)
                     ≈ 0.81 × Φ(-0.74) + 0.17 × Φ(-0.21) + 0.02 × Φ(0.28) + ...
                     ≈ 0.81(0.230) + 0.17(0.417) + 0.02(0.610) + ...
                     ≈ 26.9%
```

*(Note: illustrative parameters — actual calibration from HYPE return data required. The key insight is that even moderate jump parameters can meaningfully shift the score for leveraged positions, particularly when μ_J is negative.)*

The difference becomes more pronounced at shorter horizons and higher leverage. At 1-day horizons, the jump component can dominate the diffusion component entirely.

---

## 7. Implementation

### 7.1 Code Structure

The implementation extends the existing `compute_liquip_score()` function in `liquidation_analysis.py`:

```python
def compute_jump_diffusion_liquip_score(
    collateral_value, debt_value,
    sigma,           # diffusion volatility (daily)
    mu=0,            # drift (daily)
    lambda_j=0.02,   # jump intensity (per day)
    mu_j=-0.05,      # mean log-jump size
    sigma_j=0.05,    # std of log-jump size
    days_forward=7,
    n_terms=20       # truncation of Poisson sum
):
    S0 = collateral_value + debt_value
    phi = collateral_value - debt_value
    if phi <= 0:
        return 1.0

    t = days_forward
    k = math.exp(mu_j + sigma_j**2 / 2) - 1  # compensator
    threshold = (S0 - phi) / S0  # = debt / S0
    ln_threshold = math.log(threshold)

    liquip = 0.0
    lambda_t = lambda_j * t

    for n in range(n_terms):
        # Poisson weight
        poisson_w = math.exp(-lambda_t) * (lambda_t ** n) / math.factorial(n)

        # Conditional mean and std of log(S(t)/S0)
        m_n = (mu - sigma**2/2 - lambda_j*k) * t + n * mu_j
        v_n = math.sqrt(sigma**2 * t + n * sigma_j**2)

        if v_n == 0:
            continue

        z_n = (ln_threshold - m_n) / v_n
        liquip += poisson_w * 0.5 * (1 + math.erf(z_n / math.sqrt(2)))

    return liquip
```

### 7.2 Calibration Code

```python
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np

def merton_log_likelihood(params, returns, dt):
    """Negative log-likelihood for MLE (minimize this)."""
    mu, sigma, lam, mu_j, sigma_j = params
    if sigma <= 0 or lam < 0 or sigma_j <= 0:
        return 1e12

    k = np.exp(mu_j + sigma_j**2/2) - 1
    n_terms = 20
    ll = 0.0

    for r in returns:
        pdf_sum = 0.0
        for n in range(n_terms):
            poisson_w = np.exp(-lam*dt) * (lam*dt)**n / np.math.factorial(n)
            m_n = (mu - sigma**2/2 - lam*k)*dt + n*mu_j
            v_n2 = sigma**2*dt + n*sigma_j**2
            if v_n2 <= 0:
                continue
            pdf_sum += poisson_w * norm.pdf(r, loc=m_n, scale=np.sqrt(v_n2))

        if pdf_sum > 0:
            ll += np.log(pdf_sum)
        else:
            ll -= 100  # penalty for zero likelihood

    return -ll  # minimize negative log-likelihood

def calibrate_merton(returns, dt=1.0):
    """Two-stage calibration: heuristic + MLE."""
    # Stage 1: Heuristic
    sigma_hat = np.std(returns)
    for _ in range(3):  # iterative filtering
        mask = np.abs(returns - np.mean(returns)) < 3 * sigma_hat
        sigma_hat = np.std(returns[mask])

    jump_mask = ~mask
    n_jumps = jump_mask.sum()
    lambda_hat = n_jumps / (len(returns) * dt)

    if n_jumps > 0:
        mu_j_hat = np.mean(returns[jump_mask])
        sigma_j_hat = np.std(returns[jump_mask]) if n_jumps > 1 else sigma_hat
    else:
        mu_j_hat = -2 * sigma_hat  # prior: jumps are ~2σ events
        sigma_j_hat = sigma_hat
        lambda_hat = 0.01 / dt     # prior: ~1 jump per 100 periods

    mu_hat = np.mean(returns) / dt

    # Stage 2: MLE
    x0 = [mu_hat, sigma_hat, lambda_hat, mu_j_hat, sigma_j_hat]
    bounds = [(None, None), (1e-6, None), (0, None), (None, None), (1e-6, None)]

    result = minimize(
        merton_log_likelihood, x0, args=(returns, dt),
        method='L-BFGS-B', bounds=bounds
    )

    if result.success:
        return dict(zip(['mu','sigma','lambda','mu_j','sigma_j'], result.x))
    else:
        # Fallback to heuristic
        return {'mu': mu_hat, 'sigma': sigma_hat, 'lambda': lambda_hat,
                'mu_j': mu_j_hat, 'sigma_j': sigma_j_hat}
```

---

## 8. Future Work

### 8.1 Peaks-Over-Threshold and Extreme Value Theory

The Merton model assumes log-normal jump sizes, which may still underestimate the frequency of truly extreme moves. An alternative approach from extreme value theory (EVT) models only the tail of the return distribution using the Generalized Pareto Distribution (GPD).

**Peaks-over-threshold (POT) approach:**
1. Choose a high threshold u (e.g., the 95th percentile of |returns|)
2. Fit a GPD to exceedances: for returns r where |r| > u, model (|r| - u) ~ GPD(ξ, β)
3. The shape parameter ξ controls tail heaviness: ξ > 0 gives heavy tails (Fréchet), ξ = 0 gives exponential tails (Gumbel), ξ < 0 gives bounded tails (Weibull)
4. Use the fitted GPD to compute tail probabilities that directly feed into the Liquip Score

**Advantages over Merton:**
- Makes no parametric assumption about the overall return distribution — only models the tail
- The GPD is the theoretically justified limiting distribution for exceedances above a high threshold (Pickands-Balkema-de Haan theorem), regardless of the underlying process
- Can capture heavier tails than the Merton model if ξ > 0
- Threshold selection can be guided by mean residual life plots rather than the arbitrary 3σ cutoff

**Disadvantages:**
- Loses the clean separation between "normal" and "jump" dynamics
- Requires sufficient tail observations for reliable GPD fitting (typically 50+ exceedances)
- The resulting Liquip Score would not have a simple closed-form — would require numerical integration or Monte Carlo
- Does not provide a generative model of the full price process, only the tail behavior

**When to prefer POT/EVT:** When the primary concern is the accuracy of the Liquip Score at very low probability thresholds (e.g., 1% liquidation probability over 1 year). At these levels, the precise shape of the tail matters more than the overall distributional fit, and POT/EVT is the gold standard in extreme risk quantification.

### 8.2 Stochastic Volatility

The Merton model uses constant diffusion volatility σ. In reality, crypto volatility clusters (high-vol days follow high-vol days). A Heston-type stochastic volatility extension:

```
dS/S = μdt + √V dW₁
dV = κ(θ - V)dt + ξ√V dW₂
```

with correlated Brownian motions (dW₁·dW₂ = ρdt) would capture this effect. The Bates model combines Heston stochastic volatility with Merton jumps and could serve as a more complete framework, though at the cost of significantly increased calibration complexity (8 parameters).

### 8.3 Hawkes Process for Jump Clustering

Replace the constant-intensity Poisson process N(t) with a self-exciting Hawkes process, where each jump increases the probability of subsequent jumps:

```
λ(t) = λ₀ + Σ_{tᵢ < t} α × exp(-β(t - tᵢ))
```

This captures the empirically observed phenomenon that crypto crashes come in clusters (contagion effects, cascading liquidations triggering further price drops). Particularly relevant for the liquidation cascade dynamics explored in the companion proposal.


---

## 9. Conclusion

The Jump-Diffusion Liquip Score is a minimal, analytically tractable extension of the Anthias Liquip Score that addresses the paper's self-identified limitation of normally distributed returns. By adding three parameters (λ, μ_J, σ_J) to capture discrete price jumps, the model produces liquidation probabilities that better reflect the empirically observed fat tails of crypto asset returns. The extension nests the original score as a special case (λ = 0), enabling direct statistical comparison via likelihood ratio tests and calibration diagnostics.

The practical value is greatest for leveraged positions (Morpho Blue at 80-86% LLTV, HyperCore perps) and short time horizons (1-7 days) where jump risk dominates diffusion risk — precisely the regime that matters most for an active market-making operation.

---

## References

- Merton, R.C. (1976). "Option pricing when underlying stock returns are discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.
- Anthias Team. "Anthias Liquip Score: Measuring Liquidation Probability."
- Honoré, P. (1998). "Pitfalls in Estimating Jump-Diffusion Models." Working Paper, Aarhus School of Business.
- Ramezani, C.A. & Zeng, Y. (2007). "Maximum likelihood estimation of the double exponential jump-diffusion process." *Annals of Finance*, 3(4), 487-507.
- Pickands, J. (1975). "Statistical inference using extreme order statistics." *Annals of Statistics*, 3(1), 119-131.
- Balkema, A.A. & de Haan, L. (1974). "Residual life time at great age." *Annals of Probability*, 2(5), 792-804.
