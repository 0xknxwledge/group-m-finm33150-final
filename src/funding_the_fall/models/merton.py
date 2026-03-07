"""Merton jump-diffusion model — calibration and density.

Owner: John Beecher

Two-stage calibration:
  1. Heuristic: iterative 3σ-filtering separates jump and diffusion components
  2. MLE: L-BFGS-B optimization on the full Merton log-likelihood

The model augments GBM with a compound Poisson jump process:
  dS/S = (μ - λk)dt + σ dW + J dN
where N ~ Poisson(λ), J ~ LogNormal(μ_J, σ_J).

The log-return density is a Poisson-weighted mixture of normals:
  f(r) = Σ_n [e^{-λdt}(λdt)^n/n!] × φ(r; m_n, v_n²)
where m_n = (μ-σ²/2-λk)dt + nμ_J, v_n² = σ²dt + nσ_J².

Jump sizes are symmetric (log-normal). For asymmetric tails, see kou.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class MertonParams:
    """Fitted Merton jump-diffusion parameters."""

    sigma: float  # diffusion volatility (per period)
    lam: float  # jump intensity (jumps per period)
    mu_j: float  # mean jump size (log)
    sigma_j: float  # jump size volatility (log)
    mu: float  # drift (per period)
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    n_params: int = 5


def heuristic_calibration(
    returns: NDArray[np.floating],
    dt: float = 1.0,
    threshold_sigma: float = 3.0,
    n_iterations: int = 3,
) -> MertonParams:
    """Stage 1: heuristic calibration via iterative 3σ-filtering.

    Separates returns into 'diffusion' (within threshold) and 'jump'
    (outside threshold) components. Estimates jump distribution from outliers.
    """
    returns = np.asarray(returns, dtype=np.float64)
    sigma_hat = np.std(returns, ddof=1)
    mu_hat = np.mean(returns)
    mask = np.ones(len(returns), dtype=bool)

    for _ in range(n_iterations):
        threshold = threshold_sigma * sigma_hat
        mask = np.abs(returns - mu_hat) < threshold
        if mask.sum() > 10:
            sigma_hat = np.std(returns[mask], ddof=1)
            mu_hat = np.mean(returns[mask])

    jump_mask = ~mask
    n_jumps = int(jump_mask.sum())
    lam = n_jumps / (len(returns) * dt)

    if n_jumps >= 2:
        mu_j = float(np.mean(returns[jump_mask]))
        sigma_j = float(np.std(returns[jump_mask], ddof=1))
    elif n_jumps == 1:
        mu_j = float(returns[jump_mask][0])
        sigma_j = abs(mu_j) * 0.5
    else:
        mu_j = -2.0 * sigma_hat
        sigma_j = sigma_hat
        lam = 0.005 / dt

    k = np.exp(mu_j + sigma_j**2 / 2) - 1
    mu = mu_hat / dt + lam * k

    return MertonParams(sigma=sigma_hat, lam=lam, mu_j=mu_j, sigma_j=sigma_j, mu=mu)


def merton_log_density(
    x: NDArray[np.floating],
    params: MertonParams,
    dt: float = 1.0,
    n_terms: int = 20,
) -> NDArray[np.floating]:
    """Log-density of the Merton model (Poisson mixture of normals).

    Vectorized over x, loops over n_terms Poisson components.
    Uses logsumexp for numerical stability.
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.exp(params.mu_j + params.sigma_j**2 / 2) - 1
    drift = (params.mu - params.sigma**2 / 2 - params.lam * k) * dt

    lam_dt = max(params.lam * dt, 1e-300)
    log_poisson = -lam_dt
    components = []

    for n in range(n_terms):
        if n > 0:
            log_poisson += np.log(lam_dt) - np.log(n)

        m_n = drift + n * params.mu_j
        v_n2 = params.sigma**2 * dt + n * params.sigma_j**2
        if v_n2 <= 0:
            continue

        log_norm = -0.5 * np.log(2 * np.pi * v_n2) - 0.5 * (x - m_n) ** 2 / v_n2
        components.append(log_poisson + log_norm)

    stacked = np.array(components)
    max_log = np.max(stacked, axis=0)
    return max_log + np.log(
        np.sum(np.exp(stacked - max_log[np.newaxis, :]), axis=0)
    )


def _neg_log_likelihood(theta, returns, dt):
    """Negative log-likelihood for L-BFGS-B."""
    mu, sigma, lam, mu_j, sigma_j = theta
    params = MertonParams(sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j, mu=mu)
    ll = np.sum(merton_log_density(returns, params, dt))
    if not np.isfinite(ll):
        return 1e15
    return -ll


def mle_calibration(
    returns: NDArray[np.floating],
    dt: float = 1.0,
    heuristic_params: MertonParams | None = None,
) -> MertonParams:
    """Stage 2: MLE refinement via L-BFGS-B.

    If heuristic_params is provided, uses them as initial guess.
    Sets log_likelihood, AIC, and BIC on the returned params.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if heuristic_params is None:
        heuristic_params = heuristic_calibration(returns, dt)

    x0 = [
        heuristic_params.mu,
        heuristic_params.sigma,
        heuristic_params.lam,
        heuristic_params.mu_j,
        heuristic_params.sigma_j,
    ]
    bounds = [
        (None, None),  # mu
        (1e-8, None),  # sigma > 0
        (1e-8, None),  # lam > 0
        (None, None),  # mu_j
        (1e-8, None),  # sigma_j > 0
    ]

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(returns, dt),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    mu, sigma, lam, mu_j, sigma_j = result.x
    n = len(returns)
    ll = -result.fun
    nk = 5

    return MertonParams(
        sigma=sigma,
        lam=lam,
        mu_j=mu_j,
        sigma_j=sigma_j,
        mu=mu,
        log_likelihood=ll,
        aic=-2 * ll + 2 * nk,
        bic=-2 * ll + nk * np.log(n),
    )


def calibrate_merton(
    returns: NDArray[np.floating],
    dt: float = 1.0,
) -> MertonParams:
    """Full two-stage calibration: heuristic -> MLE."""
    h = heuristic_calibration(returns, dt)
    return mle_calibration(returns, dt, heuristic_params=h)
