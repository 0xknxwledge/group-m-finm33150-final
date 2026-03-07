"""Kou double-exponential jump-diffusion model — calibration and density.

Owner: John Beecher

The Kou (2002) model augments GBM with asymmetric jumps:
  dS/S = (μ - λκ)dt + σ dW + J dN

where N ~ Poisson(λ) and jump sizes J have a double-exponential distribution:
  f_J(x) = p · η₁ · exp(-η₁ x) · 1_{x≥0}  +  (1-p) · η₂ · exp(η₂ x) · 1_{x<0}

Parameters:
  σ   — diffusion volatility
  λ   — jump intensity (jumps per period)
  p   — probability a jump is positive (upward)
  η₁  — rate parameter for positive jumps (η₁ > 1 required for finite expectation)
  η₂  — rate parameter for negative jumps (η₂ > 0)
  μ   — drift

Density is recovered via FFT of the closed-form characteristic function:
  φ(u) = exp{dt·[iuμ̃ - σ²u²/2 + λ(pη₁/(η₁-iu) + (1-p)η₂/(η₂+iu) - 1)]}

This avoids the intractable k-fold convolution of double-exponential
distributions that a direct Poisson mixture approach would require.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class KouParams:
    """Fitted Kou double-exponential jump-diffusion parameters."""

    sigma: float  # diffusion volatility (per period)
    lam: float  # jump intensity (jumps per period)
    p: float  # probability jump is positive (0 < p < 1)
    eta1: float  # positive jump rate (η₁ > 1)
    eta2: float  # negative jump rate (η₂ > 0)
    mu: float  # drift (per period)
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    n_params: int = 6

    @property
    def mean_positive_jump(self) -> float:
        """Expected size of a positive jump: 1/η₁."""
        return 1.0 / self.eta1

    @property
    def mean_negative_jump(self) -> float:
        """Expected size of a negative jump: -1/η₂."""
        return -1.0 / self.eta2

    @property
    def jump_mean(self) -> float:
        """Expected jump size: p/η₁ - (1-p)/η₂."""
        return self.p / self.eta1 - (1 - self.p) / self.eta2

    @property
    def tail_asymmetry(self) -> float:
        """Ratio of down-jump intensity to up-jump intensity.

        Values > 1 indicate heavier left tail (crash-prone).
        """
        down_intensity = self.lam * (1 - self.p)
        up_intensity = self.lam * self.p
        if up_intensity == 0:
            return float("inf")
        return down_intensity / up_intensity


# ---------------------------------------------------------------------------
# FFT density recovery
# ---------------------------------------------------------------------------

_FFT_N = 2**14  # 16384 grid points
_FFT_L = 0.5  # half-width of return domain (±50% covers any hourly return)


def _kou_cf(
    u: NDArray,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    mu: float,
    dt: float,
) -> NDArray:
    """Characteristic function of the Kou log-return over interval dt.

    φ(u) = exp{dt·[iu·μ̃ - σ²u²/2 + λ·(p·η₁/(η₁-iu) + (1-p)·η₂/(η₂+iu) - 1)]}

    where μ̃ = μ - σ²/2 - λκ, κ = p·η₁/(η₁-1) + (1-p)·η₂/(η₂+1) - 1.
    """
    kappa = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    mu_tilde = mu - sigma**2 / 2 - lam * kappa

    drift = 1j * u * mu_tilde * dt
    diffusion = -0.5 * sigma**2 * u**2 * dt
    jump_mgf = p * eta1 / (eta1 - 1j * u) + (1 - p) * eta2 / (eta2 + 1j * u)
    jump = lam * dt * (jump_mgf - 1)

    return np.exp(drift + diffusion + jump)


def _kou_density_fft(
    x_eval: NDArray[np.floating],
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    mu: float,
    dt: float,
) -> NDArray[np.floating]:
    """Compute Kou density at arbitrary points via FFT inversion.

    Recovers f(x) = (1/2π) ∫ φ(u) e^{-iux} du on a uniform grid,
    then linearly interpolates to x_eval. The Gaussian diffusion term
    ensures φ(u) decays as e^{-σ²u²dt/2}, so no dampening is needed.
    """
    N = _FFT_N
    L = _FFT_L
    dx = 2 * L / N
    x_grid = -L + np.arange(N) * dx

    # Angular frequencies in numpy FFT order
    u = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # CF evaluated on frequency grid, phase-shifted for x_grid starting at -L
    cf = _kou_cf(u, sigma, lam, p, eta1, eta2, mu, dt)
    phase = np.exp(1j * u * L)

    # FFT gives: Σ_j (cf·phase)[j] e^{-i2πjk/N} = density * (N·dx)
    density_grid = np.fft.fft(cf * phase).real / (N * dx)

    # Interpolate to requested points
    density = np.interp(x_eval, x_grid, density_grid, left=1e-300, right=1e-300)
    return np.maximum(density, 1e-300)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def heuristic_calibration(
    returns: NDArray[np.floating],
    dt: float = 1.0,
    threshold_sigma: float = 3.0,
    n_iterations: int = 3,
) -> KouParams:
    """Stage 1: heuristic calibration via iterative 3σ-filtering.

    Like Merton heuristic, but fits separate exponentials to positive
    and negative jumps to estimate p, η₁, η₂.
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
        jump_rets = returns[jump_mask]
        pos_jumps = jump_rets[jump_rets > 0]
        neg_jumps = jump_rets[jump_rets < 0]

        p = len(pos_jumps) / n_jumps if n_jumps > 0 else 0.5
        # η = 1/mean gives the MLE of the exponential rate parameter
        eta1 = max(1.0 / np.mean(pos_jumps), 1.5) if len(pos_jumps) > 0 else 10.0
        eta2 = (
            max(1.0 / np.mean(np.abs(neg_jumps)), 0.5) if len(neg_jumps) > 0 else 10.0
        )
    elif n_jumps == 1:
        jump_ret = float(returns[jump_mask][0])
        if jump_ret > 0:
            p, eta1, eta2 = 0.7, max(1.0 / jump_ret, 1.5), 10.0
        else:
            p, eta1, eta2 = 0.3, 10.0, max(1.0 / abs(jump_ret), 0.5)
        lam = max(lam, 0.005 / dt)
    else:
        p, eta1, eta2 = 0.4, 10.0, 5.0
        lam = 0.005 / dt

    kappa = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    mu = mu_hat / dt + lam * kappa

    return KouParams(sigma=sigma_hat, lam=lam, p=p, eta1=eta1, eta2=eta2, mu=mu)


def kou_log_density(
    x: NDArray[np.floating],
    params: KouParams,
    dt: float = 1.0,
    n_terms: int = 20,
) -> NDArray[np.floating]:
    """Log-density of the Kou model via FFT of the characteristic function.

    The n_terms parameter is unused (kept for API compatibility with Merton).
    The FFT grid size is controlled by _FFT_N.
    """
    x = np.asarray(x, dtype=np.float64)
    density = _kou_density_fft(
        x, params.sigma, params.lam, params.p, params.eta1, params.eta2, params.mu, dt
    )
    return np.log(density)


def _neg_log_likelihood(theta, returns, dt):
    """Negative log-likelihood for L-BFGS-B."""
    sigma, lam, p, eta1, eta2, mu = theta
    density = _kou_density_fft(returns, sigma, lam, p, eta1, eta2, mu, dt)
    ll = np.sum(np.log(density))
    if not np.isfinite(ll):
        return 1e15
    return -ll


def mle_calibration(
    returns: NDArray[np.floating],
    dt: float = 1.0,
    heuristic_params: KouParams | None = None,
) -> KouParams:
    """Stage 2: MLE refinement via L-BFGS-B.

    Bounds:
      σ > 0, λ > 0, 0 < p < 1, η₁ > 1, η₂ > 0

    Sets log_likelihood, AIC, and BIC on the returned params.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if heuristic_params is None:
        heuristic_params = heuristic_calibration(returns, dt)

    x0 = [
        heuristic_params.sigma,
        heuristic_params.lam,
        heuristic_params.p,
        heuristic_params.eta1,
        heuristic_params.eta2,
        heuristic_params.mu,
    ]
    bounds = [
        (1e-8, None),  # sigma > 0
        (1e-8, None),  # lam > 0
        (0.01, 0.99),  # 0 < p < 1
        (1.01, 100.0),  # eta1 > 1 (finite mean requirement)
        (0.1, 100.0),  # eta2 > 0
        (None, None),  # mu
    ]

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(returns, dt),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    sigma, lam, p, eta1, eta2, mu = result.x
    n = len(returns)
    ll = -result.fun
    nk = 6

    return KouParams(
        sigma=sigma,
        lam=lam,
        p=p,
        eta1=eta1,
        eta2=eta2,
        mu=mu,
        log_likelihood=ll,
        aic=-2 * ll + 2 * nk,
        bic=-2 * ll + nk * np.log(n),
    )


def calibrate_kou(
    returns: NDArray[np.floating],
    dt: float = 1.0,
) -> KouParams:
    """Full two-stage calibration: heuristic -> MLE."""
    h = heuristic_calibration(returns, dt)
    return mle_calibration(returns, dt, heuristic_params=h)
