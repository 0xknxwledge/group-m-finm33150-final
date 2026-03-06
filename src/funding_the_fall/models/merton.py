"""Merton jump-diffusion model — calibration and density.

Owner: John Beecher

Two-stage calibration:
  1. Heuristic: iterative 3σ-filtering separates jump and diffusion components
  2. MLE: L-BFGS-B optimization on the full Merton log-likelihood

The model augments GBM with a compound Poisson jump process:
  dS/S = (μ - λk)dt + σ dW + J dN
where N ~ Poisson(λ), J ~ LogNormal(μ_J, σ_J).

Jump sizes are symmetric (log-normal). For asymmetric tails, see kou.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
    (outside threshold) components.
    """
    raise NotImplementedError


def merton_log_density(
    x: NDArray[np.floating],
    params: MertonParams,
    dt: float = 1.0,
    n_terms: int = 20,
) -> NDArray[np.floating]:
    """Log-density of the Merton model (sum over Poisson terms).

    Uses n_terms in the Poisson mixture expansion.
    """
    raise NotImplementedError


def mle_calibration(
    returns: NDArray[np.floating],
    dt: float = 1.0,
    heuristic_params: MertonParams | None = None,
) -> MertonParams:
    """Stage 2: MLE refinement via L-BFGS-B.

    If heuristic_params is provided, uses them as initial guess.
    Sets log_likelihood, AIC, and BIC on the returned params.
    """
    raise NotImplementedError


def calibrate_merton(
    returns: NDArray[np.floating],
    dt: float = 1.0,
) -> MertonParams:
    """Full two-stage calibration: heuristic → MLE."""
    h = heuristic_calibration(returns, dt)
    return mle_calibration(returns, dt, heuristic_params=h)
