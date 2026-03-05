"""Kou double-exponential jump-diffusion model — calibration and density.

Owner: John Beecher

The Kou (2002) model augments GBM with asymmetric jumps:
  dS/S = (μ - λk)dt + σ dW + J dN

where N ~ Poisson(λ) and jump sizes J have a double-exponential distribution:
  f_J(x) = p · η₁ · exp(-η₁ x) · 1_{x≥0}  +  (1-p) · η₂ · exp(η₂ x) · 1_{x<0}

Parameters:
  σ   — diffusion volatility
  λ   — jump intensity (jumps per period)
  p   — probability a jump is positive (upward)
  η₁  — rate parameter for positive jumps (η₁ > 1 required for finite expectation)
  η₂  — rate parameter for negative jumps (η₂ > 0)
  μ   — drift

Advantages over Merton for our use case:
  - Asymmetric tails: down-jumps (liquidation cascades) can be more frequent
    and heavier than up-jumps. Crypto empirically has negative skew.
  - Memoryless property: given that a jump exceeds a liquidation threshold,
    the excess is still exponential — analytically convenient for cascade model.
  - Semi-closed-form Laplace transform enables option pricing if needed.

We compare Merton vs Kou via AIC/BIC in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class KouParams:
    """Fitted Kou double-exponential jump-diffusion parameters."""

    sigma: float       # diffusion volatility (per period)
    lam: float         # jump intensity (jumps per period)
    p: float           # probability jump is positive (0 < p < 1)
    eta1: float        # positive jump rate (η₁ > 1)
    eta2: float        # negative jump rate (η₂ > 0)
    mu: float          # drift (per period)
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


def heuristic_calibration(
    returns: NDArray[np.floating],
    dt: float = 1.0,
    threshold_sigma: float = 3.0,
    n_iterations: int = 3,
) -> KouParams:
    """Stage 1: heuristic calibration via iterative 3σ-filtering.

    Like Merton heuristic, but fits separate exponentials to positive
    and negative jumps.
    """
    raise NotImplementedError


def kou_log_density(
    x: NDArray[np.floating],
    params: KouParams,
    dt: float = 1.0,
    n_terms: int = 20,
) -> NDArray[np.floating]:
    """Log-density of the Kou model.

    The density is a Poisson mixture where each term convolves the
    Gaussian diffusion with the sum of k double-exponential jumps.
    For k jumps, the jump-size sum distribution is computed via
    convolution of the double-exponential.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def calibrate_kou(
    returns: NDArray[np.floating],
    dt: float = 1.0,
) -> KouParams:
    """Full two-stage calibration: heuristic → MLE."""
    h = heuristic_calibration(returns, dt)
    return mle_calibration(returns, dt, heuristic_params=h)
