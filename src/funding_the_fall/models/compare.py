"""Model comparison — Merton vs Kou via AIC/BIC and diagnostic plots.

Owner: John Beecher

Calibrates both models on the same return series and compares:
  - Log-likelihood
  - AIC = -2L + 2k
  - BIC = -2L + k ln(n)
  - Tail fit (QQ plots, empirical vs model tail probabilities)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from funding_the_fall.models.merton import MertonParams, calibrate_merton
from funding_the_fall.models.kou import KouParams, calibrate_kou


@dataclass
class ModelComparison:
    """Side-by-side comparison of Merton and Kou fits."""

    coin: str
    n_obs: int
    merton: MertonParams
    kou: KouParams
    preferred: str             # "merton" or "kou" based on BIC

    @property
    def bic_delta(self) -> float:
        """BIC(Merton) - BIC(Kou). Positive favors Kou."""
        return self.merton.bic - self.kou.bic

    @property
    def aic_delta(self) -> float:
        """AIC(Merton) - AIC(Kou). Positive favors Kou."""
        return self.merton.aic - self.kou.aic


def compare_models(
    returns: NDArray[np.floating],
    coin: str = "",
    dt: float = 1.0,
) -> ModelComparison:
    """Calibrate both Merton and Kou, return comparison.

    The preferred model is chosen by BIC (penalizes Kou's extra parameter).
    """
    m = calibrate_merton(returns, dt)
    k = calibrate_kou(returns, dt)
    preferred = "kou" if k.bic < m.bic else "merton"
    return ModelComparison(
        coin=coin,
        n_obs=len(returns),
        merton=m,
        kou=k,
        preferred=preferred,
    )


def compare_all_tokens(
    returns_dict: dict[str, NDArray[np.floating]],
    dt: float = 1.0,
) -> dict[str, ModelComparison]:
    """Run model comparison for every token in the universe."""
    return {
        coin: compare_models(rets, coin=coin, dt=dt)
        for coin, rets in returns_dict.items()
    }
