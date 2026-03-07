from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def diffusion_returns(rng):
    """Pure diffusion returns (no jumps) — N(0.0001, 0.01)."""
    return rng.normal(0.0001, 0.01, size=500)


@pytest.fixture
def jump_returns(rng):
    """Returns with embedded jumps: diffusion + Poisson jumps."""
    n = 500
    diffusion = rng.normal(0.0001, 0.01, size=n)
    # ~5% of observations are jumps
    jump_mask = rng.random(n) < 0.05
    jumps = rng.normal(-0.05, 0.02, size=n)
    returns = diffusion.copy()
    returns[jump_mask] += jumps[jump_mask]
    return returns
