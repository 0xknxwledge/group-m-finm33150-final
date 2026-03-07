"""Tests for the Merton jump-diffusion model."""

from __future__ import annotations

import numpy as np
import pytest

from funding_the_fall.models.merton import (
    MertonParams,
    calibrate_merton,
    heuristic_calibration,
    merton_log_density,
    mle_calibration,
)


# ---------------------------------------------------------------------------
# MertonParams dataclass
# ---------------------------------------------------------------------------


class TestMertonParams:
    def test_defaults(self):
        p = MertonParams(sigma=0.01, lam=0.1, mu_j=-0.05, sigma_j=0.02, mu=0.0)
        assert p.log_likelihood == 0.0
        assert p.aic == 0.0
        assert p.bic == 0.0
        assert p.n_params == 5

    def test_field_access(self):
        p = MertonParams(sigma=0.02, lam=0.5, mu_j=-0.03, sigma_j=0.01, mu=0.001)
        assert p.sigma == 0.02
        assert p.lam == 0.5
        assert p.mu_j == -0.03
        assert p.sigma_j == 0.01
        assert p.mu == 0.001


# ---------------------------------------------------------------------------
# heuristic_calibration
# ---------------------------------------------------------------------------


class TestHeuristicCalibration:
    def test_pure_diffusion(self, diffusion_returns):
        p = heuristic_calibration(diffusion_returns)
        assert p.sigma > 0
        # With no jumps, lambda should be very small
        assert p.lam < 0.1

    def test_known_jumps(self, jump_returns):
        p = heuristic_calibration(jump_returns)
        assert p.sigma > 0
        assert p.lam > 0
        # Jump mean should be negative (we injected negative jumps)
        assert p.mu_j < 0
        assert p.sigma_j > 0

    def test_single_jump_branch(self):
        rng = np.random.default_rng(99)
        returns = rng.normal(0, 0.005, size=200)
        # Insert exactly one outlier
        returns[100] = -0.10
        p = heuristic_calibration(returns)
        assert p.lam > 0
        assert p.sigma_j > 0

    def test_no_jumps_fallback(self):
        # Very tight returns — no outliers beyond 3-sigma
        returns = np.linspace(-0.001, 0.001, 200)
        p = heuristic_calibration(returns)
        assert p.lam == pytest.approx(0.005, abs=1e-6)
        assert p.mu_j < 0

    def test_custom_dt(self, diffusion_returns):
        p1 = heuristic_calibration(diffusion_returns, dt=1.0)
        p2 = heuristic_calibration(diffusion_returns, dt=1 / 24)
        # Lambda should scale inversely with dt
        assert p2.lam != p1.lam


# ---------------------------------------------------------------------------
# merton_log_density
# ---------------------------------------------------------------------------


class TestMertonLogDensity:
    def test_output_shape(self):
        x = np.linspace(-0.1, 0.1, 50)
        p = MertonParams(sigma=0.01, lam=0.1, mu_j=-0.05, sigma_j=0.02, mu=0.0)
        ld = merton_log_density(x, p)
        assert ld.shape == (50,)

    def test_integrates_to_one(self):
        x = np.linspace(-0.5, 0.5, 10_000)
        p = MertonParams(sigma=0.01, lam=0.1, mu_j=-0.05, sigma_j=0.02, mu=0.0)
        density = np.exp(merton_log_density(x, p))
        integral = np.trapezoid(density, x)
        assert integral == pytest.approx(1.0, abs=0.05)

    def test_peak_near_drift(self):
        x = np.linspace(-0.1, 0.1, 1000)
        p = MertonParams(sigma=0.01, lam=0.01, mu_j=0.0, sigma_j=0.01, mu=0.0)
        ld = merton_log_density(x, p)
        peak_idx = np.argmax(ld)
        assert abs(x[peak_idx]) < 0.02


# ---------------------------------------------------------------------------
# mle_calibration
# ---------------------------------------------------------------------------


class TestMleCalibration:
    def test_with_heuristic(self, jump_returns):
        h = heuristic_calibration(jump_returns)
        m = mle_calibration(jump_returns, heuristic_params=h)
        assert m.log_likelihood != 0.0
        assert m.aic != 0.0
        assert m.bic != 0.0

    def test_without_heuristic(self, jump_returns):
        m = mle_calibration(jump_returns)
        assert m.log_likelihood != 0.0

    def test_ll_improvement(self, jump_returns):
        h = heuristic_calibration(jump_returns)
        m = mle_calibration(jump_returns, heuristic_params=h)
        # MLE log-likelihood should be at least as good as heuristic
        h_ll = np.sum(merton_log_density(jump_returns, h))
        assert m.log_likelihood >= h_ll - 1e-6

    def test_aic_bic_formula(self, jump_returns):
        m = mle_calibration(jump_returns)
        n = len(jump_returns)
        expected_aic = -2 * m.log_likelihood + 2 * 5
        expected_bic = -2 * m.log_likelihood + 5 * np.log(n)
        assert m.aic == pytest.approx(expected_aic)
        assert m.bic == pytest.approx(expected_bic)

    def test_positivity_constraints(self, jump_returns):
        m = mle_calibration(jump_returns)
        assert m.sigma > 0
        assert m.lam > 0
        assert m.sigma_j > 0


# ---------------------------------------------------------------------------
# calibrate_merton (end-to-end)
# ---------------------------------------------------------------------------


class TestCalibrateMerton:
    def test_end_to_end(self, jump_returns):
        m = calibrate_merton(jump_returns)
        assert isinstance(m, MertonParams)
        assert m.log_likelihood != 0.0
        assert m.sigma > 0

    def test_known_jump_recovery(self):
        rng = np.random.default_rng(42)
        n = 1000
        sigma, lam, mu_j, sigma_j = 0.01, 0.1, -0.05, 0.02
        diffusion = rng.normal(0, sigma, size=n)
        jump_mask = rng.random(n) < lam
        jumps = rng.normal(mu_j, sigma_j, size=n)
        returns = diffusion.copy()
        returns[jump_mask] += jumps[jump_mask]

        m = calibrate_merton(returns)
        # Loose tolerances — MLE on 1000 points won't recover exactly
        assert m.sigma == pytest.approx(sigma, rel=0.5)
        assert m.mu_j < 0  # negative jump mean recovered
