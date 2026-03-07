"""Tests for the Kou double-exponential jump-diffusion model."""

from __future__ import annotations

import numpy as np
import pytest

from funding_the_fall.models.kou import (
    KouParams,
    calibrate_kou,
    heuristic_calibration,
    kou_log_density,
    mle_calibration,
)


# ---------------------------------------------------------------------------
# KouParams dataclass and properties
# ---------------------------------------------------------------------------


class TestKouParams:
    def test_defaults(self):
        p = KouParams(sigma=0.01, lam=0.1, p=0.4, eta1=10.0, eta2=5.0, mu=0.0)
        assert p.log_likelihood == 0.0
        assert p.aic == 0.0
        assert p.bic == 0.0
        assert p.n_params == 6

    def test_mean_positive_jump(self):
        p = KouParams(sigma=0.01, lam=0.1, p=0.4, eta1=10.0, eta2=5.0, mu=0.0)
        assert p.mean_positive_jump == pytest.approx(0.1)

    def test_mean_negative_jump(self):
        p = KouParams(sigma=0.01, lam=0.1, p=0.4, eta1=10.0, eta2=5.0, mu=0.0)
        assert p.mean_negative_jump == pytest.approx(-0.2)

    def test_jump_mean(self):
        p = KouParams(sigma=0.01, lam=0.1, p=0.4, eta1=10.0, eta2=5.0, mu=0.0)
        expected = 0.4 / 10.0 - 0.6 / 5.0
        assert p.jump_mean == pytest.approx(expected)

    def test_tail_asymmetry(self):
        p = KouParams(sigma=0.01, lam=0.1, p=0.3, eta1=10.0, eta2=5.0, mu=0.0)
        # down = 0.1 * 0.7, up = 0.1 * 0.3
        assert p.tail_asymmetry == pytest.approx(0.7 / 0.3)

    def test_tail_asymmetry_zero_p(self):
        p = KouParams(sigma=0.01, lam=0.1, p=0.0, eta1=10.0, eta2=5.0, mu=0.0)
        assert p.tail_asymmetry == float("inf")


# ---------------------------------------------------------------------------
# heuristic_calibration
# ---------------------------------------------------------------------------


class TestKouHeuristicCalibration:
    def test_asymmetric_jumps(self, jump_returns):
        p = heuristic_calibration(jump_returns)
        assert p.sigma > 0
        assert p.lam > 0
        assert 0 <= p.p <= 1
        assert p.eta1 >= 1.0
        assert p.eta2 > 0

    def test_eta1_at_least_1_5(self, jump_returns):
        p = heuristic_calibration(jump_returns)
        assert p.eta1 >= 1.5

    def test_single_positive_jump_branch(self):
        rng = np.random.default_rng(99)
        returns = rng.normal(0, 0.005, size=200)
        returns[100] = 0.10  # single positive outlier
        p = heuristic_calibration(returns)
        assert p.p == pytest.approx(0.7)

    def test_single_negative_jump_branch(self):
        rng = np.random.default_rng(99)
        returns = rng.normal(0, 0.005, size=200)
        returns[100] = -0.10  # single negative outlier
        p = heuristic_calibration(returns)
        assert p.p == pytest.approx(0.3)

    def test_no_jumps_fallback(self):
        returns = np.linspace(-0.001, 0.001, 200)
        p = heuristic_calibration(returns)
        assert p.p == pytest.approx(0.4)
        assert p.eta1 == pytest.approx(10.0)
        assert p.eta2 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# kou_log_density
# ---------------------------------------------------------------------------


class TestKouLogDensity:
    def test_output_shape(self):
        x = np.linspace(-0.1, 0.1, 50)
        p = KouParams(sigma=0.01, lam=0.1, p=0.4, eta1=10.0, eta2=5.0, mu=0.0)
        ld = kou_log_density(x, p)
        assert ld.shape == (50,)

    def test_integrates_to_one(self):
        x = np.linspace(-0.3, 0.3, 10_000)
        p = KouParams(sigma=0.01, lam=0.1, p=0.4, eta1=10.0, eta2=5.0, mu=0.0)
        density = np.exp(kou_log_density(x, p))
        integral = np.trapz(density, x)
        assert integral == pytest.approx(1.0, abs=0.1)

    def test_finite_values(self):
        x = np.linspace(-0.1, 0.1, 100)
        p = KouParams(sigma=0.01, lam=0.1, p=0.4, eta1=10.0, eta2=5.0, mu=0.0)
        ld = kou_log_density(x, p)
        assert np.all(np.isfinite(ld))


# ---------------------------------------------------------------------------
# mle_calibration
# ---------------------------------------------------------------------------


class TestKouMleCalibration:
    def test_ll_aic_bic_populated(self, jump_returns):
        m = mle_calibration(jump_returns)
        assert m.log_likelihood != 0.0
        assert m.aic != 0.0
        assert m.bic != 0.0

    def test_bounds_enforced(self, jump_returns):
        m = mle_calibration(jump_returns)
        assert m.sigma > 0
        assert m.lam > 0
        assert 0.01 <= m.p <= 0.99
        assert m.eta1 >= 1.01
        assert m.eta2 >= 0.1

    def test_aic_bic_formula(self, jump_returns):
        m = mle_calibration(jump_returns)
        n = len(jump_returns)
        expected_aic = -2 * m.log_likelihood + 2 * 6
        expected_bic = -2 * m.log_likelihood + 6 * np.log(n)
        assert m.aic == pytest.approx(expected_aic)
        assert m.bic == pytest.approx(expected_bic)


# ---------------------------------------------------------------------------
# calibrate_kou (end-to-end)
# ---------------------------------------------------------------------------


class TestCalibrateKou:
    def test_end_to_end(self, jump_returns):
        k = calibrate_kou(jump_returns)
        assert isinstance(k, KouParams)
        assert k.log_likelihood != 0.0
        assert k.sigma > 0

    def test_asymmetry_recovery(self):
        rng = np.random.default_rng(42)
        n = 500
        diffusion = rng.normal(0, 0.01, size=n)
        # Inject mostly negative jumps (p_true ~ 0.2)
        jump_mask = rng.random(n) < 0.08
        jump_signs = rng.random(n) < 0.2  # 20% positive
        jumps = np.where(
            jump_signs, rng.exponential(0.03, n), -rng.exponential(0.05, n)
        )
        returns = diffusion.copy()
        returns[jump_mask] += jumps[jump_mask]

        k = calibrate_kou(returns)
        # Should detect heavier left tail
        assert k.p < 0.5 or k.tail_asymmetry > 1.0
