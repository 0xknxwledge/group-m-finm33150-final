"""Tests for model comparison (Merton vs Kou)."""

from __future__ import annotations

import numpy as np
import pytest

from funding_the_fall.models.compare import (
    ModelComparison,
    compare_all_tokens,
    compare_models,
)
from funding_the_fall.models.merton import MertonParams
from funding_the_fall.models.kou import KouParams


# ---------------------------------------------------------------------------
# ModelComparison dataclass
# ---------------------------------------------------------------------------


class TestModelComparison:
    @staticmethod
    def _make_comparison():
        m = MertonParams(
            sigma=0.01,
            lam=0.1,
            mu_j=-0.05,
            sigma_j=0.02,
            mu=0.0,
            log_likelihood=-100,
            aic=210,
            bic=220,
        )
        k = KouParams(
            sigma=0.01,
            lam=0.1,
            p=0.4,
            eta1=10.0,
            eta2=5.0,
            mu=0.0,
            log_likelihood=-95,
            aic=202,
            bic=215,
        )
        return ModelComparison(coin="BTC", n_obs=200, merton=m, kou=k, preferred="kou")

    def test_fields(self):
        c = self._make_comparison()
        assert c.coin == "BTC"
        assert c.n_obs == 200
        assert c.preferred == "kou"

    def test_bic_delta(self):
        c = self._make_comparison()
        assert c.bic_delta == pytest.approx(220 - 215)

    def test_aic_delta(self):
        c = self._make_comparison()
        assert c.aic_delta == pytest.approx(210 - 202)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_returns_model_comparison(self, jump_returns):
        c = compare_models(jump_returns, coin="ETH")
        assert isinstance(c, ModelComparison)

    def test_preferred_consistent_with_bic(self, jump_returns):
        c = compare_models(jump_returns, coin="ETH")
        if c.kou.bic < c.merton.bic:
            assert c.preferred == "kou"
        else:
            assert c.preferred == "merton"

    def test_n_obs_correct(self, jump_returns):
        c = compare_models(jump_returns, coin="SOL")
        assert c.n_obs == len(jump_returns)


# ---------------------------------------------------------------------------
# compare_all_tokens
# ---------------------------------------------------------------------------


class TestCompareAllTokens:
    def test_multiple_tokens(self):
        rng = np.random.default_rng(42)
        returns_dict = {
            "BTC": rng.normal(0, 0.01, size=200),
            "ETH": rng.normal(0, 0.015, size=200),
        }
        results = compare_all_tokens(returns_dict)
        assert set(results.keys()) == {"BTC", "ETH"}
        for coin, comp in results.items():
            assert isinstance(comp, ModelComparison)
            assert comp.coin == coin

    def test_single_token(self):
        rng = np.random.default_rng(42)
        returns_dict = {"SOL": rng.normal(0, 0.01, size=200)}
        results = compare_all_tokens(returns_dict)
        assert len(results) == 1
        assert "SOL" in results

    def test_empty_dict(self):
        results = compare_all_tokens({})
        assert results == {}
