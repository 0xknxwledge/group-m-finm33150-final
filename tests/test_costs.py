"""Tests for the Almgren-Chriss transaction cost model."""

from __future__ import annotations

import pytest

from funding_the_fall.backtest.costs import (
    VENUE_FEES,
    TransactionCostModel,
    make_cost_model,
)


# ---------------------------------------------------------------------------
# TransactionCostModel defaults
# ---------------------------------------------------------------------------


class TestTransactionCostModelDefaults:
    def test_epsilon(self):
        m = TransactionCostModel()
        assert m.epsilon == 0.0003

    def test_eta(self):
        m = TransactionCostModel()
        assert m.eta == 0.0001

    def test_gamma(self):
        m = TransactionCostModel()
        assert m.gamma == 0.00001


# ---------------------------------------------------------------------------
# fixed_cost
# ---------------------------------------------------------------------------


class TestFixedCost:
    def test_positive_notional(self):
        m = TransactionCostModel(epsilon=0.001)
        assert m.fixed_cost(10_000.0) == pytest.approx(10.0)

    def test_negative_notional(self):
        m = TransactionCostModel(epsilon=0.001)
        assert m.fixed_cost(-10_000.0) == pytest.approx(10.0)

    def test_symmetric(self):
        m = TransactionCostModel()
        assert m.fixed_cost(5_000.0) == pytest.approx(m.fixed_cost(-5_000.0))

    def test_zero(self):
        m = TransactionCostModel()
        assert m.fixed_cost(0.0) == 0.0


# ---------------------------------------------------------------------------
# temporary_impact
# ---------------------------------------------------------------------------


class TestTemporaryImpact:
    def test_known_value(self):
        m = TransactionCostModel(eta=0.0001)
        # (0.0001 / 1.0) * 1000^2 = 100.0
        assert m.temporary_impact(1_000.0) == pytest.approx(100.0)

    def test_tau_scaling(self):
        m = TransactionCostModel(eta=0.0001)
        notional = 1_000.0
        # tau=2 should halve the result
        assert m.temporary_impact(notional, tau=2.0) == pytest.approx(
            m.temporary_impact(notional, tau=1.0) / 2.0
        )

    def test_zero_notional(self):
        m = TransactionCostModel()
        assert m.temporary_impact(0.0) == 0.0


# ---------------------------------------------------------------------------
# permanent_impact
# ---------------------------------------------------------------------------


class TestPermanentImpact:
    def test_known_value(self):
        m = TransactionCostModel(gamma=0.00001)
        # 0.5 * 0.00001 * 10_000^2 = 500.0
        assert m.permanent_impact(10_000.0) == pytest.approx(500.0)

    def test_zero_notional(self):
        m = TransactionCostModel()
        assert m.permanent_impact(0.0) == 0.0


# ---------------------------------------------------------------------------
# total_cost (fixed + temporary, NOT permanent)
# ---------------------------------------------------------------------------


class TestTotalCost:
    def test_equals_fixed_plus_temporary(self):
        m = TransactionCostModel()
        notional = 5_000.0
        expected = m.fixed_cost(notional) + m.temporary_impact(notional)
        assert m.total_cost(notional) == pytest.approx(expected)

    def test_excludes_permanent(self):
        m = TransactionCostModel(gamma=1.0)
        notional = 1_000.0
        total = m.total_cost(notional)
        perm = m.permanent_impact(notional)
        assert total != pytest.approx(total + perm)
        assert total == pytest.approx(
            m.fixed_cost(notional) + m.temporary_impact(notional)
        )

    def test_tau_forwarded(self):
        m = TransactionCostModel()
        notional = 2_000.0
        expected = m.fixed_cost(notional) + m.temporary_impact(notional, tau=0.5)
        assert m.total_cost(notional, tau=0.5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# implementation_shortfall — hand-calculated
# ---------------------------------------------------------------------------


class TestImplementationShortfall:
    def test_hand_calculated(self):
        m = TransactionCostModel(epsilon=0.0003, eta=0.0001, gamma=0.00001)
        trades = [1_000.0, 2_000.0, -500.0]
        tau = 1.0

        # X = sum(trades) = 2500
        # eta_tilde = eta - 0.5 * gamma * tau = 0.0001 - 0.000005 = 0.000095
        # permanent = 0.5 * gamma * X^2 = 0.5 * 0.00001 * 6_250_000 = 31.25
        # fixed = epsilon * sum(|nk|) = 0.0003 * 3500 = 1.05
        # temporary = (eta_tilde / tau) * sum(nk^2) = 0.000095 * 5_250_000 = 498.75
        # total = 31.25 + 1.05 + 498.75 = 531.05
        expected = 531.05
        assert m.implementation_shortfall(trades, tau=tau) == pytest.approx(expected)

    def test_single_trade(self):
        m = TransactionCostModel(epsilon=0.0003, eta=0.0001, gamma=0.00001)
        trades = [1_000.0]
        tau = 1.0

        eta_tilde = m.eta - 0.5 * m.gamma * tau
        expected = (
            0.5 * m.gamma * 1_000.0**2
            + m.epsilon * 1_000.0
            + (eta_tilde / tau) * 1_000.0**2
        )
        assert m.implementation_shortfall(trades, tau=tau) == pytest.approx(expected)

    def test_empty_trades(self):
        m = TransactionCostModel()
        assert m.implementation_shortfall([], tau=1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VENUE_FEES
# ---------------------------------------------------------------------------


class TestVenueFees:
    def test_expected_keys(self):
        assert set(VENUE_FEES.keys()) == {"hyperliquid", "binance", "bybit", "dydx"}

    def test_values_positive(self):
        for venue, fee in VENUE_FEES.items():
            assert fee > 0, f"{venue} fee should be positive"


# ---------------------------------------------------------------------------
# make_cost_model
# ---------------------------------------------------------------------------


class TestMakeCostModel:
    def test_known_venue(self):
        m = make_cost_model("binance")
        assert m.epsilon == VENUE_FEES["binance"]

    def test_unknown_venue_default(self):
        m = make_cost_model("unknown_exchange")
        assert m.epsilon == 0.0005

    def test_returns_transaction_cost_model(self):
        m = make_cost_model("hyperliquid")
        assert isinstance(m, TransactionCostModel)
