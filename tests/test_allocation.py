"""Tests for portfolio allocation."""

from __future__ import annotations

import pandas as pd
import pytest

from funding_the_fall.strategy.allocation import (
    PositionTarget,
    _enforce_risk_limits,
    allocate_positions,
)
from funding_the_fall.strategy.carry import CarrySignal


def _make_carry_signal(
    action: str = "enter",
    spread: float = 0.15,
    coin: str = "BTC",
    long_venue: str = "binance",
    short_venue: str = "bybit",
) -> CarrySignal:
    return CarrySignal(
        timestamp=pd.Timestamp("2025-01-01"),
        coin=coin,
        long_venue=long_venue,
        short_venue=short_venue,
        spread=spread,
        action=action,
    )


def _zero_cascade() -> dict:
    return {
        "risk_score": 0.0,
        "critical_shock": None,
        "amplification_at_5pct": 1.0,
        "signal": False,
    }


def _high_cascade() -> dict:
    return {
        "risk_score": 0.8,
        "critical_shock": 0.05,
        "amplification_at_5pct": 2.5,
        "signal": True,
    }


# ---------------------------------------------------------------------------
# allocate_positions
# ---------------------------------------------------------------------------


class TestAllocatePositions:
    def test_carry_only_no_risk(self):
        signals = [_make_carry_signal()]
        targets = allocate_positions(signals, _zero_cascade(), nav=1_000_000)
        # Delta-neutral: should have long + short
        longs = [t for t in targets if t.side == "long"]
        shorts = [t for t in targets if t.side == "short"]
        assert len(longs) >= 1
        assert len(shorts) >= 1

    def test_carry_scaling_with_risk(self):
        signals = [_make_carry_signal()]
        nav = 1_000_000
        # Relax risk limits so they don't mask the carry scaling effect
        kwargs = dict(max_single_exchange_pct=1.0, max_net_delta_pct=1.0)
        t_low = allocate_positions(signals, _zero_cascade(), nav=nav, **kwargs)
        t_high = allocate_positions(signals, _high_cascade(), nav=nav, **kwargs)
        carry_low = sum(t.notional_usd for t in t_low if t.strategy == "carry")
        carry_high = sum(t.notional_usd for t in t_high if t.strategy == "carry")
        assert carry_high < carry_low

    def test_cascade_budget(self):
        signals = [_make_carry_signal()]
        nav = 1_000_000
        per_coin = {"BTC": {"risk_score": 0.8}}
        targets = allocate_positions(
            signals, _high_cascade(), nav=nav, per_coin_signals=per_coin
        )
        cascade_targets = [t for t in targets if t.strategy == "cascade"]
        assert len(cascade_targets) >= 1
        assert all(t.side == "short" for t in cascade_targets)

    def test_delta_neutral_pairs(self):
        signals = [_make_carry_signal()]
        targets = allocate_positions(signals, _zero_cascade(), nav=1_000_000)
        carry_targets = [t for t in targets if t.strategy == "carry"]
        long_not = sum(t.notional_usd for t in carry_targets if t.side == "long")
        short_not = sum(t.notional_usd for t in carry_targets if t.side == "short")
        assert long_not == pytest.approx(short_not)

    def test_spread_weighting(self):
        sig1 = _make_carry_signal(spread=0.30, coin="BTC")
        sig2 = _make_carry_signal(
            spread=0.10, coin="ETH", long_venue="okx", short_venue="kraken"
        )
        targets = allocate_positions([sig1, sig2], _zero_cascade(), nav=1_000_000)
        btc_not = sum(t.notional_usd for t in targets if t.coin == "BTC")
        eth_not = sum(t.notional_usd for t in targets if t.coin == "ETH")
        assert btc_not > eth_not

    def test_no_carry_signals(self):
        targets = allocate_positions([], _zero_cascade(), nav=1_000_000)
        assert len(targets) == 0

    def test_cascade_with_per_coin_signals(self):
        per_coin = {
            "BTC": {"risk_score": 0.9},
            "ETH": {"risk_score": 0.6},
        }
        targets = allocate_positions(
            [_make_carry_signal()],
            _high_cascade(),
            nav=1_000_000,
            per_coin_signals=per_coin,
        )
        cascade_targets = [t for t in targets if t.strategy == "cascade"]
        btc_casc = sum(t.notional_usd for t in cascade_targets if t.coin == "BTC")
        eth_casc = sum(t.notional_usd for t in cascade_targets if t.coin == "ETH")
        # BTC has higher risk_score → more weight
        assert btc_casc > eth_casc

    def test_low_risk_filtering(self):
        per_coin = {
            "BTC": {"risk_score": 0.3},  # below 0.5 threshold
        }
        targets = allocate_positions(
            [_make_carry_signal()],
            _high_cascade(),
            nav=1_000_000,
            per_coin_signals=per_coin,
        )
        cascade_targets = [t for t in targets if t.strategy == "cascade"]
        assert len(cascade_targets) == 0

    def test_exit_signals_ignored(self):
        signals = [_make_carry_signal(action="exit")]
        targets = allocate_positions(signals, _zero_cascade(), nav=1_000_000)
        carry_targets = [t for t in targets if t.strategy == "carry"]
        assert len(carry_targets) == 0


# ---------------------------------------------------------------------------
# _enforce_risk_limits
# ---------------------------------------------------------------------------


def _pt(coin, venue, side, notional, strategy="carry", leverage=4.0):
    """Helper to build a PositionTarget with margin-based fields."""
    return PositionTarget(
        pd.Timestamp("2025-01-01"), coin, venue, side,
        notional, notional / leverage, leverage, strategy,
    )


class TestEnforceRiskLimits:
    def test_gross_leverage_cap(self):
        nav = 100_000
        targets = [_pt("BTC", "binance", "long", 300_000), _pt("BTC", "bybit", "short", 300_000)]
        result = _enforce_risk_limits(targets, nav, 5.0, 1.0, 1.0)
        gross = sum(t.notional_usd for t in result)
        assert gross <= 5.0 * nav + 1e-6

    def test_single_exchange_cap(self):
        nav = 100_000
        targets = [_pt("BTC", "binance", "long", 200_000), _pt("ETH", "binance", "short", 200_000)]
        result = _enforce_risk_limits(targets, nav, 10.0, 0.30, 1.0)
        binance_total = sum(t.notional_usd for t in result if t.venue == "binance")
        assert binance_total <= 0.30 * nav + 1e-6

    def test_net_delta_cap(self):
        nav = 100_000
        targets = [_pt("BTC", "binance", "long", 200_000), _pt("BTC", "bybit", "short", 50_000)]
        result = _enforce_risk_limits(targets, nav, 10.0, 1.0, 0.10)
        net = sum(t.notional_usd * (1 if t.side == "long" else -1) for t in result)
        assert abs(net) <= 0.10 * nav + 1e-6

    def test_empty_targets(self):
        result = _enforce_risk_limits([], 100_000, 5.0, 0.40, 0.10)
        assert result == []

    def test_zero_nav(self):
        targets = [_pt("BTC", "binance", "long", 100)]
        result = _enforce_risk_limits(targets, 0.0, 5.0, 0.40, 0.10)
        # Should return targets unchanged (nav <= 0 early return)
        assert len(result) == len(targets)
