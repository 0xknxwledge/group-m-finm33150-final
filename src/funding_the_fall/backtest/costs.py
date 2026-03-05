"""Transaction cost model — Almgren-Chriss framework.

Owner: Jean Mauratille

Implements the linear impact model from Accumulation Algorithms (lecture notes):

  Permanent impact:  g(v) = γ v
  Temporary impact:  h(v) = ε sgn(v) + (η/τ) n

  Total cost of trading n units in one period:
    n · h(n/τ) = ε|n| + (η/τ) n²

Parameters:
  γ — permanent impact coefficient ($/share per share)
  ε — fixed cost (half the bid-ask spread)
  η — temporary impact coefficient

Calibrate from orderbook depth data:
  γ ≈ from Gatheral's linear permanent impact
  ε ≈ half the quoted spread
  η ≈ from regressing price impact on trade size

See: Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"
     Jusselin & Rosenbaum (2018) — propagator model, square-root law
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransactionCostModel:
    """Almgren-Chriss linear impact transaction cost model."""

    epsilon: float = 0.0003    # half spread (3 bps default)
    eta: float = 0.0001        # temporary impact coefficient
    gamma: float = 0.00001     # permanent impact coefficient

    def fixed_cost(self, notional_usd: float) -> float:
        """Fixed cost component (spread crossing): ε × |notional|."""
        return self.epsilon * abs(notional_usd)

    def temporary_impact(self, notional_usd: float, tau: float = 1.0) -> float:
        """Temporary impact cost: (η/τ) × notional²."""
        return (self.eta / tau) * notional_usd ** 2

    def permanent_impact(self, notional_usd: float) -> float:
        """Permanent impact cost: ½ γ × notional²."""
        return 0.5 * self.gamma * notional_usd ** 2

    def total_cost(self, notional_usd: float, tau: float = 1.0) -> float:
        """Total execution cost for a single trade.

        Returns cost in USD (always positive).
        """
        return (
            self.fixed_cost(notional_usd)
            + self.temporary_impact(notional_usd, tau)
        )

    def implementation_shortfall(
        self, trade_notionals: list[float], tau: float = 1.0,
    ) -> float:
        """Expected implementation shortfall for a sequence of trades.

        E[IS] = ½γX² + ε Σ|nk| + (η̃/τ) Σnk²
        where η̃ = η - ½γτ
        """
        total_quantity = sum(trade_notionals)
        eta_tilde = self.eta - 0.5 * self.gamma * tau
        permanent = 0.5 * self.gamma * total_quantity ** 2
        fixed = self.epsilon * sum(abs(n) for n in trade_notionals)
        temporary = (eta_tilde / tau) * sum(n ** 2 for n in trade_notionals)
        return permanent + fixed + temporary


# ── Per-venue fee schedules (taker fees) ──
VENUE_FEES: dict[str, float] = {
    "hyperliquid": 0.00035,   # 3.5 bps taker
    "binance": 0.0004,        # 4 bps taker (VIP0)
    "bybit": 0.00055,         # 5.5 bps taker
    "dydx": 0.0005,           # 5 bps taker
}


def make_cost_model(venue: str) -> TransactionCostModel:
    """Create a cost model with venue-specific spread estimates."""
    fee = VENUE_FEES.get(venue, 0.0005)
    return TransactionCostModel(epsilon=fee)
