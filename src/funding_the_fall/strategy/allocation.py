"""Portfolio allocation — combines carry and cascade signals.

Owner: Jean-Luc Choiseul

Allocation split:
  - 70% NAV to funding carry strategy
  - 30% NAV to cascade positioning overlay

When cascade risk is elevated, the cascade allocation may increase
at the expense of carry (dynamic rebalancing).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PositionTarget:
    """A target position to be executed."""

    timestamp: pd.Timestamp
    coin: str
    venue: str
    side: str               # "long" or "short"
    notional_usd: float     # target notional exposure
    strategy: str           # "carry" or "cascade"


def allocate_positions(
    carry_signals: list,
    cascade_signal: dict,
    nav: float,
    carry_weight: float = 0.70,
    cascade_weight: float = 0.30,
    max_leverage: float = 5.0,
    max_single_exchange_pct: float = 0.40,
    max_net_delta_pct: float = 0.10,
) -> list[PositionTarget]:
    """Convert signals into sized position targets respecting risk limits.

    Risk controls:
      - Total gross leverage ≤ max_leverage × NAV
      - Exposure to any single exchange ≤ max_single_exchange_pct × NAV
      - Net delta across all positions ≤ max_net_delta_pct × NAV

    Position size for carry is proportional to spread attractiveness.
    Cascade positions are sized by cascade risk score.
    """
    raise NotImplementedError
