"""Models — jump-diffusion calibration (Merton + Kou) and cascade simulator."""

from funding_the_fall.models.merton import calibrate_merton, merton_log_density
from funding_the_fall.models.kou import calibrate_kou, kou_log_density
from funding_the_fall.models.compare import compare_models, compare_all_tokens
from funding_the_fall.models.cascade import (
    simulate_cascade,
    cascade_risk_signal,
    per_coin_risk_signals,
    build_positions_from_oi,
    build_positions_tiered,
    compute_amplification_curve,
    sensitivity_to_leverage,
    sensitivity_to_depth,
    depth_by_coin,
    validate_cascade,
    CascadeSignal,
    generate_cascade_signals,
    MAX_LEVERAGE,
)
from funding_the_fall.models.risk import (
    jump_weighted_risk,
    jump_weighted_risk_all_coins,
)

__all__ = [
    "calibrate_merton",
    "merton_log_density",
    "calibrate_kou",
    "kou_log_density",
    "compare_models",
    "compare_all_tokens",
    "simulate_cascade",
    "cascade_risk_signal",
    "per_coin_risk_signals",
    "build_positions_from_oi",
    "compute_amplification_curve",
    "sensitivity_to_leverage",
    "sensitivity_to_depth",
    "depth_by_coin",
    "build_positions_tiered",
    "validate_cascade",
    "CascadeSignal",
    "generate_cascade_signals",
    "MAX_LEVERAGE",
    "jump_weighted_risk",
    "jump_weighted_risk_all_coins",
]
