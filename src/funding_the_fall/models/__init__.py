"""Models — jump-diffusion calibration (Merton + Kou) and cascade simulator."""

from funding_the_fall.models.merton import calibrate_merton, merton_log_density
from funding_the_fall.models.kou import calibrate_kou, kou_log_density
from funding_the_fall.models.compare import compare_models, compare_all_tokens
from funding_the_fall.models.cascade import (
    simulate_cascade,
    cascade_risk_signal,
    per_coin_risk_signals,
    build_positions_from_oi,
    compute_amplification_curve,
    sensitivity_to_leverage,
    sensitivity_to_depth,
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
]
