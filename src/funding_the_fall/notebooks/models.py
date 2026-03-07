from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from funding_the_fall.models.cascade import (
    build_positions_tiered,
    cascade_risk_signal,
    per_coin_risk_signals,
    _venue_tiers,
    MAX_LEVERAGE,
)
from funding_the_fall.models.compare import compare_all_tokens
from funding_the_fall.models.kou import calibrate_kou, kou_log_density
from funding_the_fall.models.merton import calibrate_merton, merton_log_density
from funding_the_fall.models.risk import jump_weighted_risk_all_coins

from .config import COINS, COIN_COLORS


def compute_returns(candles: pl.DataFrame) -> dict[str, np.ndarray]:
    prices = (
        candles.sort("timestamp")
        .group_by(["coin", "timestamp"])
        .agg(pl.col("c").mean().alias("close"))
        .sort("timestamp")
    )
    returns_dict = {}
    for coin in COINS:
        close = prices.filter(pl.col("coin") == coin).sort("timestamp")["close"].to_numpy()
        lr = np.diff(np.log(close))
        lr = lr[np.isfinite(lr)]
        returns_dict[coin] = lr
    return returns_dict


def calibrate_merton_all(
    returns_dict: dict[str, np.ndarray], dt: float = 1.0
) -> tuple[dict, pd.DataFrame]:
    merton_results = {}
    rows = []
    for coin in COINS:
        m = calibrate_merton(returns_dict[coin], dt=dt)
        merton_results[coin] = m
        rows.append(
            {
                "Coin": coin,
                "sigma": f"{m.sigma:.5f}",
                "lambda": f"{m.lam:.4f}",
                "mu_J": f"{m.mu_j:.5f}",
                "sigma_J": f"{m.sigma_j:.5f}",
                "LL": f"{m.log_likelihood:.1f}",
                "AIC": f"{m.aic:.1f}",
                "BIC": f"{m.bic:.1f}",
            }
        )
    return merton_results, pd.DataFrame(rows)


def plot_merton_density(returns_dict: dict[str, np.ndarray], merton_results: dict):
    fig, axes = plt.subplots(1, len(COINS), figsize=(4 * len(COINS), 4))
    for ax, coin in zip(axes, COINS):
        lr = returns_dict[coin]
        ax.grid(False)
        ax.hist(lr, bins=100, density=True, alpha=0.5, color=COIN_COLORS[coin], label="Empirical")
        x_grid = np.linspace(lr.min(), lr.max(), 500)
        log_d = merton_log_density(x_grid, merton_results[coin], dt=1.0)
        ax.plot(x_grid, np.exp(log_d), "-", color="gray", lw=1.5, label="Merton")
        ax.set_title(coin, fontweight="bold")
        ax.set_xlim(-0.05, 0.05)
        if coin == COINS[0]:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.suptitle("Merton Jump-Diffusion Fit: Empirical vs Model Density", fontsize=13)
    fig.tight_layout()
    plt.show()


def calibrate_kou_all(
    returns_dict: dict[str, np.ndarray], dt: float = 1.0
) -> tuple[dict, pd.DataFrame]:
    kou_results = {}
    rows = []
    for coin in COINS:
        k = calibrate_kou(returns_dict[coin], dt=dt)
        kou_results[coin] = k
        rows.append(
            {
                "Coin": coin,
                "sigma": f"{k.sigma:.5f}",
                "lambda": f"{k.lam:.4f}",
                "p": f"{k.p:.3f}",
                "eta1": f"{k.eta1:.2f}",
                "eta2": f"{k.eta2:.2f}",
                "LL": f"{k.log_likelihood:.1f}",
                "AIC": f"{k.aic:.1f}",
                "BIC": f"{k.bic:.1f}",
            }
        )
    return kou_results, pd.DataFrame(rows)


def compare_models_and_plot(returns_dict: dict[str, np.ndarray], dt: float = 1.0):
    comparisons = compare_all_tokens(returns_dict, dt=dt)

    rows = []
    for coin in COINS:
        c = comparisons[coin]
        rows.append(
            {
                "Coin": coin,
                "Merton BIC": f"{c.merton.bic:.1f}",
                "Kou BIC": f"{c.kou.bic:.1f}",
                "DBIC": f"{c.bic_delta:.1f}",
                "Preferred": c.preferred.upper(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    deltas = [comparisons[c].bic_delta for c in COINS]
    colors = ["steelblue" if d > 0 else "firebrick" for d in deltas]
    ax.bar(COINS, deltas, color=colors)
    ax.axhline(0, color="gray", lw=0.8)
    ax.grid(False)
    ax.set_ylabel("BIC(Merton) - BIC(Kou)")
    ax.set_title("Model Selection: DBIC (positive favors Kou)")
    fig.tight_layout()
    plt.show()

    return comparisons


def plot_qq(returns_dict: dict, merton_results: dict, kou_results: dict):
    fig, axes = plt.subplots(2, len(COINS), figsize=(4 * len(COINS), 8))
    for j, coin in enumerate(COINS):
        lr = np.sort(returns_dict[coin])
        n = len(lr)
        empirical_quantiles = (np.arange(1, n + 1) - 0.5) / n

        x_fine = np.linspace(lr.min() - 0.01, lr.max() + 0.01, 5000)

        m_density = np.exp(merton_log_density(x_fine, merton_results[coin], dt=1.0))
        m_cdf = np.cumsum(m_density) * (x_fine[1] - x_fine[0])
        m_cdf = m_cdf / m_cdf[-1]
        m_quantiles = np.interp(empirical_quantiles, m_cdf, x_fine)

        k_density = np.exp(kou_log_density(x_fine, kou_results[coin], dt=1.0))
        k_cdf = np.cumsum(k_density) * (x_fine[1] - x_fine[0])
        k_cdf = k_cdf / k_cdf[-1]
        k_quantiles = np.interp(empirical_quantiles, k_cdf, x_fine)

        axes[0, j].scatter(m_quantiles, lr, s=1, alpha=0.3, color=COIN_COLORS[coin])
        lims = [min(lr.min(), m_quantiles.min()), max(lr.max(), m_quantiles.max())]
        axes[0, j].plot(lims, lims, "--", color="gray", lw=0.8)
        axes[0, j].set_title(f"{coin} — Merton", fontweight="bold", fontsize=10)
        if j == 0:
            axes[0, j].set_ylabel("Empirical quantiles")

        axes[1, j].scatter(k_quantiles, lr, s=1, alpha=0.3, color=COIN_COLORS[coin])
        lims = [min(lr.min(), k_quantiles.min()), max(lr.max(), k_quantiles.max())]
        axes[1, j].plot(lims, lims, "--", color="gray", lw=0.8)
        axes[1, j].set_title(f"{coin} — Kou", fontweight="bold", fontsize=10)
        if j == 0:
            axes[1, j].set_ylabel("Empirical quantiles")
        axes[1, j].set_xlabel("Model quantiles")

    fig.suptitle("QQ Plots: Model vs Empirical Quantiles", fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()


def plot_tail_probabilities(returns_dict: dict, merton_results: dict, kou_results: dict):
    thresholds = np.linspace(0.005, 0.06, 30)

    fig, axes = plt.subplots(1, len(COINS), figsize=(4 * len(COINS), 4))
    for ax, coin in zip(axes, COINS):
        lr = returns_dict[coin]
        emp_tail = [np.mean(np.abs(lr) > t) for t in thresholds]

        x_fine = np.linspace(-0.15, 0.15, 8000)
        dx = x_fine[1] - x_fine[0]

        m_pdf = np.exp(merton_log_density(x_fine, merton_results[coin], dt=1.0))
        m_cdf_vals = np.cumsum(m_pdf) * dx
        m_cdf_vals = m_cdf_vals / m_cdf_vals[-1]
        m_tail = [
            float(np.interp(-t, x_fine, m_cdf_vals) + 1 - np.interp(t, x_fine, m_cdf_vals))
            for t in thresholds
        ]

        k_pdf = np.exp(kou_log_density(x_fine, kou_results[coin], dt=1.0))
        k_cdf_vals = np.cumsum(k_pdf) * dx
        k_cdf_vals = k_cdf_vals / k_cdf_vals[-1]
        k_tail = [
            float(np.interp(-t, x_fine, k_cdf_vals) + 1 - np.interp(t, x_fine, k_cdf_vals))
            for t in thresholds
        ]

        ax.semilogy(thresholds * 100, emp_tail, "o", ms=3, color=COIN_COLORS[coin], label="Empirical")
        ax.semilogy(thresholds * 100, m_tail, "-", lw=1.5, color="gray", label="Merton")
        ax.semilogy(thresholds * 100, k_tail, "--", lw=1.5, color="gray", label="Kou")
        ax.set_title(coin, fontweight="bold")
        ax.set_xlabel("Threshold (%)")
        if coin == COINS[0]:
            ax.set_ylabel("P(|r| > threshold)")
        ax.legend(fontsize=7)

    fig.suptitle("Tail Probability: Empirical vs Model", fontsize=13)
    fig.tight_layout()
    plt.show()


def compute_jump_weighted_risk(
    merton_results: dict,
    oi: pl.DataFrame,
    measured_depth: dict[str, float],
    dt: float = 1.0,
) -> tuple[dict, pd.DataFrame]:
    oi_for_risk = oi.filter(pl.col("venue").is_in(["hyperliquid", "lighter"]))
    jwr = jump_weighted_risk_all_coins(
        merton_results,
        oi_for_risk,
        dt=dt,
        depth_per_coin=measured_depth,
        tiered=True,
    )
    rows = []
    for coin in COINS:
        if coin not in jwr:
            continue
        r = jwr[coin]
        rows.append(
            {
                "Coin": coin,
                "Baseline Loss": f"{r['baseline_loss']:.4e}",
                "Amplified Loss": f"{r['amplified_loss']:.4e}",
                "Cascade Mult.": f"{r['cascade_multiplier']:.1f}x",
                "P(<=5%)": f"{r['tail_probability_5pct']:.2e}",
                "A(5%)": f"{r['amplification_at_5pct']:.2f}",
            }
        )
    return jwr, pd.DataFrame(rows)


def print_depth_and_oi_summary(
    depth_df: pl.DataFrame,
    oi_matched: pl.DataFrame,
    measured_depth: dict[str, float],
):
    print("=== Measured 1% Bid-Side Depth (USD) ===")
    for row in depth_df.sort(["coin", "venue"]).to_pandas().itertuples():
        print(f"  {row.venue:12s} {row.coin:5s}  ${row.bid_depth_usd / 1e6:.2f}M")

    print("\n=== OI Scoped to Depth-Measured Venues ===")
    oi_latest = oi_matched.sort("timestamp").group_by(["venue", "coin"]).last()
    for coin in COINS:
        coin_oi = oi_latest.filter(pl.col("coin") == coin)["oi_usd"].sum()
        depth = measured_depth.get(coin, 0)
        ratio = coin_oi / depth if depth > 0 else float("inf")
        print(
            f"  {coin:5s}  OI=${coin_oi / 1e6:>8.1f}M   depth=${depth / 1e6:.1f}M   OI/depth={ratio:>5.0f}x"
        )

    print("\n=== Sample Leverage Tiers ===")
    for venue in ["hyperliquid", "lighter"]:
        tiers = _venue_tiers(venue, "BTC")
        lev_str = ", ".join(f"{w:.0%} at {lev:.0f}x" for lev, w in tiers)
        max_lev = MAX_LEVERAGE.get((venue, "BTC"), 10)
        print(f"  {venue} BTC (max {max_lev}x): {lev_str}")


def print_per_coin_risk_signals(
    oi_matched: pl.DataFrame, measured_depth: dict[str, float]
) -> dict:
    signals = per_coin_risk_signals(oi_matched, depth_per_coin=measured_depth)

    print("=== Per-Coin Cascade Risk (tiered leverage, venue-matched) ===\n")
    print(f"{'Coin':5s}  {'OI/Depth':>9s}  {'Risk Score':>10s}  {'Crit. Shock':>11s}  {'A(5%)':>6s}")
    print("-" * 50)
    for coin in COINS:
        s = signals.get(coin, {})
        crit = s.get("critical_shock")
        crit_str = f"{crit:.1%}" if crit else "none"
        ratio = s.get("oi_depth_ratio", 0)
        print(
            f"{coin:5s}  {ratio:>8.1f}x  {s.get('risk_score', 0):>10.3f}  "
            f"{crit_str:>11s}  {s.get('amplification_at_5pct', 0):>5.1f}x"
        )
    return signals


def compute_and_plot_time_varying_risk(
    oi_matched: pl.DataFrame,
    measured_depth: dict[str, float],
):
    total_depth = sum(measured_depth.values()) / max(len(measured_depth), 1)
    oi_timestamps = oi_matched["timestamp"].unique().sort().to_list()
    risk_scores = []
    for ts in oi_timestamps:
        snap = oi_matched.filter(pl.col("timestamp") == ts)
        coin_scores = []
        for coin in COINS:
            coin_snap = snap.filter(pl.col("coin") == coin)
            if coin_snap.is_empty():
                continue
            pos_t = build_positions_tiered(coin_snap)
            if not pos_t:
                continue
            depth = measured_depth.get(coin, total_depth)
            sig = cascade_risk_signal(pos_t, current_price=1.0, orderbook_depth_usd=depth)
            coin_scores.append(sig["risk_score"])
        if coin_scores:
            risk_scores.append(
                {
                    "timestamp": ts,
                    "risk_score": np.mean(coin_scores),
                    "max_risk": max(coin_scores),
                    "min_risk": min(coin_scores),
                }
            )

    risk_df = pd.DataFrame(risk_scores)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(
        risk_df["timestamp"],
        risk_df["min_risk"],
        risk_df["max_risk"],
        alpha=0.2,
        color="steelblue",
        label="min-max range",
    )
    ax.plot(risk_df["timestamp"], risk_df["risk_score"], color="steelblue", lw=1.5, label="mean")
    ax.axhline(0.5, color="orange", ls=":", lw=1, label="Signal threshold (0.5)")
    ax.set_ylabel("Risk Score")
    ax.set_title("Time-Varying Cascade Risk Score (OI/depth-based, per-coin average)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    print(
        f"\nMean risk score: min={risk_df['risk_score'].min():.3f}  "
        f"median={risk_df['risk_score'].median():.3f}  "
        f"max={risk_df['risk_score'].max():.3f}"
    )
