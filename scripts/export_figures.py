"""Export notebook figures as PNGs for the pitchbook."""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parent.parent / "pitchbook" / "figures"

SAVEFIG_KW = dict(dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")


@contextlib.contextmanager
def capture_show(path: Path):
    """Replace plt.show with savefig, then close the figure."""
    original = plt.show

    def _save(*_args, **_kwargs):
        plt.gcf().patch.set_facecolor("white")
        for ax in plt.gcf().get_axes():
            ax.set_facecolor("white")
        plt.savefig(path, **SAVEFIG_KW)
        print(f"  saved {path.name}")

    plt.show = _save
    try:
        yield
    finally:
        plt.show = original
        plt.close("all")


def main():
    import polars as pl

    from funding_the_fall.data.storage import load_candles, load_funding, load_oi, load_orderbook_depth
    from funding_the_fall.models.cascade import depth_by_coin

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Suppress matplotlib rcParam transparency from notebook config
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
        }
    )

    print("Loading data...")
    funding = load_funding()
    candles = load_candles()
    oi = load_oi()
    depth_df = load_orderbook_depth()
    measured_depth = depth_by_coin(depth_df)
    oi_matched = oi.filter(pl.col("venue").is_in(["hyperliquid", "lighter"]))

    # 1. Spread heatmaps
    print("Generating spread heatmaps...")
    from funding_the_fall.notebooks.spreads import compute_spreads, plot_spread_heatmaps

    spreads = compute_spreads(funding)

    with capture_show(FIGURES_DIR / "spread_heatmaps.png"):
        plot_spread_heatmaps(spreads)

    # 2-4. Backtest figures (need carry signals + cascade signals + backtest)
    print("Running carry grid search...")
    from funding_the_fall.notebooks.strategy import run_carry_grid_search

    _best, all_carry_signals, _pair_results = run_carry_grid_search(spreads)

    print("Generating cascade signals...")
    from funding_the_fall.notebooks.backtest import generate_cascade_signals_from_oi, run_backtests

    cascade_signals = generate_cascade_signals_from_oi(oi_matched, measured_depth)

    print("Running backtest...")
    result, result_zero = run_backtests(
        all_carry_signals, cascade_signals, funding, candles, depth_per_coin=measured_depth
    )

    from funding_the_fall.notebooks.backtest import (
        plot_drawdown_and_rolling_sharpe,
        plot_impact_comparison,
        plot_nav_and_leverage,
        plot_pnl_decomposition,
        run_impact_comparison,
    )

    print("Generating backtest figures...")
    with capture_show(FIGURES_DIR / "nav_and_leverage.png"):
        plot_nav_and_leverage(result, result_zero)

    with capture_show(FIGURES_DIR / "pnl_decomposition.png"):
        plot_pnl_decomposition(result)

    with capture_show(FIGURES_DIR / "drawdown_rolling_sharpe.png"):
        plot_drawdown_and_rolling_sharpe(result)

    # Impact model comparison
    print("Running impact model comparison...")
    impact_results = run_impact_comparison(
        all_carry_signals, cascade_signals, funding, candles, depth_per_coin=measured_depth
    )

    with capture_show(FIGURES_DIR / "impact_model_comparison.png"):
        plot_impact_comparison(impact_results)

    print(f"Done. {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
