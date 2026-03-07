from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from .config import COINS, COIN_COLORS


def dataset_summary_table(
    funding: pl.DataFrame,
    candles: pl.DataFrame,
    oi: pl.DataFrame,
) -> pd.DataFrame:
    def _summarize(name: str, df: pl.DataFrame) -> dict:
        venues = df["venue"].unique().sort().to_list()
        coins = df["coin"].unique().sort().to_list()
        return {
            "Dataset": name,
            "Rows": f"{df.shape[0]:,}",
            "Columns": df.shape[1],
            "Start": str(df["timestamp"].min())[:10],
            "End": str(df["timestamp"].max())[:10],
            "Venues": ", ".join(venues),
            "Coins": ", ".join(coins),
        }

    return pd.DataFrame(
        [
            _summarize("Funding", funding),
            _summarize("Candles", candles),
            _summarize("OI", oi),
        ]
    )


def plot_coverage_heatmap(funding: pl.DataFrame):
    coverage = (
        funding.group_by(["venue", "coin"])
        .agg(pl.col("funding_rate").count().alias("n_obs"))
        .to_pandas()
        .pivot(index="venue", columns="coin", values="n_obs")
        .fillna(0)
        .astype(int)
    )
    coverage = coverage.reindex(columns=[c for c in COINS if c in coverage.columns])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(False)
    im = ax.imshow(coverage.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(coverage.columns)))
    ax.set_xticklabels(coverage.columns)
    ax.set_yticks(range(len(coverage.index)))
    ax.set_yticklabels(coverage.index)
    for i in range(len(coverage.index)):
        for j in range(len(coverage.columns)):
            ax.text(
                j, i, f"{coverage.values[i, j]:,}", ha="center", va="center", fontsize=9
            )
    fig.colorbar(im, ax=ax, label="Number of 8h epochs")
    ax.set_title("Funding Rate Coverage: Observations per Venue x Coin")
    fig.tight_layout()
    plt.show()


def plot_funding_distributions(funding: pl.DataFrame):
    fund_pd = funding.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fund_pd.boxplot(column="funding_rate", by="coin", ax=axes[0], grid=False)
    axes[0].set_title("Funding Rate Distribution by Coin")
    axes[0].set_xlabel("Coin")
    axes[0].set_ylabel("Funding Rate (per 8h epoch)")
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")

    fund_pd.boxplot(column="funding_rate", by="venue", ax=axes[1], grid=False)
    axes[1].set_title("Funding Rate Distribution by Venue")
    axes[1].set_xlabel("Venue")
    axes[1].set_ylabel("Funding Rate (per 8h epoch)")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("")
    fig.tight_layout()
    plt.show()


def plot_funding_timeseries(funding: pl.DataFrame):
    venues = funding["venue"].unique().sort().to_list()
    venue_colors = dict(zip(venues, plt.cm.tab10.colors[: len(venues)]))

    fig, axes = plt.subplots(len(COINS), 1, figsize=(14, 3 * len(COINS)), sharex=True)
    for ax, coin in zip(axes, COINS):
        subset = funding.filter(pl.col("coin") == coin).sort("timestamp").to_pandas()
        for venue in venues:
            vdf = subset[subset["venue"] == venue]
            if vdf.empty:
                continue
            ax.plot(
                vdf["timestamp"],
                vdf["funding_rate"],
                label=venue,
                alpha=0.7,
                lw=0.8,
                color=venue_colors[venue],
            )
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel(coin, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7, ncol=len(venues))
    axes[-1].set_xlabel("Time")
    axes[0].set_title("Funding Rates by Coin and Venue")
    fig.tight_layout()
    plt.show()


def plot_prices_and_returns(candles: pl.DataFrame):
    prices = (
        candles.sort("timestamp")
        .group_by(["coin", "timestamp"])
        .agg(pl.col("c").mean().alias("close"))
        .sort("timestamp")
    )

    fig, axes = plt.subplots(2, len(COINS), figsize=(16, 7))
    for j, coin in enumerate(COINS):
        pdf = prices.filter(pl.col("coin") == coin).to_pandas().set_index("timestamp")
        axes[0, j].plot(pdf.index, pdf["close"], color=COIN_COLORS[coin], lw=0.8)
        axes[0, j].set_title(coin, fontweight="bold")
        if j == 0:
            axes[0, j].set_ylabel("Price (USD)")
        axes[0, j].tick_params(axis="x", rotation=45, labelsize=7)

        log_ret = np.log(pdf["close"] / pdf["close"].shift(1)).dropna()
        axes[1, j].grid(False)
        axes[1, j].hist(
            log_ret, bins=60, color=COIN_COLORS[coin], alpha=0.75, density=True
        )
        axes[1, j].set_title("Log Returns")
        if j == 0:
            axes[1, j].set_ylabel("Density")

    fig.suptitle("Hourly Close Prices and Log Return Distributions", fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_open_interest(oi: pl.DataFrame):
    oi_hist = oi.filter(pl.col("venue").is_in(["hyperliquid", "lighter"])).to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for coin in COINS:
        sub = oi_hist[oi_hist["coin"] == coin].groupby("timestamp")["oi_usd"].sum()
        if sub.empty:
            continue
        axes[0].plot(
            sub.index, sub.values / 1e6, label=coin, color=COIN_COLORS[coin], lw=1
        )
    axes[0].set_ylabel("Open Interest (M USD)")
    axes[0].set_title("OI Over Time (Hyperliquid + Lighter)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    latest_oi = (
        oi.sort("timestamp")
        .group_by(["venue", "coin"])
        .last()
        .group_by("coin")
        .agg(pl.col("oi_usd").sum())
        .sort("oi_usd", descending=True)
        .to_pandas()
    )
    bar_colors = [COIN_COLORS.get(c, "gray") for c in latest_oi["coin"]]
    axes[1].bar(latest_oi["coin"], latest_oi["oi_usd"] / 1e6, color=bar_colors)
    axes[1].set_ylabel("Open Interest (M USD)")
    axes[1].set_title("OI by Coin (all venues, latest per venue)")
    axes[1].grid(False)

    fig.tight_layout()
    plt.show()


def print_oi_concentration_and_funding_extremes(
    oi: pl.DataFrame, funding: pl.DataFrame
):
    print("=== OI Concentration (latest per venue) ===")
    latest_oi_full = (
        oi.sort("timestamp")
        .group_by(["venue", "coin"])
        .last()
        .group_by("coin")
        .agg(pl.col("oi_usd").sum())
        .sort("oi_usd", descending=True)
    )
    total_oi = latest_oi_full["oi_usd"].sum()
    for row in latest_oi_full.iter_rows(named=True):
        pct = row["oi_usd"] / total_oi * 100
        print(f"  {row['coin']:5s}  ${row['oi_usd'] / 1e6:8.1f}M  ({pct:5.1f}%)")
    print(f"  {'TOTAL':5s}  ${total_oi / 1e6:8.1f}M")

    print("\n=== Funding Rate Extremes ===")
    extremes = (
        funding.group_by(["venue", "coin"])
        .agg(
            [
                pl.col("funding_rate").mean().alias("mean_rate"),
                pl.col("funding_rate").std().alias("std_rate"),
                pl.col("funding_rate").min().alias("min_rate"),
                pl.col("funding_rate").max().alias("max_rate"),
            ]
        )
        .sort("std_rate", descending=True)
    )
    print(extremes.head(10))


def plot_funding_correlations(funding: pl.DataFrame):
    fig, axes = plt.subplots(1, len(COINS), figsize=(4 * len(COINS), 4))
    for ax, coin in zip(axes, COINS):
        sub = funding.filter(pl.col("coin") == coin).select(
            "timestamp", "venue", "funding_rate"
        )
        if sub.shape[0] == 0:
            ax.set_title(f"{coin}\n(no data)")
            continue
        pivoted = sub.to_pandas().pivot_table(
            index="timestamp", columns="venue", values="funding_rate"
        )
        if pivoted.shape[1] < 2:
            ax.set_title(f"{coin}\n(single venue)")
            continue
        corr = pivoted.corr()
        ax.grid(False)
        ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title(coin, fontweight="bold")
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(
                    j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6
                )

    fig.suptitle("Cross-Venue Funding Rate Correlations", fontsize=13)
    fig.tight_layout()
    plt.show()
