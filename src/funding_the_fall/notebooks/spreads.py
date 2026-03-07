from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from .config import (
    COINS,
    EIGHT_H_RATE_VENUES,
    HOURLY_VENUES,
    VENUE_ORDER,
)


def compute_spreads(funding: pl.DataFrame) -> pd.DataFrame:
    fund_pd = funding.to_pandas()
    fund_pd["hourly_rate"] = fund_pd.apply(
        lambda r: r["funding_rate"] / 8
        if r["venue"] in EIGHT_H_RATE_VENUES
        else r["funding_rate"],
        axis=1,
    )

    venues = sorted(fund_pd["venue"].unique())

    fund_pd["period_8h"] = fund_pd["timestamp"].dt.floor("8h")
    hourly_agg_8h = (
        fund_pd[fund_pd["venue"].isin(HOURLY_VENUES)]
        .groupby(["period_8h", "venue", "coin"])["hourly_rate"]
        .sum()
        .reset_index()
        .rename(columns={"period_8h": "timestamp"})
    )

    spread_rows = []
    for coin in COINS:
        for i, v1 in enumerate(venues):
            for v2 in venues[i + 1 :]:
                v1_hourly = v1 in HOURLY_VENUES
                v2_hourly = v2 in HOURLY_VENUES
                both_hourly = v1_hourly and v2_hourly

                if both_hourly:
                    d1 = fund_pd[(fund_pd["venue"] == v1) & (fund_pd["coin"] == coin)][
                        ["timestamp", "hourly_rate"]
                    ].rename(columns={"hourly_rate": "r1"})
                    d2 = fund_pd[(fund_pd["venue"] == v2) & (fund_pd["coin"] == coin)][
                        ["timestamp", "hourly_rate"]
                    ].rename(columns={"hourly_rate": "r2"})
                    grid = "1h"
                    ann_factor = 24 * 365
                else:
                    if v1_hourly:
                        d1 = hourly_agg_8h[
                            (hourly_agg_8h["venue"] == v1) & (hourly_agg_8h["coin"] == coin)
                        ][["timestamp", "hourly_rate"]].rename(columns={"hourly_rate": "r1"})
                    else:
                        d1 = fund_pd[(fund_pd["venue"] == v1) & (fund_pd["coin"] == coin)][
                            ["period_8h", "hourly_rate"]
                        ].rename(columns={"period_8h": "timestamp", "hourly_rate": "r1"})
                        d1 = d1.copy()
                        d1["r1"] = d1["r1"] * 8

                    if v2_hourly:
                        d2 = hourly_agg_8h[
                            (hourly_agg_8h["venue"] == v2) & (hourly_agg_8h["coin"] == coin)
                        ][["timestamp", "hourly_rate"]].rename(columns={"hourly_rate": "r2"})
                    else:
                        d2 = fund_pd[(fund_pd["venue"] == v2) & (fund_pd["coin"] == coin)][
                            ["period_8h", "hourly_rate"]
                        ].rename(columns={"period_8h": "timestamp", "hourly_rate": "r2"})
                        d2 = d2.copy()
                        d2["r2"] = d2["r2"] * 8

                    grid = "8h"
                    ann_factor = 3 * 365

                merged = pd.merge(d1, d2, on="timestamp", how="inner")
                if len(merged) < 10:
                    continue
                merged["spread"] = (merged["r2"] - merged["r1"]).abs()
                merged["spread_ann"] = merged["spread"] * ann_factor
                merged["coin"] = coin
                merged["long_venue"] = np.where(merged["r1"] <= merged["r2"], v1, v2)
                merged["short_venue"] = np.where(merged["r1"] <= merged["r2"], v2, v1)
                merged["grid"] = grid
                spread_rows.append(
                    merged[
                        [
                            "timestamp",
                            "coin",
                            "long_venue",
                            "short_venue",
                            "spread",
                            "spread_ann",
                            "grid",
                        ]
                    ]
                )

    spreads = pd.concat(spread_rows, ignore_index=True)
    print(f"Total spread observations: {len(spreads):,}")
    print(f"  1h-grid (both hourly venues): {(spreads['grid'] == '1h').sum():,}")
    print(f"  8h-grid (at least one 8h venue): {(spreads['grid'] == '8h').sum():,}")
    return spreads


def print_top_pairs(spreads: pd.DataFrame):
    summary = (
        spreads.groupby(["coin", "long_venue", "short_venue", "grid"])
        .agg(
            mean_ann=("spread_ann", "mean"),
            med_ann=("spread_ann", "median"),
            p95_ann=("spread_ann", lambda x: x.quantile(0.95)),
            obs=("spread_ann", "count"),
        )
        .sort_values("mean_ann", ascending=False)
    )
    for coin in COINS:
        top = summary.loc[summary.index.get_level_values("coin") == coin].head(3)
        for (c, lv, sv, g), row in top.iterrows():
            print(
                f"  {c:5s} {lv:15s} -> {sv:15s} [{g}]  mean={row['mean_ann']:6.1%}  "
                f"med={row['med_ann']:6.1%}  p95={row['p95_ann']:6.1%}  n={int(row['obs']):,}"
            )


def plot_spread_distributions(spreads: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    spreads.boxplot(column="spread_ann", by="coin", ax=axes[0], showfliers=False, grid=False)
    axes[0].set_title("Annualized Spread by Coin")
    axes[0].set_ylabel("Annualized Spread")
    axes[0].set_xlabel("")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.sca(axes[0])
    plt.xticks(rotation=0)

    for coin in COINS:
        pair = spreads[
            (spreads["coin"] == coin)
            & (spreads["grid"] == "1h")
            & (
                ((spreads["long_venue"] == "hyperliquid") & (spreads["short_venue"] == "lighter"))
                | ((spreads["long_venue"] == "lighter") & (spreads["short_venue"] == "hyperliquid"))
            )
        ]
        if pair.empty:
            continue
        pair = pair.sort_values("timestamp").set_index("timestamp")
        rolling = pair["spread_ann"].rolling("24h").mean()
        axes[1].plot(rolling.index, rolling.values, label=coin, alpha=0.8, linewidth=1)

    axes[1].set_title("Hyperliquid-Lighter Spread, 1h grid (24h rolling)")
    axes[1].set_ylabel("Annualized Spread")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[1].legend(fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    for coin in COINS:
        pair = spreads[
            (spreads["coin"] == coin)
            & (spreads["grid"] == "1h")
            & (
                ((spreads["long_venue"] == "hyperliquid") & (spreads["short_venue"] == "kraken"))
                | ((spreads["long_venue"] == "kraken") & (spreads["short_venue"] == "hyperliquid"))
            )
        ]
        if pair.empty:
            continue
        pair = pair.sort_values("timestamp").set_index("timestamp")
        rolling = pair["spread_ann"].rolling("24h").mean()
        axes[2].plot(rolling.index, rolling.values, label=coin, alpha=0.8, linewidth=1)

    axes[2].set_title("Hyperliquid-Kraken Spread, 1h grid (24h rolling)")
    axes[2].set_ylabel("Annualized Spread")
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[2].legend(fontsize=8)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("")
    plt.tight_layout()
    plt.show()


def plot_spread_heatmaps(spreads: pd.DataFrame):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for ax, coin in zip(axes, COINS):
        matrix = pd.DataFrame(np.nan, index=VENUE_ORDER, columns=VENUE_ORDER)
        coin_spreads = spreads[spreads["coin"] == coin]
        for (lv, sv), val in (
            coin_spreads.groupby(["long_venue", "short_venue"])["spread_ann"].mean().items()
        ):
            if lv in VENUE_ORDER and sv in VENUE_ORDER:
                matrix.loc[lv, sv] = val
                matrix.loc[sv, lv] = val

        ax.grid(False)
        im = ax.imshow(matrix.values, cmap="YlOrRd", vmin=0, vmax=0.50)
        ax.set_xticks(range(len(VENUE_ORDER)))
        ax.set_xticklabels([v[:4] for v in VENUE_ORDER], fontsize=7, rotation=45)
        ax.set_yticks(range(len(VENUE_ORDER)))
        ax.set_yticklabels([v[:4] for v in VENUE_ORDER] if ax == axes[0] else [], fontsize=7)
        ax.set_title(coin, fontsize=10)
        for k in range(len(VENUE_ORDER)):
            ax.add_patch(
                plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=True, color="gray", alpha=0.3)
            )

    fig.colorbar(im, ax=axes, shrink=0.8, label="Mean Ann. Spread")
    fig.suptitle("Cross-Venue Mean Annualized Spread (settlement-grid-aligned)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()
