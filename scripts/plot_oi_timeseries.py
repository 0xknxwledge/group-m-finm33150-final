"""Plot OI time series per coin, colored by venue.

Excludes Lighter (OI reported in raw contracts, not USD) and venues with
only a single snapshot. Also shows a cross-venue snapshot comparison.
"""

import sys
sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import polars as pl
from funding_the_fall.data.storage import load_oi

COINS = ["BTC", "ETH", "SOL", "HYPE", "DOGE"]
COIN_COLORS = dict(zip(COINS, ["#F7931A", "#627EEA", "#9945FF", "#00D1A0", "#C3A634"]))

oi = load_oi()

# --- Data inventory ---
print("=== OI Data Inventory ===")
summary = (
    oi.group_by(["venue", "coin"])
    .agg([
        pl.col("timestamp").count().alias("n"),
        pl.col("timestamp").min().alias("first"),
        pl.col("timestamp").max().alias("last"),
        pl.col("oi_usd").median().alias("median_oi"),
    ])
    .sort(["coin", "venue"])
)
for row in summary.iter_rows(named=True):
    flag = ""
    if row["venue"] == "lighter":
        flag = "  ⚠ raw contracts, not USD"
    elif row["n"] == 1:
        flag = "  (single snapshot)"
    print(f"  {row['coin']:5s} {row['venue']:15s}  n={row['n']:5d}  "
          f"median=${row['median_oi']:>15,.0f}{flag}")

# --- Fig 1: Time series (only venues with >1 row, excluding Lighter) ---
ts_venues = (
    oi.filter(pl.col("venue") != "lighter")
    .group_by("venue")
    .agg(pl.col("timestamp").count().alias("n"))
    .filter(pl.col("n") > 1)["venue"]
    .to_list()
)
venue_colors = dict(zip(sorted(ts_venues), plt.cm.tab10.colors[:len(ts_venues)]))

if ts_venues:
    fig, axes = plt.subplots(len(COINS), 1, figsize=(14, 3 * len(COINS)), sharex=True)
    for ax, coin in zip(axes, COINS):
        coin_df = oi.filter(
            (pl.col("coin") == coin) & pl.col("venue").is_in(ts_venues)
        ).sort("timestamp")
        for venue in sorted(ts_venues):
            vdf = coin_df.filter(pl.col("venue") == venue)
            if vdf.shape[0] == 0:
                continue
            ax.plot(vdf["timestamp"].to_list(), [v / 1e6 for v in vdf["oi_usd"].to_list()],
                    label=venue, lw=1, alpha=0.8, color=venue_colors[venue])
        ax.set_ylabel(f"{coin}\nOI ($M)")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Time")
    axes[0].set_title("Open Interest Time Series (venues with historical data)")
    fig.tight_layout()
    plt.savefig("oi_timeseries.png", dpi=150, bbox_inches="tight")
    print("\nSaved oi_timeseries.png")
    plt.show()

# --- Fig 2: Cross-venue snapshot bar chart (latest per venue, excl Lighter) ---
latest = (
    oi.filter(pl.col("venue") != "lighter")
    .sort("timestamp")
    .group_by(["venue", "coin"])
    .last()
)

fig, axes = plt.subplots(1, len(COINS), figsize=(3.5 * len(COINS), 4))
for ax, coin in zip(axes, COINS):
    sub = latest.filter(pl.col("coin") == coin).sort("venue")
    if sub.shape[0] == 0:
        ax.set_title(coin)
        continue
    venues_list = sub["venue"].to_list()
    values = [v / 1e6 for v in sub["oi_usd"].to_list()]
    ax.barh(venues_list, values, color=COIN_COLORS[coin], alpha=0.8)
    ax.set_xlabel("OI ($M)")
    ax.set_title(coin, fontweight="bold")

fig.suptitle("Latest OI Snapshot by Venue (excl. Lighter)", fontsize=13)
fig.tight_layout()
plt.savefig("oi_snapshot.png", dpi=150, bbox_inches="tight")
print("Saved oi_snapshot.png")
plt.show()
