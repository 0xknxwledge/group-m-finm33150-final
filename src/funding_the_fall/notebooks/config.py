import matplotlib.pyplot as plt

COINS = ["BTC", "ETH", "SOL", "HYPE", "DOGE"]
COIN_COLORS = dict(zip(COINS, ["#F7931A", "#627EEA", "#9945FF", "#00D1A0", "#C3A634"]))

EIGHT_H_RATE_VENUES = {"binance", "bybit", "okx", "hyperliquid", "lighter"}
HOURLY_VENUES = {"hyperliquid", "lighter", "kraken", "dydx"}
EIGHT_H_SETTLE_VENUES = {"binance", "bybit", "okx"}

VENUE_ORDER = ["hyperliquid", "lighter", "kraken", "dydx", "binance", "bybit", "okx"]

NAV = 1_000_000
CARRY_LEV = 4.0
CASCADE_LEV = 1.5
RISK_FREE_RATE = 0.05


def apply_mpl_defaults():
    plt.rcParams.update(
        {
            "figure.figsize": (12, 5),
            "figure.dpi": 120,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
        }
    )
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("config", "InlineBackend.figure_formats = ['svg']")
            ip.run_line_magic("config", "InlineBackend.print_figure_kwargs = {'facecolor': 'none'}")
    except Exception:
        pass
