"""Exchange API fetchers for funding rates, candles, open interest, and liquidations.

Owner: Antonio Braz

Each fetcher returns a pandas DataFrame with a unified schema:
- Funding rates:  columns [timestamp, venue, coin, funding_rate]
- Candles:        columns [timestamp, venue, coin, o, h, l, c, v]
- Open interest:  columns [timestamp, venue, coin, oi_usd]
- Liquidations:   columns [timestamp, venue, coin, side, size_usd, price]

All timestamps are UTC datetime64[ns]. Funding rates are per-period (8h)
values, not annualized.

API Reference:
  Hyperliquid   POST api.hyperliquid.xyz/info          (no auth)
  Binance       GET  fapi.binance.com/fapi/v1/*        (no auth, VPN)
  Bybit         GET  api.bybit.com/v5/market/*         (no auth, VPN)
  dYdX          GET  indexer.dydx.trade/v4/*            (no auth)
  Coinalyze     GET  api.coinalyze.net/v1/*             (no auth)
"""

from __future__ import annotations

import pandas as pd

# ── API base URLs ──
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
BYBIT_API = "https://api.bybit.com/v5"
DYDX_API = "https://indexer.dydx.trade/v4"
COINALYZE_API = "https://api.coinalyze.net/v1"

# Token universe — 5+ distinct assets held simultaneously
TOKENS = ["BTC", "ETH", "SOL", "HYPE", "DOGE"]

# Venue list
VENUES = ["hyperliquid", "binance", "bybit", "dydx"]

# Binance/Bybit symbol mappings (they use different ticker formats)
BINANCE_SYMBOLS: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
    # HYPE not listed on Binance
}

BYBIT_SYMBOLS: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
    # HYPE not listed on Bybit
}

DYDX_SYMBOLS: dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    # HYPE not listed on dYdX
}


# ═══════════════════════════════════════════════════════════════════════
# Funding rate fetchers
# ═══════════════════════════════════════════════════════════════════════

def _fetch_funding_hyperliquid(coin: str, days: int = 90) -> pd.DataFrame:
    """Fetch historical funding rates from Hyperliquid.

    POST api.hyperliquid.xyz/info
    {"type": "fundingHistory", "coin": coin, "startTime": ms}
    Returns: [{"coin", "fundingRate", "premium", "time"}]
    """
    raise NotImplementedError


def _fetch_funding_binance(coin: str, days: int = 90) -> pd.DataFrame:
    """Fetch historical funding rates from Binance Futures.

    GET fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000
    Requires VPN (geo-blocked from US).
    """
    raise NotImplementedError


def _fetch_funding_bybit(coin: str, days: int = 90) -> pd.DataFrame:
    """Fetch historical funding rates from Bybit.

    GET api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT
    Requires VPN (geo-blocked from US).
    """
    raise NotImplementedError


def _fetch_funding_dydx(coin: str, days: int = 90) -> pd.DataFrame:
    """Fetch historical funding rates from dYdX v4.

    GET indexer.dydx.trade/v4/historicalFunding/BTC-USD
    """
    raise NotImplementedError


def fetch_predicted_funding() -> pd.DataFrame:
    """Fetch predicted next funding rates from Hyperliquid.

    POST api.hyperliquid.xyz/info {"type": "predictedFundings"}
    Returns cross-venue predictions (Binance, Bybit, Hyperliquid) in one call.
    Useful for live signal generation.

    Output columns: [timestamp, venue, coin, predicted_rate, next_funding_time]
    """
    raise NotImplementedError


def fetch_funding_rates(
    coins: list[str] | None = None,
    venues: list[str] | None = None,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch funding rates for all coins × venues.

    Returns DataFrame with columns: [timestamp, venue, coin, funding_rate]
    Skips (coin, venue) pairs where the coin isn't listed (e.g. HYPE on Binance).
    """
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# Candle (mark price) fetchers
# ═══════════════════════════════════════════════════════════════════════

def _fetch_candles_hyperliquid(
    coin: str, interval: str = "1h", days: int = 90,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Hyperliquid.

    POST api.hyperliquid.xyz/info
    {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval,
     "startTime": ms, "endTime": ms}}
    """
    raise NotImplementedError


def fetch_candles(
    coins: list[str] | None = None,
    interval: str = "1h",
    days: int = 90,
) -> pd.DataFrame:
    """Fetch candle data for all coins (Hyperliquid as primary venue).

    Returns DataFrame with columns: [timestamp, venue, coin, o, h, l, c, v]
    """
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# Open interest fetchers
# ═══════════════════════════════════════════════════════════════════════

def _fetch_oi_hyperliquid() -> pd.DataFrame:
    """Fetch current OI from Hyperliquid metaAndAssetCtxs.

    POST api.hyperliquid.xyz/info {"type": "metaAndAssetCtxs"}
    Returns mark price, funding, OI for every listed asset.
    """
    raise NotImplementedError


def _fetch_oi_binance(coin: str, days: int = 90) -> pd.DataFrame:
    """Fetch OI history from Binance.

    GET fapi.binance.com/fapi/v1/openInterest (current)
    GET fapi.binance.com/futures/data/openInterestHist (historical)
    Requires VPN.
    """
    raise NotImplementedError


def _fetch_oi_bybit(coin: str, days: int = 90) -> pd.DataFrame:
    """Fetch OI history from Bybit.

    GET api.bybit.com/v5/market/open-interest?category=linear&symbol=...
    Requires VPN.
    """
    raise NotImplementedError


def fetch_open_interest(
    coins: list[str] | None = None,
    venues: list[str] | None = None,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch open interest snapshots for all coins × venues.

    Returns DataFrame with columns: [timestamp, venue, coin, oi_usd]
    """
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# Liquidation fetchers
# ═══════════════════════════════════════════════════════════════════════

def _fetch_liquidations_binance(coin: str, days: int = 30) -> pd.DataFrame:
    """Fetch recent forced liquidation orders from Binance.

    GET fapi.binance.com/fapi/v1/forceOrders?symbol=BTCUSDT
    Public endpoint, no auth. Requires VPN.
    """
    raise NotImplementedError


def _fetch_liquidations_coinalyze(coin: str, days: int = 30) -> pd.DataFrame:
    """Fetch aggregated historical liquidation data from Coinalyze.

    GET api.coinalyze.net/v1/liquidation-history
    Free, no auth required.
    """
    raise NotImplementedError


def fetch_liquidations(
    coins: list[str] | None = None,
    days: int = 30,
) -> pd.DataFrame:
    """Fetch liquidation events from all available sources.

    Returns DataFrame with columns:
      [timestamp, venue, coin, side, size_usd, price]

    Sources: Binance forceOrders + Coinalyze aggregated history.
    """
    raise NotImplementedError
