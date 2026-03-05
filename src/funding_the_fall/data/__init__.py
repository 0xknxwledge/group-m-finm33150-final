"""Data pipeline — exchange fetchers, lending data, normalization, and storage."""

from funding_the_fall.data.fetchers import (
    fetch_funding_rates,
    fetch_predicted_funding,
    fetch_candles,
    fetch_open_interest,
    fetch_liquidations,
)
from funding_the_fall.data.lending import (
    fetch_hyperlend_positions,
    fetch_morpho_positions,
    fetch_tvl_history,
    fetch_all_lending_positions,
)
from funding_the_fall.data.storage import (
    load_funding,
    load_candles,
    load_oi,
    load_liquidations,
    load_lending,
)

__all__ = [
    "fetch_funding_rates",
    "fetch_predicted_funding",
    "fetch_candles",
    "fetch_open_interest",
    "fetch_liquidations",
    "fetch_hyperlend_positions",
    "fetch_morpho_positions",
    "fetch_tvl_history",
    "fetch_all_lending_positions",
    "load_funding",
    "load_candles",
    "load_oi",
    "load_liquidations",
    "load_lending",
]
