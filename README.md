# Group M FINM 33150 Final Trading Project

Cross-venue carry and cascade alpha in crypto perpetuals.

**FINM 33150 — Quantitative Trading Strategies — Winter 2026**

John Beecher (12302422), Antonio Magalhaes Torreao e Braz (12496713), Jean-Luc Choiseul (12506535), Jean Mauratille (12328252)

---

## Strategy

Two-component strategy targeting structural inefficiencies in crypto perpetual futures:

1. **Funding Rate Carry (70%)** — Delta-neutral cross-venue carry harvesting persistent funding rate spreads across 7 venues (Hyperliquid, Lighter, OKX, Kraken, Binance, Bybit, dYdX).

2. **Liquidation Cascade Positioning (30%)** — Merton jump-diffusion model monitoring cross-layer liquidation risk across Hyperliquid perps and HyperLend for convex tail-risk payoffs. Historical lending positions reconstructed via on-chain event replay.

## Data Sources

| Source | Data | Auth | Notes |
|--------|------|------|-------|
| 0xArchive | Funding, candles, OI, liquidations, prices (Hyperliquid + Lighter) | API key | Historical from Apr 2023 |
| OKX | Funding, candles, OI, liquidations | None | Free, works from US |
| Kraken Futures | Funding (hourly→8h), candles, OI | None | Free, works from US |
| Binance | Funding, candles, OI, liquidations | None | VPN required |
| Bybit | Funding, candles, OI | None | VPN required |
| dYdX | Funding, candles, OI | None | No auth needed |
| Hyperliquid | Live meta, orderbook, predicted fundings | None | Direct API fallback |
| HyperLend | Lending positions via on-chain event replay | None | HyperEVM RPC |
| DeFi Llama | TVL history | None | Free |

**Tokens:** BTC, ETH, SOL, HYPE, DOGE

## Setup

```bash
uv sync
```

## Development

```bash
make sync      # install dependencies
make test      # run tests
make lab       # launch jupyter lab
make lint      # ruff check
make clean     # remove caches and build artifacts
```

## Project Structure

```
src/funding_the_fall/   # core library
scripts/                # standalone scripts
notebooks/              # jupyter notebooks
tests/                  # pytest suite
data/                   # local data (gitignored)
```

## License

[The Unlicense](LICENSE)
