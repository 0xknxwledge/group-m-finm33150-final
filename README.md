# Group M FINM 33150 Final Trading Project

Cross-venue carry and cascade alpha in crypto perpetuals.

**FINM 33150 — Quantitative Trading Strategies — Winter 2026**

John Beecher, Antonio Magalhaes Torrea o Braz, Jean-Luc Choiseul, Jean Mauratille

---

## Strategy

Two-component strategy targeting structural inefficiencies in crypto perpetual futures:

1. **Funding Rate Carry (70%)** — Delta-neutral cross-venue carry harvesting persistent funding rate spreads across Hyperliquid, Binance, Bybit, and dYdX.

2. **Liquidation Cascade Positioning (30%)** — Merton jump-diffusion model monitoring cross-layer liquidation risk across Hyperliquid perps, HyperLend, and Morpho Blue for convex tail-risk payoffs.

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
