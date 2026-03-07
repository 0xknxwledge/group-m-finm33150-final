// Funding the Fall -- Pitchbook
// FINM 33150 Quantitative Trading Strategies, Winter 2026

#import "@preview/polylux:0.4.0": *

// ── Color palette ──────────────────────────────────────────────
#let navy = rgb("#1a365d")
#let accent = rgb("#3182ce")
#let light = rgb("#f7fafc")
#let midgray = rgb("#e2e8f0")
#let darktext = rgb("#2d3748")

// ── Page setup ─────────────────────────────────────────────────
#set page(paper: "presentation-16-9", margin: (x: 1.2cm, y: 1cm))
#set text(font: "Helvetica Neue", size: 14pt, fill: darktext, weight: "regular")
#set par(leading: 0.7em)

// ── Reusable components ────────────────────────────────────────

#let slide-header(title) = {
  rect(
    width: 100%,
    height: 1.2cm,
    fill: navy,
    inset: (x: 0.8cm, y: 0.25cm),
    text(fill: white, size: 20pt, weight: "bold", title),
  )
}

#let slide-footer() = {
  place(
    bottom + left,
    dy: 0.3cm,
    rect(
      width: 100%,
      height: 0.6cm,
      fill: light,
      inset: (x: 0.8cm, y: 0.1cm),
      {
        text(size: 9pt, fill: navy, weight: "medium", "Funding the Fall")
        h(1fr)
        text(size: 9pt, fill: navy, context counter(page).display())
      },
    ),
  )
}

#let body-slide(title, body) = {
  slide[
    #slide-header(title)
    #v(0.4cm)
    #pad(x: 0.6cm, body)
    #slide-footer()
  ]
}

#let kv(key, value) = {
  text(weight: "bold", key + ": ")
  value
}

// ── Slide 1: Title ─────────────────────────────────────────────
#slide[
  #v(1fr)
  #align(center)[
    #rect(
      width: 80%,
      inset: 1.2cm,
      fill: navy,
      radius: 8pt,
    )[
      #text(fill: white, size: 32pt, weight: "bold", "Funding the Fall")
      #v(0.3cm)
      #text(
        fill: rgb("#90cdf4"),
        size: 16pt,
        "Crypto Perpetual Futures Carry + Tail-Risk Strategy",
      )
    ]
    #v(0.8cm)
    #text(size: 13pt)[
      John Beecher (12302422) #h(0.6cm)
      Antonio Braz (12496713) \
      Jean-Luc Choiseul (12506535) #h(0.6cm)
      Jean Mauratille (12328252)
    ]
    #v(0.4cm)
    #text(
      size: 12pt,
      fill: navy,
      weight: "medium",
      "FINM 33150 | Quantitative Trading Strategies, Winter 2026",
    )
  ]
  #v(1fr)
]

// ── Slide 2: Executive Summary ─────────────────────────────────
#body-slide("Summary")[
  #v(0.1cm)
  #text(
    size: 15pt,
    weight: "bold",
    fill: navy,
    "A delta-neutral strategy that harvests cross-venue perpetual futures funding rate spreads and shorts coins vulnerable to liquidation cascades. Funding was positive, but square-root price impact, taker fees, and liquidation losses consumed the portfolio.",
  )
  #v(0.3cm)

  - *Funding Rate Carry (85% base allocation):* delta-neutral cross-venue spread harvesting across 7 exchanges
  - *Liquidation Cascade Overlay (up to 15%):* Merton jump-diffusion + OI/depth risk scoring for tail-risk shorts
  - *Universe:* BTC, ETH, SOL, HYPE, DOGE across Hyperliquid, Lighter, OKX, Kraken, Binance, Bybit, dYdX

  #v(0.3cm)
  #rect(width: 100%, fill: light, inset: 0.5cm, radius: 4pt)[
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 1cm,
      align(center)[
        #text(size: 11pt, fill: navy, weight: "medium", "Sharpe Ratio")
        #v(0.1cm)
        #text(size: 24pt, weight: "bold", fill: accent, sym.minus + "0.82")
      ],
      align(center)[
        #text(size: 11pt, fill: navy, weight: "medium", "Max Drawdown")
        #v(0.1cm)
        #text(size: 24pt, weight: "bold", fill: accent, sym.minus + "97.0%")
      ],
      align(center)[
        #text(size: 11pt, fill: navy, weight: "medium", "Total Return")
        #v(0.1cm)
        #text(size: 24pt, weight: "bold", fill: accent, sym.minus + "96.4%")
      ],
    )
  ]
  #v(0.2cm)
  #align(center, text(
    size: 12pt,
    fill: navy,
    [+\$39K funding collected #h(0.3cm) | #h(0.3cm) \$1.0M in costs (impact + fees + liq.) #h(0.3cm) | #h(0.3cm) *The carry signal works; execution must improve*],
  ))
]

// ── Slide 3: The Opportunity ───────────────────────────────────
#body-slide("The Opportunity")[
  #v(0.1cm)
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #text(weight: "bold", fill: navy, size: 14pt, "Funding Rate Dispersion")
      #v(0.15cm)
      - Perpetual futures use *funding rates* to anchor prices to spot
      - Rates vary 5 to 60%+ annualized across venues
      - *Structural cause:* venue fragmentation, different user bases, margin rules
      - Top mean spreads: HYPE 61%, SOL 21%, BTC 18%
    ],
    [
      #text(weight: "bold", fill: navy, size: 14pt, "Cascade Fragility")
      #v(0.15cm)
      - High leverage creates *liquidation feedback loops*
      - Price drop #sym.arrow.r forced selling #sym.arrow.r further drop
      - OI/depth ratios reveal structural vulnerability
      - HYPE 1048x, BTC 329x, ETH 81x, SOL 69x
    ],
  )
  #v(0.2cm)
  #align(center, image("figures/spread_heatmaps.png", width: 95%))
]

// ── Slide 4: How It Works ──────────────────────────────────────
#body-slide("How It Works")[
  #v(0.1cm)
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #text(weight: "bold", fill: navy, size: 14pt, "Carry Leg (85% baseline)")
      #v(0.15cm)
      + Scan 7 venues for funding rate spreads every 8h
      + Enter delta-neutral pairs above optimized threshold
      + Collect net funding payments (long low-rate, short high-rate)
      + Exit on mean-reversion or max holding period
      + 4x leverage per leg
    ],
    [
      #text(weight: "bold", fill: navy, size: 14pt, "Cascade Leg (up to 15%)")
      #v(0.15cm)
      + Calibrate Merton jump-diffusion on hourly returns
      + Compute OI/depth risk score per coin
      + Short the most fragile coins when score is elevated
      + 1.5x leverage, 24h rebalance
    ],
  )
  #v(0.3cm)
  #rect(width: 100%, fill: light, inset: 0.4cm, radius: 4pt)[
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 0.6cm,
      [#kv("Entry", "spread " + sym.gt.eq + " 35% ann.")],
      [#kv("Exit", "spread " + sym.lt.eq + " 1% ann.")],
      [#kv("Risk score", [#text(size: 12pt)[$1 - e^(-"OI" slash ("depth" dot 200))$]])],
    )
  ]
]

// ── Slide 5: Risk Controls ─────────────────────────────────────
#body-slide("Risk Controls")[
  #v(0.1cm)
  #text(
    size: 13pt,
    "Allocation shifts dynamically with aggregate cascade risk: carry scales down up to 30%, cascade scales up, cash buffer grows.",
  )
  #v(0.3cm)

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.6cm,
    rect(fill: light, inset: 0.5cm, radius: 4pt, height: 4cm)[
      #align(center, text(size: 28pt, fill: accent, weight: "bold", "5x"))
      #v(0.1cm)
      #align(center, text(weight: "bold", fill: navy, "Gross Leverage Cap"))
      #v(0.15cm)
      Total notional across all positions may not exceed 5x NAV. Hard limit enforced before any new trade.
    ],
    rect(fill: light, inset: 0.5cm, radius: 4pt, height: 4cm)[
      #align(center, text(size: 28pt, fill: accent, weight: "bold", "40%"))
      #v(0.1cm)
      #align(center, text(weight: "bold", fill: navy, "Single-Exchange Cap"))
      #v(0.15cm)
      No more than 40% of NAV exposed to any single exchange. Mitigates counterparty and operational risk.
    ],
    rect(fill: light, inset: 0.5cm, radius: 4pt, height: 4cm)[
      #align(center, text(size: 28pt, fill: accent, weight: "bold", "10%"))
      #v(0.1cm)
      #align(center, text(weight: "bold", fill: navy, "Net Delta Limit"))
      #v(0.15cm)
      Net directional exposure capped at 10% of NAV. Ensures the portfolio stays market-neutral.
    ],
  )

  #v(0.3cm)
  #align(center)[
    #table(
      columns: (2.5fr, 1fr, 1fr, 1fr),
      inset: 0.4cm,
      stroke: 0.5pt + midgray,
      fill: (_, row) => if row == 0 { navy } else if calc.odd(row) {
        light
      } else { white },
      table.header(
        text(fill: white, weight: "bold", "Risk Regime"),
        text(fill: white, weight: "bold", "Carry"),
        text(fill: white, weight: "bold", "Cascade"),
        text(fill: white, weight: "bold", "Cash"),
      ),
      [Low (score < 0.3)], [85%], [0%], [15%],
      [Medium (0.3 to 0.7)], [~75%], [~10%], [~15%],
      [High (score > 0.7)], [~60%], [~15%], [~25%],
    )
  ]
]

// ── Slide 6: Backtest Results ──────────────────────────────────
#body-slide("Backtest Results")[
  #v(0.1cm)
  #grid(
    columns: (1fr, 1.1fr),
    gutter: 0.6cm,
    [
      #align(center)[
        #table(
          columns: (1.5fr, 1fr, 1fr),
          inset: 0.35cm,
          stroke: 0.5pt + midgray,
          fill: (_, row) => if row == 0 { navy } else if calc.odd(row) {
            light
          } else { white },
          table.header(
            text(fill: white, weight: "bold", "Metric"),
            text(fill: white, weight: "bold", "Impact Only*"),
            text(fill: white, weight: "bold", "Full Costs"),
          ),
          [Total Return], [#sym.minus 95.4%], [#sym.minus 96.4%],
          [Ann. Volatility], [123.3%], [124.2%],
          [Sharpe Ratio], [#sym.minus 0.81], [#sym.minus 0.82],
          [Max Drawdown], [#sym.minus 96.3%], [#sym.minus 97.0%],
        )
      ]
      #v(0.2cm)
      #text(size: 10pt, fill: darktext)[
        12 months (Mar 2025 to Mar 2026) · 8,737 epochs · \$1M initial · 627 trades \
        \*Impact Only = sqrt price impact deducted from cash (no taker fees)
      ]
    ],
    [
      #image("figures/nav_and_leverage.png", width: 100%)
    ],
  )
]

// ── Slide 7: Where the Money Went ──────────────────────────────
#body-slide("Where the Money Went")[
  #v(0.1cm)
  #grid(
    columns: (1fr, 1.2fr),
    gutter: 0.6cm,
    [
      #align(center)[
        #table(
          columns: (2fr, 1fr),
          inset: 0.45cm,
          stroke: 0.5pt + midgray,
          fill: (_, row) => if row == 0 { navy } else if calc.odd(row) {
            light
          } else { white },
          table.header(
            text(fill: white, weight: "bold", "Component"),
            text(fill: white, weight: "bold", "PnL"),
          ),
          [Funding Collected], [+\$39K],
          [Price PnL], [+\$10K],
          [Price Impact], [#sym.minus \$680K],
          [Taker Fees], [#sym.minus \$96K],
          [Liquidation Losses], [#sym.minus \$236K],
          text(weight: "bold", "Net PnL"),
          text(weight: "bold", [#sym.minus \$964K]),
        )
      ]
      #v(0.3cm)
      #text(size: 11pt)[
        - 621 carry trades, \$200M notional turnover
        - Price impact was 67% of total costs (\$680K)
        - Impact #sym.arrow.r smaller positions #sym.arrow.r liquidations (\$236K)
        - Costs were 26x funding income
      ]
    ],
    [
      #image("figures/pnl_decomposition.png", width: 100%)
    ],
  )
  #v(0.2cm)
  #align(center, image("figures/impact_model_comparison.png", width: 85%))
  #v(0.1cm)
  #align(center, text(size: 10pt, fill: darktext)[
    Sqrt (base case) vs. linear (calibrated per-coin to match sqrt at median trade size) vs. no impact. Impact deducted from cash in real time.
  ])
]

// ── Slide 8: What We Learned ───────────────────────────────────
#body-slide("What We Learned")[
  #v(0.1cm)
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.8cm,
    [
      #text(weight: "bold", fill: navy, size: 14pt, "What Worked")
      #v(0.2cm)
      - Funding income was positive (+\$39K): the carry signal correctly identified real cross-venue spreads
      - Pre-cost Sharpes of 0.4 to 2.5 across coins
      - Cascade scoring flagged the right coins (HYPE, BTC)
      - Risk limits held throughout: no leverage breach, no concentration breach
    ],
    [
      #text(weight: "bold", fill: navy, size: 14pt, "What Failed")
      #v(0.2cm)
      - Price impact dominated: \$680K (67% of costs) from moving illiquid books
      - Impact #sym.arrow.r NAV bleed #sym.arrow.r margin calls: \$236K in liquidation losses
      - 4x leverage on thin venues generated \$200M notional turnover
      - Cascade leg underutilized: only 3 trades in 12 months
    ],
  )

  #v(0.3cm)
  #align(center)[
    #rect(fill: navy, inset: 0.5cm, radius: 6pt, width: 95%)[
      #text(
        fill: white,
        size: 13pt,
        weight: "bold",
        "The thesis is valid. A production version needs maker-only routing, longer holding periods, and lower leverage.",
      )
    ]
  ]
]

// ── Slide 9: Key Risks ─────────────────────────────────────────
#body-slide("Key Risks")[
  #v(0.2cm)
  #table(
    columns: (1.5fr, 2.5fr),
    inset: 0.5cm,
    stroke: 0.5pt + midgray,
    fill: (_, row) => if row == 0 { navy } else if calc.odd(row) {
      light
    } else { white },
    table.header(
      text(fill: white, weight: "bold", "Risk"),
      text(fill: white, weight: "bold", "Mitigant"),
    ),
    [*Funding rate compression*],
    [Diversification across 21+ venue pairs reduces single-spread dependence],

    [*Exchange counterparty risk*],
    [40% single-exchange cap; 7-venue diversification; isolated margin per position],

    [*Cascade model mis-specification*],
    [Conservative sizing (max 15% NAV); model validated on historical cascade events],

    [*Liquidity stress*],
    [Square-root impact model; positions sized to available depth; cash buffer in high-risk regimes],
  )
]

// ── Slide 10: Conclusion ───────────────────────────────────────
#slide[
  #slide-header("Conclusion")
  #v(0.3cm)
  #pad(x: 0.6cm)[
    #grid(
      columns: (1fr, 1fr),
      gutter: 1cm,
      [
        #text(weight: "bold", fill: navy, size: 15pt, "The Signal")
        #v(0.3cm)
        - Funding spreads are structural, not transient: top pairs averaged 18 to 61% annualized
        - Pre-cost Sharpes of 0.4 to 2.5 across five coins
        - OI/depth ratios (16x to 1048x) correctly identify cascade fragility
        - Delta-neutral carry collected +\$39K over 12 months
      ],
      [
        #text(weight: "bold", fill: navy, size: 15pt, "The Fix")
        #v(0.3cm)
        - *Thicker venues:* trade only where depth supports the position size, reducing impact
        - *Longer holding periods:* let funding accumulate before exiting
        - *Lower leverage:* 2x halves notional turnover per dollar of margin
        - *Maker-only routing:* 0bps on Hyperliquid and Lighter eliminates taker fees
      ],
    )

    #v(0.4cm)
    #align(center)[
      #rect(fill: navy, inset: 0.6cm, radius: 6pt, width: 95%)[
        #text(
          fill: white,
          size: 14pt,
          weight: "bold",
          "Funding rate carry in crypto perps is a real inefficiency.",
        )
        #v(0.1cm)
        #text(
          fill: rgb("#90cdf4"),
          size: 12pt,
          "This backtest demonstrates signal validity and quantifies the execution threshold for profitability.",
        )
      ]
    ]
  ]
  #slide-footer()
]
