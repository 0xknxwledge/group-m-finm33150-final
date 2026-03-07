# Cascade-Adjusted Liquip Score: Endogenous Liquidation Risk in DeFi Lending

**Author:** John Beecher
**Date:** February 2026
**Extends:** Anthias Liquip Score: Measuring Liquidation Probability (Anthias Team); Asset Risk Analysis - Felix USDhl (Anthias Labs)

---

## Abstract

The Anthias Liquip Score treats each lending position as an isolated unit, computing liquidation probability from exogenous price dynamics alone. In practice, liquidations are endogenous: the forced sale of collateral from one liquidation pushes prices down, potentially triggering further liquidations in a self-reinforcing cascade. We propose the "Cascade-Adjusted Liquip Score," which extends the original framework by incorporating (1) the market impact of liquidation flow against finite on-chain liquidity and (2) the feedback loop between liquidation-induced price impact and subsequent position health. We further derive a protocol-level systemic risk metric — the "Protocol Liquip Score" — that estimates the probability of aggregate liquidation volume exceeding available liquidity within a given time horizon. This metric directly addresses the "recycled liquidity" risk identified in Anthias's USDhl analysis and provides a principled basis for supply cap recommendations.

---

## 1. Motivation

### 1.1 The Independence Assumption

The current Liquip Score computes P(position i is liquidated in t days) independently for each position. The protocol-level risk is then implicitly assumed to be some aggregation of individual scores. But this ignores a critical feedback loop:

```
Price drops → Positions become undercollateralized → Liquidation triggered
→ Collateral sold on market → Price drops further → More positions liquidated → ...
```

This cascade dynamic means that the probability of position i being liquidated is not independent of the state of position j. If positions i and j both use HYPE as collateral, the liquidation of i increases the probability of j being liquidated — a correlation that is entirely absent from the current framework.

### 1.2 Evidence: The JELLY Incident

The Anthias tHLP report documents the JELLY incident on HyperCore: a whale's position caused >$10M in unrealized losses, leading to validator intervention. This is a textbook liquidation cascade — a single large position's forced liquidation overwhelmed available liquidity, moving the market far enough to trigger secondary effects. The position-level Liquip Score cannot predict the amplification; it only sees the probability of the initial trigger.

### 1.3 The Recycled Liquidity Problem

The Anthias USDhl report identifies a deeper structural issue: most LP liquidity on Felix is funded by borrowing on Felix itself. This creates a circular dependency:

```
Felix supply → Borrowing → LP deposits → AMM liquidity → Liquidation capacity
     ↑                                                          |
     └──────────────────────────────────────────────────────────┘
```

When liquidations occur, they drain the very liquidity that was meant to absorb them:
1. Liquidated collateral is sold into AMM pools, depleting pool reserves
2. Liquidators who borrowed to fund their operations may deleverage, reducing lending supply
3. Reduced lending supply tightens rates, discouraging new LP deposits
4. Lower AMM liquidity means the next liquidation has greater price impact

A cascade-aware Liquip Score would formally quantify this instability and identify the critical thresholds where the cycle becomes self-reinforcing.

---

## 2. Model

### 2.1 Setup

Consider a lending protocol with N positions, each characterized by:
- **cᵢ**: collateral amount (in units of the collateral asset)
- **dᵢ**: debt amount (in USD-denominated stablecoin)
- **MCRᵢ**: minimum collateral ratio for position i

Position i is liquidated when price P drops below its liquidation price:

```
Pᵢ* = (dᵢ × MCRᵢ) / cᵢ
```

Without loss of generality, sort positions so that P₁* ≥ P₂* ≥ ... ≥ Pₙ* (highest liquidation price first — the most vulnerable positions are liquidated earliest).

### 2.2 Market Impact Function

When position i is liquidated, its collateral cᵢ is sold on the market. The price impact of selling quantity q at current price P depends on available liquidity.

Define the market impact function I(q, P) as the fractional price decline from selling quantity q at price P:

```
P_after = P × (1 - I(q, P))
```

For an AMM with constant-product formula (x·y = k), selling Δx of the collateral asset:

```
I(Δx, P) = Δx / (x + Δx)
```

where x is the current reserve of the collateral asset. For an orderbook, the impact is determined by walking through the bid side:

```
I(q, P) = 1 - P_vwap(q) / P
```

where P_vwap(q) is the volume-weighted average execution price for selling q units.

In practice, Felix liquidations occur across multiple venues — HyperCore orderbook, Kittenswap V3, HyperSwap V3 AMMs. Define the aggregate impact function as:

```
I_agg(q, P) = I_combined(q, P; orderbook_depth, amm_reserves)
```

This can be computed empirically by aggregating the orderbook depth (from Hyperliquid `l2Book` API) and AMM pool state (from on-chain `slot0()`, `liquidity()`, `ticks()`).

### 2.3 Cascade Dynamics

Starting from an exogenous price shock that moves P from P₀ to P₁ = P₀(1 - δ):

**Round 0:** Identify positions with Pᵢ* > P₁. Let L₀ = {i : Pᵢ* > P₁} be the set of liquidated positions. Compute total collateral to be sold:

```
Q₀ = Σᵢ∈L₀ cᵢ
```

**Round 1:** The forced sale of Q₀ causes additional price impact:

```
P₂ = P₁ × (1 - I_agg(Q₀, P₁))
```

Identify newly liquidated positions: L₁ = {i ∉ L₀ : Pᵢ* > P₂}. Compute additional collateral:

```
Q₁ = Σᵢ∈L₁ cᵢ
```

**Round r:** Repeat until no new positions are liquidated:

```
Pᵣ₊₁ = Pᵣ × (1 - I_agg(Qᵣ, Pᵣ))
Lᵣ₊₁ = {i ∉ ∪ₛ₌₀ʳ Lₛ : Pᵢ* > Pᵣ₊₁}
Qᵣ₊₁ = Σᵢ∈Lᵣ₊₁ cᵢ
```

The cascade terminates when Lᵣ₊₁ = ∅ (no new liquidations triggered).

**Total cascade volume:**

```
Q_total = Σᵣ Qᵣ
```

**Terminal price:**

```
P_final = P₀ × (1-δ) × Πᵣ (1 - I_agg(Qᵣ, Pᵣ))
```

### 2.4 The Cascade Amplification Factor

Define the **cascade amplification factor** A(δ) as:

```
A(δ) = (P₀ - P_final) / (P₀ × δ)
```

This measures how much the cascade multiplies the initial price shock. A(δ) = 1 means no amplification (liquidations don't move the market). A(δ) = 3 means the final price drop is 3× the initial exogenous shock.

The cascade amplification factor is a function of:
- The distribution of liquidation prices {Pᵢ*}
- The density of positions near the current price
- Available liquidity at each price level
- The magnitude of the initial shock δ

### 2.5 Cascade-Adjusted Liquip Score

For a single position j, the **Cascade-Adjusted Liquip Score** replaces the exogenous volatility σ with an effective volatility that accounts for endogenous amplification:

```
σ_eff²(P) = σ² × A(P)²
```

More precisely, for a given price level P below current:

1. Compute the exogenous probability of reaching P: from the standard (or jump-diffusion) Liquip Score framework
2. Compute the cascade dynamics starting at P: determine P_final and the cascade amplification
3. Compute the probability of reaching P_final, accounting for the fact that if the market reaches P, it will actually fall to P_final

The cascade-adjusted Liquip Score for position j is:

```
P(position j liquidated in t days) = P(P_final < Pⱼ* in t days)
```

where P_final is the terminal price after cascade dynamics, rather than just the exogenous price.

**Simplified computation:** Rather than solving the full stochastic problem, we can approximate the cascade-adjusted score by:

1. For a grid of initial shock sizes δ ∈ {1%, 2%, ..., 50%}:
   a. Run the cascade simulation to get P_final(δ)
   b. Map δ to the effective shock: δ_eff = (P₀ - P_final) / P₀
2. Build a mapping δ → δ_eff (the "cascade function")
3. The cascade-adjusted Liquip Score for position j uses the cascade function to determine: what is the probability that the exogenous shock is large enough to cause a cascade that reaches Pⱼ*?

```
P(cascade liquidates j) = P(exogenous shock ≥ δ_min(j))
```

where δ_min(j) is the smallest initial shock that, after cascade amplification, pushes the price below Pⱼ*. This can be found by inverting the cascade function.

---

## 3. Protocol Liquip Score

### 3.1 Definition

The **Protocol Liquip Score** is a systemic risk metric that answers: what is the probability that total liquidation volume exceeds a threshold within t days?

```
Protocol Liquip Score(V, t) = P(Q_total(t) > V)
```

where Q_total is the total cascade liquidation volume resulting from whatever price path occurs over [0, t], and V is a threshold (e.g., available on-chain liquidity).

### 3.2 Computation via Monte Carlo

Unlike the position-level Liquip Score, the Protocol Liquip Score does not have a closed-form solution due to the nonlinear cascade dynamics. We compute it via Monte Carlo:

1. **Simulate price paths.** Draw M price paths from the Merton jump-diffusion model (or GBM) over [0, t]:
   ```
   For each path m = 1, ..., M:
     Simulate S_m(t) = § × exp((μ-σ²/2-λk)t + σW_m(t) + Σ jumps)
     Record minimum price along the path: P_min_m
   ```

2. **Run cascade at each path's minimum.** For each simulated path:
   ```
   δ_m = (P₀ - P_min_m) / P₀
   Run cascade dynamics to get Q_total_m
   ```

3. **Estimate Protocol Liquip Score:**
   ```
   Protocol Liquip Score(V, t) ≈ (1/M) Σ_{m=1}^M 1(Q_total_m > V)
   ```

With M = 10,000 simulations and 20-30 cascade rounds per simulation, this is computationally feasible (seconds to minutes depending on position count).

### 3.3 Connection to Supply Cap Recommendations

The Anthias USDhl report recommends supply caps based on the Debt-at-Risk (DaR) framework. The Protocol Liquip Score provides a principled way to set these caps:

**Supply cap = maximum debt level such that Protocol Liquip Score(available_liquidity, t) < α**

where α is the acceptable probability of a liquidity crisis (e.g., 1% over 30 days).

This directly extends Anthias's existing methodology: the DaR framework estimates liquidation volumes at fixed price drops, while the Protocol Liquip Score integrates over the probability-weighted distribution of price drops and accounts for cascade amplification.

---

## 4. Data Requirements and Sources

All data is available from existing sources used in `liquidation_analysis.py` and `liquidation_heatmap.py`:

| Data | Source | Use |
|------|--------|-----|
| Position data (cᵢ, dᵢ, MCRᵢ) | Felix contracts on HyperEVM (SortedTroves, TroveManager, Morpho Blue) | Liquidation price computation |
| Current prices | Hyperliquid API (`allMids`) | Position health |
| Orderbook depth | Hyperliquid API (`l2Book`) | Market impact function (orderbook component) |
| AMM pool state | On-chain (slot0, liquidity, ticks) | Market impact function (AMM component) |
| Historical returns | Hyperliquid API (`candleSnapshot`) | Volatility and jump calibration |
| Historical liquidations | Allium (`hyperliquid.dex.trades`), Felix events | Validation |

### 4.1 Liquidity Monitoring

The market impact function I_agg requires continuous monitoring of available liquidity. Key venues for HYPE:

| Venue | Type | Data Source |
|-------|------|-------------|
| HyperCore | CLOB | `l2Book` API (20 levels per side) |
| Kittenswap USDT0/WHYPE | V3 AMM | Pool at `0xec3171...e3` (~$4.76M) |
| HyperSwap feUSD/WHYPE | V3 AMM | Pool at `0x56f432...f0` |
| HyperSwap USDhl/WHYPE | V3 AMM | Pool at `0xcffd02...79` |

Aggregate liquidity as a function of price: for each price level P below current, sum the bid-side orderbook depth and AMM reserves accessible at that price.

---

## 5. Validation

### 5.1 Cascade Prediction Accuracy

**Procedure:**
1. Identify historical episodes where multiple liquidations occurred within a short window (hours)
2. At the point just before the first liquidation, compute:
   - The naive (non-cascade) prediction: total liquidation volume = Σ dᵢ for positions with Pᵢ* > P_trigger
   - The cascade prediction: run the cascade simulation from the trigger price
3. Compare both predictions against the actual total liquidation volume observed

**Metric:** Mean absolute percentage error (MAPE) of predicted vs. actual cascade volume. The cascade model should have lower MAPE, particularly for large events.

### 5.2 Protocol Liquip Score Calibration

**Procedure:**
1. Compute daily snapshots of the Protocol Liquip Score (e.g., probability of >$1M in liquidation volume within 7 days)
2. Track whether the event actually occurred in the subsequent 7 days
3. Compare predicted frequency vs. observed frequency (reliability diagram at the protocol level)

### 5.3 Stress Test Comparison

Compare the cascade model's predictions against standard stress testing approaches:

| Method | Accounts for cascade? | Accounts for liquidity? | Probabilistic? |
|--------|----------------------|------------------------|----------------|
| Fixed price drop DaR | No | No | No |
| Anthias Liquip Score | No | No | Yes |
| Historical VaR | Partially (if history includes cascades) | No | Yes |
| **Cascade-Adjusted Liquip Score** | **Yes** | **Yes** | **Yes** |

The cascade model should produce risk estimates that are (a) higher than non-cascade methods during periods of concentrated liquidation walls and thin liquidity, and (b) lower during periods of dispersed positions and deep liquidity — i.e., it should be more sensitive to the actual risk environment.

### 5.4 Case Study: USDhl Recycled Liquidity

Apply the cascade model to the specific scenario from the USDhl report:
1. Model the Felix → borrow → LP → AMM liquidity chain
2. Simulate a cascade where initial liquidations drain LP positions (which are themselves funded by Felix borrows)
3. Quantify the amplification from the liquidity feedback loop
4. Compare the cascade model's supply cap recommendation against Anthias's existing DaR-based recommendation

---

## 6. Implementation

### 6.1 Cascade Simulator

```python
def simulate_cascade(positions, current_price, initial_shock_pct,
                     orderbook_bids, amm_reserves, max_rounds=50):
    """
    Simulate a liquidation cascade from an initial price shock.

    Parameters:
        positions: list of dicts with 'collateral', 'debt', 'mcr'
        current_price: current asset price
        initial_shock_pct: initial exogenous price drop (0-1)
        orderbook_bids: list of (price, size) tuples
        amm_reserves: dict with AMM pool state
        max_rounds: maximum cascade iterations

    Returns:
        dict with terminal_price, total_liquidated_debt,
        total_liquidated_collateral, rounds, amplification_factor
    """
    # Compute liquidation prices
    for p in positions:
        p['liq_price'] = (p['debt'] * p['mcr']) / p['collateral']

    price = current_price * (1 - initial_shock_pct)
    liquidated = set()
    rounds = []

    for round_num in range(max_rounds):
        # Find newly liquidatable positions
        new_liquidations = []
        for i, p in enumerate(positions):
            if i not in liquidated and p['liq_price'] > price:
                new_liquidations.append(i)

        if not new_liquidations:
            break  # Cascade terminated

        # Compute collateral to be sold
        round_collateral = sum(positions[i]['collateral'] for i in new_liquidations)
        round_debt = sum(positions[i]['debt'] for i in new_liquidations)

        # Compute market impact
        impact = compute_aggregate_impact(
            round_collateral, price, orderbook_bids, amm_reserves
        )

        # Update state
        price = price * (1 - impact)
        liquidated.update(new_liquidations)

        rounds.append({
            'round': round_num,
            'new_liquidations': len(new_liquidations),
            'collateral_sold': round_collateral,
            'debt_liquidated': round_debt,
            'price_after': price,
            'impact': impact
        })

    total_debt = sum(positions[i]['debt'] for i in liquidated)
    total_coll = sum(positions[i]['collateral'] for i in liquidated)
    effective_shock = (current_price - price) / current_price
    amplification = effective_shock / initial_shock_pct if initial_shock_pct > 0 else 1

    return {
        'terminal_price': price,
        'total_liquidated_debt': total_debt,
        'total_liquidated_collateral': total_coll,
        'num_positions_liquidated': len(liquidated),
        'num_rounds': len(rounds),
        'amplification_factor': amplification,
        'rounds': rounds
    }


def compute_aggregate_impact(quantity, price, orderbook_bids, amm_reserves):
    """
    Compute price impact of selling `quantity` across orderbook + AMM.

    Simple model: split execution proportionally to available depth.
    """
    # Orderbook component: walk the bid side
    ob_absorbed = 0
    ob_value = 0
    remaining = quantity
    for bid_px, bid_sz in orderbook_bids:
        if bid_px > price:
            continue  # above current price, skip
        take = min(remaining, bid_sz)
        ob_absorbed += take
        ob_value += take * bid_px
        remaining -= take
        if remaining <= 0:
            break

    # AMM component: constant-product impact on remainder
    if remaining > 0 and amm_reserves.get('reserve_x', 0) > 0:
        x = amm_reserves['reserve_x']  # collateral asset reserve
        amm_impact = remaining / (x + remaining)
        return (quantity * price - ob_value - remaining * price * (1 - amm_impact)) / (quantity * price)

    if ob_absorbed > 0:
        vwap = ob_value / ob_absorbed
        return 1 - vwap / price
    return 0


def build_cascade_function(positions, current_price, orderbook_bids, amm_reserves,
                           shock_grid=None):
    """
    Build the δ → δ_eff mapping (cascade amplification curve).

    Returns arrays of (initial_shock, effective_shock, total_liquidation_volume).
    """
    if shock_grid is None:
        shock_grid = [i/200 for i in range(1, 100)]  # 0.5% to 49.5%

    results = []
    for delta in shock_grid:
        cascade = simulate_cascade(
            positions, current_price, delta, orderbook_bids, amm_reserves
        )
        results.append({
            'initial_shock': delta,
            'effective_shock': (current_price - cascade['terminal_price']) / current_price,
            'amplification': cascade['amplification_factor'],
            'total_debt_liquidated': cascade['total_liquidated_debt'],
            'num_positions': cascade['num_positions_liquidated'],
            'num_rounds': cascade['num_rounds']
        })

    return results


def protocol_liquip_score(positions, current_price, orderbook_bids, amm_reserves,
                          sigma, mu=0, lambda_j=0, mu_j=0, sigma_j=0,
                          days_forward=7, volume_threshold=1e6,
                          n_simulations=10000):
    """
    Monte Carlo estimate of Protocol Liquip Score.

    P(total cascade liquidation volume > volume_threshold within days_forward)
    """
    import numpy as np

    dt = days_forward
    k = np.exp(mu_j + sigma_j**2/2) - 1 if lambda_j > 0 else 0
    breach_count = 0

    for _ in range(n_simulations):
        # Simulate minimum price along path (using reflection principle approximation)
        # For GBM/Merton, the distribution of the running minimum is known
        Z = np.random.normal()
        log_return = (mu - sigma**2/2 - lambda_j*k) * dt + sigma * np.sqrt(dt) * Z

        # Add jumps
        n_jumps = np.random.poisson(lambda_j * dt)
        if n_jumps > 0:
            jump_returns = np.random.normal(mu_j, sigma_j, n_jumps)
            log_return += np.sum(jump_returns)

        # Approximate minimum price using Brownian bridge
        # P(min(S) < L | S(T)) ≈ exp(-2 * ln(S0/L) * ln(S(T)/L) / (σ²T))
        S_T = current_price * np.exp(log_return)
        min_price = min(current_price, S_T)  # conservative approximation

        # More accurate: sample the path minimum
        # For simplicity, use the terminal value as lower bound
        shock_pct = max(0, (current_price - min_price) / current_price)

        if shock_pct > 0.005:  # only run cascade for non-trivial shocks
            cascade = simulate_cascade(
                positions, current_price, shock_pct,
                orderbook_bids, amm_reserves
            )
            if cascade['total_liquidated_debt'] > volume_threshold:
                breach_count += 1

    return breach_count / n_simulations
```

### 6.2 Visualization: Cascade Amplification Curve

The cascade function δ → δ_eff produces a powerful diagnostic visualization:

- **x-axis:** Initial exogenous shock (%)
- **y-axis:** Terminal effective shock after cascade (%)
- **45-degree line:** No amplification (A = 1)
- **Curve above the line:** Cascade amplification
- **Steep jumps:** "Liquidation walls" where a small additional shock triggers a large new tranche of liquidations
- **Vertical asymptote:** If it exists, indicates the point where the cascade becomes self-sustaining (price impact from liquidations exceeds remaining collateral buffer)

This curve gives the MM team an at-a-glance view of where the systemic risk concentrates.

---

## 7. Connection to MM Strategy

### 7.1 Spread Adjustment Signal

The cascade amplification factor A(δ) directly informs spread management:

```
spread_adjustment = β × max(A(δ_nearby) - 1, 0) × (1 / distance_to_wall)
```

When a large liquidation wall is nearby (δ_nearby is small) and the cascade amplification is high, widen spreads to compensate for increased adverse selection risk.

### 7.2 Inventory Positioning

Before a liquidation wall: lean inventory into the expected flow direction. If a downward cascade is likely:
- Reduce long exposure
- Build short hedges on HyperCore perps
- Pre-position bids below the cascade terminal price to buy cheap collateral after forced selling exhausts

### 7.3 Liquidity Monitoring Alerts

Generate alerts when:
- Cascade amplification factor exceeds threshold (e.g., A > 2 for any shock < 20%)
- Protocol Liquip Score exceeds threshold (e.g., >5% probability of >$1M cascade in 7 days)
- Available on-chain liquidity drops below cascade-implied demand at stressed price levels

---

## 8. Limitations

### 8.1 Static Liquidity Assumption

The model assumes orderbook depth and AMM reserves are fixed during the cascade. In reality:
- Market makers may withdraw quotes as prices fall (endogenous liquidity withdrawal)
- Arbitrageurs may add liquidity between cascade rounds
- AMM LPs may rebalance their positions

The static assumption is conservative for orderbook liquidity (assuming it stays is optimistic during a crash — it usually vanishes) but may be either direction for AMM liquidity.

### 8.2 Timing and Sequencing

The model assumes all liquidations within a round execute simultaneously at the same price. In practice, liquidations are sequential transactions that may execute at progressively worse prices. The model's round-based approach approximates this but doesn't capture the exact path dependence.

### 8.3 Cross-Asset Cascades

The current model considers cascades within a single collateral asset (e.g., all HYPE-collateralized positions). In reality, HYPE liquidations can affect BTC and ETH prices through cross-asset correlations and sentiment contagion. A full treatment would require a multi-asset cascade model, which we leave for future work.

### 8.4 Oracle Delays

Felix uses RedStone pull oracles. The delay between the true market price and the oracle price used for liquidation triggers creates a "shadow zone" where cascade dynamics may differ from the model's predictions. Incorporating oracle latency distributions would improve accuracy.

---

## 9. Future Work

### 9.1 Agent-Based Cascade Simulation

Replace the aggregate market impact function with an agent-based model that includes:
- **Liquidator agents:** Each with a capital budget and strategy (inventory-based, flash-loan-based)
- **LP agents:** Who may withdraw liquidity under stress
- **Arbitrageur agents:** Who provide cross-venue liquidity
- **Borrower agents:** Who may voluntarily deleverage when they see nearby liquidations

This would capture the rich dynamics of the recycled liquidity problem identified in the USDhl report more faithfully than the aggregate approach.

### 9.2 Hawkes Process Integration

Combine the cascade model with a Hawkes self-exciting process (from Proposal A's future work), where each liquidation event increases the intensity of subsequent jumps. This formally connects the endogenous cascade dynamics to the exogenous price process.

### 9.3 Real-Time Dashboard

Build a live monitoring dashboard that:
1. Scrapes Felix position data every block
2. Computes the cascade amplification curve in real-time
3. Overlays current orderbook and AMM depth
4. Displays the Protocol Liquip Score as a single headline risk number
5. Generates alerts when systemic risk thresholds are breached

---

## 10. Conclusion

The Cascade-Adjusted Liquip Score addresses a fundamental limitation of position-level risk metrics: the assumption that liquidations are independent events driven only by exogenous price dynamics. By modeling the feedback loop between liquidation-induced selling pressure and available on-chain liquidity, this framework provides a more realistic assessment of tail risk in DeFi lending protocols.

The practical value is threefold:
1. **For risk parameterization:** Protocol-level Liquip Scores provide a principled basis for setting supply caps, collateral factors, and liquidation incentives — the core service Anthias provides to its clients
2. **For market making:** The cascade amplification curve gives the MM team advance warning of price levels where adverse selection risk spikes, enabling proactive spread and inventory management
3. **For monitoring:** The systemic risk metric provides a single number that captures the protocol's current exposure to liquidation cascades, complementing the position-level Liquip Scores already in production

This proposal directly extends two of Anthias's published works — the Liquip Score methodology and the USDhl recycled liquidity analysis — combining them into a unified framework for systemic risk assessment.

---

## References

- Anthias Team. "Anthias Liquip Score: Measuring Liquidation Probability."
- Anthias Labs. "Asset Risk Analysis - Felix USDhl." June 2025.
- Anthias Labs. "Tokenized HLP Risk Analysis." (JELLY incident documentation)
- Cifuentes, R., Ferrucci, G., & Shin, H.S. (2005). "Liquidity risk and contagion." *Journal of the European Economic Association*, 3(2-3), 556-566.
- Brunnermeier, M.K. & Pedersen, L.H. (2009). "Market liquidity and funding liquidity." *Review of Financial Studies*, 22(6), 2201-2238.
- Cont, R. & Wagalath, L. (2016). "Fire sales forensics: measuring endogenous risk." *Mathematical Finance*, 26(4), 835-866.
- Gauntlet. (2023). "DeFi Risk Management: Liquidation Cascades and Protocol Solvency." (Industry framework)
