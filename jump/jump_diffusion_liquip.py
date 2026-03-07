"""
Jump-Diffusion Liquip Score — Preliminary Results
===================================================
Extends Anthias Liquip Score with Merton jump-diffusion.
Pulls live data from Hyperliquid, calibrates the model,
and compares GBM vs Merton scores on real Felix positions.

For Anthias Labs interview — John Beecher, Feb 2026
"""

import json
import time
import math
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm, kurtosis, skew, jarque_bera, kstest
from web3 import Web3

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

HYPEREVM_RPC = "https://rpc.hyperliquid.xyz/evm"
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"

CDP_BRANCHES = {
    "WHYPE": {
        "sorted_troves": "0xd1caa4218808eb94d36e1df7247f7406f43f2ef6",
        "trove_manager": "0x3100f4e7bda2ed2452d9a57eb30260ab071bbe62",
        "price_coin": "HYPE",
        "decimals": 18,
    },
    "UBTC": {
        "sorted_troves": "0x642d979341eaac9c10623f5a58283aa72f6e2fa9",
        "trove_manager": "0xbbe5f227275f24b64bd290a91f55723a00214885",
        "price_coin": "BTC",
        "decimals": 8,
    },
    "kHYPE": {
        "sorted_troves": "0x6bc81472c10ec526c14c8b0e8faa282f9368f86f",
        "trove_manager": "0x7c07bb77b1cf9a5b40d92f805c10d90c90957e4a",
        "price_coin": "HYPE",
        "decimals": 18,
    },
}

MCR = 1.10

SORTED_TROVES_ABI = json.loads("""[
    {"inputs":[],"name":"getFirst","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"type":"uint256"}],"name":"getNext","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"getSize","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"}
]""")

TROVE_MANAGER_ABI = json.loads("""[
    {"inputs":[{"type":"uint256"}],"name":"getLatestTroveData","outputs":[{"components":[
        {"name":"entireColl","type":"uint256"},
        {"name":"entireDebt","type":"uint256"},
        {"name":"redistCollGain","type":"uint256"},
        {"name":"redistBoldDebtGain","type":"uint256"},
        {"name":"accruedInterest","type":"uint256"},
        {"name":"recordedDebt","type":"uint256"},
        {"name":"annualInterestRate","type":"uint256"},
        {"name":"weightedRecordedDebt","type":"uint256"},
        {"name":"accruedBatchManagementFee","type":"uint256"},
        {"name":"lastInterestRateAdjTime","type":"uint256"}
    ],"type":"tuple"}],"stateMutability":"view","type":"function"}
]""")


# ============================================================================
# Data Fetching
# ============================================================================

def fetch_candles(coin="HYPE", interval="1h", days=90):
    """Fetch OHLCV data from Hyperliquid."""
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 3600 * 1000)

    all_candles = []
    cursor = start_time

    # Paginate — max 5000 candles per request
    while cursor < end_time:
        resp = requests.post(HYPERLIQUID_API, json={
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval,
                    "startTime": cursor, "endTime": end_time}
        }, timeout=15)
        candles = resp.json()
        if not candles:
            break
        all_candles.extend(candles)
        # Move cursor past the last candle
        last_t = candles[-1]["t"]
        if last_t <= cursor:
            break
        cursor = last_t + 1
        time.sleep(0.3)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["t"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    df["v"] = df["v"].astype(float)
    df = df.sort_values("t").reset_index(drop=True)
    return df


def fetch_prices():
    """Get current prices from Hyperliquid."""
    resp = requests.post(HYPERLIQUID_API, json={"type": "allMids"}, timeout=10)
    mids = resp.json()
    prices = {}
    for coin in ["HYPE", "BTC", "ETH"]:
        if coin in mids:
            prices[coin] = float(mids[coin])
    return prices


def fetch_cdp_positions(w3, branch_name, config, max_troves=500):
    """Fetch all CDP positions from a branch."""
    sorted_troves = w3.eth.contract(
        address=Web3.to_checksum_address(config["sorted_troves"]),
        abi=SORTED_TROVES_ABI
    )
    trove_manager = w3.eth.contract(
        address=Web3.to_checksum_address(config["trove_manager"]),
        abi=TROVE_MANAGER_ABI
    )

    try:
        size = sorted_troves.functions.getSize().call()
    except Exception as e:
        print(f"  [{branch_name}] Error: {e}")
        return []

    if size == 0:
        return []

    print(f"  [{branch_name}] {size} troves")
    positions = []

    # Rate-limit-safe initial call
    for attempt in range(5):
        try:
            current_id = sorted_troves.functions.getFirst().call()
            break
        except Exception as e:
            if attempt < 4:
                wait = 3 * (attempt + 1)
                print(f"  [{branch_name}] Rate limited on getFirst, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [{branch_name}] Failed after 5 attempts: {e}")
                return []

    count = 0
    retries = 0

    while current_id != 0 and count < min(size, max_troves):
        try:
            trove_data = trove_manager.functions.getLatestTroveData(current_id).call()
            entire_coll = trove_data[0] / (10 ** config["decimals"])
            entire_debt = trove_data[1] / 1e18

            if entire_coll > 0 and entire_debt > 0:
                positions.append({
                    "id": current_id,
                    "branch": branch_name,
                    "collateral": entire_coll,
                    "debt": entire_debt,
                })

            time.sleep(0.4)
            current_id = sorted_troves.functions.getNext(current_id).call()
            count += 1
            retries = 0

        except Exception as e:
            if "rate limited" in str(e).lower() and retries < 5:
                retries += 1
                wait = 3 * retries
                print(f"  [{branch_name}] Rate limited, waiting {wait}s (attempt {retries}/5)...")
                time.sleep(wait)
                continue
            try:
                time.sleep(1)
                current_id = sorted_troves.functions.getNext(current_id).call()
            except:
                break
            count += 1

    print(f"  [{branch_name}] Fetched {len(positions)} positions")
    return positions


# ============================================================================
# Return Statistics
# ============================================================================

def compute_return_statistics(df, coin_name):
    """Compute comprehensive return statistics to test normality."""
    df = df.copy()
    df["log_ret"] = np.log(df["c"] / df["c"].shift(1))
    returns = df["log_ret"].dropna().values

    stats = {
        "coin": coin_name,
        "n_obs": len(returns),
        "mean_hourly": np.mean(returns),
        "std_hourly": np.std(returns, ddof=1),
        "annualized_vol": np.std(returns, ddof=1) * np.sqrt(8760),
        "skewness": skew(returns),
        "excess_kurtosis": kurtosis(returns, fisher=True),  # excess kurtosis (normal = 0)
        "min_return": np.min(returns),
        "max_return": np.max(returns),
        "pct_beyond_2sigma": np.mean(np.abs(returns - np.mean(returns)) > 2 * np.std(returns)) * 100,
        "pct_beyond_3sigma": np.mean(np.abs(returns - np.mean(returns)) > 3 * np.std(returns)) * 100,
        # Under normal: expect 4.55% beyond 2σ, 0.27% beyond 3σ
    }

    # Jarque-Bera test for normality
    jb_stat, jb_pvalue = jarque_bera(returns)
    stats["jarque_bera_stat"] = jb_stat
    stats["jarque_bera_pvalue"] = jb_pvalue

    return stats, returns


# ============================================================================
# Merton Calibration
# ============================================================================

def heuristic_calibration(returns, dt=1.0, threshold_sigma=3.0, n_iterations=3):
    """
    Stage 1: Heuristic calibration of Merton parameters.
    Iteratively filters jumps and estimates (sigma, lambda, mu_j, sigma_j).
    """
    sigma_hat = np.std(returns, ddof=1)
    mu_hat = np.mean(returns)
    mask = np.ones(len(returns), dtype=bool)

    for iteration in range(n_iterations):
        threshold = threshold_sigma * sigma_hat
        mask = np.abs(returns - mu_hat) < threshold
        if mask.sum() > 10:
            sigma_hat = np.std(returns[mask], ddof=1)
            mu_hat = np.mean(returns[mask])

    jump_mask = ~mask
    n_jumps = jump_mask.sum()
    lambda_hat = n_jumps / (len(returns) * dt)

    if n_jumps >= 2:
        mu_j_hat = np.mean(returns[jump_mask])
        sigma_j_hat = np.std(returns[jump_mask], ddof=1)
    elif n_jumps == 1:
        mu_j_hat = returns[jump_mask][0]
        sigma_j_hat = abs(mu_j_hat) * 0.5  # rough prior
    else:
        mu_j_hat = -2 * sigma_hat
        sigma_j_hat = sigma_hat
        lambda_hat = 0.005 / dt

    # Compensator
    k = np.exp(mu_j_hat + sigma_j_hat**2 / 2) - 1
    drift_hat = mu_hat / dt + lambda_hat * k

    return {
        "mu": drift_hat,
        "sigma": sigma_hat,
        "lambda": lambda_hat,
        "mu_j": mu_j_hat,
        "sigma_j": sigma_j_hat,
        "n_jumps_detected": n_jumps,
        "jump_returns": returns[jump_mask] if n_jumps > 0 else np.array([]),
        "jump_indices": np.where(jump_mask)[0],
    }


def merton_log_likelihood(params, returns, dt):
    """Negative log-likelihood for Merton jump-diffusion."""
    mu, sigma, lam, mu_j, sigma_j = params
    if sigma <= 1e-10 or lam < 0 or sigma_j <= 1e-10:
        return 1e15

    k = np.exp(mu_j + sigma_j**2 / 2) - 1
    n_terms = 20
    ll = 0.0

    for r in returns:
        pdf_sum = 0.0
        log_poisson = -lam * dt  # log of e^{-λΔt}

        for n in range(n_terms):
            if n > 0:
                log_poisson += np.log(lam * dt) - np.log(n)

            m_n = (mu - sigma**2 / 2 - lam * k) * dt + n * mu_j
            v_n2 = sigma**2 * dt + n * sigma_j**2
            if v_n2 <= 0:
                continue

            # Log of Poisson weight + normal PDF, computed in log space for stability
            log_norm = -0.5 * np.log(2 * np.pi * v_n2) - 0.5 * (r - m_n)**2 / v_n2
            pdf_sum += np.exp(log_poisson + log_norm)

        if pdf_sum > 1e-300:
            ll += np.log(pdf_sum)
        else:
            # ln(1e-300) = -691
            ll -= 700

    return -ll


def gbm_log_likelihood(returns, dt):
    """Log-likelihood under GBM (normal returns)."""
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    n = len(returns)

    ll = -n/2 * np.log(2 * np.pi) - n * np.log(sigma) - \
         np.sum((returns - mu)**2) / (2 * sigma**2)
    return ll


def mle_calibration(returns, dt=1.0, heuristic_params=None):
    """
    Stage 2: Maximum likelihood estimation of Merton parameters.
    Uses heuristic params as initialization.
    """
    if heuristic_params is None:
        heuristic_params = heuristic_calibration(returns, dt)

    x0 = [
        heuristic_params["mu"],
        heuristic_params["sigma"],
        heuristic_params["lambda"],
        heuristic_params["mu_j"],
        heuristic_params["sigma_j"],
    ]

    bounds = [
        (None, None),       # mu
        (1e-8, None),       # sigma > 0
        (1e-8, None),       # lambda >= 0
        (None, None),       # mu_j
        (1e-8, None),       # sigma_j > 0
    ]

    print("    Running MLE optimization...")
    result = minimize(
        merton_log_likelihood, x0, args=(returns, dt),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10}
    )

    if result.success:
        params = dict(zip(["mu", "sigma", "lambda", "mu_j", "sigma_j"], result.x))
        params["mle_converged"] = True
        params["neg_log_likelihood"] = result.fun
        params["log_likelihood"] = -result.fun
        print(f"    MLE converged: LL = {-result.fun:.2f}")
    else:
        print(f"    MLE did not converge: {result.message}")
        params = {
            "mu": heuristic_params["mu"],
            "sigma": heuristic_params["sigma"],
            "lambda": heuristic_params["lambda"],
            "mu_j": heuristic_params["mu_j"],
            "sigma_j": heuristic_params["sigma_j"],
            "mle_converged": False,
            "neg_log_likelihood": result.fun,
            "log_likelihood": -result.fun,
        }

    return params


# ============================================================================
# Liquip Score Implementations
# ============================================================================

def liquip_score_gbm(collateral_value, debt_value, sigma_daily, mu_daily=0, days_forward=7):
    """Original Anthias Liquip Score (GBM)."""
    if collateral_value <= debt_value:
        return 1.0
    if sigma_daily <= 0:
        return 0.0

    t = days_forward
    threshold = debt_value / collateral_value
    if threshold <= 0:
        return 0.0

    ln_threshold = math.log(threshold)
    numerator = ln_threshold - (mu_daily - sigma_daily**2 / 2) * t
    denominator = sigma_daily * math.sqrt(t)

    if denominator == 0:
        return 0.0

    z = numerator / denominator
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def liquip_score_merton(collateral_value, debt_value, sigma, mu=0,
                        lambda_j=0.02, mu_j=-0.05, sigma_j=0.05,
                        days_forward=7):
    """Jump-Diffusion Liquip Score (Merton).

    Computes P(S(t) < liquidation_threshold) under Merton jump-diffusion.
    Uses Poisson-weighted sum of conditional Normal CDFs.
    Automatically determines summation range around the Poisson peak.
    """
    S0 = collateral_value
    phi = collateral_value - debt_value

    if phi <= 0:
        return 1.0
    if sigma <= 0 and lambda_j <= 0:
        return 0.0

    t = days_forward
    threshold = debt_value / collateral_value
    if threshold <= 0:
        return 0.0

    ln_threshold = math.log(threshold)
    k = math.exp(mu_j + sigma_j**2 / 2) - 1
    lambda_t = lambda_j * t

    # Determine summation range: center on Poisson mean, extend ±4 std devs
    poisson_mean = lambda_t
    poisson_std = max(math.sqrt(lambda_t), 1)
    n_lo = max(0, int(poisson_mean - 4 * poisson_std))
    n_hi = int(poisson_mean + 4 * poisson_std) + 1
    n_hi = max(n_hi, 25)  # always sum at least 25 terms from 0

    liquip = 0.0
    # Compute log-Poisson weight starting from n_lo
    if n_lo == 0:
        log_poisson = -lambda_t
    else:
        # log(e^{-λt} (λt)^n / n!) = -λt + n*log(λt) - log(n!)
        log_poisson = -lambda_t + n_lo * math.log(lambda_t) - sum(math.log(i) for i in range(1, n_lo + 1))

    for n in range(n_lo, n_hi):
        if n > n_lo:
            log_poisson += math.log(lambda_t) - math.log(n)

        poisson_w = math.exp(log_poisson)

        m_n = (mu - sigma**2 / 2 - lambda_j * k) * t + n * mu_j
        v_n2 = sigma**2 * t + n * sigma_j**2

        if v_n2 <= 0:
            continue

        v_n = math.sqrt(v_n2)
        z_n = (ln_threshold - m_n) / v_n
        cdf_n = 0.5 * (1 + math.erf(z_n / math.sqrt(2)))

        liquip += poisson_w * cdf_n

    return liquip


def days_until_liquidation(score_func, collateral_value, debt_value,
                           target_prob=0.05, **kwargs):
    """Bisection method for days until Liquip Score reaches target_prob."""
    if collateral_value <= debt_value:
        return 0

    a, b = 0.01, 365 * 10

    fa = score_func(collateral_value, debt_value, days_forward=a, **kwargs) - target_prob
    if fa >= 0:
        return 0

    fb = score_func(collateral_value, debt_value, days_forward=b, **kwargs) - target_prob
    if fb <= 0:
        return float("inf")

    for _ in range(100):
        c = (a + b) / 2
        fc = score_func(collateral_value, debt_value, days_forward=c, **kwargs) - target_prob
        if abs(fc) < 1e-6 or (b - a) < 0.01:
            return c
        if fc < 0:
            a = c
        else:
            b = c

    return (a + b) / 2


# ============================================================================
# Likelihood Ratio Test
# ============================================================================

def likelihood_ratio_test(returns, dt, merton_params):
    """
    Test GBM (H0) vs Merton (H1) using likelihood ratio.
    GBM is nested in Merton (lambda=0).
    Test statistic D = -2(LL_GBM - LL_Merton) ~ chi2(3) under H0.
    """
    ll_gbm = gbm_log_likelihood(returns, dt)
    ll_merton = merton_params["log_likelihood"]

    D = -2 * (ll_gbm - ll_merton)
    # 3 additional parameters: lambda, mu_j, sigma_j
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(D, df=3)

    return {
        "ll_gbm": ll_gbm,
        "ll_merton": ll_merton,
        "test_statistic": D,
        "p_value": p_value,
        "reject_gbm_at_1pct": p_value < 0.01,
        "reject_gbm_at_5pct": p_value < 0.05,
    }


# ============================================================================
# Visualization
# ============================================================================

def generate_dashboard(candles_dict, stats_dict, calibrations, positions_data,
                       lr_tests, prices):
    """Generate the full comparison dashboard."""
    fig = plt.figure(figsize=(22, 28))
    fig.suptitle(
        "Jump-Diffusion Liquip Score: Preliminary Results\n"
        "Extending Anthias Liquip Score with Merton Jump-Diffusion",
        fontsize=18, fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(5, 2, hspace=0.38, wspace=0.3,
                           top=0.96, bottom=0.03, left=0.07, right=0.95)

    coins = ["HYPE", "BTC", "ETH"]
    colors = {"HYPE": "#4CAF50", "BTC": "#FF9800", "ETH": "#2196F3"}

    # ------------------------------------------------------------------
    # Panel 1: Return Distributions with Normal Overlay
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    for i, coin in enumerate(coins):
        if coin not in candles_dict:
            continue
        s = stats_dict[coin]
        df = candles_dict[coin].copy()
        df["log_ret"] = np.log(df["c"] / df["c"].shift(1))
        rets = df["log_ret"].dropna().values

        # Histogram
        bins = np.linspace(np.percentile(rets, 0.5), np.percentile(rets, 99.5), 80)
        ax1.hist(rets, bins=bins, alpha=0.35, color=colors[coin], density=True,
                 label=f"{coin} (kurt={s['excess_kurtosis']:.1f}, skew={s['skewness']:.2f})")

        # Normal overlay
        x_norm = np.linspace(bins[0], bins[-1], 200)
        ax1.plot(x_norm, norm.pdf(x_norm, np.mean(rets), np.std(rets)),
                 color=colors[coin], linewidth=1.5, linestyle="--", alpha=0.7)

    ax1.set_xlabel("Hourly Log-Return")
    ax1.set_ylabel("Density")
    ax1.set_title("Empirical Return Distributions vs Normal (dashed) — Evidence of Fat Tails")
    ax1.legend(fontsize=10)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.01)
    ax1.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2: QQ Plot for HYPE
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    if "HYPE" in candles_dict:
        df = candles_dict["HYPE"].copy()
        df["log_ret"] = np.log(df["c"] / df["c"].shift(1))
        rets = df["log_ret"].dropna().values
        rets_sorted = np.sort(rets)
        n = len(rets_sorted)
        theoretical_q = norm.ppf(np.arange(1, n + 1) / (n + 1))

        ax2.scatter(theoretical_q, rets_sorted, s=3, alpha=0.4, color="#4CAF50")
        # Reference line
        q25, q75 = np.percentile(rets_sorted, [25, 75])
        t25, t75 = norm.ppf(0.25), norm.ppf(0.75)
        slope = (q75 - q25) / (t75 - t25)
        intercept = q25 - slope * t25
        x_line = np.array([theoretical_q[0], theoretical_q[-1]])
        ax2.plot(x_line, slope * x_line + intercept, "r--", linewidth=2, label="Normal reference")
        ax2.set_xlabel("Theoretical Quantiles (Normal)")
        ax2.set_ylabel("Sample Quantiles")
        ax2.set_title("HYPE QQ-Plot: Fat Tails Visible at Extremes")
        ax2.legend()
        ax2.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 3: Normality Test Summary Table
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    table_data = []
    headers = ["", "HYPE", "BTC", "ETH"]
    rows = [
        ("Observations", "n_obs"),
        ("Ann. Volatility", "annualized_vol"),
        ("Skewness", "skewness"),
        ("Excess Kurtosis", "excess_kurtosis"),
        ("% Beyond 2σ", "pct_beyond_2sigma"),
        ("% Beyond 3σ", "pct_beyond_3sigma"),
        ("Jarque-Bera p-val", "jarque_bera_pvalue"),
    ]
    for label, key in rows:
        row = [label]
        for coin in coins:
            if coin in stats_dict:
                val = stats_dict[coin][key]
                if key == "n_obs":
                    row.append(f"{val:,}")
                elif key in ("annualized_vol",):
                    row.append(f"{val:.1%}")
                elif key in ("pct_beyond_2sigma", "pct_beyond_3sigma"):
                    expected = 4.55 if "2sigma" in key else 0.27
                    row.append(f"{val:.2f}% (exp: {expected}%)")
                elif key == "jarque_bera_pvalue":
                    row.append(f"{val:.2e}")
                else:
                    row.append(f"{val:.3f}")
            else:
                row.append("N/A")
        table_data.append(row)

    table = ax3.table(cellText=table_data, colLabels=headers,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")
    ax3.set_title("Return Distribution Statistics (Normal: kurt=0, skew=0)", fontsize=12,
                  fontweight="bold", pad=20)

    # ------------------------------------------------------------------
    # Panel 4: Merton Calibration Results
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis("off")
    cal_data = []
    cal_headers = ["Parameter", "HYPE", "BTC", "ETH"]
    param_rows = [
        ("σ (diffusion, hourly)", "sigma"),
        ("λ (jumps/hour)", "lambda"),
        ("μ_J (mean jump)", "mu_j"),
        ("σ_J (jump std)", "sigma_j"),
        ("Jumps detected", "n_jumps"),
        ("MLE converged", "mle_converged"),
    ]
    for label, key in param_rows:
        row = [label]
        for coin in coins:
            if coin in calibrations:
                cal = calibrations[coin]
                if key == "n_jumps":
                    row.append(str(cal.get("n_jumps_detected", "N/A")))
                elif key == "mle_converged":
                    mle = cal.get("mle", {})
                    row.append("Yes" if mle.get("mle_converged", False) else "No")
                else:
                    mle = cal.get("mle", cal.get("heuristic", {}))
                    val = mle.get(key, 0)
                    if key == "lambda":
                        row.append(f"{val:.5f}")
                    else:
                        row.append(f"{val:.6f}")
            else:
                row.append("N/A")
        cal_data.append(row)

    table2 = ax4.table(cellText=cal_data, colLabels=cal_headers,
                       loc="center", cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.0, 1.6)
    for (r, c), cell in table2.get_celld().items():
        if r == 0:
            cell.set_facecolor("#FFF3E0")
            cell.set_text_props(fontweight="bold")
    ax4.set_title("Merton Calibration (Heuristic → MLE)", fontsize=12,
                  fontweight="bold", pad=20)

    # ------------------------------------------------------------------
    # Panel 5: Likelihood Ratio Test Results
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    lr_data = []
    lr_headers = ["Metric", "HYPE", "BTC", "ETH"]
    lr_rows = [
        ("LL (GBM)", "ll_gbm"),
        ("LL (Merton)", "ll_merton"),
        ("LL Improvement", "ll_improvement"),
        ("Test Stat D", "test_statistic"),
        ("p-value", "p_value"),
        ("Reject GBM (5%)", "reject_gbm_at_5pct"),
    ]
    for label, key in lr_rows:
        row = [label]
        for coin in coins:
            if coin in lr_tests:
                lr = lr_tests[coin]
                if key == "ll_improvement":
                    row.append(f"{lr['ll_merton'] - lr['ll_gbm']:.1f}")
                elif key in ("ll_gbm", "ll_merton"):
                    row.append(f"{lr[key]:.1f}")
                elif key == "test_statistic":
                    row.append(f"{lr[key]:.2f}")
                elif key == "p_value":
                    row.append(f"{lr[key]:.2e}")
                elif key == "reject_gbm_at_5pct":
                    row.append("YES" if lr[key] else "No")
                else:
                    row.append(str(lr.get(key, "N/A")))
            else:
                row.append("N/A")
        lr_data.append(row)

    table3 = ax5.table(cellText=lr_data, colLabels=lr_headers,
                       loc="center", cellLoc="center")
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1.0, 1.6)
    for (r, c), cell in table3.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E8F5E9")
            cell.set_text_props(fontweight="bold")
        if r > 0 and c > 0:
            # Highlight "YES" in green
            if cell.get_text().get_text() == "YES":
                cell.set_facecolor("#C8E6C9")
    ax5.set_title("Likelihood Ratio Test: GBM vs Merton (H₀: λ=0)", fontsize=12,
                  fontweight="bold", pad=20)

    # ------------------------------------------------------------------
    # Panel 6: GBM vs Merton Liquip Scores by Time Horizon
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[3, 0])
    if positions_data:
        # Pick a representative position (median leverage from WHYPE)
        whype = positions_data.get("WHYPE", {})
        if whype.get("positions"):
            pos_list = whype["positions"]
            price = whype["price"]
            # Sort by ICR, pick median
            for p in pos_list:
                p["coll_usd"] = p["collateral"] * price
                p["icr"] = p["coll_usd"] / p["debt"] if p["debt"] > 0 else float("inf")
            pos_list.sort(key=lambda x: x["icr"])
            rep_pos = pos_list[len(pos_list) // 2]

            cal = calibrations.get("HYPE", {})
            mle = cal.get("mle", cal.get("heuristic", {}))
            sigma_h = mle.get("sigma", 0.01)
            sigma_d = sigma_h * np.sqrt(24)  # convert hourly to daily
            lambda_h = mle.get("lambda", 0)
            lambda_d = lambda_h * 24  # convert hourly to daily

            horizons = np.arange(1, 181, 1)
            gbm_scores = []
            merton_scores = []

            for t in horizons:
                gs_val = liquip_score_gbm(rep_pos["coll_usd"], rep_pos["debt"],
                                          sigma_d, days_forward=t)
                ms_val = liquip_score_merton(rep_pos["coll_usd"], rep_pos["debt"],
                                            sigma_d, lambda_j=lambda_d,
                                            mu_j=mle.get("mu_j", -0.03),
                                            sigma_j=mle.get("sigma_j", 0.03),
                                            days_forward=t)
                gbm_scores.append(gs_val)
                merton_scores.append(ms_val)

            ax6.plot(horizons, [s * 100 for s in gbm_scores], "b-", linewidth=2,
                     label="GBM Liquip Score")
            ax6.plot(horizons, [s * 100 for s in merton_scores], "r-", linewidth=2,
                     label="Merton Liquip Score")
            ax6.fill_between(horizons,
                             [g * 100 for g in gbm_scores],
                             [m * 100 for m in merton_scores],
                             alpha=0.2, color="red", label="Jump risk underestimation")
            ax6.set_xlabel("Time Horizon (days)")
            ax6.set_ylabel("Liquidation Probability (%)")
            ax6.set_title(f"GBM vs Merton: Representative WHYPE Position\n"
                          f"(ICR={rep_pos['icr']:.0%}, Debt=${rep_pos['debt']:,.0f})")
            ax6.legend(fontsize=10)
            ax6.grid(alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "No WHYPE positions loaded", ha="center", va="center")
            ax6.set_title("GBM vs Merton by Horizon")
    else:
        ax6.text(0.5, 0.5, "No position data available", ha="center", va="center")
        ax6.set_title("GBM vs Merton by Horizon")

    # ------------------------------------------------------------------
    # Panel 7: GBM vs Merton Across All Positions (scatter)
    # ------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[3, 1])
    if positions_data:
        all_gbm = []
        all_merton = []
        all_branches = []
        branch_colors = {"WHYPE": "#4CAF50", "UBTC": "#FF9800", "kHYPE": "#9C27B0"}

        for branch_name, bdata in positions_data.items():
            coin = bdata.get("coin", "HYPE")
            price = bdata["price"]
            cal = calibrations.get(coin, {})
            mle = cal.get("mle", cal.get("heuristic", {}))
            sigma_h = mle.get("sigma", 0.01)
            sigma_d = sigma_h * np.sqrt(24)
            lambda_h = mle.get("lambda", 0)
            lambda_d = lambda_h * 24

            for p in bdata["positions"]:
                coll_usd = p["collateral"] * price
                debt = p["debt"]
                gs_val = liquip_score_gbm(coll_usd, debt, sigma_d, days_forward=30)
                ms_val = liquip_score_merton(coll_usd, debt, sigma_d,
                                            lambda_j=lambda_d,
                                            mu_j=mle.get("mu_j", -0.03),
                                            sigma_j=mle.get("sigma_j", 0.03),
                                            days_forward=30)
                all_gbm.append(gs_val * 100)
                all_merton.append(ms_val * 100)
                all_branches.append(branch_name)

        for branch, color in branch_colors.items():
            mask = [b == branch for b in all_branches]
            bx = [all_gbm[i] for i in range(len(mask)) if mask[i]]
            by = [all_merton[i] for i in range(len(mask)) if mask[i]]
            if bx:
                ax7.scatter(bx, by, c=color, alpha=0.6, s=30, edgecolors="k",
                            linewidths=0.3, label=branch)

        # Diagonal reference
        max_val = max(max(all_gbm + [0.001]), max(all_merton + [0.001]))
        ax7.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="GBM = Merton")
        ax7.set_xlabel("GBM Liquip Score (30-day, %)")
        ax7.set_ylabel("Merton Liquip Score (30-day, %)")
        ax7.set_title("GBM vs Merton: All CDP Positions (30-day horizon)")
        ax7.legend(fontsize=9)
        ax7.grid(alpha=0.3)
    else:
        ax7.text(0.5, 0.5, "No position data", ha="center", va="center")

    # ------------------------------------------------------------------
    # Panel 8: Simulated Morpho Blue Comparison (high leverage)
    # ------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[4, 0])
    # Since CDPs are hella overcollateralized, simulate Morpho Blue-like positions
    # at more degenerate LTVs to show where the difference matters
    ltvs = np.arange(0.50, 0.91, 0.01)
    base_coll = 25000  # $25k collateral

    cal = calibrations.get("HYPE", {})
    mle = cal.get("mle", cal.get("heuristic", {}))
    sigma_h = mle.get("sigma", 0.01)
    sigma_d = sigma_h * np.sqrt(24)
    lambda_h = mle.get("lambda", 0)
    lambda_d = lambda_h * 24

    gbm_by_ltv = []
    merton_by_ltv = []

    for ltv in ltvs:
        debt = base_coll * ltv
        coll = base_coll
        gs_val = liquip_score_gbm(coll, debt, sigma_d, days_forward=7)
        ms_val = liquip_score_merton(coll, debt, sigma_d,
                                     lambda_j=lambda_d,
                                     mu_j=mle.get("mu_j", -0.03),
                                     sigma_j=mle.get("sigma_j", 0.03),
                                     days_forward=7)
        gbm_by_ltv.append(gs_val * 100)
        merton_by_ltv.append(ms_val * 100)

    ax8.plot(ltvs * 100, gbm_by_ltv, "b-", linewidth=2, label="GBM")
    ax8.plot(ltvs * 100, merton_by_ltv, "r-", linewidth=2, label="Merton")
    ax8.fill_between(ltvs * 100, gbm_by_ltv, merton_by_ltv, alpha=0.2, color="red")
    ax8.axvline(86, color="gray", linestyle="--", alpha=0.7, label="Felix Vanilla LLTV (86%)")
    ax8.axvline(80, color="gray", linestyle=":", alpha=0.7, label="Typical LLTV (80%)")
    ax8.set_xlabel("Loan-to-Value Ratio (%)")
    ax8.set_ylabel("7-day Liquidation Probability (%)")
    ax8.set_title("Where Jump Risk Matters: Liquip Score vs LTV\n"
                  "(Simulated HYPE-collateralized positions)")
    ax8.legend(fontsize=9)
    ax8.grid(alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 9: Jump Events Timeline (HYPE)
    # ------------------------------------------------------------------
    ax9 = fig.add_subplot(gs[4, 1])
    if "HYPE" in candles_dict and "HYPE" in calibrations:
        df = candles_dict["HYPE"].copy()
        df["log_ret"] = np.log(df["c"] / df["c"].shift(1))
        rets = df["log_ret"].dropna()

        heur = calibrations["HYPE"]["heuristic"]
        jump_idx = heur.get("jump_indices", np.array([]))

        ax9.plot(df["t"].iloc[1:].values, rets.values, color="#90CAF9", linewidth=0.5, alpha=0.7)

        if len(jump_idx) > 0:
            jump_times = df["t"].iloc[jump_idx + 1].values
            jump_rets = rets.iloc[jump_idx].values
            ax9.scatter(jump_times, jump_rets, color="red", s=40, zorder=5,
                        edgecolors="darkred", linewidths=0.5, label=f"Detected jumps ({len(jump_idx)})")

        # 3-sigma bands
        sigma_h = heur["sigma"]
        ax9.axhline(3 * sigma_h, color="orange", linestyle="--", alpha=0.5, label=f"±3σ ({3*sigma_h:.4f})")
        ax9.axhline(-3 * sigma_h, color="orange", linestyle="--", alpha=0.5)
        ax9.set_xlabel("Date")
        ax9.set_ylabel("Hourly Log-Return")
        ax9.set_title("HYPE Returns with Detected Jump Events")
        ax9.legend(fontsize=9)
        ax9.grid(alpha=0.3)
        ax9.tick_params(axis="x", rotation=30)

    output_path = "/Users/johnbeecher/Desktop/research/jump_diffusion_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nDashboard saved: {output_path}")
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  JUMP-DIFFUSION LIQUIP SCORE — PRELIMINARY RESULTS")
    print("  Extending Anthias Liquip Score with Merton Jump-Diffusion")
    print("=" * 70)

    # ---- Step 1: Fetch candle data ----
    print("\n[1/5] Fetching 90-day hourly candle data...")
    candles_dict = {}
    for coin in ["HYPE", "BTC", "ETH"]:
        print(f"  Fetching {coin}...")
        df = fetch_candles(coin, "1h", 90)
        if len(df) > 0:
            candles_dict[coin] = df
            print(f"    {len(df)} candles: {df['t'].iloc[0]} → {df['t'].iloc[-1]}")
        time.sleep(1)

    # ---- Step 2: Compute return statistics ----
    print("\n[2/5] Computing return statistics & normality tests...")
    stats_dict = {}
    returns_dict = {}
    for coin, df in candles_dict.items():
        stats, returns = compute_return_statistics(df, coin)
        stats_dict[coin] = stats
        returns_dict[coin] = returns
        print(f"\n  {coin}:")
        print(f"    N={stats['n_obs']}, Ann.Vol={stats['annualized_vol']:.1%}")
        print(f"    Skewness={stats['skewness']:.3f}, Excess Kurtosis={stats['excess_kurtosis']:.1f}")
        print(f"    Beyond 3σ: {stats['pct_beyond_3sigma']:.2f}% (normal expects 0.27%)")
        print(f"    Jarque-Bera p-value: {stats['jarque_bera_pvalue']:.2e}")
        if stats["jarque_bera_pvalue"] < 0.01:
            print(f"    → REJECT normality at 1% level")

    # ---- Step 3: Calibrate Merton model ----
    print("\n[3/5] Calibrating Merton jump-diffusion model...")
    calibrations = {}
    lr_tests = {}

    for coin, returns in returns_dict.items():
        print(f"\n  {coin}:")
        dt = 1.0  # 1 hour

        # Stage 1: Heuristic
        print("    Stage 1: Heuristic calibration...")
        heur = heuristic_calibration(returns, dt)
        print(f"      σ={heur['sigma']:.6f}, λ={heur['lambda']:.5f}/hr")
        print(f"      μ_J={heur['mu_j']:.6f}, σ_J={heur['sigma_j']:.6f}")
        print(f"      Jumps detected: {heur['n_jumps_detected']}")
        if heur["n_jumps_detected"] > 0:
            print(f"      Jump returns: {heur['jump_returns']}")

        # Stage 2: MLE
        print("    Stage 2: MLE refinement...")
        mle = mle_calibration(returns, dt, heur)
        print(f"      σ={mle['sigma']:.6f}, λ={mle['lambda']:.5f}/hr")
        print(f"      μ_J={mle['mu_j']:.6f}, σ_J={mle['sigma_j']:.6f}")

        calibrations[coin] = {
            "heuristic": heur,
            "mle": mle,
            "n_jumps_detected": heur["n_jumps_detected"],
        }

        # Likelihood ratio test
        print("    Likelihood ratio test (GBM vs Merton)...")
        lr = likelihood_ratio_test(returns, dt, mle)
        lr_tests[coin] = lr
        print(f"      LL(GBM)={lr['ll_gbm']:.1f}, LL(Merton)={lr['ll_merton']:.1f}")
        print(f"      D={lr['test_statistic']:.2f}, p={lr['p_value']:.2e}")
        if lr["reject_gbm_at_5pct"]:
            print(f"      → REJECT GBM at 5% level — Merton is significantly better")

    # ---- Step 4: Fetch Felix positions ----
    print("\n[4/5] Fetching Felix CDP positions...")
    prices = fetch_prices()
    print(f"  Prices: " + ", ".join(f"{k}=${v:,.2f}" for k, v in prices.items()))

    positions_data = {}
    w3 = Web3(Web3.HTTPProvider(HYPEREVM_RPC))
    print(f"  Connected to HyperEVM | Block: {w3.eth.block_number}")

    for i, (branch_name, config) in enumerate(CDP_BRANCHES.items()):
        if i > 0:
            print(f"  Waiting 10s before next branch to avoid rate limit...")
            time.sleep(10)
        positions = fetch_cdp_positions(w3, branch_name, config)
        if positions:
            coin = config["price_coin"]
            price = prices.get(coin, 0)
            positions_data[branch_name] = {
                "positions": positions,
                "price": price,
                "coin": coin,
            }
        time.sleep(3)

    # ---- Step 5: Compute and compare Liquip Scores ----
    print("\n[5/5] Computing GBM vs Merton Liquip Scores...")
    for branch_name, bdata in positions_data.items():
        coin = bdata["coin"]
        price = bdata["price"]
        cal = calibrations.get(coin, {})
        mle = cal.get("mle", cal.get("heuristic", {}))
        sigma_h = mle.get("sigma", 0.01)
        sigma_d = sigma_h * np.sqrt(24)
        lambda_h = mle.get("lambda", 0)
        lambda_d = lambda_h * 24

        print(f"\n  {branch_name} ({coin} @ ${price:,.2f}):")

        for p in bdata["positions"]:
            coll_usd = p["collateral"] * price
            p["coll_usd"] = coll_usd
            p["icr"] = coll_usd / p["debt"] if p["debt"] > 0 else float("inf")

            # GBM scores
            p["gbm_7d"] = liquip_score_gbm(coll_usd, p["debt"], sigma_d, days_forward=7)
            p["gbm_30d"] = liquip_score_gbm(coll_usd, p["debt"], sigma_d, days_forward=30)

            # Merton scores
            p["merton_7d"] = liquip_score_merton(
                coll_usd, p["debt"], sigma_d,
                lambda_j=lambda_d, mu_j=mle.get("mu_j", -0.03),
                sigma_j=mle.get("sigma_j", 0.03), days_forward=7
            )
            p["merton_30d"] = liquip_score_merton(
                coll_usd, p["debt"], sigma_d,
                lambda_j=lambda_d, mu_j=mle.get("mu_j", -0.03),
                sigma_j=mle.get("sigma_j", 0.03), days_forward=30
            )

        # Summary
        n = len(bdata["positions"])
        gbm_max_30 = max(p["gbm_30d"] for p in bdata["positions"])
        merton_max_30 = max(p["merton_30d"] for p in bdata["positions"])
        print(f"    Positions: {n}")
        print(f"    Max GBM 30d:    {gbm_max_30:.6%}")
        print(f"    Max Merton 30d: {merton_max_30:.6%}")
        if gbm_max_30 > 0:
            print(f"    Merton/GBM ratio: {merton_max_30/gbm_max_30:.2f}x")

    # ---- Simulated High-Leverage Comparison ----
    print("\n" + "=" * 70)
    print("  SIMULATED HIGH-LEVERAGE COMPARISON (Morpho Blue-like)")
    print("=" * 70)
    cal = calibrations.get("HYPE", {})
    mle = cal.get("mle", cal.get("heuristic", {}))
    sigma_d = mle.get("sigma", 0.01) * np.sqrt(24)
    lambda_d = mle.get("lambda", 0) * 24

    print(f"\n  Using HYPE params: σ_daily={sigma_d:.4f}, λ_daily={lambda_d:.4f}")
    print(f"  μ_J={mle.get('mu_j', 0):.4f}, σ_J={mle.get('sigma_j', 0):.4f}")
    print(f"\n  {'LTV':>6s}  {'GBM 7d':>12s}  {'Merton 7d':>12s}  {'Ratio':>8s}  {'GBM 30d':>12s}  {'Merton 30d':>12s}  {'Ratio':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")

    for ltv in [0.60, 0.70, 0.75, 0.80, 0.83, 0.86, 0.90]:
        coll = 25000
        debt = coll * ltv
        g7 = liquip_score_gbm(coll, debt, sigma_d, days_forward=7)
        m7 = liquip_score_merton(coll, debt, sigma_d, lambda_j=lambda_d,
                                  mu_j=mle.get("mu_j", -0.03),
                                  sigma_j=mle.get("sigma_j", 0.03), days_forward=7)
        g30 = liquip_score_gbm(coll, debt, sigma_d, days_forward=30)
        m30 = liquip_score_merton(coll, debt, sigma_d, lambda_j=lambda_d,
                                   mu_j=mle.get("mu_j", -0.03),
                                   sigma_j=mle.get("sigma_j", 0.03), days_forward=30)
        r7 = m7 / g7 if g7 > 1e-10 else float("inf")
        r30 = m30 / g30 if g30 > 1e-10 else float("inf")
        print(f"  {ltv:5.0%}  {g7:12.4%}  {m7:12.4%}  {r7:7.2f}x  {g30:12.4%}  {m30:12.4%}  {r30:7.2f}x")

    # ---- Generate Dashboard ----
    print("\n--- Generating Dashboard ---")
    generate_dashboard(candles_dict, stats_dict, calibrations, positions_data,
                       lr_tests, prices)

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
