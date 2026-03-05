"""Lending protocol data — HyperLend (HyperEVM) and Morpho Blue (Ethereum).

Owner: Antonio Braz

Data sources:
  HyperLend  — Aave V3 fork on HyperEVM (chain ID 999)
    RPC: https://rpc.hyperliquid.xyz/evm  (free, 100 req/min)
    Pool contract: 0x00A89d7a5A02160f20150EbEA7a2b5E4879A1A8b
    Query: getUserAccountData(address) → health factor, collateral, debt
    Discovery: scan Borrow events to find active borrowers

  Morpho Blue — Ethereum mainnet
    Subgraph: The Graph (free tier, requires Graph API key signup)
    GraphQL: query positions with market, collateral, debt, LTV
    Explorer: https://thegraph.com/explorer/profile/0x84d3e4ee...

  DeFi Llama — aggregate TVL history (free, no auth)
    GET https://api.llama.fi/protocol/{protocol}

Unified schema for lending positions:
  [timestamp, protocol, chain, borrower, collateral_usd, debt_usd,
   health_factor, liquidation_threshold, collateral_token]
"""

from __future__ import annotations

import pandas as pd

# ── HyperLend (HyperEVM / Aave V3) ──

HYPEREVM_RPC = "https://rpc.hyperliquid.xyz/evm"
HYPEREVM_CHAIN_ID = 999

# HyperLend core pool addresses (Aave V3.0.2 fork)
HYPERLEND_POOL = "0x00A89d7a5A02160f20150EbEA7a2b5E4879A1A8b"
HYPERLEND_POOL_PROVIDER = "0x72c98246a98bFe64022a3190e7710E157497170C"

# Aave V3 Pool ABI fragment — only the methods we need
AAVE_V3_POOL_ABI_FRAGMENT = [
    {
        "name": "getUserAccountData",
        "type": "function",
        "inputs": [{"name": "user", "type": "address"}],
        "outputs": [
            {"name": "totalCollateralBase", "type": "uint256"},
            {"name": "totalDebtBase", "type": "uint256"},
            {"name": "availableBorrowsBase", "type": "uint256"},
            {"name": "currentLiquidationThreshold", "type": "uint256"},
            {"name": "ltv", "type": "uint256"},
            {"name": "healthFactor", "type": "uint256"},
        ],
    },
    {
        "name": "getReserveData",
        "type": "function",
        "inputs": [{"name": "asset", "type": "address"}],
        "outputs": [
            # ReserveData struct — we only need a few fields
            {"name": "configuration", "type": "uint256"},
            {"name": "liquidityIndex", "type": "uint128"},
            {"name": "currentLiquidityRate", "type": "uint128"},
            {"name": "variableBorrowIndex", "type": "uint128"},
            {"name": "currentVariableBorrowRate", "type": "uint128"},
            {"name": "currentStableBorrowRate", "type": "uint128"},
            {"name": "lastUpdateTimestamp", "type": "uint40"},
            {"name": "id", "type": "uint16"},
            {"name": "aTokenAddress", "type": "address"},
            {"name": "stableDebtTokenAddress", "type": "address"},
            {"name": "variableDebtTokenAddress", "type": "address"},
            {"name": "interestRateStrategyAddress", "type": "address"},
            {"name": "accruedToTreasury", "type": "uint128"},
            {"name": "unbacked", "type": "uint128"},
            {"name": "isolationModeTotalDebt", "type": "uint128"},
        ],
    },
]


def discover_hyperlend_borrowers(
    from_block: int = 0,
    max_borrowers: int = 500,
) -> list[str]:
    """Scan HyperLend Borrow events to find active borrower addresses.

    Uses eth_getLogs on the HyperEVM RPC to find Borrow(address,address,...)
    events from the Pool contract.
    """
    raise NotImplementedError


def fetch_hyperlend_positions(
    borrowers: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch health factors and position data for HyperLend borrowers.

    Calls Pool.getUserAccountData(address) for each borrower via HyperEVM RPC.

    Returns DataFrame with columns:
      [timestamp, protocol, chain, borrower, collateral_usd, debt_usd,
       health_factor, liquidation_threshold, ltv]

    If borrowers is None, calls discover_hyperlend_borrowers() first.
    """
    raise NotImplementedError


def fetch_hyperlend_reserves() -> pd.DataFrame:
    """Fetch reserve-level data (utilization, rates, TVL) from HyperLend.

    Returns DataFrame with columns:
      [timestamp, asset, total_supplied_usd, total_borrowed_usd,
       utilization, supply_rate, borrow_rate]
    """
    raise NotImplementedError


# ── Morpho Blue (Ethereum via The Graph subgraph) ──

MORPHO_SUBGRAPH_URL = (
    "https://gateway.thegraph.com/api/subgraphs/id/"
    # The subgraph ID — fill in after creating a Graph API key
    "FILL_IN_SUBGRAPH_ID"
)

# Example GraphQL query for borrower positions
MORPHO_POSITIONS_QUERY = """
query GetPositions($first: Int!, $skip: Int!) {
  positions(
    first: $first,
    skip: $skip,
    where: { side: "BORROWER" }
  ) {
    id
    account { id }
    market {
      id
      inputToken { symbol decimals }
      liquidationThreshold
      liquidationPenalty
    }
    side
    balance
    isCollateral
  }
}
"""


def fetch_morpho_positions(
    min_debt_usd: float = 1000.0,
    max_positions: int = 1000,
) -> pd.DataFrame:
    """Fetch Morpho Blue borrower positions via The Graph subgraph.

    Requires GRAPH_API_KEY environment variable (free to create at
    https://thegraph.com/studio/).

    Returns DataFrame with columns:
      [timestamp, protocol, chain, borrower, market_id, collateral_usd,
       debt_usd, health_factor, liquidation_threshold, collateral_token]
    """
    raise NotImplementedError


# ── DeFi Llama (aggregate TVL) ──

DEFILLAMA_API = "https://api.llama.fi"


def fetch_tvl_history(protocol: str) -> pd.DataFrame:
    """Fetch historical TVL from DeFi Llama.

    GET https://api.llama.fi/protocol/{protocol}
    Free, no auth.

    protocol: "hyperlend", "morpho-blue", etc.

    Returns DataFrame with columns: [timestamp, protocol, tvl_usd]
    """
    raise NotImplementedError


# ── Unified lending interface ──

def fetch_all_lending_positions() -> pd.DataFrame:
    """Fetch positions from all lending protocols and unify schema.

    Returns DataFrame with columns:
      [timestamp, protocol, chain, borrower, collateral_usd, debt_usd,
       health_factor, liquidation_threshold, collateral_token]
    """
    raise NotImplementedError
