"""Lending protocol data — HyperLend (HyperEVM) historical event replay.

Owner: Antonio Braz (original), John Beecher (event replay)

Reconstructs historical lending positions by scanning on-chain events
(Supply, Withdraw, Borrow, Repay, LiquidationCall) and replaying them
to build per-borrower position timeseries with health factors.

Data sources:
  HyperLend — Aave V3 fork on HyperEVM (chain ID 999)
    RPC: https://rpc.hyperliquid.xyz/evm  (free, 100 req/min)
    Pool contract: 0x00A89d7a5A02160f20150EbEA7a2b5E4879A1A8b

  DeFi Llama — aggregate TVL history (free, no auth)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import polars as pl
import requests

from funding_the_fall.data.storage import DATA_DIR
from funding_the_fall.data.fetchers import _fetch_prices_0xa

logger = logging.getLogger(__name__)

# ── HyperLend (HyperEVM / Aave V3) ──

HYPEREVM_RPC = "https://rpc.hyperliquid.xyz/evm"
HYPEREVM_CHAIN_ID = 999

HYPERLEND_POOL = "0x00A89d7a5A02160f20150EbEA7a2b5E4879A1A8b"

# ── Event topic hashes (keccak256) ──

EVENT_TOPICS = {
    "Supply": "0x2b627736bca15cd5381dcf80b0bf11fd197d01a037c52b927a881a10fb73ba61",
    "Withdraw": "0x3115d1449a7b732c986cba18244e897a450f61e1bb8d589cd2e69e6c8924f9f7",
    "Borrow": "0xb3d084820fb1a9decffb176436bd02558d15fac9b0ddfed8c465bc7359d7dce0",
    "Repay": "0xa534c8dbe71f871f9f3530e97a74601fea17b426cae02e1c5aee42c96c784051",
    "LiquidationCall": "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286",
}

# Reverse lookup: topic0 → event name
TOPIC_TO_EVENT = {v: k for k, v in EVENT_TOPICS.items()}

# All topic0s for a single eth_getLogs filter
ALL_EVENT_TOPICS = list(EVENT_TOPICS.values())

# ── Reserve tokens on HyperLend ──
# Addresses discovered from on-chain event topics (reserve = topic1).
# Decimals and liquidation thresholds from getReserveData / Aave docs.

RESERVES: dict[str, dict[str, Any]] = {
    # Full 20-byte addresses (40 hex chars) matching _addr_from_topic output.
    "0x000000000d01dc56dcaaca66ad901c959b4011ec": {
        "symbol": "USDC",
        "decimals": 6,
        "liq_threshold": 0.85,
    },
    "0x00000000b8ce59fc3717ada4c02eadf4e3bc5c75": {
        "symbol": "USDT",
        "decimals": 6,
        "liq_threshold": 0.80,
    },
    "0x000000002e4df919ac903b3d57a97a1bcc6bfad3": {
        "symbol": "WETH",
        "decimals": 18,
        "liq_threshold": 0.825,
    },
}

# Placeholder — will be populated from first event scan if needed.
# Full 20-byte addresses are extracted from topic1 in log entries.
RESERVE_ADDRESSES: set[str] = set()

# getUserAccountData(address) selector
GET_USER_ACCOUNT_DATA_SEL = "0xbf92857c"

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
]


# ── RPC helpers ──


def _rpc_call(method: str, params: list[Any], retries: int = 5) -> Any:
    """Send a JSON-RPC call to HyperEVM with retry on rate limit."""
    for attempt in range(retries):
        resp = requests.post(
            HYPEREVM_RPC,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            if "rate limited" in str(result["error"]).lower() and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"RPC error: {result['error']}")
        return result["result"]


def _encode_call(fn_selector: str, *args_hex: str) -> str:
    """Encode a simple ABI call: 4-byte selector + 32-byte padded args."""
    data = fn_selector
    for arg in args_hex:
        data += arg.lower().removeprefix("0x").zfill(64)
    return data


def _get_block_timestamp(block_num: int, cache: dict[int, int]) -> int:
    """Get block timestamp (unix seconds), using cache to avoid redundant calls."""
    if block_num in cache:
        return cache[block_num]
    block = _rpc_call("eth_getBlockByNumber", [hex(block_num), False])
    ts = int(block["timestamp"], 16)
    cache[block_num] = ts
    return ts


# ── Event log decoding ──


def _addr_from_topic(topic: str) -> str:
    """Extract a 20-byte address from a 32-byte hex topic."""
    return "0x" + topic[-40:].lower()


def _uint256_from_hex(hex_str: str, offset: int = 0) -> int:
    """Read a uint256 from hex data at a 32-byte word offset."""
    start = offset * 64
    return int(hex_str[start : start + 64], 16)


def _decode_log(log: dict) -> dict | None:
    """Decode a single event log into a flat dict.

    Returns None if the log can't be parsed.

    Aave V3 event layouts (indexed topics + data):
      Supply(reserve[1], user[2], onBehalfOf[3], amount, referralCode)
      Withdraw(reserve[1], user[2], to[3], amount)
      Borrow(reserve[1], onBehalfOf[2], user[3], amount, interestRateMode, borrowRate, referralCode)
      Repay(reserve[1], user[2], repayer[3], amount, useATokens)
      LiquidationCall(collateralAsset[1], debtAsset[2], user[3], debtToCover, liquidatedCollateralAmount, liquidator, receiveAToken)
    """
    topics = log.get("topics", [])
    if len(topics) < 3:
        return None

    topic0 = topics[0].lower()
    event_type = TOPIC_TO_EVENT.get(topic0)
    if event_type is None:
        return None

    data_hex = log.get("data", "0x").removeprefix("0x")
    block_num = int(log["blockNumber"], 16)

    if event_type == "Supply":
        # topics: [sig, reserve, user, onBehalfOf]
        # data: [amount (uint256), referralCode (uint16)]
        if len(topics) < 4 or len(data_hex) < 64:
            return None
        return {
            "block_number": block_num,
            "event_type": event_type,
            "reserve": _addr_from_topic(topics[1]),
            "user": _addr_from_topic(topics[3]),  # onBehalfOf = actual depositor
            "amount": _uint256_from_hex(data_hex, 0),
        }

    elif event_type == "Withdraw":
        # topics: [sig, reserve, user, to]
        # data: [amount]
        if len(topics) < 4 or len(data_hex) < 64:
            return None
        return {
            "block_number": block_num,
            "event_type": event_type,
            "reserve": _addr_from_topic(topics[1]),
            "user": _addr_from_topic(topics[2]),  # user who withdrew
            "amount": _uint256_from_hex(data_hex, 0),
        }

    elif event_type == "Borrow":
        # topics: [sig, reserve, onBehalfOf, user(referrer)]
        # data: [amount, interestRateMode, borrowRate, referralCode]
        if len(topics) < 3 or len(data_hex) < 64:
            return None
        return {
            "block_number": block_num,
            "event_type": event_type,
            "reserve": _addr_from_topic(topics[1]),
            "user": _addr_from_topic(topics[2]),  # onBehalfOf = borrower
            "amount": _uint256_from_hex(data_hex, 0),
        }

    elif event_type == "Repay":
        # topics: [sig, reserve, user, repayer]
        # data: [amount, useATokens]
        if len(topics) < 3 or len(data_hex) < 64:
            return None
        return {
            "block_number": block_num,
            "event_type": event_type,
            "reserve": _addr_from_topic(topics[1]),
            "user": _addr_from_topic(topics[2]),  # user whose debt is repaid
            "amount": _uint256_from_hex(data_hex, 0),
        }

    elif event_type == "LiquidationCall":
        # topics: [sig, collateralAsset, debtAsset, user]
        # data: [debtToCover, liquidatedCollateralAmount, liquidator, receiveAToken]
        if len(topics) < 4 or len(data_hex) < 128:
            return None
        return {
            "block_number": block_num,
            "event_type": event_type,
            "reserve": _addr_from_topic(topics[2]),  # debt asset
            "collateral_asset": _addr_from_topic(topics[1]),
            "user": _addr_from_topic(topics[3]),  # liquidated borrower
            "amount": _uint256_from_hex(data_hex, 0),  # debtToCover
            "collateral_amount": _uint256_from_hex(
                data_hex, 1
            ),  # liquidatedCollateralAmount
        }

    return None


# ── Event scanning ──


def scan_hyperlend_events(
    from_block: int = 0,
    to_block: int | None = None,
    chunk_size: int = 999,
    sleep_between: float = 1.5,
) -> pl.DataFrame:
    """Scan HyperLend pool events and return decoded event log DataFrame.

    Scans Supply, Withdraw, Borrow, Repay, LiquidationCall events from the
    pool contract. Saves results to data/lending_events.parquet incrementally.

    Returns polars DataFrame:
      [block_number, timestamp, event_type, reserve, user, amount,
       collateral_asset, collateral_amount]
    """
    if to_block is None:
        latest_hex = _rpc_call("eth_blockNumber", [])
        to_block = int(latest_hex, 16)

    # Resume from previously saved events if available
    events_path = DATA_DIR / "lending_events.parquet"
    existing_rows: list[dict] = []
    if from_block == 0 and events_path.exists():
        existing = pl.read_parquet(events_path)
        if not existing.is_empty():
            from_block = existing["block_number"].max() + 1
            existing_rows = existing.to_dicts()
            logger.info(
                f"Resuming scan from block {from_block} ({len(existing_rows)} existing events)"
            )

    rows: list[dict] = []
    block_ts_cache: dict[int, int] = {}
    cursor = from_block
    total_chunks = (to_block - from_block) // chunk_size + 1

    logger.info(f"Scanning blocks {from_block}..{to_block} ({total_chunks} chunks)")

    chunk_idx = 0
    while cursor <= to_block:
        end = min(cursor + chunk_size - 1, to_block)
        try:
            logs = _rpc_call(
                "eth_getLogs",
                [
                    {
                        "address": HYPERLEND_POOL,
                        "topics": [ALL_EVENT_TOPICS],
                        "fromBlock": hex(cursor),
                        "toBlock": hex(end),
                    }
                ],
            )
        except RuntimeError as e:
            if "rate limited" in str(e).lower():
                logger.warning(f"Rate limited at block {cursor}, backing off...")
                time.sleep(10)
                continue
            raise

        for log in logs:
            decoded = _decode_log(log)
            if decoded is not None:
                rows.append(decoded)

        cursor = end + 1
        chunk_idx += 1
        if chunk_idx % 100 == 0:
            logger.info(
                f"  chunk {chunk_idx}/{total_chunks}, {len(rows)} events so far"
            )
        time.sleep(sleep_between)

    # Collect unique reserve addresses from decoded events
    RESERVE_ADDRESSES.update(r["reserve"] for r in rows if r.get("reserve"))

    if not rows and not existing_rows:
        return pl.DataFrame(
            schema={
                "block_number": pl.Int64,
                "timestamp": pl.Datetime("us", "UTC"),
                "event_type": pl.Utf8,
                "reserve": pl.Utf8,
                "user": pl.Utf8,
                "amount": pl.Int64,
                "collateral_asset": pl.Utf8,
                "collateral_amount": pl.Int64,
            }
        )

    # Resolve block timestamps for new rows
    unique_blocks = {r["block_number"] for r in rows}
    for blk in unique_blocks:
        _get_block_timestamp(blk, block_ts_cache)

    for row in rows:
        row["timestamp"] = block_ts_cache.get(row["block_number"], 0)
        # Fill missing fields for non-liquidation events
        row.setdefault("collateral_asset", None)
        row.setdefault("collateral_amount", None)

    all_rows = existing_rows + rows
    df = pl.DataFrame(all_rows)

    # Cast timestamp from unix seconds to datetime
    if "timestamp" in df.columns and df.schema["timestamp"] != pl.Datetime:
        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp")
        )

    df = df.sort("block_number")

    # Persist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(events_path)
    logger.info(f"Saved {len(df)} events to {events_path}")

    return df


def discover_reserves(
    block_range: int = 5000,
) -> set[str]:
    """Scan a small recent block range to discover unique reserve addresses.

    Useful for bootstrapping the RESERVES dict with full 20-byte addresses.
    """
    latest_hex = _rpc_call("eth_blockNumber", [])
    latest_block = int(latest_hex, 16)
    from_block = max(0, latest_block - block_range)

    reserves: set[str] = set()
    chunk_size = 999
    cursor = from_block

    while cursor <= latest_block:
        end = min(cursor + chunk_size - 1, latest_block)
        try:
            logs = _rpc_call(
                "eth_getLogs",
                [
                    {
                        "address": HYPERLEND_POOL,
                        "topics": [ALL_EVENT_TOPICS],
                        "fromBlock": hex(cursor),
                        "toBlock": hex(end),
                    }
                ],
            )
        except RuntimeError:
            cursor = end + 1
            continue
        for log in logs:
            topics = log.get("topics", [])
            if len(topics) >= 2:
                reserves.add(_addr_from_topic(topics[1]))
        cursor = end + 1
        time.sleep(1.5)

    return reserves


# ── Position replay ──


def replay_positions(
    events_df: pl.DataFrame,
    prices_df: pl.DataFrame,
    snapshot_interval_hours: int = 8,
) -> pl.DataFrame:
    """Replay events to reconstruct historical borrower positions.

    Args:
        events_df: Output of scan_hyperlend_events().
        prices_df: DataFrame with [timestamp, reserve, price_usd] for each
            reserve token at regular intervals.
        snapshot_interval_hours: How often to snapshot positions (default 8h
            to align with funding epochs).

    Returns polars DataFrame:
      [timestamp, borrower, collateral_usd, debt_usd, health_factor,
       liquidation_threshold]
    """
    if events_df.is_empty():
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "borrower": pl.Utf8,
                "collateral_usd": pl.Float64,
                "debt_usd": pl.Float64,
                "health_factor": pl.Float64,
                "liquidation_threshold": pl.Float64,
            }
        )

    events = events_df.sort("block_number").to_dicts()

    # Running state: (user, reserve) → {collateral: int, debt: int}
    balances: dict[tuple[str, str], dict[str, int]] = {}

    def _get(user: str, reserve: str) -> dict[str, int]:
        key = (user, reserve)
        if key not in balances:
            balances[key] = {"collateral": 0, "debt": 0}
        return balances[key]

    # Process events sequentially
    for ev in events:
        user = ev["user"]
        reserve = ev["reserve"]
        amount = ev["amount"]
        bal = _get(user, reserve)

        if ev["event_type"] == "Supply":
            bal["collateral"] += amount
        elif ev["event_type"] == "Withdraw":
            bal["collateral"] = max(0, bal["collateral"] - amount)
        elif ev["event_type"] == "Borrow":
            bal["debt"] += amount
        elif ev["event_type"] == "Repay":
            bal["debt"] = max(0, bal["debt"] - amount)
        elif ev["event_type"] == "LiquidationCall":
            bal["debt"] = max(0, bal["debt"] - amount)
            # Also reduce collateral on the collateral asset
            coll_asset = ev.get("collateral_asset")
            coll_amount = ev.get("collateral_amount", 0)
            if coll_asset and coll_amount:
                coll_bal = _get(user, coll_asset)
                coll_bal["collateral"] = max(0, coll_bal["collateral"] - coll_amount)

    # Build snapshot timestamps from the event time range
    ts_min = events_df["timestamp"].min()
    ts_max = events_df["timestamp"].max()
    interval = f"{snapshot_interval_hours}h"
    snap_times = pl.datetime_range(ts_min, ts_max, interval, eager=True)

    # For each snapshot, replay events up to that point and merge with prices.
    # More efficient: replay once (above gives final state). For time-series,
    # we re-replay incrementally.
    balances.clear()
    ev_idx = 0
    snapshot_rows: list[dict] = []

    for snap_t in snap_times:
        # Advance events up to snap_t
        while ev_idx < len(events) and events[ev_idx]["timestamp"] <= snap_t:
            ev = events[ev_idx]
            user = ev["user"]
            reserve = ev["reserve"]
            amount = ev["amount"]
            bal = _get(user, reserve)

            if ev["event_type"] == "Supply":
                bal["collateral"] += amount
            elif ev["event_type"] == "Withdraw":
                bal["collateral"] = max(0, bal["collateral"] - amount)
            elif ev["event_type"] == "Borrow":
                bal["debt"] += amount
            elif ev["event_type"] == "Repay":
                bal["debt"] = max(0, bal["debt"] - amount)
            elif ev["event_type"] == "LiquidationCall":
                bal["debt"] = max(0, bal["debt"] - amount)
                coll_asset = ev.get("collateral_asset")
                coll_amount = ev.get("collateral_amount", 0)
                if coll_asset and coll_amount:
                    coll_bal = _get(user, coll_asset)
                    coll_bal["collateral"] = max(
                        0, coll_bal["collateral"] - coll_amount
                    )
            ev_idx += 1

        # Get prices at this snapshot (nearest available)
        prices_at_snap = _get_prices_at(prices_df, snap_t)

        # Aggregate per borrower
        borrower_totals: dict[str, dict[str, float]] = {}
        for (user, reserve), bal in balances.items():
            if bal["collateral"] == 0 and bal["debt"] == 0:
                continue
            price = prices_at_snap.get(reserve, 0.0)
            decimals = _get_reserve_decimals(reserve)
            scale = 10**decimals

            if user not in borrower_totals:
                borrower_totals[user] = {"collateral_usd": 0.0, "debt_usd": 0.0}
            borrower_totals[user]["collateral_usd"] += (
                bal["collateral"] / scale
            ) * price
            borrower_totals[user]["debt_usd"] += (bal["debt"] / scale) * price

        for borrower, totals in borrower_totals.items():
            if totals["debt_usd"] < 10:
                continue
            liq_thresh = 0.825  # weighted average, approximation
            hf = (
                (totals["collateral_usd"] * liq_thresh) / totals["debt_usd"]
                if totals["debt_usd"] > 0
                else float("inf")
            )
            snapshot_rows.append(
                {
                    "timestamp": snap_t,
                    "borrower": borrower,
                    "collateral_usd": totals["collateral_usd"],
                    "debt_usd": totals["debt_usd"],
                    "health_factor": hf,
                    "liquidation_threshold": liq_thresh,
                }
            )

    if not snapshot_rows:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "borrower": pl.Utf8,
                "collateral_usd": pl.Float64,
                "debt_usd": pl.Float64,
                "health_factor": pl.Float64,
                "liquidation_threshold": pl.Float64,
            }
        )

    result = pl.DataFrame(snapshot_rows)

    # Persist
    out_path = DATA_DIR / "lending_history.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_path)
    logger.info(f"Saved {len(result)} position snapshots to {out_path}")

    return result


def _get_prices_at(
    prices_df: pl.DataFrame,
    snap_t: Any,
) -> dict[str, float]:
    """Get reserve→price_usd mapping at the nearest timestamp."""
    if prices_df.is_empty():
        return {}
    # Find closest timestamp <= snap_t per reserve
    filtered = prices_df.filter(pl.col("timestamp") <= snap_t)
    if filtered.is_empty():
        # Fall back to earliest available prices
        filtered = prices_df
    latest = filtered.group_by("reserve").agg(pl.all().sort_by("timestamp").last())
    return dict(
        zip(
            latest["reserve"].to_list(),
            latest["price_usd"].to_list(),
        )
    )


def _get_reserve_decimals(reserve: str) -> int:
    """Look up token decimals for a reserve address."""
    info = RESERVES.get(reserve)
    if info:
        return info["decimals"]
    # Default to 18 for unknown tokens (ERC-20 standard)
    return 18


# ── Reserve price history (for position replay) ──


def fetch_reserve_prices(days: int = 90) -> pl.DataFrame:
    """Fetch historical prices for lending reserves via 0xArchive.

    ETH price from Hyperliquid mark prices; USDC/USDT pegged at 1.0.
    Returns polars DataFrame: [timestamp, reserve, price_usd]
    """
    # ETH prices from 0xArchive
    eth_prices = _fetch_prices_0xa("ETH", "hyperliquid", days)
    frames: list[pl.DataFrame] = []

    if not eth_prices.is_empty():
        eth_df = eth_prices.with_columns(
            pl.lit("0x000000002e4df919ac903b3d57a97a1bcc6bfad3").alias("reserve"),
            pl.col("mark_price").alias("price_usd"),
        ).select("timestamp", "reserve", "price_usd")
        frames.append(eth_df)

    # Stablecoins at $1.0 — create matching timestamps from ETH data
    if not eth_prices.is_empty():
        ts_col = eth_prices.select("timestamp")
        for addr, sym in [
            ("0x000000000d01dc56dcaaca66ad901c959b4011ec", "USDC"),
            ("0x00000000b8ce59fc3717ada4c02eadf4e3bc5c75", "USDT"),
        ]:
            stable = ts_col.with_columns(
                pl.lit(addr).alias("reserve"),
                pl.lit(1.0).alias("price_usd"),
            )
            frames.append(stable)

    if not frames:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "reserve": pl.Utf8,
                "price_usd": pl.Float64,
            }
        )

    result = pl.concat(frames).sort("timestamp")

    # Persist
    out_path = DATA_DIR / "reserve_prices.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_path)
    logger.info(f"Saved {len(result)} reserve price rows to {out_path}")

    return result


# ── Current snapshot (unchanged from original) ──

KNOWN_BORROWERS: list[str] = [
    "0x9730299b10a30bdbaf81815c99b4657c685314ab",
    "0x736dde3e0f5c588ddc53ad7f0f65667c0cca2801",
    "0x3c43014aedbb496e173059f94bfe3b0ed1d55ba0",
    "0x3e4db2489cdc102db0da07a7ced836874d5fac35",
    "0xb9da98cae931ef5deacbadef63cb69223588f3a5",
]


def discover_hyperlend_borrowers(
    from_block: int = 0,
    max_borrowers: int = 500,
) -> list[str]:
    """Scan HyperLend Borrow events to find active borrower addresses."""
    latest_hex = _rpc_call("eth_blockNumber", [])
    latest_block = int(latest_hex, 16)

    chunk_size = 999
    start = max(from_block, latest_block - 10_000)
    borrowers: set[str] = set()

    while start <= latest_block and len(borrowers) < max_borrowers:
        end = min(start + chunk_size - 1, latest_block)
        try:
            logs = _rpc_call(
                "eth_getLogs",
                [
                    {
                        "address": HYPERLEND_POOL,
                        "topics": [EVENT_TOPICS["Borrow"]],
                        "fromBlock": hex(start),
                        "toBlock": hex(end),
                    }
                ],
            )
        except RuntimeError as e:
            if "rate limited" in str(e).lower():
                time.sleep(2)
                continue
            raise
        for log in logs:
            if len(log.get("topics", [])) >= 3:
                addr = "0x" + log["topics"][2][-40:]
                borrowers.add(addr.lower())
        start = end + 1
        time.sleep(1.5)

    return list(borrowers)[:max_borrowers]


def fetch_hyperlend_positions(
    borrowers: list[str] | None = None,
) -> pl.DataFrame:
    """Fetch current health factors for HyperLend borrowers via RPC snapshot.

    Returns polars DataFrame:
      [timestamp, protocol, chain, borrower, collateral_usd, debt_usd,
       health_factor, liquidation_threshold, ltv]
    """
    if borrowers is None:
        try:
            borrowers = discover_hyperlend_borrowers()
        except Exception:
            borrowers = KNOWN_BORROWERS

    rows: list[dict] = []
    now = pl.Series([0]).cast(pl.Datetime("us", "UTC"))[0]  # placeholder
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)

    for addr in borrowers:
        calldata = _encode_call(GET_USER_ACCOUNT_DATA_SEL, addr)
        try:
            raw = _rpc_call(
                "eth_call",
                [
                    {"to": HYPERLEND_POOL, "data": calldata},
                    "latest",
                ],
            )
        except Exception:
            continue

        hex_str = raw.removeprefix("0x")
        if len(hex_str) < 6 * 64:
            continue

        total_collateral = int(hex_str[0:64], 16) / 1e8
        total_debt = int(hex_str[64:128], 16) / 1e8
        liq_threshold = int(hex_str[192:256], 16) / 1e4
        ltv = int(hex_str[256:320], 16) / 1e4
        health_factor = int(hex_str[320:384], 16) / 1e18

        if total_debt < 10:
            continue

        rows.append(
            {
                "timestamp": now,
                "protocol": "hyperlend",
                "chain": "hyperevm",
                "borrower": addr,
                "collateral_usd": total_collateral,
                "debt_usd": total_debt,
                "health_factor": health_factor,
                "liquidation_threshold": liq_threshold,
                "ltv": ltv,
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "protocol": pl.Utf8,
                "chain": pl.Utf8,
                "borrower": pl.Utf8,
                "collateral_usd": pl.Float64,
                "debt_usd": pl.Float64,
                "health_factor": pl.Float64,
                "liquidation_threshold": pl.Float64,
                "ltv": pl.Float64,
            }
        )
    return pl.DataFrame(rows)


# ── DeFi Llama (aggregate TVL) ──

DEFILLAMA_API = "https://api.llama.fi"


def fetch_tvl_history(protocol: str) -> pl.DataFrame:
    """Fetch historical TVL from DeFi Llama.

    Returns polars DataFrame: [timestamp, protocol, tvl_usd]
    """
    resp = requests.get(f"{DEFILLAMA_API}/protocol/{protocol}", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    tvl_series = data.get("tvl", [])
    if not tvl_series:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "protocol": pl.Utf8,
                "tvl_usd": pl.Float64,
            }
        )

    df = pl.DataFrame(tvl_series)
    df = df.with_columns(
        pl.from_epoch(pl.col("date"), time_unit="s").alias("timestamp"),
        pl.col("totalLiquidityUSD").cast(pl.Float64).alias("tvl_usd"),
        pl.lit(protocol).alias("protocol"),
    )
    return df.select("timestamp", "protocol", "tvl_usd")


# ── Unified interface ──


def fetch_all_lending_positions() -> pl.DataFrame:
    """Fetch current positions from all lending protocols (HyperLend only).

    Returns polars DataFrame:
      [timestamp, protocol, chain, borrower, collateral_usd, debt_usd,
       health_factor, liquidation_threshold, ltv]
    """
    try:
        return fetch_hyperlend_positions()
    except Exception:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "protocol": pl.Utf8,
                "chain": pl.Utf8,
                "borrower": pl.Utf8,
                "collateral_usd": pl.Float64,
                "debt_usd": pl.Float64,
                "health_factor": pl.Float64,
                "liquidation_threshold": pl.Float64,
                "ltv": pl.Float64,
            }
        )
