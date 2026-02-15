"""Market data and portfolio helpers (stocks + crypto wallets)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import requests


YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
COINGECKO_SIMPLE_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
ETH_RPC_URL = "https://cloudflare-eth.com"
BTC_ADDRESS_URL = "https://blockstream.info/api/address/{address}"
SOL_RPC_URL = "https://api.mainnet-beta.solana.com"

COIN_ID_BY_SYMBOL = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}


def detect_wallet_chain(address: str) -> str:
    addr = str(address or "").strip()
    if re.fullmatch(r"0x[a-fA-F0-9]{40}", addr):
        return "ETH"
    if re.fullmatch(r"(bc1|[13])[a-zA-HJ-NP-Z0-9]{20,}", addr):
        return "BTC"
    if re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", addr):
        return "SOL"
    return "UNKNOWN"


def _safe_json_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {}


def fetch_stock_quotes(symbols: list[str], timeout: float = 8.0) -> pd.DataFrame:
    tickers = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not tickers:
        return pd.DataFrame(columns=["Symbol", "Name", "Price", "Currency"])

    try:
        response = requests.get(
            YAHOO_QUOTE_URL,
            params={"symbols": ",".join(tickers)},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = _safe_json_response(response)
        quotes = payload.get("quoteResponse", {}).get("result", [])
    except Exception:
        quotes = []

    rows = []
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).upper()
        if not symbol:
            continue
        rows.append(
            {
                "Symbol": symbol,
                "Name": quote.get("shortName") or quote.get("longName") or symbol,
                "Price": float(quote.get("regularMarketPrice") or 0.0),
                "Currency": quote.get("currency") or "",
            }
        )
    return pd.DataFrame(rows)


def evaluate_stock_positions(positions: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Quantity",
                "AvgBuyPrice",
                "CurrentPrice",
                "Currency",
                "CostBasis",
                "MarketValue",
                "UnrealizedPnL",
                "PnLPct",
            ]
        )

    work = positions.copy()
    work["Symbol"] = work["Symbol"].astype(str).str.upper().str.strip()
    work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce").fillna(0.0)
    work["AvgBuyPrice"] = pd.to_numeric(work["AvgBuyPrice"], errors="coerce").fillna(0.0)

    merged = work.merge(
        quotes.rename(columns={"Price": "CurrentPrice"}), on="Symbol", how="left"
    )
    merged["CurrentPrice"] = pd.to_numeric(merged["CurrentPrice"], errors="coerce").fillna(0.0)
    merged["Currency"] = merged.get("Currency", "").fillna("")
    merged["CostBasis"] = merged["Quantity"] * merged["AvgBuyPrice"]
    merged["MarketValue"] = merged["Quantity"] * merged["CurrentPrice"]
    merged["UnrealizedPnL"] = merged["MarketValue"] - merged["CostBasis"]
    merged["PnLPct"] = merged.apply(
        lambda row: (row["UnrealizedPnL"] / row["CostBasis"] * 100.0) if row["CostBasis"] else 0.0,
        axis=1,
    )
    return merged[
        [
            "Symbol",
            "Name",
            "Quantity",
            "AvgBuyPrice",
            "CurrentPrice",
            "Currency",
            "CostBasis",
            "MarketValue",
            "UnrealizedPnL",
            "PnLPct",
        ]
    ].sort_values("MarketValue", ascending=False)


def fetch_crypto_prices(symbols: list[str], vs_currency: str = "usd", timeout: float = 8.0) -> dict[str, float]:
    wanted = [str(s).upper().strip() for s in symbols if str(s).strip()]
    coin_ids = [COIN_ID_BY_SYMBOL[s] for s in wanted if s in COIN_ID_BY_SYMBOL]
    if not coin_ids:
        return {}
    try:
        response = requests.get(
            COINGECKO_SIMPLE_PRICE_URL,
            params={"ids": ",".join(sorted(set(coin_ids))), "vs_currencies": vs_currency},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = _safe_json_response(response)
    except Exception:
        return {}

    symbol_price = {}
    for symbol, coin_id in COIN_ID_BY_SYMBOL.items():
        if coin_id in payload and vs_currency in payload[coin_id]:
            symbol_price[symbol] = float(payload[coin_id][vs_currency])
    return symbol_price


def _fetch_eth_balance(address: str, timeout: float = 8.0) -> float:
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBalance",
        "params": [address, "latest"],
        "id": 1,
    }
    response = requests.post(ETH_RPC_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = _safe_json_response(response)
    result = data.get("result")
    if not isinstance(result, str):
        return 0.0
    return int(result, 16) / 1e18


def _fetch_btc_balance(address: str, timeout: float = 8.0) -> float:
    response = requests.get(BTC_ADDRESS_URL.format(address=address), timeout=timeout)
    response.raise_for_status()
    data = _safe_json_response(response)
    stats = data.get("chain_stats", {})
    funded = float(stats.get("funded_txo_sum") or 0.0)
    spent = float(stats.get("spent_txo_sum") or 0.0)
    return (funded - spent) / 1e8


def _fetch_sol_balance(address: str, timeout: float = 8.0) -> float:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [address]}
    response = requests.post(SOL_RPC_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = _safe_json_response(response)
    lamports = float(data.get("result", {}).get("value") or 0.0)
    return lamports / 1e9


def fetch_wallet_balances(wallets: pd.DataFrame, quote_currency: str = "usd") -> pd.DataFrame:
    if wallets.empty:
        return pd.DataFrame(
            columns=[
                "Label",
                "Address",
                "Chain",
                "Balance",
                "Asset",
                "Price",
                "Value",
                "QuoteCurrency",
                "Status",
            ]
        )

    rows = []
    for _, row in wallets.iterrows():
        label = str(row.get("Label", "")).strip() or "Wallet"
        address = str(row.get("Address", "")).strip()
        chain = str(row.get("Chain", "")).strip().upper() or detect_wallet_chain(address)
        asset = chain if chain in {"BTC", "ETH", "SOL"} else ""
        balance = 0.0
        status = "ok"

        if not address:
            status = "missing_address"
        else:
            try:
                if chain == "BTC":
                    balance = _fetch_btc_balance(address)
                elif chain == "ETH":
                    balance = _fetch_eth_balance(address)
                elif chain == "SOL":
                    balance = _fetch_sol_balance(address)
                else:
                    status = "unsupported_chain"
            except Exception:
                status = "fetch_error"

        rows.append(
            {
                "Label": label,
                "Address": address,
                "Chain": chain,
                "Balance": balance,
                "Asset": asset,
                "Status": status,
            }
        )

    out = pd.DataFrame(rows)
    prices = fetch_crypto_prices(out["Asset"].dropna().astype(str).tolist(), vs_currency=quote_currency)
    out["Price"] = out["Asset"].map(prices).fillna(0.0)
    out["Value"] = out["Balance"] * out["Price"]
    out["QuoteCurrency"] = quote_currency.upper()
    return out.sort_values("Value", ascending=False).reset_index(drop=True)


def portfolio_totals(stock_positions: pd.DataFrame, wallet_positions: pd.DataFrame) -> dict[str, float]:
    stock_value = float(stock_positions["MarketValue"].sum()) if not stock_positions.empty else 0.0
    stock_cost = float(stock_positions["CostBasis"].sum()) if not stock_positions.empty else 0.0
    stock_pnl = float(stock_positions["UnrealizedPnL"].sum()) if not stock_positions.empty else 0.0

    wallet_value = float(wallet_positions["Value"].sum()) if not wallet_positions.empty else 0.0
    total_value = stock_value + wallet_value
    total_cost = stock_cost
    total_pnl = stock_pnl + (wallet_value)  # wallet cost basis may be unknown, treated as zero cost.

    return {
        "stock_value": stock_value,
        "stock_cost": stock_cost,
        "stock_pnl": stock_pnl,
        "wallet_value": wallet_value,
        "total_value": total_value,
        "total_cost_known": total_cost,
        "total_pnl_known": total_pnl,
    }


def holdings_mix(stock_positions: pd.DataFrame, wallet_positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not stock_positions.empty:
        for _, row in stock_positions.iterrows():
            rows.append(
                {
                    "AssetType": "Stock",
                    "Label": row.get("Symbol", ""),
                    "Value": float(row.get("MarketValue", 0.0)),
                }
            )
    if not wallet_positions.empty:
        for _, row in wallet_positions.iterrows():
            rows.append(
                {
                    "AssetType": "CryptoWallet",
                    "Label": row.get("Label", ""),
                    "Value": float(row.get("Value", 0.0)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["AssetType", "Label", "Value"])
    return pd.DataFrame(rows).sort_values("Value", ascending=False)
