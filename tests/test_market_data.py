import pandas as pd

from market_data import (
    detect_wallet_chain,
    evaluate_stock_positions,
    holdings_mix,
    portfolio_totals,
)


def test_detect_wallet_chain_supports_btc_eth_sol() -> None:
    assert detect_wallet_chain("0x742d35Cc6634C0532925a3b844Bc454e4438f44e") == "ETH"
    assert detect_wallet_chain("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kg3g4ty") == "BTC"
    assert detect_wallet_chain("4Nd1mjqY5fW5q9Lw9x5fD4x2x2i8K8vQ8LQv2G8sL2YV") == "SOL"


def test_evaluate_stock_positions_calculates_pnl() -> None:
    positions = pd.DataFrame(
        [
            {"Symbol": "AAPL", "Quantity": 2, "AvgBuyPrice": 100},
            {"Symbol": "MSFT", "Quantity": 1, "AvgBuyPrice": 200},
        ]
    )
    quotes = pd.DataFrame(
        [
            {"Symbol": "AAPL", "Name": "Apple", "Price": 150, "Currency": "USD"},
            {"Symbol": "MSFT", "Name": "Microsoft", "Price": 180, "Currency": "USD"},
        ]
    )

    out = evaluate_stock_positions(positions, quotes)
    assert round(float(out["MarketValue"].sum()), 2) == 480.0
    assert round(float(out["CostBasis"].sum()), 2) == 400.0
    assert round(float(out["UnrealizedPnL"].sum()), 2) == 80.0


def test_portfolio_totals_and_holdings_mix() -> None:
    stocks = pd.DataFrame(
        [{"Symbol": "AAPL", "MarketValue": 300.0, "CostBasis": 200.0, "UnrealizedPnL": 100.0}]
    )
    wallets = pd.DataFrame([{"Label": "Main BTC", "Value": 250.0}])

    totals = portfolio_totals(stocks, wallets)
    mix = holdings_mix(stocks, wallets)

    assert totals["total_value"] == 550.0
    assert totals["stock_pnl"] == 100.0
    assert len(mix) == 2
