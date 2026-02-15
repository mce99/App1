"""Analytics helpers for dashboard KPIs and charts."""

from __future__ import annotations

import pandas as pd


def apply_currency_conversion(df: pd.DataFrame, conversion_rates: dict[str, float]) -> pd.DataFrame:
    """Convert debit and credit columns to CHF equivalents."""
    out = df.copy()
    out["DebitCHF"] = out.apply(
        lambda row: row["Debit"] * conversion_rates.get(str(row.get("Währung", "CHF")), float("nan")),
        axis=1,
    )
    out["CreditCHF"] = out.apply(
        lambda row: row["Credit"] * conversion_rates.get(str(row.get("Währung", "CHF")), float("nan")),
        axis=1,
    )
    return out


def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Filter transactions in inclusive date range."""
    mask = df["Date"].dt.date.between(start_date, end_date)
    return df.loc[mask].copy()


def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Return core portfolio-style KPI values."""
    total_spending = float(df["DebitCHF"].sum(skipna=True))
    total_earnings = float(df["CreditCHF"].sum(skipna=True))
    net_cashflow = total_earnings - total_spending
    tx_count = int(len(df))
    avg_spending = float(df["DebitCHF"].mean(skipna=True) if tx_count else 0.0)
    savings_rate = float((net_cashflow / total_earnings * 100.0) if total_earnings else 0.0)

    return {
        "total_spending": total_spending,
        "total_earnings": total_earnings,
        "net_cashflow": net_cashflow,
        "transactions": tx_count,
        "avg_spending": avg_spending,
        "savings_rate": savings_rate,
    }


def monthly_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate income/spending/net by calendar month."""
    out = df.copy()
    out["Month"] = out["Date"].dt.to_period("M").astype(str)
    summary = (
        out.groupby("Month", dropna=True)
        .agg(Spending=("DebitCHF", "sum"), Earnings=("CreditCHF", "sum"))
        .sort_index()
    )
    summary["Net"] = summary["Earnings"] - summary["Spending"]
    return summary


def daily_net_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """Daily net plus cumulative net trend."""
    daily = (
        df.groupby(df["Date"].dt.date, dropna=True)
        .agg(Spending=("DebitCHF", "sum"), Earnings=("CreditCHF", "sum"))
        .sort_index()
    )
    daily["Net"] = daily["Earnings"] - daily["Spending"]
    daily["CumulativeNet"] = daily["Net"].cumsum()
    daily.index = pd.to_datetime(daily.index)
    return daily


def merchant_summary(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top merchants by total spending."""
    grouped = (
        df.groupby("Merchant", dropna=True)["DebitCHF"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="SpendingCHF")
    )
    return grouped[grouped["Merchant"].astype(str).str.strip() != ""]


def recurring_transaction_candidates(df: pd.DataFrame, min_occurrences: int = 3) -> pd.DataFrame:
    """Detect likely recurring merchants based on date interval regularity."""
    working = df.copy()
    working["DateOnly"] = pd.to_datetime(working["Date"], errors="coerce").dt.normalize()
    working = working[working["DateOnly"].notna()]
    working = working[working["Merchant"].astype(str).str.strip() != ""]

    rows: list[dict[str, object]] = []
    for merchant, group in working.groupby("Merchant"):
        ordered = group.sort_values("DateOnly")
        if len(ordered) < min_occurrences:
            continue

        intervals = ordered["DateOnly"].diff().dt.days.dropna()
        if intervals.empty:
            continue

        median_days = float(intervals.median())
        if not (20 <= median_days <= 40):
            continue

        avg_spend = float(ordered["DebitCHF"].replace(0, pd.NA).dropna().mean() or 0.0)
        avg_earn = float(ordered["CreditCHF"].replace(0, pd.NA).dropna().mean() or 0.0)
        last_seen = ordered["DateOnly"].max()
        next_due = last_seen + pd.Timedelta(days=int(round(median_days)))
        confidence = 1 - min(abs(median_days - 30.0) / 20.0, 1.0)

        rows.append(
            {
                "Merchant": merchant,
                "Occurrences": int(len(ordered)),
                "CadenceDays": round(median_days, 1),
                "AvgSpendingCHF": round(avg_spend, 2),
                "AvgEarningsCHF": round(avg_earn, 2),
                "LastSeen": last_seen.date().isoformat(),
                "ExpectedNext": next_due.date().isoformat(),
                "SignalConfidence": round(max(0.0, min(confidence, 1.0)), 2),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        ["SignalConfidence", "Occurrences", "AvgSpendingCHF"],
        ascending=[False, False, False],
    )
    return out.reset_index(drop=True)


def budget_progress(df: pd.DataFrame, budget_by_category: dict[str, float]) -> pd.DataFrame:
    """Compare spending against user-provided category budgets."""
    actual = df.groupby("Category", dropna=True)["DebitCHF"].sum().rename("ActualCHF")
    budget = pd.Series(budget_by_category, name="BudgetCHF", dtype=float)
    out = pd.concat([actual, budget], axis=1).fillna(0.0)
    out["RemainingCHF"] = out["BudgetCHF"] - out["ActualCHF"]
    out["Status"] = out["RemainingCHF"].apply(lambda v: "On Track" if v >= 0 else "Over Budget")
    return out.sort_values("ActualCHF", ascending=False)
