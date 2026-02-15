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
    avg_earning = float(df["CreditCHF"].mean(skipna=True) if tx_count else 0.0)
    savings_rate = float((net_cashflow / total_earnings * 100.0) if total_earnings else 0.0)
    active_days = int(df["Date"].dropna().dt.date.nunique())
    if active_days:
        avg_spending_per_active_day = float(total_spending / active_days)
        avg_earnings_per_active_day = float(total_earnings / active_days)
        avg_transactions_per_active_day = float(tx_count / active_days)
        avg_daily_net = float(net_cashflow / active_days)
    else:
        avg_spending_per_active_day = 0.0
        avg_earnings_per_active_day = 0.0
        avg_transactions_per_active_day = 0.0
        avg_daily_net = 0.0

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        calendar_days = int((max_date.normalize() - min_date.normalize()).days + 1)
    else:
        calendar_days = 0

    if calendar_days:
        avg_spending_per_calendar_day = float(total_spending / calendar_days)
        avg_earnings_per_calendar_day = float(total_earnings / calendar_days)
    else:
        avg_spending_per_calendar_day = 0.0
        avg_earnings_per_calendar_day = 0.0

    largest_spending = float(df["DebitCHF"].max(skipna=True) if tx_count else 0.0)
    largest_earning = float(df["CreditCHF"].max(skipna=True) if tx_count else 0.0)

    return {
        "total_spending": total_spending,
        "total_earnings": total_earnings,
        "net_cashflow": net_cashflow,
        "transactions": tx_count,
        "avg_spending": avg_spending,
        "avg_earning": avg_earning,
        "savings_rate": savings_rate,
        "active_days": active_days,
        "calendar_days": calendar_days,
        "avg_spending_per_active_day": avg_spending_per_active_day,
        "avg_earnings_per_active_day": avg_earnings_per_active_day,
        "avg_spending_per_calendar_day": avg_spending_per_calendar_day,
        "avg_earnings_per_calendar_day": avg_earnings_per_calendar_day,
        "avg_transactions_per_active_day": avg_transactions_per_active_day,
        "avg_daily_net": avg_daily_net,
        "largest_spending": largest_spending,
        "largest_earning": largest_earning,
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
    daily["CumulativeSpending"] = daily["Spending"].cumsum()
    daily["CumulativeEarnings"] = daily["Earnings"].cumsum()
    daily["CumulativeNet"] = daily["Net"].cumsum()
    daily.index = pd.to_datetime(daily.index)
    return daily


def weekday_average_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """Average spending/earnings/net by weekday."""
    daily = daily_net_cashflow(df).copy()
    if daily.empty:
        return pd.DataFrame(columns=["Spending", "Earnings", "Net"])

    daily["Weekday"] = daily.index.day_name()
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    grouped = daily.groupby("Weekday")[["Spending", "Earnings", "Net"]].mean()
    grouped = grouped.reindex(order).fillna(0.0)
    return grouped


def hourly_spending_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Spending/earnings totals and averages by hour of day."""
    out = df.copy()
    hour_from_time = (
        out["Time"]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
        .str.extract(r"^(\d{1,2})")[0]
    )
    out["Hour"] = pd.to_numeric(hour_from_time, errors="coerce")
    if "SortDateTime" in out.columns:
        out["Hour"] = out["Hour"].fillna(out["SortDateTime"].dt.hour)

    out = out[out["Hour"].notna()].copy()
    out["Hour"] = out["Hour"].astype(int)
    if out.empty:
        return pd.DataFrame(
            0.0,
            index=range(24),
            columns=["Spending", "Earnings", "Net", "AvgSpending", "AvgEarnings", "Transactions"],
        )

    grouped = out.groupby("Hour").agg(
        Spending=("DebitCHF", "sum"),
        Earnings=("CreditCHF", "sum"),
        AvgSpending=("DebitCHF", "mean"),
        AvgEarnings=("CreditCHF", "mean"),
        Transactions=("Hour", "size"),
    )
    grouped["Net"] = grouped["Earnings"] - grouped["Spending"]
    grouped = grouped.reindex(range(24)).fillna(0.0)
    return grouped


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


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return spending/earnings/net and share by category."""
    out = (
        df.groupby("Category", dropna=True)
        .agg(
            SpendingCHF=("DebitCHF", "sum"),
            EarningsCHF=("CreditCHF", "sum"),
            Transactions=("Category", "size"),
        )
        .sort_values("SpendingCHF", ascending=False)
    )
    out["NetCHF"] = out["EarningsCHF"] - out["SpendingCHF"]
    total_spending = float(out["SpendingCHF"].sum())
    total_earnings = float(out["EarningsCHF"].sum())
    out["SpendingSharePct"] = (
        out["SpendingCHF"].apply(lambda x: (x / total_spending * 100.0) if total_spending else 0.0)
    )
    out["EarningsSharePct"] = (
        out["EarningsCHF"].apply(lambda x: (x / total_earnings * 100.0) if total_earnings else 0.0)
    )
    return out


def income_source_summary(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top merchants/sources by credited amount."""
    grouped = (
        df.groupby("Merchant", dropna=True)["CreditCHF"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="EarningsCHF")
    )
    grouped = grouped[grouped["Merchant"].astype(str).str.strip() != ""]
    return grouped[grouped["EarningsCHF"] > 0]


def spending_velocity(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Rolling spending/earning averages from daily totals."""
    daily = daily_net_cashflow(df)
    if daily.empty:
        return pd.DataFrame(columns=["SpendingMA", "EarningsMA", "NetMA"])
    out = daily.copy()
    out["SpendingMA"] = out["Spending"].rolling(window=window_days, min_periods=1).mean()
    out["EarningsMA"] = out["Earnings"].rolling(window=window_days, min_periods=1).mean()
    out["NetMA"] = out["Net"].rolling(window=window_days, min_periods=1).mean()
    return out[["SpendingMA", "EarningsMA", "NetMA"]]


def quality_indicators(df: pd.DataFrame) -> dict[str, float]:
    """Data quality indicators for the current filtered view."""
    total = float(len(df))
    missing_time = float(df["Time"].fillna("").astype(str).str.strip().eq("").sum())
    unknown_category = float(df["Category"].fillna("").astype(str).str.strip().eq("Other").sum())
    unknown_tod = float(df["TimeOfDay"].fillna("").astype(str).str.strip().eq("Unknown").sum())
    missing_currency = float(df["Währung"].isna().sum()) if "Währung" in df.columns else 0.0
    return {
        "rows": total,
        "missing_time_pct": (missing_time / total * 100.0) if total else 0.0,
        "other_category_pct": (unknown_category / total * 100.0) if total else 0.0,
        "unknown_timeofday_pct": (unknown_tod / total * 100.0) if total else 0.0,
        "missing_currency_pct": (missing_currency / total * 100.0) if total else 0.0,
    }
