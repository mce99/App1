"""Analytics helpers for dashboard KPIs and charts."""

from __future__ import annotations

import datetime
import io
import json
import re
import zipfile

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


_WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _chart_axis_series(df: pd.DataFrame, x_axis: str) -> pd.Series:
    if x_axis == "Date":
        return pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    if x_axis == "Month":
        return pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").astype(str)
    if x_axis == "Weekday":
        return pd.to_datetime(df["Date"], errors="coerce").dt.day_name()
    if x_axis == "Hour":
        hour_from_time = (
            df["Time"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.split(".")
            .str[0]
            .str.extract(r"^(\d{1,2})")[0]
        )
        hour = pd.to_numeric(hour_from_time, errors="coerce")
        if "SortDateTime" in df.columns:
            hour = hour.fillna(pd.to_datetime(df["SortDateTime"], errors="coerce").dt.hour)
        return hour
    if x_axis in df.columns:
        source = df[x_axis]
    else:
        source = pd.Series("Unknown", index=df.index)
    return source.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")


def _sort_chart_index(chart: pd.DataFrame, x_axis: str) -> pd.DataFrame:
    if chart.empty:
        return chart
    if x_axis == "Date":
        out = chart.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[out.index.notna()].sort_index()
        return out
    if x_axis == "Month":
        out = chart.copy()
        order = pd.Series(
            pd.to_datetime(out.index.astype(str) + "-01", errors="coerce"), index=out.index
        ).sort_values()
        return out.loc[order.index]
    if x_axis == "Weekday":
        order_map = {day: idx for idx, day in enumerate(_WEEKDAY_ORDER)}
        ordered = sorted(chart.index.tolist(), key=lambda value: order_map.get(str(value), 999))
        return chart.loc[ordered]
    if x_axis == "Hour":
        out = chart.copy()
        out.index = pd.to_numeric(out.index, errors="coerce")
        out = out[out.index.notna()]
        out.index = out.index.astype(int)
        return out.sort_index()
    return chart.sort_index()


def chart_builder_dataset(
    df: pd.DataFrame,
    x_axis: str,
    metric: str,
    aggregation: str = "Sum",
    split_by: str = "None",
    top_n: int = 20,
    cumulative: bool = False,
    date_interval: str = "Daily",
    include_transfers: bool = True,
) -> pd.DataFrame:
    """Create aggregated chart dataset for the interactive chart builder."""
    work = df.copy()
    if work.empty:
        return pd.DataFrame()
    if not include_transfers and "IsTransfer" in work.columns:
        work = work[~work["IsTransfer"].fillna(False)].copy()
    if work.empty:
        return pd.DataFrame()

    work["_x"] = _chart_axis_series(work, x_axis)
    if x_axis in {"Date", "Month"}:
        interval = str(date_interval or "Daily").strip().title()
        if interval not in {"Daily", "Weekly", "Monthly"}:
            interval = "Daily"
        dt_axis = pd.to_datetime(work["_x"], errors="coerce")
        if interval == "Weekly":
            work["_x"] = dt_axis.dt.to_period("W-SUN").dt.start_time
        elif interval == "Monthly":
            work["_x"] = dt_axis.dt.to_period("M").astype(str)
        else:
            work["_x"] = dt_axis.dt.normalize()
    work = work[work["_x"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    debit = pd.to_numeric(work.get("DebitCHF", 0.0), errors="coerce").fillna(0.0)
    credit = pd.to_numeric(work.get("CreditCHF", 0.0), errors="coerce").fillna(0.0)
    if metric == "Spending":
        work["_value"] = debit
    elif metric == "Earnings":
        work["_value"] = credit
    elif metric == "Net":
        work["_value"] = credit - debit
    elif metric == "Transactions":
        work["_value"] = 1.0
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    agg_map = {
        "Sum": "sum",
        "Average": "mean",
        "Median": "median",
        "Count": "count",
    }
    agg_name = "sum" if metric == "Transactions" else agg_map.get(aggregation, "sum")
    item_limit = max(int(top_n), 1)

    if split_by != "None" and split_by in work.columns:
        work["_split"] = (
            work[split_by].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        )
        split_rank = (
            work.groupby("_split", dropna=False)["_value"].sum().abs().sort_values(ascending=False)
        )
        work = work[work["_split"].isin(split_rank.head(item_limit).index)]
        grouped = (
            work.groupby(["_x", "_split"], dropna=False)["_value"]
            .agg(agg_name)
            .reset_index(name="Value")
        )
        chart = grouped.pivot(index="_x", columns="_split", values="Value").fillna(0.0)
    else:
        grouped = work.groupby("_x", dropna=False)["_value"].agg(agg_name)
        chart = grouped.to_frame(name=metric)

    if x_axis not in {"Date", "Month", "Weekday", "Hour"} and len(chart) > item_limit:
        rank = chart.abs().sum(axis=1).sort_values(ascending=False)
        chart = chart.loc[rank.head(item_limit).index]

    chart = _sort_chart_index(chart, x_axis)
    chart.index.name = x_axis
    if cumulative:
        chart = chart.cumsum()
    return chart.round(4)


def period_over_period_metrics(current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Compare selected period against the immediately preceding equal-length period."""
    if current_df.empty or current_df["Date"].dropna().empty:
        return pd.DataFrame(
            columns=["Metric", "CurrentValue", "PriorValue", "DeltaAbs", "DeltaPct", "Signal"]
        )

    current_start = pd.Timestamp(current_df["Date"].min()).normalize()
    current_end = pd.Timestamp(current_df["Date"].max()).normalize()
    window_days = (current_end - current_start).days + 1
    prior_end = current_start - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(days=max(window_days - 1, 0))

    prior = baseline_df[
        pd.to_datetime(baseline_df["Date"], errors="coerce").dt.normalize().between(prior_start, prior_end)
    ].copy()

    current_kpi = calculate_kpis(current_df)
    prior_kpi = calculate_kpis(prior) if not prior.empty else calculate_kpis(current_df.iloc[0:0])

    rows = [
        ("Spending (CHF)", current_kpi["total_spending"], prior_kpi["total_spending"], True),
        ("Earnings (CHF)", current_kpi["total_earnings"], prior_kpi["total_earnings"], False),
        ("Net cashflow (CHF)", current_kpi["net_cashflow"], prior_kpi["net_cashflow"], False),
        ("Savings rate (%)", current_kpi["savings_rate"], prior_kpi["savings_rate"], False),
        (
            "Avg spend / calendar day (CHF)",
            current_kpi["avg_spending_per_calendar_day"],
            prior_kpi["avg_spending_per_calendar_day"],
            True,
        ),
        ("Transactions", current_kpi["transactions"], prior_kpi["transactions"], False),
    ]

    out_rows: list[dict[str, object]] = []
    for metric, current, prior_value, lower_is_better in rows:
        curr = float(current or 0.0)
        prev = float(prior_value or 0.0)
        delta = curr - prev
        delta_pct = (delta / abs(prev) * 100.0) if prev != 0 else 0.0
        if delta == 0:
            signal = "Flat"
        elif lower_is_better:
            signal = "Improved" if delta < 0 else "Worse"
        else:
            signal = "Improved" if delta > 0 else "Worse"
        out_rows.append(
            {
                "Metric": metric,
                "CurrentValue": round(curr, 4),
                "PriorValue": round(prev, 4),
                "DeltaAbs": round(delta, 4),
                "DeltaPct": round(delta_pct, 2),
                "Signal": signal,
            }
        )

    return pd.DataFrame(out_rows)


def spending_heatmap_matrix(df: pd.DataFrame, value_metric: str = "Spending") -> pd.DataFrame:
    """Weekday x hour matrix for spending/earnings/net/transactions."""
    if df.empty:
        return pd.DataFrame(0.0, index=_WEEKDAY_ORDER, columns=range(24))

    work = df.copy()
    hour_from_time = (
        work["Time"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
        .str.extract(r"^(\d{1,2})")[0]
    )
    hour = pd.to_numeric(hour_from_time, errors="coerce")
    if "SortDateTime" in work.columns:
        hour = hour.fillna(pd.to_datetime(work["SortDateTime"], errors="coerce").dt.hour)
    work["Hour"] = hour
    work["Weekday"] = pd.to_datetime(work["Date"], errors="coerce").dt.day_name()
    work = work[(work["Hour"].notna()) & (work["Weekday"].notna())].copy()
    if work.empty:
        return pd.DataFrame(0.0, index=_WEEKDAY_ORDER, columns=range(24))

    debit = pd.to_numeric(work.get("DebitCHF", 0.0), errors="coerce").fillna(0.0)
    credit = pd.to_numeric(work.get("CreditCHF", 0.0), errors="coerce").fillna(0.0)
    if value_metric == "Spending":
        work["Value"] = debit
    elif value_metric == "Earnings":
        work["Value"] = credit
    elif value_metric == "Net":
        work["Value"] = credit - debit
    elif value_metric == "Transactions":
        work["Value"] = 1.0
    else:
        raise ValueError(f"Unsupported value_metric: {value_metric}")

    grouped = (
        work.groupby(["Weekday", "Hour"], dropna=False)["Value"].sum().reset_index()
    )
    grouped["Hour"] = pd.to_numeric(grouped["Hour"], errors="coerce").astype(int)
    matrix = (
        grouped.pivot(index="Weekday", columns="Hour", values="Value")
        .reindex(index=_WEEKDAY_ORDER, columns=range(24))
        .fillna(0.0)
    )
    return matrix


def savings_opportunity_scanner(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Identify biggest monthly savings opportunities by category and merchant."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "LeverType",
                "Name",
                "AvgMonthlySpendCHF",
                "SuggestedCutPct",
                "PotentialMonthlySavingsCHF",
                "PotentialAnnualSavingsCHF",
            ]
        )

    work = df.copy()
    work["Month"] = pd.to_datetime(work["Date"], errors="coerce").dt.to_period("M").astype(str)
    spend = work[work["DebitCHF"] > 0].copy()
    if spend.empty:
        return pd.DataFrame(
            columns=[
                "LeverType",
                "Name",
                "AvgMonthlySpendCHF",
                "SuggestedCutPct",
                "PotentialMonthlySavingsCHF",
                "PotentialAnnualSavingsCHF",
            ]
        )

    protective_keywords = [
        "RENT",
        "HOUS",
        "INSUR",
        "UTILIT",
        "HEALTH",
        "TAX",
        "LOAN",
    ]

    def _category_cut_pct(name: str) -> float:
        upper = str(name).upper()
        if any(keyword in upper for keyword in protective_keywords):
            return 0.04
        return 0.12

    cat_monthly = (
        spend.groupby(["Category", "Month"], dropna=False)["DebitCHF"]
        .sum()
        .groupby("Category")
        .mean()
        .sort_values(ascending=False)
    )
    cat_rows = []
    for name, avg_monthly in cat_monthly.items():
        cut_pct = _category_cut_pct(str(name))
        monthly_save = float(avg_monthly) * cut_pct
        cat_rows.append(
            {
                "LeverType": "Category",
                "Name": str(name),
                "AvgMonthlySpendCHF": float(avg_monthly),
                "SuggestedCutPct": cut_pct * 100.0,
                "PotentialMonthlySavingsCHF": monthly_save,
                "PotentialAnnualSavingsCHF": monthly_save * 12.0,
            }
        )

    merchant_key = "MerchantNormalized" if "MerchantNormalized" in spend.columns else "Merchant"
    mer_monthly = (
        spend.groupby([merchant_key, "Month"], dropna=False)["DebitCHF"]
        .sum()
        .groupby(merchant_key)
        .mean()
        .sort_values(ascending=False)
        .head(25)
    )
    mer_rows = []
    for name, avg_monthly in mer_monthly.items():
        monthly_save = float(avg_monthly) * 0.08
        mer_rows.append(
            {
                "LeverType": "Merchant",
                "Name": str(name),
                "AvgMonthlySpendCHF": float(avg_monthly),
                "SuggestedCutPct": 8.0,
                "PotentialMonthlySavingsCHF": monthly_save,
                "PotentialAnnualSavingsCHF": monthly_save * 12.0,
            }
        )

    table = pd.DataFrame(cat_rows + mer_rows)
    if table.empty:
        return table
    return table.sort_values("PotentialMonthlySavingsCHF", ascending=False).head(int(max(top_n, 1)))


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


def ingestion_quality_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source ingestion diagnostics for upload QA."""
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for source, group in df.groupby("SourceFile", dropna=False):
        missing_time = float(group["Time"].fillna("").astype(str).str.strip().eq("").sum())
        unknown_time = float(group["TimeOfDay"].fillna("").astype(str).str.strip().eq("Unknown").sum())
        other_category = float(group["Category"].fillna("").astype(str).str.strip().eq("Other").sum())
        duplicate_ids = 0.0
        if "TransactionId" in group.columns:
            duplicate_ids = float(group["TransactionId"].duplicated(keep=False).sum())

        date_min = group["Date"].min()
        date_max = group["Date"].max()
        statement_from = group["StatementFrom"].iloc[0] if "StatementFrom" in group.columns else pd.NaT
        statement_to = group["StatementTo"].iloc[0] if "StatementTo" in group.columns else pd.NaT

        coverage_status = "N/A"
        if pd.notna(statement_from) and pd.notna(statement_to) and pd.notna(date_min) and pd.notna(date_max):
            if date_min < statement_from or date_max > statement_to:
                coverage_status = "Outside statement period"
            else:
                coverage_status = "Within statement period"

        rows.append(
            {
                "SourceFile": source,
                "Rows": int(len(group)),
                "DateFrom": date_min.date().isoformat() if pd.notna(date_min) else "",
                "DateTo": date_max.date().isoformat() if pd.notna(date_max) else "",
                "MissingTimePct": round((missing_time / len(group) * 100.0) if len(group) else 0.0, 1),
                "UnknownTimePct": round((unknown_time / len(group) * 100.0) if len(group) else 0.0, 1),
                "OtherCategoryPct": round((other_category / len(group) * 100.0) if len(group) else 0.0, 1),
                "DuplicateTransactionIds": int(duplicate_ids),
                "CoverageStatus": coverage_status,
                "SourceAccount": str(group["SourceAccount"].iloc[0]) if "SourceAccount" in group.columns else "",
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["Rows", "MissingTimePct"], ascending=[False, False]).reset_index(drop=True)


def generate_agent_action_plan(
    kpis: dict[str, float],
    quality: dict[str, float],
    benchmark_table: pd.DataFrame,
    anomalies: pd.DataFrame,
    dupes: pd.DataFrame,
    recurring: pd.DataFrame,
) -> pd.DataFrame:
    """Create prioritized, actionable tasks for an operator-style workflow."""
    rows: list[dict[str, object]] = []

    def add(priority: int, area: str, task: str, reason: str, destination: str) -> None:
        rows.append(
            {
                "Priority": int(priority),
                "Area": area,
                "Task": task,
                "Reason": reason,
                "OpenPage": destination,
            }
        )

    if quality.get("missing_time_pct", 0.0) >= 8:
        add(
            1,
            "Data quality",
            "Audit files with missing timestamps and normalize booking-time fallback.",
            f"{quality['missing_time_pct']:.1f}% of rows miss explicit time.",
            "Data Health",
        )
    if quality.get("other_category_pct", 0.0) >= 20:
        add(
            1,
            "Categorization",
            "Review uncategorized transactions and add keyword rules.",
            f"{quality['other_category_pct']:.1f}% categorized as Other.",
            "Review Queue",
        )
    if not anomalies.empty:
        add(
            2,
            "Risk monitoring",
            "Investigate anomaly outliers and mark expected exceptions.",
            f"{len(anomalies)} anomaly candidate(s) detected.",
            "Anomalies",
        )
    if not dupes.empty:
        add(
            2,
            "Data integrity",
            "Check duplicate candidates and keep one canonical transaction.",
            f"{len(dupes)} duplicate candidate row(s) found.",
            "Anomalies",
        )
    if recurring.empty and int(kpis.get("transactions", 0)) >= 120:
        add(
            3,
            "Planning",
            "Set recurring rules for known subscriptions and standing orders.",
            "No stable recurring pattern detected despite high transaction count.",
            "Plans & Recurring",
        )

    if not benchmark_table.empty:
        over = benchmark_table[benchmark_table["Status"] == "Over"]
        low = benchmark_table[(benchmark_table["Metric"] == "Savings") & (benchmark_table["Status"] == "Low")]
        for _, row in over.head(3).iterrows():
            add(
                2,
                "Cost optimization",
                f"Reduce {row['Metric']} by about CHF {max(row['MonthlyActualCHF'] - row['MonthlyTargetCHF'], 0):,.0f}/month.",
                f"{row['Metric']} is {row['GapPct']:.1f}% above configured target.",
                "Insights & Optimization",
            )
        if not low.empty:
            gap = float(low.iloc[0]["GapPct"])
            add(
                1,
                "Savings",
                "Increase monthly auto-transfer to savings and cut discretionary categories.",
                f"Savings rate is {gap:.1f}% below target.",
                "Insights & Optimization",
            )

    if not rows:
        add(
            5,
            "Maintenance",
            "No urgent actions. Keep monthly review cadence and refresh uploads.",
            "Current dataset quality and spend metrics are within target bands.",
            "Home",
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["Priority", "Area"]).reset_index(drop=True)


_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,8}\b")
_ACCOUNT_PATTERN = re.compile(r"\b\d{4}\s\d{8}\.\d{2}\b")
_TRANSFER_KEYWORDS = [
    "TRANSFER",
    "UEBERTRAG",
    "ÜBERTRAG",
    "EIGENKONTO",
    "KONTOUEBERTRAG",
    "ACCOUNT TRANSFER",
    "REVOLUT",
    "IBAN",
]


def normalize_merchant_name(value: str) -> str:
    text = str(value or "").upper()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bPENDING\b", "", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    aliases = {
        "UBER * EATS": "UBER EATS",
        "UBER *ONE MEMBERSHIP": "UBER ONE",
        "UBER *ONE": "UBER ONE",
        "UBER * EATS PENDING": "UBER EATS",
    }
    for key, norm in aliases.items():
        if key in text:
            return norm
    return text


def enrich_transaction_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    """Add merchant normalization, transfer detection, and account intelligence."""
    out = df.copy()
    out["MerchantNormalized"] = out["Merchant"].astype(str).apply(normalize_merchant_name)
    out["CounterpartyAccount"] = ""
    out["TransferConfidence"] = 0.0
    out["IsTransfer"] = False
    out["TransferDirection"] = "N/A"

    def detect_transfer(row: pd.Series) -> tuple[str, bool, float, str]:
        text = " ".join(
            [
                str(row.get("Beschreibung1", "")),
                str(row.get("Beschreibung2", "")),
                str(row.get("Beschreibung3", "")),
                str(row.get("Fussnoten", "")),
            ]
        )
        upper = text.upper()
        matches = _IBAN_PATTERN.findall(upper) + _ACCOUNT_PATTERN.findall(upper)
        counterparty = matches[0] if matches else ""
        keyword_hits = sum(1 for kw in _TRANSFER_KEYWORDS if kw in upper)
        confidence = 0.15
        if keyword_hits:
            confidence += min(0.45, 0.1 * keyword_hits)
        if counterparty:
            confidence += 0.35
        is_transfer = confidence >= 0.5
        direction = "N/A"
        if is_transfer:
            if float(row.get("DebitCHF", 0) or 0) > 0:
                direction = "Out"
            elif float(row.get("CreditCHF", 0) or 0) > 0:
                direction = "In"
            else:
                direction = "Unknown"
        return counterparty, is_transfer, round(min(confidence, 0.99), 2), direction

    detected = out.apply(detect_transfer, axis=1, result_type="expand")
    detected.columns = ["CounterpartyAccount", "IsTransfer", "TransferConfidence", "TransferDirection"]
    out[detected.columns] = detected
    out["SourceAccount"] = out.get("SourceAccount", out.get("SourceFile", "Unknown")).fillna("Unknown")
    out["SourceAccount"] = out["SourceAccount"].astype(str).str.strip().replace("", "Unknown")
    return out


def apply_category_overrides(df: pd.DataFrame, overrides: dict[str, str]) -> pd.DataFrame:
    """Apply user-reviewed category overrides by TransactionId."""
    if not overrides:
        return df
    out = df.copy()
    if "TransactionId" not in out.columns:
        return out
    out["Category"] = out.apply(
        lambda row: overrides.get(str(row.get("TransactionId", "")), row.get("Category", "Other")),
        axis=1,
    )
    out["CategoryOverridden"] = out["TransactionId"].astype(str).isin({str(k) for k in overrides.keys()})
    return out


def review_queue(df: pd.DataFrame, min_confidence: float = 0.65) -> pd.DataFrame:
    """Transactions that should be manually reviewed for category quality."""
    out = df.copy()
    if "CategoryConfidence" not in out.columns:
        out["CategoryConfidence"] = 0.5
    queue = out[
        (out["Category"].fillna("Other") == "Other")
        | (out["CategoryConfidence"].fillna(0.0) < min_confidence)
        | (out.get("IsTransfer", False))
    ].copy()
    return queue.sort_values(["Date", "Time"], ascending=[False, False]).reset_index(drop=True)


def possible_duplicate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Find potential duplicates that may have different IDs but same economic event."""
    out = df.copy()
    out["DupKey"] = (
        out["MerchantNormalized"].astype(str)
        + "|"
        + out["Währung"].astype(str)
        + "|"
        + out["DebitCHF"].round(2).astype(str)
        + "|"
        + out["CreditCHF"].round(2).astype(str)
    )
    candidates = out[out["DupKey"].duplicated(keep=False)].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values(["DupKey", "Date", "Time"])
    return candidates[
        [
            "TransactionId",
            "Date",
            "Time",
            "SourceFile",
            "SourceAccount",
            "Merchant",
            "DebitCHF",
            "CreditCHF",
            "DupKey",
        ]
    ].reset_index(drop=True)


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """Flag spending anomalies relative to category and merchant history."""
    out = df.copy()
    spend = out[out["DebitCHF"] > 0].copy()
    if spend.empty:
        return pd.DataFrame()

    category_stats = spend.groupby("Category")["DebitCHF"].agg(["mean", "std"]).rename(
        columns={"mean": "cat_mean", "std": "cat_std"}
    )
    merchant_stats = spend.groupby("MerchantNormalized")["DebitCHF"].agg(["mean", "std"]).rename(
        columns={"mean": "mer_mean", "std": "mer_std"}
    )
    spend = spend.join(category_stats, on="Category")
    spend = spend.join(merchant_stats, on="MerchantNormalized")

    spend["CatZ"] = (spend["DebitCHF"] - spend["cat_mean"]) / spend["cat_std"].replace(0, pd.NA)
    spend["MerchZ"] = (spend["DebitCHF"] - spend["mer_mean"]) / spend["mer_std"].replace(0, pd.NA)
    spend["AnomalyScore"] = spend[["CatZ", "MerchZ"]].abs().max(axis=1).fillna(0.0)
    flagged = spend[spend["AnomalyScore"] >= z_threshold].copy()
    if flagged.empty:
        return pd.DataFrame()
    flagged["Reason"] = flagged.apply(
        lambda row: f"High spend vs baseline (score {row['AnomalyScore']:.2f})", axis=1
    )
    return flagged[
        [
            "TransactionId",
            "Date",
            "Time",
            "Merchant",
            "Category",
            "DebitCHF",
            "AnomalyScore",
            "Reason",
        ]
    ].sort_values("AnomalyScore", ascending=False)


def account_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize spending/earnings/transfers by source account."""
    out = df.copy()
    grouped = out.groupby("SourceAccount", dropna=False).agg(
        Transactions=("SourceAccount", "size"),
        SpendingCHF=("DebitCHF", "sum"),
        EarningsCHF=("CreditCHF", "sum"),
        TransferOutCHF=("DebitCHF", lambda s: float(s[out.loc[s.index, "IsTransfer"]].sum())),
        TransferInCHF=("CreditCHF", lambda s: float(s[out.loc[s.index, "IsTransfer"]].sum())),
    )
    grouped["NetCHF"] = grouped["EarningsCHF"] - grouped["SpendingCHF"]
    grouped["ExternalSpendingCHF"] = grouped["SpendingCHF"] - grouped["TransferOutCHF"]
    grouped["ExternalEarningsCHF"] = grouped["EarningsCHF"] - grouped["TransferInCHF"]
    return grouped.sort_values("SpendingCHF", ascending=False).reset_index()


def goals_progress(goal_config: dict[str, dict], current_net: float) -> pd.DataFrame:
    """Compute progress for user-defined savings goals."""
    rows = []
    for name, payload in goal_config.items():
        if isinstance(payload, dict):
            target = float(payload.get("target", 0) or 0)
            saved = float(payload.get("saved", 0) or 0)
        else:
            target = float(payload or 0)
            saved = 0.0
        projected = saved + max(current_net, 0.0)
        progress = (projected / target * 100.0) if target else 0.0
        rows.append(
            {
                "Goal": str(name),
                "TargetCHF": target,
                "SavedCHF": saved,
                "ProjectedSavedCHF": projected,
                "ProgressPct": min(progress, 100.0),
                "RemainingCHF": max(target - projected, 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("RemainingCHF")


def forecast_cashflow(df: pd.DataFrame, recurring: pd.DataFrame) -> pd.DataFrame:
    """Forecast 30/60/90 day cashflow using averages + recurring signals."""
    daily = daily_net_cashflow(df)
    if daily.empty:
        return pd.DataFrame()
    baseline_spend = float(daily["Spending"].tail(60).mean())
    baseline_earn = float(daily["Earnings"].tail(60).mean())

    rows = []
    for horizon in [30, 60, 90]:
        recurring_spend = 0.0
        recurring_earn = 0.0
        if not recurring.empty:
            for _, row in recurring.iterrows():
                cadence = max(float(row.get("CadenceDays", 30)), 1.0)
                occurrences = horizon / cadence
                recurring_spend += float(row.get("AvgSpendingCHF", 0) or 0) * occurrences
                recurring_earn += float(row.get("AvgEarningsCHF", 0) or 0) * occurrences

        expected_spend = baseline_spend * horizon + recurring_spend
        expected_earn = baseline_earn * horizon + recurring_earn
        rows.append(
            {
                "HorizonDays": horizon,
                "ExpectedSpendingCHF": round(expected_spend, 2),
                "ExpectedEarningsCHF": round(expected_earn, 2),
                "ExpectedNetCHF": round(expected_earn - expected_spend, 2),
            }
        )

    return pd.DataFrame(rows)


def data_health_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tabular health report for parsed/processed data."""
    rows = [
        ("Rows", float(len(df))),
        ("Missing Date", float(df["Date"].isna().sum())),
        ("Missing Time", float(df["Time"].fillna("").astype(str).str.strip().eq("").sum())),
        ("Missing Currency", float(df["Währung"].isna().sum()) if "Währung" in df.columns else 0.0),
        ("Category = Other", float(df["Category"].fillna("").eq("Other").sum())),
        ("Unknown TimeOfDay", float(df["TimeOfDay"].fillna("").eq("Unknown").sum())),
        ("Transfer tagged", float(df.get("IsTransfer", False).sum())),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Count"])


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _simple_pdf_from_lines(lines: list[str], lines_per_page: int = 46) -> bytes:
    """Create a simple text-only PDF without external dependencies."""
    clean_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not clean_lines:
        clean_lines = ["PulseLedger Executive Brief", "No data available."]

    page_chunks = [
        clean_lines[idx : idx + max(1, int(lines_per_page))]
        for idx in range(0, len(clean_lines), max(1, int(lines_per_page)))
    ]

    objects: list[bytes | str | None] = [None, None]  # 1: Catalog, 2: Pages

    def add_obj(payload: bytes | str) -> int:
        objects.append(payload)
        return len(objects)

    font_id = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []

    for chunk in page_chunks:
        ops = ["BT", "/F1 10 Tf", "50 800 Td"]
        for idx, line in enumerate(chunk):
            if idx > 0:
                ops.append("0 -15 Td")
            ops.append(f"({_pdf_escape(line)}) Tj")
        ops.append("ET")
        stream_text = "\n".join(ops)
        stream_bytes = stream_text.encode("latin-1", errors="replace")
        content_id = add_obj(
            f"<< /Length {len(stream_bytes)} >>\nstream\n{stream_text}\nendstream"
        )
        page_id = add_obj(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            )
        )
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    objects[0] = "<< /Type /Catalog /Pages 2 0 R >>"

    payload = bytearray()
    payload.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for idx, obj in enumerate(objects, start=1):
        offset = len(payload)
        offsets.append(offset)
        payload.extend(f"{idx} 0 obj\n".encode("ascii"))
        raw = obj if isinstance(obj, bytes) else str(obj).encode("latin-1", errors="replace")
        payload.extend(raw)
        payload.extend(b"\nendobj\n")

    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        payload.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("ascii")
    )
    return bytes(payload)


def build_executive_pdf_report(
    df: pd.DataFrame,
    kpis: dict[str, float],
    period_table: pd.DataFrame | None = None,
    opportunity_table: pd.DataFrame | None = None,
) -> bytes:
    """Build a dependency-free executive PDF brief."""
    period_table = period_table if period_table is not None else pd.DataFrame()
    opportunity_table = opportunity_table if opportunity_table is not None else pd.DataFrame()

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "PulseLedger Executive Brief",
        f"Generated: {generated_at}",
        f"Coverage: {min_date} to {max_date}",
        "",
        "Executive KPI Summary",
        f"- Transactions: {int(kpis['transactions']):,}",
        f"- Spending (CHF): {kpis['total_spending']:,.2f}",
        f"- Earnings (CHF): {kpis['total_earnings']:,.2f}",
        f"- Net cashflow (CHF): {kpis['net_cashflow']:,.2f}",
        f"- Savings rate: {kpis['savings_rate']:.1f}%",
        "",
        "Period-over-period",
    ]

    if period_table.empty:
        lines.append("- Not enough prior-period data available.")
    else:
        for _, row in period_table.head(8).iterrows():
            lines.append(
                (
                    f"- {row['Metric']}: {float(row['CurrentValue']):,.2f} "
                    f"(prior {float(row['PriorValue']):,.2f}, "
                    f"delta {float(row['DeltaPct']):+.1f}%, {row['Signal']})"
                )
            )

    lines.extend(["", "Top savings opportunities"])
    if opportunity_table.empty:
        lines.append("- No material opportunities identified.")
    else:
        for _, row in opportunity_table.head(12).iterrows():
            lines.append(
                (
                    f"- {row['LeverType']} | {row['Name']}: "
                    f"save {float(row['PotentialMonthlySavingsCHF']):,.2f} CHF/month "
                    f"({float(row['SuggestedCutPct']):.1f}% cut)"
                )
            )

    lines.extend(["", "Confidential - internal personal finance analysis"])
    return _simple_pdf_from_lines(lines)


def build_report_pack(
    df: pd.DataFrame,
    kpis: dict[str, float],
    monthly: pd.DataFrame,
    period_table: pd.DataFrame | None = None,
    opportunity_table: pd.DataFrame | None = None,
) -> tuple[str, bytes, bytes]:
    """Build markdown summary, zip pack, and executive PDF brief."""
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    period_table = period_table if period_table is not None else pd.DataFrame()
    opportunity_table = opportunity_table if opportunity_table is not None else pd.DataFrame()
    markdown = (
        f"# PulseLedger Report\n\n"
        f"- Period: {min_date} to {max_date}\n"
        f"- Transactions: {int(kpis['transactions']):,}\n"
        f"- Spending (CHF): {kpis['total_spending']:,.2f}\n"
        f"- Earnings (CHF): {kpis['total_earnings']:,.2f}\n"
        f"- Net cashflow (CHF): {kpis['net_cashflow']:,.2f}\n"
        f"- Savings rate: {kpis['savings_rate']:.1f}%\n"
    )
    executive_pdf = build_executive_pdf_report(
        df,
        kpis,
        period_table=period_table,
        opportunity_table=opportunity_table,
    )

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.md", markdown)
        zf.writestr("transactions.csv", df.to_csv(index=False))
        zf.writestr("monthly.csv", monthly.to_csv())
        zf.writestr("kpis.json", json.dumps(kpis, indent=2))
        zf.writestr("period_comparison.csv", period_table.to_csv(index=False))
        zf.writestr("opportunities.csv", opportunity_table.to_csv(index=False))
        zf.writestr("executive_brief.pdf", executive_pdf)
    return markdown, output.getvalue(), executive_pdf


def _spend_bucket(row: pd.Series) -> str:
    category = str(row.get("Category", "")).strip()
    merchant = str(row.get("MerchantNormalized", row.get("Merchant", ""))).upper()
    if category == "Transfers":
        return "Transfers"
    if category == "Groceries":
        return "Groceries"
    if category == "Restaurants & Cafes":
        return "Dining"
    if category == "Gas Stations":
        return "Transport"
    if category == "Clothing Brands":
        return "Shopping"
    if category == "Shopping (General)":
        return "Shopping"
    if category == "Food & Drink":
        grocery_keys = ["COOP", "MIGROS", "SPAR", "PRONTO", "AGROLA", "SUPERMARKT", "GROCERY"]
        if any(key in merchant for key in grocery_keys):
            return "Groceries"
        return "Dining"
    if category == "Transport":
        return "Transport"
    if category == "Utilities & Bills":
        return "Bills"
    if category == "Shopping & Retail":
        return "Shopping"
    if category == "Entertainment & Leisure":
        sub_keys = ["NETFLIX", "SPOTIFY", "APPLE", "UBER ONE", "GYM", "SUBSCRIPTION"]
        if any(key in merchant for key in sub_keys):
            return "Subscriptions"
        return "Leisure"
    return "Other"


def monthly_salary_estimate(df: pd.DataFrame) -> dict[str, object]:
    """Estimate monthly salary from recurring incoming transactions."""
    work = df.copy()
    work["Month"] = work["Date"].dt.to_period("M").astype(str)
    if "MerchantNormalized" not in work.columns:
        work["MerchantNormalized"] = work.get("Merchant", "").astype(str).str.upper()

    incoming = work[(work["CreditCHF"] > 0) & (~work.get("IsTransfer", False))]
    if incoming.empty:
        return {"avg_monthly_salary": 0.0, "salary_monthly": pd.DataFrame(), "salary_sources": pd.DataFrame()}

    monthly_by_merchant = (
        incoming.groupby(["Month", "MerchantNormalized"], dropna=True)["CreditCHF"]
        .sum()
        .reset_index()
    )
    merchant_stats = monthly_by_merchant.groupby("MerchantNormalized")["CreditCHF"].agg(
        Months="count", AvgMonthly="mean", MedianMonthly="median", MaxMonthly="max"
    )
    candidates = merchant_stats[(merchant_stats["Months"] >= 2) & (merchant_stats["MedianMonthly"] >= 1000)]
    if candidates.empty:
        candidates = merchant_stats.sort_values("AvgMonthly", ascending=False).head(1)

    candidate_merchants = set(candidates.index.tolist())
    salary_monthly = (
        monthly_by_merchant[monthly_by_merchant["MerchantNormalized"].isin(candidate_merchants)]
        .groupby("Month")["CreditCHF"]
        .sum()
        .reset_index(name="EstimatedSalaryCHF")
    )
    if salary_monthly.empty:
        salary_monthly = (
            incoming.groupby("Month")["CreditCHF"].sum().reset_index(name="EstimatedSalaryCHF")
        )
    salary_monthly = salary_monthly.sort_values("Month")
    avg_salary = float(salary_monthly["EstimatedSalaryCHF"].tail(6).mean())

    sources = candidates.sort_values("AvgMonthly", ascending=False).reset_index().rename(
        columns={"MerchantNormalized": "SalarySource"}
    )
    return {
        "avg_monthly_salary": avg_salary,
        "salary_monthly": salary_monthly,
        "salary_sources": sources,
    }


def benchmark_assessment(
    df: pd.DataFrame, avg_monthly_salary: float, benchmark_cfg: dict[str, float] | None = None
) -> pd.DataFrame:
    """Compare spending behavior to common budgeting heuristics."""
    if benchmark_cfg is None:
        benchmark_cfg = {
            "NeedsMaxPct": 50.0,
            "WantsMaxPct": 30.0,
            "SavingsMinPct": 20.0,
            "GroceriesMaxPct": 10.0,
            "DiningMaxPct": 8.0,
            "SubscriptionsMaxPct": 5.0,
            "TransportMaxPct": 15.0,
        }

    work = df.copy()
    work["Month"] = work["Date"].dt.to_period("M").astype(str)
    work["SpendBucket"] = work.apply(_spend_bucket, axis=1)

    monthly = work.groupby("Month").agg(
        Spending=("DebitCHF", "sum"),
        Earnings=("CreditCHF", "sum"),
    )
    monthly["Net"] = monthly["Earnings"] - monthly["Spending"]
    avg_monthly_earnings = float(monthly["Earnings"].mean()) if not monthly.empty else 0.0
    income_base = float(avg_monthly_salary or avg_monthly_earnings or 0.0)

    spend = work[work["DebitCHF"] > 0]
    bucket_avg = (
        spend.groupby(["Month", "SpendBucket"])["DebitCHF"].sum().groupby("SpendBucket").mean()
        if not spend.empty
        else pd.Series(dtype=float)
    )

    needs = float(sum(bucket_avg.get(k, 0.0) for k in ["Groceries", "Bills", "Transport"]))
    wants = float(sum(bucket_avg.get(k, 0.0) for k in ["Dining", "Leisure", "Shopping", "Subscriptions"]))
    groceries = float(bucket_avg.get("Groceries", 0.0))
    dining = float(bucket_avg.get("Dining", 0.0))
    subscriptions = float(bucket_avg.get("Subscriptions", 0.0))
    transport = float(bucket_avg.get("Transport", 0.0))
    savings_pct = float((monthly["Net"].mean() / income_base * 100.0) if income_base and not monthly.empty else 0.0)

    def _pct(value: float) -> float:
        return (value / income_base * 100.0) if income_base else 0.0

    checks = [
        ("Needs", needs, benchmark_cfg["NeedsMaxPct"], "max"),
        ("Wants", wants, benchmark_cfg["WantsMaxPct"], "max"),
        ("Savings", savings_pct, benchmark_cfg["SavingsMinPct"], "min"),
        ("Groceries", groceries, benchmark_cfg["GroceriesMaxPct"], "max"),
        ("Dining", dining, benchmark_cfg["DiningMaxPct"], "max"),
        ("Subscriptions", subscriptions, benchmark_cfg["SubscriptionsMaxPct"], "max"),
        ("Transport", transport, benchmark_cfg["TransportMaxPct"], "max"),
    ]

    rows = []
    for metric, actual_value, target_pct, mode in checks:
        actual_pct = actual_value if metric == "Savings" else _pct(actual_value)
        if mode == "max":
            gap = actual_pct - target_pct
            status = "Over" if gap > 0 else "Within"
        else:
            gap = target_pct - actual_pct
            status = "Low" if gap > 0 else "On Track"
        rows.append(
            {
                "Metric": metric,
                "ActualPctIncome": round(actual_pct, 2),
                "TargetPctIncome": round(target_pct, 2),
                "GapPct": round(gap, 2),
                "Status": status,
                "MonthlyActualCHF": round(actual_value if metric != "Savings" else 0.0, 2),
                "MonthlyTargetCHF": round((target_pct / 100.0) * income_base, 2) if income_base else 0.0,
            }
        )
    return pd.DataFrame(rows)


def merchant_insights(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Merchant-level behavior analytics."""
    work = df.copy()
    if "MerchantNormalized" not in work.columns:
        work["MerchantNormalized"] = work.get("Merchant", "").astype(str).str.upper()
    grouped = (
        work.groupby("MerchantNormalized")
        .agg(
            Transactions=("MerchantNormalized", "size"),
            SpendingCHF=("DebitCHF", "sum"),
            EarningsCHF=("CreditCHF", "sum"),
            AvgTicketCHF=("DebitCHF", "mean"),
            LastSeen=("Date", "max"),
        )
        .sort_values("SpendingCHF", ascending=False)
        .head(top_n)
    )
    grouped["NetCHF"] = grouped["EarningsCHF"] - grouped["SpendingCHF"]
    return grouped.reset_index().rename(columns={"MerchantNormalized": "Merchant"})


def spending_recommendations(df: pd.DataFrame, benchmark_table: pd.DataFrame) -> pd.DataFrame:
    """Actionable recommendations to reduce spending."""
    recommendations: list[dict[str, object]] = []
    spending_total = float(df["DebitCHF"].sum())

    for _, row in benchmark_table.iterrows():
        metric = str(row["Metric"])
        status = str(row["Status"])
        if metric in {"Savings"} and status == "Low":
            recommendations.append(
                {
                    "Priority": 1,
                    "Area": "Savings",
                    "Issue": f"Savings rate below target by {row['GapPct']:.1f}%",
                    "Suggestion": "Cap discretionary categories first and automate a fixed monthly transfer to savings.",
                }
            )
        elif status == "Over":
            recommendations.append(
                {
                    "Priority": 2,
                    "Area": metric,
                    "Issue": f"{metric} is {row['GapPct']:.1f}% above target.",
                    "Suggestion": f"Reduce {metric.lower()} budget by about CHF {max(row['MonthlyActualCHF'] - row['MonthlyTargetCHF'], 0):,.0f}/month.",
                }
            )

    merchant_table = merchant_insights(df, top_n=5)
    if not merchant_table.empty and spending_total > 0:
        for _, m in merchant_table.iterrows():
            share = float(m["SpendingCHF"]) / spending_total * 100.0
            if share >= 12:
                recommendations.append(
                    {
                        "Priority": 3,
                        "Area": "Merchant concentration",
                        "Issue": f"{m['Merchant']} accounts for {share:.1f}% of total spending.",
                        "Suggestion": "Set merchant-specific monthly cap and seek alternatives or bulk optimization.",
                    }
                )

    if not recommendations:
        recommendations.append(
            {
                "Priority": 5,
                "Area": "General",
                "Issue": "No major overspend flags detected.",
                "Suggestion": "Maintain current plan and focus on savings automation and periodic review.",
            }
        )

    return pd.DataFrame(recommendations).sort_values("Priority").reset_index(drop=True)


def merchant_concentration_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Merchant concentration on spending side with cumulative share."""
    work = df.copy()
    if "MerchantNormalized" not in work.columns:
        work["MerchantNormalized"] = work.get("Merchant", "").astype(str).str.upper()
    spend = (
        work.groupby("MerchantNormalized", dropna=False)["DebitCHF"]
        .sum()
        .sort_values(ascending=False)
    )
    spend = spend[spend > 0]
    if spend.empty:
        return pd.DataFrame(columns=["Merchant", "SpendingCHF", "SharePct", "CumulativeSharePct"])

    total = float(spend.sum())
    out = spend.head(int(top_n)).reset_index().rename(
        columns={"MerchantNormalized": "Merchant", "DebitCHF": "SpendingCHF"}
    )
    out["SharePct"] = out["SpendingCHF"].apply(lambda x: (x / total * 100.0) if total else 0.0)
    out["CumulativeSharePct"] = out["SharePct"].cumsum()
    return out


def income_concentration_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Income source concentration with cumulative share."""
    work = df.copy()
    if "MerchantNormalized" not in work.columns:
        work["MerchantNormalized"] = work.get("Merchant", "").astype(str).str.upper()
    income = (
        work.groupby("MerchantNormalized", dropna=False)["CreditCHF"]
        .sum()
        .sort_values(ascending=False)
    )
    income = income[income > 0]
    if income.empty:
        return pd.DataFrame(columns=["Source", "EarningsCHF", "SharePct", "CumulativeSharePct"])

    total = float(income.sum())
    out = income.head(int(top_n)).reset_index().rename(
        columns={"MerchantNormalized": "Source", "CreditCHF": "EarningsCHF"}
    )
    out["SharePct"] = out["EarningsCHF"].apply(lambda x: (x / total * 100.0) if total else 0.0)
    out["CumulativeSharePct"] = out["SharePct"].cumsum()
    return out


def cashflow_stability_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Stability diagnostics on monthly net cashflow."""
    monthly = monthly_cashflow(df)
    if monthly.empty:
        return {
            "months": 0.0,
            "negative_months": 0.0,
            "negative_month_ratio_pct": 0.0,
            "avg_net_monthly": 0.0,
            "median_net_monthly": 0.0,
            "net_std_monthly": 0.0,
            "net_coeff_var": 0.0,
            "max_drawdown_monthly_net": 0.0,
            "longest_negative_streak": 0.0,
            "net_trend_slope": 0.0,
        }

    net = monthly["Net"].astype(float)
    months = int(len(net))
    negative_mask = net < 0
    negative_months = int(negative_mask.sum())
    negative_ratio = (negative_months / months * 100.0) if months else 0.0
    avg_net = float(net.mean())
    median_net = float(net.median())
    net_std = float(net.std(ddof=0)) if months > 1 else 0.0
    coeff_var = float(net_std / abs(avg_net)) if avg_net != 0 else 0.0

    cumulative = net.cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0

    longest_streak = 0
    current_streak = 0
    for is_negative in negative_mask.tolist():
        if is_negative:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

    if months > 1:
        x = pd.Series(range(months), dtype=float)
        y = net.reset_index(drop=True)
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        num = float(((x - x_mean) * (y - y_mean)).sum())
        den = float(((x - x_mean) ** 2).sum())
        slope = num / den if den else 0.0
    else:
        slope = 0.0

    return {
        "months": float(months),
        "negative_months": float(negative_months),
        "negative_month_ratio_pct": float(negative_ratio),
        "avg_net_monthly": avg_net,
        "median_net_monthly": median_net,
        "net_std_monthly": net_std,
        "net_coeff_var": coeff_var,
        "max_drawdown_monthly_net": max_drawdown,
        "longest_negative_streak": float(longest_streak),
        "net_trend_slope": float(slope),
    }


def weekday_weekend_split(df: pd.DataFrame) -> pd.DataFrame:
    """Compare weekday vs weekend spending/earnings behavior."""
    work = df.copy()
    if work.empty:
        return pd.DataFrame(columns=["Segment", "Transactions", "SpendingCHF", "EarningsCHF", "NetCHF"])
    work["Segment"] = work["Date"].dt.dayofweek.apply(lambda v: "Weekend" if int(v) >= 5 else "Weekday")
    out = (
        work.groupby("Segment", dropna=False)
        .agg(
            Transactions=("Segment", "size"),
            SpendingCHF=("DebitCHF", "sum"),
            EarningsCHF=("CreditCHF", "sum"),
        )
        .reset_index()
    )
    out["NetCHF"] = out["EarningsCHF"] - out["SpendingCHF"]
    total_spend = float(out["SpendingCHF"].sum())
    total_earn = float(out["EarningsCHF"].sum())
    out["SpendingSharePct"] = out["SpendingCHF"].apply(lambda x: (x / total_spend * 100.0) if total_spend else 0.0)
    out["EarningsSharePct"] = out["EarningsCHF"].apply(lambda x: (x / total_earn * 100.0) if total_earn else 0.0)
    return out.sort_values("Segment").reset_index(drop=True)


def transaction_size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of transaction sizes for spending and earnings."""
    bins = [0.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, float("inf")]
    labels = [
        "0-20",
        "20-50",
        "50-100",
        "100-250",
        "250-500",
        "500-1000",
        "1000+",
    ]
    work = df.copy()
    if work.empty:
        return pd.DataFrame(
            {
                "Band": labels,
                "SpendingTx": [0] * len(labels),
                "SpendingCHF": [0.0] * len(labels),
                "EarningsTx": [0] * len(labels),
                "EarningsCHF": [0.0] * len(labels),
            }
        )

    spend = work[work["DebitCHF"] > 0].copy()
    earn = work[work["CreditCHF"] > 0].copy()
    if not spend.empty:
        spend["Band"] = pd.cut(spend["DebitCHF"], bins=bins, labels=labels, include_lowest=True, right=False)
    if not earn.empty:
        earn["Band"] = pd.cut(earn["CreditCHF"], bins=bins, labels=labels, include_lowest=True, right=False)

    spend_summary = (
        spend.groupby("Band", observed=False)["DebitCHF"]
        .agg(SpendingTx="count", SpendingCHF="sum")
        .reindex(labels, fill_value=0)
        if not spend.empty
        else pd.DataFrame(index=labels, data={"SpendingTx": 0, "SpendingCHF": 0.0})
    )
    earn_summary = (
        earn.groupby("Band", observed=False)["CreditCHF"]
        .agg(EarningsTx="count", EarningsCHF="sum")
        .reindex(labels, fill_value=0)
        if not earn.empty
        else pd.DataFrame(index=labels, data={"EarningsTx": 0, "EarningsCHF": 0.0})
    )

    out = pd.concat([spend_summary, earn_summary], axis=1).reset_index().rename(columns={"index": "Band"})
    return out


def category_volatility(df: pd.DataFrame, min_months: int = 3) -> pd.DataFrame:
    """Category volatility based on monthly spending history."""
    work = df.copy()
    if work.empty:
        return pd.DataFrame()
    work["Month"] = work["Date"].dt.to_period("M").astype(str)
    spend = work[work["DebitCHF"] > 0]
    if spend.empty:
        return pd.DataFrame()

    month_cat = spend.groupby(["Category", "Month"], dropna=False)["DebitCHF"].sum().reset_index()
    stats = month_cat.groupby("Category")["DebitCHF"].agg(
        Months="count", AvgMonthlySpend="mean", StdMonthlySpend="std", MedianMonthlySpend="median", MaxMonthlySpend="max"
    )
    stats["StdMonthlySpend"] = stats["StdMonthlySpend"].fillna(0.0)
    stats["CoeffVar"] = stats.apply(
        lambda row: (float(row["StdMonthlySpend"]) / float(row["AvgMonthlySpend"])) if float(row["AvgMonthlySpend"]) > 0 else 0.0,
        axis=1,
    )
    stats = stats[stats["Months"] >= int(min_months)]
    if stats.empty:
        return pd.DataFrame()
    return stats.sort_values("StdMonthlySpend", ascending=False).reset_index()


def spending_run_rate_projection(df: pd.DataFrame, lookback_months: int = 3) -> dict[str, float]:
    """Run-rate projection from recent monthly behavior."""
    monthly = monthly_cashflow(df)
    if monthly.empty:
        return {
            "lookback_months": float(lookback_months),
            "avg_monthly_spending": 0.0,
            "avg_monthly_earnings": 0.0,
            "avg_monthly_net": 0.0,
            "projected_annual_spending": 0.0,
            "projected_annual_earnings": 0.0,
            "projected_annual_net": 0.0,
        }

    recent = monthly.tail(max(int(lookback_months), 1))
    avg_spending = float(recent["Spending"].mean())
    avg_earnings = float(recent["Earnings"].mean())
    avg_net = float(recent["Net"].mean())

    return {
        "lookback_months": float(len(recent)),
        "avg_monthly_spending": avg_spending,
        "avg_monthly_earnings": avg_earnings,
        "avg_monthly_net": avg_net,
        "projected_annual_spending": avg_spending * 12.0,
        "projected_annual_earnings": avg_earnings * 12.0,
        "projected_annual_net": avg_net * 12.0,
    }


def monthly_trend_diagnostics(df: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
    """Monthly trend diagnostics with momentum and volatility features."""
    monthly = monthly_cashflow(df)
    if monthly.empty:
        return pd.DataFrame()

    out = monthly.copy()
    if lookback_months > 0:
        out = out.tail(int(lookback_months))

    out["SpendingMoMCHF"] = out["Spending"].diff()
    out["EarningsMoMCHF"] = out["Earnings"].diff()
    out["NetMoMCHF"] = out["Net"].diff()
    out["SpendingMoMPct"] = out["Spending"].pct_change() * 100.0
    out["EarningsMoMPct"] = out["Earnings"].pct_change() * 100.0
    out["Net3MAvg"] = out["Net"].rolling(window=3, min_periods=1).mean()
    out["Spending3MAvg"] = out["Spending"].rolling(window=3, min_periods=1).mean()
    out["NetVolatility3M"] = out["Net"].rolling(window=3, min_periods=2).std().fillna(0.0)
    return out.reset_index()


def category_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Category-level monthly momentum (latest month vs prior month)."""
    work = df.copy()
    if work.empty:
        return pd.DataFrame()
    work["Month"] = work["Date"].dt.to_period("M").astype(str)
    spend = work[work["DebitCHF"] > 0]
    if spend.empty:
        return pd.DataFrame()

    grouped = spend.groupby(["Category", "Month"], dropna=False)["DebitCHF"].sum().unstack(fill_value=0.0)
    if grouped.shape[1] < 2:
        return pd.DataFrame()

    months = sorted(grouped.columns.tolist())
    prev_month = months[-2]
    latest_month = months[-1]
    out = pd.DataFrame(
        {
            "Category": grouped.index.astype(str),
            "PrevMonth": prev_month,
            "LatestMonth": latest_month,
            "PrevSpendingCHF": grouped[prev_month].astype(float).values,
            "LatestSpendingCHF": grouped[latest_month].astype(float).values,
        }
    )
    out["ChangeCHF"] = out["LatestSpendingCHF"] - out["PrevSpendingCHF"]
    out["ChangePct"] = out.apply(
        lambda row: (row["ChangeCHF"] / row["PrevSpendingCHF"] * 100.0) if row["PrevSpendingCHF"] > 0 else 0.0,
        axis=1,
    )
    return out.sort_values("ChangeCHF", ascending=False).reset_index(drop=True)


def savings_scenario(
    df: pd.DataFrame,
    target_extra_savings_chf: float,
    max_cut_pct: float = 0.20,
    excluded_categories: list[str] | None = None,
) -> pd.DataFrame:
    """Build a category-level cut plan to reach an extra savings target."""
    work = df.copy()
    if work.empty:
        return pd.DataFrame()

    excluded = {str(item) for item in (excluded_categories or [])}
    excluded.update({"Transfers"})

    work["Month"] = work["Date"].dt.to_period("M").astype(str)
    spend = work[(work["DebitCHF"] > 0) & (~work["Category"].isin(excluded))]
    if spend.empty:
        return pd.DataFrame()

    avg_monthly = (
        spend.groupby(["Category", "Month"], dropna=False)["DebitCHF"]
        .sum()
        .groupby("Category")
        .mean()
        .sort_values(ascending=False)
    )
    if avg_monthly.empty:
        return pd.DataFrame()

    max_cut_pct = min(max(float(max_cut_pct), 0.0), 1.0)
    remaining = max(float(target_extra_savings_chf), 0.0)
    rows: list[dict[str, object]] = []

    for category, avg_spend in avg_monthly.items():
        avg_spend = float(avg_spend)
        max_cut = avg_spend * max_cut_pct
        suggested_cut = min(max_cut, remaining) if remaining > 0 else 0.0
        cut_pct = (suggested_cut / avg_spend * 100.0) if avg_spend > 0 else 0.0
        remaining_after = max(remaining - suggested_cut, 0.0)
        rows.append(
            {
                "Category": str(category),
                "AvgMonthlySpendCHF": round(avg_spend, 2),
                "MaxCutCHF": round(max_cut, 2),
                "SuggestedCutCHF": round(suggested_cut, 2),
                "SuggestedCutPct": round(cut_pct, 2),
                "TargetRemainingCHF": round(remaining_after, 2),
            }
        )
        remaining = remaining_after

    return pd.DataFrame(rows)


def balance_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Account balance trend if statement includes Saldo values."""
    if "Saldo" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["Saldo"] = pd.to_numeric(work["Saldo"], errors="coerce")
    work = work[work["Saldo"].notna()]
    if work.empty:
        return pd.DataFrame()
    grouped = (
        work.sort_values(["Date", "Time"])
        .groupby(["Date", "SourceAccount"])["Saldo"]
        .last()
        .unstack(fill_value=pd.NA)
        .sort_index()
    )
    return grouped
