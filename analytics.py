"""Analytics helpers for dashboard KPIs and charts."""

from __future__ import annotations

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


def build_report_pack(df: pd.DataFrame, kpis: dict[str, float], monthly: pd.DataFrame) -> tuple[str, bytes]:
    """Build markdown summary plus zip of raw and monthly outputs."""
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    markdown = (
        f"# PulseLedger Report\n\n"
        f"- Period: {min_date} to {max_date}\n"
        f"- Transactions: {int(kpis['transactions']):,}\n"
        f"- Spending (CHF): {kpis['total_spending']:,.2f}\n"
        f"- Earnings (CHF): {kpis['total_earnings']:,.2f}\n"
        f"- Net cashflow (CHF): {kpis['net_cashflow']:,.2f}\n"
        f"- Savings rate: {kpis['savings_rate']:.1f}%\n"
    )

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.md", markdown)
        zf.writestr("transactions.csv", df.to_csv(index=False))
        zf.writestr("monthly.csv", monthly.to_csv())
        zf.writestr("kpis.json", json.dumps(kpis, indent=2))
    return markdown, output.getvalue()


def _spend_bucket(row: pd.Series) -> str:
    category = str(row.get("Category", "")).strip()
    merchant = str(
        row.get("MerchantNormalized", row.get("Merchant", ""))
    ).upper()
    if category == "Transfers":
        return "Transfers"
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
