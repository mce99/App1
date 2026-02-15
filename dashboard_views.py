"""Modular Streamlit page renderers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from metric_guide import METRIC_GUIDE


def _fmt_chf(value: float) -> str:
    return f"{value:,.2f}"


def render_kpis(kpis: dict[str, float]) -> None:
    st.subheader("Snapshot KPIs")

    rows = [
        [
            ("Spending (CHF)", _fmt_chf(kpis["total_spending"])),
            ("Earnings (CHF)", _fmt_chf(kpis["total_earnings"])),
            ("Net cashflow", _fmt_chf(kpis["net_cashflow"])),
            ("Savings rate", f"{kpis['savings_rate']:.1f}%"),
        ],
        [
            ("Transactions", f"{int(kpis['transactions']):,}"),
            ("Active days", f"{int(kpis['active_days']):,}"),
            ("Avg spend / tx", _fmt_chf(kpis["avg_spending"])),
            ("Avg earning / tx", _fmt_chf(kpis["avg_earning"])),
        ],
        [
            ("Avg spend / active day", _fmt_chf(kpis["avg_spending_per_active_day"])),
            ("Avg earn / active day", _fmt_chf(kpis["avg_earnings_per_active_day"])),
            ("Avg tx / active day", f"{kpis['avg_transactions_per_active_day']:.2f}"),
            ("Avg daily net", _fmt_chf(kpis["avg_daily_net"])),
        ],
        [
            ("Avg spend / calendar day", _fmt_chf(kpis["avg_spending_per_calendar_day"])),
            ("Avg earn / calendar day", _fmt_chf(kpis["avg_earnings_per_calendar_day"])),
            ("Largest spend tx", _fmt_chf(kpis["largest_spending"])),
            ("Largest earning tx", _fmt_chf(kpis["largest_earning"])),
        ],
    ]

    for row in rows:
        cols = st.columns(4)
        for idx, (label, value) in enumerate(row):
            cols[idx].metric(label, value)


def render_home(
    kpis: dict[str, float],
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    category_table: pd.DataFrame,
    quality: dict[str, float],
) -> None:
    st.header("Home")
    render_kpis(kpis)

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("### Cumulative overview")
        st.line_chart(daily[["CumulativeEarnings", "CumulativeSpending", "CumulativeNet"]])
    with top_right:
        st.markdown("### Monthly net")
        st.bar_chart(monthly[["Net"]])

    mid_left, mid_right = st.columns(2)
    with mid_left:
        st.markdown("### Monthly earnings vs spending")
        st.area_chart(monthly[["Earnings", "Spending"]])
    with mid_right:
        st.markdown("### Top categories by spending")
        st.bar_chart(category_table.head(10)[["SpendingCHF"]])

    st.markdown("### Data quality signals")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing time %", f"{quality['missing_time_pct']:.1f}%")
    q2.metric("Other category %", f"{quality['other_category_pct']:.1f}%")
    q3.metric("Unknown time-of-day %", f"{quality['unknown_timeofday_pct']:.1f}%")
    q4.metric("Missing currency %", f"{quality['missing_currency_pct']:.1f}%")


def render_cashflow(daily: pd.DataFrame, monthly: pd.DataFrame, velocity: pd.DataFrame) -> None:
    st.header("Cashflow")
    st.caption("Dedicated cashflow page with daily, monthly, and rolling trend views.")

    a, b = st.columns(2)
    with a:
        st.markdown("### Daily net and cumulative net")
        st.line_chart(daily[["Net", "CumulativeNet"]])
    with b:
        st.markdown("### Rolling 7-day averages")
        st.line_chart(velocity[["SpendingMA", "EarningsMA", "NetMA"]])

    c, d = st.columns(2)
    with c:
        st.markdown("### Cumulative spending vs earnings")
        st.line_chart(daily[["CumulativeSpending", "CumulativeEarnings"]])
    with d:
        st.markdown("### Monthly totals")
        st.bar_chart(monthly[["Earnings", "Spending", "Net"]])

    st.markdown("### Daily table")
    st.dataframe(daily, use_container_width=True)


def render_spending(
    category_table: pd.DataFrame,
    top_merchants: pd.DataFrame,
    hourly: pd.DataFrame,
    weekday_avg: pd.DataFrame,
) -> None:
    st.header("Spending")
    st.caption("Deep-dive on where and when money goes out.")

    a, b = st.columns(2)
    with a:
        st.markdown("### Spending by category")
        st.bar_chart(category_table[["SpendingCHF"]].head(20))
    with b:
        st.markdown("### Category spending share (%)")
        st.bar_chart(category_table[["SpendingSharePct"]].head(20))

    c, d = st.columns(2)
    with c:
        st.markdown("### Spend by hour of day")
        st.bar_chart(hourly[["Spending"]])
    with d:
        st.markdown("### Avg spend by hour")
        st.line_chart(hourly[["AvgSpending"]])

    e, f = st.columns(2)
    with e:
        st.markdown("### Avg spending by weekday")
        st.bar_chart(weekday_avg[["Spending"]])
    with f:
        st.markdown("### Top merchants by spending")
        st.dataframe(top_merchants, use_container_width=True)


def render_earnings(
    category_table: pd.DataFrame,
    income_sources: pd.DataFrame,
    hourly: pd.DataFrame,
    weekday_avg: pd.DataFrame,
) -> None:
    st.header("Earnings")
    st.caption("Understand where incoming money comes from and when it arrives.")

    a, b = st.columns(2)
    with a:
        st.markdown("### Earnings by category")
        st.bar_chart(category_table[["EarningsCHF"]].head(20))
    with b:
        st.markdown("### Category earnings share (%)")
        st.bar_chart(category_table[["EarningsSharePct"]].head(20))

    c, d = st.columns(2)
    with c:
        st.markdown("### Earnings by hour of day")
        st.bar_chart(hourly[["Earnings"]])
    with d:
        st.markdown("### Avg earnings by hour")
        st.line_chart(hourly[["AvgEarnings"]])

    e, f = st.columns(2)
    with e:
        st.markdown("### Avg earnings by weekday")
        st.bar_chart(weekday_avg[["Earnings"]])
    with f:
        st.markdown("### Top income sources")
        if income_sources.empty:
            st.info("No positive credited transactions in current filters.")
        else:
            st.dataframe(income_sources, use_container_width=True)


def render_behavior(hourly: pd.DataFrame, weekday_avg: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.header("Behavior")
    st.caption("Timing and rhythm of your activity.")

    a, b = st.columns(2)
    with a:
        st.markdown("### Transactions by time-of-day bucket")
        st.bar_chart(filtered["TimeOfDay"].value_counts().sort_index())
    with b:
        st.markdown("### Transactions by hour")
        st.bar_chart(hourly[["Transactions"]])

    c, d = st.columns(2)
    with c:
        st.markdown("### Net by hour")
        st.line_chart(hourly[["Net"]])
    with d:
        st.markdown("### Avg net by weekday")
        st.bar_chart(weekday_avg[["Net"]])


def render_subscriptions(recurring: pd.DataFrame, budget_table: pd.DataFrame) -> None:
    st.header("Plans & Recurring")
    st.caption("Recurring-candidate detector plus budget control.")

    st.markdown("### Recurring transaction candidates")
    if recurring.empty:
        st.info("No recurring monthly patterns detected in current filters.")
    else:
        st.dataframe(recurring, use_container_width=True)

    st.markdown("### Budget progress")
    st.dataframe(budget_table, use_container_width=True)


def render_data_explorer(filtered: pd.DataFrame, source_context: pd.DataFrame) -> None:
    st.header("Data Explorer")
    st.caption("Traceability and raw details for QA and troubleshooting.")

    with st.expander("Source file context", expanded=True):
        st.dataframe(source_context, use_container_width=True)

    st.markdown("### Ordered transaction ledger")
    st.dataframe(filtered, use_container_width=True, height=520)


def render_metric_guide() -> None:
    st.header("Metric Guide")
    st.caption("Definitions and formulas behind each KPI.")
    st.dataframe(pd.DataFrame(METRIC_GUIDE), use_container_width=True, height=680)
