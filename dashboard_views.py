"""Modular Streamlit page renderers."""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
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


def render_subscriptions(
    recurring: pd.DataFrame, budget_table: pd.DataFrame, goals_table: pd.DataFrame | None = None
) -> None:
    st.header("Plans & Recurring")
    st.caption("Recurring-candidate detector plus budget control.")

    st.markdown("### Recurring transaction candidates")
    if recurring.empty:
        st.info("No recurring monthly patterns detected in current filters.")
    else:
        st.dataframe(recurring, use_container_width=True)

    st.markdown("### Budget progress")
    st.dataframe(budget_table, use_container_width=True)

    st.markdown("### Goals progress")
    if goals_table is None or goals_table.empty:
        st.info("No goals configured.")
    else:
        st.dataframe(goals_table, use_container_width=True)


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


def render_accounts(account_table: pd.DataFrame, transfers: pd.DataFrame) -> None:
    st.header("Accounts")
    st.caption("Track spending, earnings, and transfers per account.")
    st.dataframe(account_table, use_container_width=True)
    st.markdown("### Transfer ledger")
    if transfers.empty:
        st.info("No transfer-like transactions detected for current filters.")
    else:
        st.dataframe(
            transfers[
                [
                    "Date",
                    "Time",
                    "SourceAccount",
                    "CounterpartyAccount",
                    "TransferDirection",
                    "DebitCHF",
                    "CreditCHF",
                    "TransferConfidence",
                    "Merchant",
                ]
            ],
            use_container_width=True,
        )


def render_forecast(forecast_table: pd.DataFrame) -> None:
    st.header("Forecast")
    st.caption("Projected 30/60/90 day cashflow from trend + recurring signals.")
    if forecast_table.empty:
        st.info("Not enough data to generate forecast.")
        return
    st.dataframe(forecast_table, use_container_width=True)
    st.bar_chart(forecast_table.set_index("HorizonDays")[["ExpectedSpendingCHF", "ExpectedEarningsCHF"]])
    st.line_chart(forecast_table.set_index("HorizonDays")[["ExpectedNetCHF"]])


def render_anomalies(anomalies: pd.DataFrame, dupes: pd.DataFrame) -> None:
    st.header("Anomalies")
    st.caption("Unusual transactions and possible duplicates.")
    st.markdown("### Suspicious spend anomalies")
    if anomalies.empty:
        st.success("No major anomalies flagged in current filters.")
    else:
        st.dataframe(anomalies, use_container_width=True)

    st.markdown("### Possible duplicate transactions")
    if dupes.empty:
        st.success("No duplicate candidates found.")
    else:
        st.dataframe(dupes, use_container_width=True)


def render_data_health(health_table: pd.DataFrame, quality: dict[str, float]) -> None:
    st.header("Data Health")
    st.caption("Visibility into data completeness and reliability.")
    st.dataframe(health_table, use_container_width=True)
    cols = st.columns(4)
    cols[0].metric("Missing time %", f"{quality['missing_time_pct']:.1f}%")
    cols[1].metric("Other category %", f"{quality['other_category_pct']:.1f}%")
    cols[2].metric("Unknown time-of-day %", f"{quality['unknown_timeofday_pct']:.1f}%")
    cols[3].metric("Missing currency %", f"{quality['missing_currency_pct']:.1f}%")


def render_report_pack(summary_md: str, zip_bytes: bytes) -> None:
    st.header("Report Pack")
    st.caption("One-click export of summary and raw aggregates.")
    st.markdown(summary_md)
    st.download_button(
        "Download report pack (.zip)",
        data=zip_bytes,
        file_name="pulseledger_report_pack.zip",
        mime="application/zip",
    )


def render_portfolio(
    stock_positions: pd.DataFrame,
    wallet_positions: pd.DataFrame,
    totals: dict[str, float],
    holdings: pd.DataFrame,
    quote_currency: str,
) -> None:
    st.header("Portfolio")
    st.caption("Stocks + on-chain wallet balances in one place.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stock value", f"{totals['stock_value']:,.2f}")
    k2.metric("Wallet value", f"{totals['wallet_value']:,.2f} {quote_currency.upper()}")
    k3.metric("Total tracked value", f"{totals['total_value']:,.2f}")
    k4.metric("Known unrealized PnL", f"{totals['total_pnl_known']:,.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown("### Stock positions")
        if stock_positions.empty:
            st.info("No stock positions yet.")
        else:
            st.dataframe(stock_positions, use_container_width=True)
    with right:
        st.markdown("### Wallet balances")
        if wallet_positions.empty:
            st.info("No wallet balances yet.")
        else:
            st.dataframe(wallet_positions, use_container_width=True)

    st.markdown("### Holdings mix")
    if holdings.empty:
        st.info("No holdings data to chart.")
    else:
        st.bar_chart(holdings.set_index("Label")[["Value"]])


def render_insights(
    salary_info: dict[str, object],
    benchmark_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    merchant_table: pd.DataFrame,
    balance_table: pd.DataFrame,
) -> None:
    st.header("Insights & Optimization")
    st.caption("Understand behavior, compare against benchmarks, and identify concrete spend reductions.")

    avg_salary = float(salary_info.get("avg_monthly_salary", 0.0) or 0.0)
    st.metric("Estimated avg monthly salary", f"{avg_salary:,.2f}")

    salary_monthly = salary_info.get("salary_monthly", pd.DataFrame())
    salary_sources = salary_info.get("salary_sources", pd.DataFrame())

    left, right = st.columns(2)
    with left:
        st.markdown("### Salary trend")
        if isinstance(salary_monthly, pd.DataFrame) and not salary_monthly.empty:
            st.line_chart(salary_monthly.set_index("Month")[["EstimatedSalaryCHF"]])
        else:
            st.info("No recurring salary pattern confidently detected yet.")
    with right:
        st.markdown("### Salary sources")
        if isinstance(salary_sources, pd.DataFrame) and not salary_sources.empty:
            st.dataframe(salary_sources, use_container_width=True)
        else:
            st.info("No stable salary source identified.")

    st.markdown("### Benchmark comparison")
    st.dataframe(benchmark_table, use_container_width=True)

    st.markdown("### Spend reduction recommendations")
    st.dataframe(recommendations, use_container_width=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("### Merchant insights")
        st.dataframe(merchant_table, use_container_width=True)
    with lower_right:
        st.markdown("### Balance trend")
        if not balance_table.empty:
            st.line_chart(balance_table)
        else:
            st.info("No running balance data available from the current files.")


def render_spending_map(map_points: pd.DataFrame) -> None:
    st.header("Spending Map (Switzerland)")
    st.caption("Geographic concentration of spending based on transaction location text.")

    if map_points.empty:
        st.info("No geocoded spending points available for current filters.")
        return

    st.dataframe(map_points, use_container_width=True)

    center_lat = float(map_points["lat"].mean())
    center_lon = float(map_points["lon"].mean())
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=7, pitch=35)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_points,
        get_position="[lon, lat]",
        get_radius="SpendingCHF * 8",
        radius_min_pixels=4,
        radius_max_pixels=60,
        get_fill_color="[20, 120, 255, 170]",
        pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{Location}\nSpending CHF: {SpendingCHF}\nTransactions: {Transactions}"},
        map_style=None,
    )
    st.pydeck_chart(deck, use_container_width=True)
