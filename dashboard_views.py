"""Modular Streamlit page renderers."""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

from analytics import chart_builder_dataset, spending_heatmap_matrix
from metric_guide import METRIC_GUIDE


def _fmt_chf(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_delta(value: float) -> str:
    if value > 0:
        return f"+{value:,.1f}%"
    return f"{value:,.1f}%"


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
    period_table: pd.DataFrame,
) -> None:
    st.header("Home")
    render_kpis(kpis)

    st.markdown("### Period vs prior window")
    if period_table.empty:
        st.info("Not enough data to compare with the immediately prior period.")
    else:
        focus = period_table.set_index("Metric")
        c1, c2, c3 = st.columns(3)
        if "Net cashflow (CHF)" in focus.index:
            row = focus.loc["Net cashflow (CHF)"]
            c1.metric(
                "Net cashflow vs prior",
                _fmt_chf(float(row["CurrentValue"])),
                _fmt_delta(float(row["DeltaPct"])),
            )
        if "Spending (CHF)" in focus.index:
            row = focus.loc["Spending (CHF)"]
            c2.metric(
                "Spending vs prior",
                _fmt_chf(float(row["CurrentValue"])),
                _fmt_delta(float(row["DeltaPct"])),
            )
        if "Earnings (CHF)" in focus.index:
            row = focus.loc["Earnings (CHF)"]
            c3.metric(
                "Earnings vs prior",
                _fmt_chf(float(row["CurrentValue"])),
                _fmt_delta(float(row["DeltaPct"])),
            )
        st.dataframe(period_table, use_container_width=True, hide_index=True)

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

    st.markdown("### Weekday-hour heatmap")
    heat_metric = st.selectbox(
        "Heatmap metric",
        ["Spending", "Earnings", "Net", "Transactions"],
        index=0,
        key="behavior_heatmap_metric",
    )
    matrix = spending_heatmap_matrix(filtered, value_metric=heat_metric)

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(matrix.round(2), use_container_width=True)
    with right:
        hotspots = (
            matrix.stack()
            .reset_index(name="Value")
            .rename(columns={"level_0": "Weekday", "level_1": "Hour"})
        )
        hotspots = hotspots[hotspots["Value"].abs() > 0].copy()
        if hotspots.empty:
            st.info("No heatmap hotspots for current filters.")
        else:
            hotspots = hotspots.sort_values("Value", ascending=False).head(12)
            hotspots["HourBlock"] = hotspots["Hour"].apply(lambda h: f"{int(h):02d}:00")
            st.markdown("### Top hotspots")
            st.dataframe(
                hotspots[["Weekday", "HourBlock", "Value"]],
                use_container_width=True,
                hide_index=True,
            )


def render_chart_builder(filtered: pd.DataFrame) -> None:
    st.header("Chart Builder")
    st.caption("Create custom charts from your transactions with flexible dimensions and metrics.")

    dimension_candidates = [
        "Category",
        "Merchant",
        "MerchantNormalized",
        "SourceAccount",
        "TimeOfDay",
        "Währung",
        "Location",
        "TransferDirection",
        "IsTransfer",
    ]
    dimensions = [col for col in dimension_candidates if col in filtered.columns]
    x_options = ["Date", "Month", "Weekday", "Hour"] + dimensions
    split_options = ["None"] + dimensions

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_axis = st.selectbox("X axis", x_options, index=0)
        if x_axis in {"Date", "Month"}:
            interval_default = 0 if x_axis == "Date" else 2
            date_interval = st.selectbox(
                "Time interval",
                ["Daily", "Weekly", "Monthly"],
                index=interval_default,
                help="Use Daily for the most detailed spending view.",
            )
        else:
            date_interval = "Daily"
        chart_type = st.selectbox("Chart type", ["Line", "Bar", "Area"], index=0)
    with c2:
        metric = st.selectbox("Metric", ["Spending", "Earnings", "Net", "Transactions"], index=0)
        split_by = st.selectbox("Split by", split_options, index=0)
    with c3:
        if metric == "Transactions":
            aggregation = st.selectbox("Aggregation", ["Sum"], index=0, disabled=True)
        else:
            aggregation = st.selectbox(
                "Aggregation",
                ["Sum", "Average", "Median", "Count"],
                index=0,
            )
        top_n = st.slider("Top items", min_value=5, max_value=50, value=20, step=1)
    with c4:
        cumulative = st.checkbox("Cumulative view", value=False)
        include_transfers = True
        if "IsTransfer" in filtered.columns:
            include_transfers = st.checkbox(
                "Include transfers",
                value=True,
                help="Disable to focus on external spending/earnings behavior.",
            )

    chart_data = chart_builder_dataset(
        filtered,
        x_axis=x_axis,
        metric=metric,
        aggregation=aggregation,
        split_by=split_by,
        top_n=int(top_n),
        cumulative=bool(cumulative),
        date_interval=date_interval,
        include_transfers=bool(include_transfers),
    )
    if chart_data.empty:
        st.info("No data available for this chart configuration.")
        return

    st.markdown("### Preview")
    if chart_type == "Bar":
        st.bar_chart(chart_data)
    elif chart_type == "Area":
        st.area_chart(chart_data)
    else:
        st.line_chart(chart_data)

    st.markdown("### Chart data")
    table = chart_data.reset_index()
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download chart data (.csv)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="chart_builder_export.csv",
        mime="text/csv",
    )


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
    query = st.text_input(
        "Search ledger (merchant/category/notes/account)",
        value="",
        key="data_explorer_query",
    ).strip().lower()
    max_cap = max(1, min(int(len(filtered)), 5000))
    min_rows = 1 if max_cap < 50 else 50
    step = 1 if max_cap < 50 else 50
    default_rows = min(1500, max_cap)
    rows_to_show = st.slider(
        "Rows to display",
        min_value=min_rows,
        max_value=max_cap,
        value=default_rows,
        step=step,
        key="data_explorer_rows",
    )

    view = filtered.copy()
    if query:
        search_cols = [
            col
            for col in [
                "Merchant",
                "Category",
                "Beschreibung1",
                "Beschreibung2",
                "Beschreibung3",
                "Fussnoten",
                "SourceAccount",
            ]
            if col in view.columns
        ]
        if search_cols:
            text_blob = (
                view[search_cols]
                .fillna("")
                .astype(str)
                .agg(" | ".join, axis=1)
                .str.lower()
            )
            view = view[text_blob.str.contains(query, na=False)]

    st.caption(f"Showing {min(len(view), rows_to_show):,} of {len(view):,} matching rows.")
    st.dataframe(view.head(rows_to_show), use_container_width=True, height=520)


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


def render_report_pack(summary_md: str, zip_bytes: bytes, executive_pdf: bytes | None = None) -> None:
    st.header("Report Pack")
    st.caption("One-click export of summary and raw aggregates.")
    st.markdown(summary_md)
    st.download_button(
        "Download report pack (.zip)",
        data=zip_bytes,
        file_name="pulseledger_report_pack.zip",
        mime="application/zip",
    )
    if executive_pdf:
        st.download_button(
            "Download executive brief (.pdf)",
            data=executive_pdf,
            file_name="pulseledger_executive_brief.pdf",
            mime="application/pdf",
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
    opportunity_table: pd.DataFrame,
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

    st.markdown("### Opportunity scanner")
    if opportunity_table.empty:
        st.info("No material savings opportunities detected in current filters.")
    else:
        top_monthly = float(opportunity_table["PotentialMonthlySavingsCHF"].sum())
        top_annual = float(opportunity_table["PotentialAnnualSavingsCHF"].sum())
        o1, o2 = st.columns(2)
        o1.metric("Potential monthly savings (top levers)", f"{top_monthly:,.2f}")
        o2.metric("Potential annual savings (top levers)", f"{top_annual:,.2f}")
        st.dataframe(opportunity_table, use_container_width=True, hide_index=True)

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


def render_agent_console(action_plan: pd.DataFrame, ingestion_quality: pd.DataFrame) -> None:
    st.header("Agent Console")
    st.caption("Automatic operations board for what to fix and optimize next.")

    st.markdown("### Prioritized action queue")
    st.dataframe(action_plan, use_container_width=True, hide_index=True)

    st.markdown("### Upload and ingestion quality")
    if ingestion_quality.empty:
        st.info("No ingestion diagnostics available yet.")
    else:
        st.dataframe(ingestion_quality, use_container_width=True, hide_index=True)
