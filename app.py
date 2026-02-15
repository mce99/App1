"""Streamlit app for multi-file transaction analysis with rich spending/earnings insights."""

from __future__ import annotations

import datetime
import json

import pandas as pd
import streamlit as st

from analytics import (
    apply_currency_conversion,
    budget_progress,
    calculate_kpis,
    daily_net_cashflow,
    filter_by_date_range,
    merchant_summary,
    monthly_cashflow,
    recurring_transaction_candidates,
)
from categorization import DEFAULT_KEYWORD_MAP, assign_categories
from parsing import SUPPORTED_EXTENSIONS, classify_time_of_day, merge_transactions

st.set_page_config(page_title="PulseLedger", page_icon="\U0001f4c8", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }
        .stApp {
            background:
              radial-gradient(1200px 500px at 10% 0%, rgba(18, 203, 255, 0.16), transparent 60%),
              radial-gradient(900px 500px at 90% 5%, rgba(84, 255, 174, 0.15), transparent 55%),
              linear-gradient(180deg, #f4fbff 0%, #eff7ff 100%);
        }
        .hero {
            padding: 1rem 1.2rem;
            border: 1px solid rgba(55, 95, 175, 0.2);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(6px);
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(57, 90, 165, 0.25);
            border-radius: 14px;
            padding: 0.4rem 0.6rem;
            box-shadow: 0 8px 22px rgba(37, 63, 120, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">PulseLedger</h1>
            <p style="margin:0.4rem 0 0 0;">
                Multi-file spending and earnings intelligence for semicolon-delimited bank exports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _parse_json_dict(json_text: str, fallback: dict) -> dict:
    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            return fallback
        return parsed
    except Exception:
        return fallback


def _add_manual_transaction(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Add manual transaction"):
        with st.form(key="manual_form"):
            m_date = st.date_input("Date", value=datetime.date.today())
            m_time = st.time_input("Time", value=datetime.datetime.now().time())
            m_currency = st.text_input("Currency", value="CHF")
            m_debit = st.number_input("Debit", value=0.0)
            m_credit = st.number_input("Credit", value=0.0)
            m_desc1 = st.text_input("Beschreibung1 (Merchant)")
            m_desc2 = st.text_input("Beschreibung2")
            m_desc3 = st.text_input("Beschreibung3 (Location)")
            m_notes = st.text_input("Fussnoten")
            submit = st.form_submit_button("Add transaction")

        if not submit:
            return df

        temp = pd.DataFrame(
            [
                {
                    "Abschlussdatum": m_date,
                    "Abschlusszeit": m_time.strftime("%H:%M:%S"),
                    "Währung": m_currency,
                    "Belastung": m_debit,
                    "Gutschrift": m_credit,
                    "Beschreibung1": m_desc1,
                    "Beschreibung2": m_desc2,
                    "Beschreibung3": m_desc3,
                    "Fussnoten": m_notes,
                    "Debit": max(float(m_debit), 0.0),
                    "Credit": max(float(m_credit), 0.0),
                    "Merchant": m_desc1.strip(),
                    "Location": m_desc3.strip() or m_desc2.strip(),
                    "Date": pd.to_datetime(m_date),
                    "Time": m_time.strftime("%H:%M:%S"),
                    "TimeOfDay": classify_time_of_day(m_time.strftime("%H:%M:%S")),
                    "SortDateTime": pd.to_datetime(f"{m_date.isoformat()} {m_time.strftime('%H:%M:%S')}"),
                    "SourceFile": "Manual entry",
                }
            ]
        )

        merged = pd.concat([df, temp], ignore_index=True)
        return merged.sort_values(["SortDateTime", "Date", "Time"], na_position="last").reset_index(
            drop=True
        )


def _configure_timeframe_controls(df: pd.DataFrame) -> tuple[datetime.date, datetime.date]:
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    key = "timeframe_range"
    if key not in st.session_state:
        st.session_state[key] = (min_date, max_date)
    else:
        start, end = st.session_state[key]
        if start < min_date or end > max_date:
            st.session_state[key] = (min_date, max_date)

    st.sidebar.subheader("Timeframe")
    st.sidebar.caption("All KPIs and charts react to this range.")
    st.sidebar.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=st.session_state[key],
        key=key,
    )

    if st.sidebar.button("Reset to full range"):
        st.session_state[key] = (min_date, max_date)
        st.rerun()

    return st.session_state[key]


def main() -> None:
    _inject_styles()
    _render_header()

    st.caption(
        "Accepts .csv, .xlsx, .xls | CSV must be semicolon-delimited (;) | Supports multi-file uploads."
    )

    uploaded_files = st.file_uploader(
        "Upload one or many statement files",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more bank files to begin analysis.")
        return

    st.sidebar.header("Ingestion")
    drop_duplicates = st.sidebar.checkbox(
        "Auto-remove duplicate transactions across files",
        value=True,
        help="Useful when exported file ranges overlap.",
    )

    try:
        df = merge_transactions(uploaded_files, drop_duplicates=drop_duplicates)
    except Exception as exc:
        st.error(f"Could not read file(s): {exc}")
        return

    if df.empty or df["Date"].dropna().empty:
        st.warning("No valid transactions found. Check the exported format and delimiter (;).")
        return

    df = _add_manual_transaction(df)

    st.success(
        f"Loaded {len(df):,} transactions from {len(uploaded_files)} file(s). "
        f"Range: {df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    st.subheader("Currency conversion")
    conv_default = {"CHF": 1.0, "EUR": 0.96, "USD": 0.89}
    conv_json = st.text_area(
        "CHF per currency unit",
        value=json.dumps(conv_default, indent=2),
        help="Example: {\"CHF\": 1.0, \"EUR\": 0.96}",
    )
    conv_rates = _parse_json_dict(conv_json, conv_default)
    if not isinstance(conv_rates, dict):
        st.error("Invalid currency JSON. Using default rates.")
        conv_rates = conv_default

    st.subheader("Category customization")
    cat_json = st.text_area("Keyword map JSON", value=json.dumps(DEFAULT_KEYWORD_MAP, indent=2))
    raw_map = _parse_json_dict(cat_json, DEFAULT_KEYWORD_MAP)
    keyword_map = {
        str(key): [str(item).upper() for item in values]
        for key, values in raw_map.items()
        if isinstance(values, list)
    }
    if not keyword_map:
        keyword_map = DEFAULT_KEYWORD_MAP

    enriched = assign_categories(apply_currency_conversion(df, conv_rates), keyword_map)

    start_date, end_date = _configure_timeframe_controls(enriched)

    st.sidebar.subheader("Additional filters")
    selected_sources = st.sidebar.multiselect(
        "Files",
        options=sorted(enriched["SourceFile"].dropna().unique().tolist()),
        default=sorted(enriched["SourceFile"].dropna().unique().tolist()),
    )
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=sorted(enriched["Category"].dropna().unique().tolist()),
        default=sorted(enriched["Category"].dropna().unique().tolist()),
    )

    filtered = filter_by_date_range(enriched, start_date, end_date)
    filtered = filtered[filtered["SourceFile"].isin(selected_sources)]
    filtered = filtered[filtered["Category"].isin(selected_categories)]

    if filtered.empty:
        st.warning("No transactions in the selected filters/timeframe.")
        return

    kpis = calculate_kpis(filtered)

    st.subheader("Comprehensive overview")
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Spending (CHF)", f"{kpis['total_spending']:,.2f}")
    kpi_cols[1].metric("Earnings (CHF)", f"{kpis['total_earnings']:,.2f}")
    kpi_cols[2].metric("Net cashflow", f"{kpis['net_cashflow']:,.2f}")
    kpi_cols[3].metric("Transactions", f"{kpis['transactions']:,}")
    kpi_cols[4].metric("Avg spend / tx", f"{kpis['avg_spending']:,.2f}")
    kpi_cols[5].metric("Savings rate", f"{kpis['savings_rate']:.1f}%")

    monthly = monthly_cashflow(filtered)
    daily = daily_net_cashflow(filtered)
    top_merchants = merchant_summary(filtered, top_n=20)
    recurring = recurring_transaction_candidates(filtered)

    tab_overview, tab_categories, tab_merchants, tab_data = st.tabs(
        ["Overview", "Categories & Budget", "Merchants & Recurring", "Transactions"]
    )

    with tab_overview:
        st.markdown("### Cashflow trends")
        st.area_chart(monthly[["Earnings", "Spending"]])
        st.bar_chart(monthly[["Net"]])

        st.markdown("### Daily net and cumulative net")
        st.line_chart(daily[["Net", "CumulativeNet"]])

        st.markdown("### Time-of-day activity")
        st.bar_chart(filtered["TimeOfDay"].value_counts().sort_index())

    with tab_categories:
        spend_by_cat = filtered.groupby("Category")["DebitCHF"].sum().sort_values(ascending=False)
        earn_by_cat = filtered.groupby("Category")["CreditCHF"].sum().sort_values(ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Spending by category")
            st.bar_chart(spend_by_cat)
        with c2:
            st.markdown("### Earnings by category")
            st.bar_chart(earn_by_cat)

        st.markdown("### Budget tracker")
        default_budget = {
            category: round(kpis["total_spending"] / max(len(spend_by_cat), 1), 2)
            for category in spend_by_cat.index
        }
        budget_json = st.text_area(
            "Monthly budget by category (JSON)",
            value=json.dumps(default_budget, indent=2),
            key="budget_text",
        )
        budget_dict = _parse_json_dict(budget_json, default_budget)
        budget_table = budget_progress(filtered, budget_dict)
        st.dataframe(budget_table)

    with tab_merchants:
        st.markdown("### Top merchants by spending")
        st.dataframe(top_merchants, use_container_width=True)

        st.markdown("### Recurring transaction signals")
        st.caption(
            "Inspired by recurring/subscription detection in modern finance apps: shows likely monthly patterns."
        )
        if recurring.empty:
            st.info("No recurring monthly patterns detected for the selected timeframe.")
        else:
            st.dataframe(recurring, use_container_width=True)

    with tab_data:
        st.markdown("### Ordered transaction ledger")
        preview_cols = [
            "Date",
            "Time",
            "SourceFile",
            "Merchant",
            "Location",
            "Währung",
            "Debit",
            "Credit",
            "DebitCHF",
            "CreditCHF",
            "Category",
        ]
        st.dataframe(filtered[preview_cols], use_container_width=True)
        st.download_button(
            "Download filtered data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_transactions.csv",
            mime="text/csv",
        )

    st.markdown("### Product-inspiration references")
    st.markdown(
        "- [Monarch Money features](https://www.monarchmoney.com/features)\n"
        "- [Copilot Money dashboard](https://copilot.money/dashboard)\n"
        "- [YNAB help center](https://support.ynab.com/en_us/getting-started-with-ynab-an-overview-Sy_l6mQou)\n"
        "- [Firefly III docs](https://docs.firefly-iii.org/)"
    )


if __name__ == "__main__":
    main()
