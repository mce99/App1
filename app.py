"""PulseLedger Streamlit entrypoint with modular page navigation."""

from __future__ import annotations

import datetime
import json

import pandas as pd
import streamlit as st

from analytics import (
    apply_currency_conversion,
    budget_progress,
    calculate_kpis,
    category_breakdown,
    daily_net_cashflow,
    filter_by_date_range,
    hourly_spending_profile,
    income_source_summary,
    merchant_summary,
    monthly_cashflow,
    quality_indicators,
    recurring_transaction_candidates,
    spending_velocity,
    weekday_average_cashflow,
)
from categorization import DEFAULT_KEYWORD_MAP, assign_categories
from dashboard_views import (
    render_behavior,
    render_cashflow,
    render_data_explorer,
    render_earnings,
    render_home,
    render_metric_guide,
    render_spending,
    render_subscriptions,
)
from parsing import SUPPORTED_EXTENSIONS, classify_time_of_day, merge_transactions

st.set_page_config(page_title="PulseLedger", page_icon="\U0001f4ca", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        .stApp {
            background:
              radial-gradient(1200px 420px at 12% 0%, rgba(40, 214, 255, 0.15), transparent 58%),
              radial-gradient(1000px 520px at 90% 0%, rgba(95, 255, 163, 0.14), transparent 62%),
              linear-gradient(180deg, #f6fbff 0%, #eef6ff 100%);
        }
        .hero {
            margin-bottom: 0.6rem;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(30, 80, 145, 0.23);
            border-radius: 14px;
            background: rgba(255,255,255,0.82);
            box-shadow: 0 16px 36px rgba(42, 77, 140, 0.12);
        }
        .hero h1 {
            margin: 0;
            letter-spacing: 0.3px;
        }
        .hero p {
            margin: 0.35rem 0 0 0;
            color: #244674;
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.90);
            border: 1px solid rgba(45, 88, 162, 0.25);
            border-radius: 12px;
            padding: 0.45rem 0.6rem;
            box-shadow: 0 8px 20px rgba(40, 68, 120, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>PulseLedger</h1>
          <p>Modular transaction intelligence for UBS-style exports. Navigate by sections from the sidebar.</p>
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
    with st.sidebar.expander("Quick add transaction", expanded=False):
        with st.form(key="manual_form"):
            m_date = st.date_input("Date", value=datetime.date.today())
            m_time = st.time_input("Time", value=datetime.datetime.now().time())
            m_currency = st.text_input("Currency", value="CHF")
            m_debit = st.number_input("Debit", value=0.0)
            m_credit = st.number_input("Credit", value=0.0)
            m_desc1 = st.text_input("Beschreibung1 (Merchant)")
            m_desc2 = st.text_input("Beschreibung2")
            m_desc3 = st.text_input("Beschreibung3")
            m_notes = st.text_input("Fussnoten")
            submit = st.form_submit_button("Add")

        if not submit:
            return df

        time_value = m_time.strftime("%H:%M:%S")
        new_row = {
            "Abschlussdatum": m_date,
            "Abschlusszeit": time_value,
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
            "Time": time_value,
            "TimeOfDay": classify_time_of_day(time_value),
            "SortDateTime": pd.to_datetime(f"{m_date.isoformat()} {time_value}"),
            "SourceFile": "Manual entry",
        }
        out = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return out.sort_values(["SortDateTime", "Date", "Time"], na_position="last").reset_index(
            drop=True
        )


def _build_statement_context(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("SourceFile", dropna=False)
        .agg(
            LoadedTransactions=("SourceFile", "size"),
            LoadedFrom=("Date", "min"),
            LoadedTo=("Date", "max"),
        )
        .reset_index()
    )

    optional_cols = [
        "StatementAccountNumber",
        "StatementIBAN",
        "StatementFrom",
        "StatementTo",
        "StatementCurrency",
        "StatementTransactions",
    ]
    for col in optional_cols:
        if col in df.columns:
            summary[col] = df.groupby("SourceFile", dropna=False)[col].first().reset_index(drop=True)

    return summary


def _init_timeframe(df: pd.DataFrame) -> tuple[datetime.date, datetime.date]:
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    if "timeframe_range" not in st.session_state:
        st.session_state["timeframe_range"] = (min_date, max_date)
    else:
        start, end = st.session_state["timeframe_range"]
        if start < min_date or end > max_date:
            st.session_state["timeframe_range"] = (min_date, max_date)

    preset = st.sidebar.selectbox(
        "Timeframe preset",
        ["Custom", "Last 30 days", "Last 90 days", "Last 365 days", "Year to date", "Full range"],
        index=0,
    )

    if preset != "Custom":
        today = max_date
        if preset == "Last 30 days":
            st.session_state["timeframe_range"] = (max(min_date, today - datetime.timedelta(days=29)), today)
        elif preset == "Last 90 days":
            st.session_state["timeframe_range"] = (max(min_date, today - datetime.timedelta(days=89)), today)
        elif preset == "Last 365 days":
            st.session_state["timeframe_range"] = (max(min_date, today - datetime.timedelta(days=364)), today)
        elif preset == "Year to date":
            year_start = datetime.date(today.year, 1, 1)
            st.session_state["timeframe_range"] = (max(min_date, year_start), today)
        elif preset == "Full range":
            st.session_state["timeframe_range"] = (min_date, max_date)

    st.sidebar.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=st.session_state["timeframe_range"],
        key="timeframe_range",
    )

    if st.sidebar.button("Reset filters and timeframe"):
        st.session_state["timeframe_range"] = (min_date, max_date)
        st.rerun()

    return st.session_state["timeframe_range"]


def _prepare_enriched_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    st.sidebar.header("Data Setup")
    uploaded_files = st.sidebar.file_uploader(
        "Upload statements",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="CSV must be semicolon-delimited (;).",
    )

    if not uploaded_files:
        st.info("Upload one or more statement files from the sidebar to start.")
        return None, None

    drop_duplicates = st.sidebar.checkbox(
        "Auto-remove duplicates across files",
        value=True,
        help="Recommended if file exports overlap in date range.",
    )

    try:
        df = merge_transactions(uploaded_files, drop_duplicates=drop_duplicates)
    except Exception as exc:
        st.error(f"Could not read file(s): {exc}")
        return None, None

    if df.empty or df["Date"].dropna().empty:
        st.warning("No valid transactions found. Check export format and delimiter (;).")
        return None, None

    df = _add_manual_transaction(df)

    with st.sidebar.expander("Currency rates (JSON)", expanded=False):
        conv_default = {"CHF": 1.0, "EUR": 0.96, "USD": 0.89}
        conv_json = st.text_area("CHF per unit", value=json.dumps(conv_default, indent=2), key="conv_json")
        conv_rates = _parse_json_dict(conv_json, conv_default)

    with st.sidebar.expander("Category map (JSON)", expanded=False):
        cat_json = st.text_area(
            "Keyword map",
            value=json.dumps(DEFAULT_KEYWORD_MAP, indent=2),
            key="cat_json",
            height=240,
        )
        raw_map = _parse_json_dict(cat_json, DEFAULT_KEYWORD_MAP)

    keyword_map = {
        str(key): [str(item).upper() for item in values]
        for key, values in raw_map.items()
        if isinstance(values, list)
    }
    if not keyword_map:
        keyword_map = DEFAULT_KEYWORD_MAP

    enriched = assign_categories(apply_currency_conversion(df, conv_rates), keyword_map)
    source_context = _build_statement_context(enriched)

    st.sidebar.success(
        f"Loaded {len(enriched):,} rows from {len(uploaded_files)} file(s).\n"
        f"{enriched['Date'].min().date()} -> {enriched['Date'].max().date()}"
    )

    return enriched, source_context


def _apply_filters(enriched: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    start_date, end_date = _init_timeframe(enriched)

    source_options = sorted(enriched["SourceFile"].dropna().unique().tolist())
    selected_sources = st.sidebar.multiselect("Source files", source_options, default=source_options)

    category_options = sorted(enriched["Category"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect("Categories", category_options, default=category_options)

    merchant_query = st.sidebar.text_input("Merchant contains", value="").strip().lower()

    max_amount = float(enriched[["DebitCHF", "CreditCHF"]].fillna(0).max().max())
    min_amount = st.sidebar.slider("Minimum abs amount (CHF)", 0.0, max(1.0, max_amount), 0.0)

    filtered = filter_by_date_range(enriched, start_date, end_date)
    filtered = filtered[filtered["SourceFile"].isin(selected_sources)]
    filtered = filtered[filtered["Category"].isin(selected_categories)]

    if merchant_query:
        filtered = filtered[
            filtered["Merchant"].fillna("").astype(str).str.lower().str.contains(merchant_query, na=False)
        ]

    filtered = filtered[
        (filtered["DebitCHF"].abs() >= min_amount) | (filtered["CreditCHF"].abs() >= min_amount)
    ]

    return filtered


def main() -> None:
    _inject_styles()
    _render_header()

    view = st.sidebar.radio(
        "Navigate",
        [
            "Home",
            "Cashflow",
            "Spending",
            "Earnings",
            "Behavior",
            "Plans & Recurring",
            "Data Explorer",
            "Metric Guide",
        ],
    )

    enriched, source_context = _prepare_enriched_data()
    if enriched is None or source_context is None:
        return

    filtered = _apply_filters(enriched)
    if filtered.empty:
        st.warning("No transactions match your current filters.")
        return

    # Shared analytics dataset used by all pages.
    kpis = calculate_kpis(filtered)
    daily = daily_net_cashflow(filtered)
    monthly = monthly_cashflow(filtered)
    hourly = hourly_spending_profile(filtered)
    weekday_avg = weekday_average_cashflow(filtered)
    velocity = spending_velocity(filtered)
    category_table = category_breakdown(filtered)
    top_merchants = merchant_summary(filtered, top_n=20)
    income_sources = income_source_summary(filtered, top_n=20)
    recurring = recurring_transaction_candidates(filtered)
    quality = quality_indicators(filtered)

    default_budget = {
        category: round(kpis["total_spending"] / max(len(category_table), 1), 2)
        for category in category_table.index
    }
    with st.sidebar.expander("Budget setup (JSON)", expanded=False):
        budget_json = st.text_area(
            "Monthly budget by category",
            value=json.dumps(default_budget, indent=2),
            key="budget_json",
            height=220,
        )
    budget_dict = _parse_json_dict(budget_json, default_budget)
    budget_table = budget_progress(filtered, budget_dict)

    if view == "Home":
        render_home(kpis, daily, monthly, category_table, quality)
    elif view == "Cashflow":
        render_cashflow(daily, monthly, velocity)
    elif view == "Spending":
        render_spending(category_table, top_merchants, hourly, weekday_avg)
    elif view == "Earnings":
        render_earnings(category_table, income_sources, hourly, weekday_avg)
    elif view == "Behavior":
        render_behavior(hourly, weekday_avg, filtered)
    elif view == "Plans & Recurring":
        render_subscriptions(recurring, budget_table)
    elif view == "Data Explorer":
        render_data_explorer(filtered, source_context)
    elif view == "Metric Guide":
        render_metric_guide()


if __name__ == "__main__":
    main()
