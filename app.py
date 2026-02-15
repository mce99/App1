"""PulseLedger Streamlit entrypoint with modular page navigation."""

from __future__ import annotations

import datetime
import io
import json

import pandas as pd
import streamlit as st

from analytics import (
    account_summary,
    apply_category_overrides,
    apply_currency_conversion,
    balance_timeline,
    benchmark_assessment,
    budget_progress,
    build_report_pack,
    calculate_kpis,
    category_breakdown,
    daily_net_cashflow,
    data_health_report,
    detect_anomalies,
    enrich_transaction_intelligence,
    filter_by_date_range,
    forecast_cashflow,
    generate_agent_action_plan,
    goals_progress,
    hourly_spending_profile,
    ingestion_quality_by_source,
    income_source_summary,
    merchant_insights,
    monthly_salary_estimate,
    merchant_summary,
    monthly_cashflow,
    possible_duplicate_candidates,
    quality_indicators,
    recurring_transaction_candidates,
    review_queue,
    spending_recommendations,
    spending_velocity,
    weekday_average_cashflow,
)
from categorization import DEFAULT_KEYWORD_MAP, assign_categories_with_confidence
from dashboard_views import (
    render_agent_console,
    render_accounts,
    render_anomalies,
    render_behavior,
    render_cashflow,
    render_data_explorer,
    render_data_health,
    render_earnings,
    render_forecast,
    render_home,
    render_insights,
    render_metric_guide,
    render_portfolio,
    render_report_pack,
    render_spending_map,
    render_spending,
    render_subscriptions,
)
from geo_insights import spending_location_points
from market_data import (
    evaluate_stock_positions,
    fetch_stock_quotes,
    fetch_wallet_balances,
    holdings_mix,
    portfolio_totals,
)
from parsing import SUPPORTED_EXTENSIONS, classify_time_of_day, merge_transactions

st.set_page_config(page_title="PulseLedger", page_icon="\U0001f4ca", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;700&display=swap');
        html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
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
        .hero h1 { margin: 0; letter-spacing: 0.3px; }
        .hero p { margin: 0.35rem 0 0 0; color: #244674; }
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
          <p>Modular transaction intelligence with transfer/account tracking, forecasting, anomalies, and rule review.</p>
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
            "SourceAccount": "Manual entry",
            "TransactionId": f"manual-{m_date.isoformat()}-{time_value}-{m_desc1[:12]}",
        }
        out = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return out.sort_values(["SortDateTime", "Date", "Time"], na_position="last").reset_index(drop=True)


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
        "SourceAccount",
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

    preset = st.sidebar.selectbox(
        "Timeframe preset",
        ["Custom", "Last 30 days", "Last 90 days", "Last 365 days", "Year to date", "Full range"],
        index=0,
    )

    today = max_date
    if preset == "Last 30 days":
        st.session_state["timeframe_range"] = (max(min_date, today - datetime.timedelta(days=29)), today)
    elif preset == "Last 90 days":
        st.session_state["timeframe_range"] = (max(min_date, today - datetime.timedelta(days=89)), today)
    elif preset == "Last 365 days":
        st.session_state["timeframe_range"] = (max(min_date, today - datetime.timedelta(days=364)), today)
    elif preset == "Year to date":
        st.session_state["timeframe_range"] = (max(min_date, datetime.date(today.year, 1, 1)), today)
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


def _prepare_enriched_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, list[str]]]:
    st.sidebar.header("Data Setup")
    uploaded_files = st.sidebar.file_uploader(
        "Upload statements",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="CSV must be semicolon-delimited (;).",
    )
    if not uploaded_files:
        st.info("Upload one or more statement files from the sidebar to start.")
        return None, None, {}

    drop_duplicates = st.sidebar.checkbox("Auto-remove duplicates across files", value=True)

    try:
        df = merge_transactions(uploaded_files, drop_duplicates=drop_duplicates)
    except Exception as exc:
        st.error(f"Could not read file(s): {exc}")
        return None, None, {}

    if df.empty or df["Date"].dropna().empty:
        st.warning("No valid transactions found. Check export format and delimiter (;).")
        return None, None, {}

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

    category_list = sorted(keyword_map.keys()) + ["Other", "Transfers"]

    enriched = apply_currency_conversion(df, conv_rates)
    enriched = assign_categories_with_confidence(enriched, keyword_map)
    enriched = enrich_transaction_intelligence(enriched)

    # Force transfer category where confidence is high.
    enriched.loc[(enriched["IsTransfer"]) & (enriched["TransferConfidence"] >= 0.7), "Category"] = "Transfers"

    if "category_overrides" not in st.session_state:
        st.session_state["category_overrides"] = {}
    enriched = apply_category_overrides(enriched, st.session_state["category_overrides"])

    source_context = _build_statement_context(enriched)

    st.sidebar.success(
        f"Loaded {len(enriched):,} rows from {len(uploaded_files)} file(s).\n"
        f"{enriched['Date'].min().date()} -> {enriched['Date'].max().date()}"
    )

    return enriched, source_context, {"categories": category_list}


def _apply_filters(enriched: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    start_date, end_date = _init_timeframe(enriched)

    source_options = sorted(enriched["SourceFile"].dropna().unique().tolist())
    selected_sources = st.sidebar.multiselect("Source files", source_options, default=source_options)

    account_options = sorted(enriched["SourceAccount"].dropna().unique().tolist())
    selected_accounts = st.sidebar.multiselect("Accounts", account_options, default=account_options)

    category_options = sorted(enriched["Category"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect("Categories", category_options, default=category_options)

    merchant_query = st.sidebar.text_input("Merchant contains", value="").strip().lower()
    include_transfers = st.sidebar.checkbox("Include transfer transactions", value=True)

    max_amount = float(enriched[["DebitCHF", "CreditCHF"]].fillna(0).max().max())
    min_amount = st.sidebar.slider("Minimum abs amount (CHF)", 0.0, max(1.0, max_amount), 0.0)

    filtered = filter_by_date_range(enriched, start_date, end_date)
    filtered = filtered[filtered["SourceFile"].isin(selected_sources)]
    filtered = filtered[filtered["SourceAccount"].isin(selected_accounts)]
    filtered = filtered[filtered["Category"].isin(selected_categories)]

    if merchant_query:
        filtered = filtered[
            filtered["Merchant"].fillna("").astype(str).str.lower().str.contains(merchant_query, na=False)
        ]

    if not include_transfers:
        filtered = filtered[~filtered["IsTransfer"]]

    filtered = filtered[(filtered["DebitCHF"].abs() >= min_amount) | (filtered["CreditCHF"].abs() >= min_amount)]

    return filtered


def _render_review_queue(enriched: pd.DataFrame, category_options: list[str]) -> None:
    st.header("Review Queue")
    st.caption("Approve low-confidence categories and override them permanently for this session.")

    queue = review_queue(enriched)
    if queue.empty:
        st.success("No transactions currently need review.")
        return

    display_cols = [
        "TransactionId",
        "Date",
        "Time",
        "SourceAccount",
        "Merchant",
        "DebitCHF",
        "CreditCHF",
        "Category",
        "CategoryConfidence",
        "CategoryRule",
        "IsTransfer",
        "TransferConfidence",
        "CounterpartyAccount",
    ]
    editable = queue[display_cols].copy()
    editable["ReviewedCategory"] = editable["Category"]

    edited = st.data_editor(
        editable,
        column_config={
            "ReviewedCategory": st.column_config.SelectboxColumn(
                "Reviewed category", options=sorted(set(category_options + ["Other", "Transfers"]))
            )
        },
        hide_index=True,
        use_container_width=True,
    )

    if st.button("Apply reviewed categories"):
        updates = edited[edited["ReviewedCategory"] != edited["Category"]]
        if updates.empty:
            st.info("No review changes detected.")
        else:
            for _, row in updates.iterrows():
                st.session_state["category_overrides"][str(row["TransactionId"])] = str(row["ReviewedCategory"])
            st.success(f"Applied {len(updates)} category override(s).")
            st.rerun()


def _default_stock_positions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Symbol": "AAPL", "Quantity": 0.0, "AvgBuyPrice": 0.0},
            {"Symbol": "MSFT", "Quantity": 0.0, "AvgBuyPrice": 0.0},
        ]
    )


def _default_wallets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Label": "Main BTC", "Address": "", "Chain": "BTC"},
            {"Label": "Main ETH", "Address": "", "Chain": "ETH"},
            {"Label": "Main SOL", "Address": "", "Chain": "SOL"},
        ]
    )


@st.cache_data(ttl=300, show_spinner=False)
def _cached_stock_quotes(symbols: tuple[str, ...]) -> pd.DataFrame:
    return fetch_stock_quotes(list(symbols))


@st.cache_data(ttl=180, show_spinner=False)
def _cached_wallet_balances(wallets_json: str, quote_currency: str) -> pd.DataFrame:
    wallets = pd.read_json(io.StringIO(wallets_json), orient="records")
    return fetch_wallet_balances(wallets, quote_currency=quote_currency)


@st.cache_data(ttl=7200, show_spinner=False)
def _cached_spending_map_points(transactions_json: str, min_spending: float) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(transactions_json), orient="records")
    return spending_location_points(df, min_spending_chf=min_spending)


def _render_portfolio_page() -> None:
    st.header("Portfolio Setup")
    st.caption(
        "Add stock positions and crypto wallet addresses. Balances/prices are fetched automatically."
    )

    if "stock_positions" not in st.session_state:
        st.session_state["stock_positions"] = _default_stock_positions()
    if "wallets" not in st.session_state:
        st.session_state["wallets"] = _default_wallets()

    left, right = st.columns(2)
    with left:
        st.markdown("### Stock positions")
        stocks = st.data_editor(
            st.session_state["stock_positions"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="stock_positions_editor",
        )
    with right:
        st.markdown("### Crypto wallets")
        wallets = st.data_editor(
            st.session_state["wallets"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="wallets_editor",
        )

    st.session_state["stock_positions"] = stocks
    st.session_state["wallets"] = wallets

    quote_currency = st.selectbox("Wallet quote currency", ["usd", "chf"], index=0)
    manual_refresh = st.button("Refresh prices and balances")
    if manual_refresh:
        _cached_stock_quotes.clear()
        _cached_wallet_balances.clear()

    symbols = tuple(
        sorted(
            {
                str(symbol).strip().upper()
                for symbol in stocks.get("Symbol", pd.Series(dtype=str)).tolist()
                if str(symbol).strip()
            }
        )
    )
    quotes = _cached_stock_quotes(symbols) if symbols else pd.DataFrame()
    stock_positions = evaluate_stock_positions(stocks, quotes)

    wallet_payload = wallets.fillna("").to_dict(orient="records")
    wallets_json = json.dumps(wallet_payload, sort_keys=True)
    wallet_positions = _cached_wallet_balances(wallets_json, quote_currency)

    totals = portfolio_totals(stock_positions, wallet_positions)
    mix = holdings_mix(stock_positions, wallet_positions)

    render_portfolio(stock_positions, wallet_positions, totals, mix, quote_currency)


def main() -> None:
    _inject_styles()
    _render_header()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Agent Console",
            "Home",
            "Portfolio",
            "Insights & Optimization",
            "Spending Map",
            "Cashflow",
            "Spending",
            "Earnings",
            "Behavior",
            "Accounts",
            "Forecast",
            "Anomalies",
            "Review Queue",
            "Plans & Recurring",
            "Data Explorer",
            "Data Health",
            "Metric Guide",
            "Report Pack",
        ],
    )

    if page == "Portfolio":
        _render_portfolio_page()
        return

    enriched, source_context, lookup = _prepare_enriched_data()
    if enriched is None or source_context is None:
        return

    # Review queue should be available on full enriched set.
    if page == "Review Queue":
        _render_review_queue(enriched, lookup.get("categories", []))
        return

    filtered = _apply_filters(enriched)
    if filtered.empty:
        st.warning("No transactions match your current filters.")
        return

    # Shared analytics.
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
    anomalies = detect_anomalies(filtered)
    dupes = possible_duplicate_candidates(filtered)
    quality = quality_indicators(filtered)
    ingestion_quality = ingestion_quality_by_source(enriched)
    health_table = data_health_report(filtered)
    accounts = account_summary(filtered)
    balance_table = balance_timeline(filtered)
    merchant_table = merchant_insights(filtered, top_n=25)

    # Goals + budgets
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

    default_goals = {
        "Emergency Fund": {"target": 20000, "saved": 5000},
        "Travel": {"target": 5000, "saved": 1500},
    }
    with st.sidebar.expander("Goals setup (JSON)", expanded=False):
        goals_json = st.text_area(
            "Goals",
            value=json.dumps(default_goals, indent=2),
            key="goals_json",
            height=200,
        )
    goals_dict = _parse_json_dict(goals_json, default_goals)
    goals_table = goals_progress(goals_dict, kpis["net_cashflow"])

    default_benchmarks = {
        "NeedsMaxPct": 50,
        "WantsMaxPct": 30,
        "SavingsMinPct": 20,
        "GroceriesMaxPct": 10,
        "DiningMaxPct": 8,
        "SubscriptionsMaxPct": 5,
        "TransportMaxPct": 15,
    }
    with st.sidebar.expander("Benchmark setup (JSON)", expanded=False):
        benchmark_json = st.text_area(
            "Benchmarks",
            value=json.dumps(default_benchmarks, indent=2),
            key="benchmark_json",
            height=220,
        )
    benchmark_cfg = _parse_json_dict(benchmark_json, default_benchmarks)

    salary_info = monthly_salary_estimate(filtered)
    benchmark_table = benchmark_assessment(
        filtered,
        avg_monthly_salary=float(salary_info.get("avg_monthly_salary", 0.0) or 0.0),
        benchmark_cfg=benchmark_cfg,
    )
    recommendations = spending_recommendations(filtered, benchmark_table)
    action_plan = generate_agent_action_plan(
        kpis=kpis,
        quality=quality,
        benchmark_table=benchmark_table,
        anomalies=anomalies,
        dupes=dupes,
        recurring=recurring,
    )

    with st.sidebar.expander("Map settings", expanded=False):
        min_spending_for_map = st.slider("Min spending per point (CHF)", 0.0, 500.0, 20.0, key="map_min_spending")

    map_cols = [col for col in ["Date", "Location", "DebitCHF", "Merchant", "Category", "SourceAccount"] if col in filtered.columns]
    map_payload = filtered[map_cols].fillna("").to_json(orient="records", date_format="iso")
    map_points = _cached_spending_map_points(map_payload, float(min_spending_for_map))

    forecast = forecast_cashflow(filtered, recurring)

    summary_md, report_zip = build_report_pack(filtered, kpis, monthly)

    if page == "Agent Console":
        render_agent_console(action_plan, ingestion_quality)
    elif page == "Home":
        render_home(kpis, daily, monthly, category_table, quality)
    elif page == "Insights & Optimization":
        render_insights(salary_info, benchmark_table, recommendations, merchant_table, balance_table)
    elif page == "Spending Map":
        render_spending_map(map_points)
    elif page == "Cashflow":
        render_cashflow(daily, monthly, velocity)
    elif page == "Spending":
        render_spending(category_table, top_merchants, hourly, weekday_avg)
    elif page == "Earnings":
        render_earnings(category_table, income_sources, hourly, weekday_avg)
    elif page == "Behavior":
        render_behavior(hourly, weekday_avg, filtered)
    elif page == "Accounts":
        transfers = filtered[filtered["IsTransfer"]].copy()
        render_accounts(accounts, transfers)
    elif page == "Forecast":
        render_forecast(forecast)
    elif page == "Anomalies":
        render_anomalies(anomalies, dupes)
    elif page == "Plans & Recurring":
        render_subscriptions(recurring, budget_table, goals_table)
    elif page == "Data Explorer":
        render_data_explorer(filtered, source_context)
    elif page == "Data Health":
        render_data_health(health_table, quality)
    elif page == "Metric Guide":
        render_metric_guide()
    elif page == "Report Pack":
        render_report_pack(summary_md, report_zip)


if __name__ == "__main__":
    main()
