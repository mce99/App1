"""PulseLedger Streamlit entrypoint with modular page navigation."""

from __future__ import annotations

import datetime
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_assistant import generate_ai_brief
from analytics import (
    account_summary,
    apply_category_overrides,
    apply_currency_conversion,
    balance_timeline,
    benchmark_assessment,
    budget_progress,
    build_report_pack,
    cashflow_stability_metrics,
    calculate_kpis,
    category_volatility,
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
    income_concentration_table,
    income_source_summary,
    category_momentum,
    merchant_concentration_table,
    merchant_insights,
    monthly_salary_estimate,
    merchant_summary,
    monthly_trend_diagnostics,
    monthly_cashflow,
    possible_duplicate_candidates,
    quality_indicators,
    recurring_transaction_candidates,
    review_queue,
    spending_run_rate_projection,
    savings_scenario,
    spending_recommendations,
    spending_velocity,
    transaction_size_distribution,
    weekday_weekend_split,
    weekday_average_cashflow,
)
from categorization import DEFAULT_KEYWORD_MAP, assign_categories_with_confidence
from dashboard_views import (
    render_agent_console,
    render_accounts,
    render_anomalies,
    render_behavior,
    render_chart_builder,
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
from local_sources import load_local_statement_uploads
from mapping_memory import DEFAULT_MAPPING_MEMORY_PATH, load_mapping_memory, save_mapping_memory
from mapping_rules import (
    apply_pattern_rules,
    learn_pattern_rules,
    normalize_rule_map,
    suggest_category_from_rules,
    tokenize_mapping_text,
    transaction_text,
)
from parsing import SUPPORTED_EXTENSIONS, classify_time_of_day, merge_transactions

st.set_page_config(page_title="PulseLedger", page_icon="\U0001f4ca", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        :root {
            --pl-bg: #eef2f7;
            --pl-surface: #ffffff;
            --pl-surface-soft: #f6f8fb;
            --pl-border: #d5deea;
            --pl-text: #1b2e43;
            --pl-muted: #61758c;
            --pl-brand: #1f4e79;
            --pl-brand-deep: #173b5f;
            --pl-brand-accent: #0e6ea8;
        }
        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--pl-text);
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(28, 65, 103, 0.08) 0%, rgba(255, 255, 255, 0) 280px),
                repeating-linear-gradient(
                    90deg,
                    rgba(31, 78, 121, 0.028) 0,
                    rgba(31, 78, 121, 0.028) 1px,
                    transparent 1px,
                    transparent 30px
                ),
                radial-gradient(1400px 400px at 85% -20%, rgba(14, 110, 168, 0.16), transparent 70%),
                var(--pl-bg);
        }
        h1, h2, h3, h4, h5 {
            color: var(--pl-text);
            letter-spacing: 0.15px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1b3e63 0%, #122d4a 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label {
            color: #e8eff8 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f5f9ff !important;
        }
        [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        .hero {
            margin-bottom: 0.8rem;
            padding: 1.15rem 1.35rem;
            border: 1px solid var(--pl-border);
            border-left: 6px solid var(--pl-brand);
            border-radius: 12px;
            background: var(--pl-surface);
            box-shadow: 0 8px 24px rgba(20, 45, 72, 0.08);
        }
        .hero-kicker {
            margin: 0 0 0.25rem 0;
            color: var(--pl-brand);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero h1 {
            margin: 0;
            letter-spacing: 0.2px;
            font-size: 1.75rem;
        }
        .hero p {
            margin: 0.35rem 0 0 0;
            color: var(--pl-muted);
            max-width: 760px;
        }
        [data-testid="stMetric"] {
            background: var(--pl-surface);
            border: 1px solid var(--pl-border);
            border-radius: 10px;
            padding: 0.5rem 0.65rem;
            box-shadow: 0 4px 14px rgba(23, 50, 78, 0.06);
        }
        [data-testid="stMetricLabel"] {
            color: var(--pl-muted);
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.2rem;
            border-bottom: 1px solid var(--pl-border);
            padding-bottom: 0.1rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: var(--pl-surface-soft);
            border: 1px solid var(--pl-border);
            border-radius: 8px 8px 0 0;
            color: var(--pl-muted);
            font-weight: 600;
            padding: 0.4rem 0.8rem;
        }
        .stTabs [aria-selected="true"] {
            background: var(--pl-surface);
            color: var(--pl-text);
            border-bottom-color: var(--pl-surface);
        }
        .stButton > button,
        .stDownloadButton > button {
            border: 0;
            border-radius: 8px;
            background: linear-gradient(180deg, var(--pl-brand) 0%, var(--pl-brand-deep) 100%);
            color: #ffffff;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: linear-gradient(180deg, #255a8d 0%, #1a446b 100%);
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div {
            border-radius: 8px !important;
            border: 1px solid var(--pl-border) !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--pl-border);
            border-radius: 10px;
            background: var(--pl-surface);
            box-shadow: 0 3px 10px rgba(18, 43, 67, 0.05);
        }
        details[data-testid="stExpander"] {
            border: 1px solid var(--pl-border);
            border-radius: 10px;
            background: var(--pl-surface);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <p class="hero-kicker">Executive Finance Intelligence</p>
          <h1>PulseLedger Command Center</h1>
          <p>
            Corporate-grade visibility into spending, earnings, risk signals, and portfolio exposure.
            Upload statements, select a timeframe, and steer with data.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_quick_start() -> None:
    with st.expander("Executive quick start (30 seconds)", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "1. Upload files, or enable Local Sync folder in the sidebar.",
                    "2. Pick a timeframe and optional category filter.",
                    "3. Open `Plan & Improve` for concrete actions.",
                ]
            )
        )


def _parse_json_dict(json_text: str, fallback: dict) -> dict:
    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            return fallback
        return parsed
    except Exception:
        return fallback


def _ensure_state_defaults() -> None:
    if "category_overrides" not in st.session_state:
        st.session_state["category_overrides"] = {}
    if "merchant_category_rules" not in st.session_state:
        st.session_state["merchant_category_rules"] = {}
    if "pattern_category_rules" not in st.session_state:
        st.session_state["pattern_category_rules"] = {}
    if "ai_brief_text" not in st.session_state:
        st.session_state["ai_brief_text"] = ""
    if "ai_brief_mode" not in st.session_state:
        st.session_state["ai_brief_mode"] = "offline"
    if "local_sync_folder" not in st.session_state:
        st.session_state["local_sync_folder"] = "~/Downloads/ubs_statements"
    if "local_sync_recursive" not in st.session_state:
        st.session_state["local_sync_recursive"] = False
    if "use_local_sync_files" not in st.session_state:
        st.session_state["use_local_sync_files"] = False
    if "mapping_memory_path" not in st.session_state:
        st.session_state["mapping_memory_path"] = DEFAULT_MAPPING_MEMORY_PATH
    if "mapping_memory_loaded" not in st.session_state:
        st.session_state["mapping_memory_loaded"] = False
    if "mapping_memory_auto_save" not in st.session_state:
        st.session_state["mapping_memory_auto_save"] = True
    if "mapping_memory_last_saved" not in st.session_state:
        st.session_state["mapping_memory_last_saved"] = ""
    _load_mapping_memory_once()


def _load_mapping_memory_once() -> None:
    if bool(st.session_state.get("mapping_memory_loaded", False)):
        return
    path = str(st.session_state.get("mapping_memory_path", DEFAULT_MAPPING_MEMORY_PATH))
    try:
        payload = load_mapping_memory(path)
    except Exception:
        st.session_state["mapping_memory_loaded"] = True
        return
    st.session_state["category_overrides"] = payload.get("category_overrides", {})
    st.session_state["merchant_category_rules"] = payload.get("merchant_category_rules", {})
    st.session_state["pattern_category_rules"] = payload.get("pattern_category_rules", {})
    st.session_state["mapping_memory_loaded"] = True


def _save_mapping_memory_to_disk() -> tuple[bool, str]:
    path = str(st.session_state.get("mapping_memory_path", DEFAULT_MAPPING_MEMORY_PATH))
    try:
        saved = save_mapping_memory(
            path=path,
            category_overrides=st.session_state.get("category_overrides", {}),
            merchant_category_rules=st.session_state.get("merchant_category_rules", {}),
            pattern_category_rules=st.session_state.get("pattern_category_rules", {}),
        )
    except Exception as exc:
        return False, str(exc)

    try:
        pretty = str(Path(saved).expanduser())
    except Exception:
        pretty = str(saved)
    st.session_state["mapping_memory_last_saved"] = pretty
    return True, pretty


def _auto_save_mapping_memory() -> None:
    if not bool(st.session_state.get("mapping_memory_auto_save", True)):
        return
    _save_mapping_memory_to_disk()


def _apply_merchant_category_rules(df: pd.DataFrame) -> pd.DataFrame:
    rules = normalize_rule_map(st.session_state.get("merchant_category_rules", {}))
    if not rules:
        return df

    out = df.copy()
    merchant_norm = out.get("MerchantNormalized", pd.Series([""] * len(out))).astype(str).str.upper().str.strip()
    mapped = merchant_norm.map(rules)
    mask = mapped.notna()
    if mask.any():
        out.loc[mask, "Category"] = mapped.loc[mask]
        out.loc[mask, "CategoryConfidence"] = 0.99
        out.loc[mask, "CategoryRule"] = "MerchantRule"
    return out


def _apply_pattern_category_rules(df: pd.DataFrame) -> pd.DataFrame:
    rules = normalize_rule_map(st.session_state.get("pattern_category_rules", {}))
    return apply_pattern_rules(df, rules, low_confidence_threshold=0.75)


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
    _ensure_state_defaults()
    st.sidebar.header("1) Upload data")
    uploaded_files_raw = st.sidebar.file_uploader(
        "Upload statements",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="CSV must be semicolon-delimited (;).",
    )
    uploaded_files = list(uploaded_files_raw or [])

    local_files = []
    with st.sidebar.expander("Local sync folder (optional)", expanded=False):
        st.checkbox(
            "Include files from local folder",
            key="use_local_sync_files",
            help="When enabled, files from this folder are loaded on each rerun.",
        )
        st.text_input("Folder path", key="local_sync_folder")
        st.checkbox("Scan subfolders", key="local_sync_recursive")
        max_local_files = int(
            st.number_input("Max local files", min_value=10, max_value=1000, value=250, step=10)
        )
        if st.session_state["use_local_sync_files"]:
            try:
                local_files = load_local_statement_uploads(
                    folder_path=st.session_state["local_sync_folder"],
                    recursive=bool(st.session_state["local_sync_recursive"]),
                    supported_extensions=SUPPORTED_EXTENSIONS,
                    max_files=max_local_files,
                )
                st.caption(f"Found {len(local_files)} local statement file(s).")
            except Exception as exc:
                st.warning(f"Local folder read failed: {exc}")

    all_files = uploaded_files + local_files
    if not all_files:
        st.info("Upload files or enable local sync folder to start.")
        return None, None, {}

    drop_duplicates = st.sidebar.checkbox("Auto-remove duplicates across files", value=True)

    try:
        df = merge_transactions(all_files, drop_duplicates=drop_duplicates)
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
    enriched = _apply_merchant_category_rules(enriched)
    enriched = _apply_pattern_category_rules(enriched)

    # Force transfer category where confidence is high.
    enriched.loc[(enriched["IsTransfer"]) & (enriched["TransferConfidence"] >= 0.7), "Category"] = "Transfers"

    enriched = apply_category_overrides(enriched, st.session_state["category_overrides"])

    source_context = _build_statement_context(enriched)

    st.sidebar.success(
        f"Loaded {len(enriched):,} rows from {len(all_files)} file(s).\n"
        f"Uploads: {len(uploaded_files)} | Local sync: {len(local_files)}\n"
        f"{enriched['Date'].min().date()} -> {enriched['Date'].max().date()}"
    )

    return enriched, source_context, {"categories": category_list}


def _apply_filters(enriched: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("2) Filters")
    start_date, end_date = _init_timeframe(enriched)

    category_options = sorted(enriched["Category"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect("Categories", category_options, default=category_options)
    merchant_query = st.sidebar.text_input("Quick merchant search", value="").strip().lower()
    include_transfers = st.sidebar.checkbox("Include transfer transactions", value=False)

    max_amount = float(enriched[["DebitCHF", "CreditCHF"]].fillna(0).max().max())
    min_amount = 0.0

    source_options = sorted(enriched["SourceFile"].dropna().unique().tolist())
    selected_sources = source_options
    account_options = sorted(enriched["SourceAccount"].dropna().unique().tolist())
    selected_accounts = account_options
    with st.sidebar.expander("Advanced filters", expanded=False):
        selected_sources = st.multiselect("Source files", source_options, default=source_options)
        selected_accounts = st.multiselect("Accounts", account_options, default=account_options)
        min_amount = st.slider("Minimum abs amount (CHF)", 0.0, max(1.0, max_amount), 0.0)

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
            _auto_save_mapping_memory()
            st.success(f"Applied {len(updates)} category override(s).")
            st.rerun()


def _render_category_lab(enriched: pd.DataFrame, category_options: list[str]) -> None:
    st.header("Category Lab")
    st.caption("Label uncategorized or low-confidence transactions and optionally learn merchant rules.")

    confidence_cutoff = st.slider("Confidence cutoff", 0.1, 0.95, 0.65, 0.05, key="cat_lab_cutoff")

    base = enriched.copy()
    candidates = base[
        (base["Category"].fillna("Other") == "Other")
        | (base["CategoryConfidence"].fillna(0.0) < confidence_cutoff)
    ].copy()

    if candidates.empty:
        st.success("No unlabeled transactions for the selected data.")
        return

    confident = base[
        (base["Category"].fillna("") != "Other")
        & (base["CategoryConfidence"].fillna(0.0) >= 0.85)
        & (base["MerchantNormalized"].fillna("").astype(str).str.strip() != "")
    ]
    merchant_suggestion = (
        confident.groupby("MerchantNormalized")["Category"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    candidates["SuggestedCategory"] = candidates["MerchantNormalized"].map(merchant_suggestion)
    transfer_mask = (candidates["IsTransfer"]) & (candidates["TransferConfidence"] >= 0.7)
    candidates.loc[transfer_mask, "SuggestedCategory"] = "Transfers"
    candidates["SuggestedCategory"] = candidates["SuggestedCategory"].fillna("Other")
    candidates["FinalCategory"] = candidates["SuggestedCategory"]
    candidates["ApplyToMerchant"] = False

    show_cols = [
        "TransactionId",
        "Date",
        "Time",
        "Merchant",
        "MerchantNormalized",
        "DebitCHF",
        "CreditCHF",
        "Category",
        "CategoryConfidence",
        "SuggestedCategory",
        "FinalCategory",
        "ApplyToMerchant",
    ]
    editor = st.data_editor(
        candidates[show_cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "FinalCategory": st.column_config.SelectboxColumn(
                "Final category", options=sorted(set(category_options + ["Other", "Transfers"]))
            ),
            "ApplyToMerchant": st.column_config.CheckboxColumn("Learn merchant rule"),
        },
        key="category_lab_editor",
    )

    left, right = st.columns(2)
    with left:
        if st.button("Apply labels", key="category_lab_apply"):
            changed = editor[editor["FinalCategory"] != editor["Category"]]
            learned = editor[(editor["ApplyToMerchant"]) & (editor["FinalCategory"].fillna("") != "")]

            for _, row in changed.iterrows():
                st.session_state["category_overrides"][str(row["TransactionId"])] = str(row["FinalCategory"])

            learned_count = 0
            for _, row in learned.iterrows():
                merchant_key = str(row.get("MerchantNormalized", "")).upper().strip()
                if merchant_key:
                    st.session_state["merchant_category_rules"][merchant_key] = str(row["FinalCategory"])
                    learned_count += 1

            _auto_save_mapping_memory()
            st.success(
                f"Applied {len(changed)} transaction labels and learned {learned_count} merchant rule(s)."
            )
            st.rerun()

    with right:
        if st.button("Clear merchant rules", key="category_lab_clear_rules"):
            st.session_state["merchant_category_rules"] = {}
            _auto_save_mapping_memory()
            st.info("Merchant rules cleared.")
            st.rerun()

    with st.expander("Current merchant rules", expanded=False):
        rule_items = [
            {"MerchantNormalized": key, "Category": value}
            for key, value in sorted(st.session_state["merchant_category_rules"].items())
        ]
        if not rule_items:
            st.caption("No merchant rules yet.")
        else:
            st.dataframe(pd.DataFrame(rule_items), use_container_width=True, hide_index=True)


def _render_mapping_studio(enriched: pd.DataFrame, category_options: list[str]) -> None:
    st.header("Mapping Studio")
    st.caption("Map transactions to improve quality, then let the app learn those patterns.")

    merchant_rules = normalize_rule_map(st.session_state.get("merchant_category_rules", {}))
    pattern_rules = normalize_rule_map(st.session_state.get("pattern_category_rules", {}))
    override_count = len(st.session_state.get("category_overrides", {}))

    c1, c2, c3 = st.columns(3)
    c1.metric("Manual tx labels", f"{override_count:,}")
    c2.metric("Merchant rules", f"{len(merchant_rules):,}")
    c3.metric("Pattern rules", f"{len(pattern_rules):,}")

    with st.expander("Mapping memory", expanded=False):
        st.text_input("Memory file path", key="mapping_memory_path")
        st.checkbox("Auto-save mapping changes", key="mapping_memory_auto_save")
        if st.session_state.get("mapping_memory_last_saved", ""):
            st.caption(f"Last saved: {st.session_state['mapping_memory_last_saved']}")
        m1, m2, m3 = st.columns(3)
        if m1.button("Save memory now", key="mapping_memory_save_now"):
            ok, info = _save_mapping_memory_to_disk()
            if ok:
                st.success(f"Saved mapping memory: {info}")
            else:
                st.error(f"Failed to save mapping memory: {info}")
        if m2.button("Reload memory", key="mapping_memory_reload"):
            path = str(st.session_state.get("mapping_memory_path", DEFAULT_MAPPING_MEMORY_PATH))
            try:
                payload = load_mapping_memory(path)
                st.session_state["category_overrides"] = payload.get("category_overrides", {})
                st.session_state["merchant_category_rules"] = payload.get("merchant_category_rules", {})
                st.session_state["pattern_category_rules"] = payload.get("pattern_category_rules", {})
                st.success("Mapping memory reloaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load mapping memory: {exc}")
        if m3.button("Reset mapping memory", key="mapping_memory_reset"):
            st.session_state["category_overrides"] = {}
            st.session_state["merchant_category_rules"] = {}
            st.session_state["pattern_category_rules"] = {}
            _auto_save_mapping_memory()
            st.warning("Mapping memory reset.")
            st.rerun()

    st.markdown("### Learn rules from your labels")
    l1, l2, l3 = st.columns(3)
    min_examples = int(l1.number_input("Min examples per token", min_value=2, max_value=10, value=3, step=1))
    min_precision = float(l2.slider("Min token precision", 0.5, 1.0, 0.8, 0.05))
    include_high_conf = bool(l3.checkbox("Include high-confidence auto labels", value=False))

    if st.button("Auto-learn pattern rules", key="learn_pattern_rules_btn"):
        if "CategoryOverridden" in enriched.columns:
            manual_mask = enriched["CategoryOverridden"].fillna(False)
        else:
            manual_mask = pd.Series(False, index=enriched.index)
        train = enriched[manual_mask].copy()
        if include_high_conf:
            high_conf = enriched[
                (enriched["Category"].fillna("Other") != "Other")
                & (enriched["CategoryConfidence"].fillna(0.0) >= 0.9)
            ].copy()
            train = pd.concat([train, high_conf], ignore_index=True).drop_duplicates(subset=["TransactionId"], keep="first")

        learned = learn_pattern_rules(train, min_examples=min_examples, min_precision=min_precision)
        if learned.empty:
            st.info("No robust pattern rules found yet. Label more rows first.")
        else:
            for _, row in learned.iterrows():
                st.session_state["pattern_category_rules"][str(row["Token"]).upper().strip()] = str(row["Category"])
            _auto_save_mapping_memory()
            st.success(f"Learned or refreshed {len(learned)} pattern rule(s).")
            st.rerun()

    rule_left, rule_right = st.columns(2)
    with rule_left:
        st.markdown("### Merchant rules")
        merchant_rows = [
            {"MerchantNormalized": key, "Category": value}
            for key, value in sorted(merchant_rules.items())
        ]
        if merchant_rows:
            st.dataframe(pd.DataFrame(merchant_rows), use_container_width=True, hide_index=True, height=220)
        else:
            st.caption("No merchant rules.")
        if st.button("Clear merchant rules", key="mapping_clear_merchant"):
            st.session_state["merchant_category_rules"] = {}
            _auto_save_mapping_memory()
            st.rerun()

    with rule_right:
        st.markdown("### Pattern rules")
        pattern_rows = [
            {"Token": key, "Category": value}
            for key, value in sorted(pattern_rules.items())
        ]
        if pattern_rows:
            st.dataframe(pd.DataFrame(pattern_rows), use_container_width=True, hide_index=True, height=220)
        else:
            st.caption("No pattern rules.")
        if st.button("Clear pattern rules", key="mapping_clear_pattern"):
            st.session_state["pattern_category_rules"] = {}
            _auto_save_mapping_memory()
            st.rerun()

    st.markdown("### Map candidate transactions")
    queue_mode = st.selectbox(
        "Candidate scope",
        ["Other only", "Other + low confidence", "All filtered"],
        index=1,
    )
    low_conf_cutoff = st.slider("Low-confidence cutoff", 0.3, 0.95, 0.7, 0.05, key="mapping_low_conf_cutoff")
    max_rows = int(st.number_input("Max rows to edit", min_value=50, max_value=2000, value=400, step=50))

    candidates = enriched.copy()
    if queue_mode == "Other only":
        candidates = candidates[candidates["Category"].fillna("Other") == "Other"]
    elif queue_mode == "Other + low confidence":
        candidates = candidates[
            (candidates["Category"].fillna("Other") == "Other")
            | (candidates["CategoryConfidence"].fillna(0.0) < low_conf_cutoff)
        ]

    candidates = candidates.sort_values(["Date", "Time"], ascending=[False, False]).head(max_rows).copy()
    if candidates.empty:
        st.success("No candidates for mapping in this view.")
        return

    hist_source = enriched[
        (enriched["Category"].fillna("Other") != "Other")
        & (enriched["CategoryConfidence"].fillna(0.0) >= 0.85)
    ]
    history_map = (
        hist_source.groupby("MerchantNormalized")["Category"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    def _mapping_suggestion(row: pd.Series) -> tuple[str, str, str]:
        merchant_key = str(row.get("MerchantNormalized", "")).upper().strip()
        if merchant_key in merchant_rules:
            return merchant_rules[merchant_key], "MerchantRule", ""

        suggested, token, _score = suggest_category_from_rules(transaction_text(row), pattern_rules)
        if suggested:
            return suggested, f"PatternRule:{token}", token

        if merchant_key in history_map:
            return str(history_map[merchant_key]), "MerchantHistory", ""

        if bool(row.get("IsTransfer", False)) and float(row.get("TransferConfidence", 0.0) or 0.0) >= 0.7:
            return "Transfers", "TransferSignal", ""

        return "Other", "Unmapped", ""

    suggested = candidates.apply(_mapping_suggestion, axis=1, result_type="expand")
    suggested.columns = ["SuggestedCategory", "SuggestionSource", "SuggestedToken"]
    candidates[suggested.columns] = suggested
    candidates["FinalCategory"] = candidates["SuggestedCategory"]
    candidates["LearnMerchantRule"] = False
    candidates["LearnPatternRule"] = False
    candidates["PatternToken"] = candidates["SuggestedToken"]

    suggestion_stats = candidates["SuggestionSource"].value_counts(dropna=False).to_dict()
    mapped_suggestions = int(len(candidates) - suggestion_stats.get("Unmapped", 0))
    s1, s2, s3 = st.columns(3)
    s1.metric("Candidate rows", f"{len(candidates):,}")
    s2.metric("Auto-suggested", f"{mapped_suggestions:,}")
    s3.metric("Needs manual mapping", f"{int(suggestion_stats.get('Unmapped', 0)):,}")

    editable_cols = [
        "TransactionId",
        "Date",
        "Time",
        "Merchant",
        "MerchantNormalized",
        "DebitCHF",
        "CreditCHF",
        "Category",
        "CategoryConfidence",
        "SuggestedCategory",
        "SuggestionSource",
        "FinalCategory",
        "LearnMerchantRule",
        "LearnPatternRule",
        "PatternToken",
    ]
    editor = st.data_editor(
        candidates[editable_cols],
        hide_index=True,
        use_container_width=True,
        key="mapping_studio_editor",
        column_config={
            "FinalCategory": st.column_config.SelectboxColumn(
                "Final category", options=sorted(set(category_options + ["Other", "Transfers"]))
            ),
            "LearnMerchantRule": st.column_config.CheckboxColumn("Learn merchant"),
            "LearnPatternRule": st.column_config.CheckboxColumn("Learn token"),
            "PatternToken": st.column_config.TextColumn("Pattern token"),
        },
    )

    if st.button("Apply mappings", key="mapping_studio_apply"):
        changed = editor[editor["FinalCategory"] != editor["Category"]]
        merchant_learn = editor[(editor["LearnMerchantRule"]) & (editor["FinalCategory"].fillna("") != "")]
        token_learn = editor[(editor["LearnPatternRule"]) & (editor["FinalCategory"].fillna("") != "")]

        for _, row in changed.iterrows():
            st.session_state["category_overrides"][str(row["TransactionId"])] = str(row["FinalCategory"])

        merchant_count = 0
        for _, row in merchant_learn.iterrows():
            merchant_key = str(row.get("MerchantNormalized", "")).upper().strip()
            if merchant_key:
                st.session_state["merchant_category_rules"][merchant_key] = str(row["FinalCategory"])
                merchant_count += 1

        pattern_count = 0
        for _, row in token_learn.iterrows():
            token_text = str(row.get("PatternToken", "")).upper().strip()
            if not token_text:
                tokens = tokenize_mapping_text(transaction_text(row))
                token_text = tokens[0] if tokens else ""
            if token_text:
                st.session_state["pattern_category_rules"][token_text] = str(row["FinalCategory"])
                pattern_count += 1

        _auto_save_mapping_memory()
        st.success(
            f"Applied {len(changed)} transaction labels, learned {merchant_count} merchant rules, and {pattern_count} pattern rules."
        )
        st.rerun()


def _render_ai_coach(
    kpis: dict[str, float],
    benchmark_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    merchant_table: pd.DataFrame,
    anomalies: pd.DataFrame,
    recurring: pd.DataFrame,
    action_plan: pd.DataFrame,
) -> None:
    st.header("AI Coach")
    st.caption("Get a personalized analysis summary and prioritized money actions.")

    goal = st.text_input("Goal for next month", value="Increase savings by CHF 1,000")
    model = st.selectbox("AI model", ["gpt-4.1-mini", "gpt-4o-mini"], index=0)
    key_default = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API key (optional)", value=key_default, type="password")

    if st.button("Generate AI analysis", key="ai_generate"):
        mode, text = generate_ai_brief(
            kpis=kpis,
            benchmark_table=benchmark_table,
            recommendations=recommendations,
            merchant_table=merchant_table,
            anomalies=anomalies,
            recurring=recurring,
            action_plan=action_plan,
            user_goal=goal,
            api_key=api_key,
            model=model,
        )
        st.session_state["ai_brief_mode"] = mode
        st.session_state["ai_brief_text"] = text

    if st.session_state.get("ai_brief_text", "").strip():
        mode = st.session_state.get("ai_brief_mode", "offline")
        if mode == "online":
            st.success("Generated with live AI model.")
        else:
            st.info("Generated using offline fallback logic. Add API key for live AI output.")
        st.markdown(st.session_state["ai_brief_text"])


def _render_bank_sync() -> None:
    _ensure_state_defaults()
    st.header("Bank Sync (UBS)")
    st.caption("Manual login + MFA in browser, automatic statement download capture to a local folder.")

    st.markdown("### 1) Configure local sync folder")
    st.text_input("Sync folder path", key="local_sync_folder")
    st.checkbox("Scan subfolders", key="local_sync_recursive")
    st.checkbox("Include this folder in app ingestion", key="use_local_sync_files")

    folder = str(st.session_state["local_sync_folder"])
    st.markdown("### 2) Run guided sync script in terminal")
    st.code(
        (
            f"cd \"/Users/mce/Documents/New project\"\n"
            f".venv/bin/python ubs_browser_sync.py --download-dir \"{folder}\""
        ),
        language="bash",
    )
    st.caption(
        "Open script -> login manually -> export statements -> press ENTER in terminal when done."
    )

    st.markdown("### 3) Analyze in app")
    st.markdown(
        "Go to `Overview` or `Mapping` workspace. New downloaded files are auto-loaded when local sync is enabled."
    )


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
    _render_quick_start()
    _ensure_state_defaults()

    page = st.sidebar.radio(
        "Executive Workspaces",
        [
            "Overview",
            "Money In/Out",
            "Chart Builder",
            "Plan & Improve",
            "Mapping",
            "Data & QA",
            "Portfolio",
            "Guide",
        ],
    )

    if page == "Portfolio":
        _render_portfolio_page()
        return
    if page == "Guide":
        render_metric_guide()
        return

    enriched, source_context, lookup = _prepare_enriched_data()
    if enriched is None or source_context is None:
        return

    filtered = _apply_filters(enriched)
    if filtered.empty:
        st.warning("No transactions match your current filters.")
        return

    if page == "Mapping":
        _render_mapping_studio(filtered, lookup.get("categories", []))
        return
    if page == "Chart Builder":
        render_chart_builder(filtered)
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
    with st.sidebar.expander("Deep analytics settings", expanded=False):
        concentration_top_n = st.slider(
            "Top merchants / sources",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            key="deep_top_n",
        )
        run_rate_lookback = st.slider(
            "Run-rate lookback (months)",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
            key="deep_run_rate_lookback",
        )
        volatility_min_months = st.slider(
            "Category volatility min months",
            min_value=2,
            max_value=12,
            value=3,
            step=1,
            key="deep_volatility_min_months",
        )
        trend_lookback_months = st.slider(
            "Trend lookback (months)",
            min_value=3,
            max_value=36,
            value=12,
            step=1,
            key="deep_trend_lookback",
        )

    salary_info = monthly_salary_estimate(filtered)
    benchmark_table = benchmark_assessment(
        filtered,
        avg_monthly_salary=float(salary_info.get("avg_monthly_salary", 0.0) or 0.0),
        benchmark_cfg=benchmark_cfg,
    )
    recommendations = spending_recommendations(filtered, benchmark_table)
    stability_metrics = cashflow_stability_metrics(filtered)
    merchant_concentration = merchant_concentration_table(filtered, top_n=int(concentration_top_n))
    income_concentration = income_concentration_table(filtered, top_n=int(concentration_top_n))
    size_distribution = transaction_size_distribution(filtered)
    weekend_split = weekday_weekend_split(filtered)
    category_volatility_table = category_volatility(filtered, min_months=int(volatility_min_months))
    run_rate = spending_run_rate_projection(filtered, lookback_months=int(run_rate_lookback))
    trend_table = monthly_trend_diagnostics(filtered, lookback_months=int(trend_lookback_months))
    momentum_table = category_momentum(filtered)
    action_plan = generate_agent_action_plan(
        kpis=kpis,
        quality=quality,
        benchmark_table=benchmark_table,
        anomalies=anomalies,
        dupes=dupes,
        recurring=recurring,
    )

    if page == "Overview":
        render_home(kpis, daily, monthly, category_table, quality)
        st.markdown("### What to do next")
        st.dataframe(action_plan.head(6), use_container_width=True, hide_index=True)
    elif page == "Money In/Out":
        tab_cash, tab_spend, tab_earn, tab_map = st.tabs(
            ["Cashflow", "Spending", "Earnings", "Spending map"]
        )
        with tab_cash:
            render_cashflow(daily, monthly, velocity)
        with tab_spend:
            render_spending(category_table, top_merchants, hourly, weekday_avg)
            st.markdown("### Behavior timing")
            render_behavior(hourly, weekday_avg, filtered)
        with tab_earn:
            render_earnings(category_table, income_sources, hourly, weekday_avg)
        with tab_map:
            with st.sidebar.expander("Map settings", expanded=False):
                min_spending_for_map = st.slider(
                    "Min spending per point (CHF)", 0.0, 500.0, 20.0, key="map_min_spending"
                )
            map_cols = [
                col
                for col in ["Date", "Location", "DebitCHF", "Merchant", "Category", "SourceAccount"]
                if col in filtered.columns
            ]
            map_payload = filtered[map_cols].fillna("").to_json(orient="records", date_format="iso")
            map_points = _cached_spending_map_points(map_payload, float(min_spending_for_map))
            render_spending_map(map_points)
    elif page == "Plan & Improve":
        forecast = forecast_cashflow(filtered, recurring)
        tab_actions, tab_insights, tab_sim, tab_deep, tab_lab, tab_ai, tab_anom, tab_forecast, tab_plans = st.tabs(
            [
                "Action queue",
                "Insights",
                "Simulator",
                "Deep Analytics",
                "Category Lab",
                "AI Coach",
                "Anomalies",
                "Forecast",
                "Plans",
            ]
        )
        with tab_actions:
            render_agent_console(action_plan, ingestion_quality)
        with tab_insights:
            render_insights(salary_info, benchmark_table, recommendations, merchant_table, balance_table)
        with tab_sim:
            st.markdown("### Savings scenario simulator")
            left, right = st.columns(2)
            with left:
                monthly_target = st.number_input(
                    "Target extra monthly savings (CHF)",
                    min_value=0.0,
                    value=1000.0,
                    step=50.0,
                )
                max_cut_pct = st.slider(
                    "Max cut per category (%)",
                    min_value=5,
                    max_value=60,
                    value=20,
                    step=5,
                )
                excluded_categories = st.multiselect(
                    "Protected categories (do not cut)",
                    options=sorted(category_table.index.astype(str).tolist()),
                    default=["Utilities & Bills"] if "Utilities & Bills" in category_table.index else [],
                )
                scenario = savings_scenario(
                    filtered,
                    target_extra_savings_chf=float(monthly_target),
                    max_cut_pct=float(max_cut_pct) / 100.0,
                    excluded_categories=excluded_categories,
                )
            with right:
                st.markdown("### Monthly trend diagnostics")
                if trend_table.empty:
                    st.info("Not enough monthly data for trend diagnostics.")
                else:
                    st.line_chart(
                        trend_table.set_index("Month")[["Spending", "Earnings", "Net"]]
                    )
                    st.dataframe(trend_table.tail(12), use_container_width=True)

            if scenario.empty:
                st.info("No spend categories available for scenario generation.")
            else:
                planned_cut = float(scenario["SuggestedCutCHF"].sum())
                potential_cut = float(scenario["MaxCutCHF"].sum())
                coverage = (planned_cut / float(monthly_target) * 100.0) if monthly_target > 0 else 0.0
                k1, k2, k3 = st.columns(3)
                k1.metric("Planned cut (CHF)", f"{planned_cut:,.2f}")
                k2.metric("Max potential cut (CHF)", f"{potential_cut:,.2f}")
                k3.metric("Target coverage", f"{coverage:.1f}%")
                st.bar_chart(scenario.set_index("Category")[["SuggestedCutCHF", "MaxCutCHF"]])
                st.dataframe(scenario, use_container_width=True, hide_index=True)

            st.markdown("### Category momentum (latest month vs prior)")
            if momentum_table.empty:
                st.info("Not enough month-over-month category data yet.")
            else:
                st.dataframe(momentum_table.head(20), use_container_width=True, hide_index=True)
        with tab_deep:
            st.caption(
                f"Settings: top {int(concentration_top_n)} merchants/sources, "
                f"run-rate {int(run_rate_lookback)} months, "
                f"volatility min {int(volatility_min_months)} months."
            )
            st.markdown("### Cashflow stability")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Negative month ratio", f"{stability_metrics['negative_month_ratio_pct']:.1f}%")
            d2.metric("Net monthly volatility", f"{stability_metrics['net_std_monthly']:,.0f}")
            d3.metric("Longest negative streak", f"{int(stability_metrics['longest_negative_streak'])}")
            d4.metric("Max drawdown (CHF)", f"{stability_metrics['max_drawdown_monthly_net']:,.0f}")

            rr1, rr2, rr3 = st.columns(3)
            rr1.metric("Run-rate annual spending", f"{run_rate['projected_annual_spending']:,.0f}")
            rr2.metric("Run-rate annual earnings", f"{run_rate['projected_annual_earnings']:,.0f}")
            rr3.metric("Run-rate annual net", f"{run_rate['projected_annual_net']:,.0f}")

            left, right = st.columns(2)
            with left:
                st.markdown("### Merchant concentration")
                if merchant_concentration.empty:
                    st.info("No spending concentration data.")
                else:
                    st.bar_chart(merchant_concentration.set_index("Merchant")[["SharePct"]].head(10))
                    st.dataframe(merchant_concentration, use_container_width=True, hide_index=True)

            with right:
                st.markdown("### Income concentration")
                if income_concentration.empty:
                    st.info("No income concentration data.")
                else:
                    st.bar_chart(income_concentration.set_index("Source")[["SharePct"]].head(10))
                    st.dataframe(income_concentration, use_container_width=True, hide_index=True)

            mid_left, mid_right = st.columns(2)
            with mid_left:
                st.markdown("### Weekday vs weekend split")
                if weekend_split.empty:
                    st.info("No weekday/weekend split data.")
                else:
                    st.dataframe(weekend_split, use_container_width=True, hide_index=True)
            with mid_right:
                st.markdown("### Transaction size distribution")
                st.dataframe(size_distribution, use_container_width=True, hide_index=True)

            st.markdown("### Category volatility")
            if category_volatility_table.empty:
                st.info("Not enough monthly category history for volatility analysis.")
            else:
                st.dataframe(category_volatility_table, use_container_width=True, hide_index=True)
        with tab_lab:
            _render_category_lab(enriched, lookup.get("categories", []))
        with tab_ai:
            _render_ai_coach(
                kpis=kpis,
                benchmark_table=benchmark_table,
                recommendations=recommendations,
                merchant_table=merchant_table,
                anomalies=anomalies,
                recurring=recurring,
                action_plan=action_plan,
            )
        with tab_anom:
            render_anomalies(anomalies, dupes)
        with tab_forecast:
            render_forecast(forecast)
        with tab_plans:
            render_subscriptions(recurring, budget_table, goals_table)
    elif page == "Data & QA":
        summary_md, report_zip = build_report_pack(filtered, kpis, monthly)
        transfers = filtered[filtered["IsTransfer"]].copy()
        tab_explorer, tab_health, tab_accounts, tab_reports = st.tabs(
            ["Explorer", "Health", "Accounts", "Reports"]
        )
        with tab_explorer:
            render_data_explorer(filtered, source_context)
        with tab_health:
            render_data_health(health_table, quality)
        with tab_accounts:
            render_accounts(accounts, transfers)
        with tab_reports:
            render_report_pack(summary_md, report_zip)


if __name__ == "__main__":
    main()
