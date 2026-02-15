"""Streamlit application for analyzing spending and income transactions."""

import datetime
import json

import pandas as pd
import streamlit as st

from categorization import DEFAULT_KEYWORD_MAP, assign_categories
from parsing import classify_time_of_day, load_transactions


def main() -> None:
    st.title("Spending and Income Analyzer")
    st.write(
        "Upload your bank statement in CSV (semicolon-separated) or Excel format. "
        "The app will parse transactions, separate debit and credit amounts, "
        "and categorize spending using simple keyword matching. Only CHF transactions "
        "are currently supported."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=False
    )
    if uploaded_file is None:
        return

    df = load_transactions(uploaded_file)
    if df.empty:
        st.warning("No transactions found. Please check the file format.")
        return

    with st.expander("Add a manual transaction"):
        with st.form(key="manual_form"):
            m_date = st.date_input("Date", value=datetime.date.today())
            m_time = st.time_input("Time", value=datetime.datetime.now().time())
            m_currency = st.text_input("Currency", value="CHF")
            m_debit = st.number_input("Debit (negative values will be converted)", value=0.0)
            m_credit = st.number_input("Credit", value=0.0)
            m_desc1 = st.text_input("Beschreibung1 (Merchant)")
            m_desc2 = st.text_input("Beschreibung2 (Additional)")
            m_desc3 = st.text_input("Beschreibung3 (Location)")
            m_notes = st.text_input("Fussnoten (Notes)")
            submit = st.form_submit_button("Add Transaction")

        if submit:
            new_row = {
                "Abschlussdatum": m_date,
                "Abschlusszeit": m_time.strftime("%H:%M:%S"),
                "Währung": m_currency,
                "Belastung": m_debit,
                "Gutschrift": m_credit,
                "Beschreibung1": m_desc1,
                "Beschreibung2": m_desc2,
                "Beschreibung3": m_desc3,
                "Fussnoten": m_notes,
            }
            temp = pd.DataFrame([new_row])
            for col in ["Belastung", "Gutschrift"]:
                temp[col] = pd.to_numeric(temp[col], errors="coerce")
            temp["Debit"] = temp["Belastung"].fillna(0).apply(lambda x: -x if x < 0 else x)
            temp["Credit"] = temp["Gutschrift"].fillna(0)
            temp["Merchant"] = temp.get("Beschreibung1", "").astype(str).str.strip()
            temp["Location"] = temp.get("Beschreibung3", "").fillna(temp.get("Beschreibung2", "")).astype(str)
            temp["Date"] = pd.to_datetime(temp.get("Abschlussdatum"), errors="coerce")
            temp["Time"] = temp.get("Abschlusszeit").astype(str)
            temp["TimeOfDay"] = temp["Time"].apply(classify_time_of_day)
            df = pd.concat([df, temp], ignore_index=True)

    st.subheader("Currency Conversion")
    st.write(
        "Enter a JSON object mapping currency codes to CHF conversion rates "
        '(e.g., {"EUR": 0.96, "USD": 0.89}). These rates will be used to convert '
        "Debit and Credit amounts into Swiss Francs. Transactions in unknown currencies "
        "will be skipped from totals."
    )
    conv_default = {"CHF": 1.0, "EUR": 1.0, "USD": 1.0}
    conv_json = st.text_area("Currency rates (CHF per unit)", value=json.dumps(conv_default, indent=2))
    try:
        conv_rates = json.loads(conv_json)
    except Exception as exc:
        st.error(f"Invalid currency JSON: {exc}")
        conv_rates = {"CHF": 1.0}

    df["DebitCHF"] = df.apply(
        lambda row: row["Debit"] * conv_rates.get(str(row.get("Währung", "CHF")), float("nan")),
        axis=1,
    )
    df["CreditCHF"] = df.apply(
        lambda row: row["Credit"] * conv_rates.get(str(row.get("Währung", "CHF")), float("nan")),
        axis=1,
    )

    st.subheader("Customize Categories (Optional)")
    st.write(
        "Provide a JSON object mapping category names to lists of keywords. "
        "This will override the default keyword mapping used for categorization."
    )
    cat_json = st.text_area("Keyword map", value=json.dumps(DEFAULT_KEYWORD_MAP, indent=2))
    try:
        keyword_map = json.loads(cat_json)
        keyword_map = {
            str(key): [str(item).upper() for item in values] for key, values in keyword_map.items()
        }
    except Exception as exc:
        st.error(f"Invalid keyword map JSON: {exc}")
        keyword_map = DEFAULT_KEYWORD_MAP

    df = assign_categories(df, keyword_map)

    st.subheader("Preview of Transactions")
    preview_cols = [
        "Date",
        "Time",
        "Merchant",
        "Location",
        "Währung",
        "Debit",
        "Credit",
        "DebitCHF",
        "CreditCHF",
        "Category",
    ]
    st.dataframe(df[preview_cols].head(20))

    total_debit = df["DebitCHF"].sum(skipna=True)
    total_credit = df["CreditCHF"].sum(skipna=True)
    st.metric("Total Debit (CHF)", f"{total_debit:,.2f}")
    st.metric("Total Credit (CHF)", f"{total_credit:,.2f}")

    st.subheader("Weekly Summary")
    if df["Date"].notna().any():
        today = pd.Timestamp.today().normalize()
        last_week = today - pd.Timedelta(days=7)
        recent_df = df[df["Date"] >= last_week]
        if not recent_df.empty:
            summary = recent_df.groupby("Category")["DebitCHF"].sum().sort_values(ascending=False)
            summary_text = ", ".join([f"{cat}: CHF {val:,.2f}" for cat, val in summary.items()])
            st.write(f"In the last 7 days you spent: {summary_text}.")
        else:
            st.write("No transactions in the last 7 days.")

    debit_by_cat = df.groupby("Category")["DebitCHF"].sum().sort_values(ascending=False)
    credit_by_cat = df.groupby("Category")["CreditCHF"].sum().sort_values(ascending=False)

    st.subheader("Spending by Category (Debit in CHF)")
    st.bar_chart(debit_by_cat)

    st.subheader("Income by Category (Credit in CHF)")
    st.bar_chart(credit_by_cat)

    st.subheader("Transactions by Time of Day")
    st.bar_chart(df["TimeOfDay"].value_counts().sort_index())

    st.subheader("Monthly Debit and Credit Totals (CHF)")
    if df["Date"].notna().any():
        df_month = df.copy()
        df_month["Month"] = df_month["Date"].dt.to_period("M").astype(str)
        monthly_debit = df_month.groupby("Month")["DebitCHF"].sum()
        monthly_credit = df_month.groupby("Month")["CreditCHF"].sum()
        st.line_chart(pd.DataFrame({"Debit": monthly_debit, "Credit": monthly_credit}))

    st.subheader("Budget vs. Actual (CHF)")
    st.write(
        "Provide a JSON object mapping category names to monthly budget amounts. "
        "The app will compare your total debit spending per category against these budgets and highlight overspend."
    )
    default_budget = {cat: round(total_debit / max(len(debit_by_cat), 1), 2) for cat in debit_by_cat.index}
    budget_json = st.text_area(
        "Enter budgets (CHF) per category", value=json.dumps(default_budget, indent=2), key="budget_text"
    )
    try:
        budgets = json.loads(budget_json)
        budget_df = pd.DataFrame({"Actual": debit_by_cat, "Budget": pd.Series(budgets)})
        budget_df["Budget"] = budget_df["Budget"].fillna(0)
        budget_df["Overspend"] = budget_df["Actual"] - budget_df["Budget"]
        st.dataframe(budget_df)
    except Exception as exc:
        st.error(f"Invalid budget JSON: {exc}")

    processed_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download processed data as CSV",
        data=processed_csv,
        file_name="processed_transactions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
