"""
Streamlit application for analyzing spending and income transactions.

This app can load CSV (semicolon‑separated) or Excel statements with Swiss bank formatting.
It uses `Abschlussdatum` and `Abschlusszeit` as the transaction date and time,
filters for CHF currency only, and keeps debit (Belastung) and credit (Gutschrift)
amounts separate. A simple rule‑based categorization is applied using keywords
from the description fields (`Beschreibung1`, `Beschreibung2`, `Beschreibung3`, `Fussnoten`).

The app displays a preview of the data, summary metrics, and bar charts of
debit and credit totals by category. This is intended as a starting point and
can be extended with additional filters or visualization options.
"""

import streamlit as st
import pandas as pd
import json
import datetime


# Default keyword map used for rule‑based categorization. Users can override this
# via a JSON text area in the UI.
DEFAULT_KEYWORD_MAP = {
    "Food & Drink": [
        "SUSHI",
        "RESTAURANT",
        "COOP",
        "SPAR",
        "MIGROL",
        "METZGEREI",
        "CONFISERIE",
        "PIZZA",
        "UBER   * EATS",
        "UBER   *ONE",
        "STARBUCKS",
        "CAFE",
        "MCDON",
        "KINTARO",
        "MOREIRA",
        "GOURMET",
        "P&B",
    ],
    "Transport": [
        "UBER",
        "TAXI",
        "SOCAR",
        "AGROLA",
        "TANKSTELLE",
        "PARK",
        "PARKING",
        "SBB",
        "CAR",
        "ENI",
    ],
    "Utilities & Bills": [
        "SWISSCOM",
        "POST",
        "E-BANKING",
        "VERGUTUNGS",
        "FEDEX",
        "ELECTRIC",
        "POWER",
        "INSURANCE",
        "KANTON",
        "STEUER",
    ],
    "Shopping & Retail": [
        "ZARA",
        "H&M",
        "STORE",
        "SHOP",
        "MOUNTAIN AIR",
        "LEVIS",
        "ORIGINAL",
        "COOP-",
        "MIGROS",
        "MIGROL",
    ],
    "Income & Transfers": [
        "METALLUM",
        "XXXX",
        "BANK",
        "REVOLUT",
        "ENKELMANN",
        "TRANSFER",
        "UBS SWITZERLAND",
    ],
    "Entertainment & Leisure": [
        "FANVUE",
        "BILL",
        "APPLE.COM",
        "NETFLIX",
        "GYM",
        "SPA",
        "ART",
        "MUSEUM",
    ],
}

def assign_categories(df: pd.DataFrame, keyword_map: dict) -> pd.DataFrame:
    """Assign categories to transactions using a keyword map.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing transactions. Must have the description fields.
    keyword_map: dict
        Mapping of category names to lists of keywords.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'Category' column assigned.
    """
    def assign(row: pd.Series) -> str:
        desc_fields = [
            str(row.get("Beschreibung1", "")),
            str(row.get("Beschreibung2", "")),
            str(row.get("Beschreibung3", "")),
            str(row.get("Fussnoten", "")),
        ]
        description = " ".join(desc_fields).upper()
        for cat, kws in keyword_map.items():
            for kw in kws:
                if kw in description:
                    return cat
        return "Other"

    df = df.copy()
    df["Category"] = df.apply(assign, axis=1)
    return df


def load_transactions(uploaded_file: st.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Load and preprocess transactions from a CSV or Excel file.

    Parameters
    ----------
    uploaded_file: st.uploaded_file_manager.UploadedFile
        The file object returned by st.file_uploader.

    Returns
    -------
    pd.DataFrame
        A DataFrame with cleaned and enriched transaction data.
    """
    # Determine format by file extension
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, sheet_name=0, skiprows=9)
    else:
        # assume semicolon-separated CSV with 8 header rows
        df = pd.read_csv(uploaded_file, sep=";", skiprows=8)

    # Drop empty or unnamed columns
    drop_cols = [
        col
        for col in df.columns
        if str(col).startswith("Unnamed") or col in ("Einzelbetrag", "Saldo")
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert numeric fields for Belastung and Gutschrift (keep raw values; conversion later)
    for col in ["Belastung", "Gutschrift"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace("'", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create separate Debit and Credit columns (raw values; still in original currency)
    df["Debit"] = df.get("Belastung", pd.Series(dtype=float)).fillna(0).apply(lambda x: -x if x < 0 else x)
    df["Credit"] = df.get("Gutschrift", pd.Series(dtype=float)).fillna(0)

    # Merchant and location extraction
    df["Merchant"] = df.get("Beschreibung1", "").astype(str).str.strip()
    # Prefer Beschreibung3 for location; fall back to Beschreibung2
    df["Location"] = df.get("Beschreibung3", "").fillna(df.get("Beschreibung2", "")).astype(str)

    # Parse date and time
    df["Date"] = pd.to_datetime(df.get("Abschlussdatum"), errors="coerce")
    df["Time"] = df.get("Abschlusszeit").astype(str)

    # Classify time of day for each transaction based on Abschlusszeit
    def classify_time_of_day(time_str: str) -> str:
        """Return a coarse time‑of‑day bucket (Morning, Afternoon, Evening, Night)."""
        try:
            # Some time strings may include milliseconds; split off decimals
            t_obj = datetime.datetime.strptime(time_str.split(".")[0], "%H:%M:%S").time()
        except Exception:
            try:
                t_obj = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
            except Exception:
                return "Unknown"
        if datetime.time(5, 0) <= t_obj < datetime.time(11, 0):
            return "Morning"
        elif datetime.time(11, 0) <= t_obj < datetime.time(17, 0):
            return "Afternoon"
        elif datetime.time(17, 0) <= t_obj < datetime.time(21, 0):
            return "Evening"
        else:
            return "Night"

    df["TimeOfDay"] = df["Time"].apply(classify_time_of_day)

    return df


def main() -> None:
    st.title("Spending and Income Analyzer")
    st.write(
        "Upload your bank statement in CSV (semicolon‑separated) or Excel format. "
        "The app will parse transactions, separate debit and credit amounts, "
        "and categorize spending using simple keyword matching. Only CHF transactions "
        "are currently supported."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=False
    )
    if uploaded_file is not None:
        # Load and preprocess
        df = load_transactions(uploaded_file)

        if df.empty:
            st.warning("No transactions found. Please check the file format.")
            return

        # Allow user to manually add a new transaction
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
                # Create a DataFrame for the manual entry and apply minimal preprocessing
                temp = pd.DataFrame([new_row])
                # Convert numeric fields
                for col in ["Belastung", "Gutschrift"]:
                    temp[col] = pd.to_numeric(temp[col], errors="coerce")
                temp["Debit"] = temp["Belastung"].fillna(0).apply(lambda x: -x if x < 0 else x)
                temp["Credit"] = temp["Gutschrift"].fillna(0)
                temp["Merchant"] = temp.get("Beschreibung1", "").astype(str).str.strip()
                temp["Location"] = temp.get("Beschreibung3", "").fillna(temp.get("Beschreibung2", "")).astype(str)
                temp["Date"] = pd.to_datetime(temp.get("Abschlussdatum"), errors="coerce")
                temp["Time"] = temp.get("Abschlusszeit").astype(str)
                # Time of day classification
                def classify_manual(tstr: str) -> str:
                    try:
                        t_obj = datetime.datetime.strptime(tstr, "%H:%M:%S").time()
                    except Exception:
                        return "Unknown"
                    if datetime.time(5, 0) <= t_obj < datetime.time(11, 0):
                        return "Morning"
                    elif datetime.time(11, 0) <= t_obj < datetime.time(17, 0):
                        return "Afternoon"
                    elif datetime.time(17, 0) <= t_obj < datetime.time(21, 0):
                        return "Evening"
                    else:
                        return "Night"
                temp["TimeOfDay"] = temp["Time"].apply(classify_manual)
                df = pd.concat([df, temp], ignore_index=True)

        # Currency conversion mapping
        st.subheader("Currency Conversion")
        st.write(
            "Enter a JSON object mapping currency codes to CHF conversion rates (e.g., {\"EUR\": 0.96, \"USD\": 0.89}). "
            "These rates will be used to convert Debit and Credit amounts into Swiss Francs. "
            "Transactions in unknown currencies will be skipped from totals."
        )
        conv_default = {"CHF": 1.0, "EUR": 1.0, "USD": 1.0}
        conv_json = st.text_area(
            "Currency rates (CHF per unit)", value=json.dumps(conv_default, indent=2)
        )
        try:
            conv_rates = json.loads(conv_json)
        except Exception as e:
            st.error(f"Invalid currency JSON: {e}")
            conv_rates = {"CHF": 1.0}

        # Convert amounts to CHF
        df["DebitCHF"] = df.apply(
            lambda row: row["Debit"] * conv_rates.get(str(row.get("Währung", "CHF")), float('nan')),
            axis=1,
        )
        df["CreditCHF"] = df.apply(
            lambda row: row["Credit"] * conv_rates.get(str(row.get("Währung", "CHF")), float('nan')),
            axis=1,
        )

        # Prompt for keyword map customization
        st.subheader("Customize Categories (Optional)")
        st.write(
            "Provide a JSON object mapping category names to lists of keywords. "
            "This will override the default keyword mapping used for categorization."
        )
        cat_json = st.text_area("Keyword map", value=json.dumps(DEFAULT_KEYWORD_MAP, indent=2))
        try:
            keyword_map = json.loads(cat_json)
            # Ensure keys are strings and values are lists
            keyword_map = {str(k): [str(item).upper() for item in v] for k, v in keyword_map.items()}
        except Exception as e:
            st.error(f"Invalid keyword map JSON: {e}")
            keyword_map = DEFAULT_KEYWORD_MAP

        # Assign categories using custom map
        df = assign_categories(df, keyword_map)

        # Display preview
        st.subheader("Preview of Transactions")
        preview_cols = ["Date", "Time", "Merchant", "Location", "Währung", "Debit", "Credit", "DebitCHF", "CreditCHF", "Category"]
        st.dataframe(df[preview_cols].head(20))

        # Summary metrics (in CHF)
        total_debit = df["DebitCHF"].sum(skipna=True)
        total_credit = df["CreditCHF"].sum(skipna=True)
        st.metric("Total Debit (CHF)", f"{total_debit:,.2f}")
        st.metric("Total Credit (CHF)", f"{total_credit:,.2f}")

        # Weekly summary
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

        # Aggregate by category (CHF)
        debit_by_cat = df.groupby("Category")["DebitCHF"].sum().sort_values(ascending=False)
        credit_by_cat = df.groupby("Category")["CreditCHF"].sum().sort_values(ascending=False)

        st.subheader("Spending by Category (Debit in CHF)")
        st.bar_chart(debit_by_cat)

        st.subheader("Income by Category (Credit in CHF)")
        st.bar_chart(credit_by_cat)

        # Time‑of‑day distribution for all transactions
        st.subheader("Transactions by Time of Day")
        time_counts = df["TimeOfDay"].value_counts().sort_index()
        st.bar_chart(time_counts)

        # Monthly summaries of debit and credit
        st.subheader("Monthly Debit and Credit Totals (CHF)")
        if df["Date"].notna().any():
            df_month = df.copy()
            df_month["Month"] = df_month["Date"].dt.to_period("M").astype(str)
            monthly_debit = df_month.groupby("Month")["DebitCHF"].sum()
            monthly_credit = df_month.groupby("Month")["CreditCHF"].sum()
            monthly_df = pd.DataFrame({"Debit": monthly_debit, "Credit": monthly_credit})
            st.line_chart(monthly_df)

        # Simple budgeting: allow user to specify budgets in JSON format (CHF)
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
            # Compute overspend metrics
            budget_df = pd.DataFrame({"Actual": debit_by_cat, "Budget": pd.Series(budgets)})
            budget_df["Budget"] = budget_df["Budget"].fillna(0)
            budget_df["Overspend"] = budget_df["Actual"] - budget_df["Budget"]
            st.dataframe(budget_df)
        except Exception as e:
            st.error(f"Invalid budget JSON: {e}")

        # Provide a downloadable processed CSV (optional)
        processed_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download processed data as CSV",
            data=processed_csv,
            file_name="processed_transactions.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()