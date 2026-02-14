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


def assign_category(row: pd.Series) -> str:
    """Assign a category based on description fields.

    Looks at all description-related fields, converts them to uppercase,
    and matches keywords to predefined categories. If no keyword is found,
    returns 'Other'.

    Parameters
    ----------
    row: pd.Series
        A row from the transaction DataFrame.

    Returns
    -------
    str
        A category name.
    """
    # Concatenate all description fields into one string
    desc_fields = [
        str(row.get("Beschreibung1", "")),
        str(row.get("Beschreibung2", "")),
        str(row.get("Beschreibung3", "")),
        str(row.get("Fussnoten", "")),
    ]
    description = " ".join(desc_fields).upper()

    # Define keyword lists for categories
    keyword_map = {
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

    # Attempt to match keywords
    for category, keywords in keyword_map.items():
        for kw in keywords:
            if kw in description:
                return category
    return "Other"


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

    # Filter CHF only
    if "Währung" in df.columns:
        df = df[df["Währung"] == "CHF"].copy()

    # Convert numeric fields for Belastung and Gutschrift
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

    # Create separate Debit and Credit columns
    df["Debit"] = df["Belastung"].fillna(0).apply(lambda x: -x if x < 0 else x)
    df["Credit"] = df["Gutschrift"].fillna(0)

    # Assign category using descriptions
    df["Category"] = df.apply(assign_category, axis=1)

    # Merchant and location extraction
    df["Merchant"] = df.get("Beschreibung1", "").astype(str).str.strip()
    # Prefer Beschreibung3 for location; fall back to Beschreibung2
    df["Location"] = df.get("Beschreibung3", "").fillna(df.get("Beschreibung2", "")).astype(str)

    # Parse date and time
    df["Date"] = pd.to_datetime(df.get("Abschlussdatum"), errors="coerce")
    df["Time"] = df.get("Abschlusszeit").astype(str)

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
            st.warning("No transactions found after filtering. Please check the file format or currency.")
            return

        # Display preview
        st.subheader("Preview of Transactions")
        st.dataframe(df[["Date", "Time", "Merchant", "Location", "Debit", "Credit", "Category"]].head(20))

        # Summary metrics
        total_debit = df["Debit"].sum()
        total_credit = df["Credit"].sum()
        st.metric("Total Debit (CHF)", f"{total_debit:,.2f}")
        st.metric("Total Credit (CHF)", f"{total_credit:,.2f}")

        # Aggregate by category
        debit_by_cat = df.groupby("Category")["Debit"].sum().sort_values(ascending=False)
        credit_by_cat = df.groupby("Category")["Credit"].sum().sort_values(ascending=False)

        st.subheader("Spending by Category (Debit)")
        st.bar_chart(debit_by_cat)

        st.subheader("Income by Category (Credit)")
        st.bar_chart(credit_by_cat)

        # Provide a downloadable processed CSV (optional)
        processed_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download processed data as CSV", data=processed_csv, file_name="processed_transactions.csv", mime="text/csv"
        )


if __name__ == "__main__":
    main()