"""Transaction loading and normalization helpers."""

import datetime
from typing import Any

import pandas as pd


def classify_time_of_day(time_str: str) -> str:
    """Return a coarse time bucket based on HH:MM:SS."""
    try:
        t_obj = datetime.datetime.strptime(time_str.split(".")[0], "%H:%M:%S").time()
    except Exception:
        try:
            t_obj = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
        except Exception:
            return "Unknown"

    if datetime.time(5, 0) <= t_obj < datetime.time(11, 0):
        return "Morning"
    if datetime.time(11, 0) <= t_obj < datetime.time(17, 0):
        return "Afternoon"
    if datetime.time(17, 0) <= t_obj < datetime.time(21, 0):
        return "Evening"
    return "Night"


def load_transactions(uploaded_file: Any) -> pd.DataFrame:
    """Load and preprocess transactions from CSV or Excel export."""
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, sheet_name=0, skiprows=9)
    else:
        df = pd.read_csv(uploaded_file, sep=";", skiprows=8)

    drop_cols = [
        col
        for col in df.columns
        if str(col).startswith("Unnamed") or col in ("Einzelbetrag", "Saldo")
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

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

    df["Debit"] = (
        df.get("Belastung", pd.Series(dtype=float)).fillna(0).apply(lambda x: -x if x < 0 else x)
    )
    df["Credit"] = df.get("Gutschrift", pd.Series(dtype=float)).fillna(0)
    df["Merchant"] = df.get("Beschreibung1", "").astype(str).str.strip()
    df["Location"] = df.get("Beschreibung3", "").fillna(df.get("Beschreibung2", "")).astype(str)
    df["Date"] = pd.to_datetime(df.get("Abschlussdatum"), errors="coerce")
    df["Time"] = df.get("Abschlusszeit").astype(str)
    df["TimeOfDay"] = df["Time"].apply(classify_time_of_day)

    return df
