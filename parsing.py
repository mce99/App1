"""Transaction loading and normalization helpers."""

import datetime
from typing import Any

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")


def classify_time_of_day(time_str: str) -> str:
    """Return a coarse time bucket based on HH:MM:SS."""
    try:
        t_obj = datetime.datetime.strptime(str(time_str).split(".")[0], "%H:%M:%S").time()
    except Exception:
        try:
            t_obj = datetime.datetime.strptime(str(time_str), "%H:%M:%S").time()
        except Exception:
            return "Unknown"

    if datetime.time(5, 0) <= t_obj < datetime.time(11, 0):
        return "Morning"
    if datetime.time(11, 0) <= t_obj < datetime.time(17, 0):
        return "Afternoon"
    if datetime.time(17, 0) <= t_obj < datetime.time(21, 0):
        return "Evening"
    return "Night"


def _load_raw_statement(uploaded_file: Any) -> pd.DataFrame:
    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, sheet_name=0, skiprows=9)
    if name.endswith(".xls"):
        # Legacy .xls needs xlrd.
        return pd.read_excel(uploaded_file, sheet_name=0, skiprows=9, engine="xlrd")
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, sep=";", skiprows=8)
    raise ValueError(
        f"Unsupported file type: {name or '<unknown>'}. Supported: csv, xlsx, xls."
    )


def _build_sort_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    date_part = date_series.dt.strftime("%Y-%m-%d").fillna("")
    time_part = (
        time_series.fillna("").astype(str).str.strip().str.split(".").str[0].replace("nan", "")
    )
    return pd.to_datetime((date_part + " " + time_part).str.strip(), errors="coerce")


def _drop_statement_noise(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in df.columns
        if str(col).startswith("Unnamed") or col in ("Einzelbetrag", "Saldo")
    ]
    return df.drop(columns=[col for col in drop_cols if col in df.columns])


def _normalize_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Belastung", "Gutschrift"]:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace("'", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_transactions(uploaded_file: Any) -> pd.DataFrame:
    """Load and preprocess transactions from semicolon CSV or Excel exports."""
    df = _normalize_amount_columns(_drop_statement_noise(_load_raw_statement(uploaded_file)))

    for col in ["Beschreibung1", "Beschreibung2", "Beschreibung3", "Fussnoten"]:
        if col not in df.columns:
            df[col] = ""

    df["Debit"] = (
        df.get("Belastung", pd.Series(dtype=float)).fillna(0).apply(lambda x: -x if x < 0 else x)
    )
    df["Credit"] = df.get("Gutschrift", pd.Series(dtype=float)).fillna(0)
    df["Merchant"] = df.get("Beschreibung1", "").astype(str).str.strip()
    df["Location"] = df.get("Beschreibung3", "").fillna(df.get("Beschreibung2", "")).astype(str)
    df["Date"] = pd.to_datetime(df.get("Abschlussdatum"), errors="coerce", dayfirst=True)
    df["Time"] = df.get("Abschlusszeit", "").fillna("").astype(str)
    df["TimeOfDay"] = df["Time"].apply(classify_time_of_day)
    df["SortDateTime"] = _build_sort_datetime(df["Date"], df["Time"])
    df["SourceFile"] = getattr(uploaded_file, "name", "uploaded_file")

    return df


def deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove likely duplicate transactions across overlapping statement exports."""
    dedupe_candidates = [
        "Date",
        "Time",
        "Währung",
        "Debit",
        "Credit",
        "Beschreibung1",
        "Beschreibung2",
        "Beschreibung3",
        "Fussnoten",
    ]
    subset = [col for col in dedupe_candidates if col in df.columns]
    if not subset:
        return df
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def merge_transactions(uploaded_files: list[Any], drop_duplicates: bool = True) -> pd.DataFrame:
    """Load, combine, deduplicate and sort multiple statements."""
    frames = [load_transactions(uploaded_file) for uploaded_file in uploaded_files]
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    if drop_duplicates:
        merged = deduplicate_transactions(merged)

    sort_cols = [col for col in ["SortDateTime", "Date", "Time", "SourceFile"] if col in merged.columns]
    return merged.sort_values(sort_cols, na_position="last").reset_index(drop=True)
