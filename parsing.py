"""Transaction loading and normalization helpers."""

import csv
import datetime
import io
from typing import Any

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")
_DEFAULT_HEADER_ROW = 9
_EXPECTED_HEADER_COLUMNS = {
    "Abschlussdatum",
    "Abschlusszeit",
    "Buchungsdatum",
    "Valutadatum",
    "Währung",
    "Belastung",
    "Gutschrift",
    "Beschreibung1",
    "Fussnoten",
}
_HEADER_ALIASES = {
    "abschlussdatum": "Abschlussdatum",
    "abschluss": "Abschlussdatum",
    "trade date": "Abschlussdatum",
    "tradedate": "Abschlussdatum",
    "abschlusszeit": "Abschlusszeit",
    "trade time": "Abschlusszeit",
    "tradetime": "Abschlusszeit",
    "buchungsdatum": "Buchungsdatum",
    "booking date": "Buchungsdatum",
    "bookingdate": "Buchungsdatum",
    "valutadatum": "Valutadatum",
    "valuta": "Valutadatum",
    "value date": "Valutadatum",
    "valuedate": "Valutadatum",
    "währung": "Währung",
    "whrg.": "Währung",
    "currency": "Währung",
    "ccy.": "Währung",
    "ccy": "Währung",
    "belastung": "Belastung",
    "debit": "Belastung",
    "debitamount": "Belastung",
    "debit amount": "Belastung",
    "gutschrift": "Gutschrift",
    "credit": "Gutschrift",
    "creditamount": "Gutschrift",
    "credit amount": "Gutschrift",
    "transaktions-nr.": "Transaktions-Nr.",
    "transaktions-nr": "Transaktions-Nr.",
    "transaction no.": "Transaktions-Nr.",
    "transaction no": "Transaktions-Nr.",
    "transactionnr": "Transaktions-Nr.",
    "transaction nr": "Transaktions-Nr.",
    "beschreibung1": "Beschreibung1",
    "description": "Beschreibung1",
    "description 1": "Beschreibung1",
    "description1": "Beschreibung1",
    "beschreibung2": "Beschreibung2",
    "description 2": "Beschreibung2",
    "description2": "Beschreibung2",
    "beschreibung3": "Beschreibung3",
    "description 3": "Beschreibung3",
    "description3": "Beschreibung3",
    "fussnoten": "Fussnoten",
    "fußnoten": "Fussnoten",
    "footnotes": "Fussnoten",
    "footnote": "Fussnoten",
    "einzelbetrag": "Einzelbetrag",
    "individual amount": "Einzelbetrag",
    "individualamount": "Einzelbetrag",
    "saldo": "Saldo",
    "balance": "Saldo",
}
_CONTEXT_LABELS = {
    "kontonummer": "StatementAccountNumber",
    "account number": "StatementAccountNumber",
    "iban": "StatementIBAN",
    "von": "StatementFrom",
    "from": "StatementFrom",
    "date from": "StatementFrom",
    "bis": "StatementTo",
    "until": "StatementTo",
    "date to": "StatementTo",
    "bewertet in": "StatementCurrency",
    "valued in": "StatementCurrency",
    "anzahl transaktionen in diesem zeitraum": "StatementTransactions",
    "numbers of transactions in this period": "StatementTransactions",
    "number of transactions in this period": "StatementTransactions",
}


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


def _rewind(uploaded_file: Any) -> None:
    try:
        uploaded_file.seek(0)
    except Exception:
        return


def _normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    return text.replace(":", "")


def _canonical_column_name(label: Any) -> str:
    normalized = _normalize_label(label)
    return _HEADER_ALIASES.get(normalized, str(label).strip())


def _parse_datetime_value(value: Any):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return parsed


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce", dayfirst=True)
    return parsed


def _clean_csv_text(raw_text: str) -> str:
    # Banana UBS importer applies these cleanups for malformed exports.
    cleaned = raw_text.replace('"""', '"')
    cleaned = cleaned.replace(" ; ", " ")
    return cleaned


def _read_csv_text(uploaded_file: Any) -> str:
    _rewind(uploaded_file)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        return _clean_csv_text(raw.decode("utf-8-sig", errors="ignore"))
    return _clean_csv_text(str(raw))


def _read_preview(uploaded_file: Any, name: str, csv_text: str = "") -> pd.DataFrame:
    _rewind(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, sheet_name=0, header=None, nrows=40)
    if name.endswith(".xls"):
        return pd.read_excel(uploaded_file, sheet_name=0, header=None, nrows=40, engine="xlrd")
    if name.endswith(".csv"):
        if not csv_text:
            csv_text = _read_csv_text(uploaded_file)
        reader = csv.reader(io.StringIO(csv_text), delimiter=";", quotechar='"')
        rows = []
        for idx, row in enumerate(reader):
            rows.append(row)
            if idx >= 39:
                break
        if not rows:
            return pd.DataFrame()
        max_cols = max(len(row) for row in rows)
        padded = [row + [None] * (max_cols - len(row)) for row in rows]
        return pd.DataFrame(padded)
    raise ValueError(
        f"Unsupported file type: {name or '<unknown>'}. Supported: csv, xlsx, xls."
    )


def _detect_header_row(preview: pd.DataFrame) -> int:
    best_row = _DEFAULT_HEADER_ROW
    best_score = -1

    for idx, row in preview.iterrows():
        tokens = {_canonical_column_name(value) for value in row.tolist() if pd.notna(value)}
        score = len(tokens & _EXPECTED_HEADER_COLUMNS)
        if score > best_score and "Abschlussdatum" in tokens:
            best_score = score
            best_row = int(idx)

    return best_row


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = []
    used_columns: set[str] = set()
    for col in df.columns:
        canonical = _canonical_column_name(col)
        if canonical in used_columns and canonical != str(col).strip():
            renamed_columns.append(str(col).strip())
        else:
            renamed_columns.append(canonical)
            used_columns.add(canonical)
    out = df.copy()
    out.columns = renamed_columns
    return out


def _extract_statement_context(preview: pd.DataFrame) -> dict[str, Any]:
    if preview.empty or preview.shape[1] < 2:
        return {}

    context: dict[str, Any] = {}
    for _, row in preview.iloc[:12].iterrows():
        label = _normalize_label(row.iloc[0])
        if label in _CONTEXT_LABELS and pd.notna(row.iloc[1]):
            context[_CONTEXT_LABELS[label]] = row.iloc[1]

    if "StatementFrom" in context:
        context["StatementFrom"] = _parse_datetime_value(context["StatementFrom"])
    if "StatementTo" in context:
        context["StatementTo"] = _parse_datetime_value(context["StatementTo"])
    if "StatementTransactions" in context:
        context["StatementTransactions"] = pd.to_numeric(
            context["StatementTransactions"], errors="coerce"
        )

    return context


def _load_raw_statement(uploaded_file: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    name = str(getattr(uploaded_file, "name", "")).lower()
    csv_text = _read_csv_text(uploaded_file) if name.endswith(".csv") else ""
    preview = _read_preview(uploaded_file, name, csv_text=csv_text)
    context = _extract_statement_context(preview)
    header_row = _detect_header_row(preview)

    if name.endswith(".xlsx"):
        _rewind(uploaded_file)
        return _normalize_headers(pd.read_excel(uploaded_file, sheet_name=0, header=header_row)), context
    if name.endswith(".xls"):
        # Legacy .xls needs xlrd.
        _rewind(uploaded_file)
        return _normalize_headers(
            pd.read_excel(uploaded_file, sheet_name=0, header=header_row, engine="xlrd")
        ), context
    if name.endswith(".csv"):
        lines = csv_text.splitlines()
        if header_row >= len(lines):
            raise ValueError("Could not detect a valid transaction header row in CSV.")
        data_text = "\n".join(lines[header_row:])
        return _normalize_headers(
            pd.read_csv(io.StringIO(data_text), sep=";", header=0, engine="python")
        ), context
    raise ValueError(f"Unsupported file type: {name or '<unknown>'}. Supported: csv, xlsx, xls.")


def _build_sort_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    date_part = date_series.dt.strftime("%Y-%m-%d").fillna("")
    time_part = (
        time_series.fillna("").astype(str).str.strip().str.split(".").str[0].replace("nan", "")
    )
    return pd.to_datetime((date_part + " " + time_part).str.strip(), errors="coerce")


def _drop_statement_noise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    unnamed_cols = [col for col in out.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        extra_text = out[unnamed_cols].apply(
            lambda row: " | ".join(
                [
                    str(value).strip()
                    for value in row.tolist()
                    if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan"
                ]
            ),
            axis=1,
        )
        if "Fussnoten" not in out.columns:
            out["Fussnoten"] = ""
        out["Fussnoten"] = out.apply(
            lambda row: " | ".join(
                [
                    text
                    for text in [str(row.get("Fussnoten", "")).strip(), str(extra_text.loc[row.name]).strip()]
                    if text and text.lower() != "nan"
                ]
            ),
            axis=1,
        )

    drop_cols = [
        col
        for col in out.columns
        if str(col).startswith("Unnamed") or col in ("Einzelbetrag", "Saldo")
    ]
    return out.drop(columns=[col for col in drop_cols if col in out.columns])


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
    raw_df, context = _load_raw_statement(uploaded_file)
    df = _normalize_amount_columns(_drop_statement_noise(raw_df))

    for col in ["Beschreibung1", "Beschreibung2", "Beschreibung3", "Fussnoten"]:
        if col not in df.columns:
            df[col] = ""

    df["Debit"] = (
        df.get("Belastung", pd.Series(dtype=float)).fillna(0).apply(lambda x: -x if x < 0 else x)
    )
    df["Credit"] = df.get("Gutschrift", pd.Series(dtype=float)).fillna(0)
    df["Merchant"] = df.get("Beschreibung1", "").astype(str).str.strip()
    df["Location"] = df.get("Beschreibung3", "").fillna(df.get("Beschreibung2", "")).astype(str)
    df["Date"] = _parse_datetime_series(df.get("Abschlussdatum"))
    df["Time"] = df.get("Abschlusszeit", "").fillna("").astype(str)
    df["TimeOfDay"] = df["Time"].apply(classify_time_of_day)
    df["SortDateTime"] = _build_sort_datetime(df["Date"], df["Time"])
    df["SourceFile"] = getattr(uploaded_file, "name", "uploaded_file")
    for key, value in context.items():
        df[key] = value

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
