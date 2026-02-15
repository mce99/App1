"""Category mapping and assignment logic for transactions."""

import re

import pandas as pd


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


TRANSFER_KEYWORDS = [
    "TRANSFER",
    "UEBERTRAG",
    "ÜBERTRAG",
    "EIGENKONTO",
    "KONTOUEBERTRAG",
    "ACCOUNT TRANSFER",
    "IBAN",
    "REVOLUT",
    "TWINT",
]


INCOME_KEYWORDS = [
    "SALARY",
    "PAYROLL",
    "GEHALT",
    "LOHN",
    "BONUS",
    "DIVIDEND",
    "INTEREST",
    "ZINS",
    "PENSION",
    "AHV",
    "RENT",
    "RUECKERSTATTUNG",
    "REFUND",
]


def _score_keyword_match(description: str, merchant: str, keyword: str) -> float:
    score = 0.75
    if keyword in merchant:
        score += 0.15
    if re.search(rf"\b{re.escape(keyword)}\b", description):
        score += 0.06
    return min(score, 0.98)


def _to_float(value) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def assign_categories_with_confidence(df: pd.DataFrame, keyword_map: dict) -> pd.DataFrame:
    """Assign categories plus confidence and matched keyword metadata."""

    def assign(row: pd.Series) -> tuple[str, float, str]:
        desc_fields = [
            str(row.get("Beschreibung1", "")),
            str(row.get("Beschreibung2", "")),
            str(row.get("Beschreibung3", "")),
            str(row.get("Fussnoten", "")),
        ]
        description = " ".join(desc_fields).upper()
        merchant = str(row.get("Beschreibung1", "")).upper()
        debit = max(
            _to_float(row.get("Debit", 0.0)),
            _to_float(row.get("DebitCHF", 0.0)),
        )
        credit = max(
            _to_float(row.get("Credit", 0.0)),
            _to_float(row.get("CreditCHF", 0.0)),
        )
        outgoing = debit > 0 and credit == 0
        incoming = credit > 0 and debit == 0

        for keyword in TRANSFER_KEYWORDS:
            kw = str(keyword).upper()
            if kw and kw in description:
                return "Transfers", 0.93, f"Transfer:{kw}"

        if incoming:
            for keyword in INCOME_KEYWORDS:
                kw = str(keyword).upper()
                if kw and kw in description:
                    return "Income & Transfers", 0.95, f"Income:{kw}"

        for category, keywords in keyword_map.items():
            if outgoing and str(category) in {"Income & Transfers", "Transfers"}:
                continue
            if incoming and str(category) == "Transfers":
                continue
            for keyword in keywords:
                kw = str(keyword).upper()
                if kw in description:
                    return category, _score_keyword_match(description, merchant, kw), kw

        if incoming:
            return "Income & Transfers", 0.58, "Flow:Credit"
        if outgoing:
            return "Other", 0.22, "Flow:Debit"
        return "Other", 0.2, ""

    out = df.copy()
    assigned = out.apply(assign, axis=1, result_type="expand")
    assigned.columns = ["Category", "CategoryConfidence", "CategoryRule"]
    return pd.concat([out, assigned], axis=1)


def assign_categories(df: pd.DataFrame, keyword_map: dict) -> pd.DataFrame:
    """Assign categories to transactions using keyword matching."""
    return assign_categories_with_confidence(df, keyword_map)


def enforce_flow_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Correct obvious direction/category mismatches using debit/credit flow."""
    out = df.copy()
    debit = pd.to_numeric(out.get("Debit", out.get("DebitCHF", 0.0)), errors="coerce").fillna(0.0)
    credit = pd.to_numeric(out.get("Credit", out.get("CreditCHF", 0.0)), errors="coerce").fillna(0.0)
    category = out.get("Category", pd.Series(["Other"] * len(out), index=out.index)).fillna("Other").astype(str)
    transfer_mask = (
        out.get("IsTransfer", pd.Series([False] * len(out), index=out.index))
        .fillna(False)
        .astype(bool)
    )

    incoming = (credit > 0) & (debit == 0)
    outgoing = (debit > 0) & (credit == 0)

    wrong_income = outgoing & category.eq("Income & Transfers") & (~transfer_mask)
    if wrong_income.any():
        out.loc[wrong_income, "Category"] = "Other"
        if "CategoryConfidence" in out.columns:
            out.loc[wrong_income, "CategoryConfidence"] = out.loc[wrong_income, "CategoryConfidence"].apply(
                lambda v: min(_to_float(v), 0.35)
            )
        out.loc[wrong_income, "CategoryRule"] = "FlowCorrection:Outgoing"

    missing_income = incoming & category.eq("Other") & (~transfer_mask)
    if missing_income.any():
        out.loc[missing_income, "Category"] = "Income & Transfers"
        if "CategoryConfidence" in out.columns:
            out.loc[missing_income, "CategoryConfidence"] = out.loc[missing_income, "CategoryConfidence"].apply(
                lambda v: max(_to_float(v), 0.58)
            )
        out.loc[missing_income, "CategoryRule"] = "FlowCorrection:Incoming"

    return out
