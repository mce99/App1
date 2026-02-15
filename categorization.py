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


def _score_keyword_match(description: str, merchant: str, keyword: str) -> float:
    score = 0.75
    if keyword in merchant:
        score += 0.15
    if re.search(rf"\b{re.escape(keyword)}\b", description):
        score += 0.06
    return min(score, 0.98)


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
        for category, keywords in keyword_map.items():
            for keyword in keywords:
                kw = str(keyword).upper()
                if kw in description:
                    return category, _score_keyword_match(description, merchant, kw), kw
        return "Other", 0.2, ""

    out = df.copy()
    assigned = out.apply(assign, axis=1, result_type="expand")
    assigned.columns = ["Category", "CategoryConfidence", "CategoryRule"]
    return pd.concat([out, assigned], axis=1)


def assign_categories(df: pd.DataFrame, keyword_map: dict) -> pd.DataFrame:
    """Assign categories to transactions using keyword matching."""
    return assign_categories_with_confidence(df, keyword_map)
