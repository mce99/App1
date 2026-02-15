"""Category mapping and assignment logic for transactions."""

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


def assign_categories(df: pd.DataFrame, keyword_map: dict) -> pd.DataFrame:
    """Assign categories to transactions using keyword matching."""

    def assign(row: pd.Series) -> str:
        desc_fields = [
            str(row.get("Beschreibung1", "")),
            str(row.get("Beschreibung2", "")),
            str(row.get("Beschreibung3", "")),
            str(row.get("Fussnoten", "")),
        ]
        description = " ".join(desc_fields).upper()
        for category, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in description:
                    return category
        return "Other"

    out = df.copy()
    out["Category"] = out.apply(assign, axis=1)
    return out
