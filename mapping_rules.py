"""Rule learning and suggestion helpers for transaction mapping."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

_TOKEN_PATTERN = re.compile(r"[A-Z0-9]{3,}")
_STOPWORDS = {
    "THE",
    "AND",
    "PAYMENT",
    "KARTE",
    "CARD",
    "CH",
    "CHF",
    "USD",
    "EUR",
    "GMBH",
    "AG",
    "LTD",
    "SA",
    "CO",
    "COMPANY",
    "PENDING",
    "TRANSAKTIONS",
    "TRANSAKTIONS",
    "TRANSACTION",
}


def normalize_rule_map(rule_map: dict[str, str]) -> dict[str, str]:
    """Normalize a mapping dictionary to upper-case keys."""
    normalized: dict[str, str] = {}
    for key, value in (rule_map or {}).items():
        key_text = str(key).upper().strip()
        value_text = str(value).strip()
        if key_text and value_text:
            normalized[key_text] = value_text
    return normalized


def transaction_text(row: pd.Series | dict[str, Any]) -> str:
    """Build a normalized text blob for mapping suggestions."""
    source = row if isinstance(row, dict) else row.to_dict()
    fields = [
        source.get("MerchantNormalized", ""),
        source.get("Merchant", ""),
        source.get("Beschreibung1", ""),
        source.get("Beschreibung2", ""),
        source.get("Beschreibung3", ""),
        source.get("Fussnoten", ""),
    ]
    return " ".join(str(item) for item in fields if str(item).strip()).upper()


def tokenize_mapping_text(text: str) -> list[str]:
    """Extract useful tokens from transaction text."""
    tokens = []
    for token in _TOKEN_PATTERN.findall(str(text).upper()):
        if token in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return sorted(set(tokens))


def suggest_category_from_rules(text: str, rule_map: dict[str, str]) -> tuple[str, str, float]:
    """Suggest category from token rules, returning category, top token, confidence."""
    rules = normalize_rule_map(rule_map)
    if not rules:
        return "", "", 0.0

    tokens = tokenize_mapping_text(text)
    if not tokens:
        return "", "", 0.0

    category_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    for token in tokens:
        category = rules.get(token)
        if category:
            category_counter[category] += 1
            token_counter[token] += 1

    if not category_counter:
        return "", "", 0.0

    best_category, best_hits = category_counter.most_common(1)[0]
    best_token = token_counter.most_common(1)[0][0] if token_counter else ""
    confidence = best_hits / max(len(tokens), 1)
    return best_category, best_token, round(float(confidence), 3)


def learn_pattern_rules(
    labeled_df: pd.DataFrame,
    min_examples: int = 3,
    min_precision: float = 0.8,
) -> pd.DataFrame:
    """Learn token -> category rules from labeled transaction data."""
    if labeled_df is None or labeled_df.empty:
        return pd.DataFrame(columns=["Token", "Category", "Occurrences", "Precision"])

    work = labeled_df.copy()
    if "Category" not in work.columns:
        return pd.DataFrame(columns=["Token", "Category", "Occurrences", "Precision"])

    work = work[work["Category"].fillna("").astype(str).str.strip().ne("")]
    work = work[work["Category"].fillna("").astype(str).str.strip().ne("Other")]
    if work.empty:
        return pd.DataFrame(columns=["Token", "Category", "Occurrences", "Precision"])

    token_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[str] = Counter()

    for _, row in work.iterrows():
        category = str(row.get("Category", "")).strip()
        if not category:
            continue
        text = transaction_text(row)
        tokens = tokenize_mapping_text(text)
        for token in tokens:
            token_category_counts[token][category] += 1
            token_totals[token] += 1

    rows: list[dict[str, Any]] = []
    for token, category_counts in token_category_counts.items():
        total = int(token_totals[token])
        if total < int(min_examples):
            continue
        best_category, best_count = category_counts.most_common(1)[0]
        precision = best_count / max(total, 1)
        if precision < float(min_precision):
            continue
        rows.append(
            {
                "Token": token,
                "Category": best_category,
                "Occurrences": total,
                "Precision": round(float(precision), 3),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Token", "Category", "Occurrences", "Precision"])
    return pd.DataFrame(rows).sort_values(["Occurrences", "Precision"], ascending=[False, False]).reset_index(drop=True)


def apply_pattern_rules(
    df: pd.DataFrame,
    rule_map: dict[str, str],
    low_confidence_threshold: float = 0.75,
) -> pd.DataFrame:
    """Apply token-based pattern rules to low-confidence rows."""
    rules = normalize_rule_map(rule_map)
    if not rules:
        return df

    out = df.copy()
    if "CategoryConfidence" not in out.columns:
        out["CategoryConfidence"] = 0.0
    if "CategoryRule" not in out.columns:
        out["CategoryRule"] = ""

    for idx, row in out.iterrows():
        current_category = str(row.get("Category", "Other"))
        current_conf = float(row.get("CategoryConfidence", 0.0) or 0.0)
        if current_category != "Other" and current_conf >= low_confidence_threshold:
            continue

        suggested, token, score = suggest_category_from_rules(transaction_text(row), rules)
        if not suggested:
            continue

        out.at[idx, "Category"] = suggested
        out.at[idx, "CategoryConfidence"] = max(current_conf, min(0.95, 0.8 + (score * 0.15)))
        if token:
            out.at[idx, "CategoryRule"] = f"PatternRule:{token}"

    return out
