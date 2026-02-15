import pandas as pd

from mapping_rules import (
    apply_pattern_rules,
    learn_pattern_rules,
    suggest_category_from_rules,
    tokenize_mapping_text,
    transaction_text,
)


def test_tokenize_mapping_text_filters_noise() -> None:
    tokens = tokenize_mapping_text("Uber Eats Pending CH 2082 card payment")
    assert "UBER" in tokens
    assert "EATS" in tokens
    assert "CH" not in tokens
    assert "PAYMENT" not in tokens


def test_learn_pattern_rules_learns_majority_token_category() -> None:
    df = pd.DataFrame(
        [
            {"MerchantNormalized": "COOP PRONTO", "Category": "Food & Drink"},
            {"MerchantNormalized": "COOP CITY", "Category": "Food & Drink"},
            {"MerchantNormalized": "COOP MARKET", "Category": "Food & Drink"},
            {"MerchantNormalized": "UBER TRIP", "Category": "Transport"},
        ]
    )

    out = learn_pattern_rules(df, min_examples=2, min_precision=0.8)
    assert not out.empty
    coop = out[out["Token"] == "COOP"]
    assert not coop.empty
    assert coop.iloc[0]["Category"] == "Food & Drink"


def test_suggest_category_from_rules_returns_match() -> None:
    category, token, confidence = suggest_category_from_rules(
        "UBER EATS AMSTERDAM",
        {"UBER": "Transport", "EATS": "Food & Drink"},
    )
    assert category in {"Transport", "Food & Drink"}
    assert token in {"UBER", "EATS"}
    assert confidence > 0


def test_apply_pattern_rules_updates_low_conf_rows() -> None:
    df = pd.DataFrame(
        [
            {"MerchantNormalized": "COOP PRONTO", "Category": "Other", "CategoryConfidence": 0.2},
            {"MerchantNormalized": "SWISSCOM", "Category": "Utilities & Bills", "CategoryConfidence": 0.9},
        ]
    )
    out = apply_pattern_rules(df, {"COOP": "Food & Drink"}, low_confidence_threshold=0.75)
    assert out.loc[0, "Category"] == "Food & Drink"
    assert str(out.loc[0, "CategoryRule"]).startswith("PatternRule:")
    assert out.loc[1, "Category"] == "Utilities & Bills"


def test_transaction_text_combines_fields() -> None:
    row = pd.Series({"Merchant": "Coop", "Beschreibung2": "Zurich", "Fussnoten": "Debitkarte"})
    text = transaction_text(row)
    assert "COOP" in text
    assert "ZURICH" in text
