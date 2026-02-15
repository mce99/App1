from pathlib import Path

from mapping_memory import load_mapping_memory, save_mapping_memory


def test_mapping_memory_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "mapping_memory.json"
    save_mapping_memory(
        path=str(target),
        category_overrides={"tx-1": "Food & Drink"},
        merchant_category_rules={"COOP": "Food & Drink"},
        pattern_category_rules={"UBER": "Transport"},
    )

    loaded = load_mapping_memory(str(target))
    assert loaded["category_overrides"]["tx-1"] == "Food & Drink"
    assert loaded["merchant_category_rules"]["COOP"] == "Food & Drink"
    assert loaded["pattern_category_rules"]["UBER"] == "Transport"


def test_mapping_memory_missing_file_returns_empty(tmp_path: Path) -> None:
    loaded = load_mapping_memory(str(tmp_path / "missing.json"))
    assert loaded["category_overrides"] == {}
    assert loaded["merchant_category_rules"] == {}
    assert loaded["pattern_category_rules"] == {}
