"""Persistence helpers for mapping memory (overrides and learned rules)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_MAPPING_MEMORY_PATH = "data/mapping_memory.json"


def _normalize_str_dict(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            out[key_text] = value_text
    return out


def load_mapping_memory(path: str) -> dict[str, dict[str, str]]:
    """Load mapping memory from disk."""
    target = Path(path).expanduser()
    if not target.exists():
        return {
            "category_overrides": {},
            "merchant_category_rules": {},
            "pattern_category_rules": {},
        }
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {
        "category_overrides": _normalize_str_dict(payload.get("category_overrides", {})),
        "merchant_category_rules": _normalize_str_dict(payload.get("merchant_category_rules", {})),
        "pattern_category_rules": _normalize_str_dict(payload.get("pattern_category_rules", {})),
    }


def save_mapping_memory(
    path: str,
    category_overrides: dict[str, str],
    merchant_category_rules: dict[str, str],
    pattern_category_rules: dict[str, str],
) -> Path:
    """Save mapping memory to disk and return saved path."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "category_overrides": _normalize_str_dict(category_overrides),
        "merchant_category_rules": _normalize_str_dict(merchant_category_rules),
        "pattern_category_rules": _normalize_str_dict(pattern_category_rules),
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return target
