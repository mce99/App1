import io

import pandas as pd

from categorization import assign_categories
from parsing import classify_time_of_day, load_transactions, merge_transactions


class DummyUpload(io.BytesIO):
    def __init__(self, name: str, content: str) -> None:
        super().__init__(content.encode("utf-8"))
        self.name = name


def test_assign_categories_uses_keyword_map() -> None:
    df = pd.DataFrame(
        [
            {"Beschreibung1": "Coop Zurich", "Beschreibung2": "", "Beschreibung3": "", "Fussnoten": ""},
            {"Beschreibung1": "Unknown Vendor", "Beschreibung2": "", "Beschreibung3": "", "Fussnoten": ""},
        ]
    )
    keyword_map = {"Food": ["COOP"]}

    out = assign_categories(df, keyword_map)

    assert list(out["Category"]) == ["Food", "Other"]


def test_load_transactions_classifies_time_of_day() -> None:
    csv_content = "\n".join(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "h7",
            "h8",
            "Abschlussdatum;Abschlusszeit;Währung;Belastung;Gutschrift;Beschreibung1;Beschreibung2;Beschreibung3;Fussnoten",
            "2026-02-01;06:30:00;CHF;-10;0;COOP;;;",
            "2026-02-01;22:10:00;CHF;-5;0;UBER;;;",
        ]
    )
    uploaded = DummyUpload("transactions.csv", csv_content)

    out = load_transactions(uploaded)

    assert list(out["TimeOfDay"]) == ["Morning", "Night"]
    assert list(out["Debit"]) == [10.0, 5.0]


def test_merge_transactions_sorts_across_multiple_files() -> None:
    csv_one = "\n".join(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "h7",
            "h8",
            "Abschlussdatum;Abschlusszeit;Währung;Belastung;Gutschrift;Beschreibung1;Beschreibung2;Beschreibung3;Fussnoten",
            "2026-02-03;09:00:00;CHF;-7;0;StoreA;;;",
        ]
    )
    csv_two = "\n".join(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "h7",
            "h8",
            "Abschlussdatum;Abschlusszeit;Währung;Belastung;Gutschrift;Beschreibung1;Beschreibung2;Beschreibung3;Fussnoten",
            "2026-02-01;10:00:00;CHF;-3;0;StoreB;;;",
            "2026-02-02;12:00:00;CHF;-4;0;StoreC;;;",
        ]
    )

    out = merge_transactions(
        [
            DummyUpload("part_1.csv", csv_one),
            DummyUpload("part_2.csv", csv_two),
        ],
        drop_duplicates=True,
    )

    assert list(out["Merchant"]) == ["StoreB", "StoreC", "StoreA"]


def test_merge_transactions_deduplicates_overlap() -> None:
    overlapping = "\n".join(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "h7",
            "h8",
            "Abschlussdatum;Abschlusszeit;Währung;Belastung;Gutschrift;Beschreibung1;Beschreibung2;Beschreibung3;Fussnoten",
            "2026-02-01;10:00:00;CHF;-3;0;StoreB;;;",
        ]
    )
    out = merge_transactions(
        [
            DummyUpload("a.csv", overlapping),
            DummyUpload("b.csv", overlapping),
        ],
        drop_duplicates=True,
    )
    assert len(out) == 1


def test_classify_time_of_day_invalid_returns_unknown() -> None:
    assert classify_time_of_day("bad-time") == "Unknown"
