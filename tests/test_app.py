import io

import pandas as pd

from app import assign_categories, load_transactions


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
