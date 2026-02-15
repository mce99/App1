import io
import zipfile

import pandas as pd

from categorization import DEFAULT_KEYWORD_MAP, assign_categories, enforce_flow_consistency
from parsing import classify_time_of_day, load_transactions, merge_transactions


class DummyUpload(io.BytesIO):
    def __init__(self, name: str, content: str) -> None:
        super().__init__(content.encode("utf-8"))
        self.name = name


class DummyUploadBytes(io.BytesIO):
    def __init__(self, name: str, content: bytes) -> None:
        super().__init__(content)
        self.name = name


def test_assign_categories_uses_keyword_map() -> None:
    df = pd.DataFrame(
        [
            {
                "Beschreibung1": "Coop Zurich",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
            },
            {
                "Beschreibung1": "Unknown Vendor",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
            },
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


def test_load_transactions_detects_header_and_context_with_metadata_rows() -> None:
    csv_content = "\n".join(
        [
            "Kontonummer:;0000000",
            "Von:;2025-06-07 00:00:00",
            "Bis:;2026-02-13 00:00:00",
            "Bewertet in:;CHF",
            "Anzahl Transaktionen in diesem Zeitraum:;2",
            "Abschlussdatum;Abschlusszeit;Währung;Belastung;Gutschrift;Beschreibung1;Beschreibung2;Beschreibung3;Fussnoten;;;",
            "2026-02-13;17:12:00;CHF;-35.45;0;UBER EATS;AMSTERDAM;20828393;Zahlung Debitkarte;Transaktions-Nr. 1;Kosten: 1.00",
        ]
    )
    out = load_transactions(DummyUpload("styled.csv", csv_content))

    assert len(out) == 1
    assert out.loc[0, "StatementCurrency"] == "CHF"
    assert str(out.loc[0, "StatementFrom"]).startswith("2025-06-07")
    assert out.loc[0, "StatementTransactions"] == 2
    assert "Transaktions-Nr. 1" in out.loc[0, "Fussnoten"]
    assert "Kosten: 1.00" in out.loc[0, "Fussnoten"]


def test_load_transactions_uses_booking_date_when_time_missing() -> None:
    csv_content = "\n".join(
        [
            "Kontonummer:;0000000",
            "Abschlussdatum;Abschlusszeit;Buchungsdatum;Valutadatum;Währung;Belastung;Gutschrift;Beschreibung1;Fussnoten",
            "2026-02-12;;2026-02-13;2026-02-12;CHF;-10;0;NoTimeTx;",
        ]
    )
    out = load_transactions(DummyUpload("missing_time.csv", csv_content))
    assert len(out) == 1
    assert str(out.loc[0, "BookingDate"]).startswith("2026-02-13")
    assert not bool(out.loc[0, "HasExplicitTime"])
    # Noon fallback is used when no explicit trade time is present.
    assert str(out.loc[0, "SortDateTime"]).startswith("2026-02-13 12:00:00")


def test_load_transactions_maps_english_ubs_headers() -> None:
    csv_content = "\n".join(
        [
            "Account number:;1234 12345678.12",
            "IBAN:;CH20 0011 2233 4455 6677 B",
            "From:;2025-01-01",
            "Until:;2025-12-31",
            "Valued in:;CHF",
            "Numbers of transactions in this period:;1",
            "",
            "Trade date;Trade time;Booking date;Value date;Currency;Debit;Credit;Transaction no.;Description1;Description2;Description3;Footnotes;",
            "2025-01-01;00:11:22;2025-01-01;2025-01-01;CHF;-137.00;;4825794DP1572581029;John Doe;Standing order;Reference details;Costs: Standing order domestic;",
        ]
    )

    out = load_transactions(DummyUpload("english_style.csv", csv_content))

    assert len(out) == 1
    assert out.loc[0, "StatementAccountNumber"] == "1234 12345678.12"
    assert out.loc[0, "StatementIBAN"] == "CH20 0011 2233 4455 6677 B"
    assert out.loc[0, "StatementCurrency"] == "CHF"
    assert out.loc[0, "Belastung"] == -137.0
    assert out.loc[0, "Transaktions-Nr."] == "4825794DP1572581029"


def test_merge_transactions_supports_zip_bundle() -> None:
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
            "2026-02-03;09:00:00;CHF;-7;0;StoreA;;;",
            "2026-02-04;09:00:00;CHF;-8;0;StoreB;;;",
        ]
    ).encode("utf-8")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("batch_1.csv", csv_content)
    zip_upload = DummyUploadBytes("history.zip", zip_buffer.getvalue())

    out = merge_transactions([zip_upload], drop_duplicates=True)
    assert len(out) == 2
    assert list(out["Merchant"]) == ["StoreA", "StoreB"]


def test_load_transactions_splits_single_column_semicolon_payload() -> None:
    csv_content = "\n".join(
        [
            '"Kontonummer:;0000000"',
            '"Abschlussdatum;Abschlusszeit;Währung;Belastung;Gutschrift;Beschreibung1;Beschreibung2;Beschreibung3;Fussnoten"',
            '"2026-02-01;06:30:00;CHF;-10;0;COOP;;;;"',
        ]
    )
    out = load_transactions(DummyUpload("single_col.csv", csv_content))
    assert len(out) == 1
    assert out.loc[0, "Merchant"] == "COOP"
    assert float(out.loc[0, "Debit"]) == 10.0


def test_load_transactions_handles_signed_credit_and_single_amount_fallback() -> None:
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
            "Abschlussdatum;Abschlusszeit;Währung;Belastung;Gutschrift;Einzelbetrag;Beschreibung1;Fussnoten",
            "2026-02-01;09:00:00;CHF;0;-15;;CreditHasNegative;",
            "2026-02-01;10:00:00;CHF;;;25;SingleAmountIncome;",
            "2026-02-01;11:00:00;CHF;;;-40;SingleAmountSpend;",
        ]
    )
    out = load_transactions(DummyUpload("signed_amounts.csv", csv_content))

    assert len(out) == 3
    assert float(out.loc[0, "Debit"]) == 15.0
    assert float(out.loc[0, "Credit"]) == 0.0
    assert float(out.loc[1, "Credit"]) == 25.0
    assert float(out.loc[1, "Debit"]) == 0.0
    assert float(out.loc[2, "Debit"]) == 40.0
    assert float(out.loc[2, "Credit"]) == 0.0


def test_assign_categories_is_flow_aware_for_income_vs_outgoing() -> None:
    df = pd.DataFrame(
        [
            {
                "Beschreibung1": "UBS Switzerland",
                "Beschreibung2": "Salary",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 45.0,
                "Credit": 0.0,
            },
            {
                "Beschreibung1": "Unknown employer",
                "Beschreibung2": "Payroll",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 0.0,
                "Credit": 5000.0,
            },
        ]
    )
    keyword_map = {"Income & Transfers": ["UBS SWITZERLAND", "PAYROLL"], "Food": ["COOP"]}

    out = assign_categories(df, keyword_map)

    assert out.loc[0, "Category"] != "Income & Transfers"
    assert out.loc[1, "Category"] == "Income & Transfers"


def test_enforce_flow_consistency_corrects_obvious_mismatches() -> None:
    df = pd.DataFrame(
        [
            {
                "Category": "Income & Transfers",
                "CategoryConfidence": 0.9,
                "Debit": 35.0,
                "Credit": 0.0,
                "IsTransfer": False,
            },
            {
                "Category": "Other",
                "CategoryConfidence": 0.2,
                "Debit": 0.0,
                "Credit": 1200.0,
                "IsTransfer": False,
            },
            {
                "Category": "Income & Transfers",
                "CategoryConfidence": 0.9,
                "Debit": 80.0,
                "Credit": 0.0,
                "IsTransfer": True,
            },
        ]
    )

    out = enforce_flow_consistency(df)
    assert out.loc[0, "Category"] == "Other"
    assert out.loc[0, "CategoryRule"] == "FlowCorrection:Outgoing"
    assert out.loc[1, "Category"] == "Income & Transfers"
    assert out.loc[1, "CategoryRule"] == "FlowCorrection:Incoming"
    # Transfer rows are intentionally exempt from flow correction.
    assert out.loc[2, "Category"] == "Income & Transfers"


def test_assign_categories_uses_web_researched_merchant_hints() -> None:
    df = pd.DataFrame(
        [
            {
                "Beschreibung1": "AGROLA Tankstelle Regensdorf",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 60.0,
                "Credit": 0.0,
            },
            {
                "Beschreibung1": "Digitec Galaxus",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 200.0,
                "Credit": 0.0,
            },
            {
                "Beschreibung1": "Confiserie Honold AG",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 15.0,
                "Credit": 0.0,
            },
            {
                "Beschreibung1": "Metallum Metal Trading AG",
                "Beschreibung2": "Payroll",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 0.0,
                "Credit": 5000.0,
            },
            {
                "Beschreibung1": "Barbezug mit PIN",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 100.0,
                "Credit": 0.0,
            },
        ]
    )

    out = assign_categories(df, DEFAULT_KEYWORD_MAP)
    assert out.loc[0, "Category"] == "Transport"
    assert out.loc[1, "Category"] == "Shopping & Retail"
    assert out.loc[2, "Category"] == "Food & Drink"
    assert out.loc[3, "Category"] == "Income & Transfers"
    assert out.loc[4, "Category"] == "Transfers"


def test_assign_categories_handles_travel_and_avoids_short_keyword_false_positives() -> None:
    df = pd.DataFrame(
        [
            {
                "Beschreibung1": "RC Hotel Arts Barcelona",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 800.0,
                "Credit": 0.0,
            },
            {
                "Beschreibung1": "The Peninsula Tokyo",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 500.0,
                "Credit": 0.0,
            },
            {
                "Beschreibung1": "Payment card settlement",
                "Beschreibung2": "",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 25.0,
                "Credit": 0.0,
            },
            {
                "Beschreibung1": "Bargeldbezug am Bancomat im Ausland",
                "Beschreibung2": "Chuo-ku Tokyo",
                "Beschreibung3": "",
                "Fussnoten": "",
                "Debit": 100.0,
                "Credit": 0.0,
            },
        ]
    )

    out = assign_categories(df, DEFAULT_KEYWORD_MAP)
    assert out.loc[0, "Category"] == "Travel & Lodging"
    assert out.loc[1, "Category"] == "Travel & Lodging"
    # "card" should not trigger Transport via broad CAR substring anymore.
    assert out.loc[2, "Category"] != "Transport"
    assert out.loc[3, "Category"] == "Transfers"
