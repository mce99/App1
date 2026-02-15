"""Category mapping and assignment logic for transactions."""

import re

import pandas as pd


DEFAULT_KEYWORD_MAP = {
    "Groceries": [
        "COOP",
        "MIGROS",
        "SPAR",
        "DENNER",
        "AVEC",
        "LAWSON",
        "FAMILYMART",
        "SEVEN-ELEVEN",
        "MIGROLINO",
        "BAHNHOFKIOSK",
        "K KIOSK",
        "KIOSK",
        "SUPERMARKT",
        "GROCERY",
        "PRONTO",
    ],
    "Restaurants & Cafes": [
        "SUSHI",
        "RESTAURANT",
        "PIZZA",
        "STARBUCKS",
        "CAFE",
        "MCDON",
        "CONFISERIE",
        "HONOLD",
        "METZGEREI",
        "DELIVEROO",
        "UBER   * EATS",
        "UBER EATS",
        "UBERJP EATS",
        "BREZELKONIG",
        "BREZELKÖNIG",
        "FIVE GUYS",
        "BURGER KING",
    ],
    "Gas Stations": [
        "AGROLA",
        "SOCAR",
        "SHELL",
        "ENI",
        "BP",
        "MIGROL",
        "TANKSTELLE",
        "CAR WASH",
        "AUTOPSTUETZLI",
        "STUTZLIWOSCH",
    ],
    "Shopping (General)": [
        "DIGITEC",
        "GALAXUS",
        "INTERDISCOUNT",
        "JUMBO",
        "JELMOLI",
        "DAIMARU",
        "MUJIRUSHIRYOHIN",
        "YURAKUCHO MARUI",
        "GINZA SIX",
        "TOKYO ICHIBANGAI",
        "TOKYO GIFT PARETTO",
        "KIKUICHIMONNJI",
        "KODAIJI TESSAIDO",
        "RIKAWAFUKUGIONTEN",
        "DOGWOOD PLAZA",
        "LAFORET HARAJUKU",
        "HAUSTIER PARADIES",
        "OCHSNER AG",
    ],
    "Clothing Brands": [
        "ZARA",
        "H&M",
        "LEVIS",
        "UNIQLO",
        "HOUSE OF CB",
        "SKIMS",
        "KLARNA*COS",
        "COS SWITZERLA",
        "MISTREASS",
        "MAISONMAIA",
    ],
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
        "UBER EATS",
        "UBERJP EATS",
        "STARBUCKS",
        "CAFE",
        "MCDON",
        "KINTARO",
        "MOREIRA",
        "MOREIRA GOURMET",
        "CONFISERIE HONOLD",
        "HONOLD",
        "METZGEREI OBERWACHT",
        "AVEC",
        "FLEISCHLI",
        "LAWSON",
        "FAMILYMART",
        "SEVEN-ELEVEN",
        "DELIVEROO",
        "DENNER",
        "MARCHE",
        "MARCHÉ",
        "BAHNHOFKIOSK",
        "K KIOSK",
        "KIOSK",
        "BREZELKONIG",
        "BREZELKÖNIG",
        "ROYAL-PANDA",
        "CUCINA",
        "CHEGRILL",
        "ELEPHANTS HEAD",
        "LIPP",
        "JULESVERNE",
        "UBS REST.",
        "BACKEREI",
        "BÄCKEREI",
        "BUTIA A L'EN",
        "RONI'S BELSIZE",
        "THE HAVERSTOCK TAVERN",
        "HAVERSTOCK TAVERN",
        "KAIPIRAS",
        "THE MIXER",
        "PLIMSOLL",
        "WOLFPACK",
        "SP CLUBHOUSE",
        "TURBINENBRAEU",
        "BRISKET ZUERICH",
        "YARDBIRD ZUERICH",
        "TAOS BAR",
        "NICOS KITCHEN",
        "DER ZUCKERBAECKER",
        "DER ZUCKERBÄCKER",
        "YOOJI",
        "YOOJI'S",
        "OH MY GREEK",
        "MIT & OHNE KEBAB",
        "BURGER KING",
        "JULIETTE-PAIN DAMOUR",
        "FIVE GUYS",
        "RAKUTENPAY PASCALLEGAC",
        "AKASAKAYAKINIKUSHIYA",
        "GINZANOSTEAK",
        "KENZAN ANAINTAKONCHINENT",
        "KYOTOKATSUGYU",
        "YAZAWACHICKEN",
        "NIKURYORI",
        "FORTUNE GARDEN KYOTO",
        "DEIRIYAMAZAKI",
        "BELSIZE ORGANIC",
        "HOMURA",
        "MUNCHMASTER",
        "QUICK MARKET",
        "TOUJOURS PLUSS",
        "CAFFE SPETTACOLO",
        "CH GASTRO AG",
        "LS BLACKTAP",
        "SANDO ZURICH",
        "ZAPP - QUICK COMMERCE",
        "GUETS GUGGELI SAHIN",
        "HOLLY BUSH HAMPSTEAD",
        "ENGLAND'S LANE",
        "OSTERIA SCHUTZENSTULE",
        "STROZZIS",
        "SANDORATSUGU",
        "BIG TASTY",
        "HM-HELALMETZGER",
        "DORFBECK NYFELER",
        "TULLYS COFFEE",
        "NZZ BAR ZH",
        "GNUSSPUR",
        "LA ROTONDA PORT OLIMPIC",
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
        "SBB CFF FFS",
        "AGROLA",
        "SOCAR",
        "ZURICHSEE-FAHRE",
        "FAEHRE",
        "AUTOQUAI",
        "PARKHAUS",
        "NARITA AIRPORT",
        "AIRPORT",
        "VBZ",
        "LIME*RIDE",
        "BOLT.EU",
        "GREENCAB",
        "SHELL",
        "LUFTHANSA",
        "SWISS AIR",
        "MUOTTAS MURAGL BAHN",
        "SIGNALBAHN",
        "PARKPLATZ SIGNALBAHN",
        "STATION SAVOGNIN",
        "STATION ST. KATHARINA",
        "TELEVEBIER",
        "BP ST. MORITZ",
        "BAHNHOF UNTERFUHRUNG",
        "BAHNHOF UNTERFÜHRUNG",
        "HAUSERMANN STADION GARAG",
        "BALANCES - 6004 LUZERN",
        "ST.  ENGADIN ST MORITZ",
        "AUTOPSTUETZLI",
        "STUTZLIWOSCH",
        "CAR WASH",
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
        "KANTON ZURICH",
        "STEUERBEZUG",
        "SVA ZURICH",
        "DIENSTLEISTUNGSPREISABSCHLUSS",
        "UNIVERSITATSSPITAL",
        "UNIVERSITÄTSSPITAL",
        "AMAVITA",
        "TOPPHARM",
        "FREMDKOSTEN",
        "ZINSABSCHLUSS",
        "WALK-IN-LABOR",
        "UKVI ETAMOB",
        "WWW.BWLEGAL.CO.UK",
        "TERLINDEN MANAGEMENT",
        "SCC0444 SUNRISE",
        "GLOBALCHIRYOIN AKASAKA",
        "MOVINGINTERNET.CO.UK",
        "TEXTILPFLEGE ZUERISEE",
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
        "DIGITEC",
        "GALAXUS",
        "UNIQLO",
        "TAOBAO",
        "MAGIC X",
        "STEINLIN SCHMUCK",
        "JELMOLI",
        "JUMBO",
        "INTERDISCOUNT",
        "LOFT",
        "DAIMARU",
        "DIGITEC GALAXUS",
        "HAUSTIER PARADIES",
        "HOUSE OF CB",
        "SKIMS",
        "SP MISTREASS",
        "MISTREASS",
        "MAISONMAIA",
        "LANDOLT-ARBENZ",
        "DOUGLAS",
        "KLARNA*COS",
        "COCOKARAFINE",
        "YURAKUCHO MARUI",
        "GINZA SIX",
        "TOKYO ICHIBANGAI",
        "TOKYO GIFT PARETTO",
        "MUJIRUSHIRYOHIN",
        "KIKUICHIMONNJI",
        "KODAIJI TESSAIDO",
        "RIKAWAFUKUGIONTEN",
        "PARIJAN",
        "MYPROTEIN",
        "LENS4YOU",
        "OCHSNER AG",
        "KREISLADEN",
        "TOPDEN",
        "BLATTNER",
        "LOLLIPOP ZURICH",
        "DOGWOOD PLAZA",
        "LAFORET HARAJUKU",
        "SP PPFLINGERIE",
        "SP COZYLLIO",
        "SP CANPELA",
        "TENGOSOK KIKOTOJE",
        "FRONERI SWITZERLAND",
        "BURGI.CH AG",
        "PHILORO SCHWEIZ",
        "MS AKIHABARA",
        "FRESHTECH",
        "SP RYM PRODUCTS",
        "LOLIPOP ZURICH",
        "SUMUP  *EINZELFIRMA",
    ],
    "Investments & Digital Assets": [
        "CRYPTO.COM",
        "BIFINITY",
        "MOONPAY",
        "2C2P",
        "PAYPAL *LORDMILES LORD",
        "PAYPAL *CHINAFL6LDN",
    ],
    "Income & Transfers": [
        "METALLUM",
        "XXXX",
        "BANK",
        "REVOLUT",
        "SALDO ZINSABSCHLUSS",
        "ENKELMANN",
        "TRANSFER",
        "UBS SWITZERLAND",
        "WISE EUROPE",
    ],
    "Entertainment & Leisure": [
        "FANVUE",
        "BILL",
        "APPLE.COM",
        "APPLE.COM/BILL",
        "NETFLIX",
        "GYM",
        "SPA",
        "ART",
        "AVEDA",
        "STRANDBADSAUNA",
        "SKISERVICE",
        "SAGRADA FAMILIA",
        "WHOOP",
        "JINDAI BOTANICAL GARDENS",
        "TICKETMELON",
        "SUMUP  *HR HAIR REMOVAL",
        "HAIRSTYLIST PIERRE",
        "MUSEUM",
    ],
    "Travel & Lodging": [
        "HOTEL ARTS",
        "PENINSULA TOKYO",
        "ANA INTERCONTINENTAL",
        "INTERCONTINENTAL TOKYO",
        "HOTEL LAUDINELLA",
        "BERGHOTEL",
        "WASHINGTON HOTEL",
        "CHALET",
        "HOTEL",
        "MARRIOTT",
        "AIRBNB",
        "BOOKING.COM",
        "ALEX LAKE ZURICH",
        "LE FARINET VERBIER",
        "CHESA VEGLIA",
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
    "WISE",
    "BARBEZUG",
    "CASH WITHDRAWAL",
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

STRICT_WORD_KEYWORDS = {
    "CAR",
    "ART",
}


def _score_keyword_match(description: str, merchant: str, keyword: str) -> float:
    score = 0.75
    if keyword in merchant:
        score += 0.15
    if re.search(rf"\b{re.escape(keyword)}\b", description):
        score += 0.06
    return min(score, 0.98)


def _keyword_matches(description: str, keyword: str) -> bool:
    kw = str(keyword or "").upper().strip()
    if not kw:
        return False
    if kw in STRICT_WORD_KEYWORDS:
        return bool(re.search(rf"(?<![A-Z0-9]){re.escape(kw)}(?![A-Z0-9])", description))

    alnum = "".join(ch for ch in kw if ch.isalnum())
    if len(alnum) <= 4 and " " not in kw and "&" not in kw and "*" not in kw and "/" not in kw:
        return bool(re.search(rf"(?<![A-Z0-9]){re.escape(kw)}(?![A-Z0-9])", description))
    return kw in description


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

        if outgoing and (
            "BARGELDBEZUG" in description or "BANCOMAT" in description or "ATM" in description
        ):
            return "Transfers", 0.96, "Transfer:CashWithdrawal"
        if "ZINSABSCHLUSS" in description:
            return "Utilities & Bills", 0.9, "Bank:InterestSettlement"
        if "FREMDKOSTEN" in description:
            return "Utilities & Bills", 0.9, "Bank:ForeignFees"

        for keyword in TRANSFER_KEYWORDS:
            kw = str(keyword).upper()
            if kw and _keyword_matches(description, kw):
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
                if _keyword_matches(description, kw):
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
    credit = pd.to_numeric(out.get("Credit", out.get("CreditCHF", 0.0)), errors="coerce").fillna(
        0.0
    )
    category = (
        out.get("Category", pd.Series(["Other"] * len(out), index=out.index))
        .fillna("Other")
        .astype(str)
    )
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
            out.loc[wrong_income, "CategoryConfidence"] = out.loc[
                wrong_income, "CategoryConfidence"
            ].apply(lambda v: min(_to_float(v), 0.35))
        out.loc[wrong_income, "CategoryRule"] = "FlowCorrection:Outgoing"

    missing_income = incoming & category.eq("Other") & (~transfer_mask)
    if missing_income.any():
        out.loc[missing_income, "Category"] = "Income & Transfers"
        if "CategoryConfidence" in out.columns:
            out.loc[missing_income, "CategoryConfidence"] = out.loc[
                missing_income, "CategoryConfidence"
            ].apply(lambda v: max(_to_float(v), 0.58))
        out.loc[missing_income, "CategoryRule"] = "FlowCorrection:Incoming"

    return out
