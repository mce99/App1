import pandas as pd

import geo_insights
from geo_insights import spending_location_points


def test_spending_location_points_empty_when_no_location() -> None:
    df = pd.DataFrame([{"DebitCHF": 10.0}])
    out = spending_location_points(df)
    assert out.empty


def test_spending_location_points_uses_geocoder(monkeypatch) -> None:
    def fake_geocode(location: str):
        if "Regensdorf" in location:
            return 47.434, 8.468
        return None, None

    monkeypatch.setattr(geo_insights, "geocode_location_switzerland", fake_geocode)

    df = pd.DataFrame(
        [
            {"Location": "0810 Regensdorf", "DebitCHF": 20.0},
            {"Location": "0810 Regensdorf", "DebitCHF": 30.0},
        ]
    )
    out = spending_location_points(df)
    assert len(out) == 1
    assert float(out.loc[0, "SpendingCHF"]) == 50.0
