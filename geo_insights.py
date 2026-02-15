"""Geospatial helpers for spending map views."""

from __future__ import annotations

import functools
import re

import pandas as pd
import requests

GEOADMIN_SEARCH_URL = "https://api3.geo.admin.ch/rest/services/api/SearchServer"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def _clean_location(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


@functools.lru_cache(maxsize=1024)
def geocode_location_switzerland(location: str) -> tuple[float | None, float | None]:
    query = _clean_location(location)
    if not query:
        return None, None

    # Swiss official geocoder first.
    try:
        response = requests.get(
            GEOADMIN_SEARCH_URL,
            params={"searchText": query, "type": "locations", "limit": 1, "origins": "address,zipcode"},
            timeout=6.0,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if results:
            attrs = results[0].get("attrs", {})
            lat = attrs.get("lat")
            lon = attrs.get("lon")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
    except Exception:
        pass

    # Fallback geocoder.
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": f"{query}, Switzerland", "format": "jsonv2", "limit": 1},
            timeout=6.0,
            headers={"User-Agent": "PulseLedger/1.0"},
        )
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass

    return None, None


def spending_location_points(df: pd.DataFrame, min_spending_chf: float = 0.0) -> pd.DataFrame:
    """Aggregate spending by location with coordinates for map plotting."""
    if "Location" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["Location"] = work["Location"].astype(str).map(_clean_location)
    work = work[work["Location"].str.len() > 2]
    work = work[work["DebitCHF"] > 0]
    if min_spending_chf > 0:
        work = work[work["DebitCHF"] >= min_spending_chf]
    if work.empty:
        return pd.DataFrame()

    grouped = (
        work.groupby("Location")
        .agg(SpendingCHF=("DebitCHF", "sum"), Transactions=("Location", "size"))
        .reset_index()
        .sort_values("SpendingCHF", ascending=False)
    )

    coords = grouped["Location"].apply(lambda loc: pd.Series(geocode_location_switzerland(loc)))
    coords.columns = ["lat", "lon"]
    out = pd.concat([grouped, coords], axis=1)
    out = out[out["lat"].notna() & out["lon"].notna()]
    return out.reset_index(drop=True)
