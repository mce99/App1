import pandas as pd

from analytics import (
    calculate_kpis,
    category_breakdown,
    daily_net_cashflow,
    hourly_spending_profile,
    quality_indicators,
    spending_velocity,
    weekday_average_cashflow,
)


def _sample_df() -> pd.DataFrame:
    data = [
        {"Date": "2026-02-01", "Time": "08:30:00", "DebitCHF": 20.0, "CreditCHF": 0.0},
        {"Date": "2026-02-01", "Time": "18:10:00", "DebitCHF": 30.0, "CreditCHF": 0.0},
        {"Date": "2026-02-02", "Time": "08:45:00", "DebitCHF": 0.0, "CreditCHF": 100.0},
        {"Date": "2026-02-02", "Time": "22:00:00", "DebitCHF": 10.0, "CreditCHF": 0.0},
    ]
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df["SortDateTime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"])
    return df


def test_daily_net_cashflow_has_cumulative_columns() -> None:
    out = daily_net_cashflow(_sample_df())

    assert "CumulativeSpending" in out.columns
    assert "CumulativeEarnings" in out.columns
    assert "CumulativeNet" in out.columns
    assert float(out["CumulativeSpending"].iloc[-1]) == 60.0
    assert float(out["CumulativeEarnings"].iloc[-1]) == 100.0
    assert float(out["CumulativeNet"].iloc[-1]) == 40.0


def test_hourly_spending_profile_groups_by_hour() -> None:
    out = hourly_spending_profile(_sample_df())

    assert len(out) == 24
    assert float(out.loc[8, "Spending"]) == 20.0
    assert float(out.loc[8, "Earnings"]) == 100.0
    assert float(out.loc[18, "Spending"]) == 30.0
    assert float(out.loc[22, "Spending"]) == 10.0


def test_calculate_kpis_contains_daily_averages() -> None:
    kpis = calculate_kpis(_sample_df())

    assert kpis["transactions"] == 4
    assert kpis["active_days"] == 2
    assert kpis["avg_spending_per_active_day"] == 30.0
    assert kpis["avg_earnings_per_active_day"] == 50.0
    assert kpis["avg_daily_net"] == 20.0


def test_weekday_average_cashflow_returns_all_weekdays() -> None:
    out = weekday_average_cashflow(_sample_df())

    assert len(out.index) == 7
    assert "Monday" in out.index
    assert "Sunday" in out.index


def test_category_breakdown_computes_shares() -> None:
    df = _sample_df()
    df["Category"] = ["Food", "Food", "Salary", "Transport"]
    out = category_breakdown(df)

    assert "SpendingSharePct" in out.columns
    assert "EarningsSharePct" in out.columns
    assert round(float(out["SpendingCHF"].sum()), 2) == 60.0


def test_spending_velocity_contains_rolling_columns() -> None:
    out = spending_velocity(_sample_df(), window_days=2)
    assert list(out.columns) == ["SpendingMA", "EarningsMA", "NetMA"]
    assert len(out) == 2


def test_quality_indicators_are_percentages() -> None:
    df = _sample_df()
    df["Category"] = ["Other", "Food", "Other", "Transport"]
    df["TimeOfDay"] = ["Unknown", "Morning", "Unknown", "Night"]
    df["Währung"] = ["CHF", "CHF", None, "CHF"]
    indicators = quality_indicators(df)

    assert indicators["rows"] == 4.0
    assert indicators["other_category_pct"] == 50.0
    assert indicators["unknown_timeofday_pct"] == 50.0
    assert indicators["missing_currency_pct"] == 25.0
