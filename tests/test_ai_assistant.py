import pandas as pd

from ai_assistant import build_offline_ai_brief, generate_ai_brief


def test_build_offline_ai_brief_contains_core_sections() -> None:
    kpis = {"net_cashflow": 100.0, "total_spending": 500.0, "total_earnings": 600.0, "savings_rate": 16.7}
    benchmark = pd.DataFrame(
        [
            {"Metric": "Groceries", "Status": "Over", "GapPct": 4.0},
            {"Metric": "Savings", "Status": "Low", "GapPct": 8.0},
        ]
    )
    recommendations = pd.DataFrame([{"Suggestion": "Cut dining budget by CHF 120/month."}])
    merchants = pd.DataFrame([{"Merchant": "COOP"}])

    text = build_offline_ai_brief(kpis, benchmark, recommendations, merchants, user_goal="Save more")
    assert "Offline AI Finance Brief" in text
    assert "Net cashflow" in text
    assert "Reduce Groceries" in text
    assert "Cut dining budget" in text
    assert "COOP" in text


def test_generate_ai_brief_without_key_falls_back_to_offline() -> None:
    mode, text = generate_ai_brief(
        kpis={"net_cashflow": 0.0, "total_spending": 0.0, "total_earnings": 0.0, "savings_rate": 0.0},
        benchmark_table=pd.DataFrame(),
        recommendations=pd.DataFrame(),
        merchant_table=pd.DataFrame(),
        anomalies=pd.DataFrame(),
        recurring=pd.DataFrame(),
        action_plan=pd.DataFrame(),
        user_goal="",
        api_key="",
        model="gpt-4.1-mini",
    )
    assert mode == "offline"
    assert "Offline AI Finance Brief" in text
