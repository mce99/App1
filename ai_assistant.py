"""AI-assisted financial analysis helpers with offline fallback."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def _safe_table(df: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    if df is None or df.empty:
        return "(none)"
    subset = [col for col in columns if col in df.columns]
    if not subset:
        return "(none)"
    return df[subset].head(limit).to_string(index=False)


def build_offline_ai_brief(
    kpis: dict[str, float],
    benchmark_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    merchant_table: pd.DataFrame,
    user_goal: str = "",
) -> str:
    """Generate a deterministic analysis summary when API is unavailable."""
    top_net = float(kpis.get("net_cashflow", 0.0))
    total_spend = float(kpis.get("total_spending", 0.0))
    total_earn = float(kpis.get("total_earnings", 0.0))
    savings_rate = float(kpis.get("savings_rate", 0.0))

    over_rows = pd.DataFrame()
    if benchmark_table is not None and not benchmark_table.empty and "Status" in benchmark_table.columns:
        over_rows = benchmark_table[benchmark_table["Status"] == "Over"].head(3)

    lines = [
        "Offline AI Finance Brief",
        "",
        f"- Net cashflow: CHF {top_net:,.2f}",
        f"- Total spending: CHF {total_spend:,.2f}",
        f"- Total earnings: CHF {total_earn:,.2f}",
        f"- Savings rate: {savings_rate:.1f}%",
    ]
    if user_goal.strip():
        lines.append(f"- Goal focus: {user_goal.strip()}")

    lines.append("")
    lines.append("Priority actions:")
    if not over_rows.empty:
        for _, row in over_rows.iterrows():
            metric = str(row.get("Metric", "Category"))
            gap = float(row.get("GapPct", 0.0))
            lines.append(f"- Reduce {metric} (currently {gap:.1f}% above target).")
    else:
        lines.append("- Continue current spending controls and monitor monthly trends.")

    if recommendations is not None and not recommendations.empty:
        top_suggestion = str(recommendations.iloc[0].get("Suggestion", "")).strip()
        if top_suggestion:
            lines.append(f"- First recommendation: {top_suggestion}")

    if merchant_table is not None and not merchant_table.empty:
        heavy_merchant = str(merchant_table.iloc[0].get("Merchant", "")).strip()
        if heavy_merchant:
            lines.append(f"- Review concentration risk at merchant: {heavy_merchant}")

    lines.append("")
    lines.append("Note: connect OpenAI API key for deeper AI explanations.")
    return "\n".join(lines)


def build_ai_prompt(
    kpis: dict[str, float],
    benchmark_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    merchant_table: pd.DataFrame,
    anomalies: pd.DataFrame,
    recurring: pd.DataFrame,
    action_plan: pd.DataFrame,
    user_goal: str = "",
) -> str:
    """Build a concise prompt for AI analysis."""
    lines = [
        "Analyze this personal finance dataset and return concise, practical actions.",
        "Focus on: spending reduction, cashflow stability, and savings growth.",
        "Return sections: Summary, Risks, 30-day actions, 90-day actions.",
        "",
        f"User goal: {user_goal.strip() or '(not specified)'}",
        "",
        "KPIs:",
        str(kpis),
        "",
        "Benchmark table:",
        _safe_table(
            benchmark_table,
            ["Metric", "ActualPctIncome", "TargetPctIncome", "GapPct", "Status", "MonthlyActualCHF", "MonthlyTargetCHF"],
            limit=12,
        ),
        "",
        "Top recommendations:",
        _safe_table(recommendations, ["Priority", "Area", "Issue", "Suggestion"], limit=8),
        "",
        "Top merchants:",
        _safe_table(merchant_table, ["Merchant", "Transactions", "SpendingCHF", "AvgTicketCHF"], limit=12),
        "",
        "Anomalies:",
        _safe_table(anomalies, ["Date", "Merchant", "Category", "DebitCHF", "AnomalyScore"], limit=8),
        "",
        "Recurring:",
        _safe_table(recurring, ["Merchant", "CadenceDays", "AvgSpendingCHF", "AvgEarningsCHF"], limit=10),
        "",
        "Current action queue:",
        _safe_table(action_plan, ["Priority", "Area", "Task", "Reason"], limit=12),
    ]
    return "\n".join(lines)


def generate_ai_brief(
    kpis: dict[str, float],
    benchmark_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    merchant_table: pd.DataFrame,
    anomalies: pd.DataFrame,
    recurring: pd.DataFrame,
    action_plan: pd.DataFrame,
    user_goal: str = "",
    api_key: str = "",
    model: str = "gpt-4.1-mini",
) -> Tuple[str, str]:
    """Return (mode, analysis_text). Falls back to offline summary when needed."""
    offline = build_offline_ai_brief(
        kpis=kpis,
        benchmark_table=benchmark_table,
        recommendations=recommendations,
        merchant_table=merchant_table,
        user_goal=user_goal,
    )
    if not api_key.strip():
        return "offline", offline

    try:
        from openai import OpenAI
    except Exception:
        return "offline", offline

    prompt = build_ai_prompt(
        kpis=kpis,
        benchmark_table=benchmark_table,
        recommendations=recommendations,
        merchant_table=merchant_table,
        anomalies=anomalies,
        recurring=recurring,
        action_plan=action_plan,
        user_goal=user_goal,
    )

    try:
        client = OpenAI(api_key=api_key.strip())
        response = client.chat.completions.create(
            model=model,
            temperature=0.25,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict financial analyst. Give precise actions with numbers, "
                        "clear assumptions, and no generic advice."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        content = (content or "").strip()
        if not content:
            return "offline", offline
        return "online", content
    except Exception:
        return "offline", offline
