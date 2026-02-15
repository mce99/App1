import pandas as pd

from analytics import (
    apply_category_overrides,
    balance_timeline,
    benchmark_assessment,
    build_report_pack,
    cashflow_stability_metrics,
    calculate_kpis,
    chart_builder_dataset,
    category_momentum,
    category_volatility,
    category_breakdown,
    daily_net_cashflow,
    detect_anomalies,
    enrich_transaction_intelligence,
    forecast_cashflow,
    generate_agent_action_plan,
    hourly_spending_profile,
    income_concentration_table,
    ingestion_quality_by_source,
    merchant_concentration_table,
    monthly_trend_diagnostics,
    monthly_salary_estimate,
    merchant_insights,
    period_over_period_metrics,
    possible_duplicate_candidates,
    quality_indicators,
    recurring_transaction_candidates,
    savings_opportunity_scanner,
    savings_scenario,
    spending_heatmap_matrix,
    spending_run_rate_projection,
    spending_recommendations,
    spending_velocity,
    transaction_size_distribution,
    weekday_weekend_split,
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


def test_enrich_transaction_intelligence_detects_transfer() -> None:
    df = _sample_df()
    df["Beschreibung1"] = ["Transfer to savings", "Store", "Payroll", "Move to IBAN CH96 0027 4274 1271 2140 B"]
    df["Beschreibung2"] = ["", "", "", ""]
    df["Beschreibung3"] = ["", "", "", ""]
    df["Fussnoten"] = ["", "", "", "Internal account transfer"]
    df["Merchant"] = df["Beschreibung1"]
    df["SourceFile"] = "file.csv"
    df["SourceAccount"] = "CH11"
    df["Category"] = "Other"

    out = enrich_transaction_intelligence(df)
    assert bool(out.loc[3, "IsTransfer"]) is True
    assert out.loc[3, "TransferDirection"] in {"Out", "In", "Unknown"}
    assert out.loc[3, "CounterpartyAccount"] != ""


def test_apply_category_overrides_uses_transaction_id() -> None:
    df = _sample_df()
    df["TransactionId"] = ["a", "b", "c", "d"]
    df["Category"] = ["Other", "Food", "Other", "Transport"]

    out = apply_category_overrides(df, {"a": "Food", "c": "Transfers"})
    assert out.loc[0, "Category"] == "Food"
    assert out.loc[2, "Category"] == "Transfers"


def test_forecast_and_report_pack_not_empty() -> None:
    df = _sample_df()
    df["Merchant"] = ["A", "B", "C", "D"]
    df["Category"] = ["Food", "Food", "Salary", "Transport"]
    df["TimeOfDay"] = ["Morning", "Evening", "Morning", "Night"]
    df["Währung"] = ["CHF", "CHF", "CHF", "CHF"]
    df["SourceAccount"] = ["CH1", "CH1", "CH1", "CH1"]
    recurring = recurring_transaction_candidates(df.assign(Date=df["Date"], Merchant=df["Merchant"]))
    forecast = forecast_cashflow(df, recurring)
    assert not forecast.empty

    kpis = calculate_kpis(df)
    monthly = daily_net_cashflow(df).resample("ME").sum(numeric_only=True)
    summary, bundle = build_report_pack(df, kpis, monthly)
    assert "PulseLedger Report" in summary
    assert len(bundle) > 0


def test_detect_anomalies_and_duplicate_candidates() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-01-01", "Time": "10:00:00", "DebitCHF": 10.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Cafe"},
            {"Date": "2026-01-02", "Time": "10:00:00", "DebitCHF": 12.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Cafe"},
            {"Date": "2026-01-03", "Time": "10:00:00", "DebitCHF": 300.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Cafe"},
            {"Date": "2026-01-03", "Time": "10:00:00", "DebitCHF": 300.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Cafe"},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["SourceFile"] = "f1"
    df["SourceAccount"] = "CH1"
    df["Währung"] = "CHF"
    df["TimeOfDay"] = "Morning"
    df["TransactionId"] = ["t1", "t2", "t3", "t4"]
    df["CategoryConfidence"] = 0.9
    df = enrich_transaction_intelligence(df)

    anomalies = detect_anomalies(df, z_threshold=0.7)
    dupes = possible_duplicate_candidates(df)
    assert not anomalies.empty
    assert not dupes.empty


def test_salary_benchmarks_and_recommendations_pipeline() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-01-30", "Time": "09:00:00", "DebitCHF": 0.0, "CreditCHF": 6000.0, "Category": "Income & Transfers", "Merchant": "Employer AG"},
            {"Date": "2026-02-28", "Time": "09:00:00", "DebitCHF": 0.0, "CreditCHF": 6100.0, "Category": "Income & Transfers", "Merchant": "Employer AG"},
            {"Date": "2026-01-15", "Time": "18:00:00", "DebitCHF": 900.0, "CreditCHF": 0.0, "Category": "Food & Drink", "Merchant": "COOP"},
            {"Date": "2026-02-16", "Time": "18:00:00", "DebitCHF": 980.0, "CreditCHF": 0.0, "Category": "Food & Drink", "Merchant": "COOP"},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["SourceAccount"] = "CH1"
    df["IsTransfer"] = False
    df["MerchantNormalized"] = df["Merchant"].str.upper()

    salary = monthly_salary_estimate(df)
    benchmarks = benchmark_assessment(df, float(salary["avg_monthly_salary"]))
    recos = spending_recommendations(df, benchmarks)

    assert salary["avg_monthly_salary"] > 0
    assert not benchmarks.empty
    assert not recos.empty


def test_balance_timeline_and_merchant_insights() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-01-01", "Time": "10:00:00", "Saldo": 1000, "SourceAccount": "A", "Merchant": "Store", "DebitCHF": 10, "CreditCHF": 0},
            {"Date": "2026-01-02", "Time": "10:00:00", "Saldo": 950, "SourceAccount": "A", "Merchant": "Store", "DebitCHF": 50, "CreditCHF": 0},
            {"Date": "2026-01-02", "Time": "11:00:00", "Saldo": 5000, "SourceAccount": "B", "Merchant": "Employer", "DebitCHF": 0, "CreditCHF": 2000},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["MerchantNormalized"] = df["Merchant"].str.upper()

    timeline = balance_timeline(df)
    merchants = merchant_insights(df, top_n=5)
    assert not timeline.empty
    assert not merchants.empty


def test_ingestion_quality_and_agent_plan() -> None:
    df = pd.DataFrame(
        [
            {
                "SourceFile": "a.csv",
                "SourceAccount": "CH1",
                "Date": "2026-01-01",
                "Time": "",
                "TimeOfDay": "Unknown",
                "Category": "Other",
                "TransactionId": "t1",
                "StatementFrom": "2026-01-01",
                "StatementTo": "2026-01-31",
                "DebitCHF": 100.0,
                "CreditCHF": 0.0,
            },
            {
                "SourceFile": "a.csv",
                "SourceAccount": "CH1",
                "Date": "2026-01-02",
                "Time": "10:00:00",
                "TimeOfDay": "Morning",
                "Category": "Food",
                "TransactionId": "t1",
                "StatementFrom": "2026-01-01",
                "StatementTo": "2026-01-31",
                "DebitCHF": 120.0,
                "CreditCHF": 0.0,
            },
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["StatementFrom"] = pd.to_datetime(df["StatementFrom"])
    df["StatementTo"] = pd.to_datetime(df["StatementTo"])

    ingestion = ingestion_quality_by_source(df)
    assert not ingestion.empty
    assert int(ingestion.loc[0, "DuplicateTransactionIds"]) == 2

    benchmark_table = pd.DataFrame(
        [
            {"Metric": "Groceries", "Status": "Over", "GapPct": 2.5, "MonthlyActualCHF": 300.0, "MonthlyTargetCHF": 150.0},
            {"Metric": "Savings", "Status": "Low", "GapPct": 6.0, "MonthlyActualCHF": 0.0, "MonthlyTargetCHF": 0.0},
        ]
    )
    plan = generate_agent_action_plan(
        kpis={"transactions": 200},
        quality={"missing_time_pct": 10.0, "other_category_pct": 30.0},
        benchmark_table=benchmark_table,
        anomalies=pd.DataFrame([{"x": 1}]),
        dupes=pd.DataFrame([{"x": 1}]),
        recurring=pd.DataFrame(),
    )
    assert not plan.empty
    assert int(plan["Priority"].min()) == 1


def test_monthly_trend_diagnostics_and_category_momentum() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-01-01", "DebitCHF": 100.0, "CreditCHF": 0.0, "Category": "Food"},
            {"Date": "2026-01-10", "DebitCHF": 50.0, "CreditCHF": 0.0, "Category": "Transport"},
            {"Date": "2026-02-01", "DebitCHF": 180.0, "CreditCHF": 0.0, "Category": "Food"},
            {"Date": "2026-02-15", "DebitCHF": 40.0, "CreditCHF": 0.0, "Category": "Transport"},
            {"Date": "2026-02-20", "DebitCHF": 0.0, "CreditCHF": 500.0, "Category": "Income"},
            {"Date": "2026-03-01", "DebitCHF": 200.0, "CreditCHF": 0.0, "Category": "Food"},
            {"Date": "2026-03-10", "DebitCHF": 20.0, "CreditCHF": 0.0, "Category": "Transport"},
            {"Date": "2026-03-20", "DebitCHF": 0.0, "CreditCHF": 550.0, "Category": "Income"},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    trend = monthly_trend_diagnostics(df, lookback_months=12)
    momentum = category_momentum(df)

    assert not trend.empty
    assert "SpendingMoMCHF" in trend.columns
    assert "NetVolatility3M" in trend.columns
    assert not momentum.empty
    assert "ChangeCHF" in momentum.columns


def test_savings_scenario_respects_exclusions() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-01-01", "DebitCHF": 500.0, "CreditCHF": 0.0, "Category": "Food"},
            {"Date": "2026-01-15", "DebitCHF": 400.0, "CreditCHF": 0.0, "Category": "Transport"},
            {"Date": "2026-02-01", "DebitCHF": 600.0, "CreditCHF": 0.0, "Category": "Food"},
            {"Date": "2026-02-15", "DebitCHF": 300.0, "CreditCHF": 0.0, "Category": "Transport"},
            {"Date": "2026-02-20", "DebitCHF": 800.0, "CreditCHF": 0.0, "Category": "Utilities & Bills"},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    scenario = savings_scenario(
        df,
        target_extra_savings_chf=300.0,
        max_cut_pct=0.2,
        excluded_categories=["Utilities & Bills"],
    )

    assert not scenario.empty
    assert "Utilities & Bills" not in scenario["Category"].tolist()
    assert float(scenario["SuggestedCutCHF"].sum()) <= 300.0 + 1e-6


def test_deep_analytics_helpers_return_expected_shapes() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-01-05", "DebitCHF": 120.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Coop"},
            {"Date": "2026-01-07", "DebitCHF": 0.0, "CreditCHF": 6000.0, "Category": "Income", "Merchant": "Employer"},
            {"Date": "2026-02-05", "DebitCHF": 150.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Coop"},
            {"Date": "2026-02-08", "DebitCHF": 80.0, "CreditCHF": 0.0, "Category": "Transport", "Merchant": "SBB"},
            {"Date": "2026-02-20", "DebitCHF": 0.0, "CreditCHF": 6100.0, "Category": "Income", "Merchant": "Employer"},
            {"Date": "2026-03-02", "DebitCHF": 200.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Migros"},
            {"Date": "2026-03-10", "DebitCHF": 0.0, "CreditCHF": 6150.0, "Category": "Income", "Merchant": "Employer"},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    merchant = merchant_concentration_table(df, top_n=5)
    income = income_concentration_table(df, top_n=5)
    stability = cashflow_stability_metrics(df)
    split = weekday_weekend_split(df)
    size_dist = transaction_size_distribution(df)
    volatility = category_volatility(df, min_months=2)
    run_rate = spending_run_rate_projection(df, lookback_months=2)

    assert not merchant.empty
    assert "CumulativeSharePct" in merchant.columns
    assert not income.empty
    assert float(stability["months"]) == 3.0
    assert not split.empty
    assert set(split["Segment"].tolist()).issubset({"Weekday", "Weekend"})
    assert len(size_dist) == 7
    assert "CoeffVar" in volatility.columns
    assert float(run_rate["lookback_months"]) == 2.0


def test_chart_builder_dataset_supports_cumulative_daily_series() -> None:
    chart = chart_builder_dataset(
        _sample_df(),
        x_axis="Date",
        metric="Spending",
        aggregation="Sum",
        split_by="None",
        top_n=20,
        cumulative=True,
        include_transfers=True,
    )

    assert not chart.empty
    assert chart.index.name == "Date"
    assert "Spending" in chart.columns
    assert float(chart["Spending"].iloc[-1]) == 60.0


def test_chart_builder_dataset_split_and_transfer_filtering() -> None:
    df = pd.DataFrame(
        [
            {
                "Date": "2026-02-01",
                "Time": "10:00:00",
                "DebitCHF": 100.0,
                "CreditCHF": 0.0,
                "Category": "Food",
                "SourceAccount": "A",
                "IsTransfer": False,
            },
            {
                "Date": "2026-02-02",
                "Time": "10:00:00",
                "DebitCHF": 300.0,
                "CreditCHF": 0.0,
                "Category": "Transfer",
                "SourceAccount": "B",
                "IsTransfer": True,
            },
            {
                "Date": "2026-02-03",
                "Time": "10:00:00",
                "DebitCHF": 200.0,
                "CreditCHF": 0.0,
                "Category": "Food",
                "SourceAccount": "A",
                "IsTransfer": False,
            },
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    chart_with_transfers = chart_builder_dataset(
        df,
        x_axis="Category",
        metric="Spending",
        aggregation="Sum",
        split_by="SourceAccount",
        top_n=2,
        cumulative=False,
        include_transfers=True,
    )
    chart_without_transfers = chart_builder_dataset(
        df,
        x_axis="Category",
        metric="Spending",
        aggregation="Sum",
        split_by="SourceAccount",
        top_n=2,
        cumulative=False,
        include_transfers=False,
    )

    assert set(chart_with_transfers.columns.tolist()) == {"A", "B"}
    assert chart_without_transfers.columns.tolist() == ["A"]
    assert float(chart_with_transfers.sum().sum()) > float(chart_without_transfers.sum().sum())


def test_chart_builder_dataset_supports_daily_weekly_monthly_intervals() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-02-02", "Time": "10:00:00", "DebitCHF": 10.0, "CreditCHF": 0.0},
            {"Date": "2026-02-03", "Time": "10:00:00", "DebitCHF": 20.0, "CreditCHF": 0.0},
            {"Date": "2026-02-10", "Time": "10:00:00", "DebitCHF": 30.0, "CreditCHF": 0.0},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    daily = chart_builder_dataset(
        df,
        x_axis="Date",
        metric="Spending",
        aggregation="Sum",
        split_by="None",
        top_n=20,
        cumulative=False,
        date_interval="Daily",
        include_transfers=True,
    )
    weekly = chart_builder_dataset(
        df,
        x_axis="Date",
        metric="Spending",
        aggregation="Sum",
        split_by="None",
        top_n=20,
        cumulative=False,
        date_interval="Weekly",
        include_transfers=True,
    )
    monthly = chart_builder_dataset(
        df,
        x_axis="Date",
        metric="Spending",
        aggregation="Sum",
        split_by="None",
        top_n=20,
        cumulative=False,
        date_interval="Monthly",
        include_transfers=True,
    )

    assert len(daily) == 3
    assert len(weekly) == 2
    assert len(monthly) == 1
    assert float(daily["Spending"].sum()) == 60.0
    assert float(weekly["Spending"].sum()) == 60.0
    assert float(monthly["Spending"].sum()) == 60.0


def test_period_over_period_metrics_compares_equal_windows() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-01-01", "DebitCHF": 100.0, "CreditCHF": 0.0},
            {"Date": "2026-01-02", "DebitCHF": 50.0, "CreditCHF": 0.0},
            {"Date": "2026-01-03", "DebitCHF": 0.0, "CreditCHF": 500.0},
            {"Date": "2026-01-04", "DebitCHF": 90.0, "CreditCHF": 0.0},
            {"Date": "2026-01-05", "DebitCHF": 40.0, "CreditCHF": 0.0},
            {"Date": "2026-01-06", "DebitCHF": 0.0, "CreditCHF": 520.0},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    current = df[df["Date"].dt.date.between(pd.to_datetime("2026-01-04").date(), pd.to_datetime("2026-01-06").date())]

    period = period_over_period_metrics(current, df)
    assert not period.empty
    spending_row = period[period["Metric"] == "Spending (CHF)"].iloc[0]
    assert float(spending_row["CurrentValue"]) == 130.0
    assert float(spending_row["PriorValue"]) == 150.0
    assert float(spending_row["DeltaAbs"]) == -20.0
    assert spending_row["Signal"] == "Improved"


def test_spending_heatmap_matrix_and_opportunity_scanner() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2026-02-02", "Time": "08:00:00", "DebitCHF": 20.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Coop"},
            {"Date": "2026-02-02", "Time": "09:00:00", "DebitCHF": 30.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Coop"},
            {"Date": "2026-02-03", "Time": "20:00:00", "DebitCHF": 0.0, "CreditCHF": 100.0, "Category": "Income", "Merchant": "Employer"},
            {"Date": "2026-03-02", "Time": "08:00:00", "DebitCHF": 60.0, "CreditCHF": 0.0, "Category": "Food", "Merchant": "Coop"},
            {"Date": "2026-03-03", "Time": "18:00:00", "DebitCHF": 80.0, "CreditCHF": 0.0, "Category": "Dining", "Merchant": "Restaurant"},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["MerchantNormalized"] = df["Merchant"].str.upper()

    matrix = spending_heatmap_matrix(df, value_metric="Spending")
    opportunities = savings_opportunity_scanner(df, top_n=10)

    assert matrix.shape == (7, 24)
    assert float(matrix.to_numpy().sum()) == 190.0
    assert not opportunities.empty
    assert "PotentialMonthlySavingsCHF" in opportunities.columns
