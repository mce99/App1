"""Human-readable metric definitions for the app."""

METRIC_GUIDE = [
    {
        "Metric": "Spending (CHF)",
        "Meaning": "Total outgoing amount in selected period after conversion to CHF.",
        "Formula": "sum(DebitCHF)",
    },
    {
        "Metric": "Earnings (CHF)",
        "Meaning": "Total incoming amount in selected period after conversion to CHF.",
        "Formula": "sum(CreditCHF)",
    },
    {
        "Metric": "Net cashflow",
        "Meaning": "How much you gained or lost overall in the selected period.",
        "Formula": "Earnings - Spending",
    },
    {
        "Metric": "Savings rate",
        "Meaning": "Share of earnings kept after spending.",
        "Formula": "(Net cashflow / Earnings) * 100",
    },
    {
        "Metric": "Transactions",
        "Meaning": "Total number of transaction rows in current filters.",
        "Formula": "count(rows)",
    },
    {
        "Metric": "Active days",
        "Meaning": "Number of unique dates with at least one transaction.",
        "Formula": "count(unique(Date))",
    },
    {
        "Metric": "Calendar days",
        "Meaning": "Number of days from min selected date to max selected date inclusive.",
        "Formula": "(max(Date) - min(Date)) + 1",
    },
    {
        "Metric": "Avg spend / tx",
        "Meaning": "Average outgoing amount per transaction row.",
        "Formula": "mean(DebitCHF)",
    },
    {
        "Metric": "Avg earning / tx",
        "Meaning": "Average incoming amount per transaction row.",
        "Formula": "mean(CreditCHF)",
    },
    {
        "Metric": "Avg spend / active day",
        "Meaning": "Average total spending on days where transactions exist.",
        "Formula": "Spending / Active days",
    },
    {
        "Metric": "Avg earn / active day",
        "Meaning": "Average total earnings on days where transactions exist.",
        "Formula": "Earnings / Active days",
    },
    {
        "Metric": "Avg spend / calendar day",
        "Meaning": "Average spending spread across the full date range, including inactive days.",
        "Formula": "Spending / Calendar days",
    },
    {
        "Metric": "Avg earn / calendar day",
        "Meaning": "Average earnings spread across the full date range, including inactive days.",
        "Formula": "Earnings / Calendar days",
    },
    {
        "Metric": "Avg tx / active day",
        "Meaning": "Transaction density on days where activity occurred.",
        "Formula": "Transactions / Active days",
    },
    {
        "Metric": "Avg daily net",
        "Meaning": "Average daily net result on active days.",
        "Formula": "Net cashflow / Active days",
    },
    {
        "Metric": "Largest spend tx",
        "Meaning": "Single largest outgoing transaction value.",
        "Formula": "max(DebitCHF)",
    },
    {
        "Metric": "Largest earning tx",
        "Meaning": "Single largest incoming transaction value.",
        "Formula": "max(CreditCHF)",
    },
    {
        "Metric": "Cumulative spending",
        "Meaning": "Running total of spending over time.",
        "Formula": "cumsum(daily spending)",
    },
    {
        "Metric": "Cumulative earnings",
        "Meaning": "Running total of earnings over time.",
        "Formula": "cumsum(daily earnings)",
    },
    {
        "Metric": "Cumulative net",
        "Meaning": "Running result after earnings minus spending each day.",
        "Formula": "cumsum(daily net)",
    },
]
