# App1

Streamlit app for parsing bank transaction exports and summarizing spending/income by category.
Supports multi-file upload (`.csv`, `.xlsx`, `.xls`) so you can combine batched exports.
Now includes a modular finance workspace with:
- transaction intelligence (forecasts, anomalies, recurring detection, transfer/account tracking)
- stock positions (symbol, quantity, average buy price)
- crypto wallet lookups for BTC / ETH / SOL addresses
- Agent Console (auto-prioritized action queue + ingestion quality diagnostics)
- Category Lab for unlabeled transactions with merchant-rule learning
- Mapping Studio page for manual mapping + smart token-rule learning
- Mapping memory persistence (rules + labels survive app restarts)
- AI Coach (optional OpenAI API key, offline fallback if no key)
- Savings simulator and month-over-month trend diagnostics in Plan & Improve
- Deep Analytics tab (cashflow stability, run-rate projections, concentration, size bands, volatility)

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

CSV files are expected to be semicolon-delimited (`;`).
You can also upload `.zip` bundles containing many CSV/XLS/XLSX statements.
For UBS-style files with missing times, ordering now falls back to booking/value dates.
The app is organized into simple workspaces: Overview, Money In/Out, Plan & Improve, Mapping, Data & QA, Portfolio, Guide.

## Quality checks

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```
