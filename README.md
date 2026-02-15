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
- AI Coach (optional OpenAI API key, offline fallback if no key)
- Bank Sync workspace for guided UBS browser download capture + local folder auto-ingest

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

## UBS Browser Sync (local, guided)

Run this locally to open UBS in a browser, log in manually, and capture downloaded statements into a sync folder:

```bash
cd "/Users/mce/Documents/New project"
.venv/bin/python ubs_browser_sync.py --download-dir "~/Downloads/ubs_statements"
```

Then enable `Local sync folder` in the app sidebar (or `Bank Sync` workspace) to ingest those files automatically.

## Quality checks

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```
