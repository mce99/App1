# App1

Streamlit app for parsing bank transaction exports and summarizing spending/income by category.
Supports multi-file upload (`.csv`, `.xlsx`, `.xls`) so you can combine batched exports.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

CSV files are expected to be semicolon-delimited (`;`).

## Quality checks

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```
