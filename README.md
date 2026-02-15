# App1

Streamlit app for parsing bank transaction exports and summarizing spending/income by category.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Quality checks

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```
