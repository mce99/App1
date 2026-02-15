# AGENTS.md

## How to work in this repo
- Always run: `.venv/bin/python -m ruff check .` and `.venv/bin/python -m pytest` before proposing PR-ready changes
- Keep changes minimal and explain them
- Never commit secrets; never edit `.env`

## Commands
- Install: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Dev server: `.venv/bin/streamlit run app.py`
- Lint: `.venv/bin/python -m ruff check .`
- Tests: `.venv/bin/python -m pytest`

## Conventions
- <formatting, patterns, architecture notes>

## Safety boundaries
- Do not touch: <prod config paths, infra files, billing code, etc> unless explicitly requested
