#!/usr/bin/env python3
"""Guided UBS browser sync: manual login + automated download capture."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import threading
from pathlib import Path


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return cleaned or "statement.csv"


def _unique_destination(download_dir: Path, suggested_name: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _sanitize_filename(suggested_name)
    candidate = download_dir / f"{stamp}_{base}"
    counter = 1
    while candidate.exists():
        candidate = download_dir / f"{stamp}_{counter}_{base}"
        counter += 1
    return candidate


def run_sync(
    start_url: str,
    download_dir: Path,
    timeout_minutes: int,
) -> list[Path]:
    from playwright.sync_api import sync_playwright

    download_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    stop_event = threading.Event()

    def wait_for_enter() -> None:
        try:
            input("\nPress ENTER here when you are done exporting statements...\n")
        except EOFError:
            pass
        stop_event.set()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        def on_download(download) -> None:
            try:
                target = _unique_destination(download_dir, download.suggested_filename)
                download.save_as(str(target))
                saved.append(target)
                print(f"[saved] {target}")
            except Exception as exc:  # pragma: no cover - runtime-only fallback
                print(f"[warn] failed to save download: {exc}")

        context.on("download", on_download)

        page.goto(start_url, wait_until="domcontentloaded")
        print("\nUBS Sync started.")
        print("1) Log in manually in the opened browser window.")
        print("2) Complete MFA manually.")
        print("3) Export/download statement files (CSV/XLS/XLSX).")
        print("4) Downloads are captured automatically into:")
        print(f"   {download_dir}")
        print("\nNo credentials are read or stored by this script.\n")

        input_thread = threading.Thread(target=wait_for_enter, daemon=True)
        input_thread.start()

        deadline = dt.datetime.now() + dt.timedelta(minutes=int(timeout_minutes))
        while not stop_event.is_set():
            if dt.datetime.now() >= deadline:
                print("[info] sync timed out. finishing.")
                break
            page.wait_for_timeout(500)

        context.close()
        browser.close()

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture UBS statement downloads via a guided browser session.")
    parser.add_argument(
        "--start-url",
        default="https://www.ubs.com/ch/en/services/guide/accounts-and-cards/e-banking.html",
        help="URL to open first (you can navigate manually afterwards).",
    )
    parser.add_argument(
        "--download-dir",
        default="~/Downloads/ubs_statements",
        help="Folder where captured downloads are saved.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=45,
        help="Safety timeout for the guided session.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(str(args.download_dir)).expanduser()
    files = run_sync(
        start_url=str(args.start_url),
        download_dir=out_dir,
        timeout_minutes=int(args.timeout_minutes),
    )

    if not files:
        print("\nNo files captured.")
        return

    print("\nCaptured files:")
    for path in files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
