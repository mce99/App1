"""Local filesystem statement source helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable


class LocalStatementUpload(io.BytesIO):
    """In-memory file wrapper compatible with parsing.merge_transactions."""

    def __init__(self, path: Path, payload: bytes) -> None:
        super().__init__(payload)
        self.name = str(path)


def collect_statement_paths(
    folder_path: str,
    recursive: bool = False,
    supported_extensions: Iterable[str] = (),
) -> list[Path]:
    """Collect statement file paths from a folder."""
    root = Path(folder_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    exts = {str(ext).lower() for ext in supported_extensions}
    if not exts:
        exts = {".csv", ".xlsx", ".xls", ".zip"}

    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in exts
    ]
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def load_local_statement_uploads(
    folder_path: str,
    recursive: bool = False,
    supported_extensions: Iterable[str] = (),
    max_files: int = 200,
) -> list[LocalStatementUpload]:
    """Load local statement files into in-memory upload wrappers."""
    paths = collect_statement_paths(
        folder_path=folder_path,
        recursive=recursive,
        supported_extensions=supported_extensions,
    )
    uploads: list[LocalStatementUpload] = []
    for path in paths[: int(max_files)]:
        uploads.append(LocalStatementUpload(path=path, payload=path.read_bytes()))
    return uploads
