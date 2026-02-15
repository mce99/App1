from pathlib import Path

from local_sources import collect_statement_paths, load_local_statement_uploads


def test_collect_statement_paths_filters_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "b.xlsx").write_text("x")
    (tmp_path / "c.txt").write_text("x")

    out = collect_statement_paths(str(tmp_path), recursive=False, supported_extensions=[".csv", ".xlsx"])
    names = {path.name for path in out}
    assert "a.csv" in names
    assert "b.xlsx" in names
    assert "c.txt" not in names


def test_load_local_statement_uploads_returns_named_file_wrappers(tmp_path: Path) -> None:
    p = tmp_path / "sample.csv"
    p.write_text("Abschlussdatum;Abschlusszeit\n2026-01-01;12:00:00\n")

    uploads = load_local_statement_uploads(
        folder_path=str(tmp_path),
        recursive=False,
        supported_extensions=[".csv"],
        max_files=10,
    )
    assert len(uploads) == 1
    upload = uploads[0]
    assert str(p) == upload.name
    upload.seek(0)
    assert b"Abschlussdatum" in upload.read()
