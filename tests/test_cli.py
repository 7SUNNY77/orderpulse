"""CLI smoke tests: help, version, and readable errors when data or artifacts are missing."""

from __future__ import annotations

import pytest

import orderpulse
from orderpulse.cli import main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ["pipeline", "app", "ingest", "reliability"]:
        assert cmd in out


def test_version_is_release():
    assert orderpulse.__version__ == "1.0.0"


def test_missing_csv_gives_hint_not_traceback(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ORDERPULSE_RAW_DB", str(tmp_path / "processed" / "raw.duckdb"))
    code = main(["ingest", "--raw-dir", str(tmp_path / "empty")])
    err = capsys.readouterr().err
    assert code == 2
    assert "9 Olist CSV file(s) not found" in err and "--raw-dir" in err
    assert not (tmp_path / "processed" / "raw.duckdb").exists()


def test_missing_artifacts_give_hint(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ORDERPULSE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ORDERPULSE_WAREHOUSE_DB", str(tmp_path / "nope.duckdb"))
    code = main(["predict"])
    assert code == 2
    assert "python -m orderpulse pipeline" in capsys.readouterr().err
