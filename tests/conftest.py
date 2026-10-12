from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))  # Streamlit pages import `common`

from synthetic import make_tables, write_csvs  # noqa: E402

from orderpulse.build_marts import build_warehouse  # noqa: E402
from orderpulse.config import PACKAGE_ROOT, load_config  # noqa: E402
from orderpulse.ingest import ingest  # noqa: E402


def make_cfg(base: Path, tables=None):
    raw = base / "raw"
    write_csvs(tables if tables is not None else make_tables(), raw)
    return load_config(
        root=PACKAGE_ROOT,
        raw_dir=str(raw),
        processed_dir=str(base / "processed"),
        raw_db=str(base / "processed" / "raw.duckdb"),
        warehouse_db=str(base / "processed" / "warehouse.duckdb"),
        reports_dir=str(base / "reports"),
        artifacts_dir=str(base / "artifacts"),
    )


@pytest.fixture(scope="session")
def built_cfg(tmp_path_factory):
    """Synthetic dataset ingested and built once per test session (read-only use)."""
    cfg = make_cfg(tmp_path_factory.mktemp("synthetic"))
    ingest(cfg)
    build_warehouse(cfg)
    return cfg


@pytest.fixture()
def wh(built_cfg):
    con = duckdb.connect(str(built_cfg.warehouse_db), read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def pipeline_cfg(tmp_path_factory):
    """Full synthetic pipeline: marts -> train -> evaluate -> predict (shared by several test files)."""
    from orderpulse.evaluate import evaluate
    from orderpulse.predict import predict
    from orderpulse.quality import run_quality
    from orderpulse.train import train
    cfg = make_cfg(tmp_path_factory.mktemp("integration"))
    ingest(cfg)
    run_quality(cfg)
    build_warehouse(cfg)
    train(cfg)
    evaluate(cfg)
    predict(cfg)
    return cfg


def pytest_collection_modifyitems(config, items):
    """Tests that need the full synthetic pipeline or Streamlit AppTest are marked `integration`.

    No test needs the real Olist CSV: everything runs on tests/synthetic.py data.
    Quick run without them: ``python -m pytest -m "not integration"``.
    """
    for item in items:
        if "pipeline_cfg" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.integration)
