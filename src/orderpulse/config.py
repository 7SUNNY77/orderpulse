"""Configuration loading and path resolution.

The project root is resolved in this order: explicit argument, ORDERPULSE_ROOT
environment variable, the repository that contains this package. Relative paths
in configs/*.yaml are resolved against the root, so no machine-specific absolute
paths are stored anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    env = os.environ.get("ORDERPULSE_ROOT")
    if env:
        return Path(env).resolve()
    return PACKAGE_ROOT


@dataclass
class Config:
    root: Path
    data: dict[str, Any]
    model: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)

    def path(self, key: str) -> Path:
        value = self.overrides.get(key, self.data[key])
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    @property
    def raw_dir(self) -> Path:
        return self.path("raw_dir")

    @property
    def raw_db(self) -> Path:
        return self.path("raw_db")

    @property
    def warehouse_db(self) -> Path:
        return self.path("warehouse_db")

    @property
    def reports_dir(self) -> Path:
        return self.path("reports_dir")

    @property
    def artifacts_dir(self) -> Path:
        return self.path("artifacts_dir")

    @property
    def sql_dir(self) -> Path:
        return PACKAGE_ROOT / "sql"

    def csv_path(self, table: str) -> Path:
        return self.raw_dir / self.data["files"][table]


def load_config(
    root: str | Path | None = None,
    config_dir: str | Path | None = None,
    **overrides: Any,
) -> Config:
    """Load configs/data.yaml and configs/model.yaml.

    ``overrides`` may replace any top-level key of data.yaml (e.g. ``raw_dir``),
    which is how the CLI ``--raw-dir`` option and the tests point the pipeline
    at another location.
    """
    root_path = _resolve_root(root)
    cdir = Path(config_dir) if config_dir else root_path / "configs"
    if not cdir.is_absolute():
        cdir = root_path / cdir
    if not (cdir / "data.yaml").exists() or not (PACKAGE_ROOT / "sql").is_dir():
        raise FileNotFoundError(
            f"OrderPulse configs/sql not found (configs: {cdir}, sql: {PACKAGE_ROOT / 'sql'}). Run from a clone of "
            "the repository installed in editable mode (`pip install -e .`), or set ORDERPULSE_ROOT.")
    with open(cdir / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    with open(cdir / "model.yaml", encoding="utf-8") as f:
        model = yaml.safe_load(f)
    # Environment overrides: ORDERPULSE_<KEY> for any top-level key of data.yaml
    # (e.g. ORDERPULSE_RAW_DIR, ORDERPULSE_WAREHOUSE_DB); explicit arguments win.
    env = {k: os.environ[f"ORDERPULSE_{k.upper()}"] for k in data if f"ORDERPULSE_{k.upper()}" in os.environ}
    clean = {**env, **{k: v for k, v in overrides.items() if v is not None}}
    return Config(root=root_path, data=data, model=model, overrides=clean)
