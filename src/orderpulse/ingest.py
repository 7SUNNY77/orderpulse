"""CSV -> raw layer.

The raw layer keeps every value exactly as text (``all_varchar``), so nothing is
silently coerced at load time; typing happens in the staging SQL with TRY_CAST
and cast failures are counted by the quality audit.

The raw database is written to a temporary file and moved into place only after
all tables are loaded, so a failure never leaves a half-built raw.duckdb.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .config import Config
from .utils import LOG, atomic_replace, remove_if_exists, sha256_file, short_hash, timed, tmp_path_for

# Column contract: the pipeline stops if any of these files/columns is missing.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "orders": [
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ],
    "order_items": [
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    ],
    "order_payments": [
        "order_id", "payment_sequential", "payment_type",
        "payment_installments", "payment_value",
    ],
    "order_reviews": [
        "review_id", "order_id", "review_score", "review_comment_title",
        "review_comment_message", "review_creation_date", "review_answer_timestamp",
    ],
    "customers": [
        "customer_id", "customer_unique_id", "customer_zip_code_prefix",
        "customer_city", "customer_state",
    ],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "products": [
        "product_id", "product_category_name", "product_name_lenght",
        "product_description_lenght", "product_photos_qty", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    ],
    "geolocation": [
        "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
        "geolocation_city", "geolocation_state",
    ],
    "category_translation": ["product_category_name", "product_category_name_english"],
}


class SchemaError(RuntimeError):
    """Critical contract violation: a required file or column is missing."""


def _read_csv_sql(path: Path) -> str:
    p = path.as_posix().replace("'", "''")
    return (
        f"read_csv('{p}', header=true, all_varchar=true, quote='\"', escape='\"', "
        f"strict_mode=false, null_padding=false)"
    )


def check_contract(cfg: Config) -> dict[str, list[str]]:
    """Return actual columns per table; raise SchemaError on missing files/columns."""
    con = duckdb.connect()
    problems: list[str] = []
    actual: dict[str, list[str]] = {}
    for table, required in REQUIRED_COLUMNS.items():
        path = cfg.csv_path(table)
        if not path.exists():
            problems.append(f"missing file for '{table}': {path}")
            continue
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {_read_csv_sql(path)}").fetchall()]
        cols = [c.lstrip("﻿") for c in cols]
        actual[table] = cols
        missing = [c for c in required if c not in cols]
        if missing:
            problems.append(f"'{table}' is missing columns: {missing}")
    con.close()
    if problems:
        raise SchemaError("; ".join(problems))
    return actual


def data_fingerprint(cfg: Config) -> dict:
    files = {t: sha256_file(cfg.csv_path(t)) for t in REQUIRED_COLUMNS}
    return {"files": files, "combined": short_hash(files, 16)}


def ingest(cfg: Config) -> Path:
    """Load all CSVs as text into ``raw_<table>`` tables of raw.duckdb."""
    with timed("ingest"):
        check_contract(cfg)
        fp = data_fingerprint(cfg)
        target = cfg.raw_db
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_path_for(target)
        remove_if_exists(tmp)
        con = duckdb.connect(str(tmp))
        try:
            rows = []
            for table, required in REQUIRED_COLUMNS.items():
                path = cfg.csv_path(table)
                src = _read_csv_sql(path)
                # Strip a possible UTF-8 BOM from header names (seen in the translation file).
                cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
                select = ", ".join(f'"{c}" AS "{c.lstrip(chr(0xFEFF))}"' for c in cols)
                con.execute(f"CREATE TABLE raw_{table} AS SELECT {select} FROM {src}")
                n = con.execute(f"SELECT count(*) FROM raw_{table}").fetchone()[0]
                rows.append((table, path.name, n, fp["files"][table]))
                LOG.info("raw_%s: %d rows", table, n)
            con.execute(
                "CREATE TABLE meta_ingest (table_name VARCHAR, file_name VARCHAR, n_rows BIGINT, sha256 VARCHAR)"
            )
            con.executemany("INSERT INTO meta_ingest VALUES (?, ?, ?, ?)", rows)
            con.execute(
                "CREATE TABLE meta_run (key VARCHAR, value VARCHAR)"
            )
            con.executemany(
                "INSERT INTO meta_run VALUES (?, ?)",
                [
                    ("data_fingerprint", fp["combined"]),
                    ("ingested_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
                    ("data_cutoff", str(cfg.data["data_cutoff"])),
                ],
            )
        finally:
            con.close()
        atomic_replace(tmp, target)
        return target
