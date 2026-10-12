"""raw -> staging -> marts -> features, with reconciliation checks.

The warehouse is always rebuilt from scratch into ``warehouse.duckdb.tmp``;
only if every build check passes is it moved over ``warehouse.duckdb``. A
re-run therefore produces the same tables (idempotent) and a failed run leaves
the previous warehouse untouched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .config import Config
from .ingest import REQUIRED_COLUMNS
from .utils import LOG, atomic_replace, remove_if_exists, timed, tmp_path_for, write_json_atomic

LAYERS = ("staging", "marts", "features")


class BuildCheckError(RuntimeError):
    """A reconciliation invariant of the marts is violated (e.g. row multiplication)."""


def sql_params(cfg: Config) -> dict[str, str]:
    b = cfg.data["geo_bounds"]
    return {k: str(float(v)) for k, v in b.items()}


def run_sql_dir(con: duckdb.DuckDBPyConnection, directory: Path, params: dict[str, str]) -> None:
    for path in sorted(directory.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        for key, value in params.items():
            sql = sql.replace("{" + key + "}", value)
        LOG.debug("running %s", path.name)
        con.execute(sql)


def attach_raw(con: duckdb.DuckDBPyConnection, raw_db: Path) -> None:
    """Expose raw tables as temp views ``raw_<table>`` on this connection."""
    if not raw_db.exists():
        raise FileNotFoundError(f"{raw_db} not found: run `python -m orderpulse ingest` first")
    con.execute(f"ATTACH '{raw_db.as_posix()}' AS rawdb (READ_ONLY)")
    for table in REQUIRED_COLUMNS:
        con.execute(f"CREATE OR REPLACE TEMP VIEW raw_{table} AS SELECT * FROM rawdb.raw_{table}")


def build_staging(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    run_sql_dir(con, cfg.sql_dir / "staging", sql_params(cfg))


def _one(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def run_build_checks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Invariants that must hold; any failure aborts the build."""
    checks: list[dict] = []

    def check(name: str, actual, expected, description: str) -> None:
        ok = actual == expected
        checks.append({"check": name, "ok": bool(ok), "actual": str(actual), "expected": str(expected),
                       "description": description})

    n_orders = _one(con, "SELECT count(*) FROM stg_orders")
    check("mart_orders_rows", _one(con, "SELECT count(*) FROM mart_orders"), n_orders,
          "mart_orders has exactly as many rows as source orders")
    check("mart_orders_unique_order_id",
          _one(con, "SELECT count(*) - count(DISTINCT order_id) FROM mart_orders"), 0,
          "no duplicated order_id in mart_orders")
    check("items_value_reconciled",
          _one(con, "SELECT coalesce(sum(items_value), 0) FROM mart_orders"),
          _one(con, "SELECT coalesce(sum(price), 0) FROM stg_order_items"),
          "sum of item prices is identical before/after aggregation (no multiplication)")
    check("freight_value_reconciled",
          _one(con, "SELECT coalesce(sum(freight_value), 0) FROM mart_orders"),
          _one(con, "SELECT coalesce(sum(freight_value), 0) FROM stg_order_items"),
          "sum of freight is identical before/after aggregation")
    check("payment_value_reconciled",
          _one(con, "SELECT coalesce(sum(payment_value), 0) FROM mart_orders"),
          _one(con, "SELECT coalesce(sum(payment_value), 0) FROM stg_order_payments "
                    "WHERE order_id IN (SELECT order_id FROM stg_orders)"),
          "sum of payments of known orders is identical before/after aggregation")
    check("n_items_reconciled",
          _one(con, "SELECT sum(n_items) FROM mart_orders"),
          _one(con, "SELECT count(*) FROM stg_order_items WHERE order_id IN (SELECT order_id FROM stg_orders)"),
          "number of items is preserved")
    check("orders_with_review",
          _one(con, "SELECT count(*) FROM mart_orders WHERE has_review"),
          _one(con, "SELECT count(DISTINCT order_id) FROM stg_order_reviews "
                    "WHERE order_id IN (SELECT order_id FROM stg_orders)"),
          "each reviewed order has exactly one canonical review")
    check("mart_order_items_rows", _one(con, "SELECT count(*) FROM mart_order_items"),
          _one(con, "SELECT count(*) FROM stg_order_items"), "one row per item")
    check("mart_seller_orders_rows", _one(con, "SELECT count(*) FROM mart_seller_orders"),
          _one(con, "SELECT count(*) FROM (SELECT DISTINCT order_id, seller_id FROM stg_order_items)"),
          "one row per (seller, order)")
    check("mart_seller_orders_items_value",
          _one(con, "SELECT coalesce(sum(items_value), 0) FROM mart_seller_orders"),
          _one(con, "SELECT coalesce(sum(price), 0) FROM stg_order_items"),
          "seller-order item values sum to total item value")
    for table, key in [("stg_customers", "customer_id"), ("stg_sellers", "seller_id"),
                       ("stg_products", "product_id"), ("stg_geo_zip", "zip")]:
        check(f"{table}_unique_{key}",
              _one(con, f"SELECT count(*) - count(DISTINCT {key}) FROM {table}"), 0,
              f"dimension {table} is unique by {key} (joins cannot multiply rows)")
    check("features_rows",
          _one(con, "SELECT count(*) FROM feature_orders_at_approval"),
          _one(con, "SELECT count(*) FROM stg_orders WHERE approved_ts IS NOT NULL"),
          "one feature row per approved order")
    check("features_unique_order_id",
          _one(con, "SELECT count(*) - count(DISTINCT order_id) FROM feature_orders_at_approval"), 0,
          "feature rows are unique by order_id")
    check("ml_labels_rows", _one(con, "SELECT count(*) FROM ml_labels"), n_orders, "one label row per order")
    return checks


def reconciliation_report(con: duckdb.DuckDBPyConnection) -> dict:
    """Payments vs items+freight: reported, never hidden or 'fixed'."""
    row = con.execute(
        """
        SELECT
            count(*) FILTER (WHERE has_items AND n_payment_records > 0)                                AS comparable,
            count(*) FILTER (WHERE has_items AND n_payment_records > 0
                             AND abs(payment_minus_items_freight) <= 0.01)                             AS matched,
            count(*) FILTER (WHERE payment_minus_items_freight > 0.01)                                 AS payment_higher,
            count(*) FILTER (WHERE payment_minus_items_freight < -0.01)                                AS payment_lower,
            sum(payment_minus_items_freight)                                                           AS total_difference,
            count(*) FILTER (WHERE n_payment_records = 0)                                              AS orders_without_payment,
            count(*) FILTER (WHERE NOT has_items)                                                      AS orders_without_items
        FROM mart_orders
        """
    ).fetchone()
    keys = ["comparable", "matched", "payment_higher", "payment_lower", "total_difference",
            "orders_without_payment", "orders_without_items"]
    return {k: (float(v) if k == "total_difference" else int(v)) for k, v in zip(keys, row)}


def build_warehouse(cfg: Config) -> Path:
    with timed("build_marts"):
        target = cfg.warehouse_db
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_path_for(target)
        remove_if_exists(tmp)
        con = duckdb.connect(str(tmp))
        try:
            attach_raw(con, cfg.raw_db)
            params = sql_params(cfg)
            for layer in LAYERS:
                with timed(f"sql:{layer}"):
                    run_sql_dir(con, cfg.sql_dir / layer, params)
            checks = run_build_checks(con)
            recon = reconciliation_report(con)
            failed = [c for c in checks if not c["ok"]]
            con.execute("CREATE OR REPLACE TABLE meta_build (key VARCHAR, value VARCHAR)")
            fp = con.execute("SELECT value FROM rawdb.meta_run WHERE key = 'data_fingerprint'").fetchone()[0]
            built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            con.executemany(
                "INSERT INTO meta_build VALUES (?, ?)",
                [("data_fingerprint", fp), ("built_at", built_at),
                 ("data_cutoff", str(cfg.data["data_cutoff"])),
                 ("feature_version", str(cfg.model["feature_version"]))],
            )
            con.execute("DETACH rawdb")
        except Exception:
            con.close()
            remove_if_exists(tmp)
            raise
        con.close()
        report = {"built_at": built_at, "data_fingerprint": fp, "checks": checks,
                  "payments_reconciliation": recon}
        if failed:
            remove_if_exists(tmp)
            write_json_atomic(report, cfg.reports_dir / "build_checks.json")
            raise BuildCheckError(f"{len(failed)} build check(s) failed: {[c['check'] for c in failed]}")
        atomic_replace(tmp, target)
        write_json_atomic(report, cfg.reports_dir / "build_checks.json")
        LOG.info("warehouse built: %s (%d checks passed)", target, len(checks))
        return target
