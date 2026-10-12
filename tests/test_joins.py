"""Order-grain marts: no row multiplication, exact sums, canonical review, idempotency."""

from __future__ import annotations

import hashlib
from decimal import Decimal

import duckdb
import pytest

from conftest import make_cfg
from orderpulse import build_marts
from orderpulse.build_marts import BuildCheckError, build_warehouse
from orderpulse.ingest import ingest


def row(con, sql, *params):
    return con.execute(sql, list(params)).fetchone()


def test_three_items_two_payments_two_reviews_gives_one_row(wh):
    n, items_value, freight, payment, n_items, n_pay, n_rev, review_id, score = row(
        wh, """SELECT count(*), any_value(items_value), any_value(freight_value), any_value(payment_value),
                      any_value(n_items), any_value(n_payment_records), any_value(n_reviews),
                      any_value(review_id), any_value(review_score)
               FROM mart_orders WHERE order_id = 'o_multi'""")
    assert n == 1
    assert items_value == Decimal("60.50")      # 10 + 20 + 30.50, not multiplied by payments/reviews
    assert freight == Decimal("6.00")
    assert payment == Decimal("66.50")          # 40 + 26.50, not multiplied by items/reviews
    assert (n_items, n_pay, n_rev) == (3, 2, 2)
    assert review_id == "r_multi_new" and score == 5   # latest review_creation_date wins


def test_canonical_review_tie_is_deterministic(wh):
    assert row(wh, "SELECT review_id FROM mart_orders WHERE order_id = 'o_tie'")[0] == "r_tie_b"


def test_mart_orders_unique_and_complete(wh):
    n, distinct = row(wh, "SELECT count(*), count(DISTINCT order_id) FROM mart_orders")
    assert n == distinct == row(wh, "SELECT count(*) FROM stg_orders")[0]


def test_sums_reconcile_with_sources(wh):
    assert row(wh, "SELECT sum(items_value) FROM mart_orders")[0] == row(wh, "SELECT sum(price) FROM stg_order_items")[0]
    assert row(wh, "SELECT sum(payment_value) FROM mart_orders")[0] == \
        row(wh, "SELECT sum(payment_value) FROM stg_order_payments")[0]


def test_multi_seller_order_levels(wh):
    # one order, two sellers: one row at order level, two at seller-order level
    assert row(wh, "SELECT n_sellers, is_multi_seller FROM mart_orders WHERE order_id = 'o_multi'") == (2, True)
    rows = wh.execute("SELECT seller_id, items_value FROM mart_seller_orders WHERE order_id = 'o_multi' ORDER BY 1").fetchall()
    assert rows == [("s00", Decimal("30.00")), ("s01", Decimal("30.50"))]
    seller_rows = row(wh, "SELECT count(*) FROM mart_seller_orders")[0]
    unique_orders = row(wh, "SELECT count(DISTINCT order_id) FROM mart_seller_orders")[0]
    assert seller_rows > unique_orders  # summing seller statistics does not give unique orders


def test_order_without_items_is_kept_with_null_value(wh):
    items_value, n_items, has_items = row(wh, "SELECT items_value, n_items, has_items FROM mart_orders WHERE order_id = 'o_noitems'")
    assert items_value is None and n_items == 0 and has_items is False


def test_zero_price_is_zero_not_null(wh):
    assert row(wh, "SELECT items_value, has_items FROM mart_orders WHERE order_id = 'o_zero'") == (Decimal("0.00"), True)


def test_unknown_and_untranslated_category(wh):
    assert row(wh, "SELECT main_category FROM mart_orders WHERE order_id = 'o_unknowncat'")[0] == "unknown"
    assert row(wh, "SELECT main_category FROM mart_orders WHERE order_id = 'o_untranslated'")[0] == "cat_no_translation"


def test_geolocation_one_row_per_prefix_and_outliers_dropped(wh):
    n, distinct = row(wh, "SELECT count(*), count(DISTINCT zip) FROM stg_geo_zip")
    assert n == distinct
    lat = row(wh, "SELECT lat FROM stg_geo_zip WHERE zip = '01001'")[0]
    assert -24 < lat < -23  # the (40, 10) outlier did not move the median


def _warehouse_digest(path) -> dict:
    con = duckdb.connect(str(path), read_only=True)
    out = {}
    for t in ["mart_orders", "mart_order_items", "mart_seller_orders", "mart_delivery_segments",
              "feature_orders_at_approval", "ml_labels"]:
        df = con.execute(f"SELECT * FROM {t}").df()
        df = df.sort_values(list(df.columns)).reset_index(drop=True)
        out[t] = (len(df), hashlib.sha256(df.to_csv(index=False).encode()).hexdigest())
    con.close()
    return out


def test_etl_is_idempotent(tmp_path):
    cfg = make_cfg(tmp_path)
    ingest(cfg)
    build_warehouse(cfg)
    first = _warehouse_digest(cfg.warehouse_db)
    ingest(cfg)
    build_warehouse(cfg)
    second = _warehouse_digest(cfg.warehouse_db)
    assert first == second


def test_failed_build_keeps_previous_warehouse(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    ingest(cfg)
    build_warehouse(cfg)
    before = _warehouse_digest(cfg.warehouse_db)

    def broken_checks(con):
        return [{"check": "forced_failure", "ok": False, "actual": "1", "expected": "0", "description": "test"}]

    monkeypatch.setattr(build_marts, "run_build_checks", broken_checks)
    with pytest.raises(BuildCheckError):
        build_warehouse(cfg)
    assert _warehouse_digest(cfg.warehouse_db) == before
    assert not (cfg.warehouse_db.with_name(cfg.warehouse_db.name + ".tmp")).exists()
