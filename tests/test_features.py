"""Anti-leakage tests for the prediction-time feature matrix."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from conftest import make_cfg
from orderpulse.build_marts import build_warehouse
from orderpulse.features import (FEATURES, FORBIDDEN_SUBSTRINGS, LeakageError, assert_no_forbidden,
                                 load_feature_frame, to_X)
from orderpulse.ingest import ingest
from synthetic import make_tables

FORBIDDEN_COLUMNS = ["order_delivered_customer_date", "delivered_ts", "delivery_days", "delay_days", "is_late",
                     "review_score", "order_status", "outcome", "order_id", "customer_id", "customer_unique_id",
                     "seller_id", "payment_value", "carrier_ts"]


def test_allow_list_has_no_forbidden_columns():
    assert_no_forbidden(FEATURES)
    for c in FORBIDDEN_COLUMNS:
        assert c not in FEATURES
        with pytest.raises(LeakageError):
            assert_no_forbidden(FEATURES + [c])


def test_X_contains_exactly_the_allow_list(wh):
    X = to_X(load_feature_frame(wh))
    assert list(X.columns) == FEATURES
    assert not any(s in c for c in X.columns for s in FORBIDDEN_SUBSTRINGS)


def test_feature_sql_does_not_read_outcome_sources():
    from orderpulse.config import PACKAGE_ROOT
    sql = (PACKAGE_ROOT / "sql" / "features" / "01_feature_orders_at_approval.sql").read_text(encoding="utf-8")
    body = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    for token in ["delivered_ts", "carrier_ts", "order_status", "stg_order_reviews", "stg_order_payments", "mart_"]:
        assert token not in body, token


def _features(tables, tmp_path) -> pd.DataFrame:
    cfg = make_cfg(tmp_path, tables)
    ingest(cfg)
    build_warehouse(cfg)
    con = duckdb.connect(str(cfg.warehouse_db), read_only=True)
    df = load_feature_frame(con).set_index("order_id").sort_index()
    con.close()
    return df


def test_future_review_and_delivery_do_not_change_X(tmp_path):
    base = make_tables()
    changed = make_tables()
    o = changed["orders"]
    # move every actual delivery by +40 days and change statuses/carrier dates (future events)
    o["order_delivered_customer_date"] = (pd.to_datetime(o["order_delivered_customer_date"])
                                          + pd.Timedelta(days=40)).dt.strftime("%Y-%m-%d %H:%M:%S")
    o["order_delivered_carrier_date"] = None
    o.loc[o["order_status"] == "delivered", "order_status"] = "shipped"
    r = changed["order_reviews"]
    r["review_score"] = 1
    r["review_creation_date"] = "2018-08-30 00:00:00"
    pay = changed["order_payments"]
    pay["payment_value"] = "1.00"
    pay["payment_type"] = "boleto"
    a = _features(base, tmp_path / "a")
    b = _features(changed, tmp_path / "b")
    pd.testing.assert_frame_equal(a, b)


def test_label_changes_when_delivery_changes_but_features_do_not(tmp_path, built_cfg):
    # sanity check of the previous test: the mutation does change labels
    con = duckdb.connect(str(built_cfg.warehouse_db), read_only=True)
    late = con.execute("SELECT sum(is_late) FROM ml_labels").fetchone()[0]
    con.close()
    assert late > 0
