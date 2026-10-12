"""Queue composition on a hand-made example: several days, two models, filters, ties."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest

from orderpulse import queries
from orderpulse.features import FEATURES

# day 1: a(0.9, RJ) b(0.8, SP) c(0.7, SP) d(0.7, SP)   <- c/d tie
# day 2: e(0.2, SP) f(0.6, RJ)
ROWS = [
    ("a", "2018-04-01 10:00", "RJ", 0.9, 0.1),
    ("b", "2018-04-01 11:00", "SP", 0.8, 0.2),
    ("d", "2018-04-01 12:00", "SP", 0.7, 0.3),
    ("c", "2018-04-01 13:00", "SP", 0.7, 0.9),
    ("e", "2018-04-02 09:00", "SP", 0.2, 0.8),
    ("f", "2018-04-02 15:00", "RJ", 0.6, 0.7),
]


@pytest.fixture()
def qcon():
    con = duckdb.connect()
    feats = pd.DataFrame([{"order_id": r[0], "prediction_time": pd.Timestamp(r[1]), "promised_ts": pd.Timestamp("2018-05-01"),
                           **{f: 1 for f in FEATURES}, "customer_state": r[2], "main_seller_state": "SP",
                           "main_category": "x", "n_sellers": 1} for r in ROWS])
    con.register("feats_df", feats)
    con.execute("CREATE TABLE feature_orders_at_approval AS SELECT * FROM feats_df")
    preds = []
    for mv, idx in (("m_lr", 3), ("m_cb", 4)):
        for r in ROWS:
            preds.append({"order_id": r[0], "prediction_time": pd.Timestamp(r[1]), "risk_score": r[idx],
                          "model_version": mv, "risk_rank_in_day": 0, "top_factors": "", "split": "test"})
    con.register("preds_df", pd.DataFrame(preds))
    con.execute("CREATE VIEW order_predictions AS SELECT * FROM preds_df")
    yield con
    con.close()


def ids(df):
    return df["order_id"].tolist()


def test_top_k_is_taken_per_day(qcon):
    q = queries.risk_queue(qcon, "m_lr", date(2018, 4, 1), date(2018, 4, 2), 2)
    assert ids(q) == ["a", "b", "f", "e"]          # 2 per day -> 4 rows for a 2-day range (> K)
    q1 = queries.risk_queue(qcon, "m_lr", date(2018, 4, 1), date(2018, 4, 2), 1)
    assert ids(q1) == ["a", "f"]


def test_models_rank_independently(qcon):
    q = queries.risk_queue(qcon, "m_cb", date(2018, 4, 1), date(2018, 4, 2), 1)
    assert ids(q) == ["c", "e"]


def test_filters_apply_before_top_k(qcon):
    q = queries.risk_queue(qcon, "m_lr", date(2018, 4, 1), date(2018, 4, 2), 1, ("SP",))
    # day 1: 'a' (RJ) is filtered out first, so SP's best 'b' is selected; day 2: only 'e' is SP
    assert ids(q) == ["b", "e"]


def test_ties_broken_by_order_id_without_outcomes(qcon):
    q = queries.risk_queue(qcon, "m_lr", date(2018, 4, 1), date(2018, 4, 1), 3)
    assert ids(q) == ["a", "b", "c"]                # c and d tie at 0.7 -> 'c' < 'd'
    assert not {"is_late", "outcome", "delivered_ts"} & set(q.columns)


def test_days_with_fewer_orders_than_k(qcon):
    q = queries.risk_queue(qcon, "m_lr", date(2018, 4, 2), date(2018, 4, 2), 5)
    assert ids(q) == ["f", "e"]
