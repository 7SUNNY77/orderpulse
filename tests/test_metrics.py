"""Business KPIs (denominators, missing values) and model-quality metrics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from orderpulse import queries
from orderpulse.metrics import ranking_metrics, threshold_for_share, topk_daily, topk_overall
from orderpulse.queries import Filters


@pytest.fixture()
def qcon(built_cfg):
    con = queries.connect(built_cfg.warehouse_db)
    yield con
    con.close()


def test_kpi_denominators(qcon, wh):
    k = queries.kpis(qcon, Filters())
    n_orders, n_delivered, n_late = wh.execute(
        "SELECT count(*), count(*) FILTER (WHERE is_delivered), count(*) FILTER (WHERE is_late) FROM mart_orders").fetchone()
    assert k["n_orders"] == n_orders
    assert k["n_delivered"] == n_delivered
    assert k["late_rate"] == pytest.approx(n_late / n_delivered)
    # orders without items are excluded from the average order value denominator
    assert k["n_with_items"] == n_orders - 1
    assert k["avg_order_value"] == pytest.approx(k["items_value"] / k["n_with_items"])
    # undelivered orders are not in the late-rate denominator
    assert k["n_delivered"] < k["n_orders"]
    assert k["review_coverage"] == pytest.approx(k["n_reviewed"] / k["n_orders"])


def test_kpis_on_empty_selection_are_missing_not_zero(qcon):
    k = queries.kpis(qcon, Filters(date_from=date(2030, 1, 1)))
    assert k["n_orders"] == 0
    assert k["late_rate"] is None and k["avg_order_value"] is None and k["median_delivery_days"] is None


def test_seller_filter_and_multi_seller_counting(qcon):
    k = queries.kpis(qcon, Filters(sellers=("s00", "s01")))
    seg = queries.segments(qcon, Filters(), "seller")
    s = seg.set_index("segment")
    # o_multi has both s00 and s01: one order in KPIs, counted for each seller in segments
    assert k["n_orders"] < s.loc["s00", "orders"] + s.loc["s01", "orders"]


def test_segment_ci_brackets_rate(qcon):
    seg = queries.segments(qcon, Filters(), "customer_state")
    ok = seg.dropna(subset=["late_rate"])
    assert ((ok["ci_low"] <= ok["late_rate"]) & (ok["late_rate"] <= ok["ci_high"])).all()


def test_ranking_metrics_basic():
    y = np.array([0, 0, 1, 1])
    m = ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), probabilistic=True)
    assert m["ap"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)
    one_class = ranking_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), probabilistic=True)
    assert one_class["ap"] is None and one_class["roc_auc"] is None


def test_topk_overall_worked_example():
    # 20 late among the 50 selected -> Precision@50 = 40%
    y = np.array([1] * 20 + [0] * 30 + [1] * 5 + [0] * 45)
    s = np.linspace(1, 0, 100)
    r = topk_overall(y, s, np.arange(100), 50)
    assert r["precision_at_k"] == pytest.approx(0.4)
    assert r["recall_at_k"] == pytest.approx(20 / 25)
    assert r["lift"] == pytest.approx(0.4 / 0.25)


def test_topk_daily_handles_small_days():
    day = np.array(["d1"] * 5 + ["d2"] * 2)
    y = np.array([1, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.1])
    r = topk_daily(y, s, np.arange(7), day, k=3)
    assert r["days_with_fewer_than_k"] == 1
    assert r["selected"] == 3 + 2      # day 2 selects all of its 2 orders
    assert r["found_late"] == 2
    assert r["micro_precision"] == pytest.approx(2 / 5)


def test_threshold_for_share():
    s = np.arange(100, dtype=float)
    t = threshold_for_share(s, 0.1)
    assert (s >= t).sum() == 10
