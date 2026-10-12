"""Synthetic checks of the daily top-K formulas and the random baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orderpulse.queue_eval import (aggregate, daily_table, paired_bootstrap, random_monte_carlo,
                                   summarize)


def frame(days: dict[str, list[tuple[float, int]]]) -> pd.DataFrame:
    rows = [{"day": d, "score": s, "is_late": y} for d, items in days.items() for s, y in items]
    return pd.DataFrame(rows)


# Three days of different sizes, checked by hand for K = 2:
#   d1: n=4, m=2, top2 = (0.9:1, 0.8:0)            -> h=1,  k=2, precision 0.5, recall 0.5
#   d2: n=1, m=1  (n < K)                           -> h=1,  k=1, precision 1.0, recall 1.0
#   d3: n=3, m=0  (no late orders)                  -> h=0,  k=2, precision 0.0, recall undefined
DAYS = {
    "2018-01-01": [(0.9, 1), (0.8, 0), (0.7, 1), (0.1, 0)],
    "2018-01-02": [(0.5, 1)],
    "2018-01-03": [(0.6, 0), (0.5, 0), (0.4, 0)],
}


def test_manual_micro_macro():
    t = daily_table(frame(DAYS), "score", k=2)
    assert t[["n", "m", "k_d"]].values.tolist() == [[4, 2, 2], [1, 1, 1], [3, 0, 2]]
    assert t["h"].tolist() == [1.0, 1.0, 0.0]
    a = aggregate(t)
    assert a["micro_precision"] == pytest.approx(2 / 5)          # (1+1+0)/(2+1+2)
    assert a["micro_recall"] == pytest.approx(2 / 3)             # (1+1)/(2+1+0)
    assert a["macro_precision"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert a["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)   # day without late orders excluded, not 0
    assert a["days_without_late"] == 1
    assert a["reviewed"] == 5 and a["review_share"] == pytest.approx(5 / 8)
    assert a["reviewed_per_day"] == pytest.approx(5 / 3) and a["found_per_day"] == pytest.approx(2 / 3)


def test_day_with_fewer_orders_than_k_selects_all():
    s = summarize(daily_table(frame(DAYS), "score", k=2), K=2)
    assert s["model_days_n_lt_K"] == 1


def test_analytic_random_expectation():
    t = daily_table(frame(DAYS), "score", k=2)
    assert t["h_random"].tolist() == pytest.approx([2 * 2 / 4, 1 * 1 / 1, 0.0])
    s = summarize(t, K=2)
    # expected micro precision uses the same weights: sum(k m / n) / sum(k)
    assert s["random_micro_precision"] == pytest.approx((1.0 + 1.0 + 0.0) / 5)
    assert s["random_macro_precision"] == pytest.approx((2 / 4 + 1 / 1 + 0 / 3) / 3)
    assert s["random_macro_recall"] == pytest.approx((2 / 4 + 1 / 1) / 2)   # k_d / n_d on days with m_d > 0
    assert s["lift_micro_precision"] == pytest.approx((2 / 5) / (2 / 5))


def test_ties_use_expected_value_and_are_order_independent():
    # all scores equal: top-2 of 4 with 1 late -> expected hits 2 * 1/4 = 0.5, whatever the row order
    d = {"d": [(0.3, 0), (0.3, 1), (0.3, 0), (0.3, 0)]}
    a = daily_table(frame(d), "score", k=2)
    b = daily_table(frame(d).iloc[::-1], "score", k=2)
    assert a["h"].iloc[0] == pytest.approx(0.5) == b["h"].iloc[0]
    assert a["h"].iloc[0] == pytest.approx(a["h_random"].iloc[0])
    assert a["boundary_tie_size"].iloc[0] == 4
    # partial tie: one clear winner (late) + 3 tied for the last slot, 1 of them late
    d = {"d": [(0.9, 1), (0.3, 1), (0.3, 0), (0.3, 0)]}
    assert daily_table(frame(d), "score", k=2)["h"].iloc[0] == pytest.approx(1 + 1 / 3)


def test_constant_score_equals_random_expectation():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"day": rng.integers(0, 5, 300), "is_late": rng.integers(0, 2, 300), "score": 0.1})
    t = daily_table(df, "score", k=7)
    assert t["h"].to_numpy() == pytest.approx(t["h_random"].to_numpy())


def test_monte_carlo_matches_analytic_expectation():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"day": rng.integers(0, 6, 400), "is_late": (rng.uniform(size=400) < 0.2).astype(int),
                       "score": 0.0})
    t = daily_table(df, "score", k=10)
    mc = random_monte_carlo(df, k=10, reps=2000, seed=3)
    assert mc["micro_precision"].mean() == pytest.approx(aggregate(t, "h_random")["micro_precision"], abs=0.01)
    again = random_monte_carlo(df, k=10, reps=50, seed=3)
    assert again["micro_precision"].tolist() == mc["micro_precision"].head(50).tolist()  # fixed seed


def test_paired_bootstrap_uses_identical_days():
    df = frame(DAYS)
    t1 = daily_table(df, "score", k=2)
    t2 = daily_table(df.assign(score=-df["score"]), "score", k=2)
    bs = paired_bootstrap({"a": t1, "b": t2}, ("a", "b"), reps=200, seed=0)
    assert np.allclose(bs["diff_micro_precision"], bs["a_micro_precision"] - bs["b_micro_precision"])
    same = paired_bootstrap({"a": t1, "b": t1}, ("a", "b"), reps=50, seed=0)
    assert (same["diff_micro_precision"] == 0).all()
