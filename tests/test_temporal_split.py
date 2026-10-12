"""Time splits: ordering, no overlap, point-in-time label availability."""

from __future__ import annotations

import pandas as pd
import pytest

from orderpulse.config import load_config
from orderpulse.splits import Period, assign_period, make_splits, periods, validate_periods


def _frame():
    return pd.DataFrame({
        "order_id": ["a", "b", "c", "d", "e", "f"],
        "prediction_time": pd.to_datetime(["2017-03-01", "2017-11-25", "2017-12-15", "2018-01-10",
                                           "2018-04-05", "2018-08-01"]),
        "is_late": [0, 1, 0, 1, 0, 1],
        # b is approved in train but delivered only after the train label cutoff
        "label_available_at": pd.to_datetime(["2017-03-10", "2018-01-05", "2017-12-20", "2018-01-20",
                                              "2018-04-15", "2018-08-10"]),
        "ml_eligible": [True] * 6,
        "ml_exclusion_reason": [None] * 6,
    })


def test_project_config_periods_are_valid_and_ordered():
    cfg = load_config()
    ps = periods(cfg.model)
    validate_periods(ps, cfg.data["data_cutoff"])
    for a, b in zip(ps, ps[1:]):
        assert a.end <= b.start
        assert a.label_available_before <= b.start


def test_overlap_is_rejected():
    ps = [Period("train", pd.Timestamp("2017-01-01"), pd.Timestamp("2017-06-01"), pd.Timestamp("2017-06-01")),
          Period("validation", pd.Timestamp("2017-05-01"), pd.Timestamp("2017-08-01"), pd.Timestamp("2017-08-01"))]
    with pytest.raises(ValueError):
        validate_periods(ps, "2018-09-01")


def test_lookahead_labels_are_rejected():
    ps = [Period("train", pd.Timestamp("2017-01-01"), pd.Timestamp("2017-06-01"), pd.Timestamp("2017-07-15")),
          Period("validation", pd.Timestamp("2017-07-01"), pd.Timestamp("2017-08-01"), pd.Timestamp("2017-08-01"))]
    with pytest.raises(ValueError):
        validate_periods(ps, "2018-09-01")


def test_train_labels_must_be_available_before_cutoff():
    cfg = load_config()
    ds, summary = make_splits(_frame(), periods(cfg.model))
    train = ds["train"]
    assert set(train["order_id"]) == {"a"}  # b: label arrives after 2018-01-01; c: embargo month
    assert summary["train"]["excluded_label_not_yet_available"] == 1
    for name, p in zip(["train", "validation", "test"], periods(cfg.model)):
        d = ds[name]
        assert (d["label_available_at"] < p.label_available_before).all()
        assert ((d["prediction_time"] >= p.start) & (d["prediction_time"] < p.end)).all()


def test_split_sets_are_disjoint_and_ordered():
    cfg = load_config()
    ds, _ = make_splits(_frame(), periods(cfg.model))
    ids = [set(ds[n]["order_id"]) for n in ("train", "validation", "test")]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])
    assert ds["train"]["prediction_time"].max() < ds["validation"]["prediction_time"].min()
    assert ds["validation"]["prediction_time"].max() < ds["test"]["prediction_time"].min()


def test_assign_period_labels_gaps():
    cfg = load_config()
    lab = assign_period(_frame()["prediction_time"], periods(cfg.model)).tolist()
    assert lab == ["train", "train", "embargo", "validation", "test", "after_test"]


def test_assert_labels_available_rejects_future_labels():
    from orderpulse.splits import LabelAvailabilityError, assert_labels_available
    df = _frame()
    ok = df[df["label_available_at"] < pd.Timestamp("2018-01-01")]
    assert assert_labels_available(ok, "2018-01-01", "fit")["rows"] == len(ok)
    with pytest.raises(LabelAvailabilityError):
        assert_labels_available(df.head(3), "2018-01-01", "fit")  # 'b' delivered 2018-01-05
    with pytest.raises(LabelAvailabilityError):
        assert_labels_available(df.assign(label_available_at=pd.NaT).head(1), "2018-01-01", "fit")


def test_reliability_windows_are_sequential_and_before_original_test():
    import yaml
    from orderpulse.config import PACKAGE_ROOT
    rc = yaml.safe_load((PACKAGE_ROOT / "configs" / "reliability.yaml").read_text(encoding="utf-8"))
    test_start = pd.Timestamp(load_config().model["splits"]["test"]["start"])
    for name, w in rc["windows"].items():
        fit, es, ev = w["fit"], w["es"], w["eval"]
        ts = pd.Timestamp
        assert ts(fit["end"]) <= ts(fit["label_cutoff"]) <= ts(es["start"]), name
        assert ts(es["end"]) <= ts(es["label_cutoff"]) <= ts(ev["start"]), name
        assert ts(ev["end"]) <= test_start, name
    c = rc["calibration"]
    w = rc["windows"][c["model_window"]]
    assert pd.Timestamp(w["es"]["label_cutoff"]) <= pd.Timestamp(c["calib"]["start"])
    assert pd.Timestamp(c["calib"]["label_cutoff"]) <= pd.Timestamp(c["check"]["start"])
    assert pd.Timestamp(c["check"]["end"]) <= test_start
