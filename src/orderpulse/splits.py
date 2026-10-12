"""Time-based train/validation/test with point-in-time label availability.

A row belongs to a split if its prediction_time (approval) falls in the split
period. It is USED for fitting/evaluation only if additionally:
  * it is ML-eligible (approved, delivered, promised date, no temporal error, has items);
  * its label became observable (actual delivery) strictly before the split's
    ``label_available_before`` moment.
Rows dropped by the last rule are counted, never silently removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    label_available_before: pd.Timestamp


def periods(model_cfg: dict) -> list[Period]:
    out = []
    for name in SPLIT_NAMES:
        s = model_cfg["splits"][name]
        out.append(Period(name, pd.Timestamp(s["start"]), pd.Timestamp(s["end"]),
                          pd.Timestamp(s["label_available_before"])))
    return out


def validate_periods(ps: list[Period], data_cutoff: str | pd.Timestamp) -> None:
    cutoff = pd.Timestamp(data_cutoff)
    for p in ps:
        if not p.start < p.end:
            raise ValueError(f"{p.name}: start must be before end")
        if p.label_available_before < p.end:
            raise ValueError(f"{p.name}: labels must be collected at/after the period end")
        if p.label_available_before > cutoff:
            raise ValueError(f"{p.name}: label_available_before is after data_cutoff {cutoff.date()}")
    for a, b in zip(ps, ps[1:]):
        if not a.end <= b.start:
            raise ValueError(f"{a.name} and {b.name} overlap")
        # the model trained on `a` is selected/used at b.start: its labels must exist by then
        if a.label_available_before > b.start:
            raise ValueError(f"{a.name} labels are collected after {b.name} starts (look-ahead)")


def assign_period(prediction_time: pd.Series, ps: list[Period]) -> pd.Series:
    out = pd.Series("outside", index=prediction_time.index, dtype="object")
    for p in ps:
        out[(prediction_time >= p.start) & (prediction_time < p.end)] = p.name
    first, last = ps[0], ps[-1]
    out[(out == "outside") & (prediction_time < first.start)] = "before_train"
    out[(out == "outside") & (prediction_time >= last.end)] = "after_test"
    out[out == "outside"] = "embargo"
    return out


def make_splits(frame: pd.DataFrame, ps: list[Period]) -> tuple[dict[str, pd.DataFrame], dict]:
    """frame: features + labels (order_id, prediction_time, is_late, label_available_at, ml_eligible, ...)."""
    period = assign_period(frame["prediction_time"], ps)
    datasets, summary = {}, {}
    for p in ps:
        in_period = frame[period == p.name]
        eligible = in_period[in_period["ml_eligible"]]
        available = eligible["label_available_at"] < p.label_available_before
        used = eligible[available].copy()
        used["is_late"] = used["is_late"].astype(int)
        datasets[p.name] = used.sort_values(["prediction_time", "order_id"]).reset_index(drop=True)
        summary[p.name] = {
            "start": str(p.start.date()), "end_exclusive": str(p.end.date()),
            "label_available_before": str(p.label_available_before.date()),
            "approved_in_period": int(len(in_period)),
            "excluded_by_reason": {str(k): int(v) for k, v in
                                   in_period["ml_exclusion_reason"].value_counts(dropna=True).items()},
            "eligible": int(len(eligible)),
            "excluded_label_not_yet_available": int((~available).sum()),
            "used": int(len(used)),
            "positives": int(used["is_late"].sum()),
            "late_rate": float(used["is_late"].mean()) if len(used) else None,
            "max_label_available_at": str(used["label_available_at"].max()) if len(used) else None,
        }
    return datasets, summary


class LabelAvailabilityError(RuntimeError):
    """A label used for fitting/selection/calibration was not observable at its cutoff."""


def assert_labels_available(df: pd.DataFrame, cutoff, purpose: str) -> dict:
    """Every row must have an observed label strictly before ``cutoff`` and a
    prediction_time before it. ``label_available_at`` is the actual delivery
    timestamp: for a delivered order the lateness label is known at delivery."""
    cutoff = pd.Timestamp(cutoff)
    missing = int(df["label_available_at"].isna().sum())
    late_labels = int((df["label_available_at"] >= cutoff).sum())
    future_pred = int((df["prediction_time"] >= cutoff).sum())
    if missing or late_labels or future_pred:
        raise LabelAvailabilityError(
            f"{purpose}: {late_labels} labels observed at/after {cutoff}, {missing} without label time, "
            f"{future_pred} prediction_time at/after cutoff")
    return {"purpose": purpose, "cutoff": str(cutoff), "rows": int(len(df)),
            "max_label_available_at": str(df["label_available_at"].max()) if len(df) else None,
            "max_prediction_time": str(df["prediction_time"].max()) if len(df) else None}
