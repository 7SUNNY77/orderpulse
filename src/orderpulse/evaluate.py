"""Evaluate all scorers on identical validation and test sets.

Thresholds and K come from validation (model_metadata.json) or fixed product
constraints (configs/model.yaml); nothing is selected on test here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Config
from .metrics import (
    calibration_bins,
    pr_curve_points,
    ranking_metrics,
    threshold_metrics,
    topk_daily,
    topk_overall,
)
from .models import MODEL_NAMES, load_models
from .splits import make_splits, periods
from .train import load_ml_frame
from .utils import LOG, timed, write_json_atomic


def _subgroups(df: pd.DataFrame) -> dict[str, pd.Series]:
    pdays = pd.to_numeric(df["promised_days"], errors="coerce")
    return {
        "multi_seller": df["n_sellers"] > 1,
        "single_seller": df["n_sellers"] == 1,
        "interstate": df["is_interstate"] == 1,
        "same_state": df["is_interstate"] == 0,
        "promised_lt_14d": pdays < 14,
        "promised_14_28d": (pdays >= 14) & (pdays < 28),
        "promised_ge_28d": pdays >= 28,
        "geo_missing": df["geo_missing"] == 1,
    }


def evaluate_split(df: pd.DataFrame, models: dict, meta: dict, policy: dict, detailed: bool) -> dict:
    y = df["is_late"].to_numpy()
    ids = df["order_id"].to_numpy()
    day = df["prediction_time"].dt.date.to_numpy()
    out = {}
    for name in MODEL_NAMES:
        m = models[name]
        s = m.score(df)
        r = ranking_metrics(y, s, m.probabilistic)
        r["version"] = meta["models"][name]["version"]
        r["threshold_policy"] = threshold_metrics(y, s, meta["models"][name]["review_threshold"])
        r["topk_overall"] = [topk_overall(y, s, ids, k) for k in policy["k_list"]]
        r["topk_daily"] = [topk_daily(y, s, ids, day, k) for k in policy["daily_k_list"]]
        if detailed:
            r["pr_curve"] = pr_curve_points(y, s)
            r["calibration"] = calibration_bins(y, s) if m.probabilistic else []
            r["subgroups"] = {}
            for g, mask in _subgroups(df).items():
                mask = mask.fillna(False).to_numpy(bool)
                if mask.sum() >= 50:
                    sg = ranking_metrics(y[mask], s[mask], m.probabilistic)
                    r["subgroups"][g] = {k: sg[k] for k in ("n", "positives", "base_rate", "ap", "roc_auc")}
        out[name] = r
    return out


def daily_series(df: pd.DataFrame, models: dict, k: int) -> list[dict]:
    """Per-day precision@K of CatBoost vs the rule and the late rate (for the quality page)."""
    rows = []
    d = df.assign(day=df["prediction_time"].dt.date)
    scores = {n: models[n].score(d) for n in ("catboost", "rule_short_promise", "logreg")}
    for n, s in scores.items():
        d[f"s_{n}"] = s
    for day, g in d.groupby("day"):
        row = {"day": str(day), "n": int(len(g)), "late": int(g["is_late"].sum())}
        for n in scores:
            top = g.sort_values([f"s_{n}", "order_id"], ascending=[False, True]).head(k)
            row[f"precision_{n}"] = float(top["is_late"].mean())
        rows.append(row)
    return rows


def evaluate(cfg: Config) -> dict:
    with timed("evaluate"):
        meta = json.loads((cfg.artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
        models = load_models(cfg.artifacts_dir)
        frame, fp = load_ml_frame(cfg)
        if fp != meta["data_fingerprint"]:
            raise RuntimeError("warehouse data fingerprint differs from the trained model: re-run train")
        ds, summary = make_splits(frame, periods(cfg.model))
        policy = cfg.model["policy"]
        result = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_fingerprint": fp,
            "feature_version": meta["feature_version"],
            "model_versions": {n: meta["models"][n]["version"] for n in MODEL_NAMES},
            "splits": summary,
            "policy": {**meta["policy"], "k_list": policy["k_list"], "daily_k_list": policy["daily_k_list"]},
            "notes": [
                "AP = sklearn average_precision_score (not trapezoidal PR-AUC).",
                "Validation was used for CatBoost early stopping, grid choice and thresholds: "
                "validation metrics are optimistic; test is the honest estimate.",
                "rule_short_promise is a ranking score, not a probability: Brier/calibration not applicable.",
                "Population: historically delivered orders only; the model does not predict cancellation or loss.",
            ],
            "validation": evaluate_split(ds["validation"], models, meta, policy, detailed=True),
            "test": evaluate_split(ds["test"], models, meta, policy, detailed=True),
            "test_daily": daily_series(ds["test"], models, int(policy["daily_k"])),
        }
        cb = models["catboost"].model
        imp = cb.get_feature_importance(type="PredictionValuesChange")
        result["catboost_global_importance"] = sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(cb.feature_names_, imp)],
            key=lambda r: -r["importance"])
        write_json_atomic(result, cfg.artifacts_dir / "metrics.json")
        for split in ("validation", "test"):
            for n in MODEL_NAMES:
                r = result[split][n]
                d = next(x for x in r["topk_daily"] if x["k"] == policy["daily_k"])
                LOG.info("%s %-18s AP=%s ROC=%s Brier=%s P@%d/day=%.3f R@%d/day=%s",
                         split, n, _f(r["ap"]), _f(r["roc_auc"]), _f(r["brier"]), policy["daily_k"],
                         d["micro_precision"], policy["daily_k"], _f(d["micro_recall"]))
        return result


def _f(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"
