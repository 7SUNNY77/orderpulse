"""Model-quality metrics. Pure functions over (y, score[, day]) arrays.

Average Precision is sklearn's ``average_precision_score`` (step-wise, not the
trapezoidal area under the PR curve) and is always labelled "AP".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def ranking_metrics(y: np.ndarray, score: np.ndarray, probabilistic: bool) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()) if len(y) else None}
    both = len(np.unique(y)) == 2
    out["ap"] = float(average_precision_score(y, score)) if both else None
    out["roc_auc"] = float(roc_auc_score(y, score)) if both else None
    out["brier"] = float(brier_score_loss(y, score)) if probabilistic and len(y) else None
    out["mean_score"] = float(score.mean()) if probabilistic and len(y) else None
    return out


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float | None) -> dict | None:
    if threshold is None:
        return None
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "flagged": int(pred.sum()), "flagged_share": float(pred.mean()),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_share(score: np.ndarray, share: float) -> float:
    """Smallest threshold that flags ~``share`` of rows (chosen on validation only)."""
    score = np.sort(np.asarray(score, dtype=float))[::-1]
    k = max(1, int(round(share * len(score))))
    return float(score[k - 1])


def topk_overall(y: np.ndarray, score: np.ndarray, ids: np.ndarray, k: int) -> dict:
    """Top-K over the whole period as ONE cohort (ties: expected value, see queue_eval).

    Random expectation for a single cohort is the late rate, so lift = precision / late rate.
    ``ids`` is kept for API compatibility; ranking no longer depends on row order or ids.
    """
    from .queue_eval import _day_h
    y = np.asarray(y).astype(int)
    h, _ = _day_h(np.asarray(score, dtype=float), y, k)
    sel = min(k, len(y))
    base = y.mean() if len(y) else 0.0
    prec = h / sel if sel else None
    return {
        "k": int(k), "selected": int(sel), "found_late": float(h),
        "precision_at_k": prec,
        "recall_at_k": h / int(y.sum()) if y.sum() else None,
        "lift": (prec / base) if (prec is not None and base > 0) else None,
        "random_precision": float(base),
    }


def topk_daily(y: np.ndarray, score: np.ndarray, ids: np.ndarray, day: np.ndarray, k: int) -> dict:
    """Top-K per approval day; formulas and tie rule in ``orderpulse.queue_eval``.

    lift = model micro precision / EXPECTED micro precision of uniform random
    selection of the same k_d per day (sum k_d m_d / n_d / sum k_d) -- not the
    overall late rate, which differs when day sizes and late rates co-vary.
    """
    from .queue_eval import daily_table, summarize
    df = pd.DataFrame({"is_late": np.asarray(y).astype(int), "score": np.asarray(score, dtype=float),
                       "day": np.asarray(day)})
    if df.empty:
        return {"k": k, "days": 0}
    s = summarize(daily_table(df, "score", k), k)
    return {
        "k": int(k), "days": s["model_days"], "days_with_fewer_than_k": s["model_days_n_lt_K"],
        "days_without_late": s["model_days_without_late"],
        "selected": s["model_reviewed"], "found_late": s["model_found_late"], "total_late": s["model_late"],
        "micro_precision": s["model_micro_precision"], "micro_recall": s["model_micro_recall"],
        "macro_precision": s["model_macro_precision"], "macro_recall": s["model_macro_recall"],
        "random_micro_precision": s["random_micro_precision"], "random_micro_recall": s["random_micro_recall"],
        "random_found_late": s["random_found_late"],
        "lift": s["lift_micro_precision"],
        "base_rate": float(df["is_late"].mean()),
        "review_share": s["model_review_share"],
    }


def calibration_bins(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> list[dict]:
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(score, dtype=float)})
    if df.empty:
        return []
    df["bin"] = pd.qcut(df["s"].rank(method="first"), q=min(n_bins, len(df)), labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), mean_score=("s", "mean"), observed=("y", "mean"))
    return [{"bin": int(i), **{k: float(v) for k, v in r.items()}} for i, r in g.iterrows()]


def pr_curve_points(y: np.ndarray, score: np.ndarray, max_points: int = 200) -> list[dict]:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return []
    p, r, _ = precision_recall_curve(y, np.asarray(score, dtype=float))
    idx = np.unique(np.linspace(0, len(p) - 1, min(max_points, len(p))).astype(int))
    return [{"precision": float(p[i]), "recall": float(r[i])} for i in idx]
