"""Batch scoring of every approved order (historical simulation).

Output: artifacts/predictions.parquet with one row per
(order_id, prediction_time, model_version). Scores are computed once here; the
UI only reads them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from .config import Config
from .explanations import top_factor_strings
from .features import load_feature_frame
from .models import MODEL_NAMES, load_models
from .splits import assign_period, periods
from .utils import LOG, atomic_replace, remove_if_exists, timed, tmp_path_for

PREDICTION_COLUMNS = ["order_id", "prediction_time", "risk_score", "model_name", "model_version",
                      "feature_version", "scored_at", "split", "risk_rank_in_day", "risk_pct_in_day",
                      "top_factors", "data_fingerprint"]


def predictions_path(cfg: Config):
    return cfg.artifacts_dir / "predictions.parquet"


def score_frame(frame: pd.DataFrame, models: dict, meta: dict, cfg: Config, scored_at: str,
                fingerprint: str, with_factors: bool = True) -> pd.DataFrame:
    split = assign_period(frame["prediction_time"], periods(cfg.model))
    day = frame["prediction_time"].dt.date
    parts = []
    for name in MODEL_NAMES:
        m = models[name]
        s = m.score(frame)
        part = pd.DataFrame({
            "order_id": frame["order_id"].to_numpy(),
            "prediction_time": frame["prediction_time"].to_numpy(),
            "risk_score": s.astype(float),
            "model_name": name,
            "model_version": meta["models"][name]["version"],
            "feature_version": meta["feature_version"],
            "scored_at": scored_at,
            "split": split.to_numpy(),
            "_day": day.to_numpy(),
        })
        part = part.sort_values(["_day", "risk_score", "order_id"], ascending=[True, False, True])
        part["risk_rank_in_day"] = part.groupby("_day").cumcount() + 1
        part["risk_pct_in_day"] = part.groupby("_day")["risk_score"].rank(pct=True, method="max")
        part["top_factors"] = ""
        if name == "catboost" and with_factors:
            contrib, _ = m.shap(frame)
            factors = pd.Series(top_factor_strings(contrib), index=frame.index)
            part["top_factors"] = factors.loc[part.index].to_numpy()
        part["data_fingerprint"] = fingerprint
        parts.append(part.drop(columns="_day"))
    out = pd.concat(parts, ignore_index=True)[PREDICTION_COLUMNS]
    dup = out.duplicated(["order_id", "prediction_time", "model_version"]).sum()
    if dup:
        raise RuntimeError(f"{dup} duplicated (order_id, prediction_time, model_version) rows")
    return out


def predict(cfg: Config) -> pd.DataFrame:
    with timed("predict"):
        meta = json.loads((cfg.artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
        models = load_models(cfg.artifacts_dir)
        con = duckdb.connect(str(cfg.warehouse_db), read_only=True)
        try:
            frame = load_feature_frame(con)
            fp = con.execute("SELECT value FROM meta_build WHERE key = 'data_fingerprint'").fetchone()[0]
        finally:
            con.close()
        if fp != meta["data_fingerprint"]:
            raise RuntimeError("warehouse data fingerprint differs from the trained model: re-run train")
        scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out = score_frame(frame, models, meta, cfg, scored_at, fp)
        target = predictions_path(cfg)
        tmp = tmp_path_for(target)
        remove_if_exists(tmp)
        out.to_parquet(tmp, index=False)
        atomic_replace(tmp, target)
        LOG.info("predictions: %d rows (%d orders x %d models) -> %s",
                 len(out), out["order_id"].nunique(), out["model_version"].nunique(), target)
        return out
