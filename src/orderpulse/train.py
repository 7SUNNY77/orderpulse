"""Fit the constant, rule, logistic-regression and CatBoost scorers.

Only train is used for fitting; validation is used for early stopping, for the
small CatBoost grid and for the review threshold. Test is not touched here.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import __version__
from .config import Config
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_feature_frame, load_labels, to_X
from .metrics import ranking_metrics, threshold_for_share
from .models import MODEL_NAMES, CatBoostModel, ConstantModel, RuleModel, SklearnModel, save_models
from .splits import assert_labels_available, make_splits, periods, validate_periods
from .utils import LOG, short_hash, timed, write_json_atomic


def load_ml_frame(cfg: Config) -> tuple[pd.DataFrame, str]:
    con = duckdb.connect(str(cfg.warehouse_db), read_only=True)
    try:
        feats = load_feature_frame(con)
        labels = load_labels(con)
        fp = con.execute("SELECT value FROM meta_build WHERE key = 'data_fingerprint'").fetchone()[0]
    finally:
        con.close()
    frame = feats.merge(labels, on="order_id", how="left", validate="one_to_one")
    frame["ml_eligible"] = frame["ml_eligible"].fillna(False).astype(bool)
    return frame, fp


def review_share(validation: pd.DataFrame, daily_k: int) -> float:
    """Share of orders a manager can review: daily capacity / mean daily volume (validation)."""
    per_day = validation.groupby(validation["prediction_time"].dt.date).size()
    return float(min(1.0, daily_k / per_day.mean())) if len(per_day) else 0.0


def fit_logreg(train: pd.DataFrame, cfg: dict, seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], random_state=seed))])
    pipe.fit(to_X(train), train["is_late"].to_numpy())
    return pipe


def fit_catboost(train: pd.DataFrame, valid: pd.DataFrame, cfg: dict, seed: int) -> tuple[CatBoostClassifier, list[dict]]:
    tr = Pool(to_X(train), train["is_late"].to_numpy(), cat_features=CATEGORICAL_FEATURES)
    va = Pool(to_X(valid), valid["is_late"].to_numpy(), cat_features=CATEGORICAL_FEATURES)
    results, best, best_ap = [], None, -1.0
    for params in cfg["grid"]:
        model = CatBoostClassifier(
            iterations=cfg["iterations"], loss_function=cfg["loss_function"], eval_metric=cfg["eval_metric"],
            early_stopping_rounds=cfg["early_stopping_rounds"], random_seed=seed, verbose=False,
            allow_writing_files=False, thread_count=-1, **params)
        model.fit(tr, eval_set=va, use_best_model=True)
        score = model.predict_proba(va)[:, 1]
        m = ranking_metrics(valid["is_late"].to_numpy(), score, probabilistic=True)
        results.append({"params": params, "best_iteration": int(model.get_best_iteration()),
                        "validation_ap": m["ap"], "validation_roc_auc": m["roc_auc"],
                        "validation_brier": m["brier"]})
        LOG.info("catboost %s: best_iter=%d val AP=%.4f", params, model.get_best_iteration(), m["ap"])
        if m["ap"] is not None and m["ap"] > best_ap:
            best, best_ap = model, m["ap"]
    return best, results


def train(cfg: Config) -> dict:
    with timed("train"):
        mc = cfg.model
        seed = int(mc["seed"])
        np.random.seed(seed)
        ps = periods(mc)
        validate_periods(ps, cfg.data["data_cutoff"])
        frame, fp = load_ml_frame(cfg)
        ds, summary = make_splits(frame, ps)
        tr, va = ds["train"], ds["validation"]
        for name in ("train", "validation"):
            if summary[name]["used"] == 0 or summary[name]["positives"] == 0:
                raise RuntimeError(f"{name} split has no rows or no positives: {summary[name]}")
        LOG.info("splits: %s", {k: (v["used"], v["positives"]) for k, v in summary.items()})
        # Code-level guard: fitting labels known by the train cutoff; early-stopping /
        # grid / threshold labels known by the validation cutoff (= moment the model is
        # frozen and used, i.e. the start of test).
        p = {x.name: x for x in ps}
        label_checks = [
            assert_labels_available(tr, p["train"].label_available_before, "fit (train)"),
            assert_labels_available(va, p["validation"].label_available_before,
                                    "early stopping, grid, threshold (validation)"),
        ]

        models = {
            "constant": ConstantModel(tr["is_late"].mean()),
            "rule_short_promise": RuleModel(float(tr["promised_days"].median())),
        }
        with timed("fit:logreg"):
            models["logreg"] = SklearnModel(fit_logreg(tr, mc["logreg"], seed))
        with timed("fit:catboost"):
            cb, grid_results = fit_catboost(tr, va, mc["catboost"], seed)
            models["catboost"] = CatBoostModel(cb)

        # Review policy chosen on validation only.
        share = review_share(va, int(mc["policy"]["daily_k"]))
        thresholds = {}
        for name, m in models.items():
            thresholds[name] = None if name == "constant" else threshold_for_share(m.score(va), share)

        files = save_models(models, cfg.artifacts_dir)
        trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fv = mc["feature_version"]
        versions = {
            name: f"{name}-{fv}-" + short_hash({"data": fp, "splits": mc["splits"], "features": FEATURES,
                                                "seed": seed, "model": name,
                                                "params": mc.get(name if name != "rule_short_promise" else "rule_baseline"),
                                                "selected": grid_results if name == "catboost" else None}, 8)
            for name in MODEL_NAMES
        }
        best_params = {k: v for k, v in cb.get_params().items() if k in ("depth", "learning_rate", "l2_leaf_reg")}
        metadata = {
            "orderpulse_version": __version__,
            "trained_at": trained_at,
            "data_fingerprint": fp,
            "data_cutoff": str(cfg.data["data_cutoff"]),
            "feature_version": fv,
            "features": FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "prediction_time": "order_approved_at",
            "target": "date(order_delivered_customer_date) > date(order_estimated_delivery_date)",
            "population": "approved, delivered (status=delivered), with promised date, no critical temporal "
                          "error, with items; label observable before the split's label_available_before",
            "seed": seed,
            "splits": summary,
            "label_availability_checks": label_checks,
            "model_frozen_at": str(p["validation"].label_available_before.date()),
            "models": {
                name: {"version": versions[name], "file": files[name],
                       "probabilistic": models[name].probabilistic,
                       "review_threshold": thresholds[name]}
                for name in MODEL_NAMES
            },
            "catboost": {"grid": grid_results, "selected_params": best_params,
                         "best_iteration": int(cb.get_best_iteration()),
                         "selection": "max Average Precision on validation"},
            "logreg_params": mc["logreg"],
            "policy": {
                "daily_k": int(mc["policy"]["daily_k"]),
                "review_share": share,
                "threshold_rule": "per-model score threshold that flags `review_share` of VALIDATION orders; "
                                  "review_share = daily_k / mean daily validation volume",
            },
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
        }
        write_json_atomic(metadata, cfg.artifacts_dir / "model_metadata.json")
        LOG.info("models saved: %s", versions)
        return metadata
