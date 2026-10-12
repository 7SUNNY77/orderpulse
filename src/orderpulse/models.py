"""Uniform wrappers around the four scorers so that all of them are evaluated
and served through the same code path."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .features import CATEGORICAL_FEATURES, FEATURES, to_X

MODEL_NAMES = ("constant", "rule_short_promise", "logreg", "catboost")
MODEL_TITLES = {
    "constant": "Константа (доля задержек train)",
    "rule_short_promise": "Правило: короткий обещанный срок",
    "logreg": "Logistic Regression (основная)",
    "catboost": "CatBoost (сравнение)",
}


class ConstantModel:
    name = "constant"
    probabilistic = True

    def __init__(self, rate: float):
        self.rate = float(rate)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.rate)


class RuleModel:
    """Operational baseline: the shorter the promised term, the higher the risk.
    Score = -promised_days (a ranking score, not a probability)."""

    name = "rule_short_promise"
    probabilistic = False

    def __init__(self, fill_value: float):
        self.fill_value = float(fill_value)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return -pd.to_numeric(df["promised_days"], errors="coerce").fillna(self.fill_value).to_numpy(float)


class SklearnModel:
    name = "logreg"
    probabilistic = True

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(to_X(df))[:, 1]


class CatBoostModel:
    name = "catboost"
    probabilistic = True

    def __init__(self, model: CatBoostClassifier):
        self.model = model

    def pool(self, df: pd.DataFrame) -> Pool:
        return Pool(to_X(df), cat_features=CATEGORICAL_FEATURES)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.pool(df))[:, 1]

    def shap(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Local SHAP values in raw-score (log-odds) space and the expected value per row."""
        sv = self.model.get_feature_importance(self.pool(df), type="ShapValues")
        return sv[:, :-1], sv[:, -1]


def save_models(models: dict, artifacts_dir: Path) -> dict[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    base = {"constant": {"rate": models["constant"].rate},
            "rule_short_promise": {"fill_value": models["rule_short_promise"].fill_value}}
    (artifacts_dir / "baselines.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    files["constant"] = files["rule_short_promise"] = "baselines.json"
    joblib.dump(models["logreg"].pipeline, artifacts_dir / "logreg.joblib")
    files["logreg"] = "logreg.joblib"
    models["catboost"].model.save_model(str(artifacts_dir / "model.cbm"))
    files["catboost"] = "model.cbm"
    return files


def load_models(artifacts_dir: Path) -> dict:
    base = json.loads((artifacts_dir / "baselines.json").read_text(encoding="utf-8"))
    cb = CatBoostClassifier()
    cb.load_model(str(artifacts_dir / "model.cbm"))
    if list(cb.feature_names_) != FEATURES:
        raise RuntimeError("model.cbm feature list does not match the current feature allow-list")
    return {
        "constant": ConstantModel(base["constant"]["rate"]),
        "rule_short_promise": RuleModel(base["rule_short_promise"]["fill_value"]),
        "logreg": SklearnModel(joblib.load(artifacts_dir / "logreg.joblib")),
        "catboost": CatBoostModel(cb),
    }
