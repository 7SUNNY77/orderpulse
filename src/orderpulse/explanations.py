"""Local SHAP explanations for CatBoost and rule-based review hints.

SHAP values are additive contributions to the model's RAW score (log-odds):
    raw = expected_value + sum(contributions);  risk_score = sigmoid(raw)
Individual contributions are reported in log-odds, never converted to
percentage points. They describe the model, not causes of delays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_LABELS, FEATURES


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _fmt_value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "нет данных"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.1f}".replace(",", " ")
    return str(v)


def local_explanation(model, row: pd.DataFrame) -> dict:
    """Full SHAP table for one order (row = one-row feature frame)."""
    contrib, expected = model.shap(row)
    c, base = contrib[0], float(expected[0])
    table = pd.DataFrame({
        "feature": FEATURES,
        "Признак": [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "Значение": [_fmt_value(row.iloc[0][f]) for f in FEATURES],
        "Вклад (log-odds)": c,
    })
    table["Направление"] = np.where(table["Вклад (log-odds)"] > 0, "повышает", "снижает")
    table = table.reindex(table["Вклад (log-odds)"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    raw = base + c.sum()
    return {"expected_raw": base, "expected_score": float(sigmoid(base)),
            "raw": float(raw), "score": float(sigmoid(raw)), "table": table}


def top_factor_strings(contrib: np.ndarray, n: int = 3) -> list[str]:
    """Names of the n features that increased each row's prediction the most."""
    out = []
    for row in contrib:
        idx = np.argsort(-row)[:n]
        names = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in idx if row[i] > 0]
        out.append("; ".join(names))
    return out


def review_hints(features: dict, risk_percentile: float | None) -> list[str]:
    """«Подсказки для проверки»: simple documented rules, not proven optimal actions.

    Rules (see reports/decisions.md):
      * n_sellers > 1                      -> check readiness of every part of the order
      * promised_days < 10                 -> check that the promised date is feasible
      * risk in the top 5% of the day      -> prioritise checking preparation/hand-over
      * geo_missing or unknown attributes  -> check the order/product card data
    """
    hints = []
    if (features.get("n_sellers") or 0) > 1:
        hints.append("Несколько продавцов: проверить готовность всех частей заказа.")
    pdays = features.get("promised_days")
    if pdays is not None and not np.isnan(pdays) and pdays < 10:
        hints.append("Короткий обещанный срок (< 10 дней): проверить исполнимость срока.")
    if risk_percentile is not None and risk_percentile >= 0.95:
        hints.append("Высокий риск (top 5% дня): приоритетно проверить подготовку и передачу перевозчику.")
    if (features.get("geo_missing") or 0) == 1 or (features.get("share_unknown_attrs") or 0) > 0:
        hints.append("Недостаток данных (нет координат или характеристик товара): проверить карточку заказа.")
    return hints


# ----------------------------------------------------------------------------- Logistic Regression

def _original_feature(transformed: str) -> str:
    """Map a ColumnTransformer output name back to the allow-listed feature.

    num__<f>                     scaled (median-imputed) numeric value
    num__missingindicator_<f>    missing-value indicator added by SimpleImputer
    cat__<f>_<category>          one-hot component (incl. '<f>_infrequent_sklearn')
    """
    prefix, _, rest = transformed.partition("__")
    if prefix == "num":
        rest = rest.removeprefix("missingindicator_")
        if rest in FEATURES:
            return rest
    if prefix == "cat":
        for f in sorted(FEATURES, key=len, reverse=True):
            if rest.startswith(f + "_"):
                return f
    raise ValueError(f"cannot map transformed feature {transformed!r}")


def linear_contributions(pipeline, frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-row contributions of transformed features to the LogReg decision function.

    decision_function(x) = intercept + sum_j coef_j * z_j(x), where z = pipeline['pre'].transform(x)
    (the exact preprocessing used in training: median imputation + missing indicators +
    standardisation for numerics, one-hot with an 'infrequent' bucket for categoricals;
    unknown categories map to the infrequent bucket or to all-zero).
    Returns (contributions [n_rows x n_transformed], intercept). Positive values push
    towards class 1 = late delivery (clf.classes_ == [0, 1]).
    """
    from .features import to_X
    pre, clf = pipeline["pre"], pipeline["clf"]
    if list(clf.classes_) != [0, 1]:
        raise ValueError(f"unexpected classes {clf.classes_}")
    z = pre.transform(to_X(frame))
    if hasattr(z, "toarray"):
        z = z.toarray()
    contrib = np.asarray(z, dtype=float) * clf.coef_[0]
    return pd.DataFrame(contrib, columns=pre.get_feature_names_out(), index=frame.index), float(clf.intercept_[0])


def aggregate_to_features(contrib: pd.DataFrame) -> pd.DataFrame:
    """Sum transformed-column contributions per original feature (sum is preserved)."""
    groups = [_original_feature(c) for c in contrib.columns]
    agg = contrib.T.groupby(groups, sort=False).sum().T
    return agg.reindex(columns=[f for f in FEATURES if f in agg.columns])


def logreg_explanation(pipeline, row: pd.DataFrame) -> dict:
    """Local linear explanation of ONE order for the UI (logit scale, aggregated per feature)."""
    contrib, intercept = linear_contributions(pipeline, row)
    agg = aggregate_to_features(contrib).iloc[0]
    table = pd.DataFrame({
        "feature": agg.index,
        "Признак": [FEATURE_LABELS.get(f, f) for f in agg.index],
        "Значение": [_fmt_value(row.iloc[0][f]) for f in agg.index],
        "Вклад (логит)": agg.to_numpy(),
    })
    table["Направление"] = np.where(table["Вклад (логит)"] > 0, "повышает", "снижает")
    table = table.reindex(table["Вклад (логит)"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    raw = intercept + float(contrib.iloc[0].sum())
    return {"intercept": intercept, "raw": raw, "score": float(sigmoid(raw)), "table": table,
            "transformed": contrib.iloc[0]}


def logreg_top_factors(pipeline, frame: pd.DataFrame, n: int = 3) -> list[str]:
    """Top-n features (aggregated) that increased each row's linear score."""
    agg = aggregate_to_features(linear_contributions(pipeline, frame)[0])
    out = []
    for _, r in agg.iterrows():
        top = r[r > 0].sort_values(ascending=False).head(n)
        out.append("; ".join(FEATURE_LABELS.get(f, f) for f in top.index))
    return out
