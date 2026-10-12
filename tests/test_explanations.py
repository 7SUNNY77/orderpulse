"""Local linear explanations of the Logistic Regression model."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from orderpulse.explanations import (aggregate_to_features, linear_contributions, logreg_explanation,
                                     logreg_top_factors, sigmoid)
from orderpulse.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_feature_frame, to_X
from orderpulse.models import load_models
from orderpulse.predict import predictions_path


@pytest.fixture(scope="module")
def lr_setup(pipeline_cfg):
    models = load_models(pipeline_cfg.artifacts_dir)
    con = duckdb.connect(str(pipeline_cfg.warehouse_db), read_only=True)
    frame = load_feature_frame(con)
    con.close()
    return models["logreg"].pipeline, frame


def test_intercept_plus_contributions_equals_decision_function(lr_setup):
    pipe, frame = lr_setup
    contrib, intercept = linear_contributions(pipe, frame)
    np.testing.assert_allclose(intercept + contrib.sum(axis=1).to_numpy(), pipe.decision_function(to_X(frame)),
                               rtol=0, atol=1e-10)


def test_aggregation_to_original_features_preserves_sum(lr_setup):
    pipe, frame = lr_setup
    contrib, _ = linear_contributions(pipe, frame)
    agg = aggregate_to_features(contrib)
    assert list(agg.columns) == FEATURES
    np.testing.assert_allclose(agg.sum(axis=1).to_numpy(), contrib.sum(axis=1).to_numpy(), atol=1e-10)
    # every transformed column belongs to exactly one original feature (incl. missing indicators, one-hot)
    assert contrib.shape[1] == len(pipe["pre"].get_feature_names_out())


def test_unknown_category_is_handled_like_the_model(lr_setup):
    pipe, frame = lr_setup
    row = frame.head(1).copy()
    for c in CATEGORICAL_FEATURES:
        row[c] = "category_never_seen_in_training"
    contrib, intercept = linear_contributions(pipe, row)
    np.testing.assert_allclose(intercept + contrib.sum(axis=1).to_numpy(), pipe.decision_function(to_X(row)), atol=1e-10)
    agg = aggregate_to_features(contrib).iloc[0]
    enc = pipe["pre"].named_transformers_["cat"]
    for i, c in enumerate(CATEGORICAL_FEATURES):
        cols = [n for n in contrib.columns if n.startswith(f"cat__{c}_")]
        expected = contrib.iloc[0][cols].sum()
        assert agg[c] == pytest.approx(expected)
        if enc.infrequent_categories_[i] is None:   # no infrequent bucket -> all-zero encoding -> 0 contribution
            assert agg[c] == pytest.approx(0.0)


def test_missing_values_are_imputed_as_in_training(lr_setup):
    pipe, frame = lr_setup
    row = frame.head(1).copy()
    row["dist_mean_km"] = np.nan
    contrib, intercept = linear_contributions(pipe, row)
    np.testing.assert_allclose(intercept + contrib.sum(axis=1).to_numpy(), pipe.decision_function(to_X(row)), atol=1e-10)
    num = pipe["pre"].named_transformers_["num"]
    i = NUMERIC_FEATURES.index("dist_mean_km")
    median = num["impute"].statistics_[i]
    z = (median - num["scale"].mean_[i]) / num["scale"].scale_[i]
    coef = pipe["clf"].coef_[0][list(pipe["pre"].get_feature_names_out()).index("num__dist_mean_km")]
    assert contrib.iloc[0]["num__dist_mean_km"] == pytest.approx(coef * z)


def test_sign_and_positive_class_is_late(lr_setup):
    pipe, frame = lr_setup
    assert list(pipe["clf"].classes_) == [0, 1]
    contrib, intercept = linear_contributions(pipe, frame.head(50))
    raw = intercept + contrib.sum(axis=1).to_numpy()
    # sigmoid of the explained score is the predicted probability of class 1 = late
    np.testing.assert_allclose(sigmoid(raw), pipe.predict_proba(to_X(frame.head(50)))[:, 1], atol=1e-12)
    # raising the feature with the largest positive coefficient (one-hot excluded) raises P(late)
    names = list(pipe["pre"].get_feature_names_out())
    coefs = pd.Series(pipe["clf"].coef_[0], index=names)
    num = coefs[[f"num__{f}" for f in NUMERIC_FEATURES]]
    f = num.idxmax().removeprefix("num__")
    row = frame.head(1).copy()
    hi = row.assign(**{f: pd.to_numeric(row[f]).fillna(0) + 10 * pipe["pre"].named_transformers_["num"]["scale"].scale_[NUMERIC_FEATURES.index(f)]})
    assert pipe.predict_proba(to_X(hi))[0, 1] > pipe.predict_proba(to_X(row.assign(**{f: pd.to_numeric(row[f]).fillna(0)})))[0, 1]


def test_explanation_uses_the_same_model_as_the_shown_risk_score(lr_setup, pipeline_cfg):
    pipe, frame = lr_setup
    preds = pd.read_parquet(predictions_path(pipeline_cfg))
    lr = preds[preds["model_name"] == "logreg"].set_index("order_id")["risk_score"]
    for _, r in frame.head(10).iterrows():
        ex = logreg_explanation(pipe, r.to_frame().T.infer_objects())
        assert ex["score"] == pytest.approx(lr.loc[r["order_id"]], abs=1e-12)
        assert ex["table"]["Вклад (логит)"].sum() + ex["intercept"] == pytest.approx(ex["raw"])


def test_top_factors_are_positive_contributions(lr_setup):
    pipe, frame = lr_setup
    agg = aggregate_to_features(linear_contributions(pipe, frame.head(5))[0])
    tops = logreg_top_factors(pipe, frame.head(5))
    assert len(tops) == 5
    for (_, r), t in zip(agg.iterrows(), tops):
        assert len([x for x in t.split("; ") if x]) == min(3, int((r > 0).sum()))
