"""End-to-end on synthetic CSVs: marts -> features -> train -> evaluate -> predict -> UI contract."""

from __future__ import annotations

import json
import os
import shutil
from datetime import date

import duckdb
import pandas as pd
import pytest

from conftest import make_cfg
from orderpulse import queries
from orderpulse.build_marts import build_warehouse
from orderpulse.config import PACKAGE_ROOT
from orderpulse.evaluate import evaluate
from orderpulse.features import FEATURES
from orderpulse.ingest import ingest
from orderpulse.models import MODEL_NAMES, load_models
from orderpulse.predict import PREDICTION_COLUMNS, predict, predictions_path
from orderpulse.quality import run_quality
from orderpulse.train import train


def test_artifacts_and_metadata(pipeline_cfg):
    a = pipeline_cfg.artifacts_dir
    for f in ["model.cbm", "logreg.joblib", "baselines.json", "model_metadata.json", "metrics.json", "predictions.parquet"]:
        assert (a / f).exists(), f
    meta = json.loads((a / "model_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((a / "metrics.json").read_text(encoding="utf-8"))
    assert meta["features"] == FEATURES
    assert {n: m["version"] for n, m in meta["models"].items()} == metrics["model_versions"]
    for key in ["data_fingerprint", "trained_at", "splits", "seed", "policy", "catboost"]:
        assert key in meta
    assert (pipeline_cfg.reports_dir / "data_quality.md").exists()
    assert (pipeline_cfg.reports_dir / "build_checks.json").exists()


def test_models_reload_and_score_identically(pipeline_cfg):
    models = load_models(pipeline_cfg.artifacts_dir)
    preds = pd.read_parquet(predictions_path(pipeline_cfg))
    con = duckdb.connect(str(pipeline_cfg.warehouse_db), read_only=True)
    from orderpulse.features import load_feature_frame
    frame = load_feature_frame(con).head(20)
    con.close()
    p = preds[preds["model_name"] == "catboost"].set_index("order_id").loc[frame["order_id"], "risk_score"].to_numpy()
    assert models["catboost"].score(frame) == pytest.approx(p)


def test_predictions_contract(pipeline_cfg):
    preds = pd.read_parquet(predictions_path(pipeline_cfg))
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert not preds.duplicated(["order_id", "prediction_time", "model_version"]).any()
    assert set(preds["model_name"]) == set(MODEL_NAMES)
    assert preds["risk_score"].notna().all()
    assert preds["feature_version"].nunique() == 1


def test_predict_is_idempotent(pipeline_cfg):
    before = pd.read_parquet(predictions_path(pipeline_cfg)).drop(columns="scored_at")
    predict(pipeline_cfg)
    after = pd.read_parquet(predictions_path(pipeline_cfg)).drop(columns="scored_at")
    pd.testing.assert_frame_equal(before, after)


def test_queue_is_daily_top_k_and_respects_filters(pipeline_cfg):
    con = queries.connect(pipeline_cfg.warehouse_db, predictions_path(pipeline_cfg))
    meta = json.loads((pipeline_cfg.artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
    v = meta["models"]["catboost"]["version"]
    full = queries.risk_queue(con, v, date(2017, 1, 1), date(2018, 12, 31), None, ("SP",))
    q = queries.risk_queue(con, v, date(2017, 1, 1), date(2018, 12, 31), 1, ("SP",))
    assert (q["customer_state"] == "SP").all()
    full = full.assign(day=full["approved_at"].dt.date)
    q = q.assign(day=q["approved_at"].dt.date)
    # exactly min(K, n_d) orders per day, and they are the day's highest scores
    assert q.groupby("day").size().max() == 1
    assert set(q["day"]) == set(full["day"])
    best = full.sort_values(["day", "risk_score", "order_id"], ascending=[True, False, True]).groupby("day").head(1)
    assert sorted(q["order_id"]) == sorted(best["order_id"])
    # outcome columns are not part of the queue query (hidden by default)
    assert not {"outcome", "delivered_ts", "is_late", "review_score"} & set(q.columns)
    qs = queries.queue_daily_quality(con, v, date(2017, 1, 1), date(2018, 12, 31), 2)
    assert qs["model_reviewed"] <= qs["model_orders"] and qs["random_micro_precision"] > 0
    con.close()


def test_evaluation_covers_all_models_on_same_rows(pipeline_cfg):
    m = json.loads((pipeline_cfg.artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    for split in ("validation", "test"):
        ns = {m[split][n]["n"] for n in MODEL_NAMES}
        assert len(ns) == 1 and ns.pop() == m["splits"][split]["used"]


# ----------------------------------------------------------------------------- UI (AppTest)

def _apptest(page: str, env: dict, session: dict | None = None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    st.cache_resource.clear()
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        at = AppTest.from_file(str(PACKAGE_ROOT / "app" / "views" / f"{page}.py"), default_timeout=120)
        for k, v in (session or {}).items():
            at.session_state[k] = v
        at.run()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return at


def _env(cfg, artifacts=None):
    return {"ORDERPULSE_WAREHOUSE_DB": str(cfg.warehouse_db),
            "ORDERPULSE_ARTIFACTS_DIR": str(artifacts or cfg.artifacts_dir),
            "ORDERPULSE_REPORTS_DIR": str(cfg.reports_dir)}


@pytest.mark.parametrize("page", ["overview", "segments", "risk_queue", "order_card", "quality"])
def test_pages_render_without_exceptions(pipeline_cfg, page):
    at = _apptest(page, _env(pipeline_cfg), {"card_order_id": "o_multi"})
    assert not at.exception, [e.value for e in at.exception]


def test_queue_hides_outcomes_by_default(pipeline_cfg):
    at = _apptest("risk_queue", _env(pipeline_cfg))
    assert not at.exception
    cols = set(at.dataframe[0].value.columns)
    assert "Исход" not in cols and "Опоздание, дн." not in cols
    assert any("историческая симуляция" in m.value for m in at.markdown)


def test_unknown_order_id_message(pipeline_cfg):
    at = _apptest("order_card", _env(pipeline_cfg), {"card_order_id": "no-such-order"})
    assert not at.exception
    assert any("не найден" in w.value for w in at.warning)


def test_missing_artifacts_give_message_not_traceback(pipeline_cfg, tmp_path):
    empty = tmp_path / "no_artifacts"
    empty.mkdir()
    for page in ["risk_queue", "quality", "order_card"]:
        at = _apptest(page, _env(pipeline_cfg, empty), {"card_order_id": "o_multi"})
        assert not at.exception, (page, [e.value for e in at.exception])
    at = _apptest("risk_queue", _env(pipeline_cfg, empty))
    assert any("не обучена" in w.value for w in at.warning)


def test_missing_warehouse_gives_message(tmp_path, pipeline_cfg):
    env = {"ORDERPULSE_WAREHOUSE_DB": str(tmp_path / "nope.duckdb"),
           "ORDERPULSE_ARTIFACTS_DIR": str(pipeline_cfg.artifacts_dir),
           "ORDERPULSE_REPORTS_DIR": str(pipeline_cfg.reports_dir)}
    at = _apptest("overview", env)
    assert not at.exception
    assert any("не найдена" in e.value for e in at.error)


def test_empty_filter_selection_message(pipeline_cfg):
    import streamlit as st  # noqa: F401
    at = _apptest("overview", _env(pipeline_cfg))
    at.sidebar.multiselect[0].set_value(["RS"])
    at.sidebar.multiselect[2].set_value([at.sidebar.multiselect[2].options[0]]) if at.sidebar.multiselect[2].options else None
    at.run()
    assert not at.exception


def test_main_entry_point_with_navigation(pipeline_cfg):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    st.cache_resource.clear()
    env = _env(pipeline_cfg)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        at = AppTest.from_file(str(PACKAGE_ROOT / "app" / "main.py"), default_timeout=120).run()
        assert not at.exception, [e.value for e in at.exception]
        assert any(t.value == "Обзор доставки" for t in at.title)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ----------------------------------------------------------------------------- release v1.0 UI scenario

def test_queue_defaults_to_logreg_and_k20(pipeline_cfg):
    at = _apptest("risk_queue", _env(pipeline_cfg))
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == "logreg"
    assert at.number_input[0].value == 20
    df = at.dataframe[0].value
    meta = json.loads((pipeline_cfg.artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert set(df["Модель"]) == {meta["models"]["logreg"]["version"]}
    assert "Факторы ↑ (линейные вклады LogReg)" in df.columns
    assert "Факторы ↑ (SHAP CatBoost)" not in df.columns


def test_queue_multi_day_range_is_top_k_per_day(pipeline_cfg):
    at = _apptest("risk_queue", _env(pipeline_cfg))
    at.date_input[0].set_value((date(2018, 4, 1), date(2018, 6, 30)))
    at.number_input[0].set_value(1)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    df = at.dataframe[0].value
    per_day = df.groupby(pd.to_datetime(df["Подтверждён"]).dt.date).size()
    assert per_day.max() == 1 and len(df) == len(per_day) > 1


def test_card_keeps_model_from_queue_and_explains_that_model(pipeline_cfg):
    import streamlit as st  # noqa: F401
    at = _apptest("order_card", _env(pipeline_cfg), {"card_order_id": "o00010", "op_model": "logreg"})
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == "logreg"
    md = " ".join(m.value for m in at.markdown)
    assert "Вклады признаков в линейный скор" in md and "SHAP" not in md.split("Подсказки")[0].split("линейный скор")[0]
    at = _apptest("order_card", _env(pipeline_cfg), {"card_order_id": "o00010", "op_model": "catboost"})
    assert not at.exception
    assert at.selectbox[0].value == "catboost"
    md = " ".join(m.value for m in at.markdown)
    assert "Локальные SHAP-вклады" in md and "Вклады признаков в линейный скор" not in md
    # switching the model inside the card updates the explanation
    at.selectbox[0].set_value("logreg").run()
    md = " ".join(m.value for m in at.markdown)
    assert "Вклады признаков в линейный скор" in md and "Локальные SHAP-вклады" not in md
    assert at.session_state["op_model"] == "logreg"


def test_card_score_matches_queue_model_prediction(pipeline_cfg):
    at = _apptest("order_card", _env(pipeline_cfg), {"card_order_id": "o00010", "op_model": "logreg"})
    preds = pd.read_parquet(predictions_path(pipeline_cfg))
    s = preds[(preds["model_name"] == "logreg") & (preds["order_id"] == "o00010")]["risk_score"].iloc[0]
    assert any(m.value == f"{s:.3f}" for m in at.metric)


def test_queue_empty_selection_message(pipeline_cfg):
    at = _apptest("risk_queue", _env(pipeline_cfg))
    # 'cat_no_translation' exists only in one SP order (2017-03-01); with state RS nothing matches
    at.expander[0].multiselect[0].set_value(["RS"])
    at.expander[0].multiselect[1].set_value(["cat_no_translation"])
    at.run()
    assert not at.exception
    assert any("Нет подтверждённых заказов" in i.value for i in at.info)
    assert len(at.dataframe) == 0


def test_queue_outcomes_toggle_shows_random_baseline(pipeline_cfg):
    at = _apptest("risk_queue", _env(pipeline_cfg))
    at.date_input[0].set_value((date(2018, 4, 1), date(2018, 6, 30)))
    at.run()
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert "Исход" in at.dataframe[0].value.columns
    labels = [m.label for m in at.metric]
    assert "Найдено задержек в день" in labels and any("случайно" in (m.delta or "") for m in at.metric)
