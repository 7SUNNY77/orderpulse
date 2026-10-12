"""Shared UI helpers: cached connection, artifacts, filters, formatting.

No ETL, training or batch SHAP runs here: the UI only reads the warehouse,
artifacts/*.json and artifacts/predictions.parquet. Caches are keyed by the
files' modification signature, so a rebuilt warehouse or a new model is
picked up automatically.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from orderpulse import queries
from orderpulse.config import load_config
from orderpulse.queries import Filters

COLORS = {"on_time": "#2a78d6", "late": "#eb6834", "neutral": "#8a8985", "series3": "#1baf7a",
          "series4": "#6250d6", "text_muted": "#52514e"}
MODEL_COLORS = {"logreg": "#2a78d6", "catboost": "#6250d6", "rule_short_promise": "#1baf7a", "constant": "#8a8985"}
OUTCOME_RU = {"on_time": "Вовремя", "late": "Опоздал", "not_delivered": "Не доставлен",
              "canceled": "Отменён", "unavailable": "Недоступен", "delivered_missing_date": "Доставлен, нет даты"}
SPLIT_RU = {"train": "train", "validation": "validation", "test": "test", "embargo": "зазор (embargo)",
            "before_train": "до train", "after_test": "после test (неполные исходы)"}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e6e5e1; border-radius: 10px;
  padding: 12px 14px;}
div[data-testid="stMetricLabel"] p {color: #52514e; font-size: 0.85rem;}
.op-note {color: #52514e; font-size: 0.85rem; margin-top: -0.4rem;}
.op-banner {background: #fff6e8; border: 1px solid #f1d7a8; border-radius: 10px; padding: 10px 14px; margin-bottom: 0.8rem;}
.op-card {background: #ffffff; border: 1px solid #e6e5e1; border-radius: 10px; padding: 12px 16px; margin-bottom: 0.6rem;}
</style>
"""


@st.cache_resource(show_spinner=False)
def _cfg():
    return load_config()


def cfg():
    return _cfg()


def _sig(path: Path) -> tuple:
    try:
        s = path.stat()
        return (str(path), s.st_mtime_ns, s.st_size)
    except FileNotFoundError:
        return (str(path), None, None)


def predictions_file() -> Path:
    return cfg().artifacts_dir / "predictions.parquet"


def data_signature() -> tuple:
    return (_sig(cfg().warehouse_db), _sig(predictions_file()))


@st.cache_resource(show_spinner=False, max_entries=1)
def _connection(signature: tuple):
    return queries.connect(cfg().warehouse_db, predictions_file())


def con():
    """Thread-local cursor on the shared cached connection."""
    return _connection(data_signature()).cursor()


def require_warehouse() -> None:
    if not cfg().warehouse_db.exists():
        st.error("Аналитическая база не найдена. Соберите её командой:\n\n"
                 "`python -m orderpulse ingest` → `python -m orderpulse marts` (или `python -m orderpulse pipeline`).")
        st.stop()


@st.cache_data(show_spinner=False, max_entries=256)
def q(fn: str, signature: tuple, *args):
    """Cached call of ``orderpulse.queries.<fn>(cursor, *args)``."""
    return getattr(queries, fn)(con(), *args)


def run(fn: str, *args):
    return q(fn, data_signature(), *args)


@st.cache_data(show_spinner=False)
def _json(path_str: str, signature: tuple):
    p = Path(path_str)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def artifact_json(name: str):
    p = cfg().artifacts_dir / name
    return _json(str(p), _sig(p))


@st.cache_data(show_spinner=False)
def _csv(path_str: str, signature: tuple):
    p = Path(path_str)
    return pd.read_csv(p) if p.exists() else None


def experiment_csv(name: str):
    """CSV from experiments/reliability_v2 (None if the stage has not been run)."""
    p = cfg().root / "experiments" / "reliability_v2" / name
    return _csv(str(p), _sig(p))


def report_json(name: str):
    p = cfg().reports_dir / name
    return _json(str(p), _sig(p))


# ----------------------------------------------------------------------------- formatting

def fmt_int(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{int(x):,}".replace(",", " ")


def fmt_pct(x, digits: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def fmt_num(x, digits: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.{digits}f}".replace(",", " ")


def fmt_money(x) -> str:
    return "—" if x is None or pd.isna(x) else "R$ " + f"{x:,.0f}".replace(",", " ")


def fmt_money_short(x) -> str:
    """Compact money for KPI tiles (full value goes into the note)."""
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1e6:
        return f"R$ {x / 1e6:.1f} млн".replace(".", ",")
    if abs(x) >= 1e3:
        return f"R$ {x / 1e3:.0f} тыс."
    return fmt_money(x)


def note(text: str) -> None:
    st.markdown(f"<div class='op-note'>{text}</div>", unsafe_allow_html=True)


def plotly_layout(fig, height: int = 320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=48, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="#fcfcfb", font=dict(color="#0b0b0b"),
                      title=dict(y=0.98, yanchor="top"),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor="#d6d5d0")
    fig.update_yaxes(gridcolor="#ecebe7", zeroline=False)
    return fig


# ----------------------------------------------------------------------------- filters

def sidebar_filters(key: str = "f", with_sellers: bool = True) -> Filters:
    opts = run("filter_options")
    st.sidebar.header("Фильтры")
    lo, hi = opts["date_min"], opts["date_max"]
    period = st.sidebar.date_input("Период покупки", value=(lo, hi), min_value=lo, max_value=hi, key=f"{key}_period",
                                   help="Ось времени по умолчанию — дата покупки (order_purchase_timestamp).")
    if isinstance(period, (list, tuple)) and len(period) == 2:
        d_from, d_to = period
    else:  # the user is still choosing the second date
        d_from, d_to = (period[0] if isinstance(period, (list, tuple)) and period else lo), hi
    states = st.sidebar.multiselect("Штат покупателя", opts["states"], key=f"{key}_states")
    cats = st.sidebar.multiselect("Категория (заказ содержит товар категории)", opts["categories"], key=f"{key}_cats")
    sellers = []
    if with_sellers:
        sellers = st.sidebar.multiselect("Продавец (заказ содержит товар продавца)", opts["sellers"],
                                         key=f"{key}_sellers", help="Список отсортирован по числу заказов.")
    statuses = st.sidebar.multiselect("Статус заказа", opts["statuses"], key=f"{key}_status")
    return Filters(date_from=d_from, date_to=d_to, states=tuple(states), categories=tuple(cats),
                   sellers=tuple(sellers), statuses=tuple(statuses))


def model_versions_consistent() -> tuple[bool, str]:
    """Predictions, metrics and metadata must come from the same model versions."""
    meta = artifact_json("model_metadata.json")
    metrics = artifact_json("metrics.json")
    if meta is None:
        return False, "Модель не обучена: выполните `python -m orderpulse train`."
    meta_v = {n: m["version"] for n, m in meta["models"].items()}
    if metrics is None:
        return False, "Метрики не рассчитаны: выполните `python -m orderpulse evaluate`."
    if metrics.get("model_versions") != meta_v:
        return False, "Метрики рассчитаны для другой версии модели: выполните `python -m orderpulse evaluate`."
    if not predictions_file().exists():
        return False, "Прогнозы не рассчитаны: выполните `python -m orderpulse predict`."
    pv = run("prediction_models")
    pred_v = dict(zip(pv["model_name"], pv["model_version"]))
    if pred_v != meta_v:
        return False, "Прогнозы рассчитаны другой версией модели: выполните `python -m orderpulse predict`."
    return True, ""


def today_str(d: date | None) -> str:
    return "—" if d is None else d.strftime("%d.%m.%Y")


# ----------------------------------------------------------------------------- model selection & explanations

def default_model() -> str:
    return str(cfg().model["ui"].get("default_model", "logreg"))


def model_order() -> list[str]:
    names = ["logreg", "catboost", "rule_short_promise", "constant"]
    d = default_model()
    return [d] + [n for n in names if n != d]


def select_model(container, page_key: str, label: str = "Модель") -> str:
    """Model selector shared across pages via st.session_state['op_model'].

    The queue and the order card read and write the same value, so a card opened
    from the queue shows the model the queue was ranked by.
    """
    from orderpulse.models import MODEL_TITLES
    names = model_order()
    cur = st.session_state.get("op_model", default_model())
    idx = names.index(cur) if cur in names else 0
    title = lambda n: MODEL_TITLES.get(n, n)  # noqa: E731
    m = container.selectbox(label, names, index=idx, format_func=title, key=f"model_select_{page_key}")
    st.session_state["op_model"] = m
    return m


def _models_signature() -> tuple:
    a = cfg().artifacts_dir
    return tuple(_sig(a / f) for f in ("model.cbm", "logreg.joblib", "baselines.json"))


@st.cache_resource(show_spinner=False, max_entries=1)
def _models(signature: tuple):
    from orderpulse.models import load_models
    return load_models(cfg().artifacts_dir)


def models():
    return _models(_models_signature())


@st.cache_data(show_spinner=False, max_entries=500)
def _explain(model_name: str, order_id: str, signature: tuple, feature_row: dict) -> dict | None:
    """Local explanation of ONE order for the given model (cached per model artifact version)."""
    from orderpulse.explanations import local_explanation, logreg_explanation
    from orderpulse.features import FEATURES
    row = pd.DataFrame([{f: feature_row.get(f) for f in FEATURES}])
    m = _models(signature)
    if model_name == "logreg":
        ex = logreg_explanation(m["logreg"].pipeline, row)
        ex.pop("transformed")
        return ex
    if model_name == "catboost":
        return local_explanation(m["catboost"], row)
    return None


def explain(model_name: str, order_id: str, feature_row: dict):
    return _explain(model_name, order_id, _models_signature(), feature_row)


@st.cache_data(show_spinner=False, max_entries=200)
def _logreg_factors(order_ids: tuple, signature: tuple, wh_sig: tuple) -> dict:
    from orderpulse.explanations import logreg_top_factors
    feats = queries_feature_rows(list(order_ids))
    return dict(zip(feats["order_id"], logreg_top_factors(_models(signature)["logreg"].pipeline, feats)))


def queries_feature_rows(order_ids: list[str]):
    return run("feature_rows", tuple(order_ids))


def logreg_factors(order_ids: list[str]) -> dict:
    """Top linear contributions for the orders shown in the queue (computed on demand, cached)."""
    return _logreg_factors(tuple(order_ids), _models_signature(), data_signature())
