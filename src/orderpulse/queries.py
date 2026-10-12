"""Read-only analytical queries used by the UI. All SQL lives here, not in Streamlit.

Every KPI comes with its cohort/denominator so the UI can show it. Filters:
  period      purchase date (default time axis), inclusive bounds
  states      customer state
  categories  orders that contain at least one item of the category
  sellers     orders that contain at least one item of the seller
  statuses    order_status
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class Filters:
    date_from: date | None = None
    date_to: date | None = None
    states: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    sellers: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()


def where_orders(f: Filters, alias: str = "o") -> tuple[str, list]:
    """WHERE clause over mart_orders (alias ``o``) and its parameters."""
    clauses, params = ["TRUE"], []
    if f.date_from:
        clauses.append(f"{alias}.purchase_date >= ?")
        params.append(f.date_from)
    if f.date_to:
        clauses.append(f"{alias}.purchase_date <= ?")
        params.append(f.date_to)
    if f.states:
        clauses.append(f"{alias}.customer_state IN ({','.join('?' * len(f.states))})")
        params.extend(f.states)
    if f.statuses:
        clauses.append(f"{alias}.order_status IN ({','.join('?' * len(f.statuses))})")
        params.extend(f.statuses)
    if f.categories:
        clauses.append(
            f"EXISTS (SELECT 1 FROM mart_order_items fi WHERE fi.order_id = {alias}.order_id "
            f"AND fi.category IN ({','.join('?' * len(f.categories))}))")
        params.extend(f.categories)
    if f.sellers:
        clauses.append(
            f"EXISTS (SELECT 1 FROM mart_order_items fs WHERE fs.order_id = {alias}.order_id "
            f"AND fs.seller_id IN ({','.join('?' * len(f.sellers))}))")
        params.extend(f.sellers)
    return " AND ".join(clauses), params


def connect(warehouse_db: Path, predictions_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory connection with the warehouse attached read-only and predictions as a view."""
    # Persistent (non-temp) views in the in-memory catalog, so that cursors
    # created per thread (Streamlit) see them as well.
    con = duckdb.connect()
    con.execute(f"ATTACH '{Path(warehouse_db).as_posix()}' AS wh (READ_ONLY)")
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = 'wh'").fetchall()]
    for t in tables:
        con.execute(f'CREATE VIEW "{t}" AS SELECT * FROM wh.main."{t}"')
    if predictions_path is not None and Path(predictions_path).exists():
        con.execute(f"CREATE VIEW order_predictions AS SELECT * FROM read_parquet('{Path(predictions_path).as_posix()}')")
    return con


def has_table(con, name: str) -> bool:
    n = con.execute(
        "SELECT (SELECT count(*) FROM duckdb_tables() WHERE table_name = ?) "
        "     + (SELECT count(*) FROM duckdb_views() WHERE view_name = ?)", [name, name]).fetchone()[0]
    return n > 0


# ----------------------------------------------------------------------------- options

def filter_options(con) -> dict:
    lo, hi = con.execute("SELECT min(purchase_date), max(purchase_date) FROM mart_orders").fetchone()
    return {
        "date_min": lo, "date_max": hi,
        "states": [r[0] for r in con.execute("SELECT DISTINCT customer_state FROM mart_orders ORDER BY 1").fetchall()],
        "categories": [r[0] for r in con.execute("SELECT DISTINCT category FROM mart_order_items ORDER BY 1").fetchall()],
        "statuses": [r[0] for r in con.execute("SELECT DISTINCT order_status FROM mart_orders ORDER BY 1").fetchall()],
        "sellers": [r[0] for r in con.execute(
            "SELECT seller_id FROM mart_seller_orders GROUP BY 1 ORDER BY count(*) DESC, 1").fetchall()],
    }


# ----------------------------------------------------------------------------- overview

def kpis(con, f: Filters) -> dict:
    w, p = where_orders(f)
    r = con.execute(f"""
        SELECT
            count(*)                                              AS n_orders,
            count(*) FILTER (WHERE has_items)                     AS n_with_items,
            sum(items_value)                                      AS items_value,
            sum(freight_value)                                    AS freight_value,
            sum(payment_value)                                    AS payment_value,
            count(*) FILTER (WHERE n_payment_records > 0)         AS n_with_payment,
            count(*) FILTER (WHERE is_delivered)                  AS n_delivered,
            count(*) FILTER (WHERE is_late)                       AS n_late,
            median(delivery_days)                                 AS median_delivery_days,
            quantile_cont(delivery_days, 0.9)                     AS p90_delivery_days,
            avg(delay_days) FILTER (WHERE is_late)                AS mean_delay_days_late,
            median(delay_days) FILTER (WHERE is_late)             AS median_delay_days_late,
            count(*) FILTER (WHERE has_review)                    AS n_reviewed,
            avg(review_score)                                     AS avg_score,
            count(*) FILTER (WHERE is_low_review)                 AS n_low,
            count(*) FILTER (WHERE outcome = 'not_delivered')     AS n_not_delivered,
            count(*) FILTER (WHERE outcome = 'canceled')          AS n_canceled,
            count(*) FILTER (WHERE outcome = 'unavailable')       AS n_unavailable,
            count(*) FILTER (WHERE outcome = 'delivered_missing_date') AS n_delivered_missing_date
        FROM mart_orders o WHERE {w}""", p).df().iloc[0].to_dict()
    out = {k: (None if (isinstance(v, float) and math.isnan(v)) or v is None or v is pd.NA else v) for k, v in r.items()}
    for k in list(out):
        if k.startswith("n_") and out[k] is not None:
            out[k] = int(out[k])
    for k in ["items_value", "freight_value", "payment_value"]:
        out[k] = float(out[k]) if out[k] is not None else None
    out["avg_order_value"] = safe_div(out["items_value"], out["n_with_items"])
    out["avg_order_value_with_freight"] = safe_div(
        None if out["items_value"] is None else out["items_value"] + (out["freight_value"] or 0), out["n_with_items"])
    out["late_rate"] = safe_div(out["n_late"], out["n_delivered"])
    out["low_share"] = safe_div(out["n_low"], out["n_reviewed"])
    out["review_coverage"] = safe_div(out["n_reviewed"], out["n_orders"])
    out["excluded_share"] = safe_div(out["n_orders"] - out["n_delivered"], out["n_orders"])
    return out


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def monthly_trend(con, f: Filters) -> pd.DataFrame:
    w, p = where_orders(f)
    df = con.execute(f"""
        SELECT purchase_month AS month, count(*) AS orders,
               count(*) FILTER (WHERE is_delivered) AS delivered,
               count(*) FILTER (WHERE is_late) AS late,
               sum(items_value) AS items_value
        FROM mart_orders o WHERE {w} GROUP BY 1 ORDER BY 1""", p).df()
    df["late_rate"] = df["late"] / df["delivered"].where(df["delivered"] > 0)
    return df


def delivery_days_hist(con, f: Filters, cap: int = 60) -> pd.DataFrame:
    w, p = where_orders(f)
    return con.execute(f"""
        SELECT least(floor(delivery_days), {cap})::INTEGER AS days,
               count(*) FILTER (WHERE NOT is_late) AS on_time,
               count(*) FILTER (WHERE is_late) AS late
        FROM mart_orders o WHERE {w} AND is_delivered GROUP BY 1 ORDER BY 1""", p).df()


def outcome_breakdown(con, f: Filters) -> pd.DataFrame:
    w, p = where_orders(f)
    return con.execute(f"SELECT outcome, count(*) AS orders FROM mart_orders o WHERE {w} GROUP BY 1 ORDER BY 2 DESC", p).df()


def reviews_by_lateness(con, f: Filters) -> pd.DataFrame:
    """Review score distribution of delivered, reviewed orders split by lateness."""
    w, p = where_orders(f)
    return con.execute(f"""
        SELECT CASE WHEN is_late THEN 'Опоздал' ELSE 'Вовремя' END AS group_,
               review_score, count(*) AS n
        FROM mart_orders o WHERE {w} AND is_delivered AND has_review
        GROUP BY 1, 2 ORDER BY 1, 2""", p).df()


def reviews_summary_by_lateness(con, f: Filters) -> pd.DataFrame:
    w, p = where_orders(f)
    return con.execute(f"""
        SELECT CASE WHEN is_late THEN 'Опоздал' ELSE 'Вовремя' END AS "Группа",
               count(*) AS "Доставлено",
               count(*) FILTER (WHERE has_review) AS "С отзывом",
               round(avg(review_score), 2) AS "Средняя оценка",
               round(count(*) FILTER (WHERE is_low_review) / nullif(count(*) FILTER (WHERE has_review), 0), 4) AS "Доля 1–2"
        FROM mart_orders o WHERE {w} AND is_delivered GROUP BY 1 ORDER BY 1""", p).df()


# ----------------------------------------------------------------------------- segments

def wilson(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    n_ = n.astype(float).where(n > 0)
    phat = k / n_
    denom = 1 + z ** 2 / n_
    centre = (phat + z ** 2 / (2 * n_)) / denom
    half = z * ((phat * (1 - phat) / n_ + z ** 2 / (4 * n_ ** 2)) ** 0.5) / denom
    return centre - half, centre + half


SEGMENT_SQL = {
    "seller": ("mart_seller_orders", "s.seller_id", "s"),
    "seller_state": ("mart_seller_orders", "s.seller_state", "s"),
    "customer_state": ("mart_orders", "s.customer_state", "s"),
    "category": ("order_category", "s.category", "s"),
}


def segments(con, f: Filters, kind: str) -> pd.DataFrame:
    """Per-segment delivery statistics. Grain depends on kind (see SEGMENT_SQL docs)."""
    if kind not in SEGMENT_SQL:
        raise ValueError(kind)
    w, p = where_orders(f)
    if kind == "category":
        base = """(SELECT DISTINCT i.order_id, i.category, o.is_delivered, o.is_late, o.delivery_days,
                          o.has_review, o.review_score, o.is_low_review, o.items_value
                   FROM mart_order_items i JOIN mart_orders o USING (order_id) WHERE {w})"""
        seg_col = "s.category"
    elif kind in ("seller", "seller_state"):
        base = """(SELECT so.* FROM mart_seller_orders so JOIN mart_orders o USING (order_id) WHERE {w})"""
        seg_col = SEGMENT_SQL[kind][1]
    else:
        base = "(SELECT * FROM mart_orders o WHERE {w})"
        seg_col = "s.customer_state"
    df = con.execute(f"""
        SELECT {seg_col} AS segment,
               count(*) AS orders,
               count(*) FILTER (WHERE s.is_delivered) AS delivered,
               count(*) FILTER (WHERE s.is_late) AS late,
               median(s.delivery_days) AS median_delivery_days,
               count(*) FILTER (WHERE s.has_review) AS reviews,
               avg(s.review_score) AS avg_score,
               count(*) FILTER (WHERE s.is_low_review) AS low_reviews,
               sum(s.items_value) AS items_value
        FROM {base.format(w=w)} s GROUP BY 1""", p).df()
    df["late_rate"] = df["late"] / df["delivered"].where(df["delivered"] > 0)
    df["ci_low"], df["ci_high"] = wilson(df["late"], df["delivered"])
    df["low_share"] = df["low_reviews"] / df["reviews"].where(df["reviews"] > 0)
    return df.sort_values(["late_rate", "delivered"], ascending=[False, False]).reset_index(drop=True)


def segment_trend(con, kind: str, value: str) -> pd.DataFrame:
    seg_type = {"seller": "seller", "customer_state": "customer_state", "category": "category"}.get(kind)
    if seg_type is None:
        return pd.DataFrame()
    df = con.execute("""
        SELECT purchase_month AS month, sum(n_orders) AS orders, sum(n_delivered) AS delivered, sum(n_late) AS late
        FROM mart_delivery_segments WHERE segment_type = ? AND segment_value = ? GROUP BY 1 ORDER BY 1""",
                     [seg_type, value]).df()
    df["late_rate"] = df["late"] / df["delivered"].where(df["delivered"] > 0)
    return df


# ----------------------------------------------------------------------------- order card

def order_exists(con, order_id: str) -> bool:
    return con.execute("SELECT count(*) FROM mart_orders WHERE order_id = ?", [order_id]).fetchone()[0] > 0


def order_card(con, order_id: str) -> dict | None:
    df = con.execute("SELECT * FROM mart_orders WHERE order_id = ?", [order_id]).df()
    if df.empty:
        return None
    items = con.execute("""
        SELECT order_item_id, product_id, category, seller_id, seller_state, price, freight_value, weight_g
        FROM mart_order_items WHERE order_id = ? ORDER BY order_item_id""", [order_id]).df()
    sellers = con.execute("""
        SELECT seller_id, seller_state, n_items, items_value, freight_value, is_interstate
        FROM mart_seller_orders WHERE order_id = ? ORDER BY items_value DESC""", [order_id]).df()
    features = con.execute("SELECT * FROM feature_orders_at_approval WHERE order_id = ?", [order_id]).df()
    return {"order": df.iloc[0].to_dict(), "items": items, "sellers": sellers,
            "features": features.iloc[0].to_dict() if not features.empty else None}


# ----------------------------------------------------------------------------- risk queue

def prediction_models(con) -> pd.DataFrame:
    if not has_table(con, "order_predictions"):
        return pd.DataFrame(columns=["model_name", "model_version", "n"])
    return con.execute("""SELECT model_name, model_version, count(*) AS n FROM order_predictions
                          GROUP BY 1, 2 ORDER BY 1""").df()


def queue_dates(con, model_version: str) -> pd.DataFrame:
    return con.execute("""SELECT CAST(prediction_time AS DATE) AS d, count(*) AS n, any_value(split) AS split
                          FROM order_predictions WHERE model_version = ? GROUP BY 1 ORDER BY 1""",
                       [model_version]).df()


def risk_queue(con, model_version: str, day_from: date, day_to: date, k: int | None,
               states: tuple[str, ...] = (), multi_seller_only: bool = False,
               categories: tuple[str, ...] = ()) -> pd.DataFrame:
    """Orders approved in [day_from, day_to]: top-k PER APPROVAL DAY by risk score (after filters).

    k_d = min(k, orders of the day). Display ties are broken by order_id; quality
    metrics use the expected value over random tie-breaking (orderpulse.queue_eval).

    Only information available at approval time is selected here; outcome
    columns are returned separately by ``queue_outcomes`` so the UI can keep
    them hidden by default.
    """
    clauses = ["p.model_version = ?", "CAST(p.prediction_time AS DATE) BETWEEN ? AND ?"]
    params: list = [model_version, day_from, day_to]
    if states:
        clauses.append(f"f.customer_state IN ({','.join('?' * len(states))})")
        params.extend(states)
    if categories:
        clauses.append(f"f.main_category IN ({','.join('?' * len(categories))})")
        params.extend(categories)
    if multi_seller_only:
        clauses.append("f.n_sellers > 1")
    sql = f"""
        SELECT p.order_id, p.prediction_time AS approved_at, f.promised_ts AS promised_date,
               round(f.promised_days, 1) AS promised_days,
               f.items_value, f.freight_value, f.customer_state, f.main_seller_state,
               f.n_sellers, f.n_items, f.main_category,
               p.risk_score, p.risk_rank_in_day, p.top_factors,
               p.model_version, p.split
        FROM order_predictions p
        JOIN feature_orders_at_approval f USING (order_id)
        WHERE {' AND '.join(clauses)}
        {"QUALIFY row_number() OVER (PARTITION BY CAST(p.prediction_time AS DATE) "
          "ORDER BY p.risk_score DESC, p.order_id) <= " + str(int(k)) if k else ""}
        ORDER BY CAST(p.prediction_time AS DATE), p.risk_score DESC, p.order_id"""
    return con.execute(sql, params).df()


def queue_outcomes(con, order_ids: list[str]) -> pd.DataFrame:
    if not order_ids:
        return pd.DataFrame(columns=["order_id", "outcome", "delivered_ts", "delay_days", "review_score"])
    return con.execute(f"""
        SELECT order_id, outcome, delivered_ts, delay_days, review_score
        FROM mart_orders WHERE order_id IN ({','.join('?' * len(order_ids))})""", order_ids).df()


def order_predictions(con, order_id: str) -> pd.DataFrame:
    if not has_table(con, "order_predictions"):
        return pd.DataFrame()
    return con.execute("SELECT * FROM order_predictions WHERE order_id = ? ORDER BY model_name", [order_id]).df()


def queue_daily_quality(con, model_version: str, day_from: date, day_to: date, k: int,
                        states: tuple[str, ...] = (), multi_seller_only: bool = False,
                        categories: tuple[str, ...] = ()) -> dict:
    """Retrospective quality of the daily top-k queue on orders with an observed label.

    Cohort per day = filtered approved orders that were delivered with both dates
    (labelled); undelivered orders are excluded from n_d, m_d (no label).
    """
    from .queue_eval import daily_table, summarize
    pool = risk_queue(con, model_version, day_from, day_to, None, states, multi_seller_only, categories)
    if pool.empty:
        return {}
    out = con.execute(f"""SELECT order_id, is_late FROM mart_orders WHERE is_delivered
                          AND order_id IN ({','.join('?' * len(pool))})""", pool["order_id"].tolist()).df()
    lab = pool.merge(out, on="order_id")
    if lab.empty:
        return {}
    lab = lab.assign(day=pd.to_datetime(lab["approved_at"]).dt.date, is_late=lab["is_late"].astype(int))
    s = summarize(daily_table(lab, "risk_score", int(k)), int(k))
    s["unlabelled_orders"] = int(len(pool) - len(lab))
    return s


def feature_rows(con, order_ids: tuple[str, ...]) -> pd.DataFrame:
    """Allow-listed prediction-time features for the given orders (for on-demand explanations)."""
    from .features import FEATURES
    if not order_ids:
        return pd.DataFrame(columns=["order_id"] + FEATURES)
    cols = ", ".join(["order_id"] + FEATURES)
    return con.execute(f"SELECT {cols} FROM feature_orders_at_approval WHERE order_id IN "
                       f"({','.join('?' * len(order_ids))})", list(order_ids)).df()
