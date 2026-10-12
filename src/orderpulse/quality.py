"""Reproducible data-quality audit.

Runs on the raw layer (after ``ingest``), builds staging/marts in memory to
compute exclusion statistics, and writes:
  reports/data_quality.json  machine-readable results
  reports/data_quality.md    human-readable report

Each finding has a severity:
  critical - the pipeline must stop (missing columns, broken primary keys);
  warning  - anomaly that is flagged/excluded with a documented rule;
  expected - a known property of the data grain (e.g. many reviews per order).
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb

from .build_marts import LAYERS, attach_raw, run_sql_dir, sql_params
from .config import Config
from .ingest import REQUIRED_COLUMNS, check_contract
from .utils import LOG, timed, write_json_atomic, write_text_atomic


class QualityError(RuntimeError):
    """At least one critical data-quality finding."""


def _q(con, sql: str):
    return con.execute(sql).fetchone()


def _df(con, sql: str):
    return con.execute(sql).df()


class Audit:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.findings: list[dict] = []

    def add(self, section: str, check: str, severity: str, count: int | float, denominator: int | None,
            note: str, action: str = "") -> None:
        share = (count / denominator) if denominator else None
        self.findings.append({
            "section": section, "check": check, "severity": severity,
            "count": count, "denominator": denominator,
            "share": round(share, 6) if share is not None else None,
            "note": note, "action": action,
        })

    def flag(self, section: str, check: str, count: int, denominator: int | None, note: str,
             action: str = "", severity_if_any: str = "warning") -> None:
        severity = severity_if_any if count else "ok"
        self.add(section, check, severity, count, denominator, note, action)


def _audit(con: duckdb.DuckDBPyConnection, cfg: Config) -> dict:
    a = Audit(con)
    n = {t: _q(con, f"SELECT count(*) FROM raw_{t}")[0] for t in REQUIRED_COLUMNS}

    # ---------------- primary keys / grain ----------------
    def dupes(table: str, key: str) -> int:
        return _q(con, f"SELECT count(*) - count(DISTINCT ({key})) FROM raw_{table}")[0]

    for table, key in [("orders", "order_id"), ("customers", "customer_id"), ("sellers", "seller_id"),
                       ("products", "product_id"), ("order_items", "order_id, order_item_id"),
                       ("order_payments", "order_id, payment_sequential"),
                       ("order_reviews", "review_id, order_id")]:
        a.flag("keys", f"{table}: duplicates of ({key})", dupes(table, key), n[table],
               "primary key of the table grain", "pipeline stops if > 0", severity_if_any="critical")
    a.flag("keys", "category_translation: duplicated category name",
           dupes("category_translation", "product_category_name"), n["category_translation"],
           "dimension must be unique", "first English name kept")
    a.add("keys", "order_reviews: review_id linked to >1 order", "expected",
          dupes("order_reviews", "review_id"), n["order_reviews"],
          "review_id is not unique on its own; the grain is (review_id, order_id)")
    multi_rev = _q(con, "SELECT count(*) FROM (SELECT order_id FROM raw_order_reviews GROUP BY 1 HAVING count(*) > 1)")[0]
    a.add("keys", "orders with more than one review", "expected", multi_rev, n["orders"],
          "canonical review = latest review_creation_date, then answer timestamp, then review_id")
    multi_pay = _q(con, "SELECT count(*) FROM (SELECT order_id FROM raw_order_payments GROUP BY 1 HAVING count(*) > 1)")[0]
    a.add("keys", "orders with more than one payment record", "expected", multi_pay, n["orders"],
          "payments are summed per order before joining")
    multi_item = _q(con, "SELECT count(*) FROM (SELECT order_id FROM raw_order_items GROUP BY 1 HAVING count(*) > 1)")[0]
    a.add("keys", "orders with more than one item", "expected", multi_item, n["orders"],
          "items are aggregated per order before joining")
    multi_seller = _q(con, "SELECT count(*) FROM (SELECT order_id FROM raw_order_items GROUP BY 1 HAVING count(DISTINCT seller_id) > 1)")[0]
    a.add("keys", "orders with more than one seller", "expected", multi_seller, n["orders"],
          "counted once in order-level KPIs and once per seller in seller statistics")
    a.add("keys", "geolocation: points per zip prefix > 1", "expected",
          n["geolocation"] - _q(con, "SELECT count(DISTINCT geolocation_zip_code_prefix) FROM raw_geolocation")[0],
          n["geolocation"], "aggregated to one row per prefix (median lat/lng)")
    a.add("keys", "customer_unique_id with several customer_id", "expected",
          _q(con, "SELECT count(*) FROM (SELECT customer_unique_id FROM raw_customers GROUP BY 1 HAVING count(*) > 1)")[0],
          _q(con, "SELECT count(DISTINCT customer_unique_id) FROM raw_customers")[0],
          "repeat customers; customer_id is per order")

    # ---------------- referential integrity ----------------
    refs = [
        ("order_items.order_id -> orders", "raw_order_items", "order_id", "raw_orders", "order_id"),
        ("order_payments.order_id -> orders", "raw_order_payments", "order_id", "raw_orders", "order_id"),
        ("order_reviews.order_id -> orders", "raw_order_reviews", "order_id", "raw_orders", "order_id"),
        ("orders.customer_id -> customers", "raw_orders", "customer_id", "raw_customers", "customer_id"),
        ("order_items.product_id -> products", "raw_order_items", "product_id", "raw_products", "product_id"),
        ("order_items.seller_id -> sellers", "raw_order_items", "seller_id", "raw_sellers", "seller_id"),
    ]
    for name, child, ck, parent, pk in refs:
        cnt = _q(con, f"SELECT count(*) FROM {child} c WHERE c.{ck} IS NOT NULL AND NOT EXISTS "
                      f"(SELECT 1 FROM {parent} p WHERE p.{pk} = c.{ck})")[0]
        a.flag("referential", name, cnt, _q(con, f"SELECT count(*) FROM {child}")[0],
               "orphan child rows", "reported; orphans cannot enter order marts")
    cnt = _q(con, "SELECT count(*) FROM stg_customers c WHERE NOT EXISTS (SELECT 1 FROM stg_geo_zip g WHERE g.zip = c.customer_zip)")[0]
    a.flag("referential", "customer zip prefix without coordinates", cnt, n["customers"],
           "distance features are NULL", "geo_missing indicator = 1")
    cnt = _q(con, "SELECT count(*) FROM stg_sellers s WHERE NOT EXISTS (SELECT 1 FROM stg_geo_zip g WHERE g.zip = s.seller_zip)")[0]
    a.flag("referential", "seller zip prefix without coordinates", cnt, n["sellers"],
           "distance features are NULL", "geo_missing indicator = 1")
    cnt = _q(con, "SELECT count(*) FROM raw_products p WHERE p.product_category_name IS NOT NULL AND NOT EXISTS "
                  "(SELECT 1 FROM raw_category_translation t WHERE t.product_category_name = p.product_category_name)")[0]
    cats = [r[0] for r in con.execute(
        "SELECT DISTINCT product_category_name FROM raw_products p WHERE p.product_category_name IS NOT NULL AND NOT EXISTS "
        "(SELECT 1 FROM raw_category_translation t WHERE t.product_category_name = p.product_category_name) ORDER BY 1").fetchall()]
    a.flag("referential", "product category without English translation", cnt, n["products"],
           f"categories: {cats}", "original Portuguese name is kept")
    for label, sql, action in [
        ("orders without items", "SELECT count(*) FROM raw_orders o WHERE NOT EXISTS (SELECT 1 FROM raw_order_items i WHERE i.order_id = o.order_id)",
         "kept in mart_orders with NULL item value (not 0); excluded from ML"),
        ("orders without payments", "SELECT count(*) FROM raw_orders o WHERE NOT EXISTS (SELECT 1 FROM raw_order_payments i WHERE i.order_id = o.order_id)",
         "kept; payment value NULL"),
        ("orders without reviews", "SELECT count(*) FROM raw_orders o WHERE NOT EXISTS (SELECT 1 FROM raw_order_reviews i WHERE i.order_id = o.order_id)",
         "kept; review metrics use reviewed orders only"),
    ]:
        a.flag("referential", label, _q(con, sql)[0], n["orders"], "", action)

    # ---------------- typing ----------------
    casts = [
        ("orders", "order_purchase_timestamp", "TIMESTAMP"), ("orders", "order_approved_at", "TIMESTAMP"),
        ("orders", "order_delivered_carrier_date", "TIMESTAMP"), ("orders", "order_delivered_customer_date", "TIMESTAMP"),
        ("orders", "order_estimated_delivery_date", "TIMESTAMP"), ("order_items", "price", "DECIMAL(18,2)"),
        ("order_items", "freight_value", "DECIMAL(18,2)"), ("order_items", "order_item_id", "INTEGER"),
        ("order_payments", "payment_value", "DECIMAL(18,2)"), ("order_payments", "payment_installments", "INTEGER"),
        ("order_reviews", "review_score", "INTEGER"), ("order_reviews", "review_creation_date", "TIMESTAMP"),
        ("products", "product_weight_g", "DOUBLE"), ("geolocation", "geolocation_lat", "DOUBLE"),
    ]
    for table, col, typ in casts:
        cnt = _q(con, f"SELECT count(*) FROM raw_{table} WHERE {col} IS NOT NULL AND TRY_CAST({col} AS {typ}) IS NULL")[0]
        a.flag("typing", f"{table}.{col} not castable to {typ}", cnt, n[table], "", "value becomes NULL")

    # ---------------- amounts ----------------
    for label, sql, denom, sev, note in [
        ("negative item price", "SELECT count(*) FROM stg_order_items WHERE price < 0", n["order_items"], "warning", ""),
        ("negative freight", "SELECT count(*) FROM stg_order_items WHERE freight_value < 0", n["order_items"], "warning", ""),
        ("negative payment", "SELECT count(*) FROM stg_order_payments WHERE payment_value < 0", n["order_payments"], "warning", ""),
        ("zero item price", "SELECT count(*) FROM stg_order_items WHERE price = 0", n["order_items"], "warning", ""),
        ("zero freight (free shipping)", "SELECT count(*) FROM stg_order_items WHERE freight_value = 0", n["order_items"], "expected",
         "free shipping is a valid business case"),
        ("zero payment records", "SELECT count(*) FROM stg_order_payments WHERE payment_value = 0", n["order_payments"], "expected",
         "mostly vouchers"),
        ("payment_type = not_defined", "SELECT count(*) FROM stg_order_payments WHERE payment_type = 'not_defined'", n["order_payments"],
         "warning", ""),
        ("payment_installments = 0", "SELECT count(*) FROM stg_order_payments WHERE payment_installments = 0", n["order_payments"],
         "warning", ""),
    ]:
        cnt = _q(con, sql)[0]
        a.add("amounts", label, sev if cnt else "ok", cnt, denom, note, "flagged, kept")

    # ---------------- temporal ----------------
    t = _df(con, """
        SELECT
          count(*) FILTER (WHERE purchase_ts IS NULL)                       AS purchase_missing,
          count(*) FILTER (WHERE approved_ts < purchase_ts)                 AS approved_before_purchase,
          count(*) FILTER (WHERE carrier_ts < purchase_ts)                  AS carrier_before_purchase,
          count(*) FILTER (WHERE carrier_ts < approved_ts)                  AS carrier_before_approval,
          count(*) FILTER (WHERE delivered_ts < carrier_ts)                 AS delivered_before_carrier,
          count(*) FILTER (WHERE delivered_ts < approved_ts)                AS delivered_before_approval,
          count(*) FILTER (WHERE delivered_ts < purchase_ts)                AS delivered_before_purchase,
          count(*) FILTER (WHERE estimated_ts < purchase_ts)                AS estimated_before_purchase,
          count(*) FILTER (WHERE CAST(estimated_ts AS DATE) < CAST(approved_ts AS DATE)) AS estimated_before_approval,
          count(*) FILTER (WHERE order_status <> 'delivered' AND delivered_ts IS NOT NULL) AS delivered_date_not_delivered_status,
          count(*) FILTER (WHERE order_status = 'delivered' AND delivered_ts IS NULL)      AS delivered_status_no_date,
          count(*) FILTER (WHERE order_status = 'delivered' AND approved_ts IS NULL)       AS delivered_status_not_approved
        FROM stg_orders""").iloc[0].to_dict()
    temporal_actions = {
        "purchase_missing": ("critical", "purchase date is the default time axis"),
        "approved_before_purchase": ("warning", "excluded from ML (temporal_error)"),
        "carrier_before_purchase": ("warning", "flag only; carrier date is not used in features or target"),
        "carrier_before_approval": ("warning", "flag only; carrier date is not used in features or target"),
        "delivered_before_carrier": ("warning", "flag only; target uses delivered vs promised date"),
        "delivered_before_approval": ("warning", "excluded from ML (temporal_error): prediction time is unreliable"),
        "delivered_before_purchase": ("warning", "excluded from ML (temporal_error)"),
        "estimated_before_purchase": ("warning", "excluded from ML (temporal_error)"),
        "estimated_before_approval": ("warning", "kept; promised_days <= 0 is a legitimate feature value"),
        "delivered_date_not_delivered_status": ("warning", "outside the delivered cohort (status is authoritative)"),
        "delivered_status_no_date": ("warning", "outside the delivered cohort; NOT treated as on time"),
        "delivered_status_not_approved": ("warning", "excluded from ML (not_approved)"),
    }
    for k, v in t.items():
        sev, action = temporal_actions[k]
        a.add("temporal", k, sev if v else "ok", int(v), n["orders"], "", action)

    # ---------------- geolocation ----------------
    b = cfg.data["geo_bounds"]
    oob = _q(con, f"""SELECT count(*) FROM raw_geolocation WHERE
        TRY_CAST(geolocation_lat AS DOUBLE) NOT BETWEEN {b['lat_min']} AND {b['lat_max']}
        OR TRY_CAST(geolocation_lng AS DOUBLE) NOT BETWEEN {b['lng_min']} AND {b['lng_max']}""")[0]
    a.flag("geolocation", "points outside Brazil bounding box", oob, n["geolocation"], "",
           "dropped before the per-prefix median")

    # ---------------- missing values ----------------
    missing = {}
    for table, cols in REQUIRED_COLUMNS.items():
        exprs = ", ".join(f"count(*) FILTER (WHERE \"{c}\" IS NULL OR trim(\"{c}\") = '') AS \"{c}\"" for c in cols)
        row = _df(con, f"SELECT {exprs} FROM raw_{table}").iloc[0].to_dict()
        missing[table] = {k: int(v) for k, v in row.items() if v}

    # ---------------- statuses & coverage ----------------
    statuses = _df(con, """
        SELECT order_status, count(*) AS n,
               count(approved_ts) AS has_approved, count(carrier_ts) AS has_carrier,
               count(delivered_ts) AS has_delivered, count(estimated_ts) AS has_estimated
        FROM stg_orders GROUP BY 1 ORDER BY n DESC""")
    coverage = _df(con, """
        SELECT 'purchase' AS field, min(purchase_ts) AS min, max(purchase_ts) AS max FROM stg_orders
        UNION ALL SELECT 'approved', min(approved_ts), max(approved_ts) FROM stg_orders
        UNION ALL SELECT 'carrier', min(carrier_ts), max(carrier_ts) FROM stg_orders
        UNION ALL SELECT 'delivered', min(delivered_ts), max(delivered_ts) FROM stg_orders
        UNION ALL SELECT 'estimated', min(estimated_ts), max(estimated_ts) FROM stg_orders
        UNION ALL SELECT 'review_creation', min(review_creation_ts), max(review_creation_ts) FROM stg_order_reviews""")
    monthly = _df(con, """
        SELECT strftime(purchase_month, '%Y-%m') AS month, count(*) AS orders,
               count(*) FILTER (WHERE is_delivered) AS delivered,
               count(*) FILTER (WHERE is_late) AS late,
               round(avg(is_late::INTEGER), 4) AS late_rate
        FROM mart_orders GROUP BY 1 ORDER BY 1""")
    cutoff = str(cfg.data["data_cutoff"])
    after_cutoff = _q(con, f"""SELECT
        count(*) FILTER (WHERE purchase_ts >= TIMESTAMP '{cutoff}'),
        count(*) FILTER (WHERE delivered_ts >= TIMESTAMP '{cutoff}')
        FROM stg_orders""")
    a.add("coverage", f"orders purchased on/after data_cutoff {cutoff}", "expected", after_cutoff[0], n["orders"],
          "sparse tail after the effective snapshot", "excluded from ML splits by time boundaries")
    a.add("coverage", f"deliveries on/after data_cutoff {cutoff}", "expected", after_cutoff[1], n["orders"],
          "outcomes after the effective snapshot are treated as not observable", "")

    # ---------------- ML suitability ----------------
    excl = _df(con, """
        SELECT coalesce(ml_exclusion_reason, 'eligible') AS reason, count(*) AS n,
               round(count(*) / sum(count(*)) OVER (), 6) AS share
        FROM mart_orders GROUP BY 1 ORDER BY n DESC""")
    elig = _q(con, "SELECT count(*), avg(is_late::INTEGER) FROM mart_orders WHERE ml_exclusion_reason IS NULL")
    delivered = _q(con, "SELECT count(*), sum(is_late::INTEGER) FROM mart_orders WHERE is_delivered")
    outcomes = _df(con, "SELECT outcome, count(*) AS n FROM mart_orders GROUP BY 1 ORDER BY n DESC")

    summary = {
        "critical": sum(f["severity"] == "critical" for f in a.findings),
        "warning": sum(f["severity"] == "warning" for f in a.findings),
        "expected": sum(f["severity"] == "expected" for f in a.findings),
        "ok": sum(f["severity"] == "ok" for f in a.findings),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_fingerprint": _q(con, "SELECT value FROM rawdb.meta_run WHERE key = 'data_fingerprint'")[0],
        "row_counts": n,
        "summary": summary,
        "findings": a.findings,
        "missing_values": missing,
        "statuses": statuses.to_dict(orient="records"),
        "coverage": coverage.astype(str).to_dict(orient="records"),
        "monthly": monthly.to_dict(orient="records"),
        "outcomes": outcomes.to_dict(orient="records"),
        "ml": {
            "exclusions": excl.to_dict(orient="records"),
            "eligible": int(elig[0]),
            "eligible_late_rate": float(elig[1]),
            "delivered_cohort": int(delivered[0]),
            "delivered_late": int(delivered[1]),
        },
    }


def _fmt_share(x) -> str:
    return "" if x is None else f"{100 * x:.3f}%"


def render_markdown(r: dict) -> str:
    lines = [
        "# Аудит качества данных OrderPulse",
        "",
        f"Сгенерировано командой `python -m orderpulse quality` · {r['generated_at']} · "
        f"fingerprint данных `{r['data_fingerprint']}`.",
        "",
        "Отчёт создаётся автоматически; ручные правки будут перезаписаны. "
        "Решения по правилам — в `reports/decisions.md`.",
        "",
        "## Итог",
        "",
        f"- Критических проблем: **{r['summary']['critical']}**",
        f"- Предупреждений: **{r['summary']['warning']}**",
        f"- Ожидаемых особенностей данных: **{r['summary']['expected']}**",
        f"- Проверок без замечаний: {r['summary']['ok']}",
        "",
        "## Размеры таблиц",
        "",
        "| Таблица | Строк |", "|---|---:|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in r["row_counts"].items()]
    sev_ru = {"critical": "🔴 критично", "warning": "🟡 предупреждение", "expected": "🔵 ожидаемо", "ok": "🟢 ok"}
    sections = []
    for f in r["findings"]:
        if f["section"] not in sections:
            sections.append(f["section"])
    titles = {"keys": "Ключи и гранулярность", "referential": "Ссылочная целостность", "typing": "Типизация",
              "amounts": "Суммы", "temporal": "Временная последовательность", "geolocation": "Геолокация",
              "coverage": "Покрытие и срез данных"}
    for s in sections:
        lines += ["", f"## {titles.get(s, s)}", "", "| Проверка | Статус | Число | Доля | Действие / примечание |",
                  "|---|---|---:|---:|---|"]
        for f in r["findings"]:
            if f["section"] != s:
                continue
            note = "; ".join(x for x in [f["action"], f["note"]] if x)
            lines.append(f"| {f['check']} | {sev_ru[f['severity']]} | {f['count']:,} | {_fmt_share(f['share'])} | {note} |")
    lines += ["", "## Статусы заказов", "", "| Статус | Заказов | С подтверждением | С передачей перевозчику | С доставкой | С обещанной датой |",
              "|---|---:|---:|---:|---:|---:|"]
    for s in r["statuses"]:
        lines.append(f"| {s['order_status']} | {s['n']:,} | {s['has_approved']:,} | {s['has_carrier']:,} | "
                     f"{s['has_delivered']:,} | {s['has_estimated']:,} |")
    lines += ["", "## Исходы (mart_orders.outcome)", "", "| Исход | Заказов |", "|---|---:|"]
    lines += [f"| {o['outcome']} | {o['n']:,} |" for o in r["outcomes"]]
    lines += ["", "## Временное покрытие", "", "| Поле | Минимум | Максимум |", "|---|---|---|"]
    lines += [f"| {c['field']} | {c['min']} | {c['max']} |" for c in r["coverage"]]
    lines += ["", "### По месяцам покупки", "",
              "Доля задержек = опоздавшие / доставленные с обеими датами (когорта по месяцу покупки).", "",
              "| Месяц | Заказов | Доставлено | Опоздало | Доля задержек |", "|---|---:|---:|---:|---:|"]
    for m in r["monthly"]:
        lr = "" if m["late_rate"] is None or m["late_rate"] != m["late_rate"] else f"{100 * m['late_rate']:.1f}%"
        lines.append(f"| {m['month']} | {m['orders']:,} | {m['delivered']:,} | {m['late']:,} | {lr} |")
    lines += ["", "## Пропуски в исходных CSV (NULL или пустая строка)", ""]
    for table, cols in r["missing_values"].items():
        if cols:
            lines.append(f"- **{table}**: " + ", ".join(f"`{k}` {v:,}" for k, v in cols.items()))
    ml = r["ml"]
    lines += ["", "## Пригодность для ML", "",
              "Постановка: прогноз опоздания на момент `order_approved_at`; обучающая популяция — "
              "подтверждённые доставленные заказы с обещанной датой, без критических временных ошибок, с позициями.", "",
              f"- Доставленная когорта (status=delivered, есть дата доставки и обещанная дата): **{ml['delivered_cohort']:,}**, "
              f"из них опоздали **{ml['delivered_late']:,}** ({100 * ml['delivered_late'] / ml['delivered_cohort']:.2f}%).",
              f"- Пригодно для ML (до временного разбиения): **{ml['eligible']:,}**, доля задержек "
              f"{100 * ml['eligible_late_rate']:.2f}%.", "",
              "| Причина | Заказов | Доля |", "|---|---:|---:|"]
    lines += [f"| {e['reason']} | {e['n']:,} | {100 * e['share']:.2f}% |" for e in ml["exclusions"]]
    lines += ["", "Исключения по временным периодам и доступности меток фиксируются при обучении "
              "(`artifacts/model_metadata.json`, `reports/model_card.md`).", ""]
    return "\n".join(lines)


def run_quality(cfg: Config) -> dict:
    with timed("quality"):
        check_contract(cfg)  # raises SchemaError (critical) on missing files/columns
        con = duckdb.connect()
        try:
            attach_raw(con, cfg.raw_db)
            params = sql_params(cfg)
            for layer in LAYERS[:2]:  # staging + marts, in memory
                run_sql_dir(con, cfg.sql_dir / layer, params)
            report = _audit(con, cfg)
        finally:
            con.close()
        write_json_atomic(report, cfg.reports_dir / "data_quality.json")
        write_text_atomic(render_markdown(report), cfg.reports_dir / "data_quality.md")
        LOG.info("quality summary: %s", report["summary"])
        if report["summary"]["critical"]:
            crit = [f["check"] for f in report["findings"] if f["severity"] == "critical"]
            raise QualityError(f"critical data-quality findings: {crit}")
        return report
