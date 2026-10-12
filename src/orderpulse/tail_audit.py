"""Audit of the end of the dataset: what is observed, what is hypothesised.

Writes reports/dataset_tail_audit.md and experiments/reliability_v2/tail_audit.json.
All numbers are computed from the warehouse; hypotheses are labelled as such.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from .config import Config
from .utils import LOG, timed, write_json_atomic, write_text_atomic

REVIEW_LAST_CANDIDATE = "2018-08-31"


def _df(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def _md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and v is not None and not (isinstance(v, float) and math.isnan(v)):
                cells.append(fmt[c].format(v))
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compute(con, cutoff: str, test_start: str, test_end: str) -> dict:
    out = {}
    out["max_dates"] = _df(con, """
        SELECT 'покупка' AS "поле", max(purchase_ts)::VARCHAR AS "максимум" FROM mart_orders
        UNION ALL SELECT 'подтверждение', max(approved_ts)::VARCHAR FROM mart_orders
        UNION ALL SELECT 'передача перевозчику', max(carrier_ts)::VARCHAR FROM mart_orders
        UNION ALL SELECT 'фактическая доставка', max(delivered_ts)::VARCHAR FROM mart_orders
        UNION ALL SELECT 'обещанная дата', max(estimated_ts)::VARCHAR FROM mart_orders
        UNION ALL SELECT 'создание отзыва', max(review_creation_ts)::VARCHAR FROM stg_order_reviews
        UNION ALL SELECT 'ответ на отзыв', max(review_answer_ts)::VARCHAR FROM stg_order_reviews""")
    out["weekly_flows"] = _df(con, """
        WITH w AS (SELECT range::DATE AS week FROM range(DATE '2018-06-04', DATE '2018-10-22', INTERVAL 7 DAY)),
        ev AS (
            SELECT date_trunc('week', purchase_ts)::DATE AS week, 'purchases' AS kind FROM mart_orders
            UNION ALL SELECT date_trunc('week', approved_ts)::DATE, 'approvals' FROM mart_orders WHERE approved_ts IS NOT NULL
            UNION ALL SELECT date_trunc('week', carrier_ts)::DATE, 'carrier' FROM mart_orders WHERE carrier_ts IS NOT NULL
            UNION ALL SELECT date_trunc('week', delivered_ts)::DATE, 'deliveries' FROM mart_orders WHERE delivered_ts IS NOT NULL
            UNION ALL SELECT date_trunc('week', review_creation_ts)::DATE, 'reviews' FROM stg_order_reviews)
        SELECT strftime(w.week, '%Y-%m-%d') AS "неделя (пн)",
               count(*) FILTER (WHERE kind = 'purchases')  AS "покупки",
               count(*) FILTER (WHERE kind = 'approvals')  AS "подтверждения",
               count(*) FILTER (WHERE kind = 'carrier')    AS "перевозчику",
               count(*) FILTER (WHERE kind = 'deliveries') AS "доставки",
               count(*) FILTER (WHERE kind = 'reviews')    AS "отзывы создано"
        FROM w LEFT JOIN ev USING (week) GROUP BY 1 ORDER BY 1""")
    out["purchase_cohorts"] = _df(con, """
        SELECT strftime(date_trunc('week', purchase_ts), '%Y-%m-%d') AS "неделя покупки",
               count(*) AS "заказов",
               round(avg((delivered_ts IS NOT NULL)::INT), 4) AS "доля с датой доставки",
               count(*) FILTER (WHERE order_status = 'delivered') AS "delivered",
               count(*) FILTER (WHERE order_status = 'shipped') AS "shipped",
               count(*) FILTER (WHERE order_status IN ('canceled', 'unavailable')) AS "canceled+unavailable",
               count(*) FILTER (WHERE order_status NOT IN ('delivered', 'shipped', 'canceled', 'unavailable')) AS "прочие",
               round(avg(has_review::INT), 4) AS "покрытие отзывами",
               round(avg(is_late::INT), 4) AS "доля задержек (доставл.)",
               round(median(epoch(estimated_ts - purchase_ts) / 86400), 1) AS "медиана обещ. срока, дн.",
               round(median(delivery_days), 1) AS "медиана доставки, дн.",
               round(quantile_cont(delivery_days, 0.9), 1) AS "P90 доставки, дн."
        FROM mart_orders WHERE purchase_ts >= '2018-05-28' GROUP BY 1 ORDER BY 1""")
    out["estimated_cohorts"] = _df(con, """
        SELECT strftime(date_trunc('week', estimated_ts), '%Y-%m-%d') AS "неделя обещанной даты",
               count(*) AS "заказов",
               count(*) FILTER (WHERE is_delivered) AS "доставлено",
               count(*) FILTER (WHERE is_late) AS "опоздало",
               round(avg(is_late::INT), 4) AS "доля задержек",
               round(avg(has_review::INT), 4) AS "покрытие отзывами"
        FROM mart_orders WHERE estimated_ts >= '2018-07-02' AND approved_ts IS NOT NULL
          AND order_status NOT IN ('canceled', 'unavailable')
        GROUP BY 1 ORDER BY 1""")
    ref = con.execute(f"""
        SELECT avg(is_late::INT) FROM mart_orders
        WHERE is_delivered AND estimated_ts >= '2018-06-01' AND estimated_ts < '{REVIEW_LAST_CANDIDATE}'""").fetchone()[0]
    tail = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE is_late) FROM mart_orders
        WHERE is_delivered AND estimated_ts >= '{REVIEW_LAST_CANDIDATE}'""").fetchone()
    out["tail_expectation"] = {
        "reference_late_rate_est_2018_06_01_to_08_30": float(ref),
        "delivered_with_estimated_on_or_after_2018_08_31": int(tail[0]),
        "observed_late": int(tail[1]),
        "expected_late_at_reference_rate": float(tail[0] * ref),
        "binomial_p_zero_late_at_reference_rate": float((1 - ref) ** tail[0]),
    }
    out["review_timing"] = _df(con, """
        SELECT CASE WHEN is_late THEN 'опоздал' ELSE 'вовремя' END AS "группа",
               count(*) AS "заказов",
               round(median(epoch(review_creation_ts - delivered_ts) / 86400), 2) AS "медиана: отзыв − доставка, дн.",
               round(median(epoch(review_creation_ts - estimated_ts) / 86400), 2) AS "медиана: отзыв − обещанная дата, дн.",
               round(avg((review_creation_ts < delivered_ts)::INT), 4) AS "доля отзывов до доставки"
        FROM mart_orders
        WHERE is_delivered AND has_review AND purchase_ts >= '2017-06-01' AND purchase_ts < '2018-06-01'
        GROUP BY 1 ORDER BY 1""")
    out["after_cutoff_deliveries"] = _df(con, f"""
        SELECT order_status AS "статус", count(*) AS "доставок после {cutoff}",
               count(*) FILTER (WHERE is_late) AS "из них опоздали",
               count(*) FILTER (WHERE review_creation_ts < delivered_ts) AS "отзыв создан до доставки",
               max(review_creation_ts)::DATE::VARCHAR AS "макс. дата отзыва",
               max(estimated_ts)::DATE::VARCHAR AS "макс. обещанная дата",
               min(purchase_ts)::DATE::VARCHAR AS "мин. покупка", max(purchase_ts)::DATE::VARCHAR AS "макс. покупка"
        FROM mart_orders WHERE delivered_ts >= TIMESTAMP '{cutoff}' GROUP BY 1""")
    out["test_exposure"] = _df(con, f"""
        SELECT strftime(date_trunc('month', approved_ts), '%Y-%m') AS "месяц подтверждения",
               count(*) AS "подтверждено",
               count(*) FILTER (WHERE ml_exclusion_reason = 'not_delivered') AS "не доставлено",
               count(*) FILTER (WHERE estimated_ts >= '{REVIEW_LAST_CANDIDATE}') AS "обещано ≥ 2018-08-31",
               count(*) FILTER (WHERE delivered_ts >= TIMESTAMP '{cutoff}') AS "доставлено ≥ cutoff",
               round(avg((estimated_ts + INTERVAL 31 DAY <= TIMESTAMP '{cutoff}')::INT), 4) AS "доля: обещ.+31 дн. ≤ cutoff"
        FROM mart_orders WHERE approved_ts >= '2018-01-01' AND approved_ts < '2018-09-01'
        GROUP BY 1 ORDER BY 1""")
    return out


def render(r: dict, cutoff: str, test_start: str, test_end: str) -> str:
    te = r["tail_expectation"]
    pct = "{:.1%}"
    L = [
        "# Аудит конца датасета Olist",
        "",
        f"Сгенерировано `python -m orderpulse tail-audit` · {r['generated_at']}. Числа рассчитаны по "
        "`warehouse.duckdb`; гипотезы отделены от фактов.",
        "",
        "Проверяемое утверждение MVP v1: «набор обрывается около 2018-09-03, поэтому поздние доставки не попали». "
        "Ниже показано, что **обрыв потоков наблюдается**, но механизм «поздние доставки не попали» — "
        "**гипотеза**, а не установленный факт.",
        "",
        "## 1. Наблюдаемые факты",
        "",
        "### Максимальные даты",
        "",
        _md_table(r["max_dates"]),
        "",
        "### Потоки событий по неделям",
        "",
        _md_table(r["weekly_flows"]),
        "",
        "Покупки падают уже с недели 2018-08-20 и почти исчезают с 2018-08-27. Доставки и создание отзывов "
        "идут в обычном объёме по неделю 2018-08-27 включительно, затем обрываются. Новых отзывов после "
        "2018-08-31 нет, хотя ответы на отзывы продолжаются до 2018-10-29. Отдельные доставки — до 2018-10-17.",
        "",
        "### Когорты по неделе покупки",
        "",
        _md_table(r["purchase_cohorts"], {"доля с датой доставки": pct, "покрытие отзывами": pct,
                                          "доля задержек (доставл.)": pct}),
        "",
        "В последних полных неделях покупок доля заказов без даты доставки и доля `shipped` **не растут**, "
        "хотя медиана доставки ~10 дней, а P90 — более 20 дней. Если бы выгрузка была мгновенным снимком "
        "около 2018-09-03, у заказов конца августа ожидалось бы много незавершённых доставок.",
        "",
        "### Когорты по неделе обещанной даты",
        "",
        _md_table(r["estimated_cohorts"], {"доля задержек": pct, "покрытие отзывами": pct}),
        "",
        f"- Доля задержек среди доставленных заказов с обещанной датой 2018-06-01 … 2018-08-30: "
        f"**{te['reference_late_rate_est_2018_06_01_to_08_30']:.2%}**.",
        f"- Доставленных заказов с обещанной датой ≥ 2018-08-31: **{te['delivered_with_estimated_on_or_after_2018_08_31']}**, "
        f"опоздавших среди них: **{te['observed_late']}**; при эталонной доле ожидалось бы около "
        f"**{te['expected_late_at_reference_rate']:.0f}**. Вероятность получить 0 при независимых исходах с эталонной "
        f"долей ≈ {te['binomial_p_zero_late_at_reference_rate']:.1e}. Это не тест механизма: эталонная доля условна, "
        "а исходы не независимы. Но такой ноль вряд ли объясняется случайностью.",
        "",
        "### Когда создаётся отзыв (заказы, купленные 2017-06 … 2018-05)",
        "",
        _md_table(r["review_timing"], {"доля отзывов до доставки": pct}),
        "",
        "У опоздавших заказов отзыв обычно создаётся примерно через 2 дня после обещанной даты, ещё до доставки. "
        "У своевременных — вскоре после доставки.",
        "",
        "### Доставки на/после cutoff",
        "",
        _md_table(r["after_cutoff_deliveries"]),
        "",
        "Поздние доставки после cutoff **в наборе есть**, и все они опоздавшие. У этих заказов обещанная дата "
        "не позже конца августа, а отзыв создан до доставки. Поэтому формулировка MVP v1 «поздние доставки "
        "не попали» в общем виде **неточна**. Наблюдается более узкое явление: нет ни одного опоздания среди "
        "заказов с обещанной датой ≥ 2018-08-31.",
        "",
        "### Экспозиция исходного test и соседних месяцев",
        "",
        _md_table(r["test_exposure"], {"доля: обещ.+31 дн. ≤ cutoff": pct}),
        "",
        f"Исходный test: подтверждения {test_start} … {test_end} (не включая). В нём почти нет заказов с обещанной "
        "датой ≥ 2018-08-31 и доставкой после cutoff; у июля–августа такие заказы составляют заметную долю.",
        "",
        "## 2. Гипотезы о механизме (не доказаны)",
        "",
        "- **H1. Выгрузка около начала сентября 2018 (мгновенный снимок).** Согласуется с обрывом покупок, доставок и отзывов. "
        "Но тогда у заказов конца августа ожидалась бы заметная доля незавершённых (`shipped`) — её нет. "
        "Кроме того, отдельные доставки идут до 2018-10-17, а ответы на отзывы — до 2018-10-29. "
        "Чистый снимок H1 этого не объясняет.",
        "- **H2. Отбор заказов по наличию отзыва к ~2018-08-31.** Покрытие отзывами ≈ 99% во всех месяцах. "
        "Отзыв опоздавшего заказа появляется после обещанной даты. Тогда заказ с обещанной датой после "
        "~2018-08-29 мог попасть в набор только при доставке и отзыве до 31 августа, то есть только вовремя. "
        "Это объясняет 0 опозданий в хвосте и малую долю незавершённых заказов в августе. "
        "H2 согласуется с наблюдениями лучше H1, но **не доказана**: документации источника о критериях отбора нет, "
        "и **механизм отбора остаётся неизвестным**.",
        "- **H3. Реальное улучшение доставки в сентябре 2018.** Маловероятно: эффект слишком резкий и совпадает "
        "с концом потоков данных.",
        "",
        "## 3. Консервативное решение о test",
        "",
        f"- Исходный test ({test_start} … {test_end}) **не меняется** и не сдвигается на хвост ради метрик.",
        "- Июль–август 2018 **не годятся** как новый holdout: исходы там, вероятно, неполны и смещены к "
        "своевременным доставкам (H2) или незавершены (H1). При обоих механизмах доля задержек занижена.",
        f"- `data_cutoff = {cutoff}` остаётся консервативной границей наблюдаемости исходов. 4 test-заказа "
        "(все опоздавшие), доставленные после неё, исключены правилом доступности меток; влияние на метрики "
        "оценено в reliability_report.",
        "- Метрики test v1 описаны как оценка на **ранее просмотренном** holdout.",
        "",
        "## 4. Что невозможно доказать без дополнительных сведений",
        "",
        "- Точная дата и способ выгрузки, критерии включения заказов в публичный набор.",
        "- Были ли исключены заказы без отзыва, а также отменённые и потерянные заказы после определённой даты.",
        "- Полнота исходов в test-периоде: косвенно она высокая (доставки ≥ cutoff единичны), но прямой проверки нет.",
        "- Доля заказов без отзыва **среди включённых** (менее 1%) ничего не говорит о том, сколько заказов исходной "
        "генеральной совокупности не попало в набор. Число отсутствующих заказов из этих данных оценить нельзя.",
        "",
    ]
    return "\n".join(L)


def tail_audit(cfg: Config, out_dir: Path) -> dict:
    with timed("tail_audit"):
        cutoff = str(cfg.data["data_cutoff"])
        t = cfg.model["splits"]["test"]
        con = duckdb.connect(str(cfg.warehouse_db), read_only=True)
        try:
            r = compute(con, cutoff, t["start"], t["end"])
        finally:
            con.close()
        r["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_text_atomic(render(r, cutoff, t["start"], t["end"]), cfg.reports_dir / "dataset_tail_audit.md")
        js = {k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v) for k, v in r.items()}
        write_json_atomic(js, out_dir / "tail_audit.json")
        LOG.info("tail audit: %s", r["tail_expectation"])
        return js
