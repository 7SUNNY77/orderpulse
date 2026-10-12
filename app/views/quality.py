from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (COLORS, MODEL_COLORS, artifact_json, cfg, experiment_csv, fmt_int, fmt_pct, model_versions_consistent, note,
                    plotly_layout, report_json, require_warehouse)

from orderpulse.features import FEATURE_LABELS
from orderpulse.models import MODEL_NAMES, MODEL_TITLES

require_warehouse()
st.title("Качество модели и данных")

meta = artifact_json("model_metadata.json")
metrics = artifact_json("metrics.json")
dq = report_json("data_quality.json")
bc = report_json("build_checks.json")

tab_model, tab_data, tab_limits = st.tabs(["Модель", "Данные", "Ограничения"])

# ============================================================================ model
with tab_model:
    ok, msg = model_versions_consistent()
    if meta is None or metrics is None:
        st.info("Артефакты модели отсутствуют. Выполните `python -m orderpulse train` и `python -m orderpulse evaluate`.")
    else:
        if not ok:
            st.warning(msg)
        st.markdown(
            f"**Постановка.** Риск опоздания (доставка позже обещанного календарного дня) на момент "
            f"`order_approved_at`. Модель обучена {meta['trained_at']} · данные `{meta['data_fingerprint']}` · "
            f"признаки `{meta['feature_version']}` ({len(meta['features'])} шт.) · seed {meta['seed']}.")
        sp = pd.DataFrame([{
            "Выборка": n, "Подтверждение с": s["start"], "по (не вкл.)": s["end_exclusive"],
            "Метки известны до": s["label_available_before"], "Подтверждено": s["approved_in_period"],
            "Пригодно": s["eligible"], "Исключено: метка ещё не известна": s["excluded_label_not_yet_available"],
            "Используется": s["used"], "Опозданий": s["positives"], "Доля задержек": s["late_rate"]}
            for n, s in metrics["splits"].items()])
        st.dataframe(sp, hide_index=True, width="stretch",
                     column_config={"Доля задержек": st.column_config.NumberColumn(format="percent")})
        note("Периоды следуют по времени; между ними месячные зазоры (embargo), чтобы метки обучающей выборки "
             "были известны к началу следующего периода. Test не использовался для выбора признаков, "
             "параметров или порога.")

        split = st.radio("Выборка для метрик", ["test", "validation"], horizontal=True,
                         format_func=lambda s: "test (ранее просмотренный holdout)" if s == "test" else
                         "validation (использована для настройки — оптимистично)")
        res = metrics[split]
        dk_ui = int(cfg().model["ui"].get("default_queue_k", metrics["policy"]["daily_k"]))
        dk = dk_ui if dk_ui in metrics["policy"]["daily_k_list"] else int(metrics["policy"]["daily_k"])
        rows = []
        for n in MODEL_NAMES:
            r = res[n]
            daily = next(x for x in r["topk_daily"] if x["k"] == dk)
            tp = r["threshold_policy"] or {}
            rows.append({
                "Модель": MODEL_TITLES[n], "AP": r["ap"], "ROC-AUC": r["roc_auc"], "Brier": r["brier"],
                f"Precision@{dk}/день": daily["micro_precision"], f"Recall@{dk}/день": daily["micro_recall"],
                f"Lift@{dk}/день": daily["lift"],
                "Порог: precision": tp.get("precision"), "Порог: recall": tp.get("recall"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in
            ["AP", "ROC-AUC", "Brier", f"Precision@{dk}/день", f"Recall@{dk}/день", f"Lift@{dk}/день",
             "Порог: precision", "Порог: recall"]})
        base = res["catboost"]["base_rate"]
        note(f"n = {fmt_int(res['catboost']['n'])}, опозданий {fmt_int(res['catboost']['positives'])} "
             f"(доля {fmt_pct(base)}; это AP случайного ранжирования). AP — average_precision_score, не "
             f"трапецеидальная PR-AUC. Top-{dk}/день: в каждом дне подтверждения выбираются {dk} заказов с "
             f"наибольшим скором (в днях с < {dk} заказами — все); precision и recall — микро-усреднение "
             f"(сумма найденных / сумма выбранных или опоздавших); lift — к ожиданию случайного отбора тех же k_d заказов в каждом дне. "
             f"Порог: выбран на validation так, чтобы доля проверяемых = {fmt_pct(metrics['policy']['review_share'])} "
             f"({metrics['policy']['daily_k']} в день / средний дневной объём validation; политика порога MVP v1, не связана с K очереди). Brier для правила не применим (это не вероятность).")

        c1, c2 = st.columns(2)
        fig = go.Figure()
        for n in MODEL_NAMES:
            pts = res[n].get("pr_curve") or []
            if pts and n != "constant":
                fig.add_scatter(x=[p["recall"] for p in pts], y=[p["precision"] for p in pts], mode="lines",
                                name=f"{MODEL_TITLES[n]} (AP {res[n]['ap']:.3f})",
                                line=dict(color=MODEL_COLORS[n], width=2))
        fig.add_hline(y=base, line_dash="dot", line_color=COLORS["neutral"], annotation_text=f"случайный отбор {base:.1%}")
        fig.update_layout(title=f"Precision–Recall ({split})", xaxis_title="Recall", yaxis_title="Precision",
                          yaxis_tickformat=".0%", xaxis_tickformat=".0%")
        c1.plotly_chart(plotly_layout(fig, 380), width="stretch")

        tk = pd.DataFrame([{"Модель": MODEL_TITLES[n], **x} for n in MODEL_NAMES for x in res[n]["topk_overall"]])
        fig = go.Figure()
        for n in MODEL_NAMES:
            g = tk[tk["Модель"] == MODEL_TITLES[n]]
            fig.add_scatter(x=g["k"].astype(str), y=g["precision_at_k"], mode="lines+markers", name=MODEL_TITLES[n],
                            line=dict(color=MODEL_COLORS[n], width=2), marker=dict(size=8))
        fig.update_layout(title=f"Precision@K по всему периоду {split}", xaxis_title="K", yaxis_tickformat=".0%")
        c2.plotly_chart(plotly_layout(fig, 380), width="stretch")
        st.dataframe(tk.rename(columns={"k": "K", "selected": "Выбрано", "found_late": "Найдено опозданий",
                                        "precision_at_k": "Precision@K", "recall_at_k": "Recall@K", "lift": "Lift"}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Precision@K", "Recall@K", "Lift"]})

        st.markdown(f"**Top-K по дневным когортам подтверждения ({split})**")
        dd = pd.DataFrame([{"Модель": MODEL_TITLES[n], **x} for n in MODEL_NAMES for x in res[n]["topk_daily"]])
        dd = dd[["Модель", "k", "days", "days_with_fewer_than_k", "days_without_late", "selected", "found_late",
                 "total_late", "micro_precision", "micro_recall", "macro_precision", "macro_recall", "lift"]]
        st.dataframe(dd.rename(columns={"k": "K/день", "days": "Дней", "days_with_fewer_than_k": "Дней < K заказов",
                                        "days_without_late": "Дней без опозданий", "selected": "Выбрано",
                                        "found_late": "Найдено", "total_late": "Всего опозданий"}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                    ["micro_precision", "micro_recall", "macro_precision", "macro_recall", "lift"]})
        note("macro recall усредняется только по дням, где было хотя бы одно опоздание.")

        if split == "test" and metrics.get("test_daily"):
            td = pd.DataFrame(metrics["test_daily"])
            td["day"] = pd.to_datetime(td["day"])
            td["late_rate"] = td["late"] / td["n"]
            fig = go.Figure()
            for n in ["logreg", "catboost", "rule_short_promise"]:
                fig.add_scatter(x=td["day"], y=td[f"precision_{n}"], mode="lines", name=f"Precision@{metrics['policy']['daily_k']}: {MODEL_TITLES[n]}",
                                line=dict(color=MODEL_COLORS[n], width=2))
            fig.add_scatter(x=td["day"], y=td["late_rate"], mode="lines", name="Доля задержек дня",
                            line=dict(color=COLORS["neutral"], width=2, dash="dot"))
            fig.update_layout(title=f"По дням test: Precision@{metrics['policy']['daily_k']} и доля задержек", yaxis_tickformat=".0%")
            st.plotly_chart(plotly_layout(fig, 340), width="stretch")

        c1, c2 = st.columns(2)
        fig = go.Figure()
        mx = 0.0
        for n in ["logreg", "catboost"]:
            cal = pd.DataFrame(res[n].get("calibration") or [])
            if not cal.empty:
                mx = max(mx, cal["mean_score"].max(), cal["observed"].max())
                fig.add_scatter(x=cal["mean_score"], y=cal["observed"], mode="lines+markers", name=MODEL_TITLES[n],
                                line=dict(color=MODEL_COLORS[n], width=2), marker=dict(size=8),
                                customdata=cal["n"], hovertemplate="скор %{x:.3f}<br>факт %{y:.3f}<br>n=%{customdata}<extra></extra>")
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="идеальная калибровка",
                        line=dict(color=COLORS["neutral"], dash="dot", width=1))
        fig.update_layout(title=f"Калибровка ({split}), 10 квантильных корзин", xaxis_title="средний скор",
                          yaxis_title="фактическая доля задержек")
        c1.plotly_chart(plotly_layout(fig, 380), width="stretch")
        cb = res["logreg"]
        with c2:
            st.markdown("**Как читать риск-скор**")
            st.markdown(
                f"- Средний скор Logistic Regression (основная модель) на {split}: **{cb['mean_score']:.3f}**, фактическая доля задержек: "
                f"**{cb['base_rate']:.3f}**; Brier {cb['brier']:.4f} (константа: {res['constant']['brier']:.4f}).\n"
                "- Калибратор не обучался (test для него использовать нельзя, а доля задержек заметно меняется "
                "между периодами). Поэтому в интерфейсе — **риск-скор для ранжирования**, а не точная вероятность.")
            imp = pd.DataFrame(metrics.get("catboost_global_importance", [])).head(12)
            if not imp.empty:
                imp["Признак"] = imp["feature"].map(FEATURE_LABELS).fillna(imp["feature"])
                fig = go.Figure(go.Bar(x=imp["importance"][::-1], y=imp["Признак"][::-1], orientation="h",
                                       marker_color=COLORS["on_time"]))
                fig.update_layout(title="Глобальная важность CatBoost (PredictionValuesChange) — не локальное объяснение")
                st.plotly_chart(plotly_layout(fig, 360), width="stretch")

        st.markdown("### Размер очереди (K) и случайный baseline")
        dev_t, test_t = experiment_csv("topk_dev.csv"), experiment_csv("topk_test_v1.csv")
        if dev_t is None or test_t is None:
            st.info("Этап надёжности не запускался: `python -m orderpulse reliability`.")
        else:
            src = st.radio("Период", ["development (до test, 3 окна)", "исходный test (уже просмотрен)"],
                           horizontal=True, key="kq_period")
            tk_ = dev_t[dev_t["period"] == "dev_pooled"] if src.startswith("development") else test_t
            kk = st.select_slider("K в день", options=sorted(tk_["K"].unique()), value=20, key="kq_k")
            cur = tk_[(tk_["K"] == kk) & (tk_["model"].isin(["catboost", "logreg", "rule_short_promise"]))]
            cur = cur.set_index("model").loc[["logreg", "catboost", "rule_short_promise"]].reset_index()
            first = cur.iloc[0]
            tbl = pd.DataFrame({
                "Модель": [MODEL_TITLES[m] for m in cur["model"]],
                "Найдено задержек в день": cur["model_found_per_day"].round(2),
                "Precision@K (micro)": cur["model_micro_precision"],
                "Recall@K (micro)": cur["model_micro_recall"],
                "Precision@K (macro)": cur["model_macro_precision"],
                "Recall@K (macro)": cur["model_macro_recall"],
                "Lift (micro precision / random)": cur["lift_micro_precision"].round(2),
            })
            rnd = pd.DataFrame([{"Модель": "Случайный отбор (ожидание)",
                                 "Найдено задержек в день": round(first["random_found_per_day"], 2),
                                 "Precision@K (micro)": first["random_micro_precision"],
                                 "Recall@K (micro)": first["random_micro_recall"],
                                 "Precision@K (macro)": first["random_macro_precision"],
                                 "Recall@K (macro)": first["random_macro_recall"],
                                 "Lift (micro precision / random)": 1.0}])
            st.dataframe(pd.concat([tbl, rnd], ignore_index=True), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                        ["Precision@K (micro)", "Recall@K (micro)", "Precision@K (macro)", "Recall@K (macro)"]})
            note(f"Проверок в день: {first['model_reviewed_per_day']:.1f} ({fmt_pct(first['model_review_share'])} заказов); "
                 f"дней: {int(first['model_days'])}, без опозданий: {int(first['model_days_without_late'])}, "
                 f"с числом заказов < K: {int(first['model_days_n_lt_K'])}. Найденная задержка не означает предотвращённую.")
            fig = go.Figure()
            for m in ["logreg", "catboost", "rule_short_promise"]:
                g = tk_[tk_["model"] == m].sort_values("K")
                fig.add_scatter(x=g["model_reviewed_per_day"], y=g["model_found_per_day"], mode="lines+markers",
                                name=MODEL_TITLES[m], line=dict(color=MODEL_COLORS[m], width=2), marker=dict(size=8),
                                customdata=g["K"], hovertemplate="K=%{customdata}: %{y:.2f} в день<extra></extra>")
            g = tk_[tk_["model"] == "logreg"].sort_values("K")
            fig.add_scatter(x=g["model_reviewed_per_day"], y=g["random_found_per_day"], mode="lines",
                            name="Случайный отбор (ожидание)", line=dict(color=COLORS["neutral"], dash="dot", width=2))
            fig.update_layout(title="Найденные задержки в день vs число проверок в день (K = 5, 10, 20, 50, 100)",
                              xaxis_title="проверок в день", yaxis_title="найдено задержек в день")
            st.plotly_chart(plotly_layout(fig, 340), width="stretch")
            with st.expander("Методология top-K и baseline"):
                st.markdown(
                    "- Когорта дня d — доставленные заказы с меткой, подтверждённые в этот календарный день "
                    "(локальные метки времени набора). n_d — заказов, m_d — опозданий, k_d = min(K, n_d), "
                    "h_d — опозданий среди выбранных.\n"
                    "- **micro**: Σh_d/Σk_d и Σh_d/Σm_d; **macro**: среднее h_d/k_d по всем дням и среднее h_d/m_d "
                    "по дням с m_d > 0 (дни без опозданий не считаются нулём).\n"
                    "- Одинаковые скоры: берётся ожидание при случайном разрешении (не зависит от порядка строк).\n"
                    "- Случайный baseline: E[h_d] = k_d·m_d/n_d, агрегируется теми же формулами; проверен Monte Carlo "
                    "(500 повторений). Lift — отношение одной и той же метрики модели и случайного отбора.\n"
                    "- Development: 3 окна (окт.–нояб. 2017, дек. 2017–янв. 2018, февр.–март 2018), модели обучены "
                    "заново до каждого окна. Test v1 — фиксированные прогнозы MVP v1, период уже просмотрен.\n"
                    "- Подробности, bootstrap-интервалы и ablation: `reports/reliability_report.md`.")

        st.markdown("**Ограниченный анализ ошибок: AP по подгруппам**")
        sg = []
        for n in ["logreg", "catboost", "rule_short_promise"]:
            for g, v in res[n].get("subgroups", {}).items():
                sg.append({"Подгруппа": g, "Модель": MODEL_TITLES[n], "n": v["n"], "Опозданий": v["positives"],
                           "Доля задержек": v["base_rate"], "AP": v["ap"], "ROC-AUC": v["roc_auc"]})
        if sg:
            sgd = pd.DataFrame(sg).pivot_table(index=["Подгруппа", "n", "Опозданий", "Доля задержек"], columns="Модель",
                                               values="AP").reset_index()
            st.dataframe(sgd, hide_index=True, width="stretch",
                         column_config={"Доля задержек": st.column_config.NumberColumn(format="percent")})
            note("AP в подгруппе сравнивайте с долей задержек этой подгруппы (уровень случайного ранжирования).")

        with st.expander("Параметры и версии"):
            st.json({"model_versions": metrics["model_versions"], "catboost": meta["catboost"],
                     "logreg": meta["logreg_params"], "policy": meta["policy"], "features": meta["features"],
                     "environment": meta["environment"]})

# ============================================================================ data
with tab_data:
    if dq is None:
        st.info("Отчёт о качестве данных отсутствует: выполните `python -m orderpulse quality`.")
    else:
        s = dq["summary"]
        a, b, c, d = st.columns(4)
        a.metric("Критических", s["critical"])
        b.metric("Предупреждений", s["warning"])
        c.metric("Ожидаемых особенностей", s["expected"])
        d.metric("Проверок без замечаний", s["ok"])
        note(f"Аудит: {dq['generated_at']} · fingerprint данных `{dq['data_fingerprint']}` · полный отчёт — "
             "reports/data_quality.md.")
        sev = st.multiselect("Показать", ["critical", "warning", "expected", "ok"], default=["critical", "warning", "expected"])
        fd = pd.DataFrame(dq["findings"])
        fd = fd[fd["severity"].isin(sev)][["section", "check", "severity", "count", "share", "action", "note"]]
        st.dataframe(fd, hide_index=True, width="stretch",
                     column_config={"share": st.column_config.NumberColumn(format="percent")})
        st.markdown("**Исключения из ML-постановки**")
        st.dataframe(pd.DataFrame(dq["ml"]["exclusions"]), hide_index=True, width="stretch",
                     column_config={"share": st.column_config.NumberColumn(format="percent")})
    if bc:
        st.markdown("**Проверки сборки витрин (сверка сумм, отсутствие размножения строк)**")
        st.dataframe(pd.DataFrame(bc["checks"]), hide_index=True, width="stretch")
        r = bc["payments_reconciliation"]
        st.markdown(
            f"Сверка платежей с «товары + доставка»: совпадает (±0.01) у {fmt_int(r['matched'])} из "
            f"{fmt_int(r['comparable'])} сопоставимых заказов; платежи больше — {fmt_int(r['payment_higher'])}, "
            f"меньше — {fmt_int(r['payment_lower'])}; суммарная разница R$ {r['total_difference']:,.2f}. "
            f"Без платежей: {fmt_int(r['orders_without_payment'])}, без позиций: {fmt_int(r['orders_without_items'])}. "
            "Расхождения показаны, а не скрыты (например, проценты по рассрочке).")

# ============================================================================ limitations
with tab_limits:
    st.markdown("""
- **Условность прогноза.** Модель обучена и оценена только на исторически доставленных заказах. Она не прогнозирует
  отмену, потерю или недоставку; для таких заказов риск-скор описывает только риск опоздания *если* заказ будет доставлен.
- **Историческая симуляция.** Очередь риска — исторические заказы, ранжированные по прогнозу на момент подтверждения.
  Это не текущие открытые заказы магазина; ежедневное обновление прогноза не моделируется.
- **Доступность признаков — допущение.** Считается, что состав заказа и обещанная дата известны при подтверждении.
  Событийного журнала нет. Платёжные поля и `shipping_limit_date` в модель fv1 не включены.
- **Нет причинных выводов.** Сегменты и SHAP-факторы показывают статистическую связь и поведение модели,
  а не причины задержек. Продавец не объявляется «виновным».
- **Нет оценки денег.** Стоимость товаров — не выручка и не прибыль Olist; стоимость заказов в очереди — не ожидаемый убыток.
  Эффект проверки заказов не измерялся.
- **Срез данных.** После ~2018-09-03 поток событий в наборе обрывается; последние месяцы имеют неполные исходы
  и не используются для оценки.
- **Сдвиг во времени.** Доля задержек сильно меняется между периодами (например, всплески в конце 2017 и в
  феврале–марте 2018); риск-скор не откалиброван под текущий уровень.
- **Многопродавцовые заказы.** В статистике продавцов заказ учитывается у каждого продавца; задержка приписывается всем.
- **Данные.** Brazilian E-Commerce Public Dataset by Olist, лицензия CC BY-NC-SA 4.0; коммерческое использование не подразумевается.
""")
