from __future__ import annotations

import pandas as pd
import streamlit as st
from common import (OUTCOME_RU, artifact_json, cfg, fmt_int, fmt_money, fmt_num, fmt_pct, logreg_factors,
                    model_versions_consistent, note, require_warehouse, run, select_model)

from orderpulse.models import MODEL_TITLES

require_warehouse()
st.title("Очередь риска")
st.markdown("<div class='op-banner'><b>Прогнозы на момент подтверждения: историческая симуляция.</b><br>"
            "Это исторические заказы Olist, ранжированные по риск-скору, который модель выдала бы в момент "
            "подтверждения (order_approved_at). Это не список текущих открытых заказов, и прогноз не "
            "обновляется ежедневно. Модель оценивает риск опоздания только для заказов, которые будут "
            "доставлены; отмену, потерю или недоставку она не прогнозирует.</div>", unsafe_allow_html=True)

ok, msg = model_versions_consistent()
if not ok:
    st.warning(msg)
    st.stop()

meta = artifact_json("model_metadata.json")
c1, c2, c3 = st.columns([1.4, 1.2, 0.8])
model_name = select_model(c1, "queue", "Модель ранжирования")
version = meta["models"][model_name]["version"]

dates = run("queue_dates", version).copy()
dates["d"] = pd.to_datetime(dates["d"]).dt.date
dmin, dmax = dates["d"].min(), dates["d"].max()
test_days = dates.loc[dates["split"] == "test", "d"]
default_day = test_days.min() if len(test_days) else dmax  # first approval day of the test period
period = c2.date_input("Дата подтверждения (день или диапазон)", value=(default_day, default_day),
                       min_value=dmin, max_value=dmax)
if isinstance(period, (list, tuple)) and len(period) == 2:
    d_from, d_to = period
else:
    d_from = d_to = period[0] if isinstance(period, (list, tuple)) and period else default_day
K_DEFAULT = int(cfg().model["ui"].get("default_queue_k", meta["policy"]["daily_k"]))
k = c3.number_input("Top-K в день", min_value=1, max_value=500, value=K_DEFAULT, step=5,
                    help="Сколько заказов менеджер проверяет за каждый день подтверждения. K = 20 по умолчанию — "
                         "демонстрационный компромисс по development-окнам, не доказанный оптимум.")

with st.expander("Фильтры очереди (применяются ДО выбора top-K)", expanded=False):
    fc1, fc2, fc3 = st.columns(3)
    opts = run("filter_options")
    states = fc1.multiselect("Штат покупателя", opts["states"])
    cats = fc2.multiselect("Основная категория заказа", opts["categories"])
    multi = fc3.checkbox("Только заказы с несколькими продавцами")
filtered = bool(states or cats or multi)

sel_splits = dates[(dates["d"] >= d_from) & (dates["d"] <= d_to)]["split"].unique().tolist()
if any(s in ("train", "validation") for s in sel_splits):
    st.info("Выбранные даты входят в период train/validation: на них модель обучалась или настраивалась, "
            "поэтому качество здесь завышено. Ранее просмотренный holdout — период test "
            f"({meta['splits']['test']['start']} … {meta['splits']['test']['end_exclusive']}, не включая).")
if "after_test" in sel_splits:
    st.info("Даты после периода test: исходы в конце набора неполны и смещены (reports/dataset_tail_audit.md).")

q = run("risk_queue", version, d_from, d_to, int(k), tuple(states), bool(multi), tuple(cats))
if q.empty:
    st.info("Нет подтверждённых заказов под выбранные дату и фильтры. Расширьте диапазон дат или ослабьте фильтры.")
    st.stop()

pool = run("risk_queue", version, d_from, d_to, None, tuple(states), bool(multi), tuple(cats))
n_days = pool["approved_at"].dt.date.nunique()
st.markdown(f"Модель: **{MODEL_TITLES[model_name]}** (`{version}`). "
            f"Подтверждено в выбранный период{' (после фильтров)' if filtered else ''}: **{fmt_int(len(pool))}** "
            f"заказов за {n_days} дн. · в очереди — top-{int(k)} **в каждый день**: **{fmt_int(len(q))}** строк "
            f"({fmt_pct(len(q) / len(pool))} заказов) · стоимость товаров в очереди: **{fmt_money(q['items_value'].sum())}**")
note("Правило очереди: фильтры сужают заказы дня, затем в каждый день подтверждения выбираются K заказов с наибольшим "
     "риск-скором выбранной модели; при равном скоре — по order_id (детерминированно, без использования исходов). "
     "Стоимость заказов в очереди — объём под проверкой, а не ожидаемый убыток. Риск-скор не откалиброван и не "
     "является вероятностью.")

show_outcomes = st.toggle("Показать фактические исходы (для оценки симуляции)", value=False)

if model_name == "logreg":
    factors = logreg_factors(q["order_id"].tolist())
    q = q.assign(top_factors=q["order_id"].map(factors).fillna(""))
    factor_col = "Факторы ↑ (линейные вклады LogReg)"
elif model_name == "catboost":
    factor_col = "Факторы ↑ (SHAP CatBoost)"
else:
    factor_col = None

view = q.rename(columns={
    "approved_at": "Подтверждён", "promised_date": "Обещанная дата",
    "promised_days": "Срок, дн.", "items_value": "Стоимость товаров",
    "customer_state": "Штат покупателя", "main_seller_state": "Штат продавца", "n_sellers": "Продавцов",
    "main_category": "Категория", "risk_score": "Риск-скор", "top_factors": factor_col or "top_factors"})
view["Ранг в дне (после фильтров)"] = view.groupby(view["Подтверждён"].dt.date).cumcount() + 1
view["Модель"] = version
cols = ["order_id", "Подтверждён", "Обещанная дата", "Срок, дн.", "Стоимость товаров", "Штат покупателя",
        "Штат продавца", "Продавцов", "Категория", "Риск-скор", "Ранг в дне (после фильтров)"]
if factor_col:
    cols.append(factor_col)
view = view[cols + ["Модель"]]

if show_outcomes:
    out = run("queue_outcomes", q["order_id"].tolist())
    out["Исход"] = out["outcome"].map(OUTCOME_RU).fillna(out["outcome"])
    view = view.merge(out[["order_id", "Исход", "delay_days"]].rename(columns={"delay_days": "Опоздание, дн."}),
                      on="order_id", how="left")
    qs = run("queue_daily_quality", version, d_from, d_to, int(k), tuple(states), bool(multi), tuple(cats))
    if qs:
        if filtered:
            st.warning("Метрики ниже — качество **этой отфильтрованной очереди** (политика внутри сегмента). "
                       "Их нельзя сравнивать с общими метриками модели на странице «Качество».")
        a, b, c, e = st.columns(4)
        a.metric("Проверок в день (среднее)", fmt_num(qs["model_reviewed_per_day"]))
        b.metric("Найдено задержек в день", fmt_num(qs["model_found_per_day"], 2),
                 delta=f"случайно ≈ {fmt_num(qs['random_found_per_day'], 2)}", delta_color="off")
        c.metric("Precision@K (micro)", fmt_pct(qs["model_micro_precision"]),
                 delta=f"случайно ≈ {fmt_pct(qs['random_micro_precision'])}", delta_color="off")
        e.metric("Recall@K (micro)", fmt_pct(qs["model_micro_recall"]),
                 delta=f"случайно ≈ {fmt_pct(qs['random_micro_recall'])}", delta_color="off")
        with st.expander("Как считается"):
            st.markdown(
                f"Дни: {qs['model_days']}, без опозданий: {qs['model_days_without_late']}, с числом заказов < K: "
                f"{qs['model_days_n_lt_K']}. Когорта дня — доставленные заказы с меткой после фильтров "
                f"(недоставленных без метки: {fmt_int(qs['unlabelled_orders'])}). k_d = min(K, n_d); "
                "**micro** precision = Σh_d / Σk_d, **micro** recall = Σh_d / Σm_d. Macro precision (среднее по дням) = "
                f"{fmt_pct(qs['model_macro_precision'])}, macro recall (по дням с опозданиями) = "
                f"{fmt_pct(qs['model_macro_recall'])}. «Случайно» — ожидание равномерного случайного отбора тех же "
                "k_d заказов в каждом дне: E[h_d] = k_d·m_d/n_d. В метриках равные скоры на границе K учитываются "
                "как ожидание при случайном разрешении, поэтому для моделей с ties метрика может слегка отличаться "
                "от конкретного состава таблицы (там ties — по order_id). Найденная задержка не означает "
                "предотвращённую.")
    else:
        st.info("Для выбранных заказов нет доставленных с меткой — оценить очередь нельзя.")

st.dataframe(view, hide_index=True, width="stretch", height=min(700, 38 + 35 * len(view)), column_config={
    "Подтверждён": st.column_config.DatetimeColumn(format="DD.MM.YYYY HH:mm"),
    "Обещанная дата": st.column_config.DateColumn(format="DD.MM.YYYY"),
    "Стоимость товаров": st.column_config.NumberColumn(format="R$ %.2f"),
    "Риск-скор": st.column_config.ProgressColumn(format="%.3f", min_value=0.0,
                                                 max_value=float(max(pool["risk_score"].max(), 1e-9)))
    if model_name != "rule_short_promise" else st.column_config.NumberColumn(format="%.1f"),
})
if model_name == "rule_short_promise":
    note("Для правила риск-скор = −обещанный срок в днях (чем короче срок, тем выше в очереди).")
if model_name == "constant":
    note("Константа даёт всем заказам одинаковый скор: состав очереди определяется только порядком order_id.")

csv = view.to_csv(index=False).encode("utf-8-sig")
st.download_button("Экспорт CSV (ровно эта таблица)", csv,
                   file_name=f"orderpulse_queue_{model_name}_{d_from}_{d_to}_top{int(k)}_per_day.csv", mime="text/csv")
note(f"Версия модели: {version} · признаки {meta['feature_version']} · прогнозы рассчитаны пакетно; "
     "факторы LogReg вычисляются для показанных строк по той же модели.")

st.divider()
pick = st.selectbox("Открыть карточку заказа (с этой же моделью)", ["—"] + q["order_id"].tolist())
if pick != "—":
    st.session_state["card_order_id"] = pick
    st.session_state["op_model"] = model_name
    st.switch_page("views/order_card.py")
