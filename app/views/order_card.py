from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (COLORS, OUTCOME_RU, SPLIT_RU, explain, fmt_money, fmt_num, fmt_pct, model_versions_consistent,
                    note, plotly_layout, require_warehouse, run, select_model)

from orderpulse.explanations import review_hints
from orderpulse.models import MODEL_TITLES

require_warehouse()
st.title("Карточка заказа")


def contributions_chart(t: pd.DataFrame, col: str, unit: str, title: str):
    top = pd.concat([t[t[col] > 0].head(6), t[t[col] < 0].head(6)]).sort_values(col)
    fig = go.Figure(go.Bar(
        x=top[col], y=top["Признак"] + " = " + top["Значение"], orientation="h",
        marker_color=[COLORS["late"] if v > 0 else COLORS["on_time"] for v in top[col]],
        hovertemplate="%{y}<br>вклад %{x:+.3f} " + unit + "<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=f"вклад, {unit}")
    st.plotly_chart(plotly_layout(fig, height=60 + 28 * len(top)), width="stretch")


default_id = st.session_state.get("card_order_id", "")
order_id = st.text_input("order_id", value=default_id, placeholder="например, из очереди риска").strip()
if not order_id:
    st.info("Введите order_id или откройте заказ из «Очереди риска».")
    st.stop()
card = run("order_card", order_id)
if card is None:
    st.warning(f"Заказ `{order_id}` не найден.")
    st.stop()

o, feats = card["order"], card["features"]


def ts(x) -> str:
    return "—" if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%d.%m.%Y %H:%M")


def d(x) -> str:
    return "—" if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%d.%m.%Y")


left, right = st.columns([1.15, 1])
with left:
    st.subheader("На момент подтверждения")
    note("Сведения, считающиеся доступными при подтверждении заказа (допущение о составе и обещанной дате — "
         "см. reports/decisions.md).")
    st.markdown(
        f"<div class='op-card'>Покупка: <b>{ts(o['purchase_ts'])}</b><br>"
        f"Подтверждение (prediction_time): <b>{ts(o['approved_ts'])}</b><br>"
        f"Обещанная дата: <b>{d(o['estimated_ts'])}</b>"
        f"{'' if feats is None else ' · срок ' + format(feats['promised_days'], '.1f') + ' дн. от подтверждения'}</div>",
        unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Стоимость товаров", fmt_money(o["items_value"]))
    b.metric("Доставка (freight)", fmt_money(o["freight_value"]))
    c.metric("Штат покупателя", o["customer_state"])
    st.markdown("**Состав заказа**")
    if card["items"].empty:
        st.info("У заказа нет позиций в данных.")
    else:
        st.dataframe(card["items"].rename(columns={
            "order_item_id": "#", "category": "Категория", "seller_state": "Штат продавца", "price": "Цена",
            "freight_value": "Доставка", "weight_g": "Вес, г"}), hide_index=True, width="stretch")
        st.markdown(f"**Продавцы** ({len(card['sellers'])})")
        st.dataframe(card["sellers"].rename(columns={
            "seller_state": "Штат", "n_items": "Позиций", "items_value": "Товары", "freight_value": "Доставка",
            "is_interstate": "Межштатная"}), hide_index=True, width="stretch")
        if len(card["sellers"]) > 1:
            note("Многопродавцовый заказ: в статистике заказов он считается один раз, в статистике продавцов — "
                 "у каждого продавца; итоговая задержка не доказывает вину каждого продавца.")

with right:
    st.subheader("Прогноз")
    model_name = select_model(st, "card", "Модель прогноза и объяснения")
    ok, msg = model_versions_consistent()
    if feats is None:
        st.info("Заказ не подтверждён: прогноз на момент подтверждения не формируется.")
    elif not ok:
        st.warning(msg)
    else:
        preds = run("order_predictions", order_id)
        sel = preds[preds["model_name"] == model_name]
        if sel.empty:
            st.info("Для заказа нет прогноза выбранной модели.")
        else:
            p = sel.iloc[0]
            split = p["split"]
            a, b, c = st.columns(3)
            a.metric("Риск-скор", f"{p['risk_score']:.3f}")
            b.metric("Ранг в дне", f"{int(p['risk_rank_in_day'])}")
            c.metric("Перцентиль", fmt_pct(p["risk_pct_in_day"], 0))
            note(f"Модель `{p['model_version']}` · период: {SPLIT_RU.get(split, split)} · рассчитан {p['scored_at']}. "
                 "Ранг и перцентиль — среди всех заказов дня (без фильтров очереди). Риск-скор — оценка для "
                 "ранжирования, не откалиброванная вероятность.")
            if split in ("train", "validation"):
                st.caption("⚠️ Заказ из периода обучения/настройки модели: прогноз in-sample.")
            others = preds[preds["model_name"] != model_name][["model_name", "risk_score", "risk_rank_in_day"]].copy()
            others["model_name"] = others["model_name"].map(MODEL_TITLES)
            with st.expander("Скоры других моделей (для сравнения)"):
                st.dataframe(others.rename(columns={"model_name": "Модель", "risk_score": "Скор",
                                                    "risk_rank_in_day": "Ранг в дне"}), hide_index=True, width="stretch")

            ex = explain(model_name, order_id, feats)
            if model_name == "logreg" and ex is not None:
                st.markdown("**Вклады признаков в линейный скор** (Logistic Regression)")
                t = ex["table"]
                contributions_chart(t, "Вклад (логит)", "логит", "Повышают (оранжевый) и снижают (синий) линейный скор")
                st.markdown(
                    f"Свободный член модели {ex['intercept']:+.3f} + сумма вкладов "
                    f"{ex['raw'] - ex['intercept']:+.3f} = линейный скор {ex['raw']:+.3f} (логит) → риск-скор "
                    f"{ex['score']:.3f} = σ(скор), совпадает с показанным выше.")
                note("Вклад признака = коэффициент × значение после предобработки модели (медианная импутация, "
                     "стандартизация; one-hot для категорий, редкие и неизвестные категории — в группе infrequent); "
                     "вклады one-hot колонок и индикатора пропуска сложены до исходного признака. Шкала — логит: "
                     "это не процентные пункты риска и не причинное объяснение. Свободный член — параметр модели, "
                     "его смысл зависит от предобработки; это не «средний риск». Для пропусков показано «нет данных»: "
                     "модель подставляет медиану train.")
                with st.expander("Все вклады"):
                    st.dataframe(t.drop(columns=["feature"]), hide_index=True, width="stretch",
                                 column_config={"Вклад (логит)": st.column_config.NumberColumn(format="%+.4f")})
            elif model_name == "catboost" and ex is not None:
                st.markdown("**Локальные SHAP-вклады** (CatBoost, шкала log-odds)")
                t = ex["table"]
                contributions_chart(t, "Вклад (log-odds)", "log-odds", "Повышают (оранжевый) и снижают (синий) прогноз")
                st.markdown(
                    f"Ожидаемое значение SHAP {ex['expected_raw']:+.3f} log-odds + сумма вкладов "
                    f"{ex['raw'] - ex['expected_raw']:+.3f} → {ex['raw']:+.3f} log-odds (скор {ex['score']:.3f}).")
                note("SHAP-вклады аддитивны в пространстве log-odds и не переводятся в процентные пункты. Они объясняют "
                     "поведение модели, а не причину задержки.")
                with st.expander("Все вклады"):
                    st.dataframe(t.drop(columns=["feature"]), hide_index=True, width="stretch",
                                 column_config={"Вклад (log-odds)": st.column_config.NumberColumn(format="%+.4f")})
            else:
                st.info("Для baseline-модели локальное объяснение не применяется: "
                        + ("скор = −обещанный срок." if model_name == "rule_short_promise" else "скор одинаков для всех заказов."))

            st.markdown("**Подсказки для проверки**")
            hints = review_hints(feats, float(p["risk_pct_in_day"]))
            if hints:
                for h in hints:
                    st.markdown(f"- {h}")
            else:
                st.markdown("- Особых сигналов по простым правилам нет.")
            note("Подсказки — простые документированные правила, а не доказанно оптимальные действия; "
                 "«высокий риск» — по выбранной модели.")

st.divider()
st.subheader("Ретроспектива: фактический исход")
note("Эти сведения появились после момента прогноза и скрыты в прогнозном режиме. Модель их не использует.")
if st.toggle("Показать фактический исход и отзыв", value=False):
    timeline = pd.DataFrame([
        ("Покупка", ts(o["purchase_ts"]), "да"),
        ("Подтверждение", ts(o["approved_ts"]), "да (момент прогноза)"),
        ("Обещанная дата", d(o["estimated_ts"]), "да (допущение)"),
        ("Передача перевозчику", ts(o["carrier_ts"]), "нет"),
        ("Доставка покупателю", ts(o["delivered_ts"]), "нет"),
        ("Отзыв создан", ts(o["review_creation_ts"]), "нет"),
    ], columns=["Событие", "Время", "Известно на момент прогноза"])
    st.dataframe(timeline, hide_index=True, width="stretch")
    a, b, c, e = st.columns(4)
    a.metric("Статус", o["order_status"])
    b.metric("Исход", OUTCOME_RU.get(o["outcome"], o["outcome"]))
    c.metric("Опоздание, дн.", fmt_num(o["delay_days"], 0) if o["delay_days"] is not None and not pd.isna(o["delay_days"]) else "—")
    e.metric("Оценка отзыва", "—" if pd.isna(o["review_score"]) else int(o["review_score"]))
    st.markdown(f"Платежи: {int(o['n_payment_records'])} записей, сумма {fmt_money(o['payment_value'])}, "
                f"типы: {o['payment_types'] or '—'}. Платёжные поля в модель fv1 не входят.")
    flags = [k for k in o if k.startswith("flag_") and o[k]]
    if flags:
        st.warning("Временные аномалии: " + ", ".join(flags))
