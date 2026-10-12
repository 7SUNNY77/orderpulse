from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import (COLORS, OUTCOME_RU, fmt_int, fmt_money, fmt_money_short, fmt_num, fmt_pct, note, plotly_layout,
                    require_warehouse, run, sidebar_filters)

require_warehouse()
st.title("Обзор доставки")
st.caption("Что произошло: ретроспективная аналитика по фактическим исходам. "
           "Период — по дате покупки. Каждый заказ учитывается один раз.")

f = sidebar_filters("ov")
k = run("kpis", f)

if not k["n_orders"]:
    st.info("Нет заказов под выбранные фильтры. Ослабьте фильтры или расширьте период.")
    st.stop()

c = st.columns(3) + st.columns(3)
c[0].metric("Заказы", fmt_int(k["n_orders"]))
with c[0]:
    note("уникальные order_id в периоде")
c[1].metric("Стоимость товаров", fmt_money_short(k["items_value"]))
with c[1]:
    note(f"{fmt_money(k['items_value'])}: сумма цен позиций, без доставки; {fmt_int(k['n_with_items'])} заказов с позициями")
c[2].metric("Доля задержек", fmt_pct(k["late_rate"]))
with c[2]:
    note(f"{fmt_int(k['n_late'])} опоздавших / {fmt_int(k['n_delivered'])} доставленных с обеими датами")
c[3].metric("Медиана доставки", f"{fmt_num(k['median_delivery_days'])} дн.")
with c[3]:
    note(f"P90 = {fmt_num(k['p90_delivery_days'])} дн.; от покупки до доставки; {fmt_int(k['n_delivered'])} доставленных")
c[4].metric("Доля оценок 1–2", fmt_pct(k["low_share"]))
with c[4]:
    note(f"{fmt_int(k['n_low'])} / {fmt_int(k['n_reviewed'])} заказов с отзывом; средняя оценка {fmt_num(k['avg_score'], 2)}")
c[5].metric("Покрытие отзывами", fmt_pct(k["review_coverage"]))
with c[5]:
    note(f"{fmt_int(k['n_reviewed'])} / {fmt_int(k['n_orders'])} заказов; канонический отзыв — последний")

with st.expander("Дополнительные показатели и знаменатели"):
    a, b, cc = st.columns(3)
    a.markdown(
        f"- Средняя стоимость заказа (товары): **{fmt_money(k['avg_order_value'])}** — "
        f"сумма товаров / {fmt_int(k['n_with_items'])} заказов с позициями\n"
        f"- То же с доставкой: **{fmt_money(k['avg_order_value_with_freight'])}**\n"
        f"- Стоимость доставки (freight): **{fmt_money(k['freight_value'])}**")
    b.markdown(
        f"- Сумма платежей (отдельно, не равна товары+доставка): **{fmt_money(k['payment_value'])}** "
        f"по {fmt_int(k['n_with_payment'])} заказам\n"
        f"- Среднее опоздание среди опоздавших: **{fmt_num(k['mean_delay_days_late'])} дн.**, "
        f"медиана **{fmt_num(k['median_delay_days_late'], 0)} дн.** (календарные дни после обещанной даты)")
    cc.markdown(
        f"- Не доставлено (в пути/обработка): **{fmt_int(k['n_not_delivered'])}**\n"
        f"- Отменено: **{fmt_int(k['n_canceled'])}**, недоступно: **{fmt_int(k['n_unavailable'])}**\n"
        f"- Статус «доставлен» без даты: **{fmt_int(k['n_delivered_missing_date'])}**\n"
        f"- Вне знаменателя доли задержек: **{fmt_pct(k['excluded_share'])}** заказов")
    st.caption("Стоимость товаров не является выручкой или прибылью Olist. "
               "Недоставленные заказы не считаются «вовремя» и не входят в знаменатель доли задержек.")

st.subheader("Заказы и задержки по месяцам")
t = run("monthly_trend", f)
col1, col2 = st.columns(2)
fig = go.Figure()
fig.add_bar(x=t["month"], y=t["delivered"] - t["late"], name="Вовремя", marker_color=COLORS["on_time"],
            hovertemplate="%{x|%Y-%m}<br>Вовремя: %{y}<extra></extra>")
fig.add_bar(x=t["month"], y=t["late"], name="Опоздали", marker_color=COLORS["late"],
            hovertemplate="%{x|%Y-%m}<br>Опоздали: %{y}<extra></extra>")
fig.update_layout(barmode="stack", title="Доставленные заказы, по месяцу покупки")
col1.plotly_chart(plotly_layout(fig), width="stretch")
MIN_MONTH = 100  # months with fewer delivered orders are not plotted (unstable rate)
t = t.assign(late_rate=t["late_rate"].where(t["delivered"] >= MIN_MONTH))
fig = go.Figure()
fig.add_scatter(x=t["month"], y=t["late_rate"], mode="lines+markers", name="Доля задержек",
                line=dict(color=COLORS["late"], width=2), marker=dict(size=8),
                customdata=t[["late", "delivered"]],
                hovertemplate="%{x|%Y-%m}<br>%{y:.1%} (%{customdata[0]} / %{customdata[1]})<extra></extra>")
fig.update_layout(title="Доля задержек среди доставленных", yaxis_tickformat=".0%")
col2.plotly_chart(plotly_layout(fig), width="stretch")
note(f"Доля задержек показана для месяцев с ≥ {MIN_MONTH} доставленными заказами. Последние месяцы набора неполные: "
     "заказы, не доставленные к срезу данных, не попадают в знаменатель.")

st.subheader("Распределение и отзывы")
col1, col2, col3 = st.columns([1.3, 1, 1])
h = run("delivery_days_hist", f)
fig = go.Figure()
fig.add_bar(x=h["days"], y=h["on_time"], name="Вовремя", marker_color=COLORS["on_time"])
fig.add_bar(x=h["days"], y=h["late"], name="Опоздали", marker_color=COLORS["late"])
fig.update_layout(barmode="stack", title="Длительность доставки, дней (60 = 60+)", bargap=0.1)
col1.plotly_chart(plotly_layout(fig), width="stretch")

o = run("outcome_breakdown", f)
o["Исход"] = o["outcome"].map(OUTCOME_RU).fillna(o["outcome"])
fig = go.Figure(go.Bar(x=o["orders"], y=o["Исход"], orientation="h", marker_color=COLORS["neutral"],
                       text=o["orders"], textposition="auto", cliponaxis=False))
fig.update_layout(title="Исходы заказов", yaxis=dict(autorange="reversed"))
col2.plotly_chart(plotly_layout(fig), width="stretch")

r = run("reviews_by_lateness", f)
fig = go.Figure()
for grp, color in [("Вовремя", COLORS["on_time"]), ("Опоздал", COLORS["late"])]:
    g = r[r["group_"] == grp]
    share = g["n"] / g["n"].sum() if len(g) else g["n"]
    fig.add_bar(x=g["review_score"], y=share, name=grp, marker_color=color,
                hovertemplate="Оценка %{x}: %{y:.1%}<extra>" + grp + "</extra>")
fig.update_layout(barmode="group", title="Оценки: вовремя vs опоздал (доля)", yaxis_tickformat=".0%",
                  xaxis=dict(dtick=1, title="оценка"))
col3.plotly_chart(plotly_layout(fig), width="stretch")
st.dataframe(run("reviews_summary_by_lateness", f), hide_index=True, width="stretch",
             column_config={"Доля 1–2": st.column_config.NumberColumn(format="percent")})
note("Связь задержек и оценок — наблюдаемая ассоциация среди доставленных заказов с отзывами, не доказательство причины.")
