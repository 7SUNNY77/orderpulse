from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import COLORS, cfg, fmt_int, note, plotly_layout, require_warehouse, run, sidebar_filters

require_warehouse()
st.title("Проблемные сегменты")
st.caption("Где стоит разобраться: доля задержек по сегментам рядом с размером выборки и 95% доверительным "
           "интервалом (Wilson). Различия — статистическая связь, а не доказательство вины продавца или региона: "
           "на них влияют ассортимент, география и сезонность.")

f = sidebar_filters("seg")
min_n = st.sidebar.number_input("Мин. доставленных заказов в сегменте", min_value=1, max_value=5000,
                                value=int(cfg().model["ui"]["min_segment_size"]), step=10)

KINDS = {
    "Продавцы": ("seller", "Продавец (seller_id)",
                 "Зерно: продавец × заказ. Заказ с несколькими продавцами учитывается у каждого из них, "
                 "поэтому сумма заказов по продавцам больше числа уникальных заказов. Задержка заказа "
                 "приписывается всем его продавцам."),
    "Регионы покупателей": ("customer_state", "Штат покупателя", "Зерно: заказ (каждый заказ один раз)."),
    "Регионы продавцов": ("seller_state", "Штат продавца", "Зерно: продавец × заказ."),
    "Категории": ("category", "Категория",
                  "Зерно: заказ × категория. Заказ с товарами нескольких категорий учитывается в каждой."),
}
tabs = st.tabs(list(KINDS))
for tab, (title, (kind, label, grain)) in zip(tabs, KINDS.items()):
    with tab:
        note(grain)
        df = run("segments", f, kind)
        if df.empty:
            st.info("Нет данных под выбранные фильтры.")
            continue
        big = df[df["delivered"] >= min_n].copy()
        small = df[df["delivered"] < min_n]
        st.markdown(f"Сегментов: **{fmt_int(len(df))}**, из них с ≥ {min_n} доставленными: **{fmt_int(len(big))}**. "
                    f"Маленькие сегменты ({fmt_int(len(small))}) не ранжируются — их доля нестабильна.")
        if big.empty:
            st.info("Нет сегментов достаточного размера. Уменьшите порог или расширьте период.")
            continue
        sort_by = st.radio("Сортировка", ["Доля задержек", "Нижняя граница ДИ", "Число опозданий", "Объём"],
                           horizontal=True, key=f"sort_{kind}",
                           help="«Нижняя граница ДИ» — консервативная сортировка: сегмент вверху почти наверняка проблемный.")
        col = {"Доля задержек": "late_rate", "Нижняя граница ДИ": "ci_low", "Число опозданий": "late",
               "Объём": "delivered"}[sort_by]
        big = big.sort_values([col, "delivered"], ascending=[False, False]).reset_index(drop=True)
        top = big.head(20).iloc[::-1]
        fig = go.Figure(go.Scatter(
            x=top["late_rate"], y=top["segment"].astype(str), mode="markers",
            marker=dict(size=9, color=COLORS["late"]),
            error_x=dict(type="data", symmetric=False, array=top["ci_high"] - top["late_rate"],
                         arrayminus=top["late_rate"] - top["ci_low"], color="#b9b8b3", thickness=1.5),
            customdata=top[["late", "delivered"]],
            hovertemplate="%{y}<br>доля %{x:.1%} (%{customdata[0]} / %{customdata[1]})<extra></extra>"))
        overall = big["late"].sum() / big["delivered"].sum()
        fig.add_vline(x=overall, line_dash="dot", line_color=COLORS["neutral"],
                      annotation_text=f"в среднем по показанным {overall:.1%}", annotation_position="top")
        fig.update_layout(title=f"Top-20 по «{sort_by}»: доля задержек и 95% ДИ", xaxis_tickformat=".0%")
        st.plotly_chart(plotly_layout(fig, height=max(320, 22 * len(top) + 80)), width="stretch")
        show = big.rename(columns={
            "segment": label, "orders": "Заказов", "delivered": "Доставлено", "late": "Опоздало",
            "late_rate": "Доля задержек", "ci_low": "ДИ 95% от", "ci_high": "ДИ 95% до",
            "median_delivery_days": "Медиана доставки, дн.", "avg_score": "Средняя оценка",
            "reviews": "Отзывов", "low_share": "Доля оценок 1–2", "items_value": "Стоимость товаров"})
        cols = [label, "Заказов", "Доставлено", "Опоздало", "Доля задержек", "ДИ 95% от", "ДИ 95% до",
                "Медиана доставки, дн.", "Средняя оценка", "Отзывов", "Доля оценок 1–2", "Стоимость товаров"]
        st.dataframe(show[cols], hide_index=True, width="stretch", height=380, column_config={
            "Доля задержек": st.column_config.NumberColumn(format="percent"),
            "ДИ 95% от": st.column_config.NumberColumn(format="percent"),
            "ДИ 95% до": st.column_config.NumberColumn(format="percent"),
            "Доля оценок 1–2": st.column_config.NumberColumn(format="percent"),
            "Медиана доставки, дн.": st.column_config.NumberColumn(format="%.1f"),
            "Средняя оценка": st.column_config.NumberColumn(format="%.2f"),
            "Стоимость товаров": st.column_config.NumberColumn(format="R$ %.0f"),
        })
        if kind in ("seller", "customer_state", "category"):
            pick = st.selectbox("Динамика сегмента по месяцам (все периоды, без фильтров)",
                                ["—"] + big["segment"].astype(str).head(50).tolist(), key=f"trend_{kind}")
            if pick != "—":
                tr = run("segment_trend", kind, pick)
                fig = go.Figure()
                fig.add_bar(x=tr["month"], y=tr["delivered"], name="Доставлено", marker_color="#c9d9ef",
                            yaxis="y")
                fig.add_scatter(x=tr["month"], y=tr["late"], name="Опоздало", mode="lines+markers",
                                line=dict(color=COLORS["late"], width=2))
                fig.update_layout(title=f"{pick}: доставленные и опоздавшие заказы по месяцам покупки")
                st.plotly_chart(plotly_layout(fig), width="stretch")
