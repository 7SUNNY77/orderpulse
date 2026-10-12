"""OrderPulse Streamlit entry point: `python -m orderpulse app`."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

st.set_page_config(page_title="OrderPulse", page_icon="📦", layout="wide")

from common import CSS  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

pages = [
    st.Page(str(APP_DIR / "views" / "overview.py"), title="Обзор", icon="📊", default=True),
    st.Page(str(APP_DIR / "views" / "segments.py"), title="Проблемные сегменты", icon="🔎"),
    st.Page(str(APP_DIR / "views" / "risk_queue.py"), title="Очередь риска", icon="🚩"),
    st.Page(str(APP_DIR / "views" / "order_card.py"), title="Карточка заказа", icon="🧾"),
    st.Page(str(APP_DIR / "views" / "quality.py"), title="Качество модели и данных", icon="✅"),
]
st.sidebar.markdown("**OrderPulse** · контроль доставки  \nДанные: Olist (2016–2018), историческая выборка")
st.navigation(pages).run()
