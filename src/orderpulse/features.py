"""Prediction-time feature matrix with an explicit allow-list.

X is built by SELECTING the allow-listed columns from feature_orders_at_approval
(which itself only reads pre-approval sources, see the SQL header), never by
dropping the target from a wide table.
"""

from __future__ import annotations

import pandas as pd

FEATURE_VERSION = "fv1"

NUMERIC_FEATURES: list[str] = [
    # order
    "n_items", "n_products", "n_sellers", "items_value", "freight_value", "freight_ratio",
    "avg_item_price", "max_item_price", "hours_to_approval", "promised_days",
    "purchase_dow", "approval_dow", "approval_hour", "approval_month",
    # geography
    "is_interstate", "n_seller_states", "dist_mean_km", "dist_max_km", "geo_missing",
    # products
    "n_categories", "total_weight_g", "max_weight_g", "total_volume_cm3", "max_volume_cm3",
    "share_unknown_attrs",
]
CATEGORICAL_FEATURES: list[str] = ["customer_state", "main_seller_state", "main_category"]
FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Substrings that must never appear in a model feature name.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "deliver", "delay", "late", "review", "score", "status", "outcome", "label", "carrier",
    "order_id", "customer_id", "customer_unique", "seller_id", "payment", "installment",
    "shipping_limit",
)

FEATURE_LABELS: dict[str, str] = {
    "n_items": "Число позиций", "n_products": "Число товаров", "n_sellers": "Число продавцов",
    "items_value": "Стоимость товаров", "freight_value": "Стоимость доставки",
    "freight_ratio": "Доставка / товары", "avg_item_price": "Средняя цена позиции",
    "max_item_price": "Макс. цена позиции", "hours_to_approval": "Часов до подтверждения",
    "promised_days": "Обещанный срок, дней", "purchase_dow": "День недели покупки",
    "approval_dow": "День недели подтверждения", "approval_hour": "Час подтверждения",
    "approval_month": "Месяц подтверждения", "is_interstate": "Межштатная доставка",
    "n_seller_states": "Штатов продавцов", "dist_mean_km": "Среднее расстояние, км",
    "dist_max_km": "Макс. расстояние, км", "geo_missing": "Нет координат",
    "n_categories": "Число категорий", "total_weight_g": "Суммарный вес, г",
    "max_weight_g": "Макс. вес, г", "total_volume_cm3": "Суммарный объём, см³",
    "max_volume_cm3": "Макс. объём, см³", "share_unknown_attrs": "Доля неизвестных характеристик",
    "customer_state": "Штат покупателя", "main_seller_state": "Штат основного продавца",
    "main_category": "Основная категория",
}


class LeakageError(RuntimeError):
    pass


def assert_no_forbidden(columns: list[str]) -> None:
    bad = [c for c in columns if any(s in c.lower() for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise LeakageError(f"forbidden feature columns: {bad}")


assert_no_forbidden(FEATURES)


def load_feature_frame(con) -> pd.DataFrame:
    """order_id + prediction_time (+ promised_ts for display) + allow-listed features."""
    cols = ", ".join(["order_id", "prediction_time"] + FEATURES)
    return con.execute(f"SELECT {cols} FROM feature_orders_at_approval ORDER BY order_id").df()


def load_labels(con) -> pd.DataFrame:
    return con.execute(
        "SELECT order_id, is_late, label_available_at, ml_eligible, ml_exclusion_reason FROM ml_labels"
    ).df()


def to_X(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix: exactly FEATURES, in a fixed order, with stable dtypes."""
    X = df.loc[:, FEATURES].copy()
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("object").where(X[c].notna(), "unknown").astype(str)
    assert_no_forbidden(list(X.columns))
    return X
