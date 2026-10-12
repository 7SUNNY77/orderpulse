"""Synthetic Olist-shaped CSVs for tests.

Contains a random background population (so models can be trained and every
time split has both classes) plus hand-crafted edge-case orders with known
expected values.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

STATES = {"SP": ("01001", -23.55, -46.63), "RJ": ("20000", -22.90, -43.20),
          "MG": ("30000", -19.92, -43.94), "BA": ("40000", -12.97, -38.50),
          "RS": ("90000", -30.03, -51.23)}
CATEGORIES = {"cat_a": "alpha", "cat_b": "beta", "cat_c": "gamma"}
TS = "%Y-%m-%d %H:%M:%S"


def _fmt(x: datetime | None) -> str | None:
    return None if x is None else x.strftime(TS)


def make_tables(seed: int = 0, n_random: int = 900) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sellers = []
    for i, st in enumerate(list(STATES) * 2):
        sellers.append({"seller_id": f"s{i:02d}", "seller_zip_code_prefix": STATES[st][0],
                        "seller_city": "city", "seller_state": st})
    products = []
    for i in range(15):
        cat = list(CATEGORIES)[i % 3]
        products.append({"product_id": f"p{i:02d}", "product_category_name": cat,
                         "product_name_lenght": 40, "product_description_lenght": 200, "product_photos_qty": 1,
                         "product_weight_g": float(100 + 300 * i), "product_length_cm": 10.0,
                         "product_height_cm": 10.0, "product_width_cm": float(5 + i)})
    products.append({"product_id": "p_unknown", "product_category_name": None, "product_name_lenght": None,
                     "product_description_lenght": None, "product_photos_qty": None, "product_weight_g": None,
                     "product_length_cm": None, "product_height_cm": None, "product_width_cm": None})
    products.append({"product_id": "p_untranslated", "product_category_name": "cat_no_translation",
                     "product_name_lenght": 40, "product_description_lenght": 200, "product_photos_qty": 1,
                     "product_weight_g": 500.0, "product_length_cm": 10.0, "product_height_cm": 10.0,
                     "product_width_cm": 10.0})

    orders, items, payments, reviews, customers = [], [], [], [], []

    def add_order(oid, status, purchase, approved, carrier, delivered, estimated, cust_state="SP"):
        cid = f"c_{oid}"
        customers.append({"customer_id": cid, "customer_unique_id": f"u_{oid}",
                          "customer_zip_code_prefix": STATES[cust_state][0], "customer_city": "city",
                          "customer_state": cust_state})
        orders.append({"order_id": oid, "customer_id": cid, "order_status": status,
                       "order_purchase_timestamp": _fmt(purchase), "order_approved_at": _fmt(approved),
                       "order_delivered_carrier_date": _fmt(carrier),
                       "order_delivered_customer_date": _fmt(delivered),
                       "order_estimated_delivery_date": _fmt(estimated)})

    def add_item(oid, n, pid, sid, price, freight, limit):
        items.append({"order_id": oid, "order_item_id": n, "product_id": pid, "seller_id": sid,
                      "shipping_limit_date": _fmt(limit), "price": f"{price:.2f}", "freight_value": f"{freight:.2f}"})

    def add_payment(oid, seq, ptype, inst, value):
        payments.append({"order_id": oid, "payment_sequential": seq, "payment_type": ptype,
                         "payment_installments": inst, "payment_value": f"{value:.2f}"})

    def add_review(rid, oid, score, created, answered):
        reviews.append({"review_id": rid, "order_id": oid, "review_score": score,
                        "review_comment_title": None, "review_comment_message": None,
                        "review_creation_date": _fmt(created), "review_answer_timestamp": _fmt(answered)})

    # ---------------- random background population ----------------
    start = datetime(2017, 1, 1)
    span_days = (datetime(2018, 8, 25) - start).days
    state_list = list(STATES)
    for k in range(n_random):
        oid = f"o{k:05d}"
        purchase = start + timedelta(days=float(rng.uniform(0, span_days)))
        approved = purchase + timedelta(hours=float(rng.uniform(0.1, 30)))
        promised = int(rng.integers(5, 35))
        estimated = datetime.combine((approved + timedelta(days=promised)).date(), datetime.min.time())
        cust_state = state_list[int(rng.integers(0, 5))]
        sid = f"s{int(rng.integers(0, 10)):02d}"
        seller_state = sellers[int(sid[1:])]["seller_state"]
        interstate = seller_state != cust_state
        # Lateness depends on short promises and interstate shipping (learnable signal).
        p_late = 0.04 + 0.30 * (promised < 12) + 0.10 * interstate
        late = rng.uniform() < p_late
        actual_days = promised + float(rng.uniform(1, 10)) if late else promised * float(rng.uniform(0.3, 0.95))
        delivered = approved + timedelta(days=actual_days)
        status = "delivered"
        r = rng.uniform()
        if r < 0.02:
            status, delivered = "canceled", None
        elif r < 0.04:
            status, delivered = "shipped", None
        carrier = approved + timedelta(days=1) if status != "canceled" else None
        add_order(oid, status, purchase, approved, carrier, delivered, estimated, cust_state)
        pid = f"p{int(rng.integers(0, 15)):02d}"
        price = float(rng.uniform(10, 300))
        freight = float(rng.uniform(5, 40))
        add_item(oid, 1, pid, sid, price, freight, approved + timedelta(days=3))
        add_payment(oid, 1, "credit_card", int(rng.integers(1, 6)), round(price, 2) + round(freight, 2))
        if delivered is not None:
            add_review(f"r_{oid}", oid, int(2 if late else 5), delivered + timedelta(days=1),
                       delivered + timedelta(days=2))

    # ---------------- hand-crafted edge cases (all in the train period) ----------------
    p0 = datetime(2017, 3, 1, 10, 0, 0)
    a0 = p0 + timedelta(hours=1)
    # 3 items (2 sellers), 2 payments, 2 reviews
    add_order("o_multi", "delivered", p0, a0, a0 + timedelta(days=1), datetime(2017, 3, 9, 12), datetime(2017, 3, 20))
    add_item("o_multi", 1, "p00", "s00", 10.00, 1.00, a0)
    add_item("o_multi", 2, "p01", "s00", 20.00, 2.00, a0)
    add_item("o_multi", 3, "p02", "s01", 30.50, 3.00, a0)
    add_payment("o_multi", 1, "credit_card", 3, 40.00)
    add_payment("o_multi", 2, "voucher", 1, 26.50)
    add_review("r_multi_old", "o_multi", 1, datetime(2017, 3, 10), datetime(2017, 3, 11))
    add_review("r_multi_new", "o_multi", 5, datetime(2017, 3, 12), datetime(2017, 3, 13))
    # two reviews with identical timestamps -> tie broken by review_id DESC
    add_order("o_tie", "delivered", p0, a0, a0, datetime(2017, 3, 9), datetime(2017, 3, 20))
    add_item("o_tie", 1, "p03", "s02", 50.0, 5.0, a0)
    add_payment("o_tie", 1, "boleto", 1, 55.0)
    add_review("r_tie_a", "o_tie", 2, datetime(2017, 3, 10), datetime(2017, 3, 11))
    add_review("r_tie_b", "o_tie", 4, datetime(2017, 3, 10), datetime(2017, 3, 11))
    # delivered late in the evening of the promised day -> on time
    add_order("o_sameday", "delivered", p0, a0, a0, datetime(2017, 3, 20, 23, 59, 0), datetime(2017, 3, 20))
    add_item("o_sameday", 1, "p04", "s03", 70.0, 7.0, a0)
    add_payment("o_sameday", 1, "credit_card", 1, 77.0)
    # delivered just after midnight the next day -> late by 1 day
    add_order("o_nextday", "delivered", p0, a0, a0, datetime(2017, 3, 21, 0, 1, 0), datetime(2017, 3, 20))
    add_item("o_nextday", 1, "p05", "s04", 80.0, 8.0, a0)
    add_payment("o_nextday", 1, "credit_card", 1, 88.0)
    # shipped, never delivered -> no label, not 'on time'
    add_order("o_nodelivery", "shipped", p0, a0, a0, None, datetime(2017, 3, 20))
    add_item("o_nodelivery", 1, "p06", "s05", 60.0, 6.0, a0)
    add_payment("o_nodelivery", 1, "credit_card", 1, 66.0)
    # status delivered but delivery date missing -> no label
    add_order("o_delivered_nodate", "delivered", p0, a0, a0, None, datetime(2017, 3, 20))
    add_item("o_delivered_nodate", 1, "p07", "s06", 60.0, 6.0, a0)
    add_payment("o_delivered_nodate", 1, "credit_card", 1, 66.0)
    # unknown category + untranslated category
    add_order("o_unknowncat", "delivered", p0, a0, a0, datetime(2017, 3, 9), datetime(2017, 3, 20))
    add_item("o_unknowncat", 1, "p_unknown", "s07", 99.0, 9.0, a0)
    add_payment("o_unknowncat", 1, "credit_card", 1, 108.0)
    add_order("o_untranslated", "delivered", p0, a0, a0, datetime(2017, 3, 9), datetime(2017, 3, 20))
    add_item("o_untranslated", 1, "p_untranslated", "s07", 15.0, 2.0, a0)
    add_payment("o_untranslated", 1, "credit_card", 1, 17.0)
    # zero price
    add_order("o_zero", "delivered", p0, a0, a0, datetime(2017, 3, 9), datetime(2017, 3, 20))
    add_item("o_zero", 1, "p08", "s08", 0.0, 0.0, a0)
    add_payment("o_zero", 1, "voucher", 1, 0.0)
    # no items at all
    add_order("o_noitems", "unavailable", p0, a0, None, None, datetime(2017, 3, 20))
    add_payment("o_noitems", 1, "credit_card", 1, 30.0)
    # delivered before approval -> temporal error
    add_order("o_temporal", "delivered", p0, datetime(2017, 3, 5), a0, datetime(2017, 3, 4), datetime(2017, 3, 20))
    add_item("o_temporal", 1, "p09", "s09", 40.0, 4.0, a0)
    add_payment("o_temporal", 1, "credit_card", 1, 44.0)

    geo = []
    for st, (z, lat, lng) in STATES.items():
        for j in range(3):  # several points per prefix
            geo.append({"geolocation_zip_code_prefix": z, "geolocation_lat": lat + 0.01 * j,
                        "geolocation_lng": lng - 0.01 * j, "geolocation_city": "city", "geolocation_state": st})
    geo.append({"geolocation_zip_code_prefix": "01001", "geolocation_lat": 40.0, "geolocation_lng": 10.0,
                "geolocation_city": "outlier", "geolocation_state": "SP"})  # outside Brazil -> dropped
    translation = [{"product_category_name": k, "product_category_name_english": v} for k, v in CATEGORIES.items()]
    return {
        "orders": pd.DataFrame(orders), "order_items": pd.DataFrame(items),
        "order_payments": pd.DataFrame(payments), "order_reviews": pd.DataFrame(reviews),
        "customers": pd.DataFrame(customers), "sellers": pd.DataFrame(sellers),
        "products": pd.DataFrame(products), "geolocation": pd.DataFrame(geo),
        "category_translation": pd.DataFrame(translation),
    }


FILE_NAMES = {
    "orders": "olist_orders_dataset.csv", "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv", "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv", "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv", "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def write_csvs(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(directory / FILE_NAMES[name], index=False)
