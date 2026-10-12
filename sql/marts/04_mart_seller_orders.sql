-- mart_seller_orders: one row per (seller_id, order_id).
-- A multi-seller order appears once per seller, so the number of rows is NOT
-- the number of unique orders. Order-level outcomes (lateness, review) are
-- attributed to every seller of the order; this is association, not blame.
CREATE OR REPLACE TABLE mart_seller_orders AS
WITH so AS (
    SELECT order_id, seller_id,
           count(*)            AS n_items,
           sum(price)          AS items_value,
           sum(freight_value)  AS freight_value
    FROM stg_order_items
    GROUP BY order_id, seller_id
)
SELECT
    so.seller_id,
    so.order_id,
    so.n_items,
    so.items_value,
    so.freight_value,
    coalesce(s.seller_state, 'unknown')  AS seller_state,
    o.customer_state,
    (coalesce(s.seller_state, 'unknown') <> o.customer_state) AS is_interstate,
    o.n_sellers                         AS order_n_sellers,
    o.is_multi_seller,
    o.order_status,
    o.purchase_ts,
    o.purchase_date,
    o.purchase_month,
    o.approved_ts,
    o.main_category,
    o.is_delivered,
    o.is_late,
    o.delivery_days,
    o.delay_days,
    o.has_review,
    o.review_score,
    o.is_low_review
FROM so
LEFT JOIN stg_sellers s USING (seller_id)
LEFT JOIN mart_orders o USING (order_id);
