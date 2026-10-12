-- mart_order_items: one row per order item, enriched with product/seller attributes
-- and order-level outcome columns (repeated per item: do NOT sum outcomes here).
CREATE OR REPLACE TABLE mart_order_items AS
SELECT
    i.order_id,
    i.order_item_id,
    i.product_id,
    i.seller_id,
    i.shipping_limit_ts,
    i.price,
    i.freight_value,
    coalesce(p.category_en, 'unknown')         AS category,
    p.weight_g,
    p.volume_cm3,
    coalesce(s.seller_state, 'unknown')        AS seller_state,
    s.seller_city,
    o.purchase_ts,
    o.purchase_month,
    o.customer_state,
    o.order_status,
    o.is_delivered,
    o.is_late
FROM stg_order_items i
LEFT JOIN stg_products p USING (product_id)
LEFT JOIN stg_sellers  s USING (seller_id)
LEFT JOIN mart_orders  o USING (order_id);
