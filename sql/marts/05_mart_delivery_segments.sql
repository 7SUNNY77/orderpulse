-- mart_delivery_segments: additive counts per (purchase month, segment type, segment value).
-- Only additive quantities are stored so months can be summed; medians are
-- computed on demand from the base marts.
--   customer_state: order grain (each order once)
--   category      : order x category grain (an order with 2 categories counts in both)
--   seller        : seller x order grain (a multi-seller order counts for each seller)
CREATE OR REPLACE TABLE mart_delivery_segments AS
WITH order_cat AS (
    SELECT DISTINCT i.order_id, coalesce(p.category_en, 'unknown') AS segment_value
    FROM stg_order_items i
    LEFT JOIN stg_products p USING (product_id)
),
units AS (
    SELECT 'customer_state' AS segment_type, customer_state AS segment_value,
           purchase_month, order_id, is_delivered, is_late, has_review, review_score, is_low_review,
           items_value
    FROM mart_orders
    UNION ALL
    SELECT 'category', c.segment_value,
           o.purchase_month, o.order_id, o.is_delivered, o.is_late, o.has_review, o.review_score,
           o.is_low_review, NULL::DECIMAL(18,2)
    FROM order_cat c JOIN mart_orders o USING (order_id)
    UNION ALL
    SELECT 'seller', seller_id,
           purchase_month, order_id, is_delivered, is_late, has_review, review_score, is_low_review,
           items_value
    FROM mart_seller_orders
)
SELECT
    purchase_month,
    segment_type,
    segment_value,
    count(*)                                         AS n_orders,
    count(*) FILTER (WHERE is_delivered)             AS n_delivered,
    count(*) FILTER (WHERE is_late)                  AS n_late,
    count(*) FILTER (WHERE has_review)               AS n_reviews,
    count(*) FILTER (WHERE is_low_review)            AS n_low_reviews,
    sum(review_score)                                AS sum_review_score,
    sum(items_value)                                 AS items_value
FROM units
GROUP BY ALL;
