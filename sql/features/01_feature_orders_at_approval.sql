-- feature_orders_at_approval: one row per APPROVED order with features that are
-- assumed to be known at prediction_time = order_approved_at.
--
-- ANTI-LEAKAGE CONTRACT: this file may read only
--   stg_orders  (order_id, customer_id, purchase_ts, approved_ts, estimated_ts)
--   stg_customers, stg_order_items, stg_products, stg_sellers, stg_geo_zip.
-- It must never read delivery/carrier timestamps, order_status, reviews,
-- payments (excluded from fv1) or any mart_* table (marts contain outcomes).
-- tests/test_features.py mutates future fields and checks this table is unchanged.
--
-- Availability assumptions (not proven by an event log, see reports/decisions.md):
--   * the promised date (estimated_ts) is shown to the customer at purchase;
--   * order composition (items, products, sellers, prices, freight) is fixed at approval.
-- seller_id is used only internally to pick the main seller deterministically.
CREATE OR REPLACE TABLE feature_orders_at_approval AS
WITH o AS (
    SELECT o.order_id, o.purchase_ts, o.approved_ts, o.estimated_ts,
           coalesce(c.customer_state, 'unknown') AS customer_state,
           c.customer_zip
    FROM stg_orders o
    LEFT JOIN stg_customers c USING (customer_id)
    WHERE o.approved_ts IS NOT NULL
),
items AS (
    SELECT
        i.order_id, i.order_item_id, i.seller_id, i.product_id, i.price, i.freight_value,
        coalesce(p.category_en, 'unknown')               AS category_en,
        p.weight_g, p.volume_cm3,
        (p.product_id IS NULL OR p.category_missing OR p.weight_g IS NULL OR p.volume_cm3 IS NULL)
                                                         AS unknown_attrs,
        coalesce(s.seller_state, 'unknown')              AS seller_state,
        s.seller_zip
    FROM stg_order_items i
    LEFT JOIN stg_products p USING (product_id)
    LEFT JOIN stg_sellers  s USING (seller_id)
),
items_geo AS (
    SELECT
        it.*,
        o.customer_state,
        CASE WHEN gc.lat IS NOT NULL AND gs.lat IS NOT NULL THEN
            2 * 6371.0 * asin(sqrt(
                power(sin(radians(gs.lat - gc.lat) / 2), 2)
                + cos(radians(gc.lat)) * cos(radians(gs.lat))
                  * power(sin(radians(gs.lng - gc.lng) / 2), 2)))
        END AS dist_km
    FROM items it
    JOIN o USING (order_id)
    LEFT JOIN stg_geo_zip gc ON gc.zip = o.customer_zip
    LEFT JOIN stg_geo_zip gs ON gs.zip = it.seller_zip
),
main_cat AS (
    SELECT order_id, category_en,
           row_number() OVER (PARTITION BY order_id ORDER BY sum(price) DESC, category_en) AS rn
    FROM items GROUP BY order_id, category_en
),
main_seller AS (
    SELECT order_id, seller_state,
           row_number() OVER (PARTITION BY order_id ORDER BY sum(price) DESC, seller_id) AS rn
    FROM items GROUP BY order_id, seller_id, seller_state
),
agg AS (
    SELECT
        order_id,
        count(*)                                        AS n_items,
        count(DISTINCT product_id)                      AS n_products,
        count(DISTINCT seller_id)                       AS n_sellers,
        CAST(sum(price) AS DOUBLE)                      AS items_value,
        CAST(sum(freight_value) AS DOUBLE)              AS freight_value,
        CAST(avg(price) AS DOUBLE)                      AS avg_item_price,
        CAST(max(price) AS DOUBLE)                      AS max_item_price,
        count(DISTINCT category_en)                     AS n_categories,
        sum(weight_g)                                   AS total_weight_g,
        max(weight_g)                                   AS max_weight_g,
        sum(volume_cm3)                                 AS total_volume_cm3,
        max(volume_cm3)                                 AS max_volume_cm3,
        avg(unknown_attrs::INTEGER)                     AS share_unknown_attrs,
        count(DISTINCT seller_state)                    AS n_seller_states,
        max((seller_state <> customer_state)::INTEGER)  AS is_interstate,
        avg(dist_km)                                    AS dist_mean_km,
        max(dist_km)                                    AS dist_max_km,
        max((dist_km IS NULL)::INTEGER)                 AS geo_missing
    FROM items_geo
    GROUP BY order_id
)
SELECT
    o.order_id,
    o.approved_ts                                               AS prediction_time,
    o.estimated_ts                                              AS promised_ts,
    -- order
    a.n_items, a.n_products, a.n_sellers,
    a.items_value, a.freight_value,
    a.freight_value / nullif(a.items_value, 0)                  AS freight_ratio,
    a.avg_item_price, a.max_item_price,
    epoch(o.approved_ts - o.purchase_ts) / 3600.0               AS hours_to_approval,
    epoch(o.estimated_ts - o.approved_ts) / 86400.0             AS promised_days,
    dayofweek(o.purchase_ts)                                    AS purchase_dow,
    dayofweek(o.approved_ts)                                    AS approval_dow,
    hour(o.approved_ts)                                         AS approval_hour,
    month(o.approved_ts)                                        AS approval_month,
    -- geography
    o.customer_state,
    coalesce(ms.seller_state, 'unknown')                        AS main_seller_state,
    a.is_interstate, a.n_seller_states,
    a.dist_mean_km, a.dist_max_km,
    coalesce(a.geo_missing, 1)                                  AS geo_missing,
    -- products
    coalesce(mc.category_en, 'unknown')                         AS main_category,
    a.n_categories,
    a.total_weight_g, a.max_weight_g, a.total_volume_cm3, a.max_volume_cm3,
    a.share_unknown_attrs
FROM o
LEFT JOIN agg a USING (order_id)
LEFT JOIN main_cat mc ON mc.order_id = o.order_id AND mc.rn = 1
LEFT JOIN main_seller ms ON ms.order_id = o.order_id AND ms.rn = 1;
