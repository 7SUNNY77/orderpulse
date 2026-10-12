-- mart_orders: exactly one row per order_id (retrospective analytics grain).
-- Missing child records never drop an order: all joins are LEFT joins to
-- tables that are already unique per order_id.
--
-- Metric definitions (see reports/decisions.md, "Словарь метрик"):
--   delivered cohort : status = 'delivered' AND delivered_ts AND estimated_ts present
--   is_late          : date(delivered_ts) > date(estimated_ts), NULL outside the cohort
--   delivery_days    : (delivered_ts - purchase_ts) / 24h, NULL outside the cohort
--   delay_days       : max(0, date(delivered_ts) - date(estimated_ts)), NULL outside the cohort
CREATE OR REPLACE TABLE mart_orders AS
WITH base AS (
    SELECT
        o.order_id,
        o.customer_id,
        c.customer_unique_id,
        c.customer_zip,
        c.customer_city,
        coalesce(c.customer_state, 'unknown')               AS customer_state,
        o.order_status,
        o.purchase_ts,
        o.approved_ts,
        o.carrier_ts,
        o.delivered_ts,
        o.estimated_ts,
        CAST(o.purchase_ts AS DATE)                          AS purchase_date,
        CAST(date_trunc('month', o.purchase_ts) AS DATE)     AS purchase_month,
        CAST(o.approved_ts AS DATE)                          AS approved_date,
        -- items
        coalesce(i.n_items, 0)                               AS n_items,
        coalesce(i.n_sellers, 0)                             AS n_sellers,
        i.n_products,
        i.n_categories,
        i.items_value,                                       -- NULL when the order has no items
        i.freight_value,
        coalesce(i.main_category, 'unknown')                 AS main_category,
        -- payments (separate from item/freight values)
        coalesce(p.n_payment_records, 0)                     AS n_payment_records,
        p.payment_value,
        p.max_installments,
        p.payment_types,
        p.main_payment_type,
        -- canonical review
        coalesce(r.n_reviews, 0)                             AS n_reviews,
        r.review_id,
        r.review_score,
        r.has_comment_message,
        r.review_creation_ts,
        r.review_answer_ts
    FROM stg_orders o
    LEFT JOIN stg_customers               c USING (customer_id)
    LEFT JOIN int_order_items_agg         i USING (order_id)
    LEFT JOIN int_order_payments_agg      p USING (order_id)
    LEFT JOIN int_order_review_canonical  r USING (order_id)
)
SELECT
    b.*,
    b.items_value + b.freight_value                                    AS items_plus_freight,
    b.payment_value - (b.items_value + b.freight_value)                AS payment_minus_items_freight,
    (b.n_items > 0)                                                    AS has_items,
    (b.n_reviews > 0)                                                  AS has_review,
    (b.n_sellers > 1)                                                  AS is_multi_seller,
    CASE WHEN b.review_score IS NULL THEN NULL ELSE b.review_score <= 2 END AS is_low_review,
    -- delivered cohort and target
    (b.order_status = 'delivered' AND b.delivered_ts IS NOT NULL AND b.estimated_ts IS NOT NULL)
                                                                       AS is_delivered,
    CASE WHEN b.order_status = 'delivered' AND b.delivered_ts IS NOT NULL AND b.estimated_ts IS NOT NULL
         THEN CAST(b.delivered_ts AS DATE) > CAST(b.estimated_ts AS DATE) END   AS is_late,
    CASE WHEN b.order_status = 'delivered' AND b.delivered_ts IS NOT NULL AND b.estimated_ts IS NOT NULL
         THEN epoch(b.delivered_ts - b.purchase_ts) / 86400.0 END     AS delivery_days,
    CASE WHEN b.order_status = 'delivered' AND b.delivered_ts IS NOT NULL AND b.estimated_ts IS NOT NULL
         THEN greatest(0, date_diff('day', CAST(b.estimated_ts AS DATE), CAST(b.delivered_ts AS DATE))) END
                                                                       AS delay_days,
    -- outcome bucket for UI
    CASE
        WHEN b.order_status = 'delivered' AND b.delivered_ts IS NOT NULL AND b.estimated_ts IS NOT NULL
            THEN CASE WHEN CAST(b.delivered_ts AS DATE) > CAST(b.estimated_ts AS DATE) THEN 'late' ELSE 'on_time' END
        WHEN b.order_status = 'canceled'    THEN 'canceled'
        WHEN b.order_status = 'unavailable' THEN 'unavailable'
        WHEN b.order_status = 'delivered'   THEN 'delivered_missing_date'
        ELSE 'not_delivered'
    END                                                                AS outcome,
    -- temporal anomaly flags (kept, never silently dropped)
    coalesce(b.approved_ts  < b.purchase_ts, false)                    AS flag_approved_before_purchase,
    coalesce(b.carrier_ts   < b.approved_ts, false)                    AS flag_carrier_before_approval,
    coalesce(b.delivered_ts < b.carrier_ts, false)                     AS flag_delivered_before_carrier,
    coalesce(b.delivered_ts < b.approved_ts, false)                    AS flag_delivered_before_approval,
    coalesce(b.estimated_ts < b.purchase_ts, false)                    AS flag_estimated_before_purchase,
    coalesce(CAST(b.estimated_ts AS DATE) < CAST(b.approved_ts AS DATE), false) AS flag_estimated_before_approval,
    (b.order_status <> 'delivered' AND b.delivered_ts IS NOT NULL)     AS flag_delivered_date_but_not_delivered,
    -- ML population (prediction_time = approved_ts); first failing rule wins
    CASE
        WHEN b.approved_ts IS NULL                                     THEN 'not_approved'
        WHEN b.estimated_ts IS NULL                                    THEN 'no_estimated_date'
        WHEN b.order_status <> 'delivered' OR b.delivered_ts IS NULL   THEN 'not_delivered'
        WHEN b.approved_ts < b.purchase_ts
          OR b.delivered_ts < b.approved_ts
          OR b.estimated_ts < b.purchase_ts                            THEN 'temporal_error'
        WHEN coalesce(b.n_items, 0) = 0                                THEN 'no_items'
        ELSE NULL
    END                                                                AS ml_exclusion_reason
FROM base b;
