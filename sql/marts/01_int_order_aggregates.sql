-- Each child table is aggregated to the order grain SEPARATELY before any join,
-- so items x payments x reviews can never multiply rows or sums.

-- Items -> order. Main category = category with the largest summed item price;
-- ties resolved by category name (deterministic).
CREATE OR REPLACE TABLE int_order_items_agg AS
WITH items AS (
    SELECT i.*, coalesce(p.category_en, 'unknown') AS category_en
    FROM stg_order_items i
    LEFT JOIN stg_products p USING (product_id)
),
cat AS (
    SELECT order_id, category_en,
           row_number() OVER (PARTITION BY order_id ORDER BY sum(price) DESC, category_en ASC) AS rn
    FROM items
    GROUP BY order_id, category_en
)
SELECT
    i.order_id,
    count(*)                         AS n_items,
    count(DISTINCT i.product_id)     AS n_products,
    count(DISTINCT i.seller_id)      AS n_sellers,
    sum(i.price)                     AS items_value,
    sum(i.freight_value)             AS freight_value,
    count(DISTINCT coalesce(i.category_en, 'unknown')) AS n_categories,
    any_value(c.category_en)         AS main_category
FROM items i
LEFT JOIN cat c ON c.order_id = i.order_id AND c.rn = 1
GROUP BY i.order_id;

-- Payments -> order. Payment amounts are reported separately from item/freight values.
CREATE OR REPLACE TABLE int_order_payments_agg AS
SELECT
    order_id,
    count(*)                                         AS n_payment_records,
    sum(payment_value)                               AS payment_value,
    max(payment_installments)                        AS max_installments,
    string_agg(DISTINCT payment_type, ',' ORDER BY payment_type) AS payment_types,
    arg_max(payment_type, payment_value)             AS main_payment_type
FROM stg_order_payments
GROUP BY order_id;

-- Reviews -> order. Canonical review = the latest by review_creation_ts, then
-- review_answer_ts, then review_id (descending) for a deterministic tie-break.
CREATE OR REPLACE TABLE int_order_review_canonical AS
WITH ranked AS (
    SELECT
        r.*,
        count(*)     OVER (PARTITION BY order_id) AS n_reviews,
        row_number() OVER (
            PARTITION BY order_id
            ORDER BY review_creation_ts DESC NULLS LAST,
                     review_answer_ts   DESC NULLS LAST,
                     review_id          DESC
        ) AS rn
    FROM stg_order_reviews r
)
SELECT order_id, review_id, review_score, has_comment_title, has_comment_message,
       review_creation_ts, review_answer_ts, n_reviews
FROM ranked
WHERE rn = 1;
