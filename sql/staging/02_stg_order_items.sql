-- One row per (order_id, order_item_id). Money as DECIMAL to keep sums exact.
CREATE OR REPLACE TABLE stg_order_items AS
SELECT
    trim(order_id)                                   AS order_id,
    TRY_CAST(order_item_id AS INTEGER)               AS order_item_id,
    trim(product_id)                                 AS product_id,
    trim(seller_id)                                  AS seller_id,
    TRY_CAST(shipping_limit_date AS TIMESTAMP)       AS shipping_limit_ts,
    TRY_CAST(price AS DECIMAL(18, 2))                AS price,
    TRY_CAST(freight_value AS DECIMAL(18, 2))        AS freight_value
FROM raw_order_items;
