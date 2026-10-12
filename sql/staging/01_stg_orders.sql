-- One row per order (uniqueness of order_id is asserted by the quality audit and build checks).
CREATE OR REPLACE TABLE stg_orders AS
SELECT
    trim(order_id)                                         AS order_id,
    trim(customer_id)                                      AS customer_id,
    lower(trim(order_status))                              AS order_status,
    TRY_CAST(order_purchase_timestamp      AS TIMESTAMP)   AS purchase_ts,
    TRY_CAST(order_approved_at             AS TIMESTAMP)   AS approved_ts,
    TRY_CAST(order_delivered_carrier_date  AS TIMESTAMP)   AS carrier_ts,
    TRY_CAST(order_delivered_customer_date AS TIMESTAMP)   AS delivered_ts,
    TRY_CAST(order_estimated_delivery_date AS TIMESTAMP)   AS estimated_ts
FROM raw_orders;
