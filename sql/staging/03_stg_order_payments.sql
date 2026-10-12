-- One row per (order_id, payment_sequential).
CREATE OR REPLACE TABLE stg_order_payments AS
SELECT
    trim(order_id)                                   AS order_id,
    TRY_CAST(payment_sequential AS INTEGER)          AS payment_sequential,
    lower(trim(payment_type))                        AS payment_type,
    TRY_CAST(payment_installments AS INTEGER)        AS payment_installments,
    TRY_CAST(payment_value AS DECIMAL(18, 2))        AS payment_value
FROM raw_order_payments;
