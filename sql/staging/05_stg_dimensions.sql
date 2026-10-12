-- Customers: one row per customer_id. Zip prefixes are text with leading zeros.
CREATE OR REPLACE TABLE stg_customers AS
SELECT
    trim(customer_id)                                  AS customer_id,
    trim(customer_unique_id)                           AS customer_unique_id,
    lpad(trim(customer_zip_code_prefix), 5, '0')       AS customer_zip,
    lower(trim(customer_city))                         AS customer_city,
    upper(trim(customer_state))                        AS customer_state
FROM raw_customers;

CREATE OR REPLACE TABLE stg_sellers AS
SELECT
    trim(seller_id)                                    AS seller_id,
    lpad(trim(seller_zip_code_prefix), 5, '0')         AS seller_zip,
    lower(trim(seller_city))                           AS seller_city,
    upper(trim(seller_state))                          AS seller_state
FROM raw_sellers;

-- Products: missing category -> 'unknown'; untranslated categories keep the original name.
CREATE OR REPLACE TABLE stg_products AS
SELECT
    trim(p.product_id)                                                    AS product_id,
    coalesce(nullif(trim(p.product_category_name), ''), 'unknown')        AS category,
    coalesce(nullif(trim(t.product_category_name_english), ''),
             nullif(trim(p.product_category_name), ''), 'unknown')        AS category_en,
    (p.product_category_name IS NULL OR trim(p.product_category_name) = '') AS category_missing,
    TRY_CAST(p.product_photos_qty AS INTEGER)                             AS photos_qty,
    TRY_CAST(p.product_weight_g AS DOUBLE)                                AS weight_g,
    TRY_CAST(p.product_length_cm AS DOUBLE)                               AS length_cm,
    TRY_CAST(p.product_height_cm AS DOUBLE)                               AS height_cm,
    TRY_CAST(p.product_width_cm AS DOUBLE)                                AS width_cm,
    TRY_CAST(p.product_length_cm AS DOUBLE) * TRY_CAST(p.product_height_cm AS DOUBLE)
        * TRY_CAST(p.product_width_cm AS DOUBLE)                          AS volume_cm3
FROM raw_products p
LEFT JOIN (
    -- the translation table is a dimension: keep one row per Portuguese name
    SELECT product_category_name, min(product_category_name_english) AS product_category_name_english
    FROM raw_category_translation
    GROUP BY 1
) t ON trim(t.product_category_name) = trim(p.product_category_name);
