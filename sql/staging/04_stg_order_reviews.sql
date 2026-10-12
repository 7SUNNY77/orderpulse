-- One row per (review_id, order_id): review_id alone is NOT unique in Olist
-- (the same review can be linked to several orders). Comment texts are kept
-- only as presence flags; their content is never interpreted.
CREATE OR REPLACE TABLE stg_order_reviews AS
SELECT
    trim(review_id)                                   AS review_id,
    trim(order_id)                                    AS order_id,
    TRY_CAST(review_score AS INTEGER)                 AS review_score,
    (review_comment_title IS NOT NULL AND trim(review_comment_title) <> '')     AS has_comment_title,
    (review_comment_message IS NOT NULL AND trim(review_comment_message) <> '') AS has_comment_message,
    TRY_CAST(review_creation_date AS TIMESTAMP)       AS review_creation_ts,
    TRY_CAST(review_answer_timestamp AS TIMESTAMP)    AS review_answer_ts
FROM raw_order_reviews;
