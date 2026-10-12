-- ml_labels: outcome table kept SEPARATE from the feature matrix.
-- label_available_at = moment the outcome became observable (actual delivery).
-- A label may be used for training at time T only if label_available_at < T.
CREATE OR REPLACE TABLE ml_labels AS
SELECT
    order_id,
    approved_ts                                 AS prediction_time,
    CASE WHEN ml_exclusion_reason IS NULL THEN is_late::INTEGER END AS is_late,
    delivered_ts                                AS label_available_at,
    ml_exclusion_reason IS NULL                 AS ml_eligible,
    ml_exclusion_reason
FROM mart_orders;
