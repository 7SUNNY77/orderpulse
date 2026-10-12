-- Geolocation has many points per zip prefix. Pre-aggregate to exactly one row
-- per prefix: median coordinates of points inside Brazil's bounding box.
-- Points outside the box are dropped here and counted in the quality report.
-- Bounds are injected from configs/data.yaml (geo_bounds) by build_marts.py.
CREATE OR REPLACE TABLE stg_geo_zip AS
WITH typed AS (
    SELECT
        lpad(trim(geolocation_zip_code_prefix), 5, '0') AS zip,
        TRY_CAST(geolocation_lat AS DOUBLE)             AS lat,
        TRY_CAST(geolocation_lng AS DOUBLE)             AS lng
    FROM raw_geolocation
)
SELECT
    zip,
    median(lat)  AS lat,
    median(lng)  AS lng,
    count(*)     AS n_points
FROM typed
WHERE lat BETWEEN {lat_min} AND {lat_max}
  AND lng BETWEEN {lng_min} AND {lng_max}
GROUP BY zip;
