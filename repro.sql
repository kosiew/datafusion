CREATE EXTERNAL TABLE categories_raw STORED AS PARQUET LOCATION 's3://fsq-os-places-us-east-1/release/dt=2025-09-09/categories/parquet/';

CREATE EXTERNAL TABLE places STORED AS PARQUET LOCATION 's3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/';


WITH categories_arr AS (
    SELECT array_agg(category_id) AS category_ids FROM categories_raw LIMIT 500
)
SELECT COUNT(*)
    FROM places p
    WHERE date_refreshed >= CURRENT_DATE - INTERVAL '365 days' AND array_has_any(p.fsq_category_ids, (SELECT category_ids FROM categories_arr));