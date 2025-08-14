-- DuckDB httpfs + MinIO (S3) configuration
-- This file is safe to run multiple times.

INSTALL httpfs;
LOAD httpfs;

-- Configure S3/MinIO. DuckDB will also read these from environment variables.
SET s3_endpoint='${S3_ENDPOINT_URL}';
SET s3_access_key_id='${S3_ACCESS_KEY_ID}';
SET s3_secret_access_key='${S3_SECRET_ACCESS_KEY}';
SET s3_region='${S3_REGION}';
SET s3_use_ssl='${S3_USE_SSL}';
SET s3_url_style='path'; -- path-style is recommended for MinIO

CREATE SCHEMA IF NOT EXISTS lake;

-- Example view over partitioned Parquet in MinIO
-- Adjust the path to match your partition scheme.
-- Suggested layout:
-- s3://${MINIO_BUCKET}/silver/symbol=SPY/resolution=1min/dt=YYYY-MM-DD/part-*.parquet
CREATE OR REPLACE VIEW lake.spy_1min AS
SELECT
  COALESCE(utc_timestamp, "timestamp", datetime) AS ts,
  open, high, low, close, volume,
  try_cast(regexp_extract(filename, 'symbol=([^/]+)', 1) AS VARCHAR) AS symbol,
  try_cast(regexp_extract(filename, 'dt=([0-9\-]+)', 1) AS DATE) AS dt
FROM read_parquet(
  's3://${MINIO_BUCKET}/silver/symbol=SPY/resolution=1min/dt=*/part-*.parquet',
  hive_partitioning=true,
  union_by_name=true,
  filename=true
)
ORDER BY ts NULLS LAST;
