# MinIO + DuckDB Data Lake

This repo includes a local MinIO (S3-compatible) object store and a DuckDB httpfs config so we can query Parquet data directly without a separate database server.

## Quick start

1. Copy `.env.example` to `.env` and edit as needed.
2. Start MinIO:
   - make up
   - make console  # prints URL and credentials
3. Create the bucket:
   - make bucket
4. Open DuckDB CLI with httpfs configured:
   - make duck

## Suggested S3 layout

s3://$MINIO_BUCKET/silver/symbol=SPY/resolution=1min/dt=YYYY-MM-DD/part-*.parquet

This matches the example view in `configs/duckdb_init.sql` (`lake.spy_1min`). You can then query:

SELECT * FROM lake.spy_1min WHERE dt='2024-01-05' ORDER BY ts;

## Notes
- Use path-style URLs with MinIO (set `s3_url_style=path`).
- For programmatic access, export the env vars in `.env` before launching notebooks or Python scripts so DuckDB picks them up.
- If you don't have the duckdb CLI, you can run the same SQL from Python using `duckdb.sql("...")` after installing the `duckdb` package.
