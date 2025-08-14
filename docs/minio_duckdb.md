# MinIO + DuckDB Data Lake

This repo includes a local MinIO (S3-compatible) object store and a DuckDB httpfs config so we can query Parquet data directly without a separate database server.

## Quick start

1. Copy `.env.example` to `.env` and edit as needed.
2. Start MinIO via Docker Compose:
   - `docker compose up -d`
   - Console at http://localhost:9101 (creds from `.env`)
3. Create the bucket via `mc` container:
   - `docker run --rm --network antman -e MC_HOST_antman=http://$MINIO_ROOT_USER:$MINIO_ROOT_PASSWORD@minio:9000 quay.io/minio/mc mb -p antman/$MINIO_BUCKET || true`
   - `docker run --rm --network antman -e MC_HOST_antman=http://$MINIO_ROOT_USER:$MINIO_ROOT_PASSWORD@minio:9000 quay.io/minio/mc ls antman/$MINIO_BUCKET`
4. Use DuckDB (Python or CLI) and run `configs/duckdb_init.sql` if needed.

## Suggested S3 layout

s3://$MINIO_BUCKET/silver/symbol=SPY/resolution=1min/dt=YYYY-MM-DD/part-*.parquet

This matches the example view in `configs/duckdb_init.sql` (`lake.spy_1min`). You can then query:

SELECT * FROM lake.spy_1min WHERE dt='2024-01-05' ORDER BY ts;

## Notes
- Use path-style URLs with MinIO (set `s3_url_style=path`).
- For programmatic access, export the env vars in `.env` before launching notebooks or Python scripts so DuckDB picks them up.
- If you don't have the duckdb CLI, you can run the same SQL from Python using `duckdb.sql("...")` after installing the `duckdb` package.
