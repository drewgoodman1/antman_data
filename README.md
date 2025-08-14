# antman_data

Data engineering sandbox for market data using DuckDB over Parquet with a local MinIO (S3-compatible) data lake.

## Prerequisites
- Linux, macOS, or WSL
- Docker + Docker Compose
- Python 3.10+ (local venv recommended)

## 1) Clone and set up Python env
```zsh
# from project root
python -m venv antman_env
source antman_env/bin/activate
pip install -r requirements_essential.txt -r requirements.txt
```

## 2) Configure environment
Copy and edit `.env.example` → `.env`.

Important values:
- MINIO_ROOT_USER / MINIO_ROOT_PASSWORD
- MINIO_PORT=9100, MINIO_CONSOLE_PORT=9101 (avoid conflicts)
- MINIO_BUCKET=antman-lake
- S3_ENDPOINT_URL=127.0.0.1:9100 (no http://)
- S3_USE_SSL=false

## 3) Start MinIO and create the bucket
```zsh
make up            # start MinIO
make bucket        # create $MINIO_BUCKET
make console       # show console URL
```
MinIO console: http://localhost:9101

## 4) Load sample data to the lake
Option A (CLI via mc container):
```zsh
# one-day SPY parquet local → MinIO partitioned path
docker run --rm --network antman \
  -e MC_HOST_antman=http://$MINIO_ROOT_USER:$MINIO_ROOT_PASSWORD@minio:9000 \
  -v "$PWD/data/bronze/SPY:/src:ro" quay.io/minio/mc \
  cp /src/spy_1min_2025-01-02.parquet \
  antman/$MINIO_BUCKET/silver/symbol=SPY/resolution=1min/dt=2025-01-02/part-000.parquet
```

Option B (Python/DuckDB):
```zsh
source antman_env/bin/activate
DAY=2025-01-02 python scripts/upload_spy_day_to_minio.py
```

## 5) Query the lake with DuckDB
```zsh
python - <<'PY'
import os, duckdb
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_url_style='path';")
con.execute("SET s3_use_ssl='false';")
con.execute("SET s3_region='us-east-1';")
con.execute("SET s3_endpoint='127.0.0.1:9100';")
ak = os.getenv('S3_ACCESS_KEY_ID') or os.getenv('MINIO_ROOT_USER') or 'minioadmin'
sk = os.getenv('S3_SECRET_ACCESS_KEY') or os.getenv('MINIO_ROOT_PASSWORD') or 'minioadmin'
con.execute("SET s3_access_key_id=$1;", [ak])
con.execute("SET s3_secret_access_key=$1;", [sk])
print(con.execute("""
SELECT COUNT(*)
FROM read_parquet('s3://%s/silver/symbol=SPY/resolution=1min/dt=2025-01-02/*.parquet')
""" % os.getenv('MINIO_BUCKET','antman-lake')).fetchone())
PY
```

## 6) Notebooks
- See `notebooks/` for dashboards and comparisons (pandas vs DuckDB).
- If Plotly charts don’t render in VS Code, set a renderer in the notebook (pio.renderers.default).

## Troubleshooting
- Port 9000/9001 already in use: set 9100/9101 in `.env` (as shown) and re-run `make up`.
- DuckDB S3 error with `//localhost`: set `SET s3_endpoint='127.0.0.1:9100'` (no scheme) and ensure `s3_url_style='path'`.
- Bucket not listed: run `make bucket` (idempotent) or check credentials in `.env`.

## Next steps
- Wire the Alpaca fetchers to write daily bars directly to MinIO paths.
- Add dbt models or DuckDB views under `configs/duckdb_init.sql` for common queries.
