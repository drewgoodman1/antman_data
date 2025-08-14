import os
import sys
from datetime import date
import duckdb
import pandas as pd

# Inputs
DAY = os.environ.get("DAY", "2025-01-02")  # YYYY-MM-DD
TICKER = os.environ.get("TICKER", "SPY")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(PROJECT_ROOT, "data", "bronze", TICKER)
LOCAL_FILE = os.path.join(LOCAL_DIR, f"spy_1min_{DAY}.parquet")

# MinIO / S3 env
BUCKET = os.environ.get("MINIO_BUCKET", "antman-lake")
S3_PREFIX = f"s3://{BUCKET}/silver/symbol={TICKER}/resolution=1min/dt={DAY}"

# Validate local file
if not os.path.exists(LOCAL_FILE):
    print(f"Local file not found: {LOCAL_FILE}")
    sys.exit(1)

# Configure DuckDB httpfs from environment
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_url_style='path';")
endpoint = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9100")
use_ssl = os.environ.get("S3_USE_SSL", "false")
access_key = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("MINIO_ROOT_USER")
secret_key = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get(
    "MINIO_ROOT_PASSWORD"
)
con.execute(f"SET s3_endpoint='{endpoint}';")
con.execute(f"SET s3_use_ssl='{use_ssl}';")
if access_key:
    con.execute(f"SET s3_access_key_id='{access_key}';")
if secret_key:
    con.execute(f"SET s3_secret_access_key='{secret_key}';")

# Read local parquet and write to MinIO in partitioned layout
print(f"Uploading {LOCAL_FILE} -> {S3_PREFIX}")
df = pd.read_parquet(LOCAL_FILE)
# Normalize timestamp column name
if "utc_timestamp" in df.columns:
    df = df.rename(columns={"utc_timestamp": "ts"})
elif "timestamp" in df.columns:
    df = df.rename(columns={"timestamp": "ts"})
elif "datetime" in df.columns:
    df = df.rename(columns={"datetime": "ts"})

# Write to MinIO with a single part file for the day
con.execute("CREATE SCHEMA IF NOT EXISTS lake;")
con.register("bars", df)
con.execute(
    f"""
    COPY (SELECT * FROM bars)
    TO '{S3_PREFIX}/part-000.parquet'
    (FORMAT PARQUET);
    """
)

print("Done.")
