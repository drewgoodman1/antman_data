import os
import sys
import time
import argparse
from datetime import datetime, timedelta, date
import requests
import pandas as pd
import duckdb
from dotenv import load_dotenv

# Map Alpaca bar fields to our canonical schema
FIELD_MAP = {
    "t": "ts",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch 1min bars from Alpaca and write to MinIO as Parquet"
    )
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="YYYY-MM-DD (inclusive); defaults to --start")
    return p.parse_args()


def daterange(start_dt: date, end_dt: date):
    cur = start_dt
    one = timedelta(days=1)
    while cur <= end_dt:
        yield cur
        cur += one


def load_env():
    load_dotenv()
    return {
        "API_KEY": os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY"),
        "API_SECRET": os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY"),
        "BASE_URL": "https://data.alpaca.markets/v2/stocks",  # we'll append /{ticker}/bars
        "BUCKET": os.getenv("MINIO_BUCKET", "antman-lake"),
        # s3 endpoint: prefer host:port without scheme; strip if present
        "S3_ENDPOINT": (os.getenv("S3_ENDPOINT_URL") or "127.0.0.1:9100")
        .replace("http://", "")
        .replace("https://", ""),
        "S3_REGION": os.getenv("S3_REGION", "us-east-1"),
        "S3_USE_SSL": os.getenv("S3_USE_SSL", "false"),
        "S3_ACCESS_KEY": os.getenv("S3_ACCESS_KEY_ID")
        or os.getenv("MINIO_ROOT_USER", "minioadmin"),
        "S3_SECRET_KEY": os.getenv("S3_SECRET_ACCESS_KEY")
        or os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
    }


def fetch_day(
    ticker: str, day: date, api_key: str, api_secret: str, base_url: str
) -> pd.DataFrame:
    # Use US/Eastern RTH window. For simplicity, use -05:00 offset; adjust if you need DST-aware windows.
    day_str = day.strftime("%Y-%m-%d")
    start = f"{day_str}T09:30:00-05:00"
    end = f"{day_str}T16:00:00-05:00"
    url = f"{base_url}/{ticker}/bars"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "accept": "application/json",
    }
    params = {"start": start, "end": end, "timeframe": "1Min", "limit": 10000}

    all_bars = []
    next_token = None
    while True:
        if next_token:
            params["page_token"] = next_token
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Alpaca error {resp.status_code}: {resp.text}")
        data = resp.json()
        bars = data.get("bars", []) or []
        all_bars.extend(bars)
        next_token = data.get("next_page_token")
        if not next_token:
            break
        time.sleep(0.1)  # gentle on pagination

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    # Rename fields to canonical names
    for src, dst in FIELD_MAP.items():
        if src in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    # Parse timestamp
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    # Ensure column order
    cols = [
        c for c in ["ts", "open", "high", "low", "close", "volume"] if c in df.columns
    ]
    return df[cols] if cols else df


def write_day_to_minio(df: pd.DataFrame, ticker: str, day: date, env: dict):
    if df.empty:
        print(f"No data for {day}")
        return
    # Configure DuckDB httpfs
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_url_style='path';")
    con.execute(f"SET s3_use_ssl='{env['S3_USE_SSL']}';")
    con.execute(f"SET s3_region='{env['S3_REGION']}';")
    con.execute(f"SET s3_endpoint='{env['S3_ENDPOINT']}';")
    con.execute(f"SET s3_access_key_id='{env['S3_ACCESS_KEY']}';")
    con.execute(f"SET s3_secret_access_key='{env['S3_SECRET_KEY']}';")

    s3_prefix = f"s3://{env['BUCKET']}/silver/symbol={ticker}/resolution=1min/dt={day.strftime('%Y-%m-%d')}"
    # Register and write as a single file for the day
    con.register("bars", df)
    con.execute(
        f"""
        COPY (SELECT * FROM bars)
        TO '{s3_prefix}/part-000.parquet'
        (FORMAT PARQUET);
        """
    )
    print(f"Wrote {len(df)} rows to {s3_prefix}/part-000.parquet")


def main():
    args = parse_args()
    env = load_env()
    if not env["API_KEY"] or not env["API_SECRET"]:
        print("Missing Alpaca API credentials in environment (.env)")
        sys.exit(1)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else start_dt

    for d in daterange(start_dt, end_dt):
        try:
            df = fetch_day(
                args.ticker, d, env["API_KEY"], env["API_SECRET"], env["BASE_URL"]
            )
            write_day_to_minio(df, args.ticker, d, env)
        except Exception as e:
            print(f"Error on {d}: {e}")
            # continue to next day


if __name__ == "__main__":
    main()
