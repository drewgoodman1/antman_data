import os
import sys
import time
import argparse
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, Dict, Any

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.fs import S3FileSystem
from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in {"1", "true", "yes", "y"}


def s3_fs_from_env() -> Tuple[S3FileSystem, str]:
    """Create an S3FileSystem for MinIO using env vars and return fs and bucket."""
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    access = os.getenv("S3_ACCESS_KEY_ID")
    secret = os.getenv("S3_SECRET_ACCESS_KEY")
    region = os.getenv("S3_REGION", "us-east-1")
    use_ssl = _env_bool("S3_USE_SSL", False)
    bucket = os.getenv("MINIO_BUCKET", "antman-lake")

    if not access or not secret:
        raise RuntimeError("Missing S3_ACCESS_KEY_ID or S3_SECRET_ACCESS_KEY in environment")

    fs = S3FileSystem(
        access_key=access,
        secret_key=secret,
        region=region,
        scheme="https" if use_ssl else "http",
        endpoint_override=endpoint,
    )
    return fs, bucket


def fetch_alpaca_bars(symbol: str, start_iso: str, end_iso: str, timeframe: str, api_key: str, api_secret: str) -> pd.DataFrame:
    base_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "accept": "application/json",
    }
    params: Dict[str, Any] = {"start": start_iso, "end": end_iso, "timeframe": timeframe, "limit": 10000}
    all_bars = []
    next_token: Optional[str] = None
    attempt = 0
    while True:
        qp = dict(params)
        if next_token:
            qp["page_token"] = next_token
        resp = requests.get(base_url, headers=headers, params=qp, timeout=30)
        if resp.status_code == 429 and attempt < 3:
            wait = 2 ** attempt
            time.sleep(wait)
            attempt += 1
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Alpaca error {resp.status_code}: {resp.text}")
        data = resp.json()
        bars = data.get("bars") or []
        all_bars.extend(bars)
        next_token = data.get("next_page_token")
        if not next_token:
            break
    if not all_bars:
        return pd.DataFrame()
    return pd.DataFrame(all_bars)


def normalize_bars(df: pd.DataFrame, symbol: str, resolution: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])  # drop rows with bad timestamps
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["symbol"] = symbol
    df["resolution"] = resolution
    df["dt"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["ts"] = df["ts"].dt.round("ms")
    return df


def write_day_to_minio(df: pd.DataFrame, fs: S3FileSystem, bucket: str, symbol: str, resolution: str, day: date) -> str:
    if df.empty:
        return ""
    day_str = day.strftime("%Y-%m-%d")
    day_df = df[df["dt"] == day_str]
    if day_df.empty:
        return ""
    table = pa.Table.from_pandas(day_df[["ts", "open", "high", "low", "close", "volume"]], preserve_index=False)
    out_path = f"s3://{bucket}/gold/symbol={symbol}/resolution={resolution}/dt={day_str}/part-{symbol}-{day_str}.parquet"
    pq.write_table(table, out_path, filesystem=fs, compression="zstd")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Alpaca bars directly to MinIO as partitioned Parquet (gold)")
    p.add_argument("--symbol", default="SPY", help="Ticker symbol, e.g. SPY")
    p.add_argument("--timeframe", default="1Min", choices=["1Min", "5Min", "15Min"], help="Bar timeframe")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", help="End date (YYYY-MM-DD); defaults to start date")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    symbol = args.symbol.upper()
    resolution = args.timeframe.lower()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else start_dt

    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        print("ERROR: Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in environment", file=sys.stderr)
        return 2

    fs, bucket = s3_fs_from_env()

    cur = start_dt
    total_rows = 0
    written = []
    while cur <= end_dt:
        start_iso = f"{cur.isoformat()}T00:00:00Z"
        end_iso = f"{cur.isoformat()}T23:59:59Z"
        try:
            raw = fetch_alpaca_bars(symbol, start_iso, end_iso, args.timeframe, api_key, api_secret)
        except Exception as e:
            print(f"Fetch failed for {cur}: {e}")
            cur += timedelta(days=1)
            continue
        norm = normalize_bars(raw, symbol, resolution)
        total_rows += len(norm)
        out = write_day_to_minio(norm, fs, bucket, symbol, resolution, cur)
        if out:
            print(f"Wrote {len(norm[norm['dt']==cur.strftime('%Y-%m-%d')])} rows -> {out}")
            written.append(out)
        else:
            print(f"No rows for {cur}")
        cur += timedelta(days=1)

    print(f"Done. Total rows: {total_rows}. Files: {len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
