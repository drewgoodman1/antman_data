"""
Transform silver SPY 1-min parquet files to gold (analytics-ready) parquet files.
- Standardizes columns and types
- Adds session (RTH/AH) and 1-min return
- Output: data/gold/SPY/spy_1min_gold_<date>.parquet
"""

import os
import pandas as pd
import numpy as np
import pytz

SILVER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "silver", "SPY"
)
GOLD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gold", "SPY"
)
os.makedirs(GOLD_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(SILVER_DIR) if f.endswith(".parquet")])
if not files:
    raise FileNotFoundError(f"No silver files found in {SILVER_DIR}")

ny = pytz.timezone("America/New_York")


def make_gold(df):
    gold = df.copy()
    # Standardize column names
    gold = gold.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "num_trades": "num_trades",
            "vwap": "vwap",
        }
    )
    # Add session label (RTH/AH)
    gold.index = pd.to_datetime(gold.index, utc=True)
    gold["et"] = gold.index.tz_convert(ny)
    gold["session"] = np.where(
        (
            (gold["et"].dt.hour > 9)
            | ((gold["et"].dt.hour == 9) & (gold["et"].dt.minute >= 30))
        )
        & (
            (gold["et"].dt.hour < 16)
            | ((gold["et"].dt.hour == 16) & (gold["et"].dt.minute == 0))
        ),
        "RTH",
        "AH",
    )
    # Add returns
    gold["return_1m"] = gold["close"].pct_change()
    # Drop timezone for export
    gold = gold.reset_index()
    gold.rename(columns={gold.columns[0]: "utc_timestamp"}, inplace=True)
    # Reorder columns
    cols = [
        "utc_timestamp",
        "et",
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "num_trades",
        "vwap",
        "return_1m",
    ]
    gold = gold[cols]
    return gold


for f in files:
    date_str = f.split("_")[-1].replace(".parquet", "")
    path = os.path.join(SILVER_DIR, f)
    df = pd.read_parquet(path)
    gold_df = make_gold(df)
    out_path = os.path.join(GOLD_DIR, f"spy_1min_gold_{date_str}.parquet")
    gold_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(gold_df)} rows)")
