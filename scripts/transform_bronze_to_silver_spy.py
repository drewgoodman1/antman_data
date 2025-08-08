import os
import pandas as pd

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DIR = os.path.join(PROJECT_ROOT, "data", "bronze", "SPY")
SILVER_DIR = os.path.join(PROJECT_ROOT, "data", "silver", "SPY")
os.makedirs(SILVER_DIR, exist_ok=True)

# List all bronze files
files = sorted([f for f in os.listdir(BRONZE_DIR) if f.endswith(".parquet")])

for f in files:
    bronze_path = os.path.join(BRONZE_DIR, f)
    df = pd.read_parquet(bronze_path)

    # --- Silver Transformations ---
    # 1. Parse timestamp to datetime and set as index
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t").set_index("t")
    # 2. Remove duplicates (if any)
    df = df[~df.index.duplicated(keep="first")]
    # 3. (Optional) Remove rows with missing/NaN values
    df = df.dropna()
    # 4. (Optional) Rename columns for clarity
    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "num_trades",
            "vw": "vwap",
        }
    )

    # Save to silver
    silver_path = os.path.join(SILVER_DIR, f)
    df.to_parquet(silver_path)
    print(f"Saved silver file: {silver_path} ({len(df)} rows)")
