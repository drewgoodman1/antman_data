import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# --- Config ---
TICKER = os.getenv("FETCH_TICKER", "NVDA").upper()
START_STR = os.getenv("FETCH_START", "2025-01-01")  # YYYY-MM-DD
END_STR = os.getenv("FETCH_END")  # optional; defaults to now

# Load API credentials
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

BASE_URL = f"https://data.alpaca.markets/v2/stocks/{TICKER}/bars"
HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "accept": "application/json",
}


def daterange(start_dt, end_dt):
    days = (end_dt - start_dt).days
    for n in range(days + 1):
        yield start_dt + timedelta(days=n)


# Dates
start_date = datetime.strptime(START_STR, "%Y-%m-%d")
end_date = datetime.strptime(END_STR, "%Y-%m-%d") if END_STR else datetime.now()

# Save under project root: data/bronze/<TICKER>/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DIR = os.path.join(PROJECT_ROOT, "data", "bronze", TICKER)
os.makedirs(BRONZE_DIR, exist_ok=True)

for single_date in daterange(start_date, end_date):
    # NOTE: This uses a fixed -05:00 offset; for DST-aware windows, compute
    # with zoneinfo("America/New_York") and format offsets per day.
    day_str = single_date.strftime("%Y-%m-%d")
    params = {
        "start": f"{day_str}T09:30:00-05:00",
        "end": f"{day_str}T16:00:00-05:00",
        "timeframe": "1Min",
        "limit": 10000,
    }

    all_bars, next_token = [], None
    while True:
        if next_token:
            params["page_token"] = next_token
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"Error {resp.status_code} on {day_str}: {resp.text}")
            break
        data = resp.json()
        bars = data.get("bars") or []
        all_bars.extend(bars)
        next_token = data.get("next_page_token")
        if not next_token:
            break

    if not all_bars:
        print(f"No data fetched for {day_str}")
        continue

    df = pd.DataFrame(all_bars)
    out_path = os.path.join(BRONZE_DIR, f"{TICKER}_1min_{day_str}.parquet")
    print("Saving to:", out_path)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} bars to {out_path}")
