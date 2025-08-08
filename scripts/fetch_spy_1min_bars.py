import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load API credentials
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

BASE_URL = "https://data.alpaca.markets/v2/stocks/SPY/bars"

headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "accept": "application/json",
}


from datetime import timedelta


def daterange(start_dt, end_dt):
    for n in range(int((end_dt - start_dt).days) + 1):
        yield start_dt + timedelta(n)


start_date = datetime.strptime("2025-01-01", "%Y-%m-%d")
end_date = datetime.now()


# Always save to project root's data/bronze/<ticker> directory
TICKER = "SPY"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DIR = os.path.join(PROJECT_ROOT, "data", "bronze", TICKER)
os.makedirs(BRONZE_DIR, exist_ok=True)

for single_date in daterange(start_date, end_date):
    day_start = single_date.strftime("%Y-%m-%dT09:30:00-05:00")
    day_end = single_date.strftime("%Y-%m-%dT16:00:00-05:00")
    params = {"start": day_start, "end": day_end, "timeframe": "1Min", "limit": 10000}
    all_bars = []
    next_token = None
    while True:
        if next_token:
            params["page_token"] = next_token
        response = requests.get(BASE_URL, headers=headers, params=params)
        if response.status_code != 200:
            print(
                f"Error for {single_date.strftime('%Y-%m-%d')}: {response.status_code}"
            )
            print(response.text)
            break
        data = response.json()
        bars = data.get("bars", [])
        if bars is None:
            print(
                f"No 'bars' key or value is null in response for {single_date.strftime('%Y-%m-%d')}"
            )
            print("Raw response:", data)
            break
        all_bars.extend(bars)
        next_token = data.get("next_page_token")
        if not next_token:
            break
    if all_bars:
        df = pd.DataFrame(all_bars)
        out_path = os.path.join(
            BRONZE_DIR, f"spy_1min_{single_date.strftime('%Y-%m-%d')}.parquet"
        )
        print("Saving to:", out_path)
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} bars to {out_path}")
    else:
        print(f"No data fetched for {single_date.strftime('%Y-%m-%d')}")
