WASP offline viewer

This small Streamlit app reads minute-bar parquet files stored under data/{silver,gold,bronze}/SPY and draws candlesticks with EMA9/EMA21, RSI, ATR and the last zigzag fib zone (50/61.8).

How to run

1. From the repository root, create a virtualenv or use your existing one.

2. Install the small set of dependencies (you may already have pandas/numpy in the project's env):

   pip install -r apps/requirements_streamlit.txt

3. Run the app:

   streamlit run apps/streamlit_wasp_offline.py

The app will list available parquet files under data/*/SPY and let you pick a day to inspect. It is intentionally offline and reads local parquet files (these are the MinIO sync copies in this repo).
