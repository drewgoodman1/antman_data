# antman_data – Stock Algorithm Discovery via Data Engineering Pipeline

## 📌 Project Objective
Use real-time and historical stock market data to identify and test profitable trading algorithms using a robust, cloud-ready data engineering pipeline.

---

## 📊 Key Goals
- Ingest large volumes of structured market data (OHLCV, tickers, volume, spread, etc.)
- Clean and organize data using the Medallion architecture (Bronze → Silver → Gold)
- Engineer and test indicators (e.g., SMA, RSI, breakouts, spread-based signals) to inform trading logic
- Backtest multiple strategies and measure profitability
- Build dashboards for visualizing strategy performance

---

## 🔧 Phase 1: Data Engineering Infrastructure

### 1. Ingest Data
- Source API: Alpaca Markets API (100% of data ingestion)
- Data: Tickers, OHLCV, bid/ask spread, fundamentals (optional), volume
- Storage format: CSV/JSON → Parquet
- Tools: Python, requests, pandas

### 2. Store Raw Data (Bronze Layer)
- Save to AWS S3 or local object storage in Parquet format
- Organize by date, ticker

### 3. Clean + Normalize (Silver Layer)
- Load cleaned data into Snowflake (cloud data warehouse)
- Standardize column names
- Fix missing values
- Convert timezones, enforce types

### 4. Transform + Feature Engineering (Gold Layer)
- Use dbt with Snowflake to calculate SMA, EMA, RSI, MACD, Bollinger Bands, bid-ask spread, and spread-based features
- Add rolling volume stats, volatility metrics
- Tag timestamps for market open/close

### 5. Automate Workflows
- Airflow DAG to orchestrate daily ingest → clean → transform
- dbt for modeling and incremental table builds in Snowflake

---

## 🧠 Phase 2: Algorithm Testing & Analysis

### 1. Define Strategy Logic
- Example: SMA crossover (10-day vs 30-day), spread-based entry filters
- Entry/Exit signals, stop-loss logic

### 2. Backtest Strategies
- Tooling: pandas, vectorbt, or backtrader
- Key metrics: Win %, Sharpe ratio, drawdown, CAGR

### 3. Optimize and Compare
- Run over different tickers/sectors/timeframes
- Optimize parameters (e.g., moving average windows, spread thresholds)

### 4. Visualization + Reporting
- Streamlit or Power BI dashboards
- Profit curve, signal heatmaps, top performing tickers

---

## 🔁 Ongoing Tasks
- Validate data quality and handle API limits (specific to Alpaca's rate limits)
- Modularize ingestion + transformation scripts
- Schedule tests and visual reports
- Use Snowflake as the central warehouse for clean and modeled data
- Calculate and utilize spread as a key feature in signal logic
- Write README and strategy docs for future users/employers
