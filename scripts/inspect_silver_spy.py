import os
import pandas as pd

# Path to silver SPY data
silver_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "silver", "SPY"
)

# List all parquet files
files = sorted([f for f in os.listdir(silver_dir) if f.endswith(".parquet")])

if not files:
    print("No parquet files found in", silver_dir)
    exit(1)

# Inspect the first file
sample_file = os.path.join(silver_dir, files[0])
df = pd.read_parquet(sample_file)

print(f"Sample file: {sample_file}")
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nInfo:")
df.info()

# Optionally, summarize all files (row counts, date range)
total_rows = 0
dates = []
for f in files:
    path = os.path.join(silver_dir, f)
    d = pd.read_parquet(path)
    total_rows += len(d)
    dates.append(f.split("_")[-1].replace(".parquet", ""))
print(f"\nTotal files: {len(files)}")
print(f"Total rows: {total_rows}")
print(f"Date range: {min(dates)} to {max(dates)}")
