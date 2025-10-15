#!/bin/bash

# Batch script to upload SPY data for all trading days in 2025 to MinIO

# Set the ticker and year
ticker="SPY"
year="2025"

# Define the local data directory and script path
project_root="$(dirname $(dirname $(realpath $0)))"
data_dir="$project_root/data/bronze/$ticker"
upload_script="$project_root/scripts/upload_spy_day_to_minio.py"

# Check if the upload script exists
if [ ! -f "$upload_script" ]; then
  echo "Upload script not found: $upload_script"
  exit 1
fi

# Generate trading days for 2025
start_date="$year-01-01"
end_date="$year-12-31"
current_date="$start_date"
while [ "$current_date" != "$end_date" ]; do
  # Check if the current date is a weekday (Monday to Friday)
  day_of_week=$(date -d "$current_date" +%u)
  if [ "$day_of_week" -lt 6 ]; then
    # Check if the local file exists
    local_file="$data_dir/spy_1min_$current_date.parquet"
    if [ -f "$local_file" ]; then
      echo "Uploading data for $current_date..."
      DAY=$current_date TICKER=$ticker python "$upload_script"
    else
      echo "Data file not found for $current_date: $local_file"
    fi
  fi
  # Increment the date
  current_date=$(date -I -d "$current_date + 1 day")
done

echo "Batch upload complete."