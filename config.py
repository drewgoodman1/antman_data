from dotenv import load_dotenv
import os

# Load .env if present
load_dotenv()

# ---- Alpaca / trading ----
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
APCA_PAPER_BASE_URL = os.getenv("APCA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
APCA_DATA_FEED = os.getenv("APCA_DATA_FEED", "iex")

# ---- MinIO / S3 ----
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", os.getenv("S3_ACCESS_KEY_ID", "minioadmin"))
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin"))
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "antman-lake")
# DuckDB httpfs prefers host:port without scheme
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT", "127.0.0.1:9100"))
S3_USE_SSL = os.getenv("S3_USE_SSL", "false").lower() in ("1", "true", "yes")
S3_URL_STYLE = os.getenv("S3_URL_STYLE", "path")

# ---- Logging / paths ----
WASP_LOG_PATH = os.getenv("WASP_LOG_PATH", "wasp_signals.csv")

# Small helper
def env_summary():
    return {
        "APCA_API_KEY_ID": bool(APCA_API_KEY_ID),
        "MINIO_ROOT_USER": bool(MINIO_ROOT_USER),
        "S3_ENDPOINT_URL": S3_ENDPOINT_URL,
    }


__all__ = [
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_PAPER_BASE_URL",
    "APCA_DATA_FEED",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
    "MINIO_BUCKET",
    "S3_ENDPOINT_URL",
    "S3_USE_SSL",
    "WASP_LOG_PATH",
    "env_summary",
]
