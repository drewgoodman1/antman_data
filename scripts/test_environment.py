#!/usr/bin/env python3
"""
Test script to    try:
        conn = psycopg        # Test t        # Test data client
        data_client = StockHistoricalDataClient(
            api_key=os.getenv('APCA_API_KEY_ID'),
            secret_key=os.getenv('APCA_API_SECRET_KEY')
        )g client
        trading_client = TradingClient(
            api_key=os.getenv('APCA_API_KEY_ID'),
            secret_key=os.getenv('APCA_API_SECRET_KEY'),
            paper=True  # Using paper trading
        )ect(
            host=os.getenv('PGHOST'),
            database=os.getenv('PGDATABASE'),
            user=os.getenv('PGUSER'),
            password=os.getenv('PGPASSWORD'),
            port=os.getenv('PGPORT', '5432')
        )environment setup and connections
"""
import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient


def test_environment():
    """Test environment variable loading"""
    print("Testing environment variables...")
    load_dotenv()

    required_vars = [
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "PGHOST",
        "PGDATABASE",
        "PGUSER",
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("✅ All required environment variables found")
        return True


def test_postgres_connection():
    """Test PostgreSQL connection"""
    print("\nTesting PostgreSQL connection...")
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=os.getenv("POSTGRES_PORT", "5432"),
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ PostgreSQL connection successful: {version}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False


def test_alpaca_connection():
    """Test Alpaca API connection"""
    print("\nTesting Alpaca API connection...")
    try:
        # Test trading client
        trading_client = TradingClient(
            api_key=os.getenv("ALPACA_API_KEY_ID"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            paper=True,  # Using paper trading
        )
        account = trading_client.get_account()
        print(f"✅ Alpaca Trading API connection successful")
        print(f"   Account Status: {account.status}")
        print(f"   Buying Power: ${account.buying_power}")

        # Test data client
        data_client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY_ID"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
        )
        print("✅ Alpaca Data API connection successful")

        return True
    except Exception as e:
        print(f"❌ Alpaca API connection failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting antman_data environment tests...\n")

    # Load environment variables
    load_dotenv()

    tests = [test_environment, test_postgres_connection, test_alpaca_connection]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    if all(results):
        print(
            "🎉 All tests passed! Environment is ready for data pipeline development."
        )
        print("\n📝 Next Steps:")
        print("   1. Create first data ingestion script")
        print("   2. Set up database schemas")
        print("   3. Build ETL pipeline")
    else:
        print("⚠️ Some tests failed. Please fix the issues before proceeding.")

    return all(results)


if __name__ == "__main__":
    main()
