"""
Flatten any open SPY position and cancel open orders using Alpaca (paper/live via env).
Loads .env via python-dotenv.
"""

import os
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def main():
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        print("[flatten] Missing Alpaca API credentials in .env")
        return

    client = TradingClient(api_key, api_secret, paper=True)

    # Cancel any open orders first (best-effort)
    try:
        client.cancel_orders()
        print("[flatten] Canceled open orders (if any).")
    except Exception as e:
        print(f"[flatten] cancel_orders failed: {e}")

    # Flatten open position in SPY
    try:
        pos = client.get_open_position("SPY")
    except Exception:
        pos = None

    try:
        if pos is not None:
            qty = int(abs(float(pos.qty)))
            if qty > 0:
                side = OrderSide.SELL if float(pos.qty) > 0 else OrderSide.BUY
                client.submit_order(
                    MarketOrderRequest(
                        symbol="SPY",
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                print(f"[flatten] Submitted market order to flatten {qty} shares SPY.")
            else:
                print("[flatten] No SPY position to flatten.")
        else:
            print("[flatten] No SPY position open.")
    except Exception as e:
        print(f"[flatten] Flatten failed: {e}")


if __name__ == "__main__":
    main()
