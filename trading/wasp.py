"""
WASP (Wave-Aware SPY) — v1
Long-only Fib 50–61.8 retrace + EMA/RSI filters + ATR volatility gate.
Executes via Alpaca (paper) using bracket orders:
  - Half position to +1R
  - Half position to 1.618 extension
  - Common stop at swing low - ATR_MULT_STOP * ATR
Flatten all at 15:55 ET.

Prereqs:
  pip install "alpaca-py>=0.18" pandas numpy pytz python-dotenv
.env:
  APCA_API_KEY_ID=...
  APCA_API_SECRET_KEY=...
  APCA_PAPER_BASE_URL=https://paper-api.alpaca.markets
  APCA_DATA_FEED=iex
"""

import os, asyncio, math, pytz
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.stream import TradingStream

# ====================== CONFIG ======================
SYMBOL = "SPY"
TIMEZONE = pytz.timezone("America/New_York")
BAR_SECONDS = 60

# Strategy (from research best pocket)
RSI_LONG = 48
ATR_MULT_STOP = 0.30
ZZ_ATR_MULT = 1.7
ZONE_PAD_FRAC = 0.012

# Volatility gate
VOL_FILTER_STYLE = "pct"  # "pct" (top-40%) or "median"
ATR_PCT_Q = 0.60  # top 40% ATR bars in the day
ATR_MED_MULT = 1.20  # (if using "median")

# Risk / risk controls
RISK_USD = 100  # $ risk per trade (1R)
MAX_SHARES = 1000  # hard cap
CLOSE_FLATTEN_TIME = (15, 55)  # 3:55pm ET
# ====================================================

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
FEED = os.getenv("APCA_DATA_FEED", "iex")


# ----------------- indicators -----------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ----------------- fib leg via ZigZag -----------------
def zigzag_last_leg(df: pd.DataFrame, zz_atr_mult: float = 1.7):
    """ATR-threshold ZigZag to define most recent swing leg."""
    if df.empty or len(df) < 5:
        return None
    if "atr" not in df:
        df["atr"] = atr(df, 14)

    a = df["atr"].fillna(df["atr"].median())
    c = df["close"].values
    n = len(df)

    swings = [0]
    ref = c[0]
    dir_up = None
    thr = a.iloc[0] * zz_atr_mult

    for i in range(1, n):
        move = c[i] - ref
        if dir_up is None:
            if abs(move) >= thr:
                dir_up = move > 0
                swings.append(i)
                ref = c[i]
                thr = a.iloc[i] * zz_atr_mult
            continue
        if dir_up and c[i] >= ref:
            ref = c[i]
            thr = a.iloc[i] * zz_atr_mult
        elif (not dir_up) and c[i] <= ref:
            ref = c[i]
            thr = a.iloc[i] * zz_atr_mult
        else:
            if abs(c[i] - ref) >= thr:
                dir_up = not dir_up
                swings.append(i)
                ref = c[i]
                thr = a.iloc[i] * zz_atr_mult

    if len(swings) < 2:
        return None

    a_idx, b_idx = swings[-2], swings[-1]
    leg_low = float(df.loc[a_idx:b_idx, "low"].min())
    leg_high = float(df.loc[a_idx:b_idx, "high"].max())
    direction = "up" if df.loc[b_idx, "close"] >= df.loc[a_idx, "close"] else "down"

    rng = leg_high - leg_low
    if rng <= 0:
        return None

    levels = {"50%": leg_low + 0.500 * rng, "61.8%": leg_low + 0.618 * rng}
    return {"low": leg_low, "high": leg_high, "direction": direction, "levels": levels}


def now_et():
    return datetime.now(TIMEZONE)


# ----------------- WASP bot -----------------
class WaspBot:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        self.position_qty = 0
        self.entry_price = None
        self.stop_price = None
        self.tp1_price = None
        self.t161_price = None
        self.open_order_ids = set()

        self.client = TradingClient(API_KEY, API_SECRET, paper=True)
        acct = self.client.get_account()
        print(f"WASP initialized. Buying power: {acct.buying_power}")

    # ----- live bar handler -----
    async def on_bar(self, bar):
        ts = bar.t.astimezone(TIMEZONE).isoformat()
        row = {
            "ts": ts,
            "open": float(bar.o),
            "high": float(bar.h),
            "low": float(bar.l),
            "close": float(bar.c),
            "volume": float(bar.v),
        }
        self.df.loc[len(self.df)] = row

        # indicators
        self.df["ema9"] = ema(self.df["close"], 9)
        self.df["ema21"] = ema(self.df["close"], 21)
        self.df["rsi"] = rsi(self.df["close"], 14)
        self.df["atr"] = atr(self.df, 14)

        # daily risk-off
        if (
            now_et().hour == CLOSE_FLATTEN_TIME[0]
            and now_et().minute >= CLOSE_FLATTEN_TIME[1]
        ):
            await self.flatten_all("EOD")
            return

        # one position at a time
        if self.position_qty != 0:
            return

        fib = zigzag_last_leg(self.df, ZZ_ATR_MULT)
        if not fib or fib["direction"] != "up":
            return

        lo, hi = fib["low"], fib["high"]
        rng = hi - lo
        L50, L618 = fib["levels"]["50%"], fib["levels"]["61.8%"]
        zlo, zhi = min(L50, L618), max(L50, L618)
        pad = ZONE_PAD_FRAC * rng
        zlo -= pad
        zhi += pad

        last = self.df.iloc[-1]
        in_zone = (last["low"] <= zhi) and (last["high"] >= zlo)
        trend_up = last["ema9"] > last["ema21"]
        mom_ok = last["rsi"] >= RSI_LONG

        # volatility gate
        if VOL_FILTER_STYLE == "pct":
            thr = self.df["atr"].quantile(ATR_PCT_Q)
            vol_ok = last["atr"] >= thr
        else:
            med = self.df["atr"].median()
            vol_ok = last["atr"] >= ATR_MED_MULT * med

        if in_zone and trend_up and mom_ok and vol_ok:
            await self.enter_long(lo, hi, last)

    async def enter_long(self, swing_lo, swing_hi, last_row):
        atr_val = float(last_row["atr"])
        entry = float(last_row["close"])  # market now
        stop = float(swing_lo) - ATR_MULT_STOP * atr_val
        R = entry - stop
        if R <= 0:
            return

        tp1 = entry + R
        t161 = swing_hi + 0.618 * (swing_hi - swing_lo)

        qty = max(1, min(int(RISK_USD / R), MAX_SHARES))
        half1 = qty // 2
        half2 = qty - half1
        if qty <= 0:
            print("[WASP] Size rounded to 0; skip.")
            return

        print(
            f"[WASP] ENTER LONG {self.symbol} @~{entry:.2f} qty={qty}  "
            f"(R≈{R:.2f}, tp1={tp1:.2f}, t161={t161:.2f}, stop={stop:.2f})"
        )

        # Two bracket orders (½ to +1R, ½ to 1.618). Stable for paper trading.
        o1 = self.client.submit_order(
            MarketOrderRequest(
                symbol=self.symbol,
                qty=half1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(tp1, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop, 2)),
            )
        )
        o2 = self.client.submit_order(
            MarketOrderRequest(
                symbol=self.symbol,
                qty=half2,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(t161, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop, 2)),
            )
        )
        self.open_order_ids.update([o1.id, o2.id])
        self.position_qty = qty
        self.entry_price = entry
        self.stop_price = stop
        self.tp1_price = tp1
        self.t161_price = t161

    async def flatten_all(self, reason="MANUAL"):
        """Market flat + cancel open orders."""
        try:
            pos = self.client.get_open_position(self.symbol)
            if pos and float(pos.qty) != 0:
                side = OrderSide.SELL if float(pos.qty) > 0 else OrderSide.BUY
                self.client.submit_order(
                    MarketOrderRequest(
                        symbol=self.symbol,
                        qty=abs(int(float(pos.qty))),
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                print(f"[WASP] Flattened position ({reason}).")
        except Exception:
            pass
        # cancel any open orders
        try:
            for oid in list(self.open_order_ids):
                try:
                    self.client.cancel_order_by_id(oid)
                except Exception:
                    pass
            self.open_order_ids.clear()
        except Exception:
            pass
        self.position_qty = 0
        self.entry_price = None
        self.stop_price = None
        self.tp1_price = None
        self.t161_price = None


# ----------------- run streams -----------------
async def main():
    bot = WaspBot(SYMBOL)

    # Market data stream (+ trade updates stream placeholder)
    md_stream = StockDataStream(API_KEY, API_SECRET, feed=FEED)
    md_stream.subscribe_bars(lambda bar: asyncio.create_task(bot.on_bar(bar)), SYMBOL)

    # Optional: watch trade updates (fills/cancels). Useful for later: move runner stop to BE after TP1 fill.
    async def on_trade_update(data):
        # Example: print event types
        # et = data.event
        # print("[WASP] trade_update:", et, data)
        pass

    trade_stream = TradingStream(API_KEY, API_SECRET, paper=True)
    trade_stream.subscribe_trade_updates(on_trade_update)

    await asyncio.gather(md_stream.run(), trade_stream.run())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[WASP] Shutting down…")
