# wasp_v2.py — Long-only Fib 50–61.8 + EMA/RSI + ATR percentile gate
# Live/paper trading on Alpaca with: +1R partial -> BE, 1.618 preferred then 1.272, EOD 15:45
# pip install -U "alpaca-py>=0.18" pandas numpy pytz python-dotenv

import os, asyncio, csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
import time
import signal
import sys
import fcntl
import logging
import threading
from pathlib import Path

# -------- Alpaca ----------
from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
)
from alpaca.trading.stream import TradingStream

# ====================== CONFIG (defaults = your best combo) ===================
SYMBOL = "SPY"
TZ = pytz.timezone("America/New_York")

RSI_LONG = 48
ATR_MULT_STOP = 0.25
ZZ_ATR_MULT = 1.5
ZONE_PAD_FRAC = 0.02
ATR_PCT_Q = 0.60  # top 40% ATR bars (per today’s sessions only)
PREFER_EXT = "161"  # "161" then "127"

SESSIONS = [("09:35", "15:55")]
WARMUP_BARS = 63  # stabilize EMA/ATR/RSI
EOD_FLATTEN = (15, 45)

RISK_USD = 100  # 1R in dollars
MAX_SHARES = 1000
COOLDOWN_SEC = 60  # after STOP

BACKFILL_MIN_BARS = 300
VERBOSE_EVERY_BAR = True
LOG_TO_CSV = True
LOG_PATH = "wasp_signals.csv"
# ============================================================================

# -------- env / API --------
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
FEED_STR = (os.getenv("APCA_DATA_FEED") or "iex").lower()
FEED = DataFeed.SIP if "sip" in FEED_STR else DataFeed.IEX


# -------- indicators --------
def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def rsi(prices: pd.Series, length=14) -> pd.Series:
    p = prices.astype(float)
    d = p.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, length=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# -------- zigzag fib --------
def zigzag_last_leg(df: pd.DataFrame, zz_atr_mult=1.7):
    if df.empty or len(df) < 5:
        return None
    if "atr" not in df:
        df["atr"] = atr(df, 14)
    a = df["atr"].fillna(df["atr"].median())
    c = df["close"].values
    swings = [0]
    dir_up = None
    ref = c[0]
    thr = a.iloc[0] * zz_atr_mult
    for i in range(1, len(c)):
        move = c[i] - ref
        if dir_up is None:
            if abs(move) >= thr:
                dir_up = move > 0
                swings.append(i)
                ref = c[i]
                thr = a.iloc[i] * zz_atr_mult
            continue
        if (dir_up and c[i] >= ref) or ((not dir_up) and c[i] <= ref):
            ref = c[i]
            thr = a.iloc[i] * zz_atr_mult
        elif abs(c[i] - ref) >= thr:
            dir_up = not dir_up
            swings.append(i)
            ref = c[i]
            thr = a.iloc[i] * zz_atr_mult
    if len(swings) < 2:
        return None
    a_idx, b_idx = swings[-2], swings[-1]
    # swings are positional offsets (0..len(df)-1). Use iloc so we don't KeyError
    # when df.index is not a RangeIndex (e.g., after filtering).
    if not (isinstance(a_idx, int) and isinstance(b_idx, int)):
        return None
    if a_idx < 0 or b_idx < 0 or a_idx >= len(df) or b_idx >= len(df) or a_idx > b_idx:
        return None
    seg = df.iloc[a_idx : b_idx + 1]
    leg_low = float(seg["low"].min())
    leg_high = float(seg["high"].max())
    # use iloc to compare closes at the swing endpoints
    direction = (
        "up"
        if float(df.iloc[b_idx]["close"]) >= float(df.iloc[a_idx]["close"])
        else "down"
    )
    rng = leg_high - leg_low
    if rng <= 0:
        return None
    levels = {"50%": leg_low + 0.500 * rng, "61.8%": leg_low + 0.618 * rng}
    return {
        "low": leg_low,
        "high": leg_high,
        "direction": direction,
        "levels": levels,
        "range": rng,
    }


# -------- helpers --------
def to_et(ts) -> datetime:
    if hasattr(ts, "astimezone"):
        return ts.astimezone(TZ)
    return datetime.now(TZ)


def time_in_sessions(ts: datetime) -> bool:
    t = ts.time()
    for s, e in SESSIONS:
        s_t = datetime.strptime(s, "%H:%M").time()
        e_t = datetime.strptime(e, "%H:%M").time()
        if s_t <= t <= e_t:
            return True
    return False


def append_log(row: Dict[str, Any]):
    if not LOG_TO_CSV:
        return
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


# -------- bot --------
@dataclass
class PendingEntry:
    swing_lo: float
    swing_hi: float
    signaled_at: datetime


class WaspV2:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        self._last_minute_key: Optional[datetime] = None
        self.pending: Optional[PendingEntry] = None
        self.cooldown_until: Optional[datetime] = None

        self.position_qty = 0
        self.entry = None
        self.stop = None
        self.tp1 = None
        self.t161 = None
        self.t127 = None
        self.stop_order_id = None
        self.tp1_order_id = None
        self.runner_order_id = None

        self.client = TradingClient(API_KEY, API_SECRET, paper=True)
        acct = self.client.get_account()
        print(f"[WASP] Using {FEED.name} | Buying power: {acct.buying_power}")

    async def on_bar(self, bar):
        ts = (
            getattr(bar, "timestamp", None)
            or getattr(bar, "time", None)
            or getattr(bar, "t", None)
        )
        o = getattr(bar, "open", None) or getattr(bar, "o", None)
        h = getattr(bar, "high", None) or getattr(bar, "h", None)
        l = getattr(bar, "low", None) or getattr(bar, "l", None)
        c = getattr(bar, "close", None) or getattr(bar, "c", None)
        v = getattr(bar, "volume", None) or getattr(bar, "v", None)

        ts_dt = to_et(ts)
        minute_key = ts_dt.replace(second=0, microsecond=0)

        if ts_dt.second != 0:
            return  # only process final bar for each minute

        # --- update dataframe (KEEP ts as tz-aware Timestamp) ---
        row = {
            "ts": ts_dt,  # ✅ Timestamp, not string
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(v),
        }
        new_minute = (self._last_minute_key is None) or (
            minute_key != self._last_minute_key
        )
        if new_minute:
            self.df.loc[len(self.df)] = row
            self.df["ema9"] = ema(self.df["close"], 9)
            self.df["ema21"] = ema(self.df["close"], 21)
            self.df["rsi"] = rsi(self.df["close"], 14)
            self.df["atr"] = atr(self.df, 14)
            self._last_minute_key = minute_key
        else:
            self.df.iloc[-1] = row
            self.df["ema9"] = ema(self.df["close"], 9)
            self.df["ema21"] = ema(self.df["close"], 21)
            self.df["rsi"] = rsi(self.df["close"], 14)
            self.df["atr"] = atr(self.df, 14)

        # Optional one-time repair if ts somehow became strings:
        if self.df["ts"].dtype == "object":
            self.df["ts"] = pd.to_datetime(
                self.df["ts"], utc=True, format="ISO8601", errors="coerce"
            ).dt.tz_convert(TZ)

        # EOD flatten
        if ts_dt.hour == EOD_FLATTEN[0] and ts_dt.minute >= EOD_FLATTEN[1]:
            await self.flatten_all("EOD")
            return

        # Entry on next bar open if we had a pending signal
        if (
            new_minute
            and self.pending
            and self.position_qty == 0
            and time_in_sessions(ts_dt)
        ):
            await self._enter_next_open(self.pending, ts_dt)
            self.pending = None

        # In position: manage
        if self.position_qty > 0:
            await self._manage(ts_dt)
            return

        # Cooldown after STOP
        if self.cooldown_until and ts_dt < self.cooldown_until:
            return

        # No position: look for setups (only in sessions & after warmup)
        if len(self.df) < WARMUP_BARS or not time_in_sessions(ts_dt):
            # still print diagnostics each bar even if not scanning for entries
            last = self.df.iloc[-1]
            print(
                f"[BAR] {ts_dt.strftime('%H:%M')} "
                f"close={last['close']:.2f} ema9={last['ema9']:.2f} ema21={last['ema21']:.2f} "
                f"rsi={last['rsi']:.1f} atr={last['atr']:.2f}"
            )
            append_log(
                {
                    "ts": last["ts"],
                    "close": round(float(last["close"]), 2),
                    "ema9": round(float(last["ema9"]), 2),
                    "ema21": round(float(last["ema21"]), 2),
                    "rsi": round(float(last["rsi"]), 2),
                    "atr": round(float(last["atr"]), 2),
                    "trend_up": last["ema9"] > last["ema21"],
                    "touched_fib_zone": False,
                    "mom_ok": last["rsi"] >= RSI_LONG,
                    "vol_ok": False,
                    "signal": False,
                }
            )
            return

        # build “today in session” slice for ATR percentile & zigzag (no re-parsing)
        df = self.df.copy()
        today = ts_dt.date()
        in_sess = df[df["ts"].dt.date.eq(today) & df["ts"].apply(time_in_sessions)]
        if in_sess.empty:
            return

        # volatility gate from today’s session only
        thr = in_sess["atr"].quantile(ATR_PCT_Q)
        last = self.df.iloc[-1]
        vol_ok = float(last["atr"]) >= float(thr)

        # zigzag on post-OR bars (start at 09:45)
        or_time = datetime(ts_dt.year, ts_dt.month, ts_dt.day, 9, 45, tzinfo=TZ).time()
        post_or = in_sess[in_sess["ts"].dt.time >= or_time]
        fib_src = post_or if len(post_or) >= 5 else in_sess.tail(50)
        fib = zigzag_last_leg(fib_src, ZZ_ATR_MULT)

        trend_up = last["ema9"] > last["ema21"]
        mom_ok = last["rsi"] >= RSI_LONG

        touched = False
        if fib and fib["direction"] == "up":
            lo, hi, rng = fib["low"], fib["high"], fib["range"]
            L50, L618 = fib["levels"]["50%"], fib["levels"]["61.8%"]
            zlow = min(L50, L618) - ZONE_PAD_FRAC * rng
            zhigh = max(L50, L618) + ZONE_PAD_FRAC * rng
            touched = (last["low"] <= zhigh) and (last["high"] >= zlow)

        signal = touched and trend_up and mom_ok and vol_ok

        # --- print to console ---
        print(
            f"[BAR] {ts_dt.strftime('%H:%M')} "
            f"close={last['close']:.2f} ema9={last['ema9']:.2f} ema21={last['ema21']:.2f} "
            f"rsi={last['rsi']:.1f} atr={last['atr']:.2f} "
            f"trend_up={trend_up} touched={touched} mom_ok={mom_ok} vol_ok={vol_ok} "
            f"{'SIGNAL✅' if signal else ''}"
        )

        # --- log to CSV ---
        append_log(
            {
                "ts": last["ts"],  # Timestamp; CSV will serialize it
                "close": round(float(last["close"]), 2),
                "ema9": round(float(last["ema9"]), 2),
                "ema21": round(float(last["ema21"]), 2),
                "rsi": round(float(last["rsi"]), 2),
                "atr": round(float(last["atr"]), 2),
                "trend_up": trend_up,
                "touched_fib_zone": touched,
                "mom_ok": mom_ok,
                "vol_ok": vol_ok,
                "signal": signal,
            }
        )

        # Arm pending; will enter *next* bar open
        if signal:
            self.pending = PendingEntry(swing_lo=lo, swing_hi=hi, signaled_at=ts_dt)

    async def _enter_next_open(self, p: PendingEntry, ts_now: datetime):
        last = self.df.iloc[-1]
        entry = float(last["open"])  # next bar open
        swing_lo, swing_hi = p.swing_lo, p.swing_hi
        atr_val = float(last["atr"])
        stop = swing_lo - ATR_MULT_STOP * atr_val
        R = entry - stop
        if R <= 0:
            return

        t127 = swing_hi + 0.272 * (swing_hi - swing_lo)
        t161 = swing_hi + 0.618 * (swing_hi - swing_lo)
        tp1 = entry + R

        qty = max(1, min(int(RISK_USD / max(R, 0.01)), MAX_SHARES))
        if qty == 0:
            return
        half1 = qty // 2
        half2 = qty - half1

        print(
            f"[WASP] ENTER {self.symbol} {qty} @ {entry:.2f} (R={R:.2f} tp1={tp1:.2f} 1.618={t161:.2f} 1.272={t127:.2f} stop={stop:.2f})"
        )

        # Parent: market buy
        self.client.submit_order(
            MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )

        # Child orders we manage: two limits and one stop
        tp1_req = LimitOrderRequest(
            symbol=self.symbol,
            qty=half1,
            side=OrderSide.SELL,
            limit_price=round(tp1, 2),
            time_in_force=TimeInForce.DAY,
        )
        runner_price = t161 if PREFER_EXT == "161" else t127
        runner_req = LimitOrderRequest(
            symbol=self.symbol,
            qty=half2,
            side=OrderSide.SELL,
            limit_price=round(runner_price, 2),
            time_in_force=TimeInForce.DAY,
        )
        stop_req = StopOrderRequest(
            symbol=self.symbol,
            qty=qty,
            side=OrderSide.SELL,
            stop_price=round(stop, 2),
            time_in_force=TimeInForce.DAY,
        )

        tp1_ord = self.client.submit_order(tp1_req)
        runner_ord = self.client.submit_order(runner_req)
        stop_ord = self.client.submit_order(stop_req)

        # cache state
        self.position_qty = qty
        self.entry, self.stop, self.tp1 = entry, stop, tp1
        self.t161, self.t127 = t161, t127
        self.tp1_order_id = tp1_ord.id
        self.runner_order_id = runner_ord.id
        self.stop_order_id = stop_ord.id

    async def _manage(self, ts_now: datetime):
        # move stop to BE if TP1 filled; fallback to 1.272 if runner at 1.618 not hit near EOD (optional)
        try:
            if self.tp1_order_id:
                ord1 = self.client.get_order_by_id(self.tp1_order_id)
                if ord1.filled_qty and float(ord1.filled_qty) >= float(ord1.qty):
                    # TP1 done -> move remaining stop to BE once
                    if self.stop_order_id:
                        try:
                            self.client.cancel_order_by_id(self.stop_order_id)
                        except:
                            pass
                        rem = max(0, self.position_qty - int(float(ord1.qty)))
                        if rem > 0:
                            be_stop_req = StopOrderRequest(
                                symbol=self.symbol,
                                qty=rem,
                                side=OrderSide.SELL,
                                stop_price=round(self.entry, 2),
                                time_in_force=TimeInForce.DAY,
                            )
                            be_ord = self.client.submit_order(be_stop_req)
                            self.stop_order_id = be_ord.id
                    # clear so we don't repeat
                    self.tp1_order_id = None
        except Exception as e:
            print("[WASP] manage() BE move error:", e)

    async def flatten_all(self, reason="EOD"):
        try:
            # cancel children first
            for oid in [self.tp1_order_id, self.runner_order_id, self.stop_order_id]:
                if oid:
                    try:
                        self.client.cancel_order_by_id(oid)
                    except:
                        pass
        except:
            pass
        self.tp1_order_id = self.runner_order_id = self.stop_order_id = None

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
        except Exception:
            pass

        self.position_qty = 0
        self.entry = self.stop = self.tp1 = self.t161 = self.t127 = None
        self.cooldown_until = datetime.now(TZ) + timedelta(seconds=COOLDOWN_SEC)
        print(
            f"[WASP] FLAT ({reason}). Cooldown to {self.cooldown_until.strftime('%H:%M:%S')}"
        )


# -------- main --------
async def main():
    # single-instance pidfile to avoid accidental multiple connectors
    pidfile = Path(f"/tmp/wasp_v2_{SYMBOL}.pid")

    def acquire_pidfile(p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = p.open("w+")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(
                f"[WASP] another instance appears to be running (pidfile {p}). Exiting."
            )
            sys.exit(1)
        fd.truncate(0)
        fd.write(str(os.getpid()))
        fd.flush()
        return fd

    # logging to file + stdout
    logdir = Path("logs")
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logdir / "wasp.log"),
        ],
    )
    logging.info("[WASP] starting up")

    pid_fd = acquire_pidfile(pidfile)

    # cooperative shutdown event used by runner loops
    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        logging.info(f"[WASP] signal {signum} received, shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    bot = WaspV2(SYMBOL)

    # backfill to seed indicators
    try:
        hist = StockHistoricalDataClient(API_KEY, API_SECRET)
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame.Minute,
            limit=BACKFILL_MIN_BARS,
            feed=FEED,
        )
        bars = hist.get_stock_bars(req)
        if hasattr(bars, "df") and not bars.df.empty:
            df = (
                bars.df.xs(SYMBOL, level=0)
                if isinstance(bars.df.index, pd.MultiIndex)
                else bars.df
            ).reset_index()
            df = df.rename(columns={"timestamp": "ts"})
            # ✅ keep as tz-aware Timestamp; don't stringify
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TZ)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            bot.df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
            bot.df["ema9"] = ema(bot.df["close"], 9)
            bot.df["ema21"] = ema(bot.df["close"], 21)
            bot.df["rsi"] = rsi(bot.df["close"], 14)
            bot.df["atr"] = atr(bot.df, 14)
            print(f"[WASP] Backfilled {len(bot.df)} bars.")
    except Exception as e:
        print(f"[WASP] Backfill failed: {e}")

    # Run market-data and trade-update streams with automatic retries/backoff
    # and cooperative shutdown. Both runners respect stop_event.
    def md_runner():
        attempt = 0
        while not stop_event.is_set():
            try:
                md = StockDataStream(API_KEY, API_SECRET, feed=FEED)
                md.subscribe_bars(bot.on_bar, SYMBOL)
                logging.info("[WASP] starting market data stream")
                # blocking call; will return on errors or stop
                md.run()
                logging.info("[WASP] market data stream stopped")
                # if we exit without error, break
                break
            except ValueError as e:
                attempt += 1
                backoff = min(60, 2**attempt)
                logging.warning(
                    f"MD auth/conn error: {e}; retrying in {backoff}s (attempt {attempt})"
                )
                stop_event.wait(backoff)
            except Exception as e:
                attempt += 1
                backoff = min(60, 2**attempt)
                logging.exception(
                    f"MD stream error: {e}; retrying in {backoff}s (attempt {attempt})"
                )
                stop_event.wait(backoff)

    def trade_runner():
        attempt = 0
        while not stop_event.is_set():
            try:
                trade_stream = TradingStream(API_KEY, API_SECRET, paper=True)

                async def on_trade_update(data):
                    # placeholder: you may enrich this to handle fills
                    return

                trade_stream.subscribe_trade_updates(on_trade_update)
                logging.info("[WASP] starting trade update stream")
                trade_stream.run()
                logging.info("[WASP] trade update stream stopped")
                break
            except Exception as e:
                attempt += 1
                backoff = min(60, 2**attempt)
                logging.exception(
                    f"Trade stream error: {e}; retrying in {backoff}s (attempt {attempt})"
                )
                stop_event.wait(backoff)

    # run both runners concurrently; they will self-retry until stop_event
    runners = [asyncio.to_thread(md_runner), asyncio.to_thread(trade_runner)]
    try:
        await asyncio.gather(*runners)
    finally:
        logging.info("[WASP] shutting down runners")
        # release pidfile and exit
        try:
            pid_fd.close()
            pidfile.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[WASP] Shutting down…")
