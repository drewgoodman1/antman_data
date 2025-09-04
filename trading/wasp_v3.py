# wasp_v2.py — Long-only Fib 50–61.8 + EMA/RSI + ATR percentile gate
# Live/paper trading on Alpaca with: +1R partial -> BE (future), 1.618 preferred then 1.272, EOD 15:45
# pip install -U "alpaca-py>=0.18" pandas numpy pytz python-dotenv

import os, asyncio, csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
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
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.stream import TradingStream
from alpaca.common.exceptions import APIError

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
COOLDOWN_SEC = 60  # after STOP or submit error

# --- hybrid sizing (option 4) ---
RISK_PCT = 0.01  # fraction of account equity to risk per trade (1% default)
MIN_RISK_USD = 100  # don't risk less than this per trade
MAX_EXPOSURE_USD = 10_000  # cap total dollars invested per symbol
INVEST_USD_FALLBACK = 10_000  # if risk-based qty < 1, invest this amount as fallback
DRY_RUN = False  # if True, don't submit orders; useful for testing

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
    if not (isinstance(a_idx, int) and isinstance(b_idx, int)):
        return None
    if a_idx < 0 or b_idx < 0 or a_idx >= len(df) or b_idx >= len(df) or a_idx > b_idx:
        return None
    seg = df.iloc[a_idx : b_idx + 1]
    leg_low = float(seg["low"].min())
    leg_high = float(seg["high"].max())
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


def append_trade_log(row: Dict[str, Any], path: str = "wasp_trades.csv"):
    """Append a trade/update row to a separate CSV for trades/events."""
    if not LOG_TO_CSV:
        return
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
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

        # IDs of the two parent bracket orders (half/half)
        self.bracket_tp1_id = None
        self.bracket_runner_id = None

        self.client = TradingClient(API_KEY, API_SECRET, paper=True)
        acct = self.client.get_account()
        print(f"[WASP] Using {FEED.name} | Buying power: {acct.buying_power}")

    # ---------- NEW: broker sync + guards ----------
    def sync_broker_state(self) -> None:
        """Keep local state aligned with broker (called each minute and on trade updates)."""
        try:
            pos = self.client.get_open_position(self.symbol)
            qty = int(float(pos.qty))
            self.position_qty = qty
            try:
                self.entry = float(
                    getattr(pos, "avg_entry_price", self.entry) or self.entry
                )
            except Exception:
                pass
            if qty == 0:
                self.bracket_tp1_id = None
                self.bracket_runner_id = None
        except Exception:
            # no open position
            self.position_qty = 0
            self.entry = None
            self.bracket_tp1_id = None
            self.bracket_runner_id = None

    def has_open_symbol_orders(self) -> bool:
        """True if there are open orders for this symbol (pre-flight guard)."""
        try:
            try:
                orders = self.client.get_open_orders()
            except Exception:
                orders = self.client.get_orders()
            return any(getattr(o, "symbol", None) == self.symbol for o in orders)
        except Exception:
            return False

    # -----------------------------------------------
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
            "ts": ts_dt,  # tz-aware Timestamp
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

        # One-time repair if ts ever became strings:
        if self.df["ts"].dtype == "object":
            self.df["ts"] = pd.to_datetime(
                self.df["ts"], utc=True, format="ISO8601", errors="coerce"
            ).dt.tz_convert(TZ)

        # ---------- NEW: sync local state with broker once per minute ----------
        self.sync_broker_state()

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

        # In position: (future) manage BE move using trade updates/child legs
        if self.position_qty > 0:
            await self._manage(ts_dt)
            return

        # Cooldown after STOP / submit error
        if self.cooldown_until and ts_dt < self.cooldown_until:
            return

        # No position: look for setups (only in sessions & after warmup)
        if len(self.df) < WARMUP_BARS or not time_in_sessions(ts_dt):
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

        # build “today in session” slice for ATR percentile & zigzag
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
                "ts": last["ts"],
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
            self.pending = None
            return

        # ---------- NEW: pre-flight guard ----------
        self.sync_broker_state()
        if self.position_qty > 0 or self.has_open_symbol_orders():
            print("[WASP] Broker shows existing position/open orders; skipping entry.")
            self.pending = None
            self.cooldown_until = ts_now + timedelta(seconds=COOLDOWN_SEC)
            return

        t127 = swing_hi + 0.272 * (swing_hi - swing_lo)
        t161 = swing_hi + 0.618 * (swing_hi - swing_lo)
        tp1 = entry + R
        runner_price = t161 if PREFER_EXT == "161" else t127

        # --- hybrid sizing (risk-percent + caps + fallback) ---
        acct = None
        try:
            acct = self.client.get_account()
            equity = float(getattr(acct, "equity", acct.buying_power or 0))
        except Exception:
            equity = None

        risk_usd = max(MIN_RISK_USD, (equity * RISK_PCT) if equity else MIN_RISK_USD)
        qty_risk = int(risk_usd // max(R, 0.01))
        qty_exposure = int(MAX_EXPOSURE_USD // max(entry, 0.01))
        qty = max(0, min(qty_risk, qty_exposure, MAX_SHARES))

        fallback_used = False
        if qty < 1:
            # fallback to investing a fixed dollar amount (exposure-based)
            qty = int(INVEST_USD_FALLBACK // max(entry, 0.01))
            fallback_used = True

        if qty < 1:
            print(
                f"[WASP] computed qty < 1 even after fallback (entry={entry:.2f}); skipping"
            )
            self.pending = None
            return

        half1 = qty // 2
        half2 = qty - half1

        print(
            f"[WASP] sizing: equity={equity if equity else 'unknown'} risk_usd={risk_usd:.2f} R={R:.2f} qty_risk={qty_risk} qty_exposure={qty_exposure} chosen_qty={qty} fallback={fallback_used}"
        )

        # buying-power preflight
        if acct:
            try:
                bp = float(getattr(acct, "buying_power", 0))
                est_cost = entry * qty
                if est_cost > bp:
                    print(
                        f"[WASP] insufficient buying power: need ${est_cost:.2f}, have ${bp:.2f}; aborting and cooling down"
                    )
                    self.pending = None
                    self.cooldown_until = ts_now + timedelta(seconds=COOLDOWN_SEC)
                    return
            except Exception:
                pass

        if DRY_RUN:
            print(
                f"[WASP][DRY_RUN] Would submit orders: qty={qty} entry={entry:.2f} tp1={tp1:.2f} runner={runner_price:.2f} stop={stop:.2f}"
            )
            # cache state as if filled for testing flows
            self.position_qty = qty
            self.entry, self.stop, self.tp1 = entry, stop, tp1
            self.t161, self.t127 = t161, t127
            self.bracket_tp1_id = None
            self.bracket_runner_id = None
            self.sync_broker_state()
            self.pending = None
            return

        print(
            f"[WASP] ENTER {self.symbol} {qty} @ {entry:.2f} "
            f"(R={R:.2f} tp1={tp1:.2f} 1.618={t161:.2f} 1.272={t127:.2f} stop={stop:.2f})"
        )

        try:
            # ---- Half #1: bracket to +1R ----
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

            # ---- Half #2: bracket to 1.618 (or 1.272) ----
            o2 = self.client.submit_order(
                MarketOrderRequest(
                    symbol=self.symbol,
                    qty=half2,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(runner_price, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop, 2)),
                )
            )

            # cache state
            self.position_qty = qty
            self.entry, self.stop, self.tp1 = entry, stop, tp1
            self.t161, self.t127 = t161, t127
            self.bracket_tp1_id = o1.id
            self.bracket_runner_id = o2.id

            # ---------- NEW: re-sync after submit ----------
            self.sync_broker_state()

        except Exception as e:
            print("[WASP] Bracket submit failed:", e)
            # prevent repeated retries on persistent rejects
            self.pending = None
            self.cooldown_until = ts_now + timedelta(seconds=COOLDOWN_SEC)

    async def _manage(self, ts_now: datetime):
        # Placeholder: with bracket orders, moving the runner stop to BE
        # requires capturing the child stop leg ID from trade updates and
        # replacing it. We can wire this once you enable detailed on_fill events.
        return

    async def flatten_all(self, reason="EOD"):
        # Cancel both parent bracket orders (cancels attached legs)
        try:
            for oid in [self.bracket_tp1_id, self.bracket_runner_id]:
                if oid:
                    try:
                        self.client.cancel_order_by_id(oid)
                    except Exception:
                        pass
        except Exception:
            pass
        self.bracket_tp1_id = self.bracket_runner_id = None

        # Also try to close any residual position at market
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
        self.bracket_tp1_id = self.bracket_runner_id = None
        self.pending = None  # ensure no stale signal
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

    # Run market-data and trade-update streams with retries & cooperative shutdown
    def md_runner():
        attempt = 0
        while not stop_event.is_set():
            try:
                md = StockDataStream(API_KEY, API_SECRET, feed=FEED)
                md.subscribe_bars(bot.on_bar, SYMBOL)
                logging.info("[WASP] starting market data stream")
                md.run()
                logging.info("[WASP] market data stream stopped")
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
                    # Normalize event + symbol extraction across alpaca-py versions
                    ev = (
                        getattr(data, "event", None)
                        or getattr(data, "T", None)
                        or getattr(data, "event_type", None)
                    )
                    order = getattr(data, "order", None)
                    oid = getattr(order, "id", None) if order else None
                    sym = getattr(order, "symbol", None) if order else None
                    filled_qty = getattr(order, "filled_qty", None) or getattr(
                        order, "filled_quantity", None
                    )
                    status = getattr(order, "status", None)

                    # Compact debug log for trade updates
                    logging.info(
                        f"[WASP] trade_update ev={ev} sym={sym} oid={oid} filled={filled_qty} status={status}"
                    )
                    try:
                        append_trade_log(
                            {
                                "ts": datetime.now(TZ),
                                "event": str(ev),
                                "symbol": sym,
                                "order_id": oid,
                                "filled_qty": filled_qty,
                                "status": status,
                            }
                        )
                    except Exception:
                        pass

                    # Only care about updates for our symbol
                    if not sym or sym != SYMBOL:
                        return

                    # Only act on terminal / relevant events to avoid transient misreads
                    terminal_events = {"filled", "canceled", "expired", "rejected"}
                    ev_name = str(ev).lower() if ev else ""
                    if (
                        not any(t in ev_name for t in terminal_events)
                        and status not in terminal_events
                    ):
                        # ignore interim events (accepted, new, partially_filled, etc.)
                        return

                    # Attempt to sync broker state; if position reports zero, retry once after brief sleep
                    bot.sync_broker_state()
                    if bot.position_qty == 0:
                        try:
                            # short grace to avoid race with broker reporting
                            await asyncio.sleep(0.5)
                            bot.sync_broker_state()
                        except Exception:
                            pass

                    # If flat after the confirm, clear brackets/pending and resume scanning
                    if bot.position_qty == 0:
                        bot.bracket_tp1_id = bot.bracket_runner_id = None
                        bot.pending = None
                        bot.cooldown_until = datetime.now(TZ) + timedelta(
                            seconds=COOLDOWN_SEC
                        )
                        logging.info("[WASP] Trade update → flat. Back to scanning.")

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

    runners = [asyncio.to_thread(md_runner), asyncio.to_thread(trade_runner)]
    try:
        await asyncio.gather(*runners)
    finally:
        logging.info("[WASP] shutting down runners")
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
