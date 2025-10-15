import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Minimal indicator helpers (copied & simplified from wasp_v3)
TZ = pytz.timezone("America/New_York")


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


# Zigzag last-leg implementation (returns dict or None)
def zigzag_last_leg(df: pd.DataFrame, zz_atr_mult=1.7):
    if df is None or df.empty or len(df) < 5:
        return None
    if "atr" not in df:
        df = df.copy()
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


# ---- helpers ----


def load_parquet_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        st.error(f"Failed to read parquet {path}: {e}")
        return pd.DataFrame()
    # try to normalize column names
    # normalize common column name casing first
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for name in ("timestamp", "ts", "time", "date", "t"):
        if name in cols:
            mapping[cols[name]] = "ts"
            break
    for c in ["open", "high", "low", "close", "volume"]:
        if c in cols and cols[c] != c:
            mapping[cols[c]] = c
    if mapping:
        df = df.rename(columns=mapping)

    # If timestamp is stored in the index (common), reset it into a column named 'ts'
    try:
        if not ("ts" in df.columns):
            if isinstance(df.index, pd.DatetimeIndex):
                idx_name = df.index.name or "ts"
                df = df.reset_index()
                # if reset created a non-ts name, rename it to ts
                if idx_name != "ts" and idx_name in df.columns:
                    df = df.rename(columns={idx_name: "ts"})
            else:
                # try to find any datetime-like column
                dt_cols = [
                    c
                    for c in df.columns
                    if np.issubdtype(getattr(df[c], "dtype", object), np.datetime64)
                ]
                if dt_cols:
                    df = df.rename(columns={dt_cols[0]: "ts"})
                else:
                    # heuristic: find column name containing time/date substrings
                    for cand in df.columns:
                        if any(
                            k in cand.lower()
                            for k in ("time", "ts", "timestamp", "date", "t")
                        ):
                            df = df.rename(columns={cand: "ts"})
                            break
    except Exception:
        pass

    # ensure ts is datetime tz-aware
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")  # infer
        if df["ts"].isna().all():
            st.error("Timestamp column exists but could not be parsed into datetimes")
            return pd.DataFrame()
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC").dt.tz_convert(TZ)
        else:
            df["ts"] = df["ts"].dt.tz_convert(TZ)
    else:
        st.error("No timestamp column found in parquet")
        return pd.DataFrame()
    # ensure numeric columns
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---- UI ----

st.set_page_config(page_title="WASP offline viewer", layout="wide")
st.title("WASP — Offline SPY minute viewer (candles + EMA/RSI + fib zones)")

# find available files under data/*/SPY
import os

data_root = os.path.join(os.getcwd(), "data")
candidates = []
for tier in ("silver", "gold", "bronze"):
    p = os.path.join(data_root, tier, "SPY")
    if os.path.isdir(p):
        for f in sorted(os.listdir(p)):
            if f.endswith(".parquet"):
                candidates.append((tier, os.path.join(p, f)))

if not candidates:
    st.error(
        "No parquet minute files found under data/{silver,gold,bronze}/SPY. Make sure data is available."
    )
    st.stop()

# sidebar controls
tier_files = {t: [] for t, _ in candidates}
for t, fp in candidates:
    tier_files[t].append(fp)

sel_tier = st.sidebar.selectbox("Data tier", options=list(tier_files.keys()))
sel_file = st.sidebar.selectbox("Parquet file", options=tier_files[sel_tier])

# load
with st.spinner("Loading parquet..."):
    df = load_parquet_safe(sel_file)
if df.empty:
    st.stop()

# allow trimming range
start_ts = st.sidebar.time_input("Start time (ET)", value=time(9, 30))
end_ts = st.sidebar.time_input("End time (ET)", value=time(16, 0))

# filter by time range on the day(s) present
try:
    df = df.sort_values("ts").reset_index(drop=True)
    df["time"] = df["ts"].dt.time
    df = df[df["time"].between(start_ts, end_ts)]
except Exception:
    pass

# compute indicators
if len(df) < 2:
    st.warning("Not enough rows in file after filtering")
    st.dataframe(df.head())
    st.stop()

df["ema9"] = ema(df["close"], 9)
df["ema21"] = ema(df["close"], 21)
df["rsi"] = rsi(df["close"], 14)
df["atr"] = atr(df, 14)

# compute last zigzag on the visible slice
fib = zigzag_last_leg(df, zz_atr_mult=1.7)

# create plotly figure with two rows
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02
)

# candles
fig.add_trace(
    go.Candlestick(
        x=df["ts"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="candles",
    ),
    row=1,
    col=1,
)
# ema lines
fig.add_trace(
    go.Scatter(
        x=df["ts"], y=df["ema9"], mode="lines", name="EMA9", line=dict(color="orange")
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df["ts"], y=df["ema21"], mode="lines", name="EMA21", line=dict(color="blue")
    ),
    row=1,
    col=1,
)

# fib shading if present
if fib and fib.get("direction") == "up":
    lo, hi, rng = fib["low"], fib["high"], fib["range"]
    L50, L618 = fib["levels"]["50%"], fib["levels"]["61.8%"]
    zlow = min(L50, L618) - 0.02 * rng
    zhigh = max(L50, L618) + 0.02 * rng
    # shaded rectangle
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=df["ts"].iloc[0],
        x1=df["ts"].iloc[-1],
        y0=zlow,
        y1=zhigh,
        fillcolor="LightSalmon",
        opacity=0.2,
        layer="below",
        row=1,
        col=1,
    )
    # lines for 50/61.8
    fig.add_hline(y=L50, line=dict(color="red", dash="dash"), row=1, col=1)
    fig.add_hline(y=L618, line=dict(color="red", dash="dot"), row=1, col=1)

# RSI subplot
fig.add_trace(
    go.Scatter(
        x=df["ts"], y=df["rsi"], mode="lines", name="RSI", line=dict(color="purple")
    ),
    row=2,
    col=1,
)
fig.add_hline(y=70, line=dict(color="gray", dash="dash"), row=2, col=1)
fig.add_hline(y=30, line=dict(color="gray", dash="dash"), row=2, col=1)

fig.update_layout(
    height=800,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# data table + download
with st.expander("Data & indicators"):
    st.dataframe(
        df[
            [
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "ema9",
                "ema21",
                "rsi",
                "atr",
            ]
        ].tail(200)
    )
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV of full file",
        data=csv,
        file_name=os.path.basename(sel_file).replace(".parquet", ".csv"),
    )

st.caption(f"File: {sel_file}")
