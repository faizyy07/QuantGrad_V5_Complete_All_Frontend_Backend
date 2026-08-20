"""
feature_engine.py — QuantGrad v3
==================================
Full 65-feature pipeline across 7 streams:

  Stream 1 ( 8): Gradient physics — gradient, accel, jerk, persistence
  Stream 2 ( 6): Multi-scale momentum — ROC 5/20/50, divergence, alignment
  Stream 3 ( 8): Volume analysis — delta, Z-score, spike, VWAP, buying pressure
  Stream 4 (17): Market structure — swing, MSS, OB, FVG, sweep, premium (from market_structure.py)
  Stream 5 ( 9): Trend strength — ADX, DI+/-, EMA 20/50/200, spread, VWAP pos
  Stream 6 ( 7): Classic technicals — RSI, MACD, Bollinger, OB spread/imbalance
  Stream 7 (10): Macro + correlations — DXY/Gold/Crude corr, momenta, Fear&Greed, Fed, BTC dom
  ────────────────────────────────────────────────────────────────────
  Total    (65)
"""

import os, time, requests
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

BINANCE_BASE = "https://api.binance.com"
WINDOW_SIZE  = 60
FEATURE_DIM  = 65


# ══════════════════════════════════════════════════════════════════════
#  BINANCE FETCH
# ══════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol="BTCUSDT", interval="1h", total_bars=500):
    url        = f"{BINANCE_BASE}/api/v3/klines"
    all_candles = []
    end_time   = None
    remaining  = total_bars

    print(f"[binance] Fetching {total_bars} × {interval} for {symbol}...")

    while remaining > 0:
        limit  = min(remaining, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time
        try:
            resp    = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            candles = resp.json()
        except Exception as e:
            print(f"[binance] Error: {e}")
            break
        if not candles:
            break
        all_candles = candles + all_candles
        end_time    = candles[0][0] - 1
        remaining  -= len(candles)
        if len(candles) < limit:
            break
        time.sleep(0.2)

    if not all_candles:
        raise RuntimeError(f"No data for {symbol}")

    df = pd.DataFrame(all_candles, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["time"]            = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume","taker_buy_base","taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("open_time").sort_values("time").reset_index(drop=True)
    df = df.set_index("time")
    print(f"[binance] Got {len(df)} candles: {df.index[0]} -> {df.index[-1]}")
    return df


def fetch_order_book(symbol="BTCUSDT"):
    try:
        resp = requests.get(f"{BINANCE_BASE}/api/v3/depth",
                            params={"symbol": symbol, "limit": 20}, timeout=10)
        resp.raise_for_status()
        data    = resp.json()
        bids    = np.array(data["bids"], dtype=float)
        asks    = np.array(data["asks"], dtype=float)
        bid_vol = bids[:, 1].sum()
        ask_vol = asks[:, 1].sum()
        return {
            "spread":     float(asks[0,0] - bids[0,0]),
            "imbalance":  float((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)),
            "bid_volume": float(bid_vol),
            "ask_volume": float(ask_vol),
        }
    except Exception as e:
        print(f"[binance] OB error: {e}")
        return {"spread": 1.0, "imbalance": 0.0, "bid_volume": 0.0, "ask_volume": 0.0}


# ══════════════════════════════════════════════════════════════════════
#  HILL CLIMB — kept from v2
# ══════════════════════════════════════════════════════════════════════

def stochastic_hill_climb(prices, n_iter=80, step=0.12,
                           target_min=5, target_max=20):
    """Auto-tune prominence + distance for extrema detection."""
    best_prom, best_dist, best_score = 0.5, 5, -np.inf
    for _ in range(n_iter):
        prom = max(0.05, best_prom + np.random.randn() * step)
        dist = max(2, int(best_dist + np.random.randn() * step * 10))
        pk,  _ = find_peaks( prices, prominence=prom, distance=dist)
        tr,  _ = find_peaks(-prices, prominence=prom, distance=dist)
        n_ext  = len(pk) + len(tr)
        score  = n_ext if target_min <= n_ext <= target_max else -abs(n_ext - (target_min+target_max)//2)
        if score > best_score:
            best_score, best_prom, best_dist = score, prom, dist
    return best_prom, best_dist


# ══════════════════════════════════════════════════════════════════════
#  STREAM 1 — GRADIENT PHYSICS
# ══════════════════════════════════════════════════════════════════════

def compute_stream1(mid):
    """Gradient, acceleration, jerk, persistence. 8 features."""
    grad  = np.gradient(mid)
    accel = np.gradient(grad)
    jerk  = np.gradient(accel)   # 3rd derivative — momentum flip detector

    # Gradient sign persistence — consecutive candles same direction
    persistence = np.zeros(len(mid))
    count = 0
    for i in range(1, len(mid)):
        if np.sign(grad[i]) == np.sign(grad[i-1]):
            count += 1
        else:
            count = 0
        persistence[i] = count

    # Normalized gradient vs 50-period rolling mean
    grad_s     = pd.Series(grad)
    grad_mean  = grad_s.rolling(50).mean().fillna(0).values
    grad_std   = grad_s.rolling(50).std().fillna(1).values
    grad_norm  = (grad - grad_mean) / (grad_std + 1e-9)

    # Acceleration sign (is momentum building or dying)
    accel_sign = np.sign(accel)

    # Momentum cycle position: 0=early, 1=peak, based on grad/accel combo
    cycle_pos  = np.where((grad > 0) & (accel > 0), 0.25,
                 np.where((grad > 0) & (accel < 0), 0.75,
                 np.where((grad < 0) & (accel < 0), 0.25, 0.75)))

    out = pd.DataFrame()
    out["gradient"]      = grad
    out["acceleration"]  = accel
    out["jerk"]          = jerk
    out["grad_persist"]  = persistence / 50.0
    out["grad_norm"]     = np.clip(grad_norm, -3, 3) / 3
    out["accel_sign"]    = accel_sign
    out["cycle_pos"]     = cycle_pos
    out["grad_abs"]      = np.abs(grad_norm)
    return out  # 8 features


# ══════════════════════════════════════════════════════════════════════
#  STREAM 2 — MULTI-SCALE MOMENTUM
# ══════════════════════════════════════════════════════════════════════

def compute_stream2(close):
    """ROC 5/20/50, divergence, alignment. 6 features."""
    s     = pd.Series(close)
    roc5  = s.pct_change(5).fillna(0).values
    roc20 = s.pct_change(20).fillna(0).values
    roc50 = s.pct_change(50).fillna(0).values

    # Momentum alignment — all three same sign
    align = np.sign(roc5) * np.sign(roc20) * np.sign(roc50)

    # Divergence — price up but momentum declining
    price_up    = (s.diff(20) > 0).astype(float).values
    mom_decline = (pd.Series(roc20).diff() < 0).astype(float).values
    divergence  = price_up * mom_decline  # bearish divergence flag

    out = pd.DataFrame()
    out["roc5"]       = np.clip(roc5,  -0.1, 0.1) / 0.1
    out["roc20"]      = np.clip(roc20, -0.2, 0.2) / 0.2
    out["roc50"]      = np.clip(roc50, -0.3, 0.3) / 0.3
    out["mom_align"]  = align
    out["divergence"] = divergence
    out["roc_spread"] = np.clip(roc5 - roc50, -0.1, 0.1) / 0.1
    return out  # 6 features


# ══════════════════════════════════════════════════════════════════════
#  STREAM 3 — VOLUME ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def compute_stream3(df, ob):
    """Volume Z-score, spike, VWAP, buying pressure. 8 features."""
    close      = df["close"].values
    volume     = df["volume"].values
    taker_buy  = df["taker_buy_base"].values
    vol_delta  = taker_buy - (volume - taker_buy)

    # Volume Z-score (50-period)
    vol_s      = pd.Series(volume)
    vol_mean   = vol_s.rolling(50).mean().fillna(volume.mean()).values
    vol_std    = vol_s.rolling(50).std().fillna(volume.std() + 1e-9).values
    vol_zscore = (volume - vol_mean) / (vol_std + 1e-9)
    vol_spike  = (vol_zscore > 2.5).astype(float)

    # Volume trend (expanding or contracting)
    vol_trend  = pd.Series(volume).pct_change(10).fillna(0).values

    # VWAP deviation
    typical    = (df["high"].values + df["low"].values + close) / 3
    vwap       = pd.Series(typical * volume).cumsum() / pd.Series(volume).cumsum()
    vwap       = vwap.bfill().values
    vwap_dev   = (close - vwap) / (vwap + 1e-9)

    # Buying pressure ratio
    buy_ratio  = taker_buy / (volume + 1e-9)

    # OB features (broadcast)
    ob_imb     = np.full(len(close), ob["imbalance"])
    ob_spr     = np.full(len(close), ob["spread"])

    out = pd.DataFrame()
    out["vol_delta"]   = np.clip(vol_delta / (vol_mean + 1e-9), -3, 3) / 3
    out["vol_zscore"]  = np.clip(vol_zscore, -3, 3) / 3
    out["vol_spike"]   = vol_spike
    out["vol_trend"]   = np.clip(vol_trend, -1, 1)
    out["vwap_dev"]    = np.clip(vwap_dev, -0.05, 0.05) / 0.05
    out["buy_ratio"]   = buy_ratio - 0.5   # center at 0
    out["ob_imbalance"]= ob_imb
    out["ob_spread"]   = np.clip(ob_spr / 100, 0, 1)
    return out  # 8 features


# ══════════════════════════════════════════════════════════════════════
#  STREAM 5 — TREND STRENGTH
# ══════════════════════════════════════════════════════════════════════

def compute_stream5(df):
    """ADX, DI+/-, EMA alignment, VWAP position. 9 features."""
    from market_structure import compute_adx_simple
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values

    adx   = compute_adx_simple(high, low, close)

    # DI+ and DI-
    tr    = np.maximum(high - low,
            np.maximum(abs(high - np.roll(close, 1)),
                       abs(low  - np.roll(close, 1))))
    up    = np.maximum(high - np.roll(high, 1), 0)
    down  = np.maximum(np.roll(low, 1) - low,   0)
    atr14 = pd.Series(tr).rolling(14).mean().fillna(1).values
    di_p  = 100 * pd.Series(np.where(up > down, up, 0)).rolling(14).mean().fillna(0).values / (atr14 + 1e-9)
    di_n  = 100 * pd.Series(np.where(down > up, down, 0)).rolling(14).mean().fillna(0).values / (atr14 + 1e-9)

    # EMA alignment
    s      = pd.Series(close)
    ema20  = s.ewm(span=20).mean().values
    ema50  = s.ewm(span=50).mean().values
    ema200 = s.ewm(span=200).mean().values
    ema_align = np.where((ema20 > ema50) & (ema50 > ema200), 1.0,
                np.where((ema20 < ema50) & (ema50 < ema200), -1.0, 0.0))
    ema_spread = (ema20 - ema50) / (close + 1e-9)  # trend extension

    # VWAP position
    typical  = (df["high"].values + low + close) / 3
    vwap     = (pd.Series(typical * df["volume"].values).cumsum() /
                pd.Series(df["volume"].values).cumsum()).bfill().values
    vwap_pos = np.sign(close - vwap)

    # ADX slope
    adx_slope = np.gradient(adx)

    out = pd.DataFrame()
    out["adx"]         = np.clip(adx, 0, 100) / 100
    out["di_plus"]     = np.clip(di_p, 0, 100) / 100
    out["di_minus"]    = np.clip(di_n, 0, 100) / 100
    out["ema_align"]   = ema_align
    out["ema_spread"]  = np.clip(ema_spread, -0.05, 0.05) / 0.05
    out["ema20_50"]    = np.sign(ema20 - ema50)
    out["ema50_200"]   = np.sign(ema50 - ema200)
    out["vwap_pos"]    = vwap_pos
    out["adx_slope"]   = np.clip(adx_slope, -5, 5) / 5
    return out  # 9 features


# ══════════════════════════════════════════════════════════════════════
#  STREAM 6 — CLASSIC TECHNICALS (kept from v2)
# ══════════════════════════════════════════════════════════════════════

def compute_rsi(prices, period=14):
    rsi  = np.full(len(prices), 50.0)
    if len(prices) < period + 1:
        return rsi
    deltas   = np.diff(prices)
    up_arr   = np.where(deltas > 0, deltas, 0.0)
    down_arr = np.where(deltas < 0, -deltas, 0.0)
    avg_up   = up_arr[:period].mean()
    avg_down = down_arr[:period].mean()
    for i in range(period, len(prices)):
        avg_up   = (avg_up   * (period-1) + up_arr[i-1])   / period
        avg_down = (avg_down * (period-1) + down_arr[i-1]) / period
        rs       = avg_up / (avg_down + 1e-9)
        rsi[i]   = 100 - 100 / (1 + rs)
    return rsi


def compute_stream6(df, ob):
    """RSI, MACD, Bollinger, log return. 7 features."""
    close = df["close"].values
    mid   = (df["open"].values + close) / 2
    s     = pd.Series(mid)

    rsi   = compute_rsi(mid)
    fast  = s.ewm(span=12).mean()
    slow  = s.ewm(span=26).mean()
    macd  = (fast - slow).values
    sig   = pd.Series(macd).ewm(span=9).mean().values
    macd_hist = macd - sig

    roll_m = s.rolling(20).mean()
    roll_s = s.rolling(20).std()
    bb_pct = ((s - (roll_m - 2*roll_s)) / (4*roll_s + 1e-9)).fillna(0.5).values
    bb_wid = (4 * roll_s / (roll_m + 1e-9)).fillna(0).values
    log_r  = np.log(close / np.roll(close, 1) + 1e-9)
    log_r[0] = 0

    out = pd.DataFrame()
    out["rsi"]       = (rsi - 50) / 50
    out["macd"]      = np.clip(macd / (mid.mean() + 1e-9), -0.01, 0.01) / 0.01
    out["macd_hist"] = np.clip(macd_hist / (mid.mean() + 1e-9), -0.01, 0.01) / 0.01
    out["bb_pct"]    = np.clip(bb_pct, 0, 1)
    out["bb_width"]  = np.clip(bb_wid, 0, 0.1) / 0.1
    out["log_return"]= np.clip(log_r, -0.05, 0.05) / 0.05
    out["mid_price"] = (mid - mid.mean()) / (mid.std() + 1e-9)
    return out  # 7 features


# ══════════════════════════════════════════════════════════════════════
#  STREAM 7 — MACRO (kept from v2)
# ══════════════════════════════════════════════════════════════════════

def compute_stream7(df, macro):
    """Macro correlations + regime. 10 features."""
    btc = df["close"]
    idx = df.index

    def _to_hourly(dates, values):
        if not dates or not values:
            return pd.Series(np.nan, index=idx)
        daily = pd.Series(values, index=pd.to_datetime(dates))
        daily = daily[~daily.index.duplicated()]
        return daily.reindex(idx, method="ffill")

    dxy   = _to_hourly(macro["dxy"].get("dates",[]),   macro["dxy"].get("series",[]))
    gold  = _to_hourly(macro["gold"].get("dates",[]),  macro["gold"].get("series",[]))
    crude = _to_hourly(macro["crude"].get("dates",[]), macro["crude"].get("series",[]))

    w = 48
    out = pd.DataFrame(index=idx)
    out["corr_dxy"]    = btc.rolling(w).corr(dxy).fillna(0)
    out["corr_gold"]   = btc.rolling(w).corr(gold).fillna(0)
    out["corr_crude"]  = btc.rolling(w).corr(crude).fillna(0)
    out["dxy_mom"]     = macro["dxy"].get("momentum_5d", 0.0) / 10
    out["gold_mom"]    = macro["gold"].get("momentum_5d", 0.0) / 10
    out["crude_mom"]   = macro["crude"].get("momentum_5d", 0.0) / 10

    def _safe(path, default=0.0):
        try:
            v = macro
            for k in path: v = v[k]
            return float(v)
        except: return default

    out["fear_greed"]  = _safe(["fear_greed","current"], 50) / 100
    out["btc_dom"]     = _safe(["btc_dominance","btc_dominance_pct"], 52) / 100
    out["fed_rate"]    = _safe(["fed","rate"], 5.33) / 10
    out["fomc_flag"]   = _safe(["fed","event_flag_24h"], 0.0)
    return out.fillna(0)  # 10 features


# ══════════════════════════════════════════════════════════════════════
#  MASTER FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_feature_matrix(df, macro, ob, struct_df=None, auto_tune=True,
                          prominence=None, distance=None):
    """
    Build full 65-feature matrix.
    struct_df: pre-computed structure features (pass to avoid recomputing)
    """
    from market_structure import build_structure_features

    close = df["close"].values
    mid   = (df["open"].values + close) / 2

    if struct_df is None:
        print("[features] Computing market structure...")
        struct_df = build_structure_features(df)

    print("[features] Stream 1: gradient physics...")
    s1 = compute_stream1(mid)
    s1.index = df.index

    print("[features] Stream 2: multi-scale momentum...")
    s2 = compute_stream2(close)
    s2.index = df.index

    print("[features] Stream 3: volume analysis...")
    s3 = compute_stream3(df, ob)
    s3.index = df.index

    # Stream 4 = struct_df (17 features) — already computed
    s4 = struct_df.copy()

    print("[features] Stream 5: trend strength...")
    s5 = compute_stream5(df)
    s5.index = df.index

    print("[features] Stream 6: classic technicals...")
    s6 = compute_stream6(df, ob)
    s6.index = df.index

    print("[features] Stream 7: macro + correlations...")
    s7 = compute_stream7(df, macro)

    features = pd.concat([s1, s2, s3, s4, s5, s6, s7], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    actual = features.shape[1]
    if actual != FEATURE_DIM:
        print(f"[features] WARNING: expected {FEATURE_DIM}, got {actual} features")
        print(f"[features] Columns: {list(features.columns)}")

    print(f"[features] Matrix: {features.shape}")
    return features, struct_df


def get_feature_names():
    return (
        # Stream 1
        ["gradient","acceleration","jerk","grad_persist","grad_norm",
         "accel_sign","cycle_pos","grad_abs"] +
        # Stream 2
        ["roc5","roc20","roc50","mom_align","divergence","roc_spread"] +
        # Stream 3
        ["vol_delta","vol_zscore","vol_spike","vol_trend","vwap_dev",
         "buy_ratio","ob_imbalance","ob_spread"] +
        # Stream 4
        ["swing_high_flag","swing_low_flag","mss_bullish","mss_bearish",
         "trend_structure","ob_bull_active","ob_bear_active","ob_strength",
         "fvg_bull","fvg_bear","fvg_dist","sweep_bull","sweep_bear",
         "sweep_recency","premium_discount","dist_to_sh","dist_to_sl"] +
        # Stream 5
        ["adx","di_plus","di_minus","ema_align","ema_spread",
         "ema20_50","ema50_200","vwap_pos","adx_slope"] +
        # Stream 6
        ["rsi","macd","macd_hist","bb_pct","bb_width","log_return","mid_price"] +
        # Stream 7
        ["corr_dxy","corr_gold","corr_crude","dxy_mom","gold_mom","crude_mom",
         "fear_greed","btc_dom","fed_rate","fomc_flag"]
    )


# ══════════════════════════════════════════════════════════════════════
#  SLIDING WINDOWS
# ══════════════════════════════════════════════════════════════════════

def build_windows(features, labels, window=WINDOW_SIZE):
    X, y = [], []
    for i in range(window, len(features)):
        X.append(features[i-window:i])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
