"""
macro_fetcher.py
================
Fetches all macro data using FREE APIs only:
  - Yahoo Finance  : DXY, Gold, Crude Oil (no key needed)
  - FRED           : Fed Funds Rate (free key)
  - Alternative.me : Fear & Greed Index (no key needed)
  - CoinGecko      : BTC Dominance (no key needed)

Run this daily or call fetch_all_macro() from trainer/dashboard.
Output saved to: data/macro_cache.json
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
MACRO_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "macro_cache.json")


# ─────────────────────────────────────────────
#  Yahoo Finance helpers (DXY, Gold, Crude)
# ─────────────────────────────────────────────

def fetch_yfinance_series(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.Series:
    """Fetch daily close prices from Yahoo Finance. Completely free, no key."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            print(f"[macro] WARNING: No data for {ticker}")
            return pd.Series(dtype=float)
        series = data["Close"].squeeze()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return series.dropna()
    except Exception as e:
        print(f"[macro] ERROR fetching {ticker}: {e}")
        return pd.Series(dtype=float)


def fetch_dxy(period: str = "6mo") -> dict:
    """Dollar Index (DXY) via Yahoo Finance."""
    series = fetch_yfinance_series("DX-Y.NYB", period=period)
    if series.empty:
        return {"current": 103.0, "momentum_5d": 0.0, "series": []}
    current = float(series.iloc[-1])
    momentum = float((series.iloc[-1] - series.iloc[-6]) / series.iloc[-6] * 100) if len(series) >= 6 else 0.0
    return {
        "current": round(current, 4),
        "momentum_5d": round(momentum, 4),
        "series": series.tail(90).round(4).tolist(),
        "dates": series.tail(90).index.strftime("%Y-%m-%d").tolist()
    }


def fetch_gold(period: str = "6mo") -> dict:
    """Gold futures (GC=F) via Yahoo Finance."""
    series = fetch_yfinance_series("GC=F", period=period)
    if series.empty:
        return {"current": 2300.0, "momentum_5d": 0.0, "series": []}
    current = float(series.iloc[-1])
    momentum = float((series.iloc[-1] - series.iloc[-6]) / series.iloc[-6] * 100) if len(series) >= 6 else 0.0
    return {
        "current": round(current, 4),
        "momentum_5d": round(momentum, 4),
        "series": series.tail(90).round(4).tolist(),
        "dates": series.tail(90).index.strftime("%Y-%m-%d").tolist()
    }


def fetch_crude(period: str = "6mo") -> dict:
    """WTI Crude Oil futures (CL=F) via Yahoo Finance."""
    series = fetch_yfinance_series("CL=F", period=period)
    if series.empty:
        return {"current": 80.0, "momentum_5d": 0.0, "series": []}
    current = float(series.iloc[-1])
    momentum = float((series.iloc[-1] - series.iloc[-6]) / series.iloc[-6] * 100) if len(series) >= 6 else 0.0
    return {
        "current": round(current, 4),
        "momentum_5d": round(momentum, 4),
        "series": series.tail(90).round(4).tolist(),
        "dates": series.tail(90).index.strftime("%Y-%m-%d").tolist()
    }


# ─────────────────────────────────────────────
#  FRED — Federal Reserve (free key)
# ─────────────────────────────────────────────

def fetch_fed_rate() -> dict:
    """
    Fetch Fed Funds Effective Rate from FRED.
    Series: FEDFUNDS (monthly) — free API key from fred.stlouisfed.org
    Falls back to hardcoded value if no key provided.
    """
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        print("[macro] No FRED key — using fallback fed rate 5.33")
        return {"rate": 5.33, "last_change_date": "2023-07-26", "source": "fallback"}

    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "FEDFUNDS",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 12,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        observations = resp.json()["observations"]
        latest = next(o for o in observations if o["value"] != ".")
        rate = float(latest["value"])
        date = latest["date"]
        return {"rate": round(rate, 4), "last_change_date": date, "source": "FRED"}
    except Exception as e:
        print(f"[macro] FRED error: {e} — using fallback")
        return {"rate": 5.33, "last_change_date": "2023-07-26", "source": "fallback"}


def fetch_fomc_dates() -> dict:
    """
    Known upcoming FOMC dates (hardcoded — they are publicly announced months ahead).
    Returns nearest past and future FOMC dates and a ±24h event flag.
    """
    # FOMC meeting dates (end dates when decisions are announced)
    fomc_dates = [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    ]
    today = datetime.utcnow()
    fomc_dt = [datetime.strptime(d, "%Y-%m-%d") for d in fomc_dates]

    past = [d for d in fomc_dt if d <= today]
    future = [d for d in fomc_dt if d > today]

    last_fomc = past[-1] if past else today - timedelta(days=30)
    # Never fabricate a decision date. Once the official calendar list is
    # exhausted, leave the future event flag inactive until it is updated.
    next_fomc = future[0] if future else None

    hours_since = abs((today - last_fomc).total_seconds() / 3600)
    hours_until = abs((next_fomc - today).total_seconds() / 3600) if next_fomc else float("inf")
    event_flag = 1 if (hours_since <= 24 or hours_until <= 24) else 0

    return {
        "last_fomc": last_fomc.strftime("%Y-%m-%d"),
        "next_fomc": next_fomc.strftime("%Y-%m-%d") if next_fomc else "n/a",
        "hours_until_next": round(hours_until, 1) if next_fomc else None,
        "event_flag_24h": event_flag,
    }


# ─────────────────────────────────────────────
#  Alternative.me — Fear & Greed (no key)
# ─────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    """
    Crypto Fear & Greed Index from Alternative.me.
    Completely free, no API key required.
    0 = Extreme Fear, 100 = Extreme Greed
    """
    try:
        url = "https://api.alternative.me/fng/?limit=30&format=json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        current = int(data[0]["value"])
        label = data[0]["value_classification"]
        history = [int(d["value"]) for d in data]
        return {
            "current": current,
            "label": label,
            "7d_avg": round(np.mean(history[:7]), 1),
            "30d_avg": round(np.mean(history), 1),
            "history_30d": history,
        }
    except Exception as e:
        print(f"[macro] Fear & Greed error: {e} — using fallback")
        return {"current": 50, "label": "Neutral", "7d_avg": 50.0, "30d_avg": 50.0, "history_30d": [50] * 30}


# ─────────────────────────────────────────────
#  CoinGecko — BTC Dominance (no key)
# ─────────────────────────────────────────────

def fetch_btc_dominance() -> dict:
    """
    BTC market dominance from CoinGecko public API.
    No API key required for basic endpoints.
    """
    try:
        url = "https://api.coingecko.com/api/v3/global"
        headers = {"accept": "application/json"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        dominance = data["market_cap_percentage"]["btc"]
        total_mcap = data["total_market_cap"]["usd"]
        return {
            "btc_dominance_pct": round(dominance, 2),
            "total_market_cap_usd": total_mcap,
            "source": "CoinGecko"
        }
    except Exception as e:
        print(f"[macro] BTC dominance error: {e} — using fallback")
        return {"btc_dominance_pct": 52.0, "total_market_cap_usd": 0, "source": "fallback"}


# ─────────────────────────────────────────────
#  BTC post-Fed reaction (historical average)
# ─────────────────────────────────────────────

def compute_fed_reaction_feature(btc_hourly_df: pd.DataFrame, fomc_info: dict) -> dict:
    """
    Compute average BTC % return in 1h and 4h windows after the last 3 FOMC events.
    btc_hourly_df must have a DatetimeIndex and a 'close' column.
    """
    fomc_dates_str = [
        "2024-09-18", "2024-07-31", "2024-06-12",
        "2024-05-01", "2024-03-20", "2024-01-31",
    ]
    reactions_1h, reactions_4h = [], []

    if btc_hourly_df is None or btc_hourly_df.empty:
        return {"avg_return_1h": 0.0, "avg_return_4h": 0.0}

    df = btc_hourly_df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    for ds in fomc_dates_str[:3]:
        try:
            event_dt = datetime.strptime(ds, "%Y-%m-%d").replace(hour=14)  # Fed announces ~2pm ET
            mask_base = df.index[df.index <= event_dt]
            mask_1h   = df.index[df.index <= event_dt + timedelta(hours=1)]
            mask_4h   = df.index[df.index <= event_dt + timedelta(hours=4)]
            if len(mask_base) == 0 or len(mask_1h) == 0 or len(mask_4h) == 0:
                continue
            p0  = float(df.loc[mask_base[-1], "close"])
            p1h = float(df.loc[mask_1h[-1], "close"])
            p4h = float(df.loc[mask_4h[-1], "close"])
            reactions_1h.append((p1h - p0) / p0 * 100)
            reactions_4h.append((p4h - p0) / p0 * 100)
        except Exception:
            continue

    return {
        "avg_return_1h": round(np.mean(reactions_1h) if reactions_1h else 0.0, 4),
        "avg_return_4h": round(np.mean(reactions_4h) if reactions_4h else 0.0, 4),
    }


# ─────────────────────────────────────────────
#  Master fetch function
# ─────────────────────────────────────────────

def fetch_all_macro(btc_hourly_df: pd.DataFrame = None, save: bool = True) -> dict:
    """
    Fetches all macro data and optionally saves to data/macro_cache.json.
    Call this daily from a scheduler or manually.
    """
    print("[macro] Fetching DXY...")
    dxy = fetch_dxy()

    print("[macro] Fetching Gold...")
    gold = fetch_gold()

    print("[macro] Fetching Crude Oil...")
    crude = fetch_crude()

    print("[macro] Fetching Fed Rate...")
    fed = fetch_fed_rate()

    print("[macro] Fetching FOMC dates...")
    fomc = fetch_fomc_dates()

    print("[macro] Fetching Fear & Greed...")
    fg = fetch_fear_greed()

    print("[macro] Fetching BTC Dominance...")
    btc_dom = fetch_btc_dominance()

    print("[macro] Computing Fed reaction features...")
    fed_reaction = compute_fed_reaction_feature(btc_hourly_df, fomc)

    macro = {
        "fetched_at": datetime.utcnow().isoformat(),
        "dxy": dxy,
        "gold": gold,
        "crude": crude,
        "fed": {**fed, **fomc},
        "fear_greed": fg,
        "btc_dominance": btc_dom,
        "fed_reaction": fed_reaction,
    }

    if save:
        os.makedirs(os.path.dirname(MACRO_CACHE_PATH), exist_ok=True)
        with open(MACRO_CACHE_PATH, "w") as f:
            json.dump(macro, f, indent=2)
        print(f"[macro] Saved to {MACRO_CACHE_PATH}")

    return macro


def load_macro_cache() -> dict:
    """Load macro cache from disk. Warns if stale > 24h."""
    if not os.path.exists(MACRO_CACHE_PATH):
        print("[macro] No cache found — running fetch_all_macro()...")
        return fetch_all_macro()

    with open(MACRO_CACHE_PATH, "r") as f:
        cache = json.load(f)

    fetched_at = datetime.fromisoformat(cache.get("fetched_at", "2000-01-01"))
    age_hours = (datetime.utcnow() - fetched_at).total_seconds() / 3600

    if age_hours > 24:
        print(f"[macro] WARNING: Macro cache is {age_hours:.1f}h old. Run macro_fetcher.py to refresh.")

    return cache


def is_cache_stale(threshold_hours: float = 24.0) -> bool:
    if not os.path.exists(MACRO_CACHE_PATH):
        return True
    with open(MACRO_CACHE_PATH, "r") as f:
        cache = json.load(f)
    fetched_at = datetime.fromisoformat(cache.get("fetched_at", "2000-01-01"))
    age_hours = (datetime.utcnow() - fetched_at).total_seconds() / 3600
    return age_hours > threshold_hours


if __name__ == "__main__":
    print("=" * 50)
    print("QuantGrad Macro Fetcher")
    print("=" * 50)
    macro = fetch_all_macro(btc_hourly_df=None, save=True)
    print("\n--- Summary ---")
    print(f"DXY:           {macro['dxy']['current']} (5d momentum: {macro['dxy']['momentum_5d']}%)")
    print(f"Gold:          {macro['gold']['current']} (5d momentum: {macro['gold']['momentum_5d']}%)")
    print(f"Crude:         {macro['crude']['current']} (5d momentum: {macro['crude']['momentum_5d']}%)")
    print(f"Fed Rate:      {macro['fed']['rate']}%")
    print(f"Next FOMC:     {macro['fed']['next_fomc']} ({macro['fed']['hours_until_next']}h away)")
    print(f"Fear & Greed:  {macro['fear_greed']['current']} ({macro['fear_greed']['label']})")
    print(f"BTC Dominance: {macro['btc_dominance']['btc_dominance_pct']}%")
    print(f"Fed Reaction:  +1h avg: {macro['fed_reaction']['avg_return_1h']}%, +4h avg: {macro['fed_reaction']['avg_return_4h']}%")
