"""
QuantGrad v4 REST terminal server.

This file is an adapter around the existing feature engineering, quantum/PCA,
three-model inference, fusion, and backtest functions. It deliberately does
not alter the underlying ML logic or artifact contract.

Run locally:
    python server.py
Then open http://localhost:8000 after copying/building the frontend into web/.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.dirname(APP_DIR)
ARTIFACTS_DIR = os.path.join(APP_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
# The complete bundle keeps the original Python pipeline in backend/ and the
# compiled V5 interface in the adjacent web/ directory.
WEB_DIR = os.path.join(PACKAGE_DIR, "web")
MODEL_FILES = {
    "trend": "trend_model.keras",
    "structure": "structure_model.keras",
    "entry": "entry_model.keras",
    "scaler": "scaler.pkl",
    "pca": "pca.pkl",
    "quantum": "quantum_params.pkl",
}
FEATURE_GROUPS = {
    "Gradient": ["gradient", "acceleration", "jerk", "grad_norm"],
    "Momentum": ["roc5", "roc20", "roc50", "roc_spread"],
    "Volume": ["vol_delta", "vol_zscore", "vwap_dev", "buy_ratio"],
    "Structure": ["mss_bullish", "mss_bearish", "ob_strength", "fvg_dist", "sweep_recency"],
    "Trend": ["adx", "di_plus", "di_minus", "ema_spread"],
    "Technical": ["rsi", "macd", "bb_position"],
    "Macro": ["corr_dxy", "corr_gold", "corr_crude", "fear_greed", "btc_dom"],
}

app = FastAPI(title="QuantGrad v4 Terminal API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _missing_artifacts() -> list[str]:
    return [name for name, filename in MODEL_FILES.items() if not os.path.exists(os.path.join(ARTIFACTS_DIR, filename))]


def _native(value: Any) -> Any:
    """Convert pandas/numpy objects to compact JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return 0.0
    return value


def _serialise_candles(frame: pd.DataFrame) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for timestamp, row in frame.tail(360).iterrows():
        rows.append({
            "time": int(pd.Timestamp(timestamp).timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    return rows


def _feature_streams(features: pd.DataFrame, points: int = 32) -> list[dict[str, Any]]:
    tail = features.tail(points)
    output: list[dict[str, Any]] = []
    for name, requested in FEATURE_GROUPS.items():
        available = [column for column in requested if column in tail.columns]
        if not available:
            continue
        series = tail[available].mean(axis=1).to_numpy(dtype=float)
        scale = float(np.nanstd(series))
        normalised = series if scale < 1e-9 else series / scale
        output.append({"name": name, "values": np.clip(np.nan_to_num(normalised), -1.0, 1.0).tolist()})
    return output


def _backtest_stats(predictions: np.ndarray, prices: np.ndarray, hold_bars: int = 3) -> dict[str, Any]:
    """Exact dashboard rolling strategy simulation, returned as JSON-safe data."""
    equity = [1.0]
    wins = losses = 0
    gross_profit = gross_loss = 0.0
    peak = 1.0
    max_drawdown = 0.0
    trade_returns: list[float] = []
    for index in range(len(predictions) - hold_bars - 1):
        if predictions[index] == 2:
            equity.append(equity[-1])
            continue
        if index + 1 + hold_bars >= len(prices):
            break
        entry, exit_ = prices[index + 1], prices[index + 1 + hold_bars]
        if entry == 0:
            continue
        pnl = float((exit_ - entry) / entry)
        if predictions[index] in (3, 4):
            pnl = -pnl
        trade_returns.append(pnl)
        equity.append(equity[-1] * (1 + pnl))
        if pnl > 0:
            gross_profit += pnl
            wins += 1
        else:
            gross_loss += abs(pnl)
            losses += 1
        peak = max(peak, equity[-1])
        max_drawdown = max(max_drawdown, (peak - equity[-1]) / peak)
    total = wins + losses
    return {
        "profit_factor": gross_profit / (gross_loss + 1e-9),
        "win_rate": wins / (total + 1e-9),
        "max_drawdown": max_drawdown,
        "total_trades": total,
        "equity_curve": equity,
        "trade_returns": trade_returns,
        "net_return": equity[-1] - 1 if equity else 0.0,
    }


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    missing = _missing_artifacts()
    if missing:
        raise RuntimeError(f"Missing trained artifacts: {', '.join(missing)}")
    import tensorflow as tf
    from quantum_layer import QuantumPreprocessor
    from trainer_v3 import TemporalAttentionGate

    custom_objects = {"TemporalAttentionGate": TemporalAttentionGate}
    # These artifacts are created by this local QuantGrad trainer in the
    # adjacent artifacts directory. The model architecture contains a Lambda
    # layer, so Keras 3 requires explicitly opting in to local deserialization.
    trend = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, MODEL_FILES["trend"]), custom_objects=custom_objects, compile=False, safe_mode=False)
    structure = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, MODEL_FILES["structure"]), custom_objects=custom_objects, compile=False, safe_mode=False)
    entry = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, MODEL_FILES["entry"]), custom_objects=custom_objects, compile=False, safe_mode=False)
    quantum = QuantumPreprocessor.load(ARTIFACTS_DIR)
    report_path = os.path.join(ARTIFACTS_DIR, "training_report.json")
    report: dict[str, Any] = {}
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    return trend, structure, entry, quantum, report


def _transform_windows_pca(quantum: Any, windows: np.ndarray) -> np.ndarray:
    """Same PCA transform as dashboard.py; this is intentionally not model logic."""
    samples, window, feature_count = windows.shape
    scaled = quantum.scaler.transform(windows.reshape(-1, feature_count))
    transformed = quantum.decomposer.transform(scaled)
    return transformed.reshape(samples, window, -1).astype(np.float32)


def _run_inference(trend_model: Any, structure_model: Any, entry_model: Any, quantum: Any, candle_frame: pd.DataFrame, macro: dict[str, Any], order_book: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    from feature_engine import FEATURE_DIM, WINDOW_SIZE, build_feature_matrix
    from trainer_v3 import fusion_decision

    features, _ = build_feature_matrix(candle_frame, macro, order_book)
    if len(features) < WINDOW_SIZE:
        raise ValueError(f"Only {len(features)} feature rows are available; {WINDOW_SIZE} are required")
    raw_window = features.values[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, FEATURE_DIM)
    transformed_window = _transform_windows_pca(quantum, raw_window)
    trend_probs = trend_model.predict(transformed_window, verbose=0)[0]
    structure_probs = structure_model.predict(transformed_window, verbose=0)[0]
    entry_probs = entry_model.predict([raw_window, trend_probs.reshape(1, -1), structure_probs.reshape(1, -1)], verbose=0)[0]
    adx_value = float(features["adx"].iloc[-1]) if "adx" in features.columns else 0.20
    result = fusion_decision(entry_probs, trend_probs, structure_probs, adx_value, order_book.get("spread", 0))
    result.update({"entry_probs": entry_probs, "trend_probs": trend_probs, "struct_probs": structure_probs})
    return result, features


def _run_rolling_inference(trend_model: Any, structure_model: Any, entry_model: Any, quantum: Any, candle_frame: pd.DataFrame, macro: dict[str, Any], order_book: dict[str, Any], windows: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    from feature_engine import WINDOW_SIZE, build_feature_matrix, build_windows

    features, _ = build_feature_matrix(candle_frame, macro, order_book)
    if len(features) < WINDOW_SIZE + 10:
        return np.array([]), np.empty((0, 5)), features
    neutral_labels = np.full(len(features), 2, dtype=np.int32)
    raw_windows, _ = build_windows(features.values, neutral_labels)
    raw_windows = raw_windows[-windows:]
    transformed_windows = _transform_windows_pca(quantum, raw_windows)
    trend_probs = trend_model.predict(transformed_windows, verbose=0, batch_size=512)
    structure_probs = structure_model.predict(transformed_windows, verbose=0, batch_size=512)
    entry_probs = entry_model.predict([raw_windows, trend_probs, structure_probs], verbose=0, batch_size=512)
    return np.argmax(entry_probs, axis=1), entry_probs, features


@app.get("/api/status")
def status() -> dict[str, Any]:
    missing = _missing_artifacts()
    return {
        "status": "ready" if not missing else "artifacts_missing",
        "artifacts_ready": not missing,
        "missing": missing,
        "artifact_dir": ARTIFACTS_DIR,
        "api_time": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/analyze")
def analyze(
    symbol: str = Query("BTCUSDT", pattern="^[A-Z0-9]{5,16}$"),
    bars: int = Query(800, ge=300, le=2000),
    windows: int = Query(260, ge=80, le=500),
) -> dict[str, Any]:
    """Fetch market context and run the original feature → PCA → three-model → fusion pipeline."""
    if not re.fullmatch(r"[A-Z0-9]{5,16}", symbol):
        raise HTTPException(status_code=422, detail="Symbol must be an uppercase exchange pair, e.g. BTCUSDT")
    try:
        trend, structure, entry, quantum, report = _load_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        from feature_engine import fetch_binance_klines, fetch_order_book
        from macro_fetcher import fetch_all_macro, is_cache_stale, load_macro_cache

        candles = fetch_binance_klines(symbol, "1h", total_bars=bars)
        order_book = fetch_order_book(symbol)
        macro = fetch_all_macro(save=True) if is_cache_stale(24) else load_macro_cache()
        result, features = _run_inference(trend, structure, entry, quantum, candles, macro, order_book)
        predictions, rolling_probs, rolling_features = _run_rolling_inference(trend, structure, entry, quantum, candles, macro, order_book, windows)
        prices = candles["close"].values[-len(predictions) - 4:] if len(predictions) else np.array([])
        backtest = _backtest_stats(predictions, prices) if len(predictions) else _backtest_stats(np.array([]), np.array([]))
        return _native({
            "symbol": symbol,
            "interval": "1h",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "candles": _serialise_candles(candles),
            "order_book": order_book,
            "macro": macro,
            "result": result,
            "rolling": {"predictions": predictions, "probabilities": rolling_probs},
            "backtest": backtest,
            "feature_streams": _feature_streams(rolling_features if len(rolling_features) else features),
            "training_report": report,
            "artifact_status": {"ready": True, "missing": []},
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


if os.path.isdir(WEB_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")

    @app.get("/")
    def terminal() -> FileResponse:
        return FileResponse(os.path.join(WEB_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
