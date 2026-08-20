# QuantGrad v4 — Pipeline Analysis Notes

## Pipeline overview
1. `macro_fetcher.py` → fetches DXY/Gold/Crude (Yahoo), Fed rate (FRED), Fear&Greed (alternative.me), BTC dominance (CoinGecko), FOMC dates → saves `data/macro_cache.json`.
2. `feature_engine.py` → fetches Binance klines + orderbook; builds 65-feature matrix (7 streams); WINDOW_SIZE=60, FEATURE_DIM=65; `build_windows()` creates (n,60,65) arrays.
3. `market_structure.py` → swing points, MSS, HH/HL, order blocks, FVG, sweeps, premium/discount (17 features) + 3 label generators (trend 3-class, structure 6-class, entry 5-class) + compute_adx_simple.
4. `trainer_v3.py` → labels, QuantumPreprocessor (StandardScaler + PCA + optional VQC), trains 3 models sequentially (TrendModel, StructureModel, EntryModel), saves artifacts: `trend_model.h5`, `structure_model.h5`, `entry_model.h5`, `scaler.pkl`, `pca.pkl`, `quantum_params.pkl`, `training_report.json`.
5. `dashboard.py` (Streamlit) → inference via `fusion_decision()` → UI with tabs: Market, Models, Risk Lab, Macro, Eigenspace, Training.

## Inference contract (must be preserved in new API)
- `transform_windows_pca(qp, X_3d)`: scaler → PCA (NOT the quantum VQC transform; dashboard & trainer both use pca-only path).
- Entry model takes [X_q, trend_probs, struct_probs].
- `fusion_decision(entry_probs, trend_probs, struct_probs, adx, spread)` → {signal, signal_label, risk_level, trend, structure, adx, confidence}.

## Bugs / issues found
1. `feature_engine.py` line 218 & 275: `vwap.fillna(method="bfill")` — deprecated `method=` param in pandas ≥2.1 (will error). Fix: `.bfill()`.
2. `trainer_v3.py` lines 410-412: Step 3 reloads trend/structure models with `custom_objects={"loss_fn": make_label_smoothing_loss(3)}` — models saved with the smoothing-loss name `"loss_fn"` but structure model compiled with 6-class loss; using 3-class loss fn only matters for compiling gradients but compile=False would avoid it. Also loading models with custom loss may error since structure model's loss name also `loss_fn`. Not fatal if saved models can load, but safer: load with compile=False (dashboard already does this).
3. `verify.py`: references old `trainer.py` (docs/stale) — fix names.
4. `macro_fetcher.py` had its FOMC decision-date list stop in December 2025. Fixed with the official 2026 decision dates, and it now returns `next_fomc: "n/a"` rather than fabricating a future monetary-policy event after the maintained calendar ends.
5. `dashboard.py` line ~574: `st.cache_data.clear()` / `st.cache_resource.clear()` exist but fine.
6. `compute_backtest_stats` in dashboard uses `prices[i+1]` entry — logic preserved; do not touch ML logic, keep as-is.
7. `trainer_v3.py` report uses `datetime.utcnow()` — deprecation warning in Python 3.12+, harmless.
8. `setup_env.bat/sh` had no web-terminal launch command. Fixed to launch `python server.py` and open `http://localhost:8000`.
9. `_env` placeholder file — keep.
10. `quantum_layer.py` fallback `transform_single` when qiskit unavailable: `_fallback` returns cos(arccos(v)) = v multiplied by interactions — fine.

## New architecture decision
- Keep the **model architecture, feature definitions, artifact format, transform sequence, and fusion rules unchanged**. Only compatibility defects in the existing supporting code were corrected.
- New `server.py` is a thin FastAPI adapter that imports the existing modules and exposes:
  - `GET /api/status` — artifact readiness and expected artifact names.
  - `GET /api/analyze?symbol=BTCUSDT&bars=800&windows=260` — the original feature → scaler/PCA → Trend/Structure/Entry → `fusion_decision()` path, together with candles, order book, macro data, rolling predictions, feature-stream intensities and the existing rolling backtest calculation.
  - `GET /` — serves the bundled professional frontend.
- Frontend: standalone React/TypeScript terminal styled as an institutional research workstation, with TradingView's `lightweight-charts` candlestick canvas, a live API client and a clearly-labelled offline visual preview when the local inference service/artifacts are unavailable.
