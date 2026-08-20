# QuantGrad v4 Web Terminal — Implementation Report

## Scope and design boundary

The Streamlit interface has been replaced with a professional browser terminal. The existing **model logic has not been redesigned or recalibrated**: the 65-feature contract, 60-candle window, scaler/PCA transform, three-model dependency chain, fusion decision rules, model artifact names, and rolling strategy simulation remain compatible with the supplied codebase.

The new interface follows an **Observatory Ledger** visual system: a TradingView-style central price canvas, an analytical decision spine, a narrow execution ledger, restrained conviction/risk semantics, and model artifacts rendered as evidence rather than generic dashboard cards.

| Area | Implementation |
|---|---|
| Browser UI | React 19, TypeScript, Tailwind/CSS, `lightweight-charts`, responsive terminal layout |
| Local service | FastAPI, same-origin production serving, CORS for local frontend development |
| Live operation | `GET /api/analyze` performs the existing feature → PCA → model-chain → fusion flow |
| Startup | `python server.py`, then open `http://localhost:8000` |
| No-artifact behavior | Clear API `503` response and a labelled offline visual preview rather than a false live signal |

## Preserved inference pipeline

> **Macro data → Binance market/order-book data → 65 features across seven streams → scaler/PCA window transform → Trend model → Structure model → Entry model → `fusion_decision()` → terminal response.**

The adapter retains the dashboard's inference order. In particular, the Entry model continues to receive `[X_q, trend_probs, struct_probs]`, and the dashboard's `fusion_decision()` is called directly rather than replicated with altered thresholds.

| Stage | Existing module | What remains unchanged |
|---|---|---|
| Macro regime | `macro_fetcher.py` | DXY, Gold, Crude, Fed, Fear & Greed, BTC dominance inputs and cache contract |
| Feature engineering | `feature_engine.py` | 65 features, seven streams, dimensionality and `WINDOW_SIZE=60` |
| Structure/labels | `market_structure.py` | structure features plus three target-label families |
| Training/inference | `trainer_v3.py`, `quantum_layer.py` | three model architectures, PCA preprocessing and fusion rules |
| Artifact contract | `artifacts/` | `.h5` model files, scaler, PCA, quantum parameters and training report |

## Corrections made

| File | Correction | Effect |
|---|---|---|
| `feature_engine.py` | Replaced deprecated `fillna(method="bfill")` use with `.bfill()` | Restores pandas 2.1+ compatibility |
| `trainer_v3.py` | Corrected the structure-model loading loss to the six-class object | Prevents a class-count mismatch during the Step-3 reload path |
| `verify.py` | Replaced stale `trainer.py` references | Verifier commands now target `trainer_v3.py` |
| `macro_fetcher.py` | Added official 2026 FOMC decision dates; stops returning invented future dates | Eliminates the stale 2025 calendar and false event risk beyond the maintained list [^fomc] |
| `requirements.txt` | Added TensorFlow, FastAPI and Uvicorn declarations | Declares the existing ML runtime and new web-server dependencies |
| `setup_env.sh`, `setup_env.bat` | Updated post-install launch command | Starts the web terminal rather than Streamlit |

[^fomc]: Federal Reserve Board, [FOMC meeting calendars and information](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), accessed August 2026.

## Launch instructions

```bash
# Linux / macOS
bash setup_env.sh
source venv/bin/activate

# Train once if artifacts/ is not present or incomplete
python trainer_v3.py --quick

# Serve the professional web terminal and API
python server.py
```

On Windows, run `setup_env.bat`, activate the generated virtual environment if needed, then use the same Python commands. Browse to **http://localhost:8000**.

An optional `.env.example` is provided. A `FRED_API_KEY` can improve the policy-rate source; the existing macro fallback is retained when the key is absent.

## Validation performed

| Check | Result |
|---|---|
| TypeScript type check | Passed |
| Production frontend build | Passed |
| Visual verification | Passed; reviewed and refined ledger-spine composition, colour semantics and Observatory motif |
| Python syntax compilation | Passed for the service and corrected Python modules |
| FOMC helper runtime smoke test | Passed; current result identified 2026-09-16 as next official decision date |
| Unified server root route | Passed; returns the bundled terminal HTML |
| API failure path without artifacts | Passed; `/api/analyze` returns expected `503 Missing trained artifacts` |

### Validation limitation

The supplied workspace did **not** contain trained artifacts or the installed TensorFlow/Qiskit/scikit-learn runtime, so a real inference request cannot be executed in this environment. This is not masked: the terminal remains in a labelled offline visual-preview state until the six expected artifacts are present. After you train or copy the models into `artifacts/`, `GET /api/analyze` will load the existing models and run the unchanged pipeline.
