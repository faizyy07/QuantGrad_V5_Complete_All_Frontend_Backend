# QuantGrad v4 — Runbook

This folder is the runnable application. It contains the original ML pipeline, the new FastAPI adapter, and the pre-built professional web terminal. You do **not** need Node.js to train or use the terminal.

## First run

| Platform | Setup command |
|---|---|
| Linux or macOS | `bash setup_env.sh` |
| Windows | Double-click `setup_env.bat` or run it from Command Prompt |

After setup, activate the environment if your shell is new. Run `python verify.py` to confirm the ML, quantum, API and data dependencies.

## Train, test and launch

```bash
# Optional but useful: refresh macro cache
python macro_fetcher.py

# Produce the six model/preprocessing artifacts needed for inference
python trainer_v3.py --quick

# Serve the web terminal and REST API
python server.py
```

Open **http://localhost:8000**. The terminal calls the same-origin `/api/analyze` endpoint, which runs the existing feature engineering, scaler/PCA preprocessing, three model predictions and fusion logic.

## Package map

| Folder or file | Purpose |
|---|---|
| `trainer_v3.py`, `feature_engine.py`, `market_structure.py`, `quantum_layer.py`, `macro_fetcher.py` | Existing ML/training pipeline |
| `server.py` | Local FastAPI terminal server and inference adapter |
| `web/` | Compiled browser application served by `server.py` |
| `artifacts/` | Created by training; holds models, scaler, PCA and training report |
| `frontend-source/` in the ZIP root | Editable React/TypeScript source for the interface |
| `IMPLEMENTATION_REPORT.md` | Pipeline analysis, changes and validation record |

> The downloaded ZIP intentionally excludes `.env` and trained `artifacts/` because no private keys or trained models were present in the supplied workspace. Create `.env` from `.env.example` only if you have a FRED key.
