# QuantGrad V5 — Complete Local Bundle

This archive contains the original Python ML backend in `backend/`, the V5 React/tRPC source in `frontend/`, and a prebuilt browser bundle in `web/`. **For the full V5 platform and the live decision ledger, run both the Python and Node services, then open `http://localhost:3000`.**

## Windows PowerShell — first-time setup

Open the folder in VS Code. The simplest first-time setup is to double-click `setup_windows.bat` or run this in the integrated PowerShell terminal. It creates `.venv` and installs the backend dependencies using the correct virtual-environment Python executable.

```powershell
cd "C:\Users\FAIZ\Downloads\QuantGrad_V5_Complete_Local_Run\QuantGrad_V5_Complete"
.\setup_windows.bat
```

The equivalent manual commands are below. Replace the path only if you extracted the archive elsewhere.

```powershell
cd "C:\Users\FAIZ\Downloads\QuantGrad_V5_Complete_Local_Run\QuantGrad_V5_Complete"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\backend\requirements.txt
```

If PowerShell blocks activation, run this once in the same VS Code terminal and repeat the activation command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Windows PowerShell — start QuantGrad

Keep **two VS Code terminals** open. Start the Python model API in the first terminal. It must remain running.

```powershell
cd "C:\Users\FAIZ\Downloads\QuantGrad_V5_Complete_Local_Run\QuantGrad_V5_Complete"
.\.venv\Scripts\Activate.ps1
.\run_backend.bat
```

In the second terminal, start the V5 Node service. This release uses a cross-platform command, so `pnpm dev` now works in PowerShell without the prior `NODE_ENV` error.

```powershell
cd "C:\Users\FAIZ\Downloads\QuantGrad_V5_Complete_Local_Run\QuantGrad_V5_Complete\frontend"
npx pnpm@10.4.1 install --frozen-lockfile
npx pnpm@10.4.1 dev
```

> **Corepack note:** The `EPERM` error from `corepack enable` is a Windows installation-permission issue outside this project. Do not run `corepack enable` for this bundle. The supported no-admin workaround is `npx pnpm@10.4.1`, as shown above. You can alternatively run ` .\run_frontend.bat` from the bundle root; it performs the same dependency check and starts the Node service.

Open **`http://localhost:3000`** in your browser. The Node V5 server supplies the public-data panels and securely bridges the decision ledger to the local Python model API at `http://127.0.0.1:8000`.

## Trained-model artifacts

No retraining is required when you already have compatible trained artifacts. Copy your existing private artifacts into `backend/artifacts/` before starting the Python API. The folder must contain the model and preprocessor filenames expected by `backend/server.py`.

When artifacts are present, the Decision Ledger displays the real `signal_label`, `confidence`, `risk_level`, `trend`, `structure`, and `adx` returned by `/api/analyze`. If the Python API is stopped or artifacts are missing, the V5 dashboard stays visible and shows the reason safely rather than vanishing.

## Create model artifacts automatically

The bundle now includes an empty `backend\artifacts\` folder. A completed training run creates and saves the files the API needs automatically in that folder: `trend_model.keras`, `structure_model.keras`, `entry_model.keras`, `scaler.pkl`, `pca.pkl`, `quantum_params.pkl`, and `training_report.json`.

For a small functional training run, first run `setup_windows.bat`, then run:

```powershell
.\train_quick.bat
```

It trains using 1,500 BTCUSDT hourly bars and the trainer's five-epoch quick setting. It is intended only to verify the complete local pipeline; it is **not** a basis for financial decisions. After the command completes, run `run_backend.bat`, then `run_frontend.bat`.

The Python API readiness endpoint is `http://127.0.0.1:8000/api/status`. A `"status":"ready"` response means the artifact filenames were detected. The V5 UI is served at `http://localhost:3000`; `http://127.0.0.1:8000` is the Python API and static fallback, not the full tRPC-powered V5 development service.

## macOS/Linux

```bash
cd QuantGrad_V5_Complete
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
bash run_backend.sh
```

In a second terminal, run `cd frontend && pnpm install --frozen-lockfile && pnpm dev`, then open `http://localhost:3000`.

## Trained-model artifacts

For the original ML inference endpoint (`/api/analyze`) to return trained-model output, copy your existing private `artifacts/` directory into `backend/artifacts/`. It must contain the model and preprocessor files named in `backend/server.py`. The public-data V5 interface and `/api/status` work without those artifacts; `/api/analyze` will safely report that they are missing until supplied.

## Bundle structure

| Directory | Contents |
|---|---|
| `backend/` | Original FastAPI server, feature engineering, macro fetcher, quantum layer, trainer, tests, and Python requirements. |
| `frontend/` | Final V5 React/tRPC source, public-data service, routes, tests, and production configuration. |
| `web/` | Built browser files kept as a Python-served static fallback. |
| `setup_windows.bat` | Windows setup helper that creates `.venv` and installs the Python requirements. |
| `run_backend.sh` / `run_backend.bat` | Local Python API startup scripts for macOS/Linux and Windows. |
| `run_frontend.bat` | Windows V5 Node launcher using the no-Corepack `npx pnpm@10.4.1` workaround. |
| `train_quick.bat` | Windows quick-training launcher that saves the model artifacts into `backend\artifacts\`. |
