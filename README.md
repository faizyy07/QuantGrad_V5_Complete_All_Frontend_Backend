# QuantGrad

**A research-oriented market analysis platform combining multi-stream feature engineering, neural sequence models, quantum-model experiments, and a browser-based research terminal.**

QuantGrad is an experimental software project for studying how technical, market-structure, order-book, and macroeconomic information can be brought together in a common market-analysis pipeline. It includes a Python training and inference service, a React/TypeScript interface, and a set of model-comparison and backtesting experiments.

The repository is intended to support ongoing research and reproducible technical review. It is **not** a trading service, financial product, or evidence that any model can generate reliable returns.

## Project at a glance

| Area | Implementation |
|---|---|
| Market data | Binance spot OHLCV candles and order-book snapshots |
| Feature engineering | 65 features organised into seven streams |
| Sequence models | TensorFlow/Keras trend, structure, and entry classifiers |
| Dimensionality reduction | Standard scaling and PCA |
| Quantum experiments | Saved statevector feature-map experiment plus a separate supervised 4-qubit Qiskit variational classifier |
| Classical comparisons | Random Forest, Histogram Gradient Boosting, and an out-of-fold tree stacker |
| Evaluation | Chronological holdout, confidence-threshold selection, transaction-cost-aware simulation, and expanding-window walk-forward experiments |
| Python service | FastAPI |
| Web application | React 19, TypeScript, Vite, Express, and tRPC |

## How the system is organised

The project has two cooperating services:

1. **Python model service** — retrieves market inputs, constructs the feature matrix, loads trained artifacts, runs model inference, and returns the analysis payload.
2. **Web application** — provides the research terminal and public-market context panels. Its server bridges requests to the local Python service when model analysis is requested.

```mermaid
flowchart TB
    A[Binance OHLCV candles] --> C[Feature engineering]
    B[Order-book snapshot] --> C
    D[Macro and sentiment sources] --> C
    C --> E[65 features · 7 streams]
    E --> F[60-step windows]

    subgraph LIVE["Live neural inference path"]
      F --> G[StandardScaler + PCA]
      G --> H[Trend CNN–LSTM–attention model]
      G --> I[Structure CNN–LSTM–temporal attention model]
      F --> J[Entry model: seven feature-stream branches]
      H --> K[Trend probabilities · 3 classes]
      I --> L[Structure probabilities · 6 classes]
      K --> J
      L --> J
      J --> M[Entry probabilities · 5 classes]
      K --> N[Signal and risk fusion]
      L --> N
      M --> N
      O[ADX and order-book spread] --> N
      N --> P[Final signal + risk label]
    end

    subgraph EXP["Offline model-comparison experiments"]
      Q[Last-step feature snapshots] --> R[Random Forest]
      Q --> S[Histogram Gradient Boosting]
      R --> T[Probability features]
      S --> T
      T --> U[Logistic-regression OOF stacker]
      Q --> V[Separate 4-qubit Qiskit VQC]
      M --> W[Neural ensemble comparator]
      R --> X[Holdout strategy evaluation]
      S --> X
      U --> X
      V --> X
      W --> X
      X --> Y[Classification + cost-aware backtest metrics]
      R --> Z[Expanding-window walk-forward]
      S --> Z
      U --> Z
    end

    subgraph QEXP["Quantum preprocessing experiment"]
      G -. separate saved feature-map parameters .-> AA[Qiskit statevector feature-map experiment]
    end
```

**How to read this architecture:** the live inference path is a three-network neural ensemble, not a single model. Trend and structure predictions are generated independently from the PCA-transformed sequence. The Entry model receives the original 65-feature sequence plus both probability vectors, then the fusion function combines all three outputs with ADX and spread information.

The lower branch is a separate offline comparison suite. It evaluates the neural ensemble against a supervised four-qubit Qiskit variational classifier, Random Forest, Histogram Gradient Boosting, and a logistic-regression stacker trained on expanding-window out-of-fold probabilities. The walk-forward routine is applied to the classical baselines and stacker.

One implementation detail is important for an accurate research description: although `QuantumPreprocessor` fits and saves a Qiskit feature-map component, the current trainer and live API pass **scaled PCA outputs** to the Trend and Structure networks. They do not pass the feature-map transform output into those networks. The separate supervised `QuantumMarketClassifier` is the quantum model used in the model-comparison experiment.

The diagram describes the intended data flow. The separate quantum-classifier and classical-baseline experiments are part of the training/evaluation workflow; they are not all sequential stages in the live inference path.

## Repository layout

The runnable project is maintained in `QuantGrad_V5_Complete/`. The older top-level `backend/artifacts/` directory is a separate artifact snapshot; the supplied launch scripts and Python modules use `QuantGrad_V5_Complete/backend/artifacts/`.

```text
.
├── README.md
├── QuantGrad_V5_Complete/
│   ├── backend/
│   │   ├── artifacts/          # Trained models and preprocessing artifacts
│   │   ├── data/               # Macro-data cache
│   │   ├── feature_engine.py   # Market inputs and 65-feature construction
│   │   ├── market_structure.py # Structure features and labels
│   │   ├── macro_fetcher.py    # External macro-data collection
│   │   ├── quantum_layer.py    # PCA and Qiskit-based experiments
│   │   ├── trainer_v3.py       # Model training and evaluation orchestration
│   │   ├── strategy_experiments.py
│   │   ├── server.py           # FastAPI model API
│   │   └── test_*.py           # Backend tests
│   ├── frontend/
│   │   ├── client/src/         # React application
│   │   ├── server/             # Express, tRPC, public-data and model bridge
│   │   ├── shared/             # Shared TypeScript types and model schemas
│   │   ├── package.json
│   │   └── pnpm-lock.yaml
│   ├── setup_windows.bat
│   ├── run_backend.bat
│   ├── run_frontend.bat
│   └── train_quick.bat
└── backend/
    └── artifacts/              # Legacy/duplicate artifact snapshot
```

## Feature design

The Python pipeline builds a 65-column feature matrix from seven groups. The feature contract is shared by training and inference.

| Stream | Features | Examples |
|---|---:|---|
| Gradient and price dynamics | 8 | Gradient, acceleration, jerk, persistence |
| Multi-scale momentum | 6 | ROC at multiple horizons, momentum alignment, divergence |
| Volume and order flow | 8 | Volume z-score, VWAP deviation, taker-buy ratio, order-book measures |
| Market structure | 17 | Swing points, market-structure shifts, order blocks, fair-value gaps, liquidity sweeps |
| Trend strength | 9 | ADX, directional indicators, EMA alignment, VWAP position |
| Technical indicators | 7 | RSI, MACD, Bollinger features, log return |
| Macro context | 10 | DXY, gold and crude relationships, macro momentum, sentiment and policy inputs |
| **Total** | **65** | |

The model window is 60 time steps. The trend task uses three classes, the market-structure task uses six classes, and the entry task uses five signal classes.

## Models and experiments

### Live neural ensemble

The production inference route loads and runs **three distinct TensorFlow/Keras networks**. They are trained as separate models, then connected through their probability outputs.

| Model | Input | Architecture in the implementation | Output |
|---|---|---|---|
| **TrendModel** | 60-step PCA-transformed feature sequence | Causal Conv1D blocks, stacked LSTMs, four-head self-attention, residual normalisation and dense classification head | 3 classes: uptrend, downtrend, ranging |
| **StructureModel** | Same PCA-transformed sequence | Causal convolutional residual blocks, stacked LSTMs, temporal attention gate and dense classification head | 6 classes: bullish/bearish MSS, order block, fair-value gap, liquidity sweep, none |
| **EntryModel** | Original 65-feature sequence plus TrendModel and StructureModel probability vectors | Seven parallel feature-stream branches; each branch uses Conv1D, a residual projection and an LSTM. The trend and structure probabilities form an additional context branch | 5 classes: strong buy, buy, hold, sell, strong sell |

The seven EntryModel branches correspond to gradient, momentum, volume, structure, trend, technical and macro features. The EntryModel does not simply consume the two predicted labels: it receives their full probability vectors and learns from them alongside the raw feature sequence.

The final `fusion_decision` function combines the Entry, Trend and Structure probabilities with ADX and order-book spread to produce the displayed signal and risk label. This final rule-based fusion step is distinct from the neural networks themselves.

### Separate model-comparison suite

The trainer also builds an offline comparison suite. These models are **not** additional stages in the live three-network inference chain.

| Comparator | Model family | Role in the experiment |
|---|---|---|
| **QuantumMarketClassifier** | Supervised 4-qubit variational quantum classifier using Qiskit Statevector | Three-class directional prediction |
| **Random Forest** | Bagged decision trees | Classical tabular baseline |
| **Histogram Gradient Boosting** | Gradient-boosted decision trees | Classical tabular baseline |
| **Tree stacker** | Logistic regression | Learns from expanding-window out-of-fold probability outputs of Random Forest and Histogram Gradient Boosting |
| **Neural ensemble** | TrendModel + StructureModel + EntryModel | Reference model for comparison |

The classical models and stacker are trained on aligned tabular feature snapshots. The stacker is trained on out-of-fold probability features; it is not a fourth neural network and it does not replace the EntryModel in the live API. The comparison routine selects confidence thresholds on validation data, then evaluates the selected signals on the chronological holdout. The classical baselines and stacker are also included in the expanding-window walk-forward routine.

### Quantum components: an important distinction

There are two different quantum-related components in the code:

1. **QuantumPreprocessor / QuantumFeatureMap** — fits a scaler and PCA decomposition, then fits and saves a Qiskit statevector feature-map component when the local simulator is available and within the configured qubit limit. In the current trainer and API inference path, the TrendModel and StructureModel inputs are produced using the scaler and PCA decomposition; the feature-map output is not used as their model input.
2. **QuantumMarketClassifier** — a separate supervised four-qubit variational classifier trained and evaluated by the offline comparison suite. This is the Qiskit model directly compared with the neural ensemble and classical baselines.

The Qiskit routines use local statevector simulation, not quantum hardware. The feature-map code has a deterministic classical fallback when Qiskit is unavailable or the configured statevector limit is exceeded. No quantum advantage is established by the presence of these components.

### Evaluation

The training and strategy code includes:

- Chronological train, validation, and test partitions.
- A purge interval around partition boundaries for the configured maximum holding period.
- Validation-based confidence-threshold selection.
- Classification measures such as accuracy, macro precision/recall/F1, balanced accuracy, and log loss.
- Simulated strategy measures including net return, profit factor, win rate, drawdown, holding time, fees, and slippage.
- Expanding-window walk-forward evaluation for the classical baselines and stacker.

These are software evaluation procedures, not guarantees of out-of-sample performance. Reported values depend on the selected symbol, date range, data availability, labels, execution assumptions, and saved training artifacts. Results should be independently reproduced before being used in a paper.

## Web application

The browser application is organised around a set of research and market-context views:

- **Terminal** — local model status, signal details, and model evidence.
- **Discover** — public market movers and attributed news headlines.
- **Charts** — exchange candles with candle, line, and Heikin–Ashi views.
- **Strategy Lab** — a rule-specification workspace; it does not execute or backtest the displayed draft.
- **Derivatives** — exchange-specific funding, open interest, and long/short account ratios.
- **Macro** — an event calendar and market-news context.
- **Flow** — public Hyperliquid market context and optional public-wallet position lookup.
- **Integrations** — an optional Zapier Catch Hook test.

The frontend obtains public market data from Binance and Hyperliquid, and headlines from a Google News RSS search feed. The Python pipeline separately uses Binance and macro-data providers including Yahoo Finance, FRED (optional API key), Alternative.me, and CoinGecko. Provider availability and fallback values can affect the resulting feature inputs.

## Getting started

### Requirements

- Python **3.12** is the recommended version for the pinned TensorFlow/Python dependency set.
- Node.js and npm.
- pnpm **10.4.1** (the Windows launcher invokes this version through `npx`).
- Internet access for public market and macro-data requests.
- Sufficient memory and disk space for TensorFlow, model artifacts, and training. GPU acceleration is optional; the exact supported GPU setup depends on the local TensorFlow environment.

### Windows

Open PowerShell in the repository and move into the application directory:

```powershell
cd .\QuantGrad_V5_Complete
.\setup_windows.bat
```

Start the Python service in the first terminal:

```powershell
.\run_backend.bat
```

Start the web application in a second terminal:

```powershell
.\run_frontend.bat
```

Then open [http://localhost:3000](http://localhost:3000).

The Python service is available locally at [http://127.0.0.1:8000](http://127.0.0.1:8000). Its readiness endpoint is [http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status).

### macOS and Linux

From `QuantGrad_V5_Complete/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

In one terminal, start the model API:

```bash
cd backend
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

In another terminal, install and start the web application:

```bash
cd QuantGrad_V5_Complete/frontend
corepack pnpm@10.4.1 install --frozen-lockfile
corepack pnpm@10.4.1 dev
```

Open [http://localhost:3000](http://localhost:3000). If your local Corepack setup does not provide the pinned pnpm version, install pnpm 10.4.1 using your preferred Node.js package-manager workflow.

### Training

The repository currently includes model and preprocessing artifacts under `QuantGrad_V5_Complete/backend/artifacts/`. A new training run is not normally required just to start the application.

To run the small training configuration on Windows:

```powershell
cd .\QuantGrad_V5_Complete
.\train_quick.bat
```

The launcher uses 1,500 hourly BTCUSDT bars and the trainer's quick configuration. Treat this as a pipeline smoke/quick-training run, not as a research-grade experiment or a basis for financial decisions.

For a fuller run, use the trainer's command-line options from the backend directory:

```bash
python trainer_v3.py --bars 20000
```

Training fetches external data and writes model/preprocessing outputs to `backend/artifacts/`. Record the symbol, interval, date range, data snapshot, parameters, dependency versions, and resulting artifact hashes when preparing publishable experiments.

## API

The Python FastAPI service exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/status` | Checks whether the expected model and preprocessing artifacts are present |
| GET | `/api/analyze` | Fetches market context and runs the local inference pipeline |

The analysis endpoint accepts:

- `symbol` — uppercase exchange symbol; defaults to `BTCUSDT`.
- `bars` — number of hourly candles to request; defaults to 800.
- `windows` — number of rolling windows used for the rolling evaluation payload; defaults to 260.

Example:

```text
http://127.0.0.1:8000/api/analyze?symbol=BTCUSDT&bars=800&windows=260
```

The Node/tRPC application exposes the frontend's public-data and local-model bridge procedures. It is the browser application's primary entry point.

## Tests and checks

Run backend tests from `QuantGrad_V5_Complete/backend/`:

```bash
python -m unittest discover -s . -p "test_*.py" -v
python verify.py
```

Run frontend checks from `QuantGrad_V5_Complete/frontend/`:

```bash
pnpm check
pnpm test
pnpm build
```

The repository contains focused backend tests and frontend tests for selected model-bridge, presentation, and public-data behaviour. Passing these checks does not validate model performance or establish that a backtest is free of methodological bias.

## Research and reproducibility notes

QuantGrad is being developed as a research project. Before treating an experiment as a paper result, document at least:

1. **Research question and hypothesis** — define the claimed contribution before selecting metrics.
2. **Data provenance** — provider, symbol, interval, collection timestamps, missing-data treatment, and any revisions.
3. **Label construction** — thresholds, horizons, barrier definitions, and class balance.
4. **Temporal protocol** — train/validation/test dates, purge or embargo rules, and walk-forward fold boundaries.
5. **Baselines and ablations** — compare against appropriate non-quantum and simpler neural baselines under the same data and evaluation protocol.
6. **Execution assumptions** — fees, slippage, fill timing, spread, position sizing, and treatment of overlapping trades.
7. **Uncertainty** — repeated seeds where feasible, confidence intervals, sensitivity analysis, and statistical tests chosen before interpreting results.
8. **Artifacts** — dependency lockfiles, configuration, raw-data identifiers or snapshots where licensing permits, model hashes, and complete result files.

In particular, a fair quantum-versus-classical comparison requires matched data splits, comparable tuning budgets, transparent simulator settings, and a clear statement of what is quantum and what remains classical.

## Limitations and responsible use

- QuantGrad is a research prototype. It is not financial advice and does not place orders.
- Market and macro feeds are external dependencies; outages, rate limits, revised data, and fallback values can change the inputs.
- The live Python inference route expects compatible trained artifacts. The presence of files is a readiness check, not proof that those artifacts are scientifically valid or compatible with every environment.
- The current code includes local statevector simulation and a classical fallback path; it does not use a quantum processor.
- Backtests are simulations under code-defined assumptions. They do not model every aspect of market impact, liquidity, exchange outages, or real execution.
- The repository does not currently include a root-level license file. Reuse and redistribution terms should be clarified before external adoption.

## Contributing

Changes to the model, feature definitions, labels, evaluation protocol, or artifact format should be accompanied by tests and a description of their effect on reproducibility. Keep experimental outputs separate from source code, avoid committing credentials, and do not present unverified performance claims as established results.

## Citation

A formal citation should be added when the associated paper has a stable title, author list, version, and public DOI or repository release. Until then, cite the specific repository revision used in an experiment and include its commit hash in the paper's methods or supplementary material.

---

*QuantGrad — research software for studying multi-source market representation, model comparison, and evaluation.*
