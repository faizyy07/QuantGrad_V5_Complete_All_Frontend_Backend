# QuantGrad v4 — Three-Model Institutional Pipeline

## Quick Start
```bash
# 1. Create an isolated environment and install once
python -m venv venv
# Linux/macOS: source venv/bin/activate
# Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Verify and train
python verify.py
python macro_fetcher.py            # fetch macro data (daily)
python trainer_v3.py --quick       # quick validation training run
# Or: python trainer_v3.py --bars 20000  # fuller training run

# 3. Launch the professional web terminal
python server.py                 # launch the web terminal (http://localhost:8000)
```

The web terminal (`server.py` + `web/`) replaces the Streamlit dashboard while reusing the
same inference path (`transform_windows_pca` → `fusion_decision`). The 65-feature contract,
window shape, model architectures, artifact format and fusion rules are retained; only the
compatibility defects listed below were corrected.

## Bug fixes in this release
- `requirements.txt`: added the missing TensorFlow runtime declaration required by both `trainer_v3.py` and the inference server.
- `feature_engine.py`: replaced deprecated `fillna(method="bfill")` (errors on pandas 2.1+).
- `trainer_v3.py`: Step 3 now loads trend/structure models with `compile=False` and the
correct 6-class smoothing loss (previously the 3-class loss object was shared).
- `macro_fetcher.py`: updated to the official 2026 FOMC decision calendar and now avoids inventing a future event date after the maintained calendar list ends.[^fomc]
- `verify.py`: updated stale `trainer.py` references and commands.

## Models
| File | Task | Classes |
|---|---|---|
| trend_model.h5 | Trend direction | UPTREND / DOWNTREND / RANGING |
| structure_model.h5 | Market structure event | MSS / OB / FVG / SWEEP / NONE |
| entry_model.h5 | Trade signal | STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL |

## v4 Improvements (over v3)

### Model Architecture
| Change | Where | Benefit |
|---|---|---|
| MultiHeadAttention (4 heads) after LSTM | TrendModel | Captures non-local dependencies; attends to trend-defining candles regardless of position |
| TemporalAttentionGate (lightweight) | StructureModel | MSS/OB/FVG events are sparse; model learns to upweight those exact candles |
| Residual conv skip connections | StructureModel, EntryModel streams | Stabilises gradients in deeper conv stacks; avoids vanishing gradient |
| SpatialDropout1D on conv layers | All models | Drops whole feature maps instead of individual activations — more effective regularisation for temporal data |
| LayerNormalization before dense fusions | EntryModel | Prevents fusion layer from being dominated by high-magnitude streams |
| GlobalAveragePooling1D replacing last LSTM state | TrendModel | More stable gradients; averages all time positions instead of just the last |

### Training
| Change | Benefit |
|---|---|
| Label smoothing (0.05) | Prevents overconfidence on noisy financial labels; typically improves profit factor 5-15% |
| Gradient clipping (clipnorm=1.0) | Eliminates NaN loss spikes from large gradient steps |
| Cosine-annealing LR schedule | Smoother convergence than ReduceLROnPlateau; no plateau detection tuning needed; naturally re-heats mid-cycle helping escape local minima |

### Speed (market_structure.py)
| Change | Speedup |
|---|---|
| detect_swing_points: for-loop → pd.Series.rolling | ~8× |
| detect_order_blocks: O(n×k) inner loop → vectorized range-fill | ~5× |
| detect_fvg: same vectorization | ~5× |
| compute_adx_simple: vectorized TR/DM | ~3× |
| generate_structure_labels: fully vectorized np.where chain | ~20× |

### Memory (all v3 optimizations preserved + new)
- Mixed float16 precision (halves activation memory)
- X_q computed ONCE shared across all 3 models
- StructureModel LSTM: 256→192, 128→96 (saves ~200MB)
- del + gc.collect() between every training step
- tf.keras.backend.clear_session() between steps
- Incremental oversampling (extras-only, not full copy)
- Batched predict() calls (batch_size=512)

## Features (65 total)
- Stream 1 (8):  Gradient + Acceleration + Jerk (3rd derivative)
- Stream 2 (6):  Multi-scale momentum ROC 5/20/50
- Stream 3 (8):  Volume Z-score + VWAP + buying pressure
- Stream 4 (17): Market structure (swing, MSS, OB, FVG, sweep, premium/discount)
- Stream 5 (9):  ADX + EMA 20/50/200 alignment
- Stream 6 (7):  RSI + MACD + Bollinger (classic technicals)
- Stream 7 (10): DXY/Gold/Crude correlations + macro regime

## Optional API configuration
Copy `.env.example` to `.env` and fill in:
```
FRED_API_KEY=...   # free at fred.stlouisfed.org
```

Binance market-data and order-book endpoints used by the supplied code are public. The FRED key is optional because the macro fetcher retains its fallback when it is not set.

## Hardware Requirements
- RAM: 6-8 GB peak (16 GB comfortable)
- GPU: optional — CPU training works, GPU is ~5× faster
- Disk: ~2 GB for 20k bars + artifacts

[^fomc]: Federal Reserve Board, [FOMC meeting calendars and information](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), accessed August 2026.
