"""
QuantGrad v4 dashboard.

Professional Streamlit terminal for the three-model pipeline:
Trend + Structure + Entry.
"""

import json
import os
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

APP_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(APP_DIR, "artifacts")

SIGNAL_LABELS = {0: "STRONG BUY", 1: "BUY", 2: "HOLD", 3: "SELL", 4: "STRONG SELL"}
SIGNAL_COLORS = {0: "#16c784", 1: "#2dd4bf", 2: "#94a3b8", 3: "#fb7185", 4: "#ef4444"}
TREND_COLORS = {"UPTREND": "#16c784", "DOWNTREND": "#ef4444", "RANGING": "#f59e0b"}
RISK_COLORS = {
    "CONFIRMED": "#16c784",
    "STANDARD": "#60a5fa",
    "HIGH_RISK": "#f59e0b",
    "NO_TRADE": "#94a3b8",
}
ENTRY_LABELS = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
TREND_LABELS = ["Uptrend", "Downtrend", "Ranging"]
STRUCT_LABELS = ["MSS Bull", "MSS Bear", "Order Block", "FVG", "Sweep", "None"]

st.set_page_config(
    page_title="QuantGrad v4 Terminal",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        :root {
          --bg: #080b12;
          --panel: #0f1623;
          --panel-2: #121b2a;
          --line: rgba(148, 163, 184, .18);
          --muted: #94a3b8;
          --text: #e5edf8;
        }
        .stApp {
          background:
            radial-gradient(circle at 20% 0%, rgba(22,199,132,.10), transparent 28%),
            radial-gradient(circle at 80% 0%, rgba(96,165,250,.10), transparent 30%),
            var(--bg);
          color: var(--text);
        }
        .block-container { padding: 1.2rem 1.5rem 2.5rem; max-width: 1680px; }
        section[data-testid="stSidebar"] { background: #0a101b; border-right: 1px solid var(--line); }
        div[data-testid="stMetric"] {
          background: linear-gradient(180deg, rgba(18,27,42,.96), rgba(12,18,29,.96));
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .85rem .95rem;
        }
        div[data-testid="stMetric"] label { color: var(--muted) !important; }
        .qg-topbar {
          border: 1px solid var(--line);
          background: linear-gradient(135deg, rgba(15,22,35,.98), rgba(10,16,27,.98));
          border-radius: 8px;
          padding: 1rem 1.1rem;
          margin-bottom: 1rem;
        }
        .qg-title { font-size: 1.65rem; font-weight: 760; letter-spacing: 0; margin: 0; }
        .qg-subtitle { color: var(--muted); margin-top: .2rem; font-size: .92rem; }
        .qg-card {
          background: linear-gradient(180deg, rgba(18,27,42,.97), rgba(11,17,28,.97));
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 1rem;
          height: 100%;
        }
        .qg-card-title {
          color: var(--muted);
          font-size: .78rem;
          font-weight: 700;
          letter-spacing: .08em;
          text-transform: uppercase;
          margin-bottom: .45rem;
        }
        .signal-card {
          border-radius: 8px;
          border: 1px solid var(--line);
          padding: 1.15rem;
          background: linear-gradient(145deg, rgba(12,18,29,.98), rgba(18,27,42,.98));
          min-height: 188px;
        }
        .signal-label { font-size: 2rem; line-height: 1.05; font-weight: 820; margin: .25rem 0; }
        .signal-meta { color: var(--muted); font-size: .86rem; }
        .pill-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .8rem; }
        .pill {
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: .28rem .62rem;
          color: #cbd5e1;
          background: rgba(148,163,184,.07);
          font-size: .78rem;
        }
        .section-title { font-size: 1rem; font-weight: 720; margin: 1.1rem 0 .55rem; }
        .status-dot {
          display: inline-block; width: .55rem; height: .55rem; border-radius: 50%;
          margin-right: .35rem; vertical-align: middle;
        }
        .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
        .stTabs [data-baseweb="tab"] {
          border-radius: 8px;
          border: 1px solid var(--line);
          background: rgba(15,22,35,.82);
          padding: .45rem .8rem;
        }
        .stTabs [aria-selected="true"] { background: rgba(37,99,235,.22); }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def apply_plot_theme(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,11,18,.65)",
        font=dict(color="#cbd5e1", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=20, r=20, t=42, b=24),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,.10)",
        zerolinecolor="rgba(148,163,184,.12)",
        rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.12)")
    if height:
        fig.update_layout(height=height)
    return fig


@st.cache_resource(show_spinner="Loading QuantGrad artifacts...")
def load_artifacts(artifact_dir):
    try:
        paths = {
            "trend": os.path.join(artifact_dir, "trend_model.h5"),
            "structure": os.path.join(artifact_dir, "structure_model.h5"),
            "entry": os.path.join(artifact_dir, "entry_model.h5"),
            "scaler": os.path.join(artifact_dir, "scaler.pkl"),
            "pca": os.path.join(artifact_dir, "pca.pkl"),
            "quantum": os.path.join(artifact_dir, "quantum_params.pkl"),
        }
        missing = [name for name, path in paths.items() if not os.path.exists(path)]
        if missing:
            return None, None, None, None, {}, missing

        import tensorflow as tf
        from quantum_layer import QuantumPreprocessor
        from trainer_v3 import TemporalAttentionGate, make_label_smoothing_loss

        custom_objects = {
            "TemporalAttentionGate": TemporalAttentionGate,
            "loss_fn": make_label_smoothing_loss(5),
        }
        trend_model = tf.keras.models.load_model(paths["trend"], custom_objects=custom_objects, compile=False)
        struct_model = tf.keras.models.load_model(paths["structure"], custom_objects=custom_objects, compile=False)
        entry_model = tf.keras.models.load_model(paths["entry"], custom_objects=custom_objects, compile=False)
        qp = QuantumPreprocessor.load(artifact_dir)

        report_path = os.path.join(artifact_dir, "training_report.json")
        report = {}
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)

        return trend_model, struct_model, entry_model, qp, report, []
    except Exception as exc:
        return None, None, None, None, {"load_error": str(exc)}, ["runtime error"]


@st.cache_data(ttl=3600, show_spinner="Fetching macro regime...")
def get_macro(symbol):
    from macro_fetcher import fetch_all_macro, is_cache_stale, load_macro_cache

    return fetch_all_macro(save=True) if is_cache_stale(24) else load_macro_cache()


@st.cache_data(ttl=300, show_spinner="Fetching exchange candles...")
def get_candles(symbol, bars):
    from feature_engine import fetch_binance_klines

    return fetch_binance_klines(symbol, "1h", total_bars=bars)


@st.cache_data(ttl=60, show_spinner="Fetching order book...")
def get_order_book(symbol):
    from feature_engine import fetch_order_book

    return fetch_order_book(symbol)


def transform_windows_pca(qp, X_3d):
    n, w, f = X_3d.shape
    scaled = qp.scaler.transform(X_3d.reshape(-1, f))
    pca = qp.decomposer.transform(scaled)
    return pca.reshape(n, w, -1).astype(np.float32)


def run_inference(trend_m, struct_m, entry_m, qp, df, macro, ob):
    from feature_engine import FEATURE_DIM, WINDOW_SIZE, build_feature_matrix
    from trainer_v3 import fusion_decision

    feat_df, struct_df = build_feature_matrix(df, macro, ob)
    if len(feat_df) < WINDOW_SIZE:
        return None, feat_df, struct_df

    X = feat_df.values[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, FEATURE_DIM)
    X_q = transform_windows_pca(qp, X)
    trend_probs = trend_m.predict(X_q, verbose=0)[0]
    struct_probs = struct_m.predict(X_q, verbose=0)[0]
    entry_probs = entry_m.predict(
        [X_q, trend_probs.reshape(1, -1), struct_probs.reshape(1, -1)],
        verbose=0,
    )[0]

    adx_value = float(feat_df["adx"].iloc[-1]) if "adx" in feat_df.columns else 0.20
    result = fusion_decision(entry_probs, trend_probs, struct_probs, adx_value, ob.get("spread", 0))
    result["entry_probs"] = entry_probs
    result["trend_probs"] = trend_probs
    result["struct_probs"] = struct_probs
    return result, feat_df, struct_df


def run_rolling_inference(trend_m, struct_m, entry_m, qp, df, macro, ob, max_windows=260):
    from feature_engine import WINDOW_SIZE, build_feature_matrix, build_windows

    feat_df, _ = build_feature_matrix(df, macro, ob)
    if len(feat_df) < WINDOW_SIZE + 10:
        return np.array([]), np.empty((0, 5)), feat_df

    labels = np.full(len(feat_df), 2, dtype=np.int32)
    X_raw, _ = build_windows(feat_df.values, labels)
    X_raw = X_raw[-max_windows:]
    X_q = transform_windows_pca(qp, X_raw)
    trend_p = trend_m.predict(X_q, verbose=0, batch_size=512)
    struct_p = struct_m.predict(X_q, verbose=0, batch_size=512)
    entry_p = entry_m.predict([X_q, trend_p, struct_p], verbose=0, batch_size=512)
    preds = np.argmax(entry_p, axis=1)
    return preds, entry_p, feat_df


def compute_backtest_stats(preds, prices, hold_bars=3):
    equity = [1.0]
    wins, losses, gp, gl = 0, 0, 0.0, 0.0
    peak, max_dd = 1.0, 0.0
    trade_returns = []

    for i in range(len(preds) - hold_bars - 1):
        if preds[i] == 2:
            equity.append(equity[-1])
            continue
        if i + 1 + hold_bars >= len(prices):
            break
        entry = prices[i + 1]
        exit_ = prices[i + 1 + hold_bars]
        if entry == 0:
            continue
        pnl = (exit_ - entry) / entry
        if preds[i] in (3, 4):
            pnl = -pnl
        trade_returns.append(pnl)
        equity.append(equity[-1] * (1 + pnl))
        if pnl > 0:
            gp += pnl
            wins += 1
        else:
            gl += abs(pnl)
            losses += 1
        peak = max(peak, equity[-1])
        max_dd = max(max_dd, (peak - equity[-1]) / peak)

    total = wins + losses
    eq = np.array(equity)
    return {
        "profit_factor": gp / (gl + 1e-9),
        "win_rate": wins / (total + 1e-9),
        "max_drawdown": max_dd,
        "total_trades": total,
        "equity_curve": equity,
        "trade_returns": trade_returns,
        "net_return": eq[-1] - 1 if len(eq) else 0,
    }


def topbar(symbol, bars):
    st.markdown(
        f"""
        <div class="qg-topbar">
          <div class="qg-title">QuantGrad v4 Terminal</div>
          <div class="qg-subtitle">Three-model market intelligence for {symbol} on 1h candles · {bars} bars loaded</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_panel(result, symbol, latest_price):
    sig = result["signal"]
    color = SIGNAL_COLORS.get(sig, "#94a3b8")
    risk = result["risk_level"]
    trend = result["trend"]
    st.markdown(
        f"""
        <div class="signal-card" style="border-color:{color};">
          <div class="qg-card-title">Live Signal</div>
          <div class="signal-label" style="color:{color};">{result['signal_label']}</div>
          <div class="signal-meta">{symbol} · ${latest_price:,.2f} · confidence {result['confidence']*100:.1f}%</div>
          <div class="pill-row">
            <span class="pill" style="border-color:{RISK_COLORS.get(risk, '#94a3b8')};">{risk}</span>
            <span class="pill" style="border-color:{TREND_COLORS.get(trend, '#94a3b8')};">{trend}</span>
            <span class="pill">{result['structure']}</span>
            <span class="pill">ADX {result['adx']:.1f}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_market_chart(df, preds=None, feat_df=None):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.62, 0.18, 0.20],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )
    candle_df = df.tail(320)
    fig.add_trace(
        go.Candlestick(
            x=candle_df.index,
            open=candle_df["open"],
            high=candle_df["high"],
            low=candle_df["low"],
            close=candle_df["close"],
            increasing_line_color="#16c784",
            decreasing_line_color="#ef4444",
            name="Price",
        ),
        row=1,
        col=1,
    )
    close = candle_df["close"]
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    fig.add_trace(go.Scatter(x=candle_df.index, y=ema20, mode="lines", line=dict(color="#60a5fa", width=1.2), name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=candle_df.index, y=ema50, mode="lines", line=dict(color="#f59e0b", width=1.2), name="EMA 50"), row=1, col=1)

    if preds is not None and len(preds):
        signal_times = df.index[-len(preds):]
        signal_prices = df["close"].values[-len(preds):]
        for cls, marker in [(0, "triangle-up"), (1, "triangle-up"), (3, "triangle-down"), (4, "triangle-down")]:
            idx = np.where(preds == cls)[0]
            if len(idx):
                fig.add_trace(
                    go.Scatter(
                        x=signal_times[idx],
                        y=signal_prices[idx],
                        mode="markers",
                        marker=dict(symbol=marker, color=SIGNAL_COLORS[cls], size=11 if cls in (0, 4) else 9, line=dict(width=1, color="#020617")),
                        name=SIGNAL_LABELS[cls],
                    ),
                    row=1,
                    col=1,
                )

    colors = np.where(candle_df["close"] >= candle_df["open"], "#16c784", "#ef4444")
    fig.add_trace(go.Bar(x=candle_df.index, y=candle_df["volume"], marker_color=colors, opacity=.58, name="Volume"), row=2, col=1)

    if feat_df is not None and "gradient" in feat_df:
        grad = feat_df["gradient"].reindex(candle_df.index).fillna(0)
        fig.add_trace(go.Scatter(x=candle_df.index, y=grad, mode="lines", fill="tozeroy", line=dict(color="#a78bfa", width=1), name="Gradient"), row=3, col=1)

    fig.update_layout(title="Market Structure, Execution Signals, and Flow", height=760)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Gradient", row=3, col=1)
    return apply_plot_theme(fig)


def plot_probability_distribution(result):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Entry Signal", "Trend State", "Structure Event"),
        horizontal_spacing=.08,
    )
    fig.add_trace(
        go.Bar(
            x=ENTRY_LABELS,
            y=result["entry_probs"],
            marker_color=[SIGNAL_COLORS[i] for i in range(5)],
            text=[f"{v*100:.1f}%" for v in result["entry_probs"]],
            textposition="outside",
            name="Entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=TREND_LABELS,
            y=result["trend_probs"],
            marker_color=["#16c784", "#ef4444", "#f59e0b"],
            text=[f"{v*100:.1f}%" for v in result["trend_probs"]],
            textposition="outside",
            name="Trend",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=STRUCT_LABELS,
            y=result["struct_probs"],
            marker_color=["#16c784", "#ef4444", "#a78bfa", "#f59e0b", "#60a5fa", "#94a3b8"],
            text=[f"{v*100:.1f}%" for v in result["struct_probs"]],
            textposition="outside",
            name="Structure",
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_layout(title="Model Probability Distributions", showlegend=False)
    return apply_plot_theme(fig, height=365)


def plot_signal_mix(preds):
    counts = Counter(preds.tolist()) if len(preds) else {}
    values = [counts.get(i, 0) for i in range(5)]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=ENTRY_LABELS,
                values=values,
                hole=.62,
                marker=dict(colors=[SIGNAL_COLORS[i] for i in range(5)]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(title="Rolling Signal Distribution")
    return apply_plot_theme(fig, height=360)


def plot_equity(stats):
    eq = np.array(stats["equity_curve"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(eq))), y=eq, mode="lines", fill="tozeroy", line=dict(color="#16c784" if eq[-1] >= 1 else "#ef4444", width=2), name="Equity"))
    fig.add_hline(y=1, line_dash="dash", line_color="rgba(148,163,184,.45)")
    fig.update_layout(title="Rolling Strategy Equity Simulation", yaxis_tickformat=".2f")
    return apply_plot_theme(fig, height=360)


def plot_return_distribution(stats):
    returns = stats.get("trade_returns", [])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.array(returns) * 100, nbinsx=32, marker_color="#60a5fa", opacity=.82, name="Trade returns"))
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(title="Trade Return Distribution", xaxis_title="Return %", yaxis_title="Trades")
    return apply_plot_theme(fig, height=360)


def plot_feature_heatmap(feat_df):
    groups = {
        "Gradient": ["gradient", "acceleration", "jerk", "grad_norm"],
        "Momentum": ["roc5", "roc20", "roc50", "roc_spread"],
        "Volume": ["vol_delta", "vol_zscore", "vwap_dev", "buy_ratio"],
        "Structure": ["mss_bullish", "mss_bearish", "ob_strength", "fvg_dist", "sweep_recency"],
        "Trend": ["adx", "di_plus", "di_minus", "ema_spread"],
        "Macro": ["corr_dxy", "corr_gold", "corr_crude", "fear_greed", "btc_dom"],
    }
    rows, labels = [], []
    tail = feat_df.tail(96)
    for group, cols in groups.items():
        existing = [c for c in cols if c in tail]
        if not existing:
            continue
        rows.append(tail[existing].mean(axis=1).values)
        labels.append(group)
    fig = go.Figure(data=go.Heatmap(z=rows, x=tail.index, y=labels, colorscale="RdBu", zmid=0, colorbar=dict(title="Intensity")))
    fig.update_layout(title="Feature Stream Intensity Heatmap")
    return apply_plot_theme(fig, height=330)


def plot_macro_matrix(feat_df):
    cols = [c for c in ["corr_dxy", "corr_gold", "corr_crude"] if c in feat_df]
    if not cols:
        return None
    data = feat_df[cols].tail(120).rename(columns={"corr_dxy": "DXY", "corr_gold": "Gold", "corr_crude": "Crude"})
    fig = go.Figure(data=go.Heatmap(z=data.T.values, x=data.index, y=data.columns, colorscale="RdYlGn", zmin=-1, zmax=1, colorbar=dict(title="Corr")))
    fig.update_layout(title="Cross-Asset Correlation Regime")
    return apply_plot_theme(fig, height=280)


def plot_eigenspace(qp, report):
    summary = qp.get_eigenvalue_summary()
    if not summary and report:
        summary = report.get("quantum", {})
    ratios = summary.get("explained_variance_ratio", [])
    eigenvalues = summary.get("eigenvalues", [])
    if not ratios:
        return None
    x = list(range(1, len(ratios) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Eigenvalues", "Cumulative Explained Variance"))
    fig.add_trace(go.Bar(x=x, y=eigenvalues, marker_color="#60a5fa", name="Eigenvalue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=np.cumsum(ratios) * 100, mode="lines+markers", line=dict(color="#16c784", width=2), name="Cumulative"), row=1, col=2)
    fig.add_hline(y=95, line_dash="dash", line_color="#f59e0b", row=1, col=2)
    fig.update_layout(title="Eigenspace Monitor")
    return apply_plot_theme(fig, height=330)


def render_artifact_empty_state(missing, artifact_dir, report):
    st.markdown("### QuantGrad Terminal")
    st.warning("No complete trained artifact set was found. Train once, then this terminal will unlock live model inference.")
    st.code(
        "python macro_fetcher.py\npython trainer_v3.py --quick\nstreamlit run dashboard.py",
        language="bash",
    )
    st.write("Artifact folder:")
    st.code(artifact_dir, language="text")
    if missing:
        st.write("Missing:")
        st.write(", ".join(missing))
    if report.get("load_error"):
        st.error(report["load_error"])


with st.sidebar:
    st.markdown("### QuantGrad v4")
    st.caption("Institutional signal terminal")
    symbol = st.selectbox("Trading Pair", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    bars = st.slider("Candle History", 300, 2000, 800, 100)
    rolling_windows = st.slider("Rolling Inference Windows", 80, 500, 260, 20)
    artifact_dir_input = st.text_input("Artifacts Folder", value=ARTIFACTS_DIR)
    st.divider()
    use_overrides = st.checkbox("Manual Macro Override", value=False)
    dxy_override = st.slider("DXY", 90.0, 120.0, 103.0, 0.1, disabled=not use_overrides)
    fg_override = st.slider("Fear & Greed", 0, 100, 50, 1, disabled=not use_overrides)
    st.divider()
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)
    refresh = st.button("Clear Cache", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.cache_resource.clear()


topbar(symbol, bars)
trend_m, struct_m, entry_m, qp, report, missing = load_artifacts(artifact_dir_input)

if missing or trend_m is None:
    render_artifact_empty_state(missing, artifact_dir_input, report)
    st.stop()

if not run_button:
    st.info("Models are available. Configure the sidebar and run analysis.")

with st.spinner("Running market data, feature, and model pipeline..."):
    df = get_candles(symbol, bars)
    ob = get_order_book(symbol)
    macro = get_macro(symbol)
    if use_overrides:
        macro["dxy"]["current"] = dxy_override
        macro["fear_greed"]["current"] = fg_override

    result, feat_df, struct_df = run_inference(trend_m, struct_m, entry_m, qp, df, macro, ob)
    preds_roll, entry_roll, feat_roll = run_rolling_inference(
        trend_m, struct_m, entry_m, qp, df, macro, ob, max_windows=rolling_windows
    )

if result is None:
    st.error("Not enough data for the configured window.")
    st.stop()

latest_price = float(df["close"].iloc[-1])
prev_price = float(df["close"].iloc[-25]) if len(df) > 25 else float(df["close"].iloc[0])
change_24h = (latest_price - prev_price) / prev_price
macro_age = ""
if macro.get("fetched_at"):
    try:
        age_h = (datetime.utcnow() - datetime.fromisoformat(macro["fetched_at"])).total_seconds() / 3600
        macro_age = f"{age_h:.1f}h"
    except Exception:
        macro_age = "unknown"

left, right = st.columns([1.3, 3.2])
with left:
    signal_panel(result, symbol, latest_price)
with right:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Last Price", f"${latest_price:,.2f}", f"{change_24h:+.2%} 24h")
    m2.metric("Spread", f"${ob.get('spread', 0):.2f}", f"imb {ob.get('imbalance', 0):+.3f}")
    m3.metric("DXY", f"{macro['dxy'].get('current', 0):.2f}", f"{macro['dxy'].get('momentum_5d', 0):+.2f}%")
    m4.metric("Fear & Greed", f"{macro['fear_greed'].get('current', 50)}", macro["fear_greed"].get("label", "Neutral"))
    m5.metric("BTC Dominance", f"{macro['btc_dominance'].get('btc_dominance_pct', 0):.1f}%", f"macro age {macro_age}")

tabs = st.tabs(["Market", "Models", "Risk Lab", "Macro", "Eigenspace", "Training"])

with tabs[0]:
    st.plotly_chart(plot_market_chart(df, preds_roll, feat_df), use_container_width=True)

with tabs[1]:
    st.plotly_chart(plot_probability_distribution(result), use_container_width=True)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.plotly_chart(plot_signal_mix(preds_roll), use_container_width=True)
    with c2:
        st.plotly_chart(plot_feature_heatmap(feat_df), use_container_width=True)

with tabs[2]:
    stats = compute_backtest_stats(preds_roll, df["close"].values[-len(preds_roll) - 4 :]) if len(preds_roll) else None
    if stats:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Profit Factor", f"{stats['profit_factor']:.3f}")
        r2.metric("Win Rate", f"{stats['win_rate']:.1%}")
        r3.metric("Max Drawdown", f"{stats['max_drawdown']:.1%}")
        r4.metric("Net Return", f"{stats['net_return']:+.2%}", f"{stats['total_trades']} trades")
        e1, e2 = st.columns(2)
        with e1:
            st.plotly_chart(plot_equity(stats), use_container_width=True)
        with e2:
            st.plotly_chart(plot_return_distribution(stats), use_container_width=True)
    else:
        st.info("No rolling predictions available for the selected window.")

with tabs[3]:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="qg-card"><div class="qg-card-title">Macro Regime</div>', unsafe_allow_html=True)
        st.metric("Gold", f"${macro['gold'].get('current', 0):,.0f}", f"{macro['gold'].get('momentum_5d', 0):+.2f}% 5d")
        st.metric("WTI Crude", f"${macro['crude'].get('current', 0):.1f}", f"{macro['crude'].get('momentum_5d', 0):+.2f}% 5d")
        st.metric("Fed Rate", f"{macro['fed'].get('rate', 0):.2f}%", f"Next FOMC {macro['fed'].get('next_fomc', 'n/a')}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        fig = plot_macro_matrix(feat_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    fig = plot_eigenspace(qp, report)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    q1, q2, q3 = st.columns(3)
    summary = qp.get_eigenvalue_summary()
    q1.metric("PCA Components", summary.get("n_components", "n/a"))
    q2.metric("Feature Dimension", feat_df.shape[1])
    q3.metric("Window Size", 60)

with tabs[5]:
    if report:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Trained Symbol", report.get("symbol", "n/a"))
        t2.metric("Bars", report.get("bars", "n/a"))
        t3.metric("Best Entry PF", f"{report.get('best_entry_pf', 0):.3f}" if isinstance(report.get("best_entry_pf"), (int, float)) else "n/a")
        t4.metric("Trained At", str(report.get("trained_at", "n/a"))[:10])
        st.json(report)
    else:
        st.info("No training report was found in the artifacts folder.")
