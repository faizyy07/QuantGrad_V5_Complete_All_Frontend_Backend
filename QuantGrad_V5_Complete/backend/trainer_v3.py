"""
trainer_v3.py — QuantGrad v4 Three-Model Pipeline (Memory-Optimized + Improved)
=================================================================================
Run: python trainer_v3.py --bars 20000
     python trainer_v3.py --quick

Improvements over original v3:
  Model architecture:
  - MultiHeadAttention (4 heads) after LSTM stacks in TrendModel
  - TemporalAttentionGate in StructureModel (sparse event detection)
  - Residual conv connections in StructureModel + EntryModel streams
  - SpatialDropout1D on conv layers (drops whole feature maps, better than Dropout)
  - LayerNormalization before dense fusions (stabler training)

  Training:
  - Label smoothing loss (0.05) → prevents overconfidence on noisy labels
  - Gradient clipping (clipnorm=1.0) → no NaN spikes
  - Cosine-annealing LR schedule → smoother than ReduceLROnPlateau

  Memory (all v3 optimizations preserved):
  - Mixed float16 precision
  - X_q computed once and shared
  - del + gc.collect() between steps
  - tf.keras.backend.clear_session() between steps
  - Incremental oversampling
"""

import os, sys, json, argparse, gc
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from feature_engine import (
    fetch_binance_klines, fetch_order_book,
    build_feature_matrix, build_windows,
    stochastic_hill_climb, WINDOW_SIZE, FEATURE_DIM
)
from market_structure import (
    build_structure_features,
    generate_trend_labels,
    generate_structure_labels,
    generate_entry_labels,
)
from quantum_layer import QuantumPreprocessor
from macro_fetcher import fetch_all_macro

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

STREAM_RANGES = {
    "gradient":   (0,  8),
    "momentum":   (8,  14),
    "volume":     (14, 22),
    "structure":  (22, 39),
    "trend":      (39, 48),
    "technical":  (48, 55),
    "macro":      (55, 65),
}

TREND_LABELS = {0: "UPTREND", 1: "DOWNTREND", 2: "RANGING"}
STRUCTURE_LABELS = {
    0: "MSS BULLISH",
    1: "MSS BEARISH",
    2: "ORDER BLOCK",
    3: "FAIR VALUE GAP",
    4: "LIQUIDITY SWEEP",
    5: "NONE",
}
SIGNAL_LABELS = {
    0: "STRONG BUY",
    1: "BUY",
    2: "HOLD",
    3: "SELL",
    4: "STRONG SELL",
}


def fusion_decision(entry_probs, trend_probs, struct_probs, adx_value, spread):
    """Fuse model probabilities into a trade signal and execution risk label."""
    entry_probs = np.asarray(entry_probs, dtype=float)
    trend_probs = np.asarray(trend_probs, dtype=float)
    struct_probs = np.asarray(struct_probs, dtype=float)

    signal = int(np.argmax(entry_probs))
    trend = int(np.argmax(trend_probs))
    structure = int(np.argmax(struct_probs))
    confidence = float(entry_probs[signal])
    adx_display = float(adx_value * 100 if adx_value <= 1.5 else adx_value)

    bullish_signal = signal in (0, 1)
    bearish_signal = signal in (3, 4)
    trend_aligned = (
        (bullish_signal and trend == 0)
        or (bearish_signal and trend == 1)
        or signal == 2
    )
    structure_support = (
        (bullish_signal and structure in (0, 2, 3, 4))
        or (bearish_signal and structure in (1, 2, 3, 4))
        or signal == 2
    )

    if signal == 2 or confidence < 0.28:
        risk_level = "NO_TRADE"
    elif confidence >= 0.55 and trend_aligned and structure_support and adx_display >= 20:
        risk_level = "CONFIRMED"
    elif not trend_aligned or spread > 100:
        risk_level = "HIGH_RISK"
    else:
        risk_level = "STANDARD"

    return {
        "signal": signal,
        "signal_label": SIGNAL_LABELS.get(signal, "UNKNOWN"),
        "risk_level": risk_level,
        "trend": TREND_LABELS.get(trend, "UNKNOWN"),
        "structure": STRUCTURE_LABELS.get(structure, "UNKNOWN"),
        "adx": adx_display,
        "confidence": confidence,
    }


# ── Label smoothing loss ──────────────────────────────────────────────
def make_label_smoothing_loss(num_classes, smoothing=0.05):
    def loss_fn(y_true, y_pred):
        # Keras supplies sparse labels as either (batch,) or (batch, 1)
        # depending on the adapter; reshape handles both safely.
        y_true   = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_hot    = tf.one_hot(y_true, num_classes, dtype=tf.float32)
        y_smooth = y_hot * (1.0 - smoothing) + smoothing / num_classes
        y_pred   = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0), tf.float32)
        return -tf.reduce_mean(tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1))
    return loss_fn


# ── Temporal attention gate ───────────────────────────────────────────
class TemporalAttentionGate(layers.Layer):
    """Learns which timesteps matter most — critical for sparse MSS/OB events."""
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="glorot_uniform", name="W")
        self.b = self.add_weight(shape=(input_shape[1], 1),  initializer="zeros",          name="b")
        super().build(input_shape)

    def call(self, x):
        score   = tf.matmul(x, tf.cast(self.W, x.dtype)) + tf.cast(self.b, x.dtype)
        weights = tf.cast(tf.nn.softmax(tf.cast(score, tf.float32), axis=1), x.dtype)
        return tf.reduce_sum(x * weights, axis=1)

    def get_config(self):
        return super().get_config()


# ── Cosine annealing LR ───────────────────────────────────────────────
def make_cosine_lr(initial_lr=3e-4, min_lr=1e-6, total_epochs=60):
    def schedule(epoch):
        cos = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        return float(min_lr + (initial_lr - min_lr) * cos)
    return callbacks.LearningRateScheduler(schedule, verbose=0)


# ── Oversampling ──────────────────────────────────────────────────────
def manual_oversample(X, y, target_ratio=0.20):
    counts   = Counter(y)
    target_n = int(len(y) * target_ratio)
    print(f"[oversample] Before: {dict(counts)}")
    extras_X, extras_y = [], []
    for cls, cnt in counts.items():
        if cls == 2 and len(counts) == 5: continue
        need = max(0, target_n - cnt)
        if need == 0: continue
        idx   = np.where(y == cls)[0]
        extra = np.random.choice(idx, size=need, replace=True)
        noise = np.random.normal(0, 0.005, X[extra].shape).astype(np.float32)
        extras_X.append(X[extra] + noise)
        extras_y.append(np.full(need, cls, dtype=np.int32))
    if extras_X:
        X_b = np.concatenate([X] + extras_X).astype(np.float32)
        y_b = np.concatenate([y] + extras_y).astype(np.int32)
    else:
        X_b, y_b = X.astype(np.float32), y.astype(np.int32)
    del extras_X, extras_y; gc.collect()
    perm = np.random.permutation(len(X_b))
    X_b, y_b = X_b[perm], y_b[perm]
    print(f"[oversample] After:  {dict(Counter(y_b))}")
    return X_b, y_b


# ── Profit factor callback ─────────────────────────────────────────────
class ProfitFactorCallback(callbacks.Callback):
    def __init__(self, X_val, y_val, prices_val, patience=8, save_path=None, signal_map=None):
        super().__init__()
        self.X_val=X_val; self.y_val=y_val; self.prices_val=prices_val
        self.patience=patience; self.save_path=save_path
        self.signal_map=signal_map or {1:"BUY",3:"SELL",0:"STRONG_BUY",4:"STRONG_SELL"}
        self.best_pf=-np.inf; self.wait=0; self.history_pf=[]

    def on_epoch_end(self, epoch, logs=None):
        preds  = np.argmax(self.model.predict(self.X_val, verbose=0, batch_size=512), axis=1)
        pf, wr = self._pf(preds, self.prices_val)
        trades = int((preds != 2).sum())
        self.history_pf.append(pf)
        print(f"  → PF:{pf:.4f} WR:{wr:.1%} trades:{trades} dist:{dict(Counter(preds.tolist()))}")
        if pf > self.best_pf:
            self.best_pf, self.wait = pf, 0
            if self.save_path: self.model.save_weights(self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"[trainer] Early stop — best PF: {self.best_pf:.4f}")

    def _pf(self, preds, prices, hold=3):
        gp, gl, wins, total = 0., 0., 0, 0
        for i in range(len(preds)-hold-1):
            if preds[i] == 2: continue
            if i+1+hold >= len(prices): break
            entry, exit_ = prices[i+1], prices[i+1+hold]
            if entry == 0: continue
            pnl = (exit_-entry)/entry
            if preds[i] in [3,4]: pnl = -pnl
            if pnl > 0: gp += pnl; wins += 1
            else: gl += abs(pnl)
            total += 1
        return round(gp/(gl+1e-9), 4), round(wins/(total+1e-9), 4)


# ── Model 1: TrendModel ───────────────────────────────────────────────
def build_trend_model(window, n_features):
    inp  = keras.Input(shape=(window, n_features))
    x    = layers.Conv1D(64, 5, padding="causal", activation="relu")(inp)
    x    = layers.BatchNormalization()(x)
    x    = layers.SpatialDropout1D(0.10)(x)
    x    = layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.MaxPooling1D(2)(x)
    x    = layers.LSTM(128, return_sequences=True)(x)
    x    = layers.LSTM(64,  return_sequences=True)(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x    = layers.LayerNormalization()(x + attn)
    x    = layers.GlobalAveragePooling1D()(x)
    x    = layers.Dropout(0.2)(x)
    x    = layers.Dense(32, activation="relu")(x)
    out  = layers.Dense(3, activation="softmax", name="trend_out", dtype="float32")(x)
    m    = keras.Model(inp, out, name="TrendModel")
    m.compile(optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0),
              loss=make_label_smoothing_loss(3, 0.05), metrics=["accuracy"])
    return m


# ── Model 2: StructureModel ───────────────────────────────────────────
def build_structure_model(window, n_features):
    inp    = keras.Input(shape=(window, n_features))
    x      = layers.Conv1D(128, 3, padding="causal", activation="relu")(inp)
    x      = layers.BatchNormalization()(x)
    x      = layers.SpatialDropout1D(0.15)(x)
    x_skip = layers.Conv1D(64, 1)(x)
    x      = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Add()([x, x_skip])
    x      = layers.MaxPooling1D(2)(x)
    x      = layers.SpatialDropout1D(0.15)(x)
    x      = layers.LSTM(192, return_sequences=True)(x)   # 256→192 saves RAM
    x      = layers.LSTM(96,  return_sequences=True)(x)   # 128→96
    x      = TemporalAttentionGate()(x)
    x      = layers.Dropout(0.25)(x)
    x      = layers.Dense(64, activation="relu")(x)
    out    = layers.Dense(6, activation="softmax", name="structure_out", dtype="float32")(x)
    m      = keras.Model(inp, out, name="StructureModel")
    m.compile(optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0),
              loss=make_label_smoothing_loss(6, 0.05), metrics=["accuracy"])
    return m


# ── Model 3: EntryModel ───────────────────────────────────────────────
def build_entry_model(window, n_features, n_trend_probs=3, n_struct_probs=6):
    feat_inp   = keras.Input(shape=(window, n_features), name="features")
    trend_inp  = keras.Input(shape=(n_trend_probs,),     name="trend_probs")
    struct_inp = keras.Input(shape=(n_struct_probs,),    name="struct_probs")

    def stream_lstm(x_slice, units, name):
        s     = layers.Conv1D(32, 3, padding="causal", activation="relu", name=f"conv_{name}")(x_slice)
        s     = layers.BatchNormalization(name=f"bn_{name}")(s)
        s_res = layers.Conv1D(32, 1, name=f"res_{name}")(x_slice)
        s     = layers.Add(name=f"add_{name}")([s, s_res])
        return layers.LSTM(units, name=f"lstm_{name}")(s)

    def feat_slice(start, end):
        return layers.Lambda(lambda x: x[:, :, start:end],
                             output_shape=(window, end-start))(feat_inp)

    sr = STREAM_RANGES
    streams = [
        stream_lstm(feat_slice(*sr["gradient"]),  32, "grad"),
        stream_lstm(feat_slice(*sr["momentum"]),  24, "mom"),
        stream_lstm(feat_slice(*sr["volume"]),    32, "vol"),
        stream_lstm(feat_slice(*sr["structure"]), 48, "str"),
        stream_lstm(feat_slice(*sr["trend"]),     32, "trnd"),
        stream_lstm(feat_slice(*sr["technical"]), 24, "tech"),
        stream_lstm(feat_slice(*sr["macro"]),     24, "mac"),
    ]
    ctx    = layers.LayerNormalization()(layers.Dense(32, activation="relu")(
                layers.Concatenate()([trend_inp, struct_inp])))
    merged = layers.Concatenate()(streams + [ctx])
    x      = layers.Dense(128, activation="relu")(merged)
    x      = layers.LayerNormalization()(x)
    x      = layers.Dropout(0.3)(x)
    x      = layers.Dense(64, activation="relu")(x)
    x      = layers.Dropout(0.2)(x)
    out    = layers.Dense(5, activation="softmax", name="entry_out", dtype="float32")(x)
    m      = keras.Model([feat_inp, trend_inp, struct_inp], out, name="EntryModel")
    m.compile(optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0),
              loss=make_label_smoothing_loss(5, 0.05), metrics=["accuracy"])
    return m


# ── Main train function ───────────────────────────────────────────────
def train(symbol="BTCUSDT", total_bars=20000, quick=False):
    print(f"\n{'='*65}\n  QuantGrad v4 — {symbol}  bars={total_bars}  quick={quick}\n{'='*65}\n")

    df    = fetch_binance_klines(symbol=symbol, interval="1h", total_bars=total_bars)
    ob    = fetch_order_book(symbol=symbol)
    macro = fetch_all_macro(btc_hourly_df=df, save=True)

    struct_df = build_structure_features(df)
    feat_df, struct_aligned = build_feature_matrix(df, macro, ob, struct_df=struct_df)
    df_aligned = df.loc[feat_df.index]
    prices     = df_aligned["close"].values.astype(np.float32)

    trend_labels  = generate_trend_labels(df_aligned, struct_aligned)
    struct_labels = generate_structure_labels(struct_aligned)
    mid           = ((df_aligned["open"] + df_aligned["close"]) / 2).values
    prom, dist    = stochastic_hill_climb(mid, n_iter=100, target_min=10, target_max=40)
    entry_labels  = generate_entry_labels(df_aligned, struct_aligned, trend_labels,
                                           prominence=max(prom,50.0), distance=max(dist,4))

    print(f"Labels — Trend:{dict(Counter(trend_labels.tolist()))}  "
          f"Structure:{dict(Counter(struct_labels.tolist()))}  "
          f"Entry:{dict(Counter(entry_labels.tolist()))}")

    feat_vals = feat_df.values
    X_trend,  y_trend  = build_windows(feat_vals, trend_labels)
    X_struct, y_struct = build_windows(feat_vals, struct_labels)
    X_entry,  y_entry  = build_windows(feat_vals, entry_labels)
    prices_w           = prices[WINDOW_SIZE:]

    del feat_df, struct_df, feat_vals; gc.collect()

    # Quantum
    qp = QuantumPreprocessor()
    qp.fit(X_trend.reshape(-1, X_trend.shape[2]),
           n_quantum_train_samples=min(100, len(X_trend)))
    qp.save(ARTIFACTS_DIR)

    def to_q(X_3d):
        n, w, f = X_3d.shape
        return qp.decomposer.transform(
            qp.scaler.transform(X_3d.reshape(-1, f))
        ).reshape(n, w, -1).astype(np.float32)

    print("\n[QUANTUM] Shared X_q transform...")
    X_q = to_q(X_trend)
    n_q_feat = X_q.shape[2]
    # The EntryModel uses stream slices defined over the original 65-feature
    # schema. Trend and Structure consume PCA windows, but Entry must retain
    # the aligned raw windows or its macro stream becomes an empty slice.
    X_entry_raw = X_entry[:len(X_q)].astype(np.float32)
    del X_trend, X_struct, X_entry; gc.collect()

    n, n_train, n_val = len(X_q), int(len(X_q)*0.70), int(len(X_q)*0.15)
    def split3(y): return y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    ytr_t, yv_t, _ = split3(y_trend[:n])
    ytr_s, yv_s, _ = split3(y_struct[:n])
    ytr_e, yv_e, _ = split3(y_entry[:n])
    Xtr_q = X_q[:n_train]
    Xv_q  = X_q[n_train:n_train+n_val]
    Xtr_entry = X_entry_raw[:n_train]
    Xv_entry = X_entry_raw[n_train:n_train+n_val]
    p_val = prices_w[n_train:n_train+n_val]
    epochs = 5 if quick else 60

    # ── STEP 1: Trend ────────────────────────────────────────────────
    print("\n" + "─"*50 + "\n[STEP 1] TrendModel")
    Xtr_t_os, ytr_t_os = manual_oversample(Xtr_q, ytr_t, 0.25)
    cw_t = {c: max(1.0, len(ytr_t_os)/(3*Counter(ytr_t_os).get(c,1))) for c in range(3)}
    tm = build_trend_model(WINDOW_SIZE, n_q_feat)
    tm.fit(Xtr_t_os, ytr_t_os, validation_data=(Xv_q, yv_t),
           epochs=epochs, batch_size=32, class_weight=cw_t,
           callbacks=[make_cosine_lr(total_epochs=epochs)], verbose=1)
    tm.save(os.path.join(ARTIFACTS_DIR, "trend_model.keras"))
    del Xtr_t_os, ytr_t_os; gc.collect(); tf.keras.backend.clear_session()
    print("[STEP 1] Done.")

    # ── STEP 2: Structure ────────────────────────────────────────────
    print("\n" + "─"*50 + "\n[STEP 2] StructureModel")
    Xtr_s_os, ytr_s_os = manual_oversample(Xtr_q, ytr_s, 0.15)
    cw_s = {c: max(1.0, len(ytr_s_os)/(6*Counter(ytr_s_os).get(c,1))) for c in range(6)}
    sm = build_structure_model(WINDOW_SIZE, n_q_feat)
    sm.fit(Xtr_s_os, ytr_s_os, validation_data=(Xv_q, yv_s),
           epochs=epochs, batch_size=32, class_weight=cw_s,
           callbacks=[make_cosine_lr(total_epochs=epochs)], verbose=1)
    sm.save(os.path.join(ARTIFACTS_DIR, "structure_model.keras"))
    del Xtr_s_os, ytr_s_os; gc.collect(); tf.keras.backend.clear_session()
    print("[STEP 2] Done.")

    # ── STEP 3: Entry ────────────────────────────────────────────────
    print("\n" + "─"*50 + "\n[STEP 3] EntryModel")
    custom_objs = {
        "TemporalAttentionGate": TemporalAttentionGate,
        "loss_fn": make_label_smoothing_loss(6),
    }
    tm2 = keras.models.load_model(os.path.join(ARTIFACTS_DIR, "trend_model.keras"),
                                   custom_objects=custom_objs, compile=False)
    sm2 = keras.models.load_model(os.path.join(ARTIFACTS_DIR, "structure_model.keras"),
                                   custom_objects=custom_objs, compile=False)
    tp_all = tm2.predict(X_q, verbose=0, batch_size=512)
    sp_all = sm2.predict(X_q, verbose=0, batch_size=512)
    ttr, tv     = tp_all[:n_train], tp_all[n_train:n_train+n_val]
    str_tr, str_v = sp_all[:n_train], sp_all[n_train:n_train+n_val]
    del tp_all, sp_all; gc.collect()

    cnt_e  = Counter(ytr_e.tolist())
    n_hold = cnt_e.get(2, 1)
    cw_e   = {0:max(3.0,n_hold/cnt_e.get(0,1)), 1:max(2.0,n_hold/cnt_e.get(1,1)),
              2:1.0, 3:max(2.0,n_hold/cnt_e.get(3,1)), 4:max(3.0,n_hold/cnt_e.get(4,1))}
    em = build_entry_model(WINDOW_SIZE, Xtr_entry.shape[2])
    pf_cb = ProfitFactorCallback(X_val=[Xv_entry,tv,str_v], y_val=yv_e, prices_val=p_val,
                                  patience=8,
                                  save_path=os.path.join(ARTIFACTS_DIR,"best_entry.weights.h5"))
    em.fit([Xtr_entry,ttr,str_tr], ytr_e, validation_data=([Xv_entry,tv,str_v], yv_e),
           epochs=epochs, batch_size=32, class_weight=cw_e,
           callbacks=[pf_cb, make_cosine_lr(total_epochs=epochs)], verbose=1)
    em.save(os.path.join(ARTIFACTS_DIR, "entry_model.keras"))

    report = {
        "trained_at": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "bars": int(total_bars),
        "quick": bool(quick),
        "best_entry_pf": float(pf_cb.best_pf),
        "entry_pf_history": [float(x) for x in pf_cb.history_pf],
        "feature_dim": int(FEATURE_DIM),
        "window_size": int(WINDOW_SIZE),
        "quantum": qp.get_eigenvalue_summary(),
    }
    with open(os.path.join(ARTIFACTS_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*65}\n[trainer] Done. Best PF: {pf_cb.best_pf:.4f}\n{'='*65}\n")
    return tm, sm, em, qp


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--bars",   type=int, default=20000)
    p.add_argument("--quick",  action="store_true")
    a = p.parse_args()
    train(symbol=a.symbol, total_bars=a.bars, quick=a.quick)
