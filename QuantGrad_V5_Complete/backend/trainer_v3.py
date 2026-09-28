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
    WINDOW_SIZE, FEATURE_DIM
)
from market_structure import (
    build_structure_features,
    generate_trend_labels,
    generate_structure_labels,
)
from quantum_layer import QuantumMarketClassifier, QuantumPreprocessor
from macro_fetcher import fetch_all_macro
from strategy_experiments import (
    DEFAULT_STRATEGY_CONFIG,
    build_barrier_labels,
    classification_metrics as experiment_classification_metrics,
    directional_labels,
    directional_probabilities_to_signals,
    evaluate_walk_forward_baselines,
    five_class_probabilities_to_direction,
    predict_aligned_probabilities,
    simulate_variable_exit,
    stack_features,
    strategy_config_dict,
    train_tabular_baselines,
    train_oof_tree_stacker,
    tune_confidence_threshold,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
TRADE_FEE_RATE = 0.0005
TRADE_SLIPPAGE_RATE = 0.0002
VALIDATION_CONFIDENCE_THRESHOLDS = (0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

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


def fusion_decision(entry_probs, trend_probs, struct_probs, adx_value, spread, min_direction_confidence=0.0):
    """Fuse model probabilities into a trade signal and execution risk label."""
    entry_probs = np.asarray(entry_probs, dtype=float)
    trend_probs = np.asarray(trend_probs, dtype=float)
    struct_probs = np.asarray(struct_probs, dtype=float)

    signal = int(decode_entry_predictions(
        entry_probs.reshape(1, -1), min_direction_confidence
    )[0])
    trend = int(np.argmax(trend_probs))
    structure = int(np.argmax(struct_probs))
    directional_probs = np.array([
        entry_probs[3] + entry_probs[4],
        entry_probs[2],
        entry_probs[0] + entry_probs[1],
    ])
    confidence = float(directional_probs[0 if signal in (3, 4) else 2 if signal in (0, 1) else 1])
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


def decode_entry_predictions(probabilities, min_direction_confidence=0.0):
    """Combine strong/regular classes before deciding direction; hold ties."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("Entry probabilities must have five class columns")
    buy = probabilities[:, 0] + probabilities[:, 1]
    hold = probabilities[:, 2]
    sell = probabilities[:, 3] + probabilities[:, 4]
    predictions = np.full(len(probabilities), 2, dtype=np.int32)
    buy_wins = (buy > hold) & (buy > sell) & (buy >= min_direction_confidence)
    sell_wins = (sell > hold) & (sell > buy) & (sell >= min_direction_confidence)
    predictions[buy_wins] = np.where(
        probabilities[buy_wins, 0] >= probabilities[buy_wins, 1], 0, 1
    )
    predictions[sell_wins] = np.where(
        probabilities[sell_wins, 4] >= probabilities[sell_wins, 3], 4, 3
    )
    return predictions


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


def classification_metrics(y_true, probabilities):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, precision_score, recall_score

    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, 1e-8, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.arange(probabilities.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
    }


def forward_direction_labels(prices, horizon=3, threshold=0.001):
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) <= horizon + 1 or np.any(prices[1:-horizon] <= 0):
        raise ValueError("Not enough positive prices to build forward direction labels")
    forward_returns = prices[horizon + 1:] / prices[1:-horizon] - 1.0
    return np.where(forward_returns < -threshold, 0,
                    np.where(forward_returns > threshold, 2, 1)).astype(np.int32)


def build_model_comparison(X_train, y_train, X_test, y_test, ensemble_probabilities, artifact_dir=ARTIFACTS_DIR):
    quantum_model = QuantumMarketClassifier(n_qubits=4, n_classes=3)
    quantum_model.fit(X_train, y_train, epochs=4, batch_size=16, max_samples=96)
    quantum_probabilities = quantum_model.predict_proba(X_test)
    quantum_model.save(os.path.join(artifact_dir, "quantum_classifier.pkl"))
    return {
        "evaluation": "chronological_holdout",
        "target_definition": "next-close entry, 3-bar return: down below -0.1%, flat within +/-0.1%, up above +0.1%",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "quantum": {
            "status": "trained",
            "simulator": "Qiskit Statevector",
            "n_qubits": quantum_model.n_qubits,
            "trained_epochs": quantum_model.trained_epochs,
            **classification_metrics(y_test, quantum_probabilities),
        },
        "ensemble": {
            "status": "evaluated",
            **classification_metrics(y_test, ensemble_probabilities),
        },
    }


def _strategy_result(model_id, label, y_test, probabilities, signals, bars_test, threshold):
    backtest = simulate_variable_exit(
        signals,
        bars_test,
        y_test,
        config=DEFAULT_STRATEGY_CONFIG,
    )
    return {
        "model_id": model_id,
        "label": label,
        "confidence_threshold": float(threshold),
        "classification": experiment_classification_metrics(y_test, probabilities),
        "backtest": backtest,
    }


def evaluate_strategy_models(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    bars_val,
    bars_test,
    X_all,
    y_all,
    bars_all,
    entry_val_probabilities,
    entry_test_probabilities,
    artifact_dir=ARTIFACTS_DIR,
):
    validation_bars = bars_val
    test_bars = bars_test
    model_specs = []

    neural_decoder = lambda probabilities, threshold: decode_entry_predictions(probabilities, threshold)
    neural_threshold, neural_validation = tune_confidence_threshold(
        entry_val_probabilities, neural_decoder, validation_bars, DEFAULT_STRATEGY_CONFIG
    )
    neural_direction_probabilities = five_class_probabilities_to_direction(entry_test_probabilities)
    neural_signals = neural_decoder(entry_test_probabilities, neural_threshold)
    model_specs.append(_strategy_result(
        "neural_ensemble", "CNN/LSTM ensemble", y_test, neural_direction_probabilities,
        neural_signals, test_bars, neural_threshold,
    ))

    quantum_model = QuantumMarketClassifier(n_qubits=4, n_classes=3)
    quantum_model.fit(X_train, y_train, epochs=4, batch_size=16, max_samples=96)
    quantum_model.save(os.path.join(artifact_dir, "quantum_classifier.pkl"))
    quantum_val = quantum_model.predict_proba(X_val)
    quantum_test = quantum_model.predict_proba(X_test)
    quantum_threshold, _ = tune_confidence_threshold(
        quantum_val, directional_probabilities_to_signals, validation_bars, DEFAULT_STRATEGY_CONFIG
    )
    model_specs.append(_strategy_result(
        "qiskit_vqc", "Qiskit · 4 qubits", y_test, quantum_test,
        directional_probabilities_to_signals(quantum_test, quantum_threshold),
        test_bars, quantum_threshold,
    ))

    baselines = train_tabular_baselines(X_train, y_train)
    import joblib
    validation_base_probabilities = {}
    test_base_probabilities = {}
    for model_id, label in (
        ("random_forest", "Random Forest · bagging"),
        ("hist_gradient_boosting", "HistGradientBoosting · boosting"),
    ):
        model = baselines[model_id]
        joblib.dump(model, os.path.join(artifact_dir, f"{model_id}.joblib"))
        val_probabilities = predict_aligned_probabilities(model, X_val)
        test_probabilities = predict_aligned_probabilities(model, X_test)
        validation_base_probabilities[model_id] = val_probabilities
        test_base_probabilities[model_id] = test_probabilities
        threshold, _ = tune_confidence_threshold(
            val_probabilities, directional_probabilities_to_signals,
            validation_bars, DEFAULT_STRATEGY_CONFIG,
        )
        model_specs.append(_strategy_result(
            model_id, label, y_test, test_probabilities,
            directional_probabilities_to_signals(test_probabilities, threshold),
            test_bars, threshold,
        ))

    stacker, stacker_oof_samples = train_oof_tree_stacker(X_train, y_train)
    joblib.dump(stacker, os.path.join(artifact_dir, "tree_stacker.joblib"))
    validation_stacked = predict_aligned_probabilities(
        stacker,
        stack_features(
            validation_base_probabilities["random_forest"],
            validation_base_probabilities["hist_gradient_boosting"],
        ),
    )
    test_stacked = predict_aligned_probabilities(
        stacker,
        stack_features(
            test_base_probabilities["random_forest"],
            test_base_probabilities["hist_gradient_boosting"],
        ),
    )
    stacker_threshold, _ = tune_confidence_threshold(
        validation_stacked, directional_probabilities_to_signals,
        validation_bars, DEFAULT_STRATEGY_CONFIG,
    )
    model_specs.append(_strategy_result(
        "tree_stacker", "OOF stack · Random Forest + boosting", y_test,
        test_stacked, directional_probabilities_to_signals(test_stacked, stacker_threshold),
        test_bars, stacker_threshold,
    ))

    walk_forward = evaluate_walk_forward_baselines(
        X_all,
        y_all,
        bars_all,
        DEFAULT_STRATEGY_CONFIG,
        folds=3,
    )
    sequence_comparison = {
        "evaluation": "chronological_holdout",
        "target_definition": "next-open entry; ATR barriers; model HOLD/reversal exit; max 24 bars",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "quantum": {
            "status": "trained",
            "simulator": "Qiskit Statevector",
            "n_qubits": quantum_model.n_qubits,
            "trained_epochs": quantum_model.trained_epochs,
            **experiment_classification_metrics(y_test, quantum_test),
        },
        "ensemble": {
            "status": "evaluated",
            **experiment_classification_metrics(y_test, neural_direction_probabilities),
        },
    }
    return {
        "strategy_results": model_specs,
        "walk_forward": walk_forward,
        "model_comparison": sequence_comparison,
        "holdout_backtest": model_specs[0]["backtest"],
        "validation_thresholds": {
            "neural_ensemble": neural_threshold,
            "qiskit_vqc": quantum_threshold,
            "random_forest": next(row["confidence_threshold"] for row in model_specs if row["model_id"] == "random_forest"),
            "hist_gradient_boosting": next(row["confidence_threshold"] for row in model_specs if row["model_id"] == "hist_gradient_boosting"),
            "tree_stacker": stacker_threshold,
        },
        "stacker_oof_samples": stacker_oof_samples,
        "validation_neural_backtest": {key: value for key, value in neural_validation.items() if key != "trades"},
    }


def macro_data_status(macro):
    fallback_inputs = []
    for name, label in (("dxy", "DXY"), ("gold", "gold"), ("crude", "crude")):
        if not macro.get(name, {}).get("series"):
            fallback_inputs.append(label)
    if macro.get("fed", {}).get("source") == "fallback":
        fallback_inputs.append("Fed rate")
    if macro.get("btc_dominance", {}).get("source") == "fallback":
        fallback_inputs.append("BTC dominance")
    if not macro.get("fear_greed", {}).get("history_30d"):
        fallback_inputs.append("Fear & Greed")
    return {
        "fetched_at": macro.get("fetched_at"),
        "fallback_inputs": fallback_inputs,
        "warning": "Some macro inputs used provider fallbacks." if fallback_inputs else None,
    }


# ── Profit factor callback ─────────────────────────────────────────────
class ProfitFactorCallback(callbacks.Callback):
    def __init__(self, X_val, y_val, bars_val, patience=8, save_path=None, signal_map=None):
        super().__init__()
        self.X_val=X_val; self.y_val=y_val; self.bars_val=bars_val
        self.patience=patience; self.save_path=save_path
        self.signal_map=signal_map or {1:"BUY",3:"SELL",0:"STRONG_BUY",4:"STRONG_SELL"}
        self.best_pf=-np.inf; self.best_threshold=0.0; self.wait=0; self.history_pf=[]

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(self.X_val, verbose=0, batch_size=512)
        threshold, stats = tune_confidence_threshold(
            probabilities,
            lambda values, gate: decode_entry_predictions(values, gate),
            self.bars_val,
            DEFAULT_STRATEGY_CONFIG,
        )
        pf = float(stats["profit_factor"])
        wr = float(stats["win_rate"])
        trades = int(stats["total_trades"])
        preds = decode_entry_predictions(probabilities, threshold)
        self.history_pf.append(pf)
        print(f"  → PF:{pf:.4f} WR:{wr:.1%} trades:{trades} gate:{threshold:.2f} dist:{dict(Counter(preds.tolist()))}")
        if pf > self.best_pf:
            self.best_pf, self.best_threshold, self.wait = pf, threshold, 0
            if self.save_path: self.model.save_weights(self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"[trainer] Early stop — best PF: {self.best_pf:.4f}")

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
def train(symbol="BTCUSDT", total_bars=20000, quick=False, compare_existing=False):
    print(f"\n{'='*65}\n  QuantGrad v4 — {symbol}  bars={total_bars}  quick={quick}\n{'='*65}\n")

    df    = fetch_binance_klines(symbol=symbol, interval="1h", total_bars=total_bars)
    ob    = fetch_order_book(symbol=symbol)
    macro = fetch_all_macro(btc_hourly_df=df, save=False)

    struct_df = build_structure_features(df)
    feat_df, struct_aligned = build_feature_matrix(df, macro, ob, struct_df=struct_df)
    df_aligned = df.loc[feat_df.index]
    prices     = df_aligned["close"].values.astype(np.float32)

    trend_labels  = generate_trend_labels(df_aligned, struct_aligned)
    struct_labels = generate_structure_labels(struct_aligned)
    entry_labels  = build_barrier_labels(df_aligned, DEFAULT_STRATEGY_CONFIG)

    print(f"Labels — Trend:{dict(Counter(trend_labels.tolist()))}  "
          f"Structure:{dict(Counter(struct_labels.tolist()))}  "
          f"Entry:{dict(Counter(entry_labels.tolist()))}")

    feat_vals = feat_df.values
    X_trend,  y_trend  = build_windows(feat_vals, trend_labels)
    X_struct, y_struct = build_windows(feat_vals, struct_labels)
    X_entry,  y_entry  = build_windows(feat_vals, entry_labels)
    prices_w           = prices[WINDOW_SIZE:]
    bars_w              = df_aligned.iloc[WINDOW_SIZE:].copy()

    del feat_df, struct_df, feat_vals; gc.collect()

    # Quantum
    if compare_existing:
        qp = QuantumPreprocessor.load(ARTIFACTS_DIR)
    else:
        qp = QuantumPreprocessor()
        preprocessor_train_end = max(1, int(len(X_trend) * 0.70) - DEFAULT_STRATEGY_CONFIG.max_holding_bars)
        qp.fit(X_trend[:preprocessor_train_end].reshape(-1, X_trend.shape[2]),
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

    n = len(X_q)
    train_boundary = int(n * 0.70)
    validation_boundary = int(n * 0.85)
    purge = DEFAULT_STRATEGY_CONFIG.max_holding_bars
    train_end = max(1, train_boundary - purge)
    validation_start = train_boundary
    validation_end = max(validation_start + 1, validation_boundary - purge)
    test_start = validation_boundary
    direction_labels = directional_labels(y_entry[:n])
    comparison_end = min(n, len(direction_labels))
    ytr_t = y_trend[:train_end]
    yv_t = y_trend[validation_start:validation_end]
    ytr_s = y_struct[:train_end]
    yv_s = y_struct[validation_start:validation_end]
    ytr_e = y_entry[:train_end]
    yv_e = y_entry[validation_start:validation_end]
    Xtr_q = X_q[:train_end]
    Xv_q  = X_q[validation_start:validation_end]
    Xtr_entry = X_entry_raw[:train_end]
    Xv_entry = X_entry_raw[validation_start:validation_end]
    X_test_q = X_q[test_start:comparison_end]
    X_test_entry = X_entry_raw[test_start:comparison_end]
    y_train_direction = direction_labels[:train_end]
    y_validation_direction = direction_labels[validation_start:validation_end]
    y_test_direction = direction_labels[test_start:comparison_end]
    bars_w = bars_w[:comparison_end]
    X_entry_snapshots = X_entry_raw[:comparison_end, -1, :]
    epochs = 5 if quick else 60

    if compare_existing:
        custom_objects = {"TemporalAttentionGate": TemporalAttentionGate}
        models = [
            keras.models.load_model(
                os.path.join(ARTIFACTS_DIR, filename),
                custom_objects=custom_objects,
                compile=False,
                safe_mode=False,
            )
            for filename in ("trend_model.keras", "structure_model.keras", "entry_model.keras")
        ]
        trend_val = models[0].predict(Xv_q, verbose=0, batch_size=512)
        structure_val = models[1].predict(Xv_q, verbose=0, batch_size=512)
        entry_val_probabilities = models[2].predict(
            [Xv_entry, trend_val, structure_val], verbose=0, batch_size=512
        )
        trend_test = models[0].predict(X_test_q, verbose=0, batch_size=512)
        structure_test = models[1].predict(X_test_q, verbose=0, batch_size=512)
        entry_test_probabilities = models[2].predict(
            [X_test_entry, trend_test, structure_test], verbose=0, batch_size=512
        )
        experiment = evaluate_strategy_models(
            X_entry_snapshots[:train_end], y_train_direction,
            X_entry_snapshots[validation_start:validation_end], y_validation_direction,
            X_entry_snapshots[test_start:comparison_end], y_test_direction,
            bars_w[validation_start:validation_end], bars_w[test_start:comparison_end],
            X_entry_snapshots, direction_labels[:comparison_end], bars_w,
            entry_val_probabilities, entry_test_probabilities,
        )
        report_path = os.path.join(ARTIFACTS_DIR, "training_report.json")
        report = {}
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        report.update({
            "comparison_generated_at": datetime.utcnow().isoformat(),
            "comparison_symbol": symbol,
            "comparison_bars": int(total_bars),
            "model_comparison": experiment["model_comparison"],
            "strategy_results": experiment["strategy_results"],
            "walk_forward": experiment["walk_forward"],
            "holdout_backtest": experiment["holdout_backtest"],
            "macro_fetched_at": macro.get("fetched_at"),
            "macro_status": macro_data_status(macro),
            "entry_confidence_threshold": experiment["validation_thresholds"]["neural_ensemble"],
            "stacker_oof_samples": experiment["stacker_oof_samples"],
            "strategy_config": strategy_config_dict(DEFAULT_STRATEGY_CONFIG),
        })
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[trainer] Existing ensemble vs Qiskit comparison saved to {report_path}")
        return None, None, None, qp

    t_test = s_test = None

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
    ttr, tv     = tp_all[:train_end], tp_all[validation_start:validation_end]
    str_tr, str_v = sp_all[:train_end], sp_all[validation_start:validation_end]
    t_test, s_test = tp_all[test_start:comparison_end], sp_all[test_start:comparison_end]
    del tp_all, sp_all; gc.collect()

    cnt_e  = Counter(ytr_e.tolist())
    n_hold = cnt_e.get(2, 1)
    cw_e   = {0:max(3.0,n_hold/cnt_e.get(0,1)), 1:max(2.0,n_hold/cnt_e.get(1,1)),
              2:1.0, 3:max(2.0,n_hold/cnt_e.get(3,1)), 4:max(3.0,n_hold/cnt_e.get(4,1))}
    em = build_entry_model(WINDOW_SIZE, Xtr_entry.shape[2])
    validation_bars = bars_w[validation_start:validation_end]
    pf_cb = ProfitFactorCallback(X_val=[Xv_entry,tv,str_v], y_val=yv_e, bars_val=validation_bars,
                                  patience=8,
                                  save_path=os.path.join(ARTIFACTS_DIR,"best_entry.weights.h5"))
    em.fit([Xtr_entry,ttr,str_tr], ytr_e, validation_data=([Xv_entry,tv,str_v], yv_e),
           epochs=epochs, batch_size=32, class_weight=cw_e,
           callbacks=[pf_cb, make_cosine_lr(total_epochs=epochs)], verbose=1)
    best_entry_weights = os.path.join(ARTIFACTS_DIR, "best_entry.weights.h5")
    if os.path.exists(best_entry_weights):
        em.load_weights(best_entry_weights)
    em.save(os.path.join(ARTIFACTS_DIR, "entry_model.keras"))

    validation_probabilities = em.predict(
        [Xv_entry, tv, str_v], verbose=0, batch_size=512
    )
    test_probabilities = em.predict(
        [X_test_entry, t_test, s_test], verbose=0, batch_size=512
    )
    experiment = evaluate_strategy_models(
        X_entry_snapshots[:train_end], y_train_direction,
        X_entry_snapshots[validation_start:validation_end], y_validation_direction,
        X_entry_snapshots[test_start:comparison_end], y_test_direction,
        bars_w[validation_start:validation_end], bars_w[test_start:comparison_end],
        X_entry_snapshots, direction_labels[:comparison_end], bars_w,
        validation_probabilities, test_probabilities,
    )

    report = {
        "trained_at": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "bars": int(total_bars),
        "quick": bool(quick),
        "best_entry_pf": float(pf_cb.best_pf),
        "entry_confidence_threshold": float(pf_cb.best_threshold),
        "entry_pf_history": [float(x) for x in pf_cb.history_pf],
        "entry_label_counts": {str(label): int(count) for label, count in sorted(Counter(entry_labels.tolist()).items())},
        "entry_label_target": "next-open entry; ATR stop/target outcomes; model reversal/HOLD exit; maximum holding period",
        "macro_fetched_at": macro.get("fetched_at"),
        "macro_status": macro_data_status(macro),
        "strategy_config": strategy_config_dict(DEFAULT_STRATEGY_CONFIG),
        "holdout_backtest": experiment["holdout_backtest"],
        "strategy_results": experiment["strategy_results"],
        "walk_forward": experiment["walk_forward"],
        "feature_dim": int(FEATURE_DIM),
        "window_size": int(WINDOW_SIZE),
        "quantum": qp.get_eigenvalue_summary(),
        "model_comparison": experiment["model_comparison"],
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
    p.add_argument("--compare-existing", action="store_true",
                   help="Train only the small Qiskit classifier and compare it with saved ensemble artifacts")
    a = p.parse_args()
    train(symbol=a.symbol, total_bars=a.bars, quick=a.quick,
          compare_existing=a.compare_existing)
