"""Shared triple-barrier labels, variable-exit backtests, and tabular baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, precision_score, recall_score


@dataclass(frozen=True)
class StrategyConfig:
    atr_period: int = 14
    stop_atr: float = 1.25
    target_atr: float = 1.75
    max_holding_bars: int = 24
    min_entry_return: float = 0.0015
    strong_entry_return: float = 0.005
    fee_rate_per_side: float = 0.0005
    slippage_rate_per_side: float = 0.0002
    confidence_thresholds: tuple[float, ...] = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.fee_rate_per_side + self.slippage_rate_per_side)


DEFAULT_STRATEGY_CONFIG = StrategyConfig()


def average_true_range(bars: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    close = bars["close"].to_numpy(dtype=np.float64)
    previous_close = np.r_[close[0], close[:-1]]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)))
    return pd.Series(true_range).rolling(period, min_periods=1).mean().to_numpy(dtype=np.float64)


def _side_for_signal(signal: int) -> int:
    if signal in (0, 1):
        return 1
    if signal in (3, 4):
        return -1
    return 0


def _resolve_trade(
    bars: pd.DataFrame,
    signal_index: int,
    side: int,
    atr_values: np.ndarray,
    config: StrategyConfig,
    model_signals: np.ndarray | None = None,
) -> dict[str, Any] | None:
    entry_index = signal_index + 1
    if entry_index >= len(bars) or not np.isfinite(atr_values[signal_index]) or atr_values[signal_index] <= 0:
        return None

    entry_price = float(bars["open"].iloc[entry_index])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    atr = float(atr_values[signal_index])
    stop_price = entry_price - side * config.stop_atr * atr
    target_price = entry_price + side * config.target_atr * atr
    final_index = min(entry_index + config.max_holding_bars - 1, len(bars) - 1)

    for bar_index in range(entry_index, final_index + 1):
        high = float(bars["high"].iloc[bar_index])
        low = float(bars["low"].iloc[bar_index])
        stop_hit = low <= stop_price if side > 0 else high >= stop_price
        target_hit = high >= target_price if side > 0 else low <= target_price
        if stop_hit and target_hit:
            exit_index, exit_price, exit_reason = bar_index, stop_price, "stop_and_target_same_bar_stop_first"
            break
        if stop_hit:
            exit_index, exit_price, exit_reason = bar_index, stop_price, "stop_loss"
            break
        if target_hit:
            exit_index, exit_price, exit_reason = bar_index, target_price, "take_profit"
            break

        if model_signals is not None and bar_index < len(bars) - 1:
            if _side_for_signal(int(model_signals[bar_index])) != side:
                exit_index = bar_index + 1
                exit_price = float(bars["open"].iloc[exit_index])
                exit_reason = "model_exit"
                break
    else:
        exit_index = final_index
        exit_price = float(bars["close"].iloc[exit_index])
        exit_reason = "max_holding_time"

    if exit_index < entry_index or not np.isfinite(exit_price) or exit_price <= 0:
        return None
    gross_return = side * (exit_price - entry_price) / entry_price
    net_return = gross_return - config.round_trip_cost
    entry_time = bars.index[entry_index]
    exit_time = bars.index[exit_index]
    return {
        "entry_index": int(entry_index),
        "exit_index": int(exit_index),
        "entry_time": pd.Timestamp(entry_time).isoformat() if not isinstance(entry_time, str) else entry_time,
        "exit_time": pd.Timestamp(exit_time).isoformat() if not isinstance(exit_time, str) else exit_time,
        "side": "LONG" if side > 0 else "SHORT",
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "holding_bars": max(1, int(exit_index - entry_index)),
        "gross_return": float(gross_return),
        "net_return": float(net_return),
    }


def build_barrier_labels(
    bars: pd.DataFrame,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> np.ndarray:
    """Create five entry labels from next-open, ATR stop/target outcomes."""
    labels = np.full(len(bars), 2, dtype=np.int32)
    atr_values = average_true_range(bars, config.atr_period)
    for signal_index in range(len(bars) - 1):
        long_trade = _resolve_trade(bars, signal_index, 1, atr_values, config)
        short_trade = _resolve_trade(bars, signal_index, -1, atr_values, config)
        outcomes = [(1, long_trade), (-1, short_trade)]
        viable = [(side, trade) for side, trade in outcomes if trade and trade["net_return"] >= config.min_entry_return]
        if not viable:
            continue
        side, trade = max(viable, key=lambda item: item[1]["net_return"])
        strong = trade["net_return"] >= config.strong_entry_return
        labels[signal_index] = 0 if side > 0 and strong else 1 if side > 0 else 4 if strong else 3
    return labels


def directional_labels(entry_labels: np.ndarray) -> np.ndarray:
    entry_labels = np.asarray(entry_labels, dtype=np.int32)
    return np.where(np.isin(entry_labels, (0, 1)), 2,
                    np.where(np.isin(entry_labels, (3, 4)), 0, 1)).astype(np.int32)


def directional_probabilities_to_signals(probabilities: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Directional probabilities must have down, flat, and up columns")
    down, flat, up = probabilities[:, 0], probabilities[:, 1], probabilities[:, 2]
    signals = np.full(len(probabilities), 2, dtype=np.int32)
    long_wins = (up > flat) & (up > down) & (up >= threshold)
    short_wins = (down > flat) & (down > up) & (down >= threshold)
    signals[long_wins] = 1
    signals[short_wins] = 3
    return signals


def five_class_probabilities_to_direction(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.column_stack((
        probabilities[:, 3] + probabilities[:, 4],
        probabilities[:, 2],
        probabilities[:, 0] + probabilities[:, 1],
    ))


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, 1e-8, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.arange(3)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
    }


def simulate_variable_exit(
    signals: np.ndarray,
    bars: pd.DataFrame,
    true_direction_labels: np.ndarray | None = None,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    include_trades: bool = True,
    evaluation: str = "chronological_holdout",
) -> dict[str, Any]:
    """Single-position ATR barrier backtest with model reversal/HOLD exits."""
    signals = np.asarray(signals, dtype=np.int32)
    if len(signals) != len(bars):
        raise ValueError("Signals and OHLC bars must be aligned and have equal lengths")
    atr_values = average_true_range(bars, config.atr_period)
    trade_log: list[dict[str, Any]] = []
    equity = [1.0]
    next_signal_index = 0
    signal_index = 0
    while signal_index < len(signals) - 1:
        if signal_index < next_signal_index:
            signal_index = next_signal_index
            continue
        side = _side_for_signal(int(signals[signal_index]))
        if side == 0:
            signal_index += 1
            continue
        trade = _resolve_trade(bars, signal_index, side, atr_values, config, signals)
        if trade is None:
            signal_index += 1
            continue
        trade["trade_number"] = len(trade_log) + 1
        trade_log.append(trade)
        equity.append(max(0.0, equity[-1] * (1 + trade["net_return"])))
        next_signal_index = trade["exit_index"]
        signal_index = next_signal_index

    returns = np.asarray([trade["net_return"] for trade in trade_log], dtype=np.float64)
    positive = returns[returns > 0]
    negative = returns[returns <= 0]
    equity_values = np.asarray(equity, dtype=np.float64)
    peaks = np.maximum.accumulate(np.r_[1.0, equity_values])
    drawdowns = (peaks - np.r_[1.0, equity_values]) / np.maximum(peaks, 1e-12)
    result: dict[str, Any] = {
        "evaluation": evaluation,
        "exit_policy": "ATR stop/target; close on model HOLD/reversal; 24-bar maximum hold",
        "atr_period": config.atr_period,
        "stop_atr": config.stop_atr,
        "target_atr": config.target_atr,
        "max_holding_bars": config.max_holding_bars,
        "fee_rate_per_side": config.fee_rate_per_side,
        "slippage_rate_per_side": config.slippage_rate_per_side,
        "round_trip_cost": config.round_trip_cost,
        "total_trades": int(len(returns)),
        "net_return": float(equity_values[-1] - 1) if len(equity_values) else 0.0,
        "max_drawdown": float(np.max(drawdowns)) if len(drawdowns) else 0.0,
        "win_rate": float(np.mean(returns > 0)) if len(returns) else 0.0,
        "profit_factor": float(positive.sum() / (np.abs(negative).sum() + 1e-12)) if len(returns) else 0.0,
        "average_trade_return": float(np.mean(returns)) if len(returns) else 0.0,
        "average_win": float(np.mean(positive)) if len(positive) else 0.0,
        "average_loss": float(np.mean(np.abs(negative))) if len(negative) else 0.0,
        "average_holding_bars": float(np.mean([trade["holding_bars"] for trade in trade_log])) if trade_log else 0.0,
        "time_in_market": float(sum(trade["holding_bars"] for trade in trade_log) / max(len(bars), 1)),
        "evaluated_windows": int(len(signals)),
    }
    if true_direction_labels is not None:
        direction_predictions = np.where(np.isin(signals, (0, 1)), 2,
                                np.where(np.isin(signals, (3, 4)), 0, 1))
        y_true = np.asarray(true_direction_labels, dtype=np.int32)[:len(direction_predictions)]
        scores = classification_metrics(y_true, np.eye(3)[direction_predictions[:len(y_true)]])
        result.update({key: value for key, value in scores.items() if key != "log_loss"})
        result["directional_accuracy"] = scores["accuracy"]
        result["classification_metrics"] = {key: value for key, value in scores.items() if key != "log_loss"}
    if include_trades:
        result["trades"] = trade_log
    return result


def tune_confidence_threshold(
    probabilities: np.ndarray,
    decoder: Callable[[np.ndarray, float], np.ndarray],
    validation_bars: pd.DataFrame,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    minimum_trade_fraction: float = 0.02,
) -> tuple[float, dict[str, Any]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    minimum_trades = max(5, int(len(probabilities) * minimum_trade_fraction))
    candidates = []
    for threshold in config.confidence_thresholds:
        signals = decoder(probabilities, threshold)
        stats = simulate_variable_exit(signals, validation_bars, config=config, include_trades=False, evaluation="validation")
        if stats["total_trades"] >= minimum_trades:
            candidates.append((stats["profit_factor"], stats["net_return"], threshold, stats))
    if not candidates:
        signals = decoder(probabilities, 0.0)
        return 0.0, simulate_variable_exit(signals, validation_bars, config=config, include_trades=False, evaluation="validation")
    best = max(candidates, key=lambda row: (row[0], row[1]))
    return float(best[2]), best[3]


def train_tabular_baselines(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Tabular classifiers need at least two training classes")
    class_weights = {int(label): len(y) / (len(classes) * int(count)) for label, count in zip(classes, counts)}
    sample_weights = np.asarray([class_weights[int(label)] for label in y], dtype=np.float64)

    bagging = RandomForestClassifier(
        n_estimators=240,
        max_depth=8,
        min_samples_leaf=8,
        max_features=0.7,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    boosting = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )
    bagging.fit(X, y)
    boosting.fit(X, y, sample_weight=sample_weights)
    return {"random_forest": bagging, "hist_gradient_boosting": boosting}


def predict_aligned_probabilities(model: Any, X: np.ndarray, class_count: int = 3) -> np.ndarray:
    raw = np.asarray(model.predict_proba(np.asarray(X, dtype=np.float32)), dtype=np.float64)
    aligned = np.full((len(raw), class_count), 1e-8, dtype=np.float64)
    for source_index, label in enumerate(model.classes_):
        if 0 <= int(label) < class_count:
            aligned[:, int(label)] = raw[:, source_index]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def stack_features(random_forest_probabilities: np.ndarray, boosting_probabilities: np.ndarray) -> np.ndarray:
    return np.concatenate((random_forest_probabilities, boosting_probabilities), axis=1)


def train_oof_tree_stacker(
    X: np.ndarray,
    y: np.ndarray,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    seed: int = 42,
    folds: int = 3,
) -> tuple[LogisticRegression, int]:
    """Fit a logistic meta-model only on expanding-window out-of-fold probabilities."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    warmup = max(50, int(len(y) * 0.40))
    available = len(y) - warmup
    fold_size = available // folds
    oof_features = []
    oof_labels = []
    for fold_index in range(folds):
        train_end = warmup + fold_index * fold_size
        validation_start = train_end + config.max_holding_bars
        validation_end = warmup + (fold_index + 1) * fold_size
        if validation_end - validation_start < 10:
            continue
        estimators = train_tabular_baselines(X[:train_end], y[:train_end], seed + fold_index)
        random_forest_probabilities = predict_aligned_probabilities(
            estimators["random_forest"], X[validation_start:validation_end]
        )
        boosting_probabilities = predict_aligned_probabilities(
            estimators["hist_gradient_boosting"], X[validation_start:validation_end]
        )
        oof_features.append(stack_features(random_forest_probabilities, boosting_probabilities))
        oof_labels.append(y[validation_start:validation_end])
    if not oof_features:
        raise ValueError("Not enough observations to create out-of-fold stacker predictions")
    meta_model = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5, random_state=seed)
    meta_model.fit(np.concatenate(oof_features), np.concatenate(oof_labels))
    return meta_model, int(sum(len(labels) for labels in oof_labels))


def evaluate_walk_forward_baselines(
    X: np.ndarray,
    y: np.ndarray,
    bars: pd.DataFrame,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    folds: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Expanding-window folds; validation tunes the gate, next block is test."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    if len(X) != len(y) or len(y) != len(bars):
        raise ValueError("Walk-forward features, labels, and OHLC data must be aligned")
    fold_size = len(y) // 10
    validation_size = fold_size
    purge = config.max_holding_bars
    initial_train_end = len(y) - (folds + 2) * fold_size
    if fold_size < 10 or initial_train_end <= 0:
        raise ValueError("Not enough samples for the requested chronological walk-forward folds")

    summaries: dict[str, list[dict[str, Any]]] = {
        "random_forest": [], "hist_gradient_boosting": [], "tree_stacker": [],
    }
    fold_details = []
    for fold_index in range(folds):
        train_end = initial_train_end + fold_index * fold_size
        validation_start = train_end + purge
        validation_end = min(validation_start + validation_size, len(y))
        test_start = validation_end + purge
        test_end = min(test_start + fold_size, len(y))
        if test_end - test_start < 10:
            continue
        estimators = train_tabular_baselines(X[:train_end], y[:train_end], seed + fold_index)
        fold_detail: dict[str, Any] = {
            "fold": fold_index + 1,
            "train_samples": int(train_end),
            "validation_samples": int(validation_end - validation_start),
            "test_samples": int(test_end - test_start),
            "results": {},
        }
        for model_id, model in estimators.items():
            validation_probabilities = predict_aligned_probabilities(model, X[validation_start:validation_end])
            threshold, _ = tune_confidence_threshold(
                validation_probabilities,
                directional_probabilities_to_signals,
                bars.iloc[validation_start:validation_end],
                config,
            )
            test_probabilities = predict_aligned_probabilities(model, X[test_start:test_end])
            signals = directional_probabilities_to_signals(test_probabilities, threshold)
            backtest = simulate_variable_exit(
                signals, bars.iloc[test_start:test_end], y[test_start:test_end], config,
                evaluation="walk_forward_test",
            )
            summaries[model_id].append(backtest)
            fold_detail["results"][model_id] = {
                "threshold": threshold,
                "backtest": {key: value for key, value in backtest.items() if key != "trades"},
            }
        stacker, oof_samples = train_oof_tree_stacker(X[:train_end], y[:train_end], config, seed + fold_index)
        validation_rf = predict_aligned_probabilities(estimators["random_forest"], X[validation_start:validation_end])
        validation_hgb = predict_aligned_probabilities(estimators["hist_gradient_boosting"], X[validation_start:validation_end])
        validation_stack = predict_aligned_probabilities(stacker, stack_features(validation_rf, validation_hgb))
        stack_threshold, _ = tune_confidence_threshold(
            validation_stack, directional_probabilities_to_signals,
            bars.iloc[validation_start:validation_end], config,
        )
        test_rf = predict_aligned_probabilities(estimators["random_forest"], X[test_start:test_end])
        test_hgb = predict_aligned_probabilities(estimators["hist_gradient_boosting"], X[test_start:test_end])
        test_stack = predict_aligned_probabilities(stacker, stack_features(test_rf, test_hgb))
        stacked_signals = directional_probabilities_to_signals(test_stack, stack_threshold)
        stacked_backtest = simulate_variable_exit(
            stacked_signals, bars.iloc[test_start:test_end], y[test_start:test_end], config,
            evaluation="walk_forward_test",
        )
        summaries["tree_stacker"].append(stacked_backtest)
        fold_detail["results"]["tree_stacker"] = {
            "threshold": stack_threshold,
            "oof_train_samples": oof_samples,
            "backtest": {key: value for key, value in stacked_backtest.items() if key != "trades"},
        }
        fold_details.append(fold_detail)

    aggregate: dict[str, Any] = {}
    for model_id, fold_results in summaries.items():
        returns = [trade["net_return"] for fold in fold_results for trade in fold.get("trades", [])]
        returns_array = np.asarray(returns, dtype=np.float64)
        equity = np.cumprod(1 + returns_array) if len(returns_array) else np.array([1.0])
        peaks = np.maximum.accumulate(np.r_[1.0, equity])
        drawdowns = (peaks - np.r_[1.0, equity]) / np.maximum(peaks, 1e-12)
        gains = returns_array[returns_array > 0]
        losses = returns_array[returns_array <= 0]
        aggregate[model_id] = {
            "folds": len(fold_results),
            "total_trades": int(len(returns_array)),
            "net_return": float(equity[-1] - 1),
            "profit_factor": float(gains.sum() / (np.abs(losses).sum() + 1e-12)) if len(returns_array) else 0.0,
            "win_rate": float(np.mean(returns_array > 0)) if len(returns_array) else 0.0,
            "max_drawdown": float(np.max(drawdowns)) if len(drawdowns) else 0.0,
            "average_trade_return": float(np.mean(returns_array)) if len(returns_array) else 0.0,
        }
    return {
        "method": "expanding_train_validation_test_with_24-bar_purge",
        "fold_count": len(fold_details),
        "folds": fold_details,
        "aggregate": aggregate,
    }


def strategy_config_dict(config: StrategyConfig = DEFAULT_STRATEGY_CONFIG) -> dict[str, Any]:
    values = asdict(config)
    values["confidence_thresholds"] = list(config.confidence_thresholds)
    values["round_trip_cost"] = config.round_trip_cost
    return values
