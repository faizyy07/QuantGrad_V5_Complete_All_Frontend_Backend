import unittest

import numpy as np
import pandas as pd

from strategy_experiments import (
    StrategyConfig,
    build_barrier_labels,
    evaluate_walk_forward_baselines,
    simulate_variable_exit,
    train_tabular_baselines,
)


class StrategyExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig(
            atr_period=1,
            stop_atr=1.0,
            target_atr=1.5,
            max_holding_bars=3,
            min_entry_return=0.0,
            strong_entry_return=0.005,
            fee_rate_per_side=0.0,
            slippage_rate_per_side=0.0,
        )
        self.index = pd.date_range("2026-01-01", periods=12, freq="h")
        self.bars = pd.DataFrame({
            "open": [100.0] * 12,
            "high": [100.5, 101, 102, 101, 101, 101, 101, 101, 101, 101, 101, 101],
            "low": [99.5, 99.5, 100, 99, 99, 99, 99, 99, 99, 99, 99, 99],
            "close": [100.0] * 12,
        }, index=self.index)

    def test_take_profit_and_model_exit(self):
        labels = build_barrier_labels(self.bars, self.config)
        target_result = simulate_variable_exit(
            np.array([1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
            self.bars,
            config=self.config,
        )
        model_exit_result = simulate_variable_exit(
            np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
            self.bars,
            config=self.config,
        )
        self.assertEqual(labels.shape, (len(self.bars),))
        self.assertEqual(target_result["trades"][0]["exit_reason"], "take_profit")
        self.assertEqual(model_exit_result["trades"][0]["exit_reason"], "model_exit")

    def test_stop_wins_ambiguous_same_bar_and_trades_do_not_overlap(self):
        ambiguous = self.bars.copy()
        ambiguous.loc[self.index[1], "high"] = 103.0
        ambiguous.loc[self.index[1], "low"] = 97.0
        result = simulate_variable_exit(
            np.array([1] + [2] * 11), ambiguous, config=self.config
        )
        self.assertEqual(result["trades"][0]["exit_reason"], "stop_and_target_same_bar_stop_first")
        self.assertGreaterEqual(result["trades"][0]["holding_bars"], 1)

        continuous_entries = simulate_variable_exit(
            np.array([1] * len(self.bars)), self.bars, config=self.config
        )
        self.assertLessEqual(continuous_entries["time_in_market"], 1.0)
        for first, second in zip(continuous_entries["trades"], continuous_entries["trades"][1:]):
            self.assertGreaterEqual(second["entry_index"], first["exit_index"])

    def test_tabular_baselines_and_walk_forward(self):
        rng = np.random.default_rng(7)
        sample_count = 300
        index = pd.date_range("2026-01-01", periods=sample_count, freq="h")
        direction = rng.integers(0, 3, size=sample_count, dtype=np.int32)
        features = rng.normal(size=(sample_count, 8)).astype(np.float32)
        features[:, 0] += direction
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, size=sample_count)))
        bars = pd.DataFrame({
            "open": close,
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
        }, index=index)

        estimators = train_tabular_baselines(features[:180], direction[:180])
        self.assertEqual(set(estimators), {"random_forest", "hist_gradient_boosting"})
        result = evaluate_walk_forward_baselines(
            features, direction, bars, config=self.config, folds=3
        )
        self.assertEqual(result["fold_count"], 3)
        self.assertEqual(set(result["aggregate"]), {"random_forest", "hist_gradient_boosting", "tree_stacker"})


if __name__ == "__main__":
    unittest.main()
