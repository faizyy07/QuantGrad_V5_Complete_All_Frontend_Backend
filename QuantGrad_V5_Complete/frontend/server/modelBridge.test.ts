import { afterEach, describe, expect, it, vi } from "vitest";
import { getLocalModelAnalysis } from "./modelBridge";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("local Python model bridge", () => {
  it("returns a clear artifacts-missing state without attempting inference", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: "artifacts_missing", artifacts_ready: false, missing: ["entry_model.keras", "scaler.pkl"] }) });
    vi.stubGlobal("fetch", fetchMock);

    await expect(getLocalModelAnalysis("BTCUSDT")).resolves.toMatchObject({
      state: "artifacts_missing",
      available: false,
      missingArtifacts: ["entry_model.keras", "scaler.pkl"],
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("maps a ready Python inference result into the typed terminal ledger model", async () => {
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "ready", artifacts_ready: true, missing: [] }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({
        symbol: "ETHUSDT",
        result: { signal_label: "STRONG BUY", confidence: 0.64, risk_level: "CONFIRMED", trend: "BULLISH", structure: "HIGHER LOW", adx: 27.4, factor_scores: { momentum: 0.72, volatility: 0.41 }, atr: 1832.4 },
        backtest: { evaluation: "chronological_holdout", profit_factor: 1.31, win_rate: 0.58, max_drawdown: 0.12, total_trades: 24, net_return: 0.18, average_trade_return: 0.01, average_win: 0.03, average_loss: 0.02, fee_rate_per_side: 0.0005, slippage_rate_per_side: 0.0002, round_trip_cost: 0.0014, evaluated_windows: 80, directional_accuracy: 0.56, precision_macro: 0.55, recall_macro: 0.53, f1_macro: 0.54, balanced_accuracy: 0.53 },
        training_report: { model_comparison: {
          evaluation: "chronological_holdout", target_definition: "3-bar forward direction", train_samples: 120, test_samples: 30,
          quantum: { status: "trained", simulator: "Qiskit Statevector", n_qubits: 4, trained_epochs: 4, accuracy: 0.52, precision_macro: 0.5, recall_macro: 0.48, f1_macro: 0.49, balanced_accuracy: 0.48, log_loss: 1.42 },
          ensemble: { status: "evaluated", accuracy: 0.61, precision_macro: 0.59, recall_macro: 0.58, f1_macro: 0.58, balanced_accuracy: 0.58, log_loss: 1.1 },
        }, macro_status: { fetched_at: "2026-09-29T00:00:00Z", fallback_inputs: ["DXY", "gold", "Fed rate"], warning: "Some macro inputs used provider fallbacks." }, strategy_results: [{
          model_id: "random_forest", label: "Random Forest · bagging", confidence_threshold: 0.45,
          classification: { accuracy: 0.61, precision_macro: 0.59, recall_macro: 0.58, f1_macro: 0.58, balanced_accuracy: 0.58, log_loss: 1.1 },
          backtest: { evaluation: "chronological_holdout", exit_policy: "ATR barriers", atr_period: 14, stop_atr: 1.25, target_atr: 1.75, max_holding_bars: 24, fee_rate_per_side: 0.0005, slippage_rate_per_side: 0.0002, round_trip_cost: 0.0014, total_trades: 1, net_return: 0.01, max_drawdown: 0.01, win_rate: 1, profit_factor: 100, average_trade_return: 0.01, average_win: 0.01, average_loss: 0, average_holding_bars: 3, time_in_market: 0.1, evaluated_windows: 30, directional_accuracy: 0.61, precision_macro: 0.59, recall_macro: 0.58, f1_macro: 0.58, balanced_accuracy: 0.58, trades: [{ trade_number: 1, side: "LONG", entry_time: "2026-01-01T01:00:00", exit_time: "2026-01-01T04:00:00", entry_price: 100, exit_price: 101, exit_reason: "take_profit", holding_bars: 3, gross_return: 0.01, net_return: 0.01 }] },
        }], walk_forward: {
          method: "expanding_train_validation_test_with_24-bar_purge", fold_count: 1,
          folds: [{ fold: 1, train_samples: 100, validation_samples: 20, test_samples: 20, results: { random_forest: { threshold: 0.45, backtest: { net_return: 0.01, profit_factor: 1.2, total_trades: 4, max_drawdown: 0.02 } } } }],
          aggregate: { random_forest: { folds: 1, total_trades: 4, net_return: 0.01, profit_factor: 1.2, win_rate: 0.5, max_drawdown: 0.02, average_trade_return: 0.0025 } },
        } },
      }) }));

    const result = await getLocalModelAnalysis("ETHUSDT");
    expect(result).toMatchObject({
      state: "ready",
      available: true,
      symbol: "ETHUSDT",
      signalLabel: "STRONG BUY",
      confidence: 0.64,
      riskLevel: "CONFIRMED",
      trend: "BULLISH",
      structure: "HIGHER LOW",
      adx: 27.4,
      backtest: { evaluation: "chronological_holdout", profitFactor: 1.31, winRate: 0.58, maxDrawdown: 0.12, totalTrades: 24, netReturn: 0.18, averageTradeReturn: 0.01, averageWin: 0.03, averageLoss: 0.02, feeRatePerSide: 0.0005, slippageRatePerSide: 0.0002, roundTripCost: 0.0014, evaluatedWindows: 80, directionalAccuracy: 0.56, precisionMacro: 0.55, recallMacro: 0.53, f1Macro: 0.54, balancedAccuracy: 0.53 },
      modelComparison: {
        evaluation: "chronological_holdout",
        targetDefinition: "3-bar forward direction",
        trainSamples: 120,
        testSamples: 30,
        quantum: { status: "trained", simulator: "Qiskit Statevector", nQuBits: 4, trainedEpochs: 4, accuracy: 0.52, precisionMacro: 0.5, recallMacro: 0.48, f1Macro: 0.49, balancedAccuracy: 0.48, logLoss: 1.42 },
        ensemble: { status: "evaluated", accuracy: 0.61, precisionMacro: 0.59, recallMacro: 0.58, f1Macro: 0.58, balancedAccuracy: 0.58, logLoss: 1.1 },
      },
      strategyResults: [{
        modelId: "random_forest",
        label: "Random Forest · bagging",
        confidenceThreshold: 0.45,
        classification: { status: "evaluated", accuracy: 0.61, precisionMacro: 0.59, recallMacro: 0.58, f1Macro: 0.58, balancedAccuracy: 0.58, logLoss: 1.1 },
        backtest: { evaluation: "chronological_holdout", exitPolicy: "ATR barriers", profitFactor: 100, winRate: 1, maxDrawdown: 0.01, totalTrades: 1, netReturn: 0.01, averageTradeReturn: 0.01, averageWin: 0.01, averageLoss: 0, feeRatePerSide: 0.0005, slippageRatePerSide: 0.0002, roundTripCost: 0.0014, evaluatedWindows: 30, directionalAccuracy: 0.61, precisionMacro: 0.59, recallMacro: 0.58, f1Macro: 0.58, balancedAccuracy: 0.58, averageHoldingBars: 3, timeInMarket: 0.1 },
        trades: [{ tradeNumber: 1, side: "LONG", entryTime: "2026-01-01T01:00:00", exitTime: "2026-01-01T04:00:00", entryPrice: 100, exitPrice: 101, exitReason: "take_profit", holdingBars: 3, grossReturn: 0.01, netReturn: 0.01 }],
      }],
      macroFallbackInputs: ["DXY", "gold", "Fed rate"],
      walkForward: {
        method: "expanding_train_validation_test_with_24-bar_purge",
        foldCount: 1,
        folds: [{ fold: 1, trainSamples: 100, validationSamples: 20, testSamples: 20, results: [{ modelId: "random_forest", threshold: 0.45, backtest: { netReturn: 0.01, profitFactor: 1.2, totalTrades: 4, maxDrawdown: 0.02 } }] }],
        aggregate: [{ modelId: "random_forest", folds: 1, totalTrades: 4, netReturn: 0.01, profitFactor: 1.2, winRate: 0.5, maxDrawdown: 0.02, averageTradeReturn: 0.0025 }],
      },
    });
    expect(result.modelDetails).toEqual(expect.arrayContaining([
      expect.objectContaining({ key: "factor_scores.momentum", numericValue: 0.72, kind: "score" }),
      expect.objectContaining({ key: "factor_scores.volatility", numericValue: 0.41, kind: "score" }),
      expect.objectContaining({ key: "atr", numericValue: 1832.4, kind: "metric" }),
    ]));
  });

  it("returns an intentional offline state when the Python server cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("connect ECONNREFUSED")));
    await expect(getLocalModelAnalysis("BTCUSDT")).resolves.toMatchObject({ state: "unavailable", available: false });
  });
});
