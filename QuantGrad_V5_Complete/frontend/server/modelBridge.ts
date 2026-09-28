import { modelFieldLabel, type BacktestMetrics, type ClassificationMetrics, type ModelComparison, type ModelDetail, type StrategyResult, type StrategyTrade, type WalkForwardResults } from "../shared/modelAnalysis";

export type LocalModelAnalysis = {
  state: "ready" | "artifacts_missing" | "unavailable" | "analysis_failed";
  available: boolean;
  source: string;
  checkedAt: number;
  reason?: string;
  missingArtifacts: string[];
  symbol?: string;
  signalLabel?: string;
  confidence?: number;
  riskLevel?: string;
  trend?: string;
  structure?: string;
  adx?: number;
  modelDetails: ModelDetail[];
  backtest?: BacktestMetrics;
  modelComparison?: ModelComparison;
  strategyResults?: StrategyResult[];
  walkForward?: WalkForwardResults;
  macroFallbackInputs: string[];
};

type UnknownRecord = Record<string, unknown>;

const DEFAULT_LOCAL_MODEL_URL = "http://127.0.0.1:8000";

function isRecord(value: unknown): value is UnknownRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function backtestMetrics(value: unknown): BacktestMetrics | undefined {
  if (!isRecord(value)) return undefined;
  const keys = ["profit_factor", "win_rate", "max_drawdown", "total_trades", "net_return", "average_trade_return", "average_win", "average_loss", "fee_rate_per_side", "slippage_rate_per_side", "round_trip_cost", "evaluated_windows", "directional_accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy"] as const;
  const values = keys.map(key => numberValue(value[key]));
  if (values.some(metric => metric === undefined)) return undefined;
  return {
    evaluation: stringValue(value.evaluation),
    profitFactor: values[0]!, winRate: values[1]!, maxDrawdown: values[2]!, totalTrades: values[3]!,
    netReturn: values[4]!, averageTradeReturn: values[5]!, averageWin: values[6]!, averageLoss: values[7]!,
    feeRatePerSide: values[8]!, slippageRatePerSide: values[9]!, roundTripCost: values[10]!,
    evaluatedWindows: values[11]!, directionalAccuracy: values[12]!, precisionMacro: values[13]!,
    recallMacro: values[14]!, f1Macro: values[15]!, balancedAccuracy: values[16]!,
    exitPolicy: stringValue(value.exit_policy),
    averageHoldingBars: numberValue(value.average_holding_bars),
    timeInMarket: numberValue(value.time_in_market),
  };
}

function classificationMetrics(value: unknown, expectedStatus: ClassificationMetrics["status"]): ClassificationMetrics | undefined {
  if (!isRecord(value)) return undefined;
  const accuracy = numberValue(value.accuracy);
  const precisionMacro = numberValue(value.precision_macro);
  const recallMacro = numberValue(value.recall_macro);
  const f1Macro = numberValue(value.f1_macro);
  const balancedAccuracy = numberValue(value.balanced_accuracy);
  const logLoss = numberValue(value.log_loss);
  if ([accuracy, precisionMacro, recallMacro, f1Macro, balancedAccuracy, logLoss].some(metric => metric === undefined)) return undefined;
  const status = value.status === "trained" || value.status === "evaluated" ? value.status : expectedStatus;
  return {
    status, accuracy: accuracy!, precisionMacro: precisionMacro!, recallMacro: recallMacro!,
    f1Macro: f1Macro!, balancedAccuracy: balancedAccuracy!, logLoss: logLoss!,
    nQuBits: numberValue(value.n_qubits), simulator: stringValue(value.simulator),
    trainedEpochs: numberValue(value.trained_epochs),
  };
}

function modelComparison(value: unknown): ModelComparison | undefined {
  if (!isRecord(value)) return undefined;
  const quantum = classificationMetrics(value.quantum, "trained");
  const ensemble = classificationMetrics(value.ensemble, "evaluated");
  const trainSamples = numberValue(value.train_samples);
  const testSamples = numberValue(value.test_samples);
  if (!quantum || !ensemble || trainSamples === undefined || testSamples === undefined) return undefined;
  return {
    evaluation: stringValue(value.evaluation) || "chronological_holdout",
    targetDefinition: stringValue(value.target_definition) || "3-bar forward direction",
    trainSamples, testSamples, quantum, ensemble,
  };
}

function strategyTrade(value: unknown): StrategyTrade | undefined {
  if (!isRecord(value)) return undefined;
  const tradeNumber = numberValue(value.trade_number);
  const side = value.side === "LONG" || value.side === "SHORT" ? value.side : undefined;
  const entryTime = stringValue(value.entry_time);
  const exitTime = stringValue(value.exit_time);
  const entryPrice = numberValue(value.entry_price);
  const exitPrice = numberValue(value.exit_price);
  const exitReason = stringValue(value.exit_reason);
  const holdingBars = numberValue(value.holding_bars);
  const grossReturn = numberValue(value.gross_return);
  const netReturn = numberValue(value.net_return);
  if ([tradeNumber, entryPrice, exitPrice, holdingBars, grossReturn, netReturn].some(item => item === undefined) || !side || !entryTime || !exitTime || !exitReason) return undefined;
  return { tradeNumber: tradeNumber!, side, entryTime, exitTime, entryPrice: entryPrice!, exitPrice: exitPrice!, exitReason, holdingBars: holdingBars!, grossReturn: grossReturn!, netReturn: netReturn! };
}

function strategyResults(value: unknown): StrategyResult[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.flatMap(item => {
    if (!isRecord(item)) return [];
    const modelId = stringValue(item.model_id);
    const label = stringValue(item.label);
    const confidenceThreshold = numberValue(item.confidence_threshold);
    const classification = classificationMetrics(item.classification, "evaluated");
    const backtestPayload = isRecord(item.backtest) ? item.backtest : undefined;
    const backtest = backtestMetrics(backtestPayload);
    if (!modelId || !label || confidenceThreshold === undefined || !classification || !backtest) return [];
    const trades = Array.isArray(backtestPayload?.trades)
      ? backtestPayload.trades.flatMap((trade: unknown) => {
          const parsed = strategyTrade(trade);
          return parsed ? [parsed] : [];
        })
      : [];
    return [{ modelId, label, confidenceThreshold, classification, backtest, trades }];
  });
}

function walkForwardResults(value: unknown): WalkForwardResults | undefined {
  if (!isRecord(value) || !isRecord(value.aggregate) || !Array.isArray(value.folds)) return undefined;
  const aggregate = Object.entries(value.aggregate).flatMap(([modelId, row]) => {
    if (!isRecord(row)) return [];
    const folds = numberValue(row.folds);
    const totalTrades = numberValue(row.total_trades);
    const netReturn = numberValue(row.net_return);
    const profitFactor = numberValue(row.profit_factor);
    const winRate = numberValue(row.win_rate);
    const maxDrawdown = numberValue(row.max_drawdown);
    const averageTradeReturn = numberValue(row.average_trade_return);
    if ([folds, totalTrades, netReturn, profitFactor, winRate, maxDrawdown, averageTradeReturn].some(item => item === undefined)) return [];
    return [{ modelId, folds: folds!, totalTrades: totalTrades!, netReturn: netReturn!, profitFactor: profitFactor!, winRate: winRate!, maxDrawdown: maxDrawdown!, averageTradeReturn: averageTradeReturn! }];
  });
  const folds = value.folds.flatMap(item => {
    if (!isRecord(item) || !isRecord(item.results)) return [];
    const fold = numberValue(item.fold);
    const trainSamples = numberValue(item.train_samples);
    const validationSamples = numberValue(item.validation_samples);
    const testSamples = numberValue(item.test_samples);
    if ([fold, trainSamples, validationSamples, testSamples].some(count => count === undefined)) return [];
    const results = Object.entries(item.results).flatMap(([modelId, result]) => {
      if (!isRecord(result) || !isRecord(result.backtest)) return [];
      const threshold = numberValue(result.threshold);
      const netReturn = numberValue(result.backtest.net_return);
      const profitFactor = numberValue(result.backtest.profit_factor);
      const totalTrades = numberValue(result.backtest.total_trades);
      const maxDrawdown = numberValue(result.backtest.max_drawdown);
      if ([threshold, netReturn, profitFactor, totalTrades, maxDrawdown].some(metric => metric === undefined)) return [];
      return [{ modelId, threshold: threshold!, backtest: { netReturn: netReturn!, profitFactor: profitFactor!, totalTrades: totalTrades!, maxDrawdown: maxDrawdown! } }];
    });
    return [{ fold: fold!, trainSamples: trainSamples!, validationSamples: validationSamples!, testSamples: testSamples!, results }];
  });
  const foldCount = numberValue(value.fold_count);
  if (foldCount === undefined) return undefined;
  return { method: stringValue(value.method) || "expanding walk-forward", foldCount, folds, aggregate };
}

function macroFallbackInputs(value: unknown): string[] {
  if (!isRecord(value) || !Array.isArray(value.fallback_inputs)) return [];
  return value.fallback_inputs.filter((item): item is string => typeof item === "string");
}

const PRIMARY_RESULT_FIELDS = new Set([
  "signal_label", "confidence", "risk_level", "trend", "structure", "adx",
]);

function detailKind(key: string, numericValue?: number): ModelDetail["kind"] {
  if (numericValue === undefined) return "field";
  return /(score|confidence|probability|prob|weight|importance|ratio|pct|percent)/i.test(key) ? "score" : "metric";
}

function collectModelDetails(result: UnknownRecord) {
  const details: ModelDetail[] = [];
  const addDetail = (key: string, value: string, numericValue?: number) => {
    if (details.length >= 36 || PRIMARY_RESULT_FIELDS.has(key)) return;
    details.push({ key, label: modelFieldLabel(key), value, numericValue, kind: detailKind(key, numericValue) });
  };
  const visit = (value: unknown, key: string, depth: number) => {
    if (details.length >= 36 || PRIMARY_RESULT_FIELDS.has(key)) return;
    const numericValue = numberValue(value);
    if (numericValue !== undefined) { addDetail(key, String(numericValue), numericValue); return; }
    const text = stringValue(value);
    if (text) { addDetail(key, text.slice(0, 160)); return; }
    if (typeof value === "boolean") { addDetail(key, value ? "True" : "False"); return; }
    if (Array.isArray(value) && value.length <= 8 && value.every(item => numberValue(item) !== undefined)) {
      value.forEach((item, index) => visit(item, `${key}.${index + 1}`, depth + 1));
      return;
    }
    if (isRecord(value) && depth < 2) {
      Object.entries(value).forEach(([childKey, childValue]) => visit(childValue, `${key}.${childKey}`, depth + 1));
    }
  };

  Object.entries(result).forEach(([key, value]) => visit(value, key, 0));
  return details;
}

function localModelBaseUrl() {
  return (process.env.QUANTGRAD_MODEL_API_URL || DEFAULT_LOCAL_MODEL_URL).replace(/\/+$/, "");
}

async function responseDetail(response: Response) {
  try {
    const payload: unknown = await response.json();
    return isRecord(payload) ? stringValue(payload.detail) : undefined;
  } catch {
    return undefined;
  }
}

function unavailable(state: LocalModelAnalysis["state"], reason: string, missingArtifacts: string[] = []): LocalModelAnalysis {
  return {
    state,
    available: false,
    source: "Local QuantGrad Python API",
    checkedAt: Date.now(),
    reason,
    missingArtifacts,
    modelDetails: [],
    macroFallbackInputs: [],
  };
}

export async function getLocalModelAnalysis(symbol: string): Promise<LocalModelAnalysis> {
  const baseUrl = localModelBaseUrl();
  let statusResponse: Response;

  try {
    statusResponse = await fetch(`${baseUrl}/api/status`, { signal: AbortSignal.timeout(8_000) });
  } catch {
    return unavailable("unavailable", "Local Python model API is not running. Start run_backend.bat, then refresh this page.");
  }

  if (!statusResponse.ok) {
    return unavailable("unavailable", `Local Python model status returned ${statusResponse.status}.`);
  }

  let statusPayload: unknown;
  try {
    statusPayload = await statusResponse.json();
  } catch {
    return unavailable("unavailable", "Local Python model status returned an unreadable response.");
  }

  const status = isRecord(statusPayload) ? statusPayload : {};
  const missingArtifacts = Array.isArray(status.missing) ? status.missing.filter((item): item is string => typeof item === "string") : [];
  if (status.status !== "ready" || status.artifacts_ready !== true) {
    return unavailable(
      "artifacts_missing",
      missingArtifacts.length ? `Model artifacts are missing: ${missingArtifacts.join(", ")}. Reuse or copy your trained artifacts into backend/artifacts/.` : "The local Python model is not ready.",
      missingArtifacts,
    );
  }

  let analysisResponse: Response;
  try {
    analysisResponse = await fetch(`${baseUrl}/api/analyze?symbol=${encodeURIComponent(symbol)}`, { signal: AbortSignal.timeout(90_000) });
  } catch {
    return unavailable("analysis_failed", "The local model did not finish analysis in time. Check the Python terminal for details and try again.");
  }

  if (!analysisResponse.ok) {
    const detail = await responseDetail(analysisResponse);
    const state = analysisResponse.status === 503 ? "artifacts_missing" : "analysis_failed";
    return unavailable(state, detail || `Local model analysis returned ${analysisResponse.status}.`);
  }

  let analysisPayload: unknown;
  try {
    analysisPayload = await analysisResponse.json();
  } catch {
    return unavailable("analysis_failed", "The local model returned an unreadable analysis response.");
  }

  const body = isRecord(analysisPayload) ? analysisPayload : {};
  const result = isRecord(body.result) ? body.result : {};
  const trainingReport = isRecord(body.training_report) ? body.training_report : {};
  const confidence = numberValue(result.confidence);
  const signalLabel = stringValue(result.signal_label);
  if (confidence === undefined || !signalLabel) {
    return unavailable("analysis_failed", "The local model response did not include a usable signal result.");
  }

  return {
    state: "ready",
    available: true,
    source: "Local QuantGrad Python API",
    checkedAt: Date.now(),
    missingArtifacts: [],
    symbol: stringValue(body.symbol) || symbol,
    signalLabel,
    confidence: Math.max(0, Math.min(1, confidence)),
    riskLevel: stringValue(result.risk_level),
    trend: stringValue(result.trend),
    structure: stringValue(result.structure),
    adx: numberValue(result.adx),
    modelDetails: collectModelDetails(result),
    backtest: backtestMetrics(body.backtest),
    modelComparison: modelComparison(trainingReport.model_comparison),
    strategyResults: strategyResults(trainingReport.strategy_results),
    walkForward: walkForwardResults(trainingReport.walk_forward),
    macroFallbackInputs: macroFallbackInputs(trainingReport.macro_status),
  };
}
