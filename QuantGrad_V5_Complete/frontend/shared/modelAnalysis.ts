export type ModelDetailKind = "metric" | "score" | "field";

export type ModelDetail = {
  key: string;
  label: string;
  value: string;
  numericValue?: number;
  kind: ModelDetailKind;
};

export type BacktestMetrics = {
  evaluation?: string;
  exitPolicy?: string;
  profitFactor: number;
  winRate: number;
  maxDrawdown: number;
  totalTrades: number;
  netReturn: number;
  averageTradeReturn: number;
  averageWin: number;
  averageLoss: number;
  feeRatePerSide: number;
  slippageRatePerSide: number;
  roundTripCost: number;
  evaluatedWindows: number;
  directionalAccuracy: number;
  precisionMacro: number;
  recallMacro: number;
  f1Macro: number;
  balancedAccuracy: number;
  averageHoldingBars?: number;
  timeInMarket?: number;
};

export type ClassificationMetrics = {
  status: "trained" | "evaluated";
  accuracy: number;
  precisionMacro: number;
  recallMacro: number;
  f1Macro: number;
  balancedAccuracy: number;
  logLoss: number;
  nQuBits?: number;
  simulator?: string;
  trainedEpochs?: number;
};

export type ModelComparison = {
  evaluation: string;
  targetDefinition: string;
  trainSamples: number;
  testSamples: number;
  quantum: ClassificationMetrics;
  ensemble: ClassificationMetrics;
};

export type StrategyTrade = {
  tradeNumber: number;
  side: "LONG" | "SHORT";
  entryTime: string;
  exitTime: string;
  entryPrice: number;
  exitPrice: number;
  exitReason: string;
  holdingBars: number;
  grossReturn: number;
  netReturn: number;
};

export type StrategyResult = {
  modelId: string;
  label: string;
  confidenceThreshold: number;
  classification: ClassificationMetrics;
  backtest: BacktestMetrics;
  trades: StrategyTrade[];
};

export type WalkForwardFoldResult = {
  fold: number;
  trainSamples: number;
  validationSamples: number;
  testSamples: number;
  results: Array<{
    modelId: string;
    threshold: number;
    backtest: Partial<BacktestMetrics>;
  }>;
};

export type WalkForwardAggregate = {
  modelId: string;
  folds: number;
  totalTrades: number;
  netReturn: number;
  profitFactor: number;
  winRate: number;
  maxDrawdown: number;
  averageTradeReturn: number;
};

export type WalkForwardResults = {
  method: string;
  foldCount: number;
  folds: WalkForwardFoldResult[];
  aggregate: WalkForwardAggregate[];
};

export function modelFieldLabel(key: string) {
  return key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, character => character.toUpperCase());
}
