import { modelFieldLabel, type ModelDetail } from "../shared/modelAnalysis";

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
  };
}
