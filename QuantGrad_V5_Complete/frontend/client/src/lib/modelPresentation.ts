import type { ModelDetail } from "@shared/modelAnalysis";

export function modelScorePercent(value?: number) {
  if (!Number.isFinite(value)) return undefined;
  return Math.round(Math.max(0, Math.min(1, value ?? 0)) * 100);
}

export function formatModelDetail(detail: ModelDetail) {
  if (detail.numericValue === undefined) return detail.value;

  const value = detail.numericValue;
  const isFractionalScore = detail.kind === "score" && Math.abs(value) <= 1;
  if (isFractionalScore) return `${(value * 100).toFixed(1)}%`;

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: Math.abs(value) < 10 ? 4 : 2,
  }).format(value);
}

export function getDetailBarMagnitude(detail: ModelDetail) {
  const value = detail.numericValue;
  if (!Number.isFinite(value) || (value ?? 0) < 0) return undefined;
  if ((value ?? 0) <= 1) return Math.round((value ?? 0) * 100);
  if ((value ?? 0) <= 100) return Math.round(value ?? 0);
  return undefined;
}
