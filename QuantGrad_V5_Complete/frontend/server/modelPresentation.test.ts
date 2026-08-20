import { describe, expect, it } from "vitest";
import { formatModelDetail, getDetailBarMagnitude, modelScorePercent } from "../client/src/lib/modelPresentation";

describe("model presentation helpers", () => {
  it("normalizes confidence scores to the decision-grade percentage scale", () => {
    expect(modelScorePercent(0.643)).toBe(64);
    expect(modelScorePercent(4)).toBe(100);
    expect(modelScorePercent(undefined)).toBeUndefined();
  });

  it("formats fractional calculation scores and exposes a bounded visual magnitude", () => {
    const detail = { key: "factorScores.momentum", label: "Factor Scores Momentum", value: "0.625", numericValue: 0.625, kind: "score" as const };
    expect(formatModelDetail(detail)).toBe("62.5%");
    expect(getDetailBarMagnitude(detail)).toBe(63);
  });

  it("does not invent a bar scale for unbounded metrics", () => {
    const detail = { key: "atr", label: "ATR", value: "1832.4", numericValue: 1832.4, kind: "metric" as const };
    expect(formatModelDetail(detail)).toBe("1,832.4");
    expect(getDetailBarMagnitude(detail)).toBeUndefined();
  });
});
