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
      .mockResolvedValueOnce({ ok: true, json: async () => ({ symbol: "ETHUSDT", result: { signal_label: "STRONG BUY", confidence: 0.64, risk_level: "CONFIRMED", trend: "BULLISH", structure: "HIGHER LOW", adx: 27.4, factor_scores: { momentum: 0.72, volatility: 0.41 }, atr: 1832.4 } }) }));

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
