import { afterEach, describe, expect, it, vi } from "vitest";
import { clearPublicDataCache, getCandles, normalizeKlines, validateWalletAddress, validateZapierHook } from "./publicData";

afterEach(() => {
  clearPublicDataCache();
  vi.unstubAllGlobals();
});

describe("public market data normalizers", () => {
  it("converts Binance klines into safe candle records", () => {
    expect(normalizeKlines([[1_700_000_000_000, "100", "104", "98", "102", "55"]])).toEqual([{ time: 1_700_000_000, open: 100, high: 104, low: 98, close: 102, volume: 55 }]);
    expect(normalizeKlines([["bad"]])).toEqual([]);
  });

  it("only accepts official Zapier Catch Hook targets", () => {
    expect(validateZapierHook("https://hooks.zapier.com/hooks/catch/123/abc/")).toContain("hooks.zapier.com");
    expect(() => validateZapierHook("https://example.com/hooks/catch/123")).toThrow("Only an HTTPS");
  });

  it("accepts only valid public EVM addresses", () => {
    expect(validateWalletAddress("0x1234567890abcdef1234567890abcdef12345678")).toHaveLength(42);
    expect(() => validateWalletAddress("not-a-wallet")).toThrow("valid public EVM");
  });

  it("returns a typed unavailable candle state when an upstream source fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    await expect(getCandles("BTCUSDT", "1h")).resolves.toMatchObject({
      symbol: "BTCUSDT",
      interval: "1h",
      candles: [],
      service: { available: false, source: "Binance spot public market data", notice: "Public data source returned 503." },
    });
  });

  it("returns a deliberate empty candle model when an upstream response has no usable rows", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await expect(getCandles("BTCUSDT", "1h")).resolves.toMatchObject({
      symbol: "BTCUSDT",
      interval: "1h",
      candles: [],
      service: { available: true, source: "Binance spot public market data" },
    });
  });
});
