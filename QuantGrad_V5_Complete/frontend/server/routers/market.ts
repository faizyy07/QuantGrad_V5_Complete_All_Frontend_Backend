import { z } from "zod";
import { publicProcedure, router } from "../_core/trpc";
import { getLocalModelAnalysis } from "../modelBridge";
import { getCandles, getDerivatives, getHyperliquid, getMacroCalendar, getMarketOverview, getMovers, getNews, getWallet, intervalSchema, symbolSchema, validateZapierHook } from "../publicData";

export const marketRouter = router({
  overview: publicProcedure.input(z.object({ symbol: symbolSchema.default("BTCUSDT") })).query(({ input }) => getMarketOverview(input.symbol)),
  candles: publicProcedure.input(z.object({ symbol: symbolSchema.default("BTCUSDT"), interval: intervalSchema.default("1h") })).query(({ input }) => getCandles(input.symbol, input.interval)),
  movers: publicProcedure.query(() => getMovers()),
  derivatives: publicProcedure.input(z.object({ symbol: symbolSchema.default("BTCUSDT") })).query(({ input }) => getDerivatives(input.symbol)),
  hyperliquid: publicProcedure.query(() => getHyperliquid()),
  wallet: publicProcedure.input(z.object({ address: z.string().min(1) })).query(({ input }) => getWallet(input.address)),
  macro: publicProcedure.query(() => getMacroCalendar()),
  news: publicProcedure.query(() => getNews()),
  analyze: publicProcedure.input(z.object({ symbol: symbolSchema.default("BTCUSDT") })).query(({ input }) => getLocalModelAnalysis(input.symbol)),
});

export const integrationRouter = router({
  testZapier: publicProcedure.input(z.object({ hookUrl: z.string().url(), event: z.enum(["signal", "strategy", "market-alert"]).default("market-alert") })).mutation(async ({ input }) => {
    const hookUrl = validateZapierHook(input.hookUrl);
    const response = await fetch(hookUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source: "QuantGrad", event: input.event, sentAt: new Date().toISOString(), message: "QuantGrad public-data webhook test" }), signal: AbortSignal.timeout(8_000) });
    if (!response.ok) throw new Error(`Zapier returned ${response.status}. Check that the Catch Hook URL is active.`);
    return { delivered: true, deliveredAt: Date.now() };
  }),
});
