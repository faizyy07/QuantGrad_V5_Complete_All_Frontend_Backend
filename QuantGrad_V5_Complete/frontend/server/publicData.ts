import { z } from "zod";

const BINANCE_SPOT = "https://data-api.binance.vision";
const BINANCE_FUTURES = "https://fapi.binance.com";
const HYPERLIQUID_INFO = "https://api.hyperliquid.xyz/info";
const CACHE_TTL_MS = 35_000;

type CacheEntry<T> = { expiresAt: number; value: T };
const cache = new Map<string, CacheEntry<unknown>>();

/** Clears transient public-source results for deterministic tests and manual refreshes. */
export function clearPublicDataCache() {
  cache.clear();
}

export type Candle = { time: number; open: number; high: number; low: number; close: number; volume: number };
export type ServiceState = { available: boolean; source: string; refreshedAt: number; notice?: string };

export const symbolSchema = z.string().regex(/^[A-Z0-9]{5,20}$/, "Use an uppercase exchange symbol such as BTCUSDT.");
export const intervalSchema = z.enum(["1m", "5m", "15m", "1h", "4h", "1d"]);

async function cached<T>(key: string, loader: () => Promise<T>): Promise<T> {
  const found = cache.get(key) as CacheEntry<T> | undefined;
  if (found && found.expiresAt > Date.now()) return found.value;
  const value = await loader();
  cache.set(key, { value, expiresAt: Date.now() + CACHE_TTL_MS });
  return value;
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...init, signal: AbortSignal.timeout(8_000), headers: { Accept: "application/json", ...(init?.headers ?? {}) } });
  if (!response.ok) throw new Error(`Public data source returned ${response.status}.`);
  return response.json() as Promise<T>;
}

function success<T>(source: string, value: T, notice?: string): T & { service: ServiceState } {
  return { ...(value as object), service: { available: true, source, refreshedAt: Date.now(), notice } } as T & { service: ServiceState };
}

function reason(error: unknown) {
  if (error instanceof Error) {
    if (error.name === "TimeoutError" || error.name === "AbortError") return "The upstream source timed out. Retry in a moment.";
    return error.message;
  }
  return "The upstream source is unavailable. Retry in a moment.";
}

function unavailable<T>(source: string, value: T, error: unknown): T & { service: ServiceState } {
  return { ...(value as object), service: { available: false, source, refreshedAt: Date.now(), notice: reason(error) } } as T & { service: ServiceState };
}

async function publicResult<T>(key: string, source: string, fallback: T, loader: () => Promise<T>, notice?: string): Promise<T & { service: ServiceState }> {
  return cached(key, async () => {
    try {
      return success(source, await loader(), notice);
    } catch (error) {
      return unavailable(source, fallback, error);
    }
  });
}

export function normalizeKlines(payload: unknown): Candle[] {
  if (!Array.isArray(payload)) return [];
  return payload.flatMap((item) => {
    if (!Array.isArray(item) || item.length < 6) return [];
    const [time, open, high, low, close, volume] = item;
    const values = [Number(time), Number(open), Number(high), Number(low), Number(close), Number(volume)];
    return values.every(Number.isFinite) ? [{ time: Math.floor(values[0] / 1000), open: values[1], high: values[2], low: values[3], close: values[4], volume: values[5] }] : [];
  });
}

export function validateZapierHook(value: string): string {
  const url = new URL(value);
  const allowed = url.protocol === "https:" && url.hostname === "hooks.zapier.com" && url.pathname.startsWith("/hooks/catch/");
  if (!allowed) throw new Error("Only an HTTPS Webhooks by Zapier Catch Hook URL is accepted.");
  return url.toString();
}

export function validateWalletAddress(value: string): string {
  const address = value.trim().toLowerCase();
  if (!/^0x[a-f0-9]{40}$/.test(address)) throw new Error("Enter a valid public EVM wallet address.");
  return address;
}

export async function getMarketOverview(symbol: string) {
  return publicResult(`overview:${symbol}`, "Binance spot public market data", { symbol, price: 0, changePct: 0, quoteVolume: 0, high: 0, low: 0, candles: [] as Candle[] }, async () => {
    const [ticker, klines] = await Promise.all([
      getJson<{ lastPrice: string; priceChangePercent: string; quoteVolume: string; highPrice: string; lowPrice: string }>(`${BINANCE_SPOT}/api/v3/ticker/24hr?symbol=${symbol}`),
      getJson<unknown>(`${BINANCE_SPOT}/api/v3/klines?symbol=${symbol}&interval=1h&limit=120`),
    ]);
    return { symbol, price: Number(ticker.lastPrice), changePct: Number(ticker.priceChangePercent), quoteVolume: Number(ticker.quoteVolume), high: Number(ticker.highPrice), low: Number(ticker.lowPrice), candles: normalizeKlines(klines) };
  });
}

export async function getCandles(symbol: string, interval: z.infer<typeof intervalSchema>) {
  return publicResult(`candles:${symbol}:${interval}`, "Binance spot public market data", { symbol, interval, candles: [] as Candle[] }, async () => {
    const payload = await getJson<unknown>(`${BINANCE_SPOT}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=180`);
    return { symbol, interval, candles: normalizeKlines(payload) };
  });
}

export async function getMovers() {
  const symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"];
  return publicResult("movers", "Binance spot public market data", { rows: [] as Array<{ symbol: string; price: number; changePct: number; quoteVolume: number }> }, async () => {
    const encoded = encodeURIComponent(JSON.stringify(symbols));
    const payload = await getJson<Array<{ symbol: string; lastPrice: string; priceChangePercent: string; quoteVolume: string }>>(`${BINANCE_SPOT}/api/v3/ticker/24hr?symbols=${encoded}`);
    const rows = payload.map((item) => ({ symbol: item.symbol, price: Number(item.lastPrice), changePct: Number(item.priceChangePercent), quoteVolume: Number(item.quoteVolume) })).sort((a, b) => Math.abs(b.changePct) - Math.abs(a.changePct));
    return { rows };
  });
}

export async function getDerivatives(symbol: string) {
  type Ratio = { time: number; longShortRatio: number; longAccount: number; shortAccount: number };
  return publicResult(`derivatives:${symbol}`, "Binance USDⓈ-M Futures public market data", { symbol, markPrice: 0, fundingRate: 0, nextFundingTime: 0, openInterest: 0, ratio: [] as Ratio[] }, async () => {
    const [premium, interest, ratio] = await Promise.all([
      getJson<{ lastFundingRate: string; markPrice: string; nextFundingTime: number }>(`${BINANCE_FUTURES}/fapi/v1/premiumIndex?symbol=${symbol}`),
      getJson<{ openInterest: string; time: number }>(`${BINANCE_FUTURES}/fapi/v1/openInterest?symbol=${symbol}`),
      getJson<Array<{ longShortRatio: string; longAccount: string; shortAccount: string; timestamp: number }>>(`${BINANCE_FUTURES}/futures/data/globalLongShortAccountRatio?symbol=${symbol}&period=1h&limit=24`),
    ]);
    return { symbol, markPrice: Number(premium.markPrice), fundingRate: Number(premium.lastFundingRate), nextFundingTime: premium.nextFundingTime, openInterest: Number(interest.openInterest), ratio: ratio.map((point) => ({ time: point.timestamp, longShortRatio: Number(point.longShortRatio), longAccount: Number(point.longAccount), shortAccount: Number(point.shortAccount) })) };
  }, "Exchange-specific data, not an all-market aggregate.");
}

export async function getHyperliquid() {
  type FlowRow = { coin: string; markPrice: number; openInterest: number; dayVolume: number; funding: number };
  return publicResult("hyperliquid", "Hyperliquid public API", { rows: [] as FlowRow[] }, async () => {
    const payload = await getJson<[unknown, Array<{ coin: string; markPx: string; openInterest: string; dayNtlVlm: string; funding: string }>]>(HYPERLIQUID_INFO, { method: "POST", body: JSON.stringify({ type: "metaAndAssetCtxs" }), headers: { "Content-Type": "application/json" } });
    const rows = (payload[1] ?? []).filter((item) => ["BTC", "ETH", "SOL", "HYPE"].includes(item.coin)).map((item) => ({ coin: item.coin, markPrice: Number(item.markPx), openInterest: Number(item.openInterest), dayVolume: Number(item.dayNtlVlm), funding: Number(item.funding) }));
    return { rows };
  }, "Public market context only. Add a public wallet to inspect its disclosed positions.");
}

export async function getWallet(addressInput: string) {
  const address = validateWalletAddress(addressInput);
  type Position = { coin: string; size: number; entryPrice: number; unrealizedPnl: number; leverage: number };
  return publicResult(`wallet:${address}`, "Hyperliquid public API", { address, positions: [] as Position[] }, async () => {
    const payload = await getJson<{ assetPositions?: Array<{ position?: { coin?: string; szi?: string; entryPx?: string; unrealizedPnl?: string; leverage?: { value?: number } } }> }>(HYPERLIQUID_INFO, { method: "POST", body: JSON.stringify({ type: "clearinghouseState", user: address }), headers: { "Content-Type": "application/json" } });
    const positions = (payload.assetPositions ?? []).flatMap((item) => item.position ? [{ coin: item.position.coin ?? "Unknown", size: Number(item.position.szi ?? 0), entryPrice: Number(item.position.entryPx ?? 0), unrealizedPnl: Number(item.position.unrealizedPnl ?? 0), leverage: Number(item.position.leverage?.value ?? 0) }] : []);
    return { address, positions };
  });
}

export async function getMacroCalendar() {
  const events = [
    { date: "2026-09-15", endDate: "2026-09-16", title: "FOMC meeting", impact: "High", note: "Summary of Economic Projections expected" },
    { date: "2026-10-27", endDate: "2026-10-28", title: "FOMC meeting", impact: "High", note: "Policy statement scheduled" },
    { date: "2026-12-08", endDate: "2026-12-09", title: "FOMC meeting", impact: "High", note: "Summary of Economic Projections expected" },
  ];
  return success("Federal Reserve FOMC calendar", { events }, "Dates are based on the official FOMC schedule and should be reconfirmed at source before acting.");
}

export async function getNews() {
  return publicResult("news", "Google News RSS public feed", { items: [] as Array<{ title: string; link: string; publishedAt: string; source: string }> }, async () => {
    const response = await fetch("https://news.google.com/rss/search?q=cryptocurrency%20markets%20when%3A2d&hl=en-US&gl=US&ceid=US:en", { signal: AbortSignal.timeout(8_000) });
    if (!response.ok) throw new Error(`News source returned ${response.status}.`);
    const xml = await response.text();
    const items = Array.from(xml.matchAll(/<item>([\s\S]*?)<\/item>/g)).slice(0, 8).map((match) => {
      const text = match[1];
      const field = (name: string) => (text.match(new RegExp(`<${name}>(?:<!\\[CDATA\\[)?([\\s\\S]*?)(?:\\]\\]>)?<\/${name}>`))?.[1] ?? "").replace(/<[^>]*>/g, "").trim();
      return { title: field("title"), link: field("link"), publishedAt: field("pubDate"), source: field("source") || "Google News" };
    }).filter((item) => item.title && item.link);
    return { items };
  }, "Headlines are external reporting, not QuantGrad research or trade guidance.");
}
