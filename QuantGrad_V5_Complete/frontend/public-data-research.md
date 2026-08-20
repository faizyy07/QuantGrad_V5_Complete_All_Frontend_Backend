# Public Data and Integration Research

## Verified sources

| Capability | Source | No-cost implementation | Notes |
|---|---|---|---|
| Spot charts, quotes, movers and candles | [Binance Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api) | Use public market-data responses from `data-api.binance.vision` | The documentation directs public market-only requests to the dedicated base URL and requires exponential backoff on `429` rate-limit responses. |
| Derivatives dashboard | [Binance USDⓈ-M Futures market data](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data) | Read public funding, open-interest, long/short and mark-price data | The UI must identify the exchange and time window; it must not imply an all-exchange Coinglass aggregate. |
| Hyperliquid flow monitor | [Hyperliquid public API](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api) | Read public market and account data from `https://api.hyperliquid.xyz` | “Whale” is an analytical label defined transparently by size/activity thresholds, not private wallet intelligence. |
| Macro event calendar | [Federal Reserve FOMC calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | Curated calendar derived from the official schedule, with a source link | The Fed states that the FOMC has eight regular meetings per year; upcoming calendar dates are visible on the official page. |
| Optional workflow delivery | [Zapier Webhooks](https://help.zapier.com/hc/en-us/articles/8496288690317-Trigger-Zap-workflows-from-webhooks) | User supplies a free-plan Catch Hook URL in Settings; send an explicit, user-triggered event payload | Zapier documents Catch Hook and Catch Raw Hook for GET, PUT or POST requests. The URL is user-controlled and must never be hard-coded. |

## Product decisions

The expansion remains **information-only**. It will not place trades, connect exchange accounts, claim that signals are recommendations, or present influencer content as verified. Every public data card will show an exchange/source label, last refresh time, and a resilient unavailable-data state.

Live no-key requests use a conservative client cache, in-flight de-duplication, manual refresh controls, and a backoff state after errors. A local preview fixture will preserve each page’s layout when a public endpoint is unavailable or blocked by browser policy.

The initial integration screen stores the optional Zapier hook only in the user’s browser. A later server-side integration can protect the hook, support scheduled delivery, and avoid cross-origin limitations once the user chooses a host.
