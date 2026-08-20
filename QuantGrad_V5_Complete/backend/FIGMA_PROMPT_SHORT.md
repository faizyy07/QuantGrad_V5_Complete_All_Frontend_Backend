# QuantGrad v4 — Figma Prompt (under 2000 characters)

Copy the block below into Figma's Make Design tool.

```
Design a 1680px dark crypto trading research terminal "QuantGrad v4" — institutional research ledger style, NOT a SaaS dashboard. Quiet, dense, evidence-first, like a dim dealing room.

Colors: background #080B12 with faint green (#22C98D, 10% at top-left) and blue (rgba(96,165,250,10%) top-right) glows. Panels #0F1623/#121B2A, 1px border rgba(148,163,184,.18), 8px radius. Text #E5EDF8, muted #94A3B8. Signal green #22C98D ONLY for positive conviction/buy/active; soft coral #F07167 for sell/downside; ochre #D97706 for warnings. No decorative gradients.

Fonts: DM Mono for all values/timestamps/axis labels; Manrope for body UI; Space Grotesk bold for headings and signal labels. Section titles: 12px uppercase, letter-spacing .08em, muted.

Layout "ledger spine": 64px left nav (#0A101B) with aperture logo (4 graphite wedges around a green square) and 6 tabs: Market, Models, Risk Lab, Macro, Eigenspace, Training. Top header: title, "BTCUSDT · 1h" subtitle, symbol selector, green "Recalculate inference window" button, monospace timestamp. Center: large TradingView-style candlestick chart (green up/coral down candles, hairline grid, volume histogram). Right 240px ledger: big signal label (STRONG BUY green / HOLD ochre / STRONG SELL coral, 32px bold), probability bar, pills for trend, structure, risk, ADX, spread. Below chart: 5 KPI cards — Last Price, Spread, DXY, Fear & Greed, BTC Dominance.

Tabs: Market (default), Models (3 model cards, probability charts), Risk Lab (Profit Factor, Win Rate, Max Drawdown, Net Return, equity curve, return histogram), Macro (Gold, Crude, Fed Rate w/ Next FOMC, heatmap), Eigenspace (PCA scatter), Training (artifact status of 6 files, training report).
Details: thin conviction track beside panels, monospace source labels, pills 999px radius, 180ms hover. Offline state: aperture motif, "Offline preview — artifacts not trained". Desktop 1680x1024 plus mobile 390x844 stacked variant.
```

Length check: the code block is under 2000 characters (excluding the code fences). If Figma still complains, delete the trailing "plus mobile 390x844 stacked variant." sentence — it's the least essential part.
