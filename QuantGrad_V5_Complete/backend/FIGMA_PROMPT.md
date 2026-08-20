# QuantGrad v4 Web Terminal — Figma Design Prompt

Copy everything inside the code block below and paste it into Figma (e.g., Figma's AI design assistant / Make Design tool, or as a design brief for a designer). It fully describes the terminal built from your codebase so Figma reproduces the exact look.

---

```
Design a full-width desktop web application (1680px artboard, dark theme) called "QuantGrad v4 — Three-Model Institutional Pipeline". It is a professional crypto trading research terminal, NOT a generic SaaS dashboard.

DESIGN DIRECTION: "Observatory Ledger" — institutional editorialism meets Swiss technical documentation. It should feel like a dimly lit institutional dealing room and a research ledger: quiet, dense, evidence-first. No flashy gradients used as decoration, no emoji, no cartoon illustration. Charts, model probabilities, and risk state have visual primacy.

COLOR SYSTEM (use these exact values):
- Background: #080B12 (blue-black graphite), with two very subtle radial glows at the top edge: green glow rgba(22,199,132,.10) at 20% 0% and blue glow rgba(96,165,250,.10) at 80% 0%
- Panel background: #0F1623 with a slightly lighter variant #121B2A, each with a 1px hairline border rgba(148,163,184,.18) and 8px rounded corners
- Primary text: #E5EDF8 (warm mineral off-white)
- Muted text / labels: #94A3B8
- Signal green (Quant Signal Green #22C98D): reserved STRICTLY for positive conviction, active states, buy signals, and the center node of the Observatory Mark — never decorative
- Coral (soft red #F07167 style): reserved for downside / sell / negative conviction
- Ochre / amber (#D97706 style): reserved for non-directional / conditional risk / warning states
- Grid lines, ticks, and axes in very low-contrast graphite hairlines

TYPOGRAPHY:
- DM Mono (or JetBrains Mono fallback): ALL values, timestamps, labels, axis metadata, source labels, monospace coordinate labels
- Manrope (or Inter fallback): readable UI copy, body text
- Space Grotesk: bold headings and signal labels
- Card section titles: 12px, font-weight 700, letter-spacing 0.08em, uppercase, muted color

LAYOUT: "Ledger spine" composition — asymmetric, evidence-focused:
1. FIXED LEFT VERTICAL NAVIGATION STRIP (64px wide, background #0A101B, right border hairline): contains the Observatory Mark logo (four small graphite wedge shapes framing a tiny signal-green #22C98D square aperture), and 6 vertical tab icons + labels stacked below: Market, Models, Risk Lab, Macro, Eigenspace, Training. Active tab highlighted with a thin green vertical track.
2. ANALYTICAL HEADER BAR (full width, top): rounded 8px card, gradient background rgba(15,22,35,.98)→rgba(10,16,27,.98), containing title "QuantGrad" in Space Grotesk 26px bold, subtitle "Three-Model Institutional Pipeline — BTCUSDT · 1h" in muted, and on the right: symbol selector dropdown (BTCUSDT / ETHUSDT / SOLUSDT), a "Recalculate full 65-feature inference window" primary button in signal green (#22C98D text on dark, or green fill with dark text), and a small monospace timestamp of last refresh.
3. MAIN MARKET CANVAS (center, largest area, ~60% width): a large TradingView-style candlestick chart (green #22C98D up candles, coral down candles, hairline grid), with a thin colored conviction track ruler to its left edge, monospace price/timestamp labels on axes, and a volume histogram along the bottom.
4. NARROW RIGHT DECISION LEDGER RAIL (~240px): the "execution ledger" — a signal card (min 188px tall) showing the live model decision: a large signal label (e.g., "STRONG BUY" in signal green, "HOLD" in ochre, "STRONG SELL" in coral, Space Grotesk 32px weight 820), signal meta line in muted monospace, a probability distribution chart (horizontal stacked bar of the 5 entry probabilities), and pills (rounded 999px, border hairline, muted background) for trend (UPTREND/DOWNTREND/RANGING), structure (MSS/OB/FVG/SWEEP/NONE), risk level, ADX value, and spread.
5. LOWER PANELS: unequal, evidence-focused modules (not a symmetric card grid): a row of KPI metric cards and below it the tab-specific content.

KPI METRIC CARDS (top of main canvas, 5 in a row):
- "Last Price" $XX,XXX.XX with "+X.XX% 24h" delta
- "Spread" with imbalance delta
- "DXY" with 5d momentum delta
- "Fear & Greed" index number with sentiment label
- "BTC Dominance" percentage with macro age note
Each card: dark gradient background, 1px hairline border, 8px radius, muted uppercase label, large DM Mono value.

TAB CONTENT (design all 6 views):
- MARKET (default): KPI row + central candlestick chart + right decision ledger.
- MODELS: three model artifact cards (TrendModel 3-class, StructureModel 6-class, EntryModel 5-class) each showing architecture diagram placeholders, probability distribution charts, and training metrics; a probability mix chart (stacked area of the 5 entry signals across the rolling window).
- RISK LAB: four metric cards (Profit Factor, Win Rate %, Max Drawdown %, Net Return % with trade count), an equity curve line chart starting at 1.0, and a return distribution histogram; coral for losses, green for gains.
- MACRO: metric cards for Gold ($), WTI Crude ($), Fed Rate (% with "Next FOMC 2026-09-16" note), DXY, Fear & Greed, BTC Dominance; a macro correlation heatmap module; monospace source labels.
- EIGENSPACE: PCA eigenspace scatter visualization placeholder, metric cards for PCA Components, Feature Dimension (65), Window Size (60).
- TRAINING: artifact status ledger (6 items: trend_model.keras, structure_model.keras, entry_model.keras, scaler.pkl, pca.pkl, quantum_preprocessor.pkl) with ready/missing states, training report metrics (Trained Symbol, Bars, Best Entry Profit Factor, Trained At date).

DETAIL ELEMENTS:
- Signal ruler: a 3-4px vertical colored track beside decision panels encoding conviction (green=constructive, coral=downside, ochre=conditional).
- Coordinate labels: low-contrast monospace metadata ticks, axis units, and data-source labels ("Binance 1h · alternative.me · FRED") that make the terminal feel instrument-grade.
- Pills: border-radius 999px, 1px hairline border, rgba(148,163,184,.07) background, 12.5px muted text.
- Buttons: tactile, 8px radius; primary action in signal green, secondary ghost with hairline border; hover 180ms ease-out.
- Empty/offline state: when artifacts are missing, show a muted placeholder screen with the Observatory Mark aperture motif (4 graphite wedges around a green square), the headline "Offline preview — artifacts not trained" and a monospace list of missing artifact files; never show fake live signals.
- Microcopy tone: plain and evidentiary, e.g. "Model consensus is constructive; execution quality remains conditional." — never promotional.

ANIMATION (document in notes): 180ms ease-out hover/panel entry; header tools arrive first at load, then decision ledger, then chart panels in a 45ms cascade; status pulses breathe slowly; respect prefers-reduced-motion.

OUTPUT: one 1680x1024 desktop frame plus a 390x844 mobile variant (stacked single column, chart full-width, ledger moves below).
```

---

## Tips for getting the best result in Figma

1. Paste the prompt into Figma's AI design tool (Make Design) and iterate: ask follow-ups like "make the right decision rail narrower" or "increase chart height".
2. Figma's export gives you **static UI code** (HTML/Tailwind/React). That is fine for the look, but your terminal's **working data logic** — the candlestick chart fed by `lightweight-charts`, the live `/api/analyze` calls to `server.py`, the fusion signal updates — still needs to be wired into the React app. The prompt above is designed to match the existing codebase (`ideas.md` style, `dashboard.py` metrics/tabs) so a developer (or me) can replace the frontend layer without changing the ML pipeline.
3. Keep the Observatory Mark (4 graphite wedges + green square), DM Mono values, and the reserved-color semantics (green = positive conviction only) — those are the three signature elements that make the design ownable.
4. If you use Figma to Code plugins (Locofy, Builder.io, Anima), export screen-by-screen: the 6 tabs map to 6 React views that already exist in your project.
