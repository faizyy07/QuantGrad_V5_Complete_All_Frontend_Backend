import { useMemo, useState } from "react";
import { useLocation } from "wouter";
import {
  Activity, ArrowDownRight, ArrowUpRight, Bell, BookOpen, Bot, Calculator, CandlestickChart, ChartNoAxesCombined,
  ChevronDown, ChevronRight, ChevronUp, CircleAlert, CircleCheck, CircleDotDashed, Clock3, DatabaseZap, FlaskConical, Gauge,
  Layers3, LineChart, Menu, Network, Newspaper, Radar, RefreshCw, Search, Send, Settings2,
  ShieldAlert, Sigma, SlidersHorizontal, Sparkles, Target, TrendingDown, TrendingUp, WalletCards, Waves,
  X, Zap,
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { formatModelDetail, getDetailBarMagnitude, modelScorePercent } from "@/lib/modelPresentation";
import type { ModelDetail } from "@shared/modelAnalysis";
import "@/model-analytics.css";

type ChartMode = "candles" | "line" | "heikin";
type Interval = "1m" | "5m" | "15m" | "1h" | "4h" | "1d";
type NavKey = "terminal" | "discover" | "charts" | "strategy" | "derivatives" | "macro" | "flow" | "integrations";
type Candle = { time: number; open: number; high: number; low: number; close: number; volume: number };
type Service = { source: string; refreshedAt: number; notice?: string };
type MarketOverview = { symbol: string; price: number; changePct: number; quoteVolume: number; high: number; low: number; candles: Candle[]; service: Service };
type Movers = { rows: Array<{ symbol: string; price: number; changePct: number; quoteVolume: number }>; service: Service };
type News = { items: Array<{ title: string; link: string; publishedAt: string; source: string }>; service: Service };
type CandleFeed = { symbol: string; interval: Interval; candles: Candle[]; service: Service };
type Derivatives = { symbol: string; markPrice: number; fundingRate: number; nextFundingTime: number; openInterest: number; ratio: Array<{ time: number; longShortRatio: number; longAccount: number; shortAccount: number }>; service: Service };
type Hyperliquid = { rows: Array<{ coin: string; markPrice: number; openInterest: number; dayVolume: number; funding: number }>; service: Service };
type Macro = { events: Array<{ date: string; endDate?: string; title: string; impact: string; note: string }>; service: Service };
type ModelAnalysis = { state: "ready" | "artifacts_missing" | "unavailable" | "analysis_failed"; available: boolean; source: string; checkedAt: number; reason?: string; missingArtifacts: string[]; symbol?: string; signalLabel?: string; confidence?: number; riskLevel?: string; trend?: string; structure?: string; adx?: number; modelDetails: ModelDetail[] };
type PublicQuery<T> = { data?: T; isLoading: boolean; isFetching?: boolean; error?: unknown; refetch?: () => Promise<unknown> };

const NAV: Array<{ key: NavKey; label: string; icon: typeof Activity; eyebrow: string }> = [
  { key: "terminal", label: "Terminal", icon: ChartNoAxesCombined, eyebrow: "Decision workspace" },
  { key: "discover", label: "Discover", icon: Newspaper, eyebrow: "News & movers" },
  { key: "charts", label: "Charts", icon: CandlestickChart, eyebrow: "Chart studio" },
  { key: "strategy", label: "Strategy Lab", icon: FlaskConical, eyebrow: "Rule composer" },
  { key: "derivatives", label: "Derivatives", icon: Gauge, eyebrow: "Funding & positioning" },
  { key: "macro", label: "Macro", icon: Radar, eyebrow: "Event calendar" },
  { key: "flow", label: "Flow", icon: Waves, eyebrow: "Hyperliquid context" },
  { key: "integrations", label: "Integrations", icon: Network, eyebrow: "Automation settings" },
];
const MARKETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"];
const INTERVALS: Interval[] = ["1m", "5m", "15m", "1h", "4h", "1d"];

function currency(value?: number, digits?: number) {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: digits ?? (Math.abs(value ?? 0) >= 1000 ? 0 : 2) }).format(value ?? 0);
}
function compact(value?: number) {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(value ?? 0);
}
function percent(value?: number, fraction = false) {
  if (!Number.isFinite(value)) return "—";
  const number = fraction ? (value ?? 0) * 100 : value ?? 0;
  return `${number >= 0 ? "+" : ""}${number.toFixed(2)}%`;
}
function errorText(error: unknown) {
  return error instanceof Error ? error.message.replace(/^TRPCClientError:\s*/, "") : "The public data source is unavailable. Try again shortly.";
}
function title(symbol: string) { return symbol.replace("USDT", "/USD"); }

function ChartSurface({ candles, mode, height = "large" }: { candles: Candle[]; mode: ChartMode; height?: "large" | "small" }) {
  const series = useMemo(() => {
    if (mode !== "heikin") return candles;
    let previousOpen = candles[0]?.open ?? 0;
    let previousClose = candles[0]?.close ?? 0;
    return candles.map((candle) => {
      const close = (candle.open + candle.high + candle.low + candle.close) / 4;
      const open = (previousOpen + previousClose) / 2;
      const high = Math.max(candle.high, open, close);
      const low = Math.min(candle.low, open, close);
      previousOpen = open; previousClose = close;
      return { ...candle, open, close, high, low };
    });
  }, [candles, mode]);
  if (!series.length) return <div className={`chart-surface ${height} chart-empty market-evidence-state`}><div className="evidence-state-head"><span><CircleDotDashed size={14}/>Market evidence state</span><b>public feed</b></div><div className="evidence-grid"><i/><i/><i/><i/></div><svg className="evidence-trace" viewBox="0 0 100 50" preserveAspectRatio="none" aria-label="Market evidence preview"><path d="M0 33 L12 29 L20 37 L33 20 L46 25 L56 11 L67 17 L78 9 L88 21 L100 14"/><path d="M0 41 L12 38 L20 43 L33 31 L46 35 L56 25 L67 29 L78 22 L88 31 L100 25"/></svg><div className="evidence-horizon"><span>Confidence horizon</span><i/><i/><i/></div><div className="evidence-state-foot"><span><i/>Negotiating source telemetry</span><b>preview only</b></div></div>;
  const data = series.slice(-112);
  const min = Math.min(...data.map(item => item.low)); const max = Math.max(...data.map(item => item.high)); const range = Math.max(max - min, 1);
  const x = (index: number) => 3 + (index / Math.max(data.length - 1, 1)) * 94;
  const y = (value: number) => 89 - ((value - min) / range) * 76;
  const closePoints = data.map((item, index) => `${x(index)},${y(item.close)}`).join(" L ");
  return <div className={`chart-surface ${height}`}>
    <div className="chart-grid" />
    <span className="chart-axis high">{currency(max)}</span><span className="chart-axis low">{currency(min)}</span>
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-label={`${mode} price chart`} role="img">
      <defs><linearGradient id="qg-area" x1="0" x2="0" y1="0" y2="1"><stop stopColor="#5b8cff" stopOpacity=".30"/><stop offset="1" stopColor="#5b8cff" stopOpacity="0"/></linearGradient></defs>
      <path d={`M ${closePoints} L 97,93 L 3,93 Z`} fill="url(#qg-area)" />
      {mode === "line" ? <path d={`M ${closePoints}`} fill="none" stroke="#75a1ff" strokeWidth=".65" vectorEffect="non-scaling-stroke"/> : data.map((item, index) => { const bullish = item.close >= item.open; const xpos = x(index); const top = y(Math.max(item.open, item.close)); const body = Math.max(.8, Math.abs(y(item.open) - y(item.close))); return <g key={item.time}><line x1={xpos} x2={xpos} y1={y(item.high)} y2={y(item.low)} stroke={bullish ? "#55d7af" : "#f27491"} strokeWidth=".22"/><rect x={xpos - .3} y={top} width=".6" height={body} rx=".1" fill={bullish ? "#55d7af" : "#f27491"}/></g>; })}
      <path d={`M ${closePoints}`} fill="none" stroke="#78a9ff" strokeWidth={mode === "line" ? ".14" : ".18"} vectorEffect="non-scaling-stroke" opacity=".8"/>
    </svg>
  </div>;
}

function QueryState({ loading, error, children, label = "Loading public data" }: { loading?: boolean; error?: unknown; children: React.ReactNode; label?: string }) {
  if (loading) return <div className="data-state"><RefreshCw className="spin" size={16}/><span>{label}</span></div>;
  if (error) return <div className="data-state warning"><CircleAlert size={16}/><span>{errorText(error)}</span></div>;
  return <>{children}</>;
}

function MetricCard({ label, value, trend, note, tone = "blue" }: { label: string; value: string; trend?: string; note?: string; tone?: "blue" | "green" | "red" | "gold" }) {
  return <article className={`metric-card tone-${tone}`}><span>{label}</span><strong>{value}</strong>{trend ? <b className={trend.startsWith("-") ? "down" : "up"}>{trend.startsWith("-") ? <ArrowDownRight size={13}/> : <ArrowUpRight size={13}/>}{trend}</b> : <small>{note}</small>}</article>;
}

function Notice({ children }: { children: React.ReactNode }) { return <p className="source-note"><DatabaseZap size={13}/>{children}</p>; }

export default function Home() {
  const [location, navigate] = useLocation();
  const routeSegment = location.split("/")[1];
  const pathKey = (routeSegment === "strategies" ? "strategy" : routeSegment) as NavKey | undefined;
  const active = NAV.some(item => item.key === pathKey) ? pathKey as NavKey : "terminal";
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setInterval] = useState<Interval>("1h");
  const [chartMode, setChartMode] = useState<ChartMode>("candles");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const overview = trpc.market.overview.useQuery({ symbol }, { staleTime: 25_000, refetchInterval: 60_000, retry: 1 });
  const candles = trpc.market.candles.useQuery({ symbol, interval }, { staleTime: 25_000, refetchInterval: 60_000, retry: 1 });
  const movers = trpc.market.movers.useQuery(undefined, { staleTime: 25_000, refetchInterval: 60_000, retry: 1 });
  const derivatives = trpc.market.derivatives.useQuery({ symbol }, { staleTime: 25_000, refetchInterval: 60_000, retry: 1 });
  const hyperliquid = trpc.market.hyperliquid.useQuery(undefined, { staleTime: 25_000, refetchInterval: 60_000, retry: 1 });
  const macro = trpc.market.macro.useQuery(undefined, { staleTime: 300_000, retry: 1 });
  const news = trpc.market.news.useQuery(undefined, { staleTime: 90_000, refetchInterval: 180_000, retry: 1 });
  const model = trpc.market.analyze.useQuery({ symbol }, { staleTime: 300_000, refetchInterval: 300_000, retry: false });
  const go = (key: NavKey) => { setDrawerOpen(false); navigate(key === "terminal" ? "/" : `/${key}`); };
  const selectedNav = NAV.find(item => item.key === active) ?? NAV[0];

  return <main className="qg-app">
    <aside className={`qg-sidebar ${drawerOpen ? "open" : ""}`} aria-label="QuantGrad navigation">
      <div className="qg-logo"><span className="logo-orbit"><i/><i/><i/></span><span>Quant<span>Grad</span></span><em>V5</em><button className="mobile-close" onClick={() => setDrawerOpen(false)} aria-label="Close navigation"><X size={18}/></button></div>
      <div className="qg-nav-label">Intelligence</div>
      <nav>{NAV.slice(0, 4).map(item => <NavItem key={item.key} item={item} active={active === item.key} onClick={() => go(item.key)}/>)}</nav>
      <div className="qg-nav-label">Market structure</div>
      <nav>{NAV.slice(4, 7).map(item => <NavItem key={item.key} item={item} active={active === item.key} onClick={() => go(item.key)}/>)}</nav>
      <div className="qg-nav-label">System</div>
      <nav><NavItem item={NAV[7]} active={active === "integrations"} onClick={() => go("integrations")}/></nav>
      <div className="sidebar-foot"><div className="data-health"><span><i/> Public feeds</span><b>Resilient</b></div><p>Market intelligence, not financial advice.</p></div>
    </aside>
    {drawerOpen && <button className="drawer-scrim" aria-label="Close navigation" onClick={() => setDrawerOpen(false)}/>} 
    <section className="qg-main">
      <header className="qg-topbar"><button className="menu-button" onClick={() => setDrawerOpen(true)} aria-label="Open navigation"><Menu size={19}/></button><div className="crumb"><span>{selectedNav.eyebrow}</span><h1>{selectedNav.label}</h1></div><div className="topbar-tools"><div className="public-pulse"><i/>Public data</div><button className="top-icon" aria-label="Refresh market data" onClick={() => { void overview.refetch(); void candles.refetch(); void model.refetch(); }}><RefreshCw size={15}/></button><button className="top-icon" aria-label="Notifications"><Bell size={15}/></button><div className="desk-avatar">QG</div></div></header>
      <div className="page-wrap">
        {active === "terminal" && <TerminalPage symbol={symbol} setSymbol={setSymbol} overview={overview} movers={movers} model={model} chartMode={chartMode} setChartMode={setChartMode} candles={candles.data?.candles ?? []} onOpenCharts={() => go("charts")}/>} 
        {active === "discover" && <DiscoverPage movers={movers} news={news}/>} 
        {active === "charts" && <ChartsPage symbol={symbol} setSymbol={setSymbol} interval={interval} setInterval={setInterval} chartMode={chartMode} setChartMode={setChartMode} candles={candles}/>} 
        {active === "strategy" && <StrategyPage symbol={symbol} lastPrice={overview.data?.price}/>} 
        {active === "derivatives" && <DerivativesPage symbol={symbol} setSymbol={setSymbol} query={derivatives}/>} 
        {active === "macro" && <MacroPage query={macro} news={news}/>} 
        {active === "flow" && <FlowPage query={hyperliquid}/>} 
        {active === "integrations" && <IntegrationsPage/>}
      </div>
    </section>
  </main>;
}

function NavItem({ item, active, onClick }: { item: typeof NAV[number]; active: boolean; onClick: () => void }) { const Icon = item.icon; return <button className={`nav-item ${active ? "active" : ""}`} onClick={onClick}><Icon size={17}/><span>{item.label}</span>{active && <ChevronRight size={14}/>}</button>; }

function SymbolControl({ symbol, setSymbol }: { symbol: string; setSymbol: (value: string) => void }) { return <div className="symbol-control"><Search size={15}/><select value={symbol} onChange={event => setSymbol(event.target.value)} aria-label="Select market">{MARKETS.map(item => <option value={item} key={item}>{title(item)}</option>)}</select></div>; }

function TerminalPage({ symbol, setSymbol, overview, movers, model, chartMode, setChartMode, candles, onOpenCharts }: { symbol: string; setSymbol: (value: string) => void; overview: PublicQuery<MarketOverview>; movers: PublicQuery<Movers>; model: PublicQuery<ModelAnalysis>; chartMode: ChartMode; setChartMode: (value: ChartMode) => void; candles: Candle[]; onOpenCharts: () => void }) {
  const data = overview.data;
  return <>
    <section className="hero-line"><div><span className="eyebrow"><Sparkles size={13}/>Quantitative decision cockpit</span><h2>See structure <i>before it resolves.</i></h2><p>Live public-market context, transparent source labels, and a durable workspace that remains useful when a feed goes offline.</p></div><div className="hero-action"><span>Selected market</span><SymbolControl symbol={symbol} setSymbol={setSymbol}/><button onClick={onOpenCharts}>Open chart studio <ChevronRight size={15}/></button></div></section>
    <QueryState loading={overview.isLoading} error={overview.error} label="Connecting to public spot market data"><><section className="metric-row"><MetricCard label="Spot price" value={currency(data?.price)} trend={percent(data?.changePct)} tone={(data?.changePct ?? 0) >= 0 ? "green" : "red"}/><MetricCard label="24h quote volume" value={compact(data?.quoteVolume)} note="Public exchange volume"/><MetricCard label="24h range" value={`${currency(data?.low)} — ${currency(data?.high)}`} note="Session high / low" tone="gold"/><MetricCard label="Signal posture" value="Observe" note="No autonomous execution" tone="blue"/></section><Notice>{data?.service.source} · refreshed {data?.service.refreshedAt ? new Date(data.service.refreshedAt).toLocaleTimeString() : "—"}</Notice></></QueryState>
    <section className="terminal-layout"><article className="panel chart-panel"><div className="panel-head"><div><span className="mini">Market canvas</span><h3>{title(symbol)} <em>spot</em></h3></div><div className="segment" role="group" aria-label="Chart rendering type">{(["candles", "line", "heikin"] as ChartMode[]).map(mode => <button className={chartMode === mode ? "active" : ""} onClick={() => setChartMode(mode)} key={mode}>{mode === "heikin" ? "Heikin-Ashi" : mode[0].toUpperCase() + mode.slice(1)}</button>)}</div></div><ChartSurface candles={candles} mode={chartMode}/><div className="chart-footer"><span><i className="green-dot"/>Spot data via public API</span><button onClick={onOpenCharts}>Indicators & overlays <SlidersHorizontal size={14}/></button></div></article><ModelDecisionLedger analysis={model.data} loading={model.isLoading} onOpenCharts={onOpenCharts}/></section>
    <section className="lower-grid"><article className="panel movers-panel"><div className="panel-head"><div><span className="mini">Cross-market pulse</span><h3>Market movers</h3></div><span className="source-chip">public</span></div><QueryState loading={movers.isLoading} error={movers.error}><div className="mover-list">{movers.data?.rows.slice(0, 6).map(item => <button key={item.symbol} onClick={() => setSymbol(item.symbol)}><span className="token-icon">{item.symbol.slice(0, 1)}</span><span><b>{title(item.symbol)}</b><small>{compact(item.quoteVolume)} vol.</small></span><strong>{currency(item.price)}</strong><em className={item.changePct >= 0 ? "up" : "down"}>{percent(item.changePct)}</em></button>)}</div></QueryState></article><article className="panel source-panel"><div className="panel-head"><div><span className="mini">Methods note</span><h3>What this page is</h3></div><BookOpen size={17}/></div><p>QuantGrad surfaces market information from public sources. It does not represent a broker, exchange, investment adviser, or an automated trading service.</p><div className="method-grid"><span><CircleCheck size={14}/>Source-labelled</span><span><CircleCheck size={14}/>Failure-tolerant</span><span><CircleCheck size={14}/>Manual decisions</span></div></article></section>
  </>;
}

function ModelDecisionLedger({ analysis, loading, onOpenCharts }: { analysis?: ModelAnalysis; loading: boolean; onOpenCharts: () => void }) {
  const [showCalculations, setShowCalculations] = useState(true);
  const ready = analysis?.state === "ready";
  const confidence = ready ? modelScorePercent(analysis?.confidence) : undefined;
  const rawConfidence = analysis?.confidence ?? 0;
  const details = analysis?.modelDetails ?? [];
  const stateLabel = loading ? "checking" : ready ? "model live" : "offline";
  const detail = loading ? "Checking the local Python model and its trained artifacts." : analysis?.reason ?? "Start the local Python model to load signals.";
  const adx = Number.isFinite(analysis?.adx) ? analysis?.adx ?? 0 : undefined;
  const gaugeOffset = 301.6 * (1 - (confidence ?? 0) / 100);

  return <aside className="panel decision-panel model-decision-panel">
    <div className="panel-head model-ledger-header"><div><span className="mini">Decision ledger</span><h3>Model evidence</h3></div><span className={`live-tag ${ready ? "" : "unavailable"}`}><i/>{stateLabel}</span></div>
    {ready ? <>
      <div className="model-score-hero">
        <div className="model-score-copy"><span><CircleDotDashed size={13}/>Local inference</span><h4>{analysis?.signalLabel ?? "MODEL READ"}</h4><p>Decision score calculated by the connected local model for {title(analysis?.symbol ?? "BTCUSDT")}.</p></div>
        <div className="score-gauge" aria-label={`Model confidence ${confidence ?? 0} out of 100`}><svg className="score-gauge-svg" viewBox="0 0 120 120" role="img"><circle className="score-gauge-track" cx="60" cy="60" r="48"/><circle className="score-gauge-progress" cx="60" cy="60" r="48" strokeDasharray="301.6" strokeDashoffset={gaugeOffset}/></svg><div className="score-gauge-core"><b>{confidence ?? "—"}</b><span>/ 100</span></div></div>
      </div>
      <div className="decision-formula"><span><Calculator size={13}/>Confidence normalization</span><code>{rawConfidence.toFixed(4)} × 100 = {confidence ?? "—"}%</code></div>
      <div className="signal-topology">
        <div className="signal-topology-head"><span>Signal topology</span><span>returned inputs</span></div>
        <SignalTrack label="Trend" value={analysis?.trend ?? "Not returned"} note="market regime" progress={confidence ?? 0}/>
        <SignalTrack label="Structure" value={analysis?.structure ?? "Not returned"} note="price geometry" progress={confidence ?? 0}/>
        <SignalTrack label="ADX" value={adx === undefined ? "Not returned" : adx.toFixed(1)} note={adx === undefined ? "trend strength" : "visual scale: 0–50"} progress={adx === undefined ? 0 : Math.min(100, Math.round((adx / 50) * 100))}/>
        <SignalTrack label="Risk" value={analysis?.riskLevel ?? "Not returned"} note="model risk label" progress={confidence ?? 0}/>
      </div>
      <button className="calculation-toggle" onClick={() => setShowCalculations(value => !value)} aria-expanded={showCalculations}><span>Calculation trace · {details.length} returned fields</span>{showCalculations ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}</button>
      {showCalculations && <>{details.length ? <div className="calculation-ledger"><div className="calculation-ledger-head"><span className="calculation-kicker"><Sigma size={12}/>Model-returned values</span><p>Bars appear only when the returned number has an explicit 0–1 or 0–100 scale.</p></div>{details.map(detailItem => <CalculationRow detail={detailItem} key={detailItem.key}/>)}</div> : <div className="calculation-empty"><CircleAlert size={14}/>The connected model returned the decision fields above, but no additional factor, feature, or calculation values in this run.</div>}</>}
      <div className="model-run-facts"><span>Source<b>{analysis?.source}</b></span><span>Processed<b>{analysis?.checkedAt ? new Date(analysis.checkedAt).toLocaleTimeString() : "—"}</b></span><span>Market<b>{title(analysis?.symbol ?? "BTCUSDT")}</b></span><span>Payload fields<b>{details.length + 6} visible</b></span></div>
    </> : <div className="model-not-ready"><div><Bot size={16}/></div><h4>{loading ? "Validating local inference" : "Local model not ready"}</h4><p>{detail}</p>{analysis?.missingArtifacts?.length ? <div className="model-artifact-note">MISSING · {analysis.missingArtifacts.join(" · ")}</div> : null}</div>}
    <button className="wide-button" onClick={onOpenCharts}>Inspect market structure <ArrowUpRight size={15}/></button>
  </aside>;
}

function SignalTrack({ label, value, note, progress }: { label: string; value: string; note: string; progress: number }) { return <div className="signal-track"><span className="signal-track-name">{label}</span><div className="signal-track-copy"><b>{value}</b><small>{note}</small></div><strong className="signal-track-value">{Math.max(0, Math.min(100, progress))}%</strong><div className="signal-track-meter"><i style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}/></div></div>; }

function CalculationRow({ detail }: { detail: ModelDetail }) { const magnitude = getDetailBarMagnitude(detail); return <div className="calculation-row"><span className="calculation-row-icon"><Sigma size={12}/></span><div className="calculation-row-copy"><div className="calculation-row-title"><b title={detail.label}>{detail.label}</b><small>{detail.kind === "score" ? "SCORE" : detail.kind === "metric" ? "METRIC" : "FIELD"}</small></div>{magnitude !== undefined ? <div className="calculation-meter"><i style={{ width: `${magnitude}%` }}/></div> : null}</div><strong title={detail.value}>{formatModelDetail(detail)}</strong></div>; }

function DiscoverPage({ movers, news }: { movers: PublicQuery<Movers>; news: PublicQuery<News> }) {
  const stories = news.data?.items ?? []; const board = movers.data?.rows ?? [];
  return <><section className="page-intro"><span className="eyebrow"><Newspaper size={13}/>Public reporting, attributed</span><h2>Discover the information <i>around the tape.</i></h2><p>Headlines are provided for situational awareness. Read the original reporting before drawing conclusions.</p></section><section className="discover-grid"><article className="panel headline-panel"><div className="panel-head"><div><span className="mini">Market reporting</span><h3>Latest headlines</h3></div><span className="source-chip">RSS</span></div><QueryState loading={news.isLoading} error={news.error} label="Loading public news feed">{stories.length ? <div className="news-list">{stories.map((item, index) => <a href={item.link} target="_blank" rel="noreferrer" key={`${item.link}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><div><b>{item.title}</b><small>{item.source} · {item.publishedAt}</small></div><ArrowUpRight size={15}/></a>)}</div> : <EmptyPanel copy={news.data?.service.notice ?? "No public headlines were returned by this source yet."}/>}<Notice>{news.data?.service.notice ?? "External news coverage"}</Notice></QueryState></article><article className="panel movers-heat"><div className="panel-head"><div><span className="mini">Exchange activity</span><h3>Momentum board</h3></div><TrendingUp size={17}/></div><QueryState loading={movers.isLoading} error={movers.error}>{board.length ? <div className="heat-list">{board.map(item => <div key={item.symbol}><span>{title(item.symbol)}</span><div><i style={{ width: `${Math.min(100, Math.abs(item.changePct) * 10 + 12)}%` }} className={item.changePct >= 0 ? "positive-fill" : "negative-fill"}/></div><b className={item.changePct >= 0 ? "up" : "down"}>{percent(item.changePct)}</b></div>)}</div> : <EmptyPanel copy="The exchange did not return a movers board."/>}</QueryState></article></section><section className="panel narrative-panel"><div><span className="mini">Narrative radar</span><h3>Influence monitoring needs a source you can verify.</h3><p>QuantGrad does not fabricate “influencer trades” or infer private intent. Connect a public RSS feed or use a verified source link in your own research workflow.</p></div><div className="narrative-tags"><span>ETF flow</span><span>Layer 1</span><span>Macro</span><span>Stablecoins</span><span>Regulation</span></div></section></>;
}

function ChartsPage({ symbol, setSymbol, interval, setInterval, chartMode, setChartMode, candles }: { symbol: string; setSymbol: (value: string) => void; interval: Interval; setInterval: (value: Interval) => void; chartMode: ChartMode; setChartMode: (value: ChartMode) => void; candles: PublicQuery<CandleFeed> }) {
  const data = candles.data;
  return <><section className="studio-head"><div><span className="eyebrow"><CandlestickChart size={13}/>Chart studio</span><h2>Choose the lens, <i>keep the source visible.</i></h2></div><SymbolControl symbol={symbol} setSymbol={setSymbol}/></section><article className="panel studio-panel"><div className="studio-controls"><div className="segment">{INTERVALS.map(item => <button className={interval === item ? "active" : ""} onClick={() => setInterval(item)} key={item}>{item}</button>)}</div><div className="segment">{(["candles", "line", "heikin"] as ChartMode[]).map(mode => <button className={chartMode === mode ? "active" : ""} onClick={() => setChartMode(mode)} key={mode}>{mode === "heikin" ? "Heikin-Ashi" : mode}</button>)}</div></div><QueryState loading={candles.isLoading} error={candles.error} label="Loading price history"><><ChartSurface candles={data?.candles ?? []} mode={chartMode}/><div className="volume-strip">{(data?.candles ?? []).slice(-60).map(point => <i key={point.time} style={{ height: `${Math.min(100, Math.max(8, Math.log10(point.volume + 1) * 18))}%` }}/>)}</div><Notice>{data?.service.source} · {chartMode === "heikin" ? "Heikin-Ashi values are derived client-side from the source candles." : "Raw exchange candle values."}</Notice></></QueryState></article><section className="chart-notes"><article><Layers3 size={18}/><div><b>Rendering modes</b><p>Candles, line, and Heikin-Ashi are visual perspectives over the same public exchange data.</p></div></article><article><Target size={18}/><div><b>Indicators next</b><p>Use Strategy Lab to express rules; it will never claim a backtest was run unless results are generated.</p></div></article><article><Clock3 size={18}/><div><b>Refresh discipline</b><p>Public data is cached briefly to respect source limits and reduce noisy refreshes.</p></div></article></section></>;
}

function StrategyPage({ symbol, lastPrice }: { symbol: string; lastPrice?: number }) {
  const [trend, setTrend] = useState(true); const [volume, setVolume] = useState(true); const [risk, setRisk] = useState("medium");
  const rules = [trend && "Trend filter: price above 50-period moving average", volume && "Participation filter: volume above 20-period median", `Risk posture: ${risk} volatility tolerance`].filter(Boolean) as string[];
  return <><section className="page-intro"><span className="eyebrow"><FlaskConical size={13}/>Strategy definition</span><h2>Build rules before <i>you build conviction.</i></h2><p>Compose a transparent strategy specification. This workspace does not simulate or execute orders.</p></section><section className="strategy-grid"><article className="panel rule-builder"><div className="panel-head"><div><span className="mini">Rule composer</span><h3>{title(symbol)} checklist</h3></div><span className="source-chip">draft</span></div><div className="rule-control"><div><b>Trend confirmation</b><small>Only surface a setup when trend structure agrees.</small></div><button className={`toggle ${trend ? "on" : ""}`} onClick={() => setTrend(!trend)} aria-pressed={trend}><i/></button></div><div className="rule-control"><div><b>Volume participation</b><small>Require a relative volume confirmation.</small></div><button className={`toggle ${volume ? "on" : ""}`} onClick={() => setVolume(!volume)} aria-pressed={volume}><i/></button></div><div className="risk-choice"><b>Volatility tolerance</b><div className="choice-row">{["low", "medium", "high"].map(item => <button key={item} onClick={() => setRisk(item)} className={risk === item ? "active" : ""}>{item}</button>)}</div></div></article><article className="panel strategy-preview"><div className="panel-head"><div><span className="mini">Readable specification</span><h3>Strategy preview</h3></div><Bot size={18}/></div><div className="spec-card"><span>IF</span>{rules.map(rule => <p key={rule}>{rule}</p>)}<span>THEN</span><p>Flag the setup for manual review.</p><span>NOT</span><p>Place an order or make a prediction.</p></div><div className="backtest-empty"><ChartNoAxesCombined size={18}/><div><b>No backtest run</b><small>Connect your own historical dataset and explicitly initiate a backtest before performance statistics can appear.</small></div></div><div className="strategy-price">Reference spot: <strong>{currency(lastPrice)}</strong></div></article></section></>;
}

function DerivativesPage({ symbol, setSymbol, query }: { symbol: string; setSymbol: (value: string) => void; query: PublicQuery<Derivatives> }) {
  const data = query.data; const lastRatio = data?.ratio.at(-1);
  return <><section className="studio-head"><div><span className="eyebrow"><Gauge size={13}/>Derivatives dashboard</span><h2>Positioning is context, <i>not a verdict.</i></h2></div><SymbolControl symbol={symbol} setSymbol={setSymbol}/></section><QueryState loading={query.isLoading} error={query.error} label="Loading public futures metrics"><><section className="metric-row"><MetricCard label="Mark price" value={currency(data?.markPrice)} note="Futures market"/><MetricCard label="Funding rate" value={percent(data?.fundingRate, true)} note="Latest published rate" tone={((data?.fundingRate ?? 0) >= 0) ? "green" : "red"}/><MetricCard label="Open interest" value={compact(data?.openInterest)} note="Contract units" tone="gold"/><MetricCard label="Long / short ratio" value={(lastRatio?.longShortRatio ?? 0).toFixed(2)} note="Global accounts"/></section><Notice>{data?.service.source} · {data?.service.notice}</Notice><section className="deriv-grid"><article className="panel ratio-panel"><div className="panel-head"><div><span className="mini">Account positioning</span><h3>Long / short ratio</h3></div><span className="source-chip">1h</span></div><div className="ratio-chart">{data?.ratio.map(point => <div key={point.time}><i style={{ height: `${Math.min(100, point.longShortRatio * 42)}%` }} className={point.longShortRatio >= 1 ? "positive-fill" : "negative-fill"}/><span>{point.longShortRatio.toFixed(2)}</span></div>)}</div><p>Values above 1 indicate more reported long accounts than short accounts in the exchange’s global-account ratio.</p></article><article className="panel derivative-method"><div className="panel-head"><div><span className="mini">Interpretation guardrail</span><h3>Do not overread a ratio</h3></div><ShieldAlert size={17}/></div><p>Funding, open interest, and long/short account ratios are exchange-specific snapshots. They can change rapidly and do not identify counterparties or predict future returns.</p><div className="next-funding"><Clock3 size={16}/><span>Next funding timestamp</span><b>{data?.nextFundingTime ? new Date(data.nextFundingTime).toLocaleString() : "—"}</b></div></article></section></></QueryState></>;
}

function MacroPage({ query, news }: { query: PublicQuery<Macro>; news: PublicQuery<News> }) { const stories = news.data?.items ?? []; return <><section className="page-intro"><span className="eyebrow"><Radar size={13}/>Macro event ledger</span><h2>Know when liquidity <i>changes character.</i></h2><p>An event calendar for planning attention. Confirm timing against the original source before acting.</p></section><section className="macro-grid"><article className="panel calendar-panel"><div className="panel-head"><div><span className="mini">Official schedule</span><h3>FOMC calendar</h3></div><span className="source-chip">Fed</span></div><QueryState loading={query.isLoading} error={query.error}><><div className="event-list">{query.data?.events.map(event => <div key={event.date}><div className="date-block"><b>{new Date(`${event.date}T12:00:00Z`).toLocaleString("en-US", { month: "short" })}</b><strong>{new Date(`${event.date}T12:00:00Z`).getUTCDate()}</strong></div><div><span className="impact">{event.impact} impact</span><h4>{event.title}</h4><p>{event.note}</p></div><small>{event.endDate ? `${event.date} → ${event.endDate}` : event.date}</small></div>)}</div><Notice>{query.data?.service.source} · {query.data?.service.notice}</Notice></></QueryState></article><article className="panel macro-news"><div className="panel-head"><div><span className="mini">Market context</span><h3>Recent coverage</h3></div><Newspaper size={17}/></div><QueryState loading={news.isLoading} error={news.error}>{stories.length ? <div className="compact-news">{stories.slice(0, 4).map(item => <a href={item.link} target="_blank" rel="noreferrer" key={item.link}>{item.title}<ArrowUpRight size={14}/></a>)}</div> : <EmptyPanel copy={news.data?.service.notice ?? "No attributed coverage is available from this feed."}/>}</QueryState></article></section></>;
}

function FlowPage({ query }: { query: PublicQuery<Hyperliquid> }) {
  const [address, setAddress] = useState(""); const canQuery = /^0x[a-fA-F0-9]{40}$/.test(address.trim());
  const wallet = trpc.market.wallet.useQuery({ address: address.trim() || "0x0000000000000000000000000000000000000000" }, { enabled: canQuery, retry: 1 });
  const rows = query.data?.rows ?? [];
  return <><section className="page-intro flow-intro"><span className="eyebrow"><Waves size={13}/>Public onchain market context</span><h2>Follow disclosed positions, <i>not folklore.</i></h2><p>Hyperliquid market data and public wallet positions are visible only when a valid publicly disclosed address is supplied.</p></section><section className="flow-grid"><article className="panel hyper-panel"><div className="panel-head"><div><span className="mini">Venue overview</span><h3>Hyperliquid context</h3></div><span className="source-chip">public API</span></div><QueryState loading={query.isLoading} error={query.error}>{rows.length ? <><div className="hyper-list"><div className="hyper-header"><span>Asset</span><span>Mark</span><span>Open interest</span><span>24h volume</span><span>Funding</span></div>{rows.map(row => <div key={row.coin}><b>{row.coin}</b><span>{currency(row.markPrice)}</span><span>{compact(row.openInterest)}</span><span>{compact(row.dayVolume)}</span><em className={row.funding >= 0 ? "up" : "down"}>{percent(row.funding, true)}</em></div>)}</div><Notice>{query.data?.service.notice}</Notice></> : <EmptyPanel copy={query.data?.service.notice ?? "No market rows were returned by the public Hyperliquid endpoint."}/>}</QueryState></article><article className="panel wallet-panel"><div className="panel-head"><div><span className="mini">Wallet monitor</span><h3>Inspect a public address</h3></div><WalletCards size={18}/></div><p>Paste a public EVM address. QuantGrad will only read exposed position data; it cannot access or move funds.</p><div className="wallet-input"><input value={address} onChange={event => setAddress(event.target.value)} placeholder="0x… public wallet" aria-label="Public wallet address"/><button disabled={!canQuery || wallet.isFetching} onClick={() => void wallet.refetch()}><Search size={15}/></button></div>{address && !canQuery && <div className="inline-warning"><CircleAlert size={14}/>Enter a valid 42-character EVM address.</div>}<QueryState loading={wallet.isFetching} error={wallet.error} label="Loading disclosed positions">{wallet.data && <div className="position-list">{wallet.data.positions.length ? wallet.data.positions.map(position => <div key={position.coin}><b>{position.coin}</b><span>Size {position.size}</span><span>PnL {currency(position.unrealizedPnl)}</span></div>) : <div className="empty-inline">No open positions were returned for this public address.</div>}</div>}</QueryState></article></section></>;
}

function EmptyPanel({ copy }: { copy: string }) { return <div className="empty-panel"><DatabaseZap size={17}/><span>{copy}</span></div>; }

function IntegrationsPage() {
  const [hookUrl, setHookUrl] = useState(""); const [event, setEvent] = useState<"signal" | "strategy" | "market-alert">("market-alert");
  const testHook = trpc.integrations.testZapier.useMutation();
  const submit = () => { if (hookUrl) testHook.mutate({ hookUrl, event }); };
  return <><section className="page-intro"><span className="eyebrow"><Network size={13}/>No-code automation</span><h2>Connect your workflow, <i>keep control local.</i></h2><p>Optional webhook delivery only. QuantGrad does not store a webhook URL in this prototype and never sends an order instruction.</p></section><section className="integration-grid"><article className="panel integration-card"><div className="integration-logo"><Zap size={22}/></div><div><span className="mini">Optional outbound webhook</span><h3>Zapier Catch Hook</h3><p>Send a test event to an active Webhooks by Zapier Catch Hook URL. Zapier plan availability is determined by Zapier; this application uses no paid market-data API.</p></div><label>Catch Hook URL<input value={hookUrl} onChange={event => setHookUrl(event.target.value)} placeholder="https://hooks.zapier.com/hooks/catch/..."/></label><div className="event-options">{(["market-alert", "signal", "strategy"] as const).map(item => <button className={event === item ? "active" : ""} onClick={() => setEvent(item)} key={item}>{item}</button>)}</div><button className="wide-button" onClick={submit} disabled={!hookUrl || testHook.isPending}>{testHook.isPending ? <RefreshCw className="spin" size={15}/> : <Send size={15}/>}Send test event</button>{testHook.isSuccess && <div className="success-line"><CircleCheck size={15}/>Test event delivered.</div>}{testHook.error && <div className="inline-warning"><CircleAlert size={14}/>{errorText(testHook.error)}</div>}</article><article className="panel integration-guide"><div className="panel-head"><div><span className="mini">Connection map</span><h3>How to use it</h3></div><Settings2 size={18}/></div><ol><li><span>01</span><p>Create a Catch Hook in your own Zapier account.</p></li><li><span>02</span><p>Paste that URL temporarily and send a test event.</p></li><li><span>03</span><p>Map the event to your destination in Zapier.</p></li></ol><div className="integration-safe"><ShieldAlert size={16}/><p>Never place exchange keys, private keys, recovery phrases, or account credentials in a webhook field.</p></div></article></section></>;
}
