# Figma Reference Notes — "Trading website - Glowing & Gradient Text Style" (Capi Product, community file 1299626894568887805)

URL in editor: https://www.figma.com/design/C8V9JZH25vElrX6xmBXsgz/ (user logged in via My Browser)
Pages in file: Thumbnail (marketing cover), Shot (design shots in 5 aspect ratios: 1800x1200, 1600x1200, 1000x1000, 1200x900, pinterest), UI Design.

## Overall style (from community page + editor canvas)
- **Glowing & Gradient Text Style**: near-black background (#000000/#050810 range) with large soft nebula glows — deep blue (#1D4ED8–#2F6BFF) at bottom-right corner glow, faint cyan/teal glows, occasional purple trace.
- Electric blue #2F6BFF / #3B82F6 primary accent; glowing buttons (solid electric blue fill with outer glow shadow); pill buttons (999px radius) with hairline borders.
- Headlines in white, bold, some words treated with gradient text (cyan→blue gradient fill).
- Rounded cards (8-16px radius) with 1px rgba(255,255,255,.08) borders, subtle translucent fills.
- Body text gray (#9CA3AF-ish), small, uppercase letter-spaced labels for metadata.

## Landing hero (Shot page frames)
- Top nav: small square CapiTrade logo mark (blue icon in rounded square), brand wordmark "CapiTrade"; CENTER pill navigation: segmented pill with tabs Chart | Markets | News | Community | More; right: "Login" text link + "Start Free Trial" outlined pill.
- Small centered pill badge above headline: sparkle icon + "New: Our AI integration just landed" (glass, hairline border, subtle glow).
- Headline centered: "Discover endless possibilities in the world of Trading." — white bold, two lines.
- Subtext paragraph centered, muted gray.
- Three check rows: icon + "Fast Trading", icon + "Secure & Reliable", icon + "Continuous Market Updates".
- Two CTAs: "Start Trading" (glowing electric-blue filled pill with flame icon) and "Try Demo" (ghost/outline pill).
- Right side: floating tilted mockup cards of the platform UI (dark dashboards with area charts), one card has a circular play button.

## Dashboard/app screens (bottom strip of landing shot)
- Dark app window (like a trading app screenshot): top bar with tabs Overview | AI Support, right-side buttons Type + date range "Sep 24 → Oct 10", "Download" button.
- Sidebar left: sections "Request", "Questions", "Insight bot" with small icons.
- "Performance" section header with info icon; metric rows below: "Today's", "Profit/Loss" (0.0 / -0.0), "Number of..." style metric tiles in dark cards with values; area chart rows.
- Bottom: "Today", "Profile", "Total Buy", "Total Sell(?)", "MiniChart" row labels.
- Cards use dark navy (#0A0F1E-ish), hairline borders, blue accent numbers, small area charts under metrics.
- Floating dark modal/card: "Start Trading / Be smart & decisive / Continuous Market updates" three lines with icons, plus Start Trading / Try Free buttons.

## File notes
- The file is a **landing page concept kit**, not a full app UI kit; dashboard details come from the mockup screenshots inside it. The Shot page contains the same hero design at 5 aspect ratios. The IMAGE layer in UI Design page holds the full design as images.
- License CC BY 4.0 — free to use/adapt.

## Design translation for QuantGrad v4 rebuild (glowing gradient style)
- Replace Observatory Ledger graphite/green with: bg #05080F (near-black navy), nebula glows blue/cyan, primary electric blue #2F6BFF + gradient text (cyan→blue), pills everywhere (nav, CTAs, badges), glassmorphic cards, hairline white 8% borders, glowing blue "Start Trading"-style primary buttons.
- Keep QuantGrad brand: keep logo mark structure but glow style; tabs: Market, Models, Risk Lab, Macro, Eigenspace, Training as centered pill nav; KPI cards as dark metric tiles with blue accent values and mini area charts; right decision ledger card with glowing signal label; chart canvas dominant.

## Final detailed observations (from editor close-ups)
The reference is Capi Product's "Trading webiste - Glowing & Gradient Text Style" (CC BY 4.0). Verified details from the editor and community preview:

1. **Global**: near-black background with a huge radial nebula glow, deep blue at bottom-right, faint cyan trace; canvas dark navy #05080F–#0A0F1E. White sans headline (Poppins-like, bold 48-64px). Muted gray subtext (#9CA3AF).
2. **Nav bar** (floating, ~56px): left = small logo square (blue icon in rounded white/light square) + "CapiTrade" wordmark in white; CENTER = pill-shaped segmented nav ("Chart", "Markets", "News", "Community", "More") with light translucent fill and active tab highlighted; right = "Login" ghost text link + "Start Free Trial" pill with white hairline border.
3. **Badge**: small glass pill with sparkle icon + "New: Our AI integration just landed", hairline border.
4. **Headline**: "Discover endless possibilities in the world of Trading." — "endless possibilities" in light gray, rest white; the word "Trading" ends with a gradient dot/glow.
5. **Checks row**: pen/pen icons with "Fast Trading", shield "Secure & Reliable", clock "Continuous Market Updates" in small gray text.
6. **CTAs**: "Start Trading" = electric blue #2F6BFF filled pill with small flame/rocket icon and soft blue glow; "Try Demo" = dark pill, hairline border, globe icon.
7. **Right mockups**: floating tilted dashboard cards (dark, angled) with area charts; one has circular play button (video demo).
8. **Bottom app mockup** (dark desktop app): title bar with traffic-light dots + tabs "Overview | AI Support" + right pill "Add keyword", "All Steps", dropdown "Sep 24 → Oct 10", download icon button; left sidebar rows: Project (dropdown), Quick actions, Insight box; main area header "Performance" with info icon; grid of dark metric tiles ("Today" value 92.28? blue, "Profit/Loss 0.0 / -0.0", "Number of..."), each with tiny area chart; labels row below: "Today, Profile, Total Buy, Total Sell, MiniChart".
9. **Floating feature card** (dark glass): 3 rows with icons — "Start Trading", "Be smart & decisive", "Continuous Market updates"; buttons "Start Trading" (blue glow) / "Try Free" ghost.
10. Buttons everywhere = pills (999px radius), hairline 1px rgba(255,255,255,.08) borders, glass fills rgba(255,255,255,.05).
11. Accent palette to reuse: primary #2F6BFF, glow #3B82F6, gradient text cyan #22D3EE → blue #3B82F6, surface #0A0F1E, border rgba(255,255,255,.08), text #FFFFFF / #9CA3AF, muted label #6B7280.
12. Figma community preview URL for hero shot: https://s3-alpha.figma.com/thumbnails/... (not extracted); not needed further.

## QuantGrad v4 mapping (final, for frontend build)
Build a single-page React app (Vite) "QuantGrad" with: landing/hero intro → terminal dashboard. Landing hero mirrors reference exactly (logo pill, center pill nav, glass badge, gradient headline, checks, glow CTA + ghost CTA, floating tilted chart cards, bottom app mockup). Dashboard: same nav with 6 tabs (Market, Models, Risk Lab, Macro, Eigenspace, Training); left KPI column (3 tiles, blue accent numbers + mini area charts); center TradingView chart canvas (lightweight-charts); right decision ledger card (signal pill glowing blue, entries); top bar brand + connection status + analyze button (Start Analysis glowing pill); offline state card matching reference when server down.
