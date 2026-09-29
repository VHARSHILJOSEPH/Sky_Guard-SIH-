---
name: SkyGuard AI Precision Telemetry
colors:
  surface: "#f8f9ff"
  surface-dim: "#cbdbf5"
  surface-bright: "#f8f9ff"
  surface-container-lowest: "#ffffff"
  surface-container-low: "#eff4ff"
  surface-container: "#e5eeff"
  surface-container-high: "#dce9ff"
  surface-container-highest: "#d3e4fe"
  on-surface: "#0b1c30"
  on-surface-variant: "#3f4850"
  inverse-surface: "#213145"
  inverse-on-surface: "#eaf1ff"
  outline: "#707881"
  outline-variant: "#bfc7d2"
  surface-tint: "#006398"
  primary: "#006194"
  on-primary: "#ffffff"
  primary-container: "#007bb9"
  on-primary-container: "#fdfcff"
  inverse-primary: "#93ccff"
  secondary: "#565e74"
  on-secondary: "#ffffff"
  secondary-container: "#dae2fd"
  on-secondary-container: "#5c647a"
  tertiary: "#00628d"
  on-tertiary: "#ffffff"
  tertiary-container: "#007cb1"
  on-tertiary-container: "#fcfcff"
  error: "#ba1a1a"
  on-error: "#ffffff"
  error-container: "#ffdad6"
  on-error-container: "#93000a"
  primary-fixed: "#cce5ff"
  primary-fixed-dim: "#93ccff"
  on-primary-fixed: "#001d31"
  on-primary-fixed-variant: "#004b73"
  secondary-fixed: "#dae2fd"
  secondary-fixed-dim: "#bec6e0"
  on-secondary-fixed: "#131b2e"
  on-secondary-fixed-variant: "#3f465c"
  tertiary-fixed: "#c9e6ff"
  tertiary-fixed-dim: "#89ceff"
  on-tertiary-fixed: "#001e2f"
  on-tertiary-fixed-variant: "#004c6e"
  background: "#f8f9ff"
  on-background: "#0b1c30"
  surface-variant: "#d3e4fe"
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 30px
    fontWeight: "700"
    lineHeight: 38px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: "600"
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: "600"
    lineHeight: 24px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: "600"
    lineHeight: 20px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: "400"
    lineHeight: 22px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: "400"
    lineHeight: 18px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: "400"
    lineHeight: 16px
  label-numeric-lg:
    fontFamily: JetBrains Mono
    fontSize: 22px
    fontWeight: "600"
    lineHeight: 26px
    letterSpacing: -0.02em
  label-numeric-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: "500"
    lineHeight: 18px
    letterSpacing: 0em
  label-numeric-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: "500"
    lineHeight: 14px
    letterSpacing: 0.02em
  caption-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: "600"
    lineHeight: 14px
    letterSpacing: 0.06em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter-xs: 0.25rem
  gutter-sm: 0.5rem
  gutter-md: 0.75rem
  gutter-lg: 1rem
  gutter-xl: 1.5rem
  container-padding: 1.25rem
  sidebar-width-expanded: 16rem
  sidebar-width-collapsed: 4rem
  header-height: 3.5rem
---

## Brand & Style

This design system serves mission-critical aerospace, meteorological, and environmental operations. It targets operational meteorologists, atmospheric scientists, aviation dispatchers, and autonomous system operators who require split-second situational awareness and zero-latency data comprehension.

The aesthetic philosophy is **Technical Instrument Modernism**:

- **Utilitarian Rigor:** Prioritizes scan-efficiency, typographic precision, and functional data density over decorative visual flourish.
- **Instrumental Focus:** Surfaces simulate a high-fidelity scientific console. High contrast, precise hairline partitions, and calculated spatial economy prevent cognitive fatigue during extended operational monitoring.
- **Controlled Chromatic Hierarchy:** Color is reserved strictly for operational state changes, status vectors, and environmental threshold alerts. Decorative gradients and heavy frosted glassmorphism are explicitly avoided.

## Colors

The palette balances a luminous, low-fatigue workspace with a commanding command-dock sidebar and semantic telemetry alerts.

### Workspace & Frame Tokens

- **Canvas Base:** `#f8fafc` (Slate 50) transitioning to `#f1f5f9` (Slate 100) for structured page backgrounds.
- **Surface Card:** `#ffffff` pure white for interactive and telemetry viewport containers.
- **Surface Muted:** `#f1f5f9` for table headers, embedded sub-panels, and control grooves.
- **Surface Border:** `#e2e8f0` (Slate 200) for clean, structural 1px hairline separation.
- **Surface Border Strong:** `#cbd5e1` (Slate 300) for interactive focus zones and active dividers.

### Operational Rail / Dark Frame

- **Sidebar Surface:** `#0f172a` (Slate 900) to `#090d16` (Deep Obsidian).
- **Sidebar Border:** `#1e293b` (Slate 800) hairline perimeter.
- **Sidebar Text Muted:** `#94a3b8` (Slate 400).
- **Sidebar Text Active:** `#ffffff` with `#38bdf8` (Ice Cyan) glow or indicator dots.

### Semantic Telemetry & Accents

- **Primary / Telemetry Sky:** `#0284c7` (Primary interactive) and `#0ea5e9` (Active telemetry focus).
- **Operational Nominal / Live:** `#10b981` (Emerald 500) paired with background `#ecfdf5`.
- **Advisory / Warning:** `#f59e0b` (Amber 500) paired with background `#fffbeb`.
- **Critical / Anomaly:** `#ef4444` (Red 500) paired with background `#fef2f2`.
- **Atmospheric Pressure / Neutral Gauge:** `#64748b` (Slate 500).

Color contrast must meet WCAG AAA requirements for all critical telemetry readouts and AA for peripheral UI metadata.

## Typography

Typographic hierarchy enforces quick triage through dual font engine distribution:

1. **Primary Interface (`Inter`):** Handles semantic navigation, operational headings, contextual commentary, and structured data tables. All numeric strings rendered in `Inter` must enforce `font-feature-settings: "tnum" 1, "cv05" 1` to guarantee vertical numeric alignment across stream updates.
2. **Telemetry / Value Metrics (`JetBrains Mono`):** Dedicated to sensor values, geographic coordinates (lat/long), timestamp clocks (UTC), wind vectors, barometric shifts, and confidence coefficients.

### Rules & Formatting

- **Data Densities:** Never use weights below `400`. Headline titles in operational cards max out at `14px` (`headline-sm`) to preserve primary real estate for visualization and data matrices.
- **Uppercase Badges & Column Headers:** Utilize `caption-caps` transformed to `uppercase` with `0.06em` letter tracking to establish structural boundaries within data grids.

## Layout & Spacing

The operational interface is structured on a 4px base rhythm and an adaptive 12-column grid configured for high-density spatial economy.

### Frame Architecture

- **Global Left Operational Rail:** Fixed dark rail (`#0f172a`), toggleable between `16rem` (expanded) and `4rem` (icon-only telemetry mode).
- **Sub-Header Console:** Fixed `3.5rem` utility strip displaying live UTC synchronizers, system integrity diagnostics, and global threat filters.
- **Main Canvas:** Fluid viewport with `1.25rem` outer padding, ensuring operational consoles fill wide screen displays without dead whitespace.

### Grid & Density Rhythms

- **Telemetry Grid:** Uses `0.75rem` (`gutter-md`) gaps between real-time chart widgets and metric cards to maximize operational screen real estate.
- **Internal Card Padding:** Compact `0.75rem` to `1rem` ceiling. Avoid generous whitespace cushions typical of standard consumer SaaS.
- **Breakpoints:**
  - **Desktop Ultra-Wide (≥1600px):** 12-column continuous telemetry with simultaneous multi-radar stream and log feeds.
  - **Desktop Standard (1024px – 1599px):** 12-column with responsive collapse of secondary diagnostic rails into drawers.
  - **Tactical Tablet (768px – 1023px):** 6-column reflow; sidebar collapses into persistent 4rem instrument dock.

## Elevation & Depth

Visual hierarchy uses crisp hairline perimeters paired with disciplined, near-ambient low-opacity drop shadows. This preserves a flat, technical instrument look while maintaining clear layered focus.

### Surface Elevation Levels

- **Level 0 (Workspace Canvas):** `#f8fafc` flat background.
- **Level 1 (Operational Cards & Data Cells):** Pure `#ffffff` framed by a 1px solid border in `#e2e8f0`. Shadow: `0 1px 2px 0 rgba(15, 23, 42, 0.04)`.
- **Level 2 (Active Tooltips, Popovers, & Dropdown Trays):** Pure `#ffffff` surface with a 1px `#cbd5e1` border. Shadow: `0 4px 12px -2px rgba(15, 23, 42, 0.08), 0 2px 4px -1px rgba(15, 23, 42, 0.04)`.
- **Level 3 (Modal Alerts & Emergency Override Panels):** Pure `#ffffff` surface with a 1px `#94a3b8` border. Shadow: `0 20px 25px -5px rgba(15, 23, 42, 0.12), 0 8px 10px -6px rgba(15, 23, 42, 0.06)`.

### Operational Glass & Highlighting

- Heavy blurred glass is prohibited within the daylight workspace.
- The only permissible backdrop filter is an ultra-subtle `backdrop-filter: blur(4px)` with an `rgba(255, 255, 255, 0.85)` tint applied strictly to sticky metric table headers during vertical telemetry scrolling.

## Shapes

The design system adopts a **Soft Technical (`1`)** shape geometry.

- **Standard Cards, Data Panes, & Viewports:** `0.375rem` (6px) or `rounded` (`0.25rem` / 4px) to retain sharp architectural lines reminiscent of hardware physical monitors.
- **Buttons, Toggle Switches, & Form Fields:** `0.25rem` (4px).
- **Status Badges, Indicators, & Vector Tags:** `0.25rem` (4px) with crisp corners or strict circular radius `9999px` purely for pulsing live indicators.
- **Pill shapes (`rounded-full`) are strictly restricted** to operational status dots or micro-chips indicating sensor connectivity (e.g., satellite ping nodes).

## Components

### Buttons & Action Triggers

- **Primary Telemetry Button:** Filled with `#0284c7`, text in white (`#ffffff`), `0.25rem` border radius, font `13px` weight `500`. Hover: `#0369a1`. Active: `#075985`. Focus ring: 2px offset with `#38bdf8`.
- **Secondary Tool Button:** `#ffffff` surface, 1px border `#cbd5e1`, text `#334155`. Hover: background `#f8fafc`, border `#94a3b8`.
- **Destructive/Override Button:** Background `#fef2f2`, 1px border `#fca5a5`, text `#b91c1c`. Active: `#dc2626` background with white text.

### Metric Cards & Telemetry Tiles

- **Structure:** White container (`#ffffff`), 1px `#e2e8f0` border, `0.75rem` internal padding.
- **Header:** Card title in `caption-caps` (`#64748b`), right-aligned auxiliary unit or delta badge.
- **Value Core:** Primary sensor reading set in `label-numeric-lg` (`#0f172a`), inline engineering unit (e.g., `hPa`, `kts`, `dBZ`) set in `body-sm` (`#64748b`).
- **Footer:** Micro sparkline or delta readout: green `#10b981` with leading `+` for positive trends, red `#ef4444` for critical drops, accompanied by reference time in `JetBrains Mono` (`11px`).

### Status Badges & Chips

- **Nominal / Live:** Light emerald background `#ecfdf5`, border `#a7f3d0`, text `#065f46`, accompanied by an active `#10b981` pulsing indicator dot.
- **Advisory / Warning:** Light amber background `#fffbeb`, border `#fde68a`, text `#92400e`.
- **Critical Anomaly:** Light red background `#fef2f2`, border `#fecaca`, text `#991b1b`, bold weight `600`.
- All badges use `label-numeric-sm` or `caption-caps`, height fixed at `20px`, padding `0 6px`.

### Data Tables (High-Density Telemetry View)

- **Header Row:** Height `28px`, background `#f8fafc`, border-bottom `1px solid #cbd5e1`, text in `caption-caps` `#475569`.
- **Data Rows:** Height `32px` for high-density, alternating row background optional (even: `#ffffff`, odd: `#f8fafc`). Border-bottom `1px solid #f1f5f9`.
- **Cell Content:** Left-aligned strings (`Inter`, `13px`), right-aligned numeric metrics (`JetBrains Mono`, `13px`). Hover state: `#f1f5f9`.

### Form Controls & Filter Bars

- **Inputs & Selects:** Height `32px`, border `1px solid #cbd5e1`, background `#ffffff`, text `#0f172a`, font size `13px`. Placeholder `#94a3b8`. Focus: border `#0284c7`, box-shadow `0 0 0 1px #0284c7`.
- **Segmented Range Controls (Time Horizon):** Enclosed pill switch in `#f1f5f9` tray. Active segment: white card with 1px border `#cbd5e1`, text `#0f172a`, font weight `500`.
