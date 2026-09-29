import { Link, useRouterState, useNavigate } from "@tanstack/react-router";
import { useState, type ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  Clock,
  Cpu,
  Globe,
  LayoutDashboard,
  MapPin,
  Menu,
  Radio,
  RefreshCw,
  Sliders,
  Sparkles,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSkyGuard } from "@/data/store";
import { DemoControlBar } from "@/components/DemoControlBar";
import { AtmosphereBackground } from "@/components/AtmosphereBackground";

const NAV_TABS = [
  {
    label: "Overview",
    to: "/",
    icon: LayoutDashboard,
    badge: null,
  },
  {
    label: "Stations & Map",
    to: "/station",
    icon: Globe,
    badge: null,
  },
  {
    label: "Anomaly Detection",
    to: "/diagnostics",
    icon: AlertTriangle,
    badge: "anomalies",
  },
  {
    label: "Station Health",
    to: "/sensors",
    icon: Activity,
    badge: null,
  },
  {
    label: "Calibration & Admin",
    to: "/admin",
    icon: Sliders,
    badge: null,
  },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const {
    dataset,
    selectedStationId,
    setSelectedStationId,
    connectionState,
    clock,
    recentAnomalies,
    refresh,
    refreshing,
    dataSource,
    startDemo,
    returnToLive,
  } = useSkyGuard();

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  const activeStation =
    dataset.stations.find((s) => s.station_id === selectedStationId) || dataset.stations[0];

  const anomalyCount = recentAnomalies.length;

  const connectionMeta = (() => {
    switch (connectionState) {
      case "WARMUP":
        return {
          label: "WARMUP",
          dot: "bg-amber-400",
          text: "text-amber-400",
          bg: "bg-amber-400/10 border-amber-400/20",
        };
      case "SIMULATION":
        return {
          label: "SIMULATION",
          dot: "bg-amber-400",
          text: "text-amber-400",
          bg: "bg-amber-400/10 border-amber-400/20",
        };
      case "LOST":
        return {
          label: "OFFLINE",
          dot: "bg-rose-500",
          text: "text-rose-400",
          bg: "bg-rose-500/10 border-rose-500/20",
        };
      case "LIVE":
      default:
        return {
          label: "LIVE",
          dot: "bg-emerald-400",
          text: "text-emerald-300",
          bg: "bg-emerald-500/10 border-emerald-500/20",
        };
    }
  })();

  return (
    <div className="relative min-h-screen text-foreground font-sans antialiased selection:bg-sky-500/20 selection:text-white pb-12">
      {/* Calm ambient background */}
      <div className="sg-atmosphere" aria-hidden="true" />
      <AtmosphereBackground />

      {/* ==================== PROFESSIONAL METEOROLOGICAL TOP NAVIGATION ==================== */}
      <header className="sticky top-0 z-50 w-full border-b border-border/80 bg-card/95 backdrop-blur-md">
        <div className="max-w-[1720px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
          {/* Left: Meteorological Identity */}
          <div className="flex items-center gap-3 shrink-0">
            <button
              type="button"
              className="lg:hidden p-1.5 text-muted-foreground hover:text-foreground rounded-lg hover:bg-muted transition-colors"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>

            <Link to="/" className="flex items-center gap-2.5 group cursor-pointer">
              <div className="w-8 h-8 rounded-lg bg-sky-950/70 border border-sky-500/30 flex items-center justify-center text-sky-400">
                <Radio className="h-4 w-4" />
              </div>
              <div className="flex flex-col">
                <div className="flex items-center gap-1.5">
                  <span className="text-sm font-semibold tracking-tight text-foreground font-sans">
                    SkyGuard AI
                  </span>
                  <span className="text-[10px] font-mono font-medium px-1.5 py-0.2 rounded bg-muted text-muted-foreground border border-border">
                    AWS OPS
                  </span>
                </div>
                <span className="text-[10px] text-muted-foreground hidden sm:inline">
                  Weather Station Telemetry & Anomaly Intelligence
                </span>
              </div>
            </Link>
          </div>

          {/* Center: Clean Professional Tabs */}
          <nav className="hidden lg:flex items-center gap-1 bg-muted/40 p-1 rounded-lg border border-border">
            {NAV_TABS.map((tab) => {
              const isActive = pathname === tab.to;
              const Icon = tab.icon;
              const hasAlert = tab.badge === "anomalies" && anomalyCount > 0;

              return (
                <Link
                  key={tab.to}
                  to={tab.to}
                  className={cn(
                    "relative flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors select-none",
                    isActive
                      ? "bg-secondary text-foreground shadow-xs border border-border"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/60",
                  )}
                >
                  <Icon
                    className={cn(
                      "h-3.5 w-3.5",
                      isActive ? "text-sky-400" : "text-muted-foreground",
                    )}
                  />
                  <span>{tab.label}</span>
                  {hasAlert && (
                    <span className="inline-flex items-center justify-center px-1.5 py-0.2 text-[10px] font-mono font-bold bg-rose-500 text-white rounded-full min-w-[16px]">
                      {anomalyCount}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Right: Station Quick Selector + Live State + Clock */}
          <div className="flex items-center gap-2.5 shrink-0">
            {/* Station Quick Selector */}
            <div className="relative flex items-center">
              <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-muted/50 border border-border text-xs">
                <MapPin className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <select
                  aria-label="Select AWS Station"
                  value={selectedStationId}
                  onChange={(e) => setSelectedStationId(e.target.value)}
                  className="bg-transparent text-xs font-medium text-foreground cursor-pointer outline-hidden pr-2 max-w-[140px] sm:max-w-[200px] truncate"
                >
                  {dataset.stations.map((s) => (
                    <option
                      key={s.station_id}
                      value={s.station_id}
                      className="bg-popover text-foreground"
                    >
                      {s.station_name} ({s.station_id})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Truthful Connection Status Pill */}
            <div
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-mono font-medium",
                connectionMeta.bg,
                connectionMeta.text,
              )}
              title={`Connection status: ${connectionMeta.label}`}
            >
              <span className={cn("w-1.5 h-1.5 rounded-full", connectionMeta.dot)} />
              <span>{connectionMeta.label}</span>
            </div>

            {/* Live / Demo Mode Toggle */}
            {dataSource === "LIVE" ? (
              <button
                type="button"
                onClick={() => {
                  navigate({ to: "/" });
                  startDemo();
                }}
                className="hidden sm:inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground border border-border text-xs font-medium transition-colors cursor-pointer"
                title="Launch Synthetic Fault Simulation Testbed"
              >
                <Sparkles className="h-3 w-3 text-amber-400" />
                <span>Simulation</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={returnToLive}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/25 text-xs font-medium transition-colors cursor-pointer"
                title="Return to Live Telemetry Feed"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                <span>Return Live</span>
              </button>
            )}

            {/* Refresh Button */}
            <button
              onClick={refresh}
              title="Refresh Telemetry Feed"
              disabled={refreshing}
              className="text-muted-foreground hover:text-foreground p-1.5 rounded-md hover:bg-muted transition-colors cursor-pointer border border-border"
            >
              <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin text-sky-400")} />
            </button>

            {/* Sync Timestamp */}
            <div className="hidden md:flex items-center gap-1 px-2 py-1 rounded-md bg-muted/30 border border-border/60 text-xs font-mono text-muted-foreground">
              <Clock className="h-3 w-3" />
              <span>{clock}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Drawer Navigation */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm lg:hidden flex flex-col justify-end p-3">
          <div className="bg-card text-foreground p-4 rounded-xl space-y-3 border border-border shadow-lg">
            <div className="flex items-center justify-between pb-2 border-b border-border">
              <div className="flex items-center gap-2">
                <Radio className="h-4 w-4 text-sky-400" />
                <span className="font-semibold text-sm">SkyGuard AI Navigation</span>
              </div>
              <button
                onClick={() => setMobileMenuOpen(false)}
                className="text-muted-foreground hover:text-foreground p-1"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <nav className="flex flex-col gap-1">
              {NAV_TABS.map((tab) => {
                const Icon = tab.icon;
                const isActive = pathname === tab.to;
                return (
                  <Link
                    key={tab.to}
                    to={tab.to}
                    onClick={() => setMobileMenuOpen(false)}
                    className={cn(
                      "flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors",
                      isActive
                        ? "bg-secondary text-foreground border border-border"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground",
                    )}
                  >
                    <div className="flex items-center gap-2.5">
                      <Icon className="h-3.5 w-3.5 text-sky-400" />
                      <span>{tab.label}</span>
                    </div>
                    {tab.badge === "anomalies" && anomalyCount > 0 && (
                      <span className="px-1.5 py-0.2 text-[10px] font-mono font-bold bg-rose-500 text-white rounded-full">
                        {anomalyCount}
                      </span>
                    )}
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
      )}

      {/* ==================== MAIN CONTENT AREA ==================== */}
      <main className="relative z-10 max-w-[1720px] mx-auto px-4 sm:px-6 pt-4">
        <DemoControlBar />
        {children}
      </main>
    </div>
  );
}
