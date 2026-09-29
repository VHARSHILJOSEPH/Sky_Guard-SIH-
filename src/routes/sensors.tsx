import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Gauge,
  HelpCircle,
  Radio,
  RefreshCw,
  Search,
  ShieldAlert,
  Wrench,
  Activity,
  Flame,
  Thermometer,
  Droplets,
  ArrowRight,
  Filter,
  MapPin,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSkyGuard } from "@/data/store";
import { formatDateTime, type SensorHealth, isImdStation } from "@/data/skyguard";
import { WeatherService } from "@/data/weather-service";

export const Route = createFileRoute("/sensors")({
  head: () => ({
    meta: [
      { title: "Sensor Hardware Health & Diagnostics | SkyGuard AI" },
      {
        name: "description",
        content:
          "AWS hardware reliability, sensor drift tracking, communication latency, and predictive maintenance triage.",
      },
    ],
  }),
  component: SensorHealthPage,
});

export type CanonicalHealthState =
  "ALL" | "HEALTHY" | "DEGRADED" | "UNHEALTHY" | "OFFLINE" | "WARMUP";

export interface StationHealthAuditRecord {
  station_id?: string;
  status?: string;
  health_score?: number;
  score?: number;
  reasons?: string[];
  contributors?: string[];
  sensor_breakdowns?: Record<
    string,
    {
      status?: string;
      temperature_anomaly_rate?: number;
      pressure_anomaly_rate?: number;
      humidity_anomaly_rate?: number;
    }
  >;
  [key: string]: unknown;
}

export function SensorHealthPage() {
  const { dataset, setSelectedStationId } = useSkyGuard();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<CanonicalHealthState>("ALL");
  const [search, setSearch] = useState("");
  const [healthMap, setHealthMap] = useState<Record<string, StationHealthAuditRecord>>({});

  useEffect(() => {
    WeatherService.fetchStationHealth().then((res) => {
      const data = res as { records?: StationHealthAuditRecord[] } | null;
      if (data && Array.isArray(data.records)) {
        const mapped: Record<string, StationHealthAuditRecord> = {};
        for (const rec of data.records) {
          if (rec.station_id) {
            mapped[rec.station_id] = rec;
          }
        }
        setHealthMap(mapped);
      }
    });
  }, []);

  type StationItem = (typeof dataset.stations)[number];

  const getStationCanonicalHealth = useCallback(
    (
      s: StationItem,
    ): {
      status: "HEALTHY" | "DEGRADED" | "UNHEALTHY" | "OFFLINE" | "WARMUP";
      score: number;
      contributors: string[];
      primaryReason: string;
    } => {
      const hr = healthMap[s.station_id];
      if (hr) {
        const rawStatus = (hr.status || "").toUpperCase();
        let normStatus: "HEALTHY" | "DEGRADED" | "UNHEALTHY" | "OFFLINE" | "WARMUP" = "HEALTHY";
        if (rawStatus.includes("OFFLINE")) normStatus = "OFFLINE";
        else if (rawStatus.includes("UNHEALTHY") || rawStatus.includes("CRITICAL"))
          normStatus = "UNHEALTHY";
        else if (rawStatus.includes("DEGRADED") || rawStatus.includes("WARNING"))
          normStatus = "DEGRADED";
        else if (rawStatus.includes("WARMUP")) normStatus = "WARMUP";
        else normStatus = "HEALTHY";

        return {
          status: normStatus,
          score:
            typeof hr.health_score === "number"
              ? hr.health_score
              : typeof hr.score === "number"
                ? hr.score
                : 95,
          contributors:
            Array.isArray(hr.contributors) && hr.contributors.length > 0
              ? hr.contributors
              : Array.isArray(hr.reasons) && hr.reasons.length > 0
                ? hr.reasons
                : ["Transducer impedance nominal", "Zero drift within calibrated threshold"],
          primaryReason:
            Array.isArray(hr.reasons) && hr.reasons.length > 0 && typeof hr.reasons[0] === "string"
              ? hr.reasons[0]
              : normStatus === "HEALTHY"
                ? "All channels operating normally"
                : `Active flag: ${rawStatus || "Telemetry inconsistency"}`,
        };
      }

      // Baseline fallback derivation from station state
      if (s.health === "CRITICAL" || s.maintenance_priority === "P1") {
        return {
          status: "UNHEALTHY",
          score: 38,
          contributors: [
            "Persistent hardware anomaly or data corruption",
            "Transducer validation failure",
          ],
          primaryReason:
            "Critical hardware degradation or persistent data corruption requiring field maintenance",
        };
      }
      if (s.health === "WARNING" || s.maintenance_priority === "P2") {
        return {
          status: "DEGRADED",
          score: 74,
          contributors: [
            "Systematic baseline drift detected",
            "Spatial neighbor residual monitoring active",
          ],
          primaryReason: "Operational sensor degradation: recurring departures or transducer drift",
        };
      }
      if (s.health === "OFFLINE") {
        return {
          status: "OFFLINE",
          score: 0,
          contributors: [
            "Complete telemetry packet interruption",
            "No transmission received within timeout",
          ],
          primaryReason:
            "Telemetry communication gap: station transmission interrupted or missing packet",
        };
      }

      return {
        status: "HEALTHY",
        score: 98,
        contributors: [
          "All physical parameters within certified operational envelopes",
          "Normal packet reception",
        ],
        primaryReason:
          "Nominal operation: all sensor channels calibrated and transmitting within tolerances",
      };
    },
    [healthMap],
  );

  const counts = useMemo(() => {
    const c = { HEALTHY: 0, DEGRADED: 0, UNHEALTHY: 0, OFFLINE: 0, WARMUP: 0 };
    dataset.stations.forEach((s) => {
      const h = getStationCanonicalHealth(s);
      c[h.status] += 1;
    });
    return c;
  }, [dataset.stations, getStationCanonicalHealth]);

  const cards = useMemo(() => {
    const q = search.trim().toLowerCase();
    return dataset.stations
      .map((s) => ({
        ...s,
        canonicalHealth: getStationCanonicalHealth(s),
      }))
      .filter((item) => {
        const matchesFilter = filter === "ALL" || item.canonicalHealth.status === filter;
        const matchesSearch =
          !q ||
          item.station_id.toLowerCase().includes(q) ||
          item.station_name.toLowerCase().includes(q) ||
          item.state.toLowerCase().includes(q);
        return matchesFilter && matchesSearch;
      })
      .sort((a, b) => a.maintenance_priority.localeCompare(b.maintenance_priority));
  }, [dataset.stations, filter, search, getStationCanonicalHealth]);

  return (
    <div className="space-y-5">
      {/* ==================== 1. EXECUTIVE HEADER & SEARCH ==================== */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-1">
        <div>
          <div className="flex items-center gap-2">
            <span className="provenance-tag provenance-live">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Hardware Reliability Audit
            </span>
            <span className="text-xs font-mono text-muted-foreground">
              {dataset.stations.length} Monitored AWS Stations
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground mt-1 font-sans">
            Station Health & Sensor Hardware Diagnostics
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5 max-w-3xl">
            Transducer calibration drift, packet continuity, frozen sensor variance tests, and
            predictive maintenance triage.
          </p>
        </div>

        {/* Search Field */}
        <div className="relative w-full sm:w-72 shrink-0">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search station ID, name, city..."
            className="h-9 w-full pl-9 pr-3 text-xs bg-muted/40 border border-border rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-hidden focus:border-sky-500/50"
          />
        </div>
      </div>

      {/* ==================== 2. HEALTH CATEGORY TABS ==================== */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <button
          type="button"
          onClick={() => setFilter("ALL")}
          className={cn(
            "bg-card border rounded-lg p-3 text-left transition-colors cursor-pointer shadow-xs",
            filter === "ALL" ? "border-sky-500/50 bg-secondary" : "border-border hover:bg-muted/30",
          )}
        >
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>All Stations</span>
            <Activity className="h-3.5 w-3.5 text-sky-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold font-sans text-foreground">
            {dataset.stations.length}
          </div>
          <p className="text-[10px] text-muted-foreground mt-0.5">Full fleet registry</p>
        </button>

        <button
          type="button"
          onClick={() => setFilter("HEALTHY")}
          className={cn(
            "bg-card border rounded-lg p-3 text-left transition-colors cursor-pointer shadow-xs",
            filter === "HEALTHY"
              ? "border-emerald-500/50 bg-secondary"
              : "border-border hover:bg-muted/30",
          )}
        >
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Healthy</span>
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold font-sans text-emerald-400">
            {counts.HEALTHY}
          </div>
          <p className="text-[10px] text-muted-foreground mt-0.5">Certified operational</p>
        </button>

        <button
          type="button"
          onClick={() => setFilter("DEGRADED")}
          className={cn(
            "bg-card border rounded-lg p-3 text-left transition-colors cursor-pointer shadow-xs",
            filter === "DEGRADED"
              ? "border-amber-500/50 bg-secondary"
              : "border-border hover:bg-muted/30",
          )}
        >
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Degraded</span>
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold font-sans text-amber-400">{counts.DEGRADED}</div>
          <p className="text-[10px] text-muted-foreground mt-0.5">Transducer drift active</p>
        </button>

        <button
          type="button"
          onClick={() => setFilter("UNHEALTHY")}
          className={cn(
            "bg-card border rounded-lg p-3 text-left transition-colors cursor-pointer shadow-xs",
            filter === "UNHEALTHY"
              ? "border-rose-500/50 bg-secondary"
              : "border-border hover:bg-muted/30",
          )}
        >
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Unhealthy</span>
            <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold font-sans text-rose-400">{counts.UNHEALTHY}</div>
          <p className="text-[10px] text-muted-foreground mt-0.5">Field inspection required</p>
        </button>

        <button
          type="button"
          onClick={() => setFilter("OFFLINE")}
          className={cn(
            "bg-card border rounded-lg p-3 text-left transition-colors cursor-pointer shadow-xs",
            filter === "OFFLINE"
              ? "border-slate-500/50 bg-secondary"
              : "border-border hover:bg-muted/30",
          )}
        >
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Offline</span>
            <Radio className="h-3.5 w-3.5 text-slate-400" />
          </div>
          <div className="mt-1.5 text-xl font-bold font-sans text-slate-400">{counts.OFFLINE}</div>
          <p className="text-[10px] text-muted-foreground mt-0.5">Packet reception timeout</p>
        </button>
      </div>

      {/* ==================== 3. SENSOR HARDWARE TILES ==================== */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {cards.map((station) => {
          const health = station.canonicalHealth;
          const isHealthy = health.status === "HEALTHY";
          const isDegraded = health.status === "DEGRADED";
          const isUnhealthy = health.status === "UNHEALTHY";

          return (
            <div
              key={station.station_id}
              className={cn(
                "bg-card border rounded-xl p-4 shadow-xs flex flex-col justify-between space-y-3 transition-colors",
                isUnhealthy
                  ? "border-rose-500/40"
                  : isDegraded
                    ? "border-amber-500/40"
                    : "border-border hover:border-border/80",
              )}
            >
              <div className="space-y-2.5">
                {/* Header: Station Name & Status */}
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="font-semibold text-sm text-foreground">
                      {station.station_name}
                    </h3>
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground mt-0.5 font-mono">
                      <MapPin className="h-3 w-3 text-sky-400" />
                      <span>
                        {station.district}, {station.state} · {station.station_id}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col items-end gap-0.5">
                    <span
                      className={cn(
                        "text-[10px] font-mono font-medium px-2 py-0.5 rounded border uppercase",
                        isHealthy
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                          : isDegraded
                            ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                            : "bg-rose-500/10 text-rose-400 border-rose-500/20",
                      )}
                    >
                      {health.status}
                    </span>
                    <span className="text-xs font-mono font-semibold text-foreground">
                      Health: {health.score}%
                    </span>
                  </div>
                </div>

                {/* Transducer Channel Readouts */}
                <div className="grid grid-cols-3 gap-1.5 pt-2 border-t border-border/60 text-center text-xs">
                  <div className="p-1.5 rounded bg-muted/30 border border-border/50">
                    <Thermometer className="h-3.5 w-3.5 text-sky-400 mx-auto" />
                    <span className="text-[10px] text-muted-foreground block mt-0.5">Temp</span>
                    <span className="font-mono font-semibold text-foreground text-xs">
                      {station.temperature !== null ? `${station.temperature.toFixed(1)}°C` : "—"}
                    </span>
                  </div>

                  <div className="p-1.5 rounded bg-muted/30 border border-border/50">
                    <Droplets className="h-3.5 w-3.5 text-emerald-400 mx-auto" />
                    <span className="text-[10px] text-muted-foreground block mt-0.5">Humidity</span>
                    <span className="font-mono font-semibold text-foreground text-xs">
                      {station.humidity !== null ? `${Math.round(station.humidity)}%` : "—"}
                    </span>
                  </div>

                  <div className="p-1.5 rounded bg-muted/30 border border-border/50">
                    <Gauge className="h-3.5 w-3.5 text-indigo-400 mx-auto" />
                    <span className="text-[10px] text-muted-foreground block mt-0.5">Pressure</span>
                    <span className="font-mono font-semibold text-foreground text-xs">
                      {station.pressure !== null ? `${station.pressure.toFixed(0)}` : "—"}
                    </span>
                  </div>
                </div>

                {/* Primary Assessment */}
                <div className="p-2 rounded bg-muted/20 border border-border/50 space-y-0.5 text-xs">
                  <span className="text-[10px] font-mono text-muted-foreground uppercase font-medium">
                    Diagnostic Assessment:
                  </span>
                  <p className="text-muted-foreground text-[11px] leading-relaxed">
                    {health.primaryReason}
                  </p>
                </div>
              </div>

              {/* Card Footer: Priority & Action */}
              <div className="pt-2 border-t border-border/60 flex items-center justify-between text-xs">
                <span className="text-[10.5px] font-mono text-muted-foreground">
                  Maintenance:{" "}
                  <strong className="text-foreground">Tier {station.maintenance_priority}</strong>
                </span>

                <button
                  type="button"
                  onClick={() => {
                    setSelectedStationId(station.station_id);
                    navigate({ to: "/station" });
                  }}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-muted hover:bg-secondary text-foreground text-xs font-medium border border-border transition-colors cursor-pointer"
                >
                  <span>Inspect</span>
                  <ArrowRight className="h-3 w-3" />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
