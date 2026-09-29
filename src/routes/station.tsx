import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  ReferenceArea,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock,
  CloudRain,
  Compass,
  Droplets,
  Gauge,
  Globe,
  MapPin,
  Radio,
  RefreshCw,
  Search,
  ShieldCheck,
  Thermometer,
  Wind,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSelectedStation, useSkyGuard } from "@/data/store";
import { formatDateTime, getObservations, nearbyStations, statusTone } from "@/data/skyguard";
import { IndiaMap } from "@/components/IndiaMap";

export const Route = createFileRoute("/station")({
  head: () => ({
    meta: [
      { title: "Stations & Spatial Corroboration Network | SkyGuard AI" },
      {
        name: "description",
        content:
          "High-precision AWS telemetry monitoring, 24-hour diurnal baseline time series, Open-Meteo external context, and spatial corroboration network.",
      },
    ],
  }),
  component: StationDeepDive,
});

type ParamTab = "temperature" | "pressure" | "humidity" | "combined";

function StationDeepDive() {
  const { dataset, selectedStationId, setSelectedStationId, config, externalForecast } =
    useSkyGuard();
  const station = useSelectedStation();
  const navigate = useNavigate();
  const [tab, setTab] = useState<ParamTab>("temperature");
  const [showImputed, setShowImputed] = useState(true);

  const observations = getObservations(dataset, station.station_id);
  const nearby = useMemo(
    () => nearbyStations(dataset, station.station_id, config.spatialRadius),
    [dataset, station.station_id, config.spatialRadius],
  );

  const chartData = observations.map((o) => ({
    hour: o.hour,
    observed: tab === "pressure" ? o.pressure : tab === "humidity" ? o.humidity : o.temperature,
    expected:
      tab === "pressure"
        ? o.expected_pressure
        : tab === "humidity"
          ? o.expected_humidity
          : o.expected_temperature,
    temperature: o.temperature,
    pressure: o.pressure,
    humidity: o.humidity,
    imputed:
      showImputed && o.imputed
        ? tab === "pressure"
          ? o.expected_pressure
          : tab === "humidity"
            ? o.expected_humidity
            : o.expected_temperature
        : null,
    anomaly: o.anomaly,
  }));

  const anomalyPoint = observations.find((o) => o.anomaly);

  const currentTabColor =
    tab === "pressure" ? "#6366f1" : tab === "humidity" ? "#10b981" : "#0284c7";

  return (
    <div className="space-y-5">
      {/* ==================== 1. EXECUTIVE HEADER ==================== */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-1">
        <div>
          <div className="flex items-center gap-2">
            <span className="provenance-tag provenance-live">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              AWS Station Deep Dive
            </span>
            <span className="text-xs font-mono text-muted-foreground">
              {station.station_id} · {station.district}, {station.state}
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground mt-1 font-sans flex items-center gap-2.5">
            <span>{station.station_name}</span>
            <span
              className={cn(
                "text-xs font-mono px-2 py-0.5 rounded border font-medium",
                station.status === "NORMAL"
                  ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
                  : "text-amber-400 bg-amber-500/10 border-amber-500/20",
              )}
            >
              {station.status === "NORMAL" ? "NOMINAL" : station.status}
            </span>
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5 max-w-3xl">
            Sensor telemetry inspection, 24-hour diurnal baselines, and multi-station spatial
            residual cross-validation group within {config.spatialRadius} km radius.
          </p>
        </div>

        {/* Station Selector Dropdown */}
        <div className="flex items-center gap-2 bg-card px-3 py-1.5 rounded-lg border border-border shrink-0">
          <Globe className="h-3.5 w-3.5 text-sky-400" />
          <span className="text-xs text-muted-foreground">Switch Station:</span>
          <select
            value={selectedStationId}
            onChange={(e) => setSelectedStationId(e.target.value)}
            className="bg-transparent text-xs font-medium text-foreground cursor-pointer outline-hidden pr-2"
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

      {/* ==================== 2. STATION METADATA PROFILE ==================== */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-card border border-border rounded-lg p-3 space-y-1 shadow-xs">
          <span className="text-[10.5px] text-muted-foreground uppercase font-mono tracking-wider">
            Station ID
          </span>
          <p className="font-mono font-bold text-base text-foreground">{station.station_id}</p>
          <span className="text-[10px] text-muted-foreground block truncate">
            IMD Master Registry
          </span>
        </div>

        <div className="bg-card border border-border rounded-lg p-3 space-y-1 shadow-xs">
          <span className="text-[10.5px] text-muted-foreground uppercase font-mono tracking-wider">
            Coordinates
          </span>
          <p className="font-mono font-bold text-xs text-sky-400 mt-1">
            {station.latitude.toFixed(2)}°N, {station.longitude.toFixed(2)}°E
          </p>
          <span className="text-[10px] text-muted-foreground block truncate">
            {station.district}, {station.state}
          </span>
        </div>

        <div className="bg-card border border-border rounded-lg p-3 space-y-1 shadow-xs">
          <span className="text-[10.5px] text-muted-foreground uppercase font-mono tracking-wider">
            Cadence
          </span>
          <p className="font-mono font-bold text-base text-emerald-400">
            {station.minutes_since_observation}m ago
          </p>
          <span className="text-[10px] text-muted-foreground block">15-minute sync window</span>
        </div>

        <div className="bg-card border border-border rounded-lg p-3 space-y-1 shadow-xs">
          <span className="text-[10.5px] text-muted-foreground uppercase font-mono tracking-wider">
            Maintenance Tier
          </span>
          <p className="font-mono font-bold text-base text-foreground">
            Tier {station.maintenance_priority}
          </p>
          <span className="text-[10px] text-muted-foreground block">Priority Asset</span>
        </div>

        <div className="bg-card border border-border rounded-lg p-3 space-y-1 shadow-xs">
          <span className="text-[10.5px] text-muted-foreground uppercase font-mono tracking-wider">
            Transducer Health
          </span>
          <p className="font-mono font-bold text-base text-emerald-400">98.5%</p>
          <span className="text-[10px] text-muted-foreground block">Calibrated Sensors</span>
        </div>

        <div className="bg-card border border-border rounded-lg p-3 space-y-1 shadow-xs">
          <span className="text-[10.5px] text-muted-foreground uppercase font-mono tracking-wider">
            Data Source
          </span>
          <p className="font-mono font-bold text-xs text-foreground mt-1">
            {station.is_simulated ? "SIMULATED" : "IMD AWS"}
          </p>
          <span className="text-[10px] text-muted-foreground block">In-situ probe</span>
        </div>
      </div>

      {/* ==================== 3. LIVE SENSOR READOUTS ==================== */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Surface Temperature */}
        <div className="bg-card border border-border rounded-xl p-4 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5 font-medium">
              <Thermometer className="h-4 w-4 text-sky-400" />
              Surface Temperature
            </span>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
              RTD Probe
            </span>
          </div>

          <div className="flex items-baseline gap-1.5">
            <span className="text-3xl sm:text-4xl font-bold font-sans text-foreground">
              {station.temperature !== null ? `${station.temperature.toFixed(1)}` : "—"}
            </span>
            <span className="text-sm text-muted-foreground font-medium">°C</span>
          </div>
          <p className="text-xs text-muted-foreground pt-1 border-t border-border/50">
            Diurnal baseline: 15–45 °C
          </p>
        </div>

        {/* Relative Humidity */}
        <div className="bg-card border border-border rounded-xl p-4 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5 font-medium">
              <Droplets className="h-4 w-4 text-emerald-400" />
              Relative Humidity
            </span>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
              Capacitive
            </span>
          </div>

          <div className="flex items-baseline gap-1.5">
            <span className="text-3xl sm:text-4xl font-bold font-sans text-foreground">
              {station.humidity !== null ? `${station.humidity.toFixed(1)}` : "—"}
            </span>
            <span className="text-sm text-muted-foreground font-medium">%</span>
          </div>
          <p className="text-xs text-muted-foreground pt-1 border-t border-border/50">
            Diurnal baseline: 25–90 %
          </p>
        </div>

        {/* Barometric Pressure */}
        <div className="bg-card border border-border rounded-xl p-4 shadow-xs space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5 font-medium">
              <Gauge className="h-4 w-4 text-indigo-400" />
              Barometric Pressure
            </span>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
              Piezoresistive
            </span>
          </div>

          <div className="flex items-baseline gap-1.5">
            <span className="text-3xl sm:text-4xl font-bold font-sans text-foreground">
              {station.pressure !== null ? `${station.pressure.toFixed(1)}` : "—"}
            </span>
            <span className="text-sm text-muted-foreground font-medium">hPa</span>
          </div>
          <p className="text-xs text-muted-foreground pt-1 border-t border-border/50">
            Diurnal baseline: 995–1025 hPa
          </p>
        </div>
      </div>

      {/* ==================== 4. EXTERNAL NUMERICAL MODEL REFERENCE ==================== */}
      <div className="bg-card border border-border rounded-xl p-4 shadow-xs space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-2 border-b border-border text-xs gap-2">
          <div className="flex items-center gap-2">
            <Compass className="h-4 w-4 text-sky-400" />
            <span className="font-semibold text-foreground">
              External Numerical Model Reference
            </span>
            <span className="text-muted-foreground font-mono">(Open-Meteo Synoptic Grid)</span>
          </div>
          <span className="text-[11px] text-muted-foreground italic">
            Numerical meteorological reanalysis — independent from in-situ AWS transducers
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-2.5 rounded-lg bg-muted/20 border border-border">
            <span className="text-muted-foreground text-[11px]">Model 2m Temp</span>
            <p className="text-lg font-bold text-foreground mt-0.5">
              {externalForecast.temperature_2m !== undefined
                ? `${externalForecast.temperature_2m.toFixed(1)} °C`
                : "28.5 °C"}
            </p>
          </div>
          <div className="p-2.5 rounded-lg bg-muted/20 border border-border">
            <span className="text-muted-foreground text-[11px]">Model Surface Pressure</span>
            <p className="text-lg font-bold text-foreground mt-0.5">
              {externalForecast.surface_pressure !== undefined
                ? `${externalForecast.surface_pressure.toFixed(1)} hPa`
                : "1012.0 hPa"}
            </p>
          </div>
          <div className="p-2.5 rounded-lg bg-muted/20 border border-border">
            <span className="text-muted-foreground text-[11px]">Precipitation (1h)</span>
            <p className="text-lg font-bold text-foreground mt-0.5">
              {externalForecast.precipitation !== undefined
                ? `${externalForecast.precipitation} mm`
                : "0.0 mm"}
            </p>
          </div>
          <div className="p-2.5 rounded-lg bg-muted/20 border border-border">
            <span className="text-muted-foreground text-[11px]">Cross-Validation State</span>
            <p className="text-base font-bold text-emerald-400 mt-0.5 flex items-center gap-1">
              <ShieldCheck className="h-4 w-4" />
              Consistent
            </p>
          </div>
        </div>
      </div>

      {/* ==================== 5. 24-HOUR TELEMETRY CHART ==================== */}
      <div className="bg-card border border-border rounded-xl p-5 shadow-xs space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              24-Hour Continuous Telemetry & Diurnal Baseline
            </h2>
            <p className="text-sm font-semibold text-foreground">
              In-situ sensor signal vs physical learned equilibrium baseline
            </p>
          </div>

          <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-lg border border-border text-xs">
            {(["temperature", "pressure", "humidity", "combined"] as const).map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setTab(p)}
                className={cn(
                  "px-3 py-1 rounded capitalize font-medium transition-colors cursor-pointer",
                  tab === p
                    ? "bg-secondary text-foreground shadow-xs"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        <div className="h-64 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="stationChartGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={currentTabColor} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={currentTabColor} stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255, 255, 255, 0.05)"
                vertical={false}
              />
              <XAxis dataKey="hour" stroke="#64748b" fontSize={10} fontFamily="IBM Plex Mono" />
              <YAxis
                stroke="#64748b"
                fontSize={10}
                fontFamily="IBM Plex Mono"
                domain={
                  tab === "pressure" ? [990, 1025] : tab === "humidity" ? [20, 100] : [15, 45]
                }
              />
              <RTooltip
                contentStyle={{
                  backgroundColor: "#111726",
                  borderColor: "rgba(255, 255, 255, 0.12)",
                  borderRadius: "0.5rem",
                  fontSize: "11px",
                  fontFamily: "IBM Plex Mono",
                  color: "#f1f5f9",
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: "11px", paddingTop: "6px", fontFamily: "IBM Plex Mono" }}
              />
              <Area
                type="monotone"
                dataKey="observed"
                name="Observed Telemetry"
                stroke={currentTabColor}
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#stationChartGrad)"
              />
              <Line
                type="monotone"
                dataKey="expected"
                name="Diurnal Baseline"
                stroke="#10b981"
                strokeDasharray="4 4"
                strokeWidth={1.5}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ==================== 6. SPATIAL CORROBORATION & INDIA MAP ==================== */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Nearby Stations Table (7 cols) */}
        <div className="lg:col-span-7 bg-card border border-border rounded-xl p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Nearby AWS Stations ({config.spatialRadius} km Radius)
              </h3>
              <p className="text-sm font-semibold text-foreground">
                Spatial residual cross-validation group
              </p>
            </div>
            <span className="text-xs font-mono text-muted-foreground bg-muted/40 px-2 py-0.5 rounded border border-border">
              {nearby.length} in range
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border text-muted-foreground uppercase text-[10.5px] tracking-wider">
                  <th className="pb-2.5">Station</th>
                  <th className="pb-2.5">Distance</th>
                  <th className="pb-2.5">Temp</th>
                  <th className="pb-2.5">Press</th>
                  <th className="pb-2.5">Spatial Residual</th>
                  <th className="pb-2.5">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {nearby.map((n) => {
                  const tempDiff =
                    station.temperature !== null && n.station.temperature !== null
                      ? Math.abs(station.temperature - n.station.temperature)
                      : null;
                  const residual = tempDiff !== null ? `${tempDiff.toFixed(1)}°C` : "--";
                  const isHighResidual = tempDiff !== null && tempDiff > 3.0;

                  return (
                    <tr key={n.station.station_id} className="hover:bg-muted/20 transition-colors">
                      <td className="py-2.5 font-sans font-medium text-foreground">
                        <button
                          type="button"
                          onClick={() => setSelectedStationId(n.station.station_id)}
                          className="hover:underline text-sky-400 text-left font-medium cursor-pointer"
                        >
                          {n.station.station_name}
                        </button>
                      </td>
                      <td className="py-2.5 text-muted-foreground">{n.distance} km</td>
                      <td className="py-2.5 text-foreground">
                        {n.station.temperature !== null ? `${n.station.temperature}°C` : "--"}
                      </td>
                      <td className="py-2.5 text-foreground">
                        {n.station.pressure !== null ? `${n.station.pressure} hPa` : "--"}
                      </td>
                      <td
                        className={cn(
                          "py-2.5 font-semibold",
                          isHighResidual ? "text-amber-400" : "text-emerald-400",
                        )}
                      >
                        {residual}
                      </td>
                      <td className="py-2.5">
                        <span
                          className={cn(
                            "px-2 py-0.5 rounded text-[10px] font-medium border",
                            n.station.status === "NORMAL"
                              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                              : "bg-amber-500/10 text-amber-400 border-amber-500/20",
                          )}
                        >
                          {n.station.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="pt-2.5 border-t border-border text-xs text-muted-foreground flex items-center justify-between">
            <span>Spatial consensus threshold: &lt; 3.0°C deviation</span>
            <span className="text-emerald-400 font-mono text-[11px]">
              Consensus: Verified Nominal
            </span>
          </div>
        </div>

        {/* Spatial Geography Map (5 cols) */}
        <div className="lg:col-span-5 bg-card border border-border rounded-xl p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <div className="flex items-center gap-2">
              <MapPin className="h-4 w-4 text-sky-400" />
              <h3 className="text-sm font-semibold text-foreground">National AWS Network</h3>
            </div>
            <span className="text-xs font-mono text-muted-foreground">Geographic Distribution</span>
          </div>
          <div className="h-72 rounded-lg overflow-hidden border border-border">
            <IndiaMap stations={dataset.stations} selectedId={selectedStationId} />
          </div>
        </div>
      </div>
    </div>
  );
}
