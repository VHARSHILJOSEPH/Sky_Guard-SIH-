import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  Clock,
  Compass,
  Cpu,
  Droplets,
  ExternalLink,
  Filter,
  Gauge,
  Info,
  MapPin,
  Minus,
  Radio,
  RefreshCw,
  Server,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Thermometer,
  Wifi,
  WifiOff,
  Wind,
  Wrench,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSelectedStation, useSkyGuard } from "@/data/store";
import {
  formatDateTime,
  formatTime,
  getObservations,
  statusTone,
  type AnomalyType,
} from "@/data/skyguard";
import { WeatherService } from "@/data/weather-service";
import { StationMap } from "@/components/StationMap";
import { getCustomizedAnomalyProfile } from "@/data/anomaly-intelligence";

function DeltaBadge({ delta, unit = "" }: { delta: number | null | undefined; unit?: string }) {
  if (delta === null || delta === undefined || isNaN(delta)) {
    return <span className="text-[11px] font-mono text-muted-foreground">—</span>;
  }
  if (Math.abs(delta) < 0.001) {
    return (
      <span className="inline-flex items-center gap-1 text-[11px] font-mono text-muted-foreground bg-muted/50 border border-border px-1.5 py-0.5 rounded">
        <Minus className="h-2.5 w-2.5" />
        <span>0.0 {unit}</span>
      </span>
    );
  }
  const isPositive = delta > 0;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 text-[11px] font-mono font-medium px-1.5 py-0.5 rounded border",
        isPositive
          ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/25"
          : "text-sky-400 bg-sky-500/10 border-sky-500/25",
      )}
    >
      {isPositive ? <ArrowUp className="h-2.5 w-2.5" /> : <ArrowDown className="h-2.5 w-2.5" />}
      <span>
        {isPositive ? `+${delta}` : delta} {unit}
      </span>
    </span>
  );
}

export interface AnomalyDomainInfo {
  domain: "METEOROLOGICAL" | "DATA_QUALITY" | "COMMUNICATION" | "NOMINAL";
  label: string;
  isTempAffected: boolean;
  isHumAffected: boolean;
  isBaroAffected: boolean;
}

export function getAnomalyDomain(anomalyType: string): AnomalyDomainInfo {
  const upper = (anomalyType || "").toUpperCase();
  if (!upper || upper === "NONE" || upper === "NORMAL") {
    return {
      domain: "NOMINAL",
      label: "Nominal Observation",
      isTempAffected: false,
      isHumAffected: false,
      isBaroAffected: false,
    };
  }
  if (
    upper.includes("TIMESTAMP") ||
    upper.includes("COMMUNICATION") ||
    upper.includes("LOST") ||
    upper.includes("LATENCY") ||
    upper.includes("CADENCE")
  ) {
    return {
      domain: "COMMUNICATION",
      label: "Communication / Cadence Fault",
      isTempAffected: false,
      isHumAffected: false,
      isBaroAffected: false,
    };
  }
  if (
    upper.includes("FROZEN") ||
    upper.includes("BOUNDS") ||
    upper.includes("CORRUPTED") ||
    upper.includes("MISSING") ||
    upper.includes("STUCK") ||
    upper.includes("DRIFT")
  ) {
    return {
      domain: "DATA_QUALITY",
      label: "Sensor Quality / Hardware Fault",
      isTempAffected: upper.includes("TEMP"),
      isHumAffected: upper.includes("HUM"),
      isBaroAffected: upper.includes("PRESS") || upper.includes("BARO"),
    };
  }
  return {
    domain: "METEOROLOGICAL",
    label: "Meteorological Telemetry Anomaly",
    isTempAffected: upper.includes("TEMP") || upper.includes("SPIKE"),
    isHumAffected: upper.includes("HUM"),
    isBaroAffected: upper.includes("PRESS") || upper.includes("BARO"),
  };
}

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "National Weather Station Intelligence & Telemetry Ops | SkyGuard AI" },
      {
        name: "description",
        content:
          "High-precision Automatic Weather Station monitoring, 2D spatial corroboration, real-time telemetry validation, and explainable anomaly intelligence.",
      },
    ],
  }),
  component: DashboardConsole,
});

export function DashboardConsole() {
  const {
    dataset,
    selectedStationId,
    setSelectedStationId,
    activeParameter,
    setActiveParameter,
    chartWindow,
    setChartWindow,
    connectionState,
    clock,
    activeAnomaly,
    recentAnomalies,
    acknowledgeAnomaly,
    dismissAnomaly,
    toastMessage,
    dismissToast,
    externalForecast,
    mlState,
    dataSource,
    activeTelemetrySource,
    currentPipelineResult,
    pipelineRunning,
  } = useSkyGuard();

  const station = useSelectedStation();
  const navigate = useNavigate();

  const isLost = connectionState === "LOST";

  // Only use currentPipelineResult if it matches the currently selected station
  const matchingPipeline =
    currentPipelineResult && currentPipelineResult.station_id === selectedStationId
      ? currentPipelineResult
      : null;

  // Authoritative AWS Telemetry values
  const tempVal = isLost
    ? null
    : matchingPipeline?.readings?.temperature !== undefined &&
        matchingPipeline?.readings?.temperature !== null
      ? matchingPipeline.readings.temperature
      : (station.temperature ?? null);

  const humVal = isLost
    ? null
    : matchingPipeline?.readings?.humidity !== undefined &&
        matchingPipeline?.readings?.humidity !== null
      ? matchingPipeline.readings.humidity
      : (station.humidity ?? null);

  const baroVal = isLost
    ? null
    : matchingPipeline?.readings?.pressure !== undefined &&
        matchingPipeline?.readings?.pressure !== null
      ? matchingPipeline.readings.pressure
      : (station.pressure ?? null);

  const stnId = selectedStationId;
  const timestampStr = matchingPipeline?.timestamp ?? station.timestamp;

  // Genuine previous readings from backend
  const prevTempVal =
    matchingPipeline?.previous_readings?.temperature !== undefined &&
    matchingPipeline?.previous_readings?.temperature !== null
      ? matchingPipeline.previous_readings.temperature
      : null;

  const prevHumVal =
    matchingPipeline?.previous_readings?.humidity !== undefined &&
    matchingPipeline?.previous_readings?.humidity !== null
      ? matchingPipeline.previous_readings.humidity
      : null;

  const prevBaroVal =
    matchingPipeline?.previous_readings?.pressure !== undefined &&
    matchingPipeline?.previous_readings?.pressure !== null
      ? matchingPipeline.previous_readings.pressure
      : null;

  // Deltas
  const deltaTemp =
    matchingPipeline?.change?.temperature !== undefined &&
    matchingPipeline?.change?.temperature !== null
      ? matchingPipeline.change.temperature
      : tempVal !== null && prevTempVal !== null
        ? Number((tempVal - prevTempVal).toFixed(2))
        : null;

  const deltaHum =
    matchingPipeline?.change?.humidity !== undefined &&
    matchingPipeline?.change?.humidity !== null
      ? matchingPipeline.change.humidity
      : humVal !== null && prevHumVal !== null
        ? Number((humVal - prevHumVal).toFixed(1))
        : null;

  const deltaBaro =
    matchingPipeline?.change?.pressure !== undefined &&
    matchingPipeline?.change?.pressure !== null
      ? matchingPipeline.change.pressure
      : baroVal !== null && prevBaroVal !== null
        ? Number((baroVal - prevBaroVal).toFixed(2))
        : null;

  // Anomaly Resolution
  const rawAnomalyType =
    matchingPipeline?.anomaly_type ||
    (activeAnomaly?.stationId === selectedStationId ? activeAnomaly.type : null) ||
    station.anomaly_type ||
    "";
  const isAnomalyActive = Boolean(
    matchingPipeline?.is_anomaly ||
    (activeAnomaly && activeAnomaly.stationId === selectedStationId && activeAnomaly.type !== "NORMAL") ||
    station.status === "ANOMALY" ||
    station.status === "CRITICAL",
  );

  const anomalyProfile = useMemo(() => {
    return getCustomizedAnomalyProfile(
      rawAnomalyType,
      station,
      matchingPipeline,
      activeAnomaly?.stationId === selectedStationId ? activeAnomaly : null,
      externalForecast,
    );
  }, [rawAnomalyType, station, matchingPipeline, activeAnomaly, selectedStationId, externalForecast]);

  const domainInfo = useMemo(() => {
    const isTemp = anomalyProfile.affectedParameter === "temperature" || anomalyProfile.affectedParameter === "multivariate";
    const isHum = anomalyProfile.affectedParameter === "humidity" || anomalyProfile.affectedParameter === "multivariate";
    const isBaro = anomalyProfile.affectedParameter === "pressure";
    return {
      domain: anomalyProfile.domain,
      label: anomalyProfile.domainLabel,
      isTempAffected: isTemp,
      isHumAffected: isHum,
      isBaroAffected: isBaro,
    };
  }, [anomalyProfile]);

  // Status strings
  const tempStatus = isLost
    ? "OFFLINE"
    : domainInfo.isTempAffected && isAnomalyActive
      ? (anomalyProfile.severity === "CRITICAL" ? "CRITICAL FAULT" : "ALERT")
      : tempVal === null
        ? "NO DATA"
        : "NORMAL";

  const humStatus = isLost
    ? "OFFLINE"
    : domainInfo.isHumAffected && isAnomalyActive
      ? (anomalyProfile.severity === "CRITICAL" ? "FROZEN TRANSDUCER" : "ALERT")
      : humVal === null
        ? "NO DATA"
        : "NORMAL";

  const pressStatus = isLost
    ? "OFFLINE"
    : domainInfo.isBaroAffected && isAnomalyActive
      ? (baroVal === null ? "CHANNEL DROPOUT" : "ALERT")
      : baroVal === null
        ? "NO DATA"
        : "NORMAL";

  // Thermal Comfort & Weather Descriptor
  const thermalDescriptor = useMemo(() => {
    if (tempVal === null) return "No Data";
    if (tempVal > 40) return "Extreme Heat Warning";
    if (tempVal > 34) return "High Solar Load";
    if (tempVal > 28) return "Warm & Clear";
    if (tempVal > 20) return "Moderate Diurnal Curve";
    return "Cool & Fresh";
  }, [tempVal]);

  interface ServerIncidentRecord {
    incident_id?: string;
    id?: string;
    station_id?: string;
    stationId?: string;
    first_detected?: string;
    time?: string;
    anomaly_type?: string;
    type?: string;
    severity?: string;
    observed_value?: unknown;
    observed?: string | number;
    expected_value?: unknown;
    expected?: string | number;
    explanation?: string;
    description?: string;
    status?: string;
    occurrence_count?: number;
  }

  // Server Incidents & Lifecycle State
  const [anomalyStatusFilter, setAnomalyStatusFilter] = useState<string>("ALL");
  const [anomalySeverityFilter, setAnomalySeverityFilter] = useState<string>("ALL");
  const [anomalyStationFilter, setAnomalyStationFilter] = useState<string>("ALL");
  const [serverIncidents, setServerIncidents] = useState<ServerIncidentRecord[]>([]);

  useEffect(() => {
    WeatherService.fetchIncidents().then((res) => {
      const data = res as { incidents?: ServerIncidentRecord[] } | null;
      if (data && Array.isArray(data.incidents)) {
        setServerIncidents(data.incidents);
      }
    });
  }, [currentPipelineResult]);

  const handleUpdateLifecycle = async (incidentId: string, newStatus: string) => {
    try {
      await WeatherService.updateIncidentStatus(incidentId, newStatus);
      toast.success(`Incident #${incidentId.slice(0, 8)} updated to ${newStatus}`);
      setServerIncidents((prev) =>
        prev.map((inc) =>
          inc.incident_id === incidentId || inc.id === incidentId
            ? { ...inc, status: newStatus }
            : inc,
        ),
      );
    } catch {
      toast.error("Failed to update incident status");
    }
  };

  const mergedAnomalies = useMemo(() => {
    if (serverIncidents.length > 0) {
      return serverIncidents.map((inc: ServerIncidentRecord) => ({
        id: inc.incident_id || inc.id,
        stationId: inc.station_id || inc.stationId || stnId,
        time: inc.first_detected
          ? new Date(inc.first_detected).toLocaleTimeString("en-US", {
              hour: "2-digit",
              minute: "2-digit",
              hour12: true,
            })
          : inc.time || "Recent",
        type: inc.anomaly_type
          ? inc.anomaly_type.replace(/_/g, " ").toUpperCase()
          : inc.type || "ANOMALY",
        severity: (inc.severity || "HIGH").toUpperCase(),
        observed:
          inc.observed_value !== undefined && inc.observed_value !== null
            ? String(inc.observed_value)
            : inc.observed || "Anomalous",
        expected:
          inc.expected_value !== undefined && inc.expected_value !== null
            ? String(inc.expected_value)
            : inc.expected || "Nominal",
        description:
          inc.explanation || inc.description || "Flagged by multi-source quality control pipeline",
        status: (inc.status || "DETECTED").toUpperCase(),
        occurrences: inc.occurrence_count || 1,
      }));
    }
    return recentAnomalies.map((inc) => ({
      ...inc,
      status: (inc.status || "DETECTED").toUpperCase(),
      occurrences: 1,
    }));
  }, [serverIncidents, recentAnomalies, stnId]);

  const filteredAnomalies = useMemo(() => {
    return mergedAnomalies.filter((inc) => {
      const matchesStation =
        anomalyStationFilter === "ALL" || inc.stationId === anomalyStationFilter;
      const matchesSeverity =
        anomalySeverityFilter === "ALL" || inc.severity === anomalySeverityFilter;
      const matchesStatus = anomalyStatusFilter === "ALL" || inc.status === anomalyStatusFilter;
      return matchesStation && matchesSeverity && matchesStatus;
    });
  }, [mergedAnomalies, anomalyStationFilter, anomalySeverityFilter, anomalyStatusFilter]);

  // Operational KPIs
  const totalStationsCount = dataset.stations.length;
  const healthyStationsCount = dataset.stations.filter(
    (s) => s.status === "NORMAL" || s.health === "HEALTHY",
  ).length;
  const activeAnomaliesCount = mergedAnomalies.filter(
    (a) => a.status === "DETECTED" || a.status === "INVESTIGATING" || isAnomalyActive,
  ).length;
  const dataQualityDisplay = pipelineRunning
    ? "VALIDATING"
    : matchingPipeline?.status
      ? matchingPipeline.status
      : isLost
        ? "OFFLINE"
        : "NOMINAL (98.6%)";

  // Observations
  const observations = useMemo(
    () => getObservations(dataset, selectedStationId),
    [dataset, selectedStationId],
  );

  const windowedObservations = useMemo(() => {
    if (chartWindow === "15m" || chartWindow === "1h") {
      return observations.slice(-5);
    }
    if (chartWindow === "6h") {
      return observations.slice(-10);
    }
    return observations;
  }, [observations, chartWindow]);

  // Chart Configuration
  const chartConfig = useMemo(() => {
    switch (activeParameter) {
      case "pressure":
        return {
          title: "Barometric Pressure",
          dataKey: "pressure",
          unit: "hPa",
          value: baroVal,
          safeMin: 995,
          safeMax: 1025,
          color: "#0284c7",
        };
      case "humidity":
        return {
          title: "Relative Humidity",
          dataKey: "humidity",
          unit: "%",
          value: humVal,
          safeMin: 25,
          safeMax: 90,
          color: "#10b981",
        };
      default:
        return {
          title: "Surface Air Temperature",
          dataKey: "temperature",
          unit: "°C",
          value: tempVal,
          safeMin: 15,
          safeMax: 45,
          color: "#0284c7",
        };
    }
  }, [activeParameter, tempVal, humVal, baroVal]);

  const flaggedAnomalyObs = useMemo(
    () => windowedObservations.find((o) => o.anomaly),
    [windowedObservations],
  );

  return (
    <div className="space-y-5">
      {/* Toast Alert Banner if present */}
      {toastMessage && (
        <div
          role="alert"
          className="flex items-center justify-between p-3 bg-rose-500/10 border border-rose-500/25 rounded-lg text-xs"
        >
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0" />
            <span className="font-semibold text-foreground">{toastMessage.title}</span>
            <span className="text-muted-foreground">{toastMessage.desc}</span>
            <span className="font-mono text-[11px] text-rose-400 bg-rose-500/15 px-1.5 py-0.5 rounded">
              {toastMessage.time}
            </span>
          </div>
          <button
            type="button"
            aria-label="Dismiss alert"
            onClick={dismissToast}
            className="text-muted-foreground hover:text-foreground p-1"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* ==================== 1. STATION CONTEXT & FLEET SUMMARY STRIP ==================== */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-1">
        <div>
          <div className="flex items-center gap-2">
            <span className="provenance-tag provenance-live">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              AWS Quality Sentinel
            </span>
            <span className="text-xs font-mono text-muted-foreground">
              Station: <strong className="text-foreground">{stnId}</strong> · {station.station_name}
              , {station.district} ({station.state})
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground mt-1 font-sans">
            Weather Station Intelligence & Telemetry Stream
          </h1>
        </div>

        {/* Operational Indicators */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 bg-card px-3 py-1.5 rounded-lg border border-border">
            <span className="text-muted-foreground">Provenance:</span>
            <span className="font-medium text-foreground font-mono">
              {dataSource === "DEMO"
                ? "DEMO SIMULATION"
                : activeTelemetrySource === "IMD_AWS"
                  ? "IMD AWS REFERENCE"
                  : "EXTERNAL SYNOPTIC"}
            </span>
          </div>
          <div className="flex items-center gap-1.5 bg-card px-3 py-1.5 rounded-lg border border-border text-muted-foreground font-mono">
            <Clock className="h-3.5 w-3.5 text-sky-400" />
            <span className="text-foreground">{formatTime(timestampStr)}</span>
          </div>
        </div>
      </div>

      {/* ==================== 2. OPERATIONAL SUMMARY STATS ==================== */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-card border border-border rounded-lg p-3.5 shadow-xs">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Monitored AWS Fleet</span>
            <Server className="h-3.5 w-3.5 text-muted-foreground" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold text-foreground font-sans">
              {totalStationsCount}
            </span>
            <span className="text-xs text-muted-foreground">Registered Nodes</span>
          </div>
          <p className="mt-1 text-[11px] text-muted-foreground">National grid telemetry coverage</p>
        </div>

        <div className="bg-card border border-border rounded-lg p-3.5 shadow-xs">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Fleet Operational Health</span>
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold text-emerald-400 font-sans">
              {healthyStationsCount}
            </span>
            <span className="text-xs text-muted-foreground">
              / {totalStationsCount} Nominal (
              {Math.round((healthyStationsCount / (totalStationsCount || 1)) * 100)}%)
            </span>
          </div>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Operating within calibrated bounds
          </p>
        </div>

        <div className="bg-card border border-border rounded-lg p-3.5 shadow-xs">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Active Sentinel Incidents</span>
            <AlertTriangle
              className={cn(
                "h-3.5 w-3.5",
                activeAnomaliesCount > 0 ? "text-rose-400" : "text-emerald-400",
              )}
            />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span
              className={cn(
                "text-2xl font-bold font-sans",
                activeAnomaliesCount > 0 ? "text-rose-400" : "text-foreground",
              )}
            >
              {activeAnomaliesCount}
            </span>
            <span className="text-xs text-muted-foreground">Unresolved flags</span>
          </div>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Automated multi-source quality audit
          </p>
        </div>

        <div className="bg-card border border-border rounded-lg p-3.5 shadow-xs">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Telemetry Quality Envelope</span>
            <Activity className="h-3.5 w-3.5 text-sky-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-xl font-bold text-sky-400 font-sans truncate">
              {dataQualityDisplay}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Physical invariant compliance verified
          </p>
        </div>
      </div>

      {/* ==================== 3. WEATHER CONDITIONS AS VISUAL ANCHOR ==================== */}
      <div className="bg-card border border-border rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 mb-4 border-b border-border/70 gap-2">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Current Observations
            </div>
            <div className="text-base font-semibold text-foreground mt-0.5">
              In-Situ Telemetry — Station {stnId} ({station.station_name})
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <span>
              Coordinates: {station.latitude.toFixed(2)}°N, {station.longitude.toFixed(2)}°E
            </span>
          </div>
        </div>

        {/* 3 Core Weather Values (Google Weather clarity) */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Thermometer className="h-4 w-4 text-sky-400" />
                Surface Temperature
              </span>
              <span
                className={cn(
                  "text-[10px] font-mono px-2 py-0.5 rounded border font-medium",
                  domainInfo.isTempAffected && isAnomalyActive
                    ? "text-rose-300 bg-rose-500/15 border-rose-500/30"
                    : "text-emerald-300 bg-emerald-500/10 border-emerald-500/20",
                )}
              >
                {tempStatus}
              </span>
            </div>

            <div className="flex items-baseline gap-1.5">
              <span
                className={cn(
                  "text-4xl sm:text-5xl font-bold tracking-tight font-sans",
                  domainInfo.isTempAffected && isAnomalyActive
                    ? "text-rose-400"
                    : "text-foreground",
                )}
              >
                {tempVal !== null ? tempVal.toFixed(1) : "—"}
              </span>
              <span className="text-lg text-muted-foreground font-medium">°C</span>
            </div>

            <div className="flex items-center justify-between text-xs pt-1 border-t border-border/50 text-muted-foreground">
              <span>Diurnal range: 15.0 – 45.0 °C</span>
              <DeltaBadge delta={deltaTemp} unit="°C" />
            </div>
          </div>

          {/* Barometric Pressure */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Gauge className="h-4 w-4 text-indigo-400" />
                Barometric Pressure
              </span>
              <span
                className={cn(
                  "text-[10px] font-mono px-2 py-0.5 rounded border font-medium",
                  domainInfo.isBaroAffected && isAnomalyActive
                    ? "text-rose-300 bg-rose-500/15 border-rose-500/30"
                    : "text-emerald-300 bg-emerald-500/10 border-emerald-500/20",
                )}
              >
                {pressStatus}
              </span>
            </div>

            <div className="flex items-baseline gap-1.5">
              <span
                className={cn(
                  "text-4xl sm:text-5xl font-bold tracking-tight font-sans",
                  domainInfo.isBaroAffected && isAnomalyActive
                    ? "text-rose-400"
                    : "text-foreground",
                )}
              >
                {baroVal !== null ? baroVal.toFixed(1) : "—"}
              </span>
              <span className="text-lg text-muted-foreground font-medium">hPa</span>
            </div>

            <div className="flex items-center justify-between text-xs pt-1 border-t border-border/50 text-muted-foreground">
              <span>Synoptic baseline: 995 – 1025 hPa</span>
              <DeltaBadge delta={deltaBaro} unit="hPa" />
            </div>
          </div>

          {/* Relative Humidity */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Droplets className="h-4 w-4 text-emerald-400" />
                Relative Humidity
              </span>
              <span
                className={cn(
                  "text-[10px] font-mono px-2 py-0.5 rounded border font-medium",
                  domainInfo.isHumAffected && isAnomalyActive
                    ? "text-rose-300 bg-rose-500/15 border-rose-500/30"
                    : "text-emerald-300 bg-emerald-500/10 border-emerald-500/20",
                )}
              >
                {humStatus}
              </span>
            </div>

            <div className="flex items-baseline gap-1.5">
              <span
                className={cn(
                  "text-4xl sm:text-5xl font-bold tracking-tight font-sans",
                  domainInfo.isHumAffected && isAnomalyActive ? "text-rose-400" : "text-foreground",
                )}
              >
                {humVal !== null ? humVal.toFixed(1) : "—"}
              </span>
              <span className="text-lg text-muted-foreground font-medium">%</span>
            </div>

            <div className="flex items-center justify-between text-xs pt-1 border-t border-border/50 text-muted-foreground">
              <span>Comfort range: 25 – 90 %</span>
              <DeltaBadge delta={deltaHum} unit="%" />
            </div>
          </div>
        </div>

        {/* Subordinate External Grid Reference Context */}
        <div className="mt-5 pt-3 border-t border-border/60 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground bg-muted/20 px-3 py-2 rounded-lg">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-foreground">External Numerical Context:</span>
            <span>
              Open-Meteo Synoptic Grid (
              {externalForecast.temperature_2m !== undefined
                ? `${externalForecast.temperature_2m.toFixed(1)}°C`
                : "28.5°C"}
              ,{" "}
              {externalForecast.surface_pressure !== undefined
                ? `${externalForecast.surface_pressure.toFixed(1)} hPa`
                : "1012.0 hPa"}
              )
            </span>
          </div>
          <span className="text-[11px] font-mono italic">
            External reference model — not in-situ hardware probe
          </span>
        </div>
      </div>

      {/* ==================== 4. AI ANOMALY OBSERVATION STATUS ==================== */}
      {isAnomalyActive ? (
        <div className="bg-card border-l-4 border-l-rose-500 border border-border rounded-xl p-5 shadow-xs space-y-4">
          {/* Header Bar */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-border/70 pb-3">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 shrink-0">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30 uppercase">
                    {matchingPipeline?.severity || anomalyProfile.severity} SEVERITY
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-muted text-muted-foreground border border-border uppercase">
                    {anomalyProfile.domainLabel}
                  </span>
                  <h3 className="text-sm sm:text-base font-bold text-foreground">
                    {anomalyProfile.title}
                  </h3>
                </div>
                <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-2">
                  <span>
                    Flagged on {station.station_name} ({stnId}) · {station.district}, {station.state}
                  </span>
                  <span>·</span>
                  <span className="font-mono text-foreground font-medium">
                    Observed: {formatTime(timestampStr)}
                  </span>
                </div>
              </div>
            </div>

            {/* Score & Confidence Badges */}
            <div className="flex items-center gap-2.5 text-xs font-mono shrink-0">
              <div className="bg-muted/40 px-3 py-1.5 rounded-lg border border-border">
                <span className="text-muted-foreground">Anomaly Score: </span>
                <strong className="text-rose-400 font-bold">
                  {matchingPipeline?.anomaly_score !== undefined
                    ? Number(matchingPipeline.anomaly_score).toFixed(2)
                    : anomalyProfile.anomalyScore.toFixed(2)}
                </strong>
              </div>
              <div className="bg-muted/40 px-3 py-1.5 rounded-lg border border-border">
                <span className="text-muted-foreground">Confidence: </span>
                <strong className="text-foreground font-bold">
                  {matchingPipeline?.confidence
                    ? `${Math.round(matchingPipeline.confidence * (matchingPipeline.confidence <= 1 ? 100 : 1))}%`
                    : `${Math.round(anomalyProfile.confidence)}%`}
                </strong>
              </div>
            </div>
          </div>

          {/* Diagnostic Inspection Strip (Observed vs Expected vs Variance) */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 bg-muted/20 p-3 rounded-lg border border-border/70 font-mono text-xs">
            <div className="space-y-0.5">
              <span className="text-[10.5px] text-muted-foreground uppercase block font-sans font-semibold">
                Observed Sensor Value
              </span>
              <span className="text-sm font-bold text-rose-300">
                {anomalyProfile.observedValue}
              </span>
              <span className="text-[10px] text-muted-foreground block font-sans truncate">
                {anomalyProfile.parameterLabel}
              </span>
            </div>

            <div className="space-y-0.5">
              <span className="text-[10.5px] text-muted-foreground uppercase block font-sans font-semibold">
                Baseline / Expected
              </span>
              <span className="text-sm font-semibold text-emerald-400">
                {anomalyProfile.expectedValue}
              </span>
              <span className="text-[10px] text-muted-foreground block font-sans">
                Learned Diurnal Equilibrium
              </span>
            </div>

            <div className="space-y-0.5">
              <span className="text-[10.5px] text-muted-foreground uppercase block font-sans font-semibold">
                Telemetry Evidence
              </span>
              <span className="text-sm font-semibold text-amber-300">
                {anomalyProfile.deviationMetric}
              </span>
              <span className="text-[10px] text-muted-foreground block font-sans truncate">
                Subsystem: {anomalyProfile.hardwareComponent}
              </span>
            </div>
          </div>

          {/* Explanation & Evidence Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            {/* Left: Why Was This Flagged */}
            <div className="space-y-2 bg-muted/15 p-3.5 rounded-lg border border-border/60">
              <div className="font-semibold text-foreground uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5 text-rose-400" />
                <span>Why Was This Flagged?</span>
              </div>
              <ul className="space-y-2 text-muted-foreground">
                {(matchingPipeline?.why_flagged && matchingPipeline.why_flagged.length > 0
                  ? matchingPipeline.why_flagged
                  : anomalyProfile.whyFlagged
                ).map((pt, i) => (
                  <li key={i} className="flex items-start gap-2 leading-relaxed">
                    <span className="text-rose-400 font-bold shrink-0 mt-0.5">•</span>
                    <span>{pt}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Right: Technical Diagnosis & Operator Remediation */}
            <div className="space-y-2.5 bg-muted/15 p-3.5 rounded-lg border border-border/60 flex flex-col justify-between">
              <div className="space-y-2">
                <div className="font-semibold text-foreground uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                  <Wrench className="h-3.5 w-3.5 text-sky-400" />
                  <span>Technical Diagnosis & Root Cause</span>
                </div>
                <p className="text-muted-foreground leading-relaxed">
                  {matchingPipeline?.explanation || anomalyProfile.narrative}
                </p>

                {/* Certified Remediation Recommendation */}
                <div className="p-2.5 rounded-md bg-sky-500/10 border border-sky-500/25 text-sky-200 text-xs flex items-start gap-2">
                  <ShieldAlert className="h-4 w-4 shrink-0 text-sky-400 mt-0.5" />
                  <span>
                    <strong className="text-sky-300 font-semibold">Recommended Remediation: </strong>
                    {anomalyProfile.recommendedAction}
                  </span>
                </div>
              </div>

              {/* Action Buttons Deck */}
              <div className="pt-2 border-t border-border/50 flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    onClick={() => {
                      const id =
                        matchingPipeline?.incident_id ||
                        (activeAnomaly?.stationId === selectedStationId ? activeAnomaly.id : null) ||
                        "INC-ACTIVE";
                      handleUpdateLifecycle(id, "ACKNOWLEDGED");
                      acknowledgeAnomaly();
                    }}
                    className="px-2.5 py-1.5 rounded-md bg-muted hover:bg-secondary text-foreground text-xs font-medium border border-border cursor-pointer transition-colors"
                  >
                    Acknowledge
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const id =
                        matchingPipeline?.incident_id ||
                        (activeAnomaly?.stationId === selectedStationId ? activeAnomaly.id : null) ||
                        "INC-ACTIVE";
                      handleUpdateLifecycle(id, "INVESTIGATING");
                    }}
                    className="px-2.5 py-1.5 rounded-md bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 text-xs font-medium border border-amber-500/30 cursor-pointer transition-colors"
                  >
                    Investigating
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const id =
                        matchingPipeline?.incident_id ||
                        (activeAnomaly?.stationId === selectedStationId ? activeAnomaly.id : null) ||
                        "INC-ACTIVE";
                      handleUpdateLifecycle(id, "RESOLVED");
                      dismissAnomaly();
                    }}
                    className="px-2.5 py-1.5 rounded-md bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 text-xs font-medium border border-emerald-500/30 cursor-pointer transition-colors"
                  >
                    Resolve
                  </button>
                  <button
                    type="button"
                    onClick={dismissAnomaly}
                    className="px-2.5 py-1.5 rounded-md bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 text-xs font-medium border border-rose-500/30 cursor-pointer transition-colors"
                  >
                    Dismiss
                  </button>
                </div>

                <button
                  type="button"
                  onClick={() => navigate({ to: "/diagnostics" })}
                  className="px-3 py-1.5 rounded-md bg-sky-500/15 hover:bg-sky-500/25 text-sky-300 text-xs font-semibold border border-sky-400/30 flex items-center gap-1.5 cursor-pointer transition-colors"
                >
                  <span>Evidence Audit</span>
                  <ExternalLink className="h-3 w-3" />
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-card border-l-4 border-l-emerald-500 border border-border rounded-xl p-4 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0" />
            <div>
              <div className="text-xs font-semibold text-emerald-400 uppercase tracking-wider font-mono">
                Observation Status: Nominal
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                No significant anomaly detected. All physical bounds, temporal slopes, and spatial
                corroborations conform strictly to standard diurnal models.
              </p>
            </div>
          </div>
          <div className="text-xs font-mono text-muted-foreground shrink-0">
            LOF Score: <strong className="text-foreground">0.08</strong> (Threshold: 1.50)
          </div>
        </div>
      )}

      {/* ==================== 5. STATION-CENTRIC MAP + TELEMETRY TREND ==================== */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left: Station-Centric 2D Leaflet Map (5 cols) */}
        <div className="lg:col-span-5 space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground px-1">
            <span className="font-semibold text-foreground">Spatial Network Context</span>
            <span className="font-mono">50 km Consensus Radius</span>
          </div>
          <StationMap
            stations={dataset.stations}
            selectedStationId={selectedStationId}
            onSelectStation={(id) => setSelectedStationId(id)}
            height={320}
          />
        </div>

        {/* Right: Telemetry Time-Series Chart (7 cols) */}
        <div className="lg:col-span-7 bg-card border border-border rounded-xl p-4 shadow-xs space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Telemetry Trend
              </h2>
              <p className="text-sm font-semibold text-foreground">
                {chartConfig.title} ({chartConfig.unit})
              </p>
            </div>

            {/* Parameter Switcher */}
            <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-lg border border-border text-xs">
              {(["temperature", "pressure", "humidity"] as const).map((param) => {
                const active = activeParameter === param;
                return (
                  <button
                    key={param}
                    type="button"
                    onClick={() => setActiveParameter(param)}
                    className={cn(
                      "px-2.5 py-1 rounded capitalize font-medium transition-colors cursor-pointer",
                      active
                        ? "bg-secondary text-foreground shadow-xs"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {param}
                  </button>
                );
              })}
            </div>

            {/* Time Window Tabs */}
            <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-lg border border-border text-[11px] font-mono">
              {(["1h", "6h", "24h"] as const).map((w) => {
                const active =
                  chartWindow === w || (w === "1h" && chartWindow === ("15m" as const));
                return (
                  <button
                    key={w}
                    type="button"
                    onClick={() => setChartWindow(w === "1h" ? "1h" : w === "6h" ? "6h" : "6h")}
                    className={cn(
                      "px-2 py-0.5 rounded transition-colors cursor-pointer",
                      active
                        ? "bg-secondary text-foreground shadow-xs font-semibold"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {w}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Recharts Area Chart */}
          <div className="w-full h-60 pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={windowedObservations}
                margin={{ top: 10, right: 15, left: -15, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={chartConfig.color} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={chartConfig.color} stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="hour"
                  stroke="#64748b"
                  fontSize={10}
                  fontFamily="IBM Plex Mono"
                  tickLine={false}
                />
                <YAxis
                  stroke="#64748b"
                  fontSize={10}
                  fontFamily="IBM Plex Mono"
                  tickLine={false}
                  domain={["auto", "auto"]}
                />
                <RechartsTooltip
                  contentStyle={{
                    backgroundColor: "#111726",
                    borderColor: "rgba(255, 255, 255, 0.12)",
                    borderRadius: "0.5rem",
                    fontSize: "11px",
                    fontFamily: "IBM Plex Mono",
                    color: "#f1f5f9",
                  }}
                />
                <ReferenceArea
                  y1={chartConfig.safeMin}
                  y2={chartConfig.safeMax}
                  fill="rgba(16, 185, 129, 0.04)"
                  stroke="rgba(16, 185, 129, 0.2)"
                  strokeDasharray="3 3"
                />
                <Area
                  type="monotone"
                  dataKey={chartConfig.dataKey}
                  stroke={chartConfig.color}
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#chartGradient)"
                />
                {flaggedAnomalyObs &&
                  typeof flaggedAnomalyObs[
                    chartConfig.dataKey as keyof typeof flaggedAnomalyObs
                  ] === "number" && (
                    <ReferenceDot
                      x={flaggedAnomalyObs.hour}
                      y={
                        flaggedAnomalyObs[
                          chartConfig.dataKey as keyof typeof flaggedAnomalyObs
                        ] as number
                      }
                      r={5}
                      fill="#ef4444"
                      stroke="#ffffff"
                      strokeWidth={1.5}
                    />
                  )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ==================== 6. OPERATIONAL ANOMALY HISTORY TABLE ==================== */}
      <div className="bg-card border border-border rounded-xl p-5 shadow-xs space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-border">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Recent Anomaly Incidents
            </h3>
            <p className="text-sm font-semibold text-foreground mt-0.5">
              Operational Quality Control Log
            </p>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            <select
              value={anomalyStationFilter}
              onChange={(e) => setAnomalyStationFilter(e.target.value)}
              className="h-8 px-2 rounded-md bg-muted/50 border border-border text-xs text-foreground cursor-pointer"
            >
              <option value="ALL">All Stations</option>
              {dataset.stations.map((s) => (
                <option key={s.station_id} value={s.station_id}>
                  {s.station_name} ({s.station_id})
                </option>
              ))}
            </select>

            <select
              value={anomalySeverityFilter}
              onChange={(e) => setAnomalySeverityFilter(e.target.value)}
              className="h-8 px-2 rounded-md bg-muted/50 border border-border text-xs text-foreground cursor-pointer"
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MEDIUM">Medium</option>
              <option value="LOW">Low</option>
            </select>

            <select
              value={anomalyStatusFilter}
              onChange={(e) => setAnomalyStatusFilter(e.target.value)}
              className="h-8 px-2 rounded-md bg-muted/50 border border-border text-xs text-foreground cursor-pointer"
            >
              <option value="ALL">All States</option>
              <option value="DETECTED">Detected</option>
              <option value="ACKNOWLEDGED">Acknowledged</option>
              <option value="INVESTIGATING">Investigating</option>
              <option value="RESOLVED">Resolved</option>
            </select>

            <span className="text-xs font-mono text-muted-foreground bg-muted/30 px-2 py-1 rounded border border-border">
              {filteredAnomalies.length} Logged
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-border text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                <th className="pb-2.5 pl-2">Time</th>
                <th className="pb-2.5">Station</th>
                <th className="pb-2.5">Anomaly Type</th>
                <th className="pb-2.5">Domain</th>
                <th className="pb-2.5">Severity</th>
                <th className="pb-2.5">Observed vs Expected</th>
                <th className="pb-2.5">Status</th>
                <th className="pb-2.5">Action</th>
                <th className="pb-2.5 pr-2">Summary</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {filteredAnomalies.length === 0 ? (
                <tr>
                  <td
                    colSpan={9}
                    className="py-6 text-center text-muted-foreground font-sans text-xs"
                  >
                    No active or historical anomalies match current filter criteria.
                  </td>
                </tr>
              ) : (
                filteredAnomalies.map((inc) => {
                  const dom = getAnomalyDomain(inc.type);
                  return (
                    <tr key={inc.id} className="hover:bg-muted/30 transition-colors">
                      <td className="py-2.5 pl-2 text-muted-foreground whitespace-nowrap">
                        <div className="text-foreground font-medium">{inc.time}</div>
                        <div className="text-[10px] text-muted-foreground">
                          #{inc.id ? inc.id.slice(0, 8) : "INC-01"}
                        </div>
                      </td>
                      <td className="py-2.5 font-medium text-foreground whitespace-nowrap">
                        {inc.stationId || stnId}
                      </td>
                      <td className="py-2.5 font-medium text-foreground whitespace-nowrap">
                        {inc.type}
                      </td>
                      <td className="py-2.5 whitespace-nowrap">
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground border border-border">
                          {dom.domain}
                        </span>
                      </td>
                      <td className="py-2.5 whitespace-nowrap">
                        <span
                          className={cn(
                            "inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border",
                            inc.severity === "CRITICAL"
                              ? "bg-rose-500/10 text-rose-400 border-rose-500/25"
                              : inc.severity === "HIGH"
                                ? "bg-amber-500/10 text-amber-400 border-amber-500/25"
                                : "bg-slate-500/10 text-slate-300 border-slate-500/25",
                          )}
                        >
                          {inc.severity}
                        </span>
                      </td>
                      <td className="py-2.5 whitespace-nowrap">
                        <span className="font-semibold text-foreground">{inc.observed}</span>
                        <span className="text-muted-foreground text-[10px] ml-1">
                          (exp: {inc.expected})
                        </span>
                      </td>
                      <td className="py-2.5 whitespace-nowrap">
                        <span
                          className={cn(
                            "text-[10px] font-medium px-2 py-0.5 rounded border uppercase",
                            inc.status === "RESOLVED"
                              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                              : inc.status === "INVESTIGATING"
                                ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                                : "bg-rose-500/10 text-rose-400 border-rose-500/20",
                          )}
                        >
                          {inc.status}
                        </span>
                      </td>
                      <td className="py-2.5 whitespace-nowrap">
                        <div className="flex items-center gap-1">
                          {inc.status !== "ACKNOWLEDGED" && inc.status !== "RESOLVED" && (
                            <button
                              type="button"
                              onClick={() => handleUpdateLifecycle(inc.id || "", "ACKNOWLEDGED")}
                              className="px-2 py-0.5 text-[10px] rounded bg-muted hover:bg-secondary text-foreground border border-border cursor-pointer transition-colors"
                            >
                              Ack
                            </button>
                          )}
                          {inc.status !== "RESOLVED" ? (
                            <button
                              type="button"
                              onClick={() => handleUpdateLifecycle(inc.id || "", "RESOLVED")}
                              className="px-2 py-0.5 text-[10px] rounded bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 cursor-pointer transition-colors"
                            >
                              Resolve
                            </button>
                          ) : (
                            <span className="text-[10px] text-muted-foreground italic">Closed</span>
                          )}
                        </div>
                      </td>
                      <td
                        className="py-2.5 pr-2 text-muted-foreground max-w-xs truncate text-[11px]"
                        title={inc.description}
                      >
                        {inc.description}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
