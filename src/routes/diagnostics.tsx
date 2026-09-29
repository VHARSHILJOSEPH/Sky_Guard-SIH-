import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMemo } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Brain,
  CheckCircle2,
  Cpu,
  FileText,
  HelpCircle,
  Layers,
  Scale,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  WifiOff,
  Activity,
  Wrench,
  Check,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSelectedStation, useSkyGuard } from "@/data/store";
import { nearbyStations } from "@/data/skyguard";
import { getAnomalyDomain } from "./index";

interface EvidenceStageItem {
  stage: number;
  name: string;
  verdict: string;
  anomaly_delta?: number;
  weight?: number;
  status: string;
  detail?: string;
}

export const Route = createFileRoute("/diagnostics")({
  head: () => ({
    meta: [
      { title: "Explainable Anomaly Intelligence & Evidence Fusion | SkyGuard AI" },
      {
        name: "description",
        content:
          "Authoritative multi-source evidence fusion audit, backend decision logic, LOF anomaly scoring, and operator explanation for AWS sensor quality.",
      },
    ],
  }),
  component: DiagnosticsConsole,
});

export function DiagnosticsConsole() {
  const {
    dataset,
    selectedStationId,
    setSelectedStationId,
    config,
    mlState,
    currentPipelineResult,
    activeAnomaly,
    acknowledgeAnomaly,
    dismissAnomaly,
  } = useSkyGuard();
  const station = useSelectedStation();
  const navigate = useNavigate();

  const nearby = useMemo(
    () => nearbyStations(dataset, station.station_id, config.spatialRadius),
    [dataset, station.station_id, config.spatialRadius],
  );

  const rawAnomalyType =
    currentPipelineResult?.anomaly_type || activeAnomaly?.type || station.anomaly_type || "";
  const isAnomaly = Boolean(
    currentPipelineResult?.is_anomaly ||
    (activeAnomaly && activeAnomaly.type !== "NORMAL") ||
    station.status === "ANOMALY" ||
    station.status === "CRITICAL",
  );
  const domainInfo = getAnomalyDomain(rawAnomalyType);

  const structuredExp = currentPipelineResult?.structured_explanation;
  const waterfall = currentPipelineResult?.evidence_waterfall || [];

  const evidenceScores = useMemo(() => {
    const severity =
      currentPipelineResult?.severity || activeAnomaly?.severity || (isAnomaly ? "HIGH" : "NORMAL");

    const explanation =
      structuredExp?.what_happened ||
      currentPipelineResult?.explanation ||
      activeAnomaly?.description ||
      "All physical bounds and spatial correlations conform to nominal diurnal baselines.";

    const whyFlagged: string[] =
      Array.isArray(structuredExp?.why_flagged) && structuredExp.why_flagged.length > 0
        ? (structuredExp.why_flagged as string[])
        : typeof structuredExp?.why_flagged === "string"
          ? [structuredExp.why_flagged]
          : currentPipelineResult?.why_flagged && currentPipelineResult.why_flagged.length > 0
            ? currentPipelineResult.why_flagged
            : isAnomaly
              ? [
                  activeAnomaly?.description ||
                    "Observed telemetry exceeded statistical boundaries.",
                ]
              : ["Signal envelope falls strictly within calibrated physical bounds."];

    const recommendedAction =
      structuredExp?.recommended_action ||
      currentPipelineResult?.classification ||
      (isAnomaly
        ? domainInfo.domain === "COMMUNICATION"
          ? "Inspect telemetry transmitter link, packet receiver buffer, and antenna line."
          : domainInfo.domain === "DATA_QUALITY"
            ? "Perform remote soft-reset of sensor transducer; verify calibration coefficients."
            : "Dispatch regional field team to inspect AWS probe and radiation shield."
        : "No maintenance action required. Station nominal.");

    return {
      fusedDecision: isAnomaly
        ? rawAnomalyType.replace(/_/g, " ").toUpperCase()
        : "NOMINAL OBSERVATION",
      severity,
      confidence:
        currentPipelineResult?.confidence_status || (isAnomaly ? "HIGH_CONFIDENCE" : "STABLE"),
      explanation,
      whyFlagged,
      recommendedAction,
      domain: domainInfo.domain,
      domainLabel: domainInfo.label,
    };
  }, [currentPipelineResult, structuredExp, activeAnomaly, isAnomaly, rawAnomalyType, domainInfo]);

  return (
    <div className="space-y-5">
      {/* ==================== 1. EXECUTIVE HEADER ==================== */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-1">
        <div>
          <div className="flex items-center gap-2">
            <span className="provenance-tag provenance-live">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Quality Control Engine
            </span>
            <span className="text-xs font-mono text-muted-foreground">
              {station.station_id} · {station.station_name}
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground mt-1 font-sans">
            Explainable Anomaly Intelligence & Evidence Fusion
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5 max-w-3xl">
            Multi-stage evidence verification: physical invariants, temporal gradients, spatial
            neighbor consensus, and multivariate Local Outlier Factor (LOF) density audit.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-card px-3 py-1.5 rounded-lg border border-border shrink-0">
          <span className="text-xs text-muted-foreground">Station:</span>
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

      {/* ==================== 2. PRIMARY CONSENSUS BANNER ==================== */}
      <div
        className={cn(
          "border rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xs",
          isAnomaly
            ? "bg-card border-l-4 border-l-rose-500 border-border"
            : "bg-card border-l-4 border-l-emerald-500 border-border",
        )}
      >
        <div className="flex items-center gap-3.5">
          <div
            className={cn(
              "w-10 h-10 rounded-lg flex items-center justify-center shrink-0 border",
              isAnomaly
                ? "bg-rose-500/10 text-rose-400 border-rose-500/25"
                : "bg-emerald-500/10 text-emerald-400 border-emerald-500/25",
            )}
          >
            {isAnomaly ? <ShieldAlert className="h-5 w-5" /> : <ShieldCheck className="h-5 w-5" />}
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-mono uppercase font-semibold text-muted-foreground">
                Decision:
              </span>
              <span
                className={cn(
                  "text-[10px] font-mono font-medium px-2 py-0.5 rounded border uppercase",
                  isAnomaly
                    ? "bg-rose-500/15 text-rose-300 border-rose-500/30"
                    : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
                )}
              >
                {evidenceScores.severity} SEVERITY
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-muted text-muted-foreground border border-border">
                {evidenceScores.domainLabel}
              </span>
            </div>
            <h2 className="text-lg font-bold tracking-tight text-foreground mt-0.5 font-sans">
              {evidenceScores.fusedDecision}
            </h2>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
          <div className="flex flex-col bg-muted/40 px-3 py-1.5 rounded-lg border border-border">
            <span className="text-muted-foreground uppercase text-[10px]">Confidence</span>
            <span className="font-semibold text-foreground mt-0.5">
              {evidenceScores.confidence}
            </span>
          </div>
          <div className="flex flex-col bg-muted/40 px-3 py-1.5 rounded-lg border border-border">
            <span className="text-muted-foreground uppercase text-[10px]">Domain</span>
            <span className="font-semibold text-sky-400 mt-0.5">{evidenceScores.domain}</span>
          </div>
        </div>
      </div>

      {/* ==================== 3. 7-CHANNEL EVIDENCE WATERFALL & OPERATOR TRIAGE ==================== */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left: 7-Stage Evidence Waterfall (7 cols) */}
        <div className="lg:col-span-7 bg-card border border-border rounded-xl p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                7-Stage Quality Control Pipeline
              </h3>
              <p className="text-sm font-semibold text-foreground">
                Sequential verification stages
              </p>
            </div>
            <span className="text-xs font-mono text-muted-foreground bg-muted/40 px-2 py-0.5 rounded border border-border">
              7 Stages Active
            </span>
          </div>

          <div className="space-y-2 text-xs">
            {(waterfall.length > 0
              ? waterfall
              : [
                  {
                    stage: 1,
                    name: "Data Quality & Physical Envelopes",
                    verdict: "BOUNDS_SATISFIED",
                    anomaly_delta: 0.0,
                    weight: 0.25,
                    status: "NORMAL" as const,
                    detail:
                      "Operating ranges and physical thermodynamics conform to certified limits.",
                  },
                  {
                    stage: 2,
                    name: "Temporal Rate & Cadence",
                    verdict: "TEMPORAL_CONFORMING",
                    anomaly_delta: 0.0,
                    weight: 0.2,
                    status: "NORMAL" as const,
                    detail:
                      "15-minute rate of change, gradient velocity, and persistence are within tolerances.",
                  },
                  {
                    stage: 3,
                    name: "Statistical Diurnal Baseline",
                    verdict: "STATISTICAL_NOMINAL",
                    anomaly_delta: 0.0,
                    weight: 0.15,
                    status: "NORMAL" as const,
                    detail:
                      "Diurnal distribution distance within rolling interquartile range (IQR).",
                  },
                  {
                    stage: 4,
                    name: "Local Outlier Factor (LOF Density)",
                    verdict: "NOMINAL_DENSITY",
                    anomaly_delta: 0.08,
                    weight: 0.15,
                    status: "NORMAL" as const,
                    detail:
                      "Local Outlier Factor density ratio 0.08 vs calibrated decision boundary 1.50.",
                  },
                  {
                    stage: 5,
                    name: "Multivariate Vapor Coupling",
                    verdict: "CONSISTENT_CORRELATION",
                    anomaly_delta: 0.0,
                    weight: 0.1,
                    status: "NORMAL" as const,
                    detail: "Thermodynamic Magnus-Tetens vapor pressure coupling is consistent.",
                  },
                  {
                    stage: 6,
                    name: "Spatial Neighbor Corroboration",
                    verdict: "SPATIAL_CORROBORATED",
                    anomaly_delta: 0.0,
                    weight: 0.1,
                    status: "NORMAL" as const,
                    detail: `Consensus across ${nearby.length} neighboring AWS stations verified.`,
                  },
                  {
                    stage: 7,
                    name: "Macro Synoptic Weather Context",
                    verdict: "MACRO_WEATHER_CONSISTENT",
                    anomaly_delta: 0.0,
                    weight: 0.05,
                    status: "INFO" as const,
                    detail:
                      "Regional numerical model confirms macro-scale meteorological baseline.",
                  },
                ]
            ).map((stage: EvidenceStageItem) => {
              const isFlagged = stage.status === "WARNING" || stage.status === "CRITICAL";

              return (
                <div
                  key={stage.stage}
                  className={cn(
                    "p-3 rounded-lg border transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-2.5",
                    isFlagged
                      ? "bg-rose-500/5 border-rose-500/25"
                      : "bg-muted/20 border-border/60 hover:bg-muted/40",
                  )}
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className={cn(
                        "w-5 h-5 rounded flex items-center justify-center font-mono font-semibold text-[11px] shrink-0",
                        isFlagged ? "bg-rose-500 text-white" : "bg-muted text-muted-foreground",
                      )}
                    >
                      {stage.stage}
                    </span>
                    <div>
                      <h4 className="font-semibold text-foreground text-xs">{stage.name}</h4>
                      <p className="text-[11px] text-muted-foreground mt-0.5">{stage.detail}</p>
                    </div>
                  </div>

                  <div className="shrink-0 self-end sm:self-center font-mono">
                    <span
                      className={cn(
                        "text-[10px] font-medium px-2 py-0.5 rounded border uppercase",
                        isFlagged
                          ? "bg-rose-500/10 text-rose-300 border-rose-500/25"
                          : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
                      )}
                    >
                      {stage.verdict}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Explainable Operator Diagnostics & Triage (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-card border border-border rounded-xl p-5 shadow-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-border">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-sky-400" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Operator Incident Narrative
                </h3>
              </div>
            </div>

            <div className="space-y-3 text-xs">
              <div className="p-3 rounded-lg bg-muted/20 border border-border/60 space-y-1">
                <span className="text-[10px] font-semibold text-muted-foreground uppercase font-mono">
                  Observation Assessment
                </span>
                <p className="text-foreground leading-relaxed text-xs">
                  {evidenceScores.explanation}
                </p>
              </div>

              <div className="p-3 rounded-lg bg-muted/20 border border-border/60 space-y-1">
                <span className="text-[10px] font-semibold text-muted-foreground uppercase font-mono">
                  Why Flagged / Verified
                </span>
                <ul className="space-y-1 mt-1 text-muted-foreground text-xs list-disc list-inside">
                  {evidenceScores.whyFlagged.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="p-3 rounded-lg bg-sky-500/5 border border-sky-500/20 space-y-1">
                <span className="text-[10px] font-semibold text-sky-400 uppercase font-mono flex items-center gap-1.5">
                  <Wrench className="h-3.5 w-3.5" />
                  Recommended Operator Remediation
                </span>
                <p className="text-foreground leading-relaxed text-xs">
                  {evidenceScores.recommendedAction}
                </p>
              </div>
            </div>
          </div>

          {/* Quick Action Triage Deck */}
          <div className="bg-card border border-border rounded-xl p-4 shadow-xs space-y-2.5">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Incident Remediation Actions
            </h4>
            <div className="grid grid-cols-2 gap-2 text-xs font-medium">
              <button
                type="button"
                onClick={acknowledgeAnomaly}
                className="p-2.5 rounded-lg bg-muted hover:bg-secondary text-foreground border border-border flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
              >
                <Check className="h-3.5 w-3.5 text-sky-400" />
                <span>Acknowledge</span>
              </button>
              <button
                type="button"
                onClick={dismissAnomaly}
                className="p-2.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
              >
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                <span>Resolve Incident</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
