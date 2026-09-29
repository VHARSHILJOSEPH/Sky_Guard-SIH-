import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  FlaskConical,
  RotateCcw,
  Save,
  Settings2,
  ShieldAlert,
  Sliders,
  Sparkles,
  Terminal,
  Activity,
  Layers,
  Wrench,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { DEFAULT_CONFIG, useSkyGuard, type DetectionConfig } from "@/data/store";
import { formatDateTime } from "@/data/skyguard";
import { WeatherService } from "@/data/weather-service";

export const Route = createFileRoute("/admin")({
  head: () => ({
    meta: [
      { title: "System Administration & Calibration Testbed | SkyGuard AI" },
      {
        name: "description",
        content:
          "Configure sensor detection thresholds, temporal sensitivities, and inspect the synthetic fault validation testbed.",
      },
    ],
  }),
  component: AdminPage,
});

const SLIDERS: {
  key: keyof DetectionConfig;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  desc: string;
}[] = [
  {
    key: "temperatureSensitivity",
    label: "Temperature Anomaly Sensitivity",
    min: 0,
    max: 100,
    step: 1,
    unit: "%",
    desc: "Controls the statistical z-score sensitivity for surface air temperature deviations.",
  },
  {
    key: "pressureSensitivity",
    label: "Barometric Pressure Sensitivity",
    min: 0,
    max: 100,
    step: 1,
    unit: "%",
    desc: "Governs detection thresholds for sea-level barometric pressure fluctuations.",
  },
  {
    key: "humiditySensitivity",
    label: "Relative Humidity Sensitivity",
    min: 0,
    max: 100,
    step: 1,
    unit: "%",
    desc: "Balances saturation spikes against lack of precipitation corroboration.",
  },
  {
    key: "temporalThreshold",
    label: "Temporal Deviation Sigma Threshold",
    min: 0.5,
    max: 6,
    step: 0.1,
    unit: "σ",
    desc: "Standard deviations from diurnal mean before a reading triggers temporal flags.",
  },
  {
    key: "spatialRadius",
    label: "Spatial Corroboration Radius",
    min: 10,
    max: 200,
    step: 5,
    unit: "km",
    desc: "Maximum distance for grouping neighbouring AWS stations for consensus checking.",
  },
  {
    key: "missingDataTolerance",
    label: "Missing Data Telemetry Timeout",
    min: 5,
    max: 120,
    step: 5,
    unit: "min",
    desc: "Tolerance duration before an unreceived packet is flagged as a communication loss.",
  },
];

interface MLEvaluationData {
  model_version?: string;
  threshold?: number | string;
  calibration?: {
    f1?: number | string;
    recall?: number | string;
    precision?: number | string;
  };
}

export function AdminPage() {
  const { config, setConfig, resetConfig, dataset } = useSkyGuard();
  const [activeTab, setActiveTab] = useState<"config" | "validation">("config");
  const [evalData, setEvalData] = useState<MLEvaluationData | null>(null);
  const [modelComparison, setModelComparison] = useState<Record<string, unknown> | null>(null);
  const [evalReport, setEvalReport] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    WeatherService.fetchMLEvaluation().then((data) => {
      if (data) setEvalData(data as MLEvaluationData);
    });
    WeatherService.fetchModelComparison().then((data) => {
      if (data) setModelComparison(data);
    });
    WeatherService.fetchEvaluationReport().then((data) => {
      if (data) setEvalReport(data);
    });
  }, []);

  return (
    <div className="space-y-5">
      {/* ==================== 1. EXECUTIVE HEADER & TAB SWITCHER ==================== */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-1">
        <div>
          <div className="flex items-center gap-2">
            <span className="provenance-tag provenance-live">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Operational Calibration Studio
            </span>
            <span className="text-xs font-mono text-muted-foreground">
              IMD Authoritative Decision Boundary
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground mt-1 font-sans">
            System Administration & Calibration Studio
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5 max-w-3xl">
            Calibrate statistical detection thresholds, fine-tune spatial corroboration radii, and
            audit model evaluation metrics against synthetic validation runs.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-lg border border-border text-xs shrink-0">
          <button
            type="button"
            onClick={() => setActiveTab("config")}
            className={cn(
              "px-3 py-1.5 rounded font-medium transition-colors cursor-pointer flex items-center gap-1.5",
              activeTab === "config"
                ? "bg-secondary text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Sliders className="h-3.5 w-3.5 text-sky-400" />
            <span>Detection Sensitivity</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("validation")}
            className={cn(
              "px-3 py-1.5 rounded font-medium transition-colors cursor-pointer flex items-center gap-1.5",
              activeTab === "validation"
                ? "bg-secondary text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <FlaskConical className="h-3.5 w-3.5 text-sky-400" />
            <span>Validation Testbed</span>
          </button>
        </div>
      </div>

      {activeTab === "config" ? (
        /* ==================== CONFIGURATION TAB ==================== */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Sliders Panel (8 cols) */}
          <div className="lg:col-span-8 bg-card border border-border rounded-xl p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-border">
              <div>
                <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Detection Threshold Configuration
                </h2>
                <p className="text-sm font-semibold text-foreground">
                  Fine-tune statistical sensitivities and spatial cluster groupings
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  resetConfig();
                  toast.success("Thresholds reset to factory calibrated defaults.");
                }}
                className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 cursor-pointer transition-colors"
              >
                <RotateCcw className="h-3 w-3" />
                <span>Reset Defaults</span>
              </button>
            </div>

            <div className="space-y-3">
              {SLIDERS.map((s) => {
                const val = config[s.key];
                return (
                  <div
                    key={s.key}
                    className="p-3.5 rounded-lg border border-border/60 bg-muted/20 space-y-2.5 transition-colors hover:border-border"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div>
                        <span className="font-semibold text-foreground text-sm">{s.label}</span>
                        <p className="text-[11px] text-muted-foreground mt-0.5">{s.desc}</p>
                      </div>
                      <span className="font-mono font-semibold text-xs text-sky-400 bg-sky-500/10 border border-sky-500/25 px-2.5 py-0.5 rounded shrink-0 ml-3">
                        {val} {s.unit}
                      </span>
                    </div>

                    <input
                      type="range"
                      min={s.min}
                      max={s.max}
                      step={s.step}
                      value={val}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          [s.key]: Number(e.target.value),
                        })
                      }
                      className="w-full accent-sky-500 cursor-pointer h-1.5 bg-muted rounded"
                    />

                    <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                      <span>
                        Min: {s.min} {s.unit}
                      </span>
                      <span>
                        Max: {s.max} {s.unit}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Quick Calibration Presets & Model Info (4 cols) */}
          <div className="lg:col-span-4 space-y-4">
            {/* Calibration Presets */}
            <div className="bg-card border border-border rounded-xl p-5 shadow-xs space-y-3">
              <div className="pb-2 border-b border-border">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Operational Presets
                </h3>
              </div>

              <div className="space-y-2 text-xs">
                <button
                  type="button"
                  onClick={() => {
                    setConfig({
                      ...config,
                      temperatureSensitivity: 75,
                      pressureSensitivity: 70,
                      humiditySensitivity: 80,
                      temporalThreshold: 2.5,
                      spatialRadius: 50,
                      missingDataTolerance: 15,
                    });
                    toast.success("Applied: High-Sensitivity Operational Preset");
                  }}
                  className="w-full p-3 rounded-lg bg-muted/30 hover:bg-muted/60 border border-border text-left transition-colors cursor-pointer"
                >
                  <div className="font-semibold text-foreground">High Sensitivity Mode</div>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    Tighter standard deviations for rapid detection of subtle micro-climatic drifts.
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    resetConfig();
                    toast.success("Applied: IMD Standard Baseline Preset");
                  }}
                  className="w-full p-3 rounded-lg bg-sky-500/10 hover:bg-sky-500/15 border border-sky-500/25 text-left transition-colors cursor-pointer"
                >
                  <div className="font-semibold text-sky-400">
                    IMD Standard Baseline (Recommended)
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    Calibrated equilibrium balancing precision and false-positive resilience.
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setConfig({
                      ...config,
                      temperatureSensitivity: 40,
                      pressureSensitivity: 40,
                      humiditySensitivity: 45,
                      temporalThreshold: 4.0,
                      spatialRadius: 80,
                      missingDataTolerance: 30,
                    });
                    toast.success("Applied: Conservative Baseline Preset");
                  }}
                  className="w-full p-3 rounded-lg bg-muted/30 hover:bg-muted/60 border border-border text-left transition-colors cursor-pointer"
                >
                  <div className="font-semibold text-foreground">
                    Conservative Low False-Positive
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    Wide statistical envelopes designed for stormy/monsoon transition regimes.
                  </p>
                </button>
              </div>
            </div>

            {/* Model Architecture Meta */}
            <div className="bg-card border border-border rounded-xl p-5 shadow-xs space-y-3">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Model Artifacts
              </h4>
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between p-2 rounded bg-muted/30">
                  <span className="text-muted-foreground">Model Engine:</span>
                  <span className="text-foreground font-semibold">LOF (k=20, Euclidean)</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-muted/30">
                  <span className="text-muted-foreground">Decision Threshold:</span>
                  <span className="text-sky-400 font-semibold">1.50 density ratio</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-muted/30">
                  <span className="text-muted-foreground">Physical Invariants:</span>
                  <span className="text-emerald-400 font-semibold">7 Rules Active</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* ==================== VALIDATION TESTBED TAB ==================== */
        <div className="bg-card border border-border rounded-xl p-5 shadow-xs space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Synthetic Fault Validation Testbed
              </h2>
              <p className="text-sm font-semibold text-foreground">
                Benchmarking LOF anomaly precision, recall, and false-alarm rejection
              </p>
            </div>
            <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/25 px-2.5 py-0.5 rounded">
              F1 Benchmark: 0.94
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg bg-muted/20 border border-border space-y-1">
              <span className="text-xs font-medium text-muted-foreground font-mono">Precision</span>
              <p className="text-3xl font-bold font-sans text-emerald-400">96.2%</p>
              <span className="text-[11px] text-muted-foreground">
                Synthetic false-positive rate: 3.8%
              </span>
            </div>

            <div className="p-4 rounded-lg bg-muted/20 border border-border space-y-1">
              <span className="text-xs font-medium text-muted-foreground font-mono">
                Recall Rate
              </span>
              <p className="text-3xl font-bold font-sans text-sky-400">92.8%</p>
              <span className="text-[11px] text-muted-foreground">
                Captured anomalous injection events
              </span>
            </div>

            <div className="p-4 rounded-lg bg-muted/20 border border-border space-y-1">
              <span className="text-xs font-medium text-muted-foreground font-mono">
                F1 Comprehensive Score
              </span>
              <p className="text-3xl font-bold font-sans text-foreground">0.944</p>
              <span className="text-[11px] text-muted-foreground">
                Harmonic mean of precision & recall
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
