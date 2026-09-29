import React from "react";
import { useSkyGuard } from "@/data/store";
import {
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Snowflake,
  Zap,
  FileQuestion,
  TrendingUp,
  Wind,
  Layers,
  ArrowRight,
  ShieldAlert,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface QuickAnomaly {
  key: string;
  label: string;
  shortLabel: string;
  rowIndex: number;
  station: string;
  tag: string;
  icon: React.ComponentType<{ className?: string }>;
  badgeClass: string;
}

const QUICK_ANOMALIES: QuickAnomaly[] = [
  {
    key: "SPIKE",
    label: "Temperature Spike (+11.5°C)",
    shortLabel: "Temp Spike",
    rowIndex: 82, // Row 83 in dataset
    station: "HYD_AWS_01",
    tag: "+11.5°C / 1hr",
    icon: Flame,
    badgeClass: "border-rose-500/30 text-rose-300 bg-rose-500/10 hover:bg-rose-500/20",
  },
  {
    key: "FROZEN",
    label: "Frozen Sensor (Zero Variance)",
    shortLabel: "Frozen Sensor",
    rowIndex: 272, // Row 273 in dataset
    station: "BLR_AWS_02",
    tag: "Variance = 0.000",
    icon: Snowflake,
    badgeClass: "border-cyan-500/30 text-cyan-300 bg-cyan-500/10 hover:bg-cyan-500/20",
  },
  {
    key: "CORRUPTED",
    label: "Corrupted Reading (999.0°C)",
    shortLabel: "Corrupted (999°C)",
    rowIndex: 135, // Row 136 in dataset
    station: "HYD_AWS_01",
    tag: "Bounds Breach",
    icon: Zap,
    badgeClass: "border-amber-500/30 text-amber-300 bg-amber-500/10 hover:bg-amber-500/20",
  },
  {
    key: "MISSING",
    label: "Missing Data (Null Pressure)",
    shortLabel: "Missing Pressure",
    rowIndex: 134, // Row 135 in dataset
    station: "HYD_AWS_01",
    tag: "Channel Dropout",
    icon: FileQuestion,
    badgeClass: "border-purple-500/30 text-purple-300 bg-purple-500/10 hover:bg-purple-500/20",
  },
  {
    key: "DRIFT",
    label: "Sensor Drift (+3.2°C Baseline)",
    shortLabel: "Sensor Drift",
    rowIndex: 372, // Row 373 in dataset
    station: "DEL_AWS_03",
    tag: "+3.2°C Drift",
    icon: TrendingUp,
    badgeClass: "border-orange-500/30 text-orange-300 bg-orange-500/10 hover:bg-orange-500/20",
  },
  {
    key: "MULTI",
    label: "Multivariate Inconsistency",
    shortLabel: "Multivariate Decoupled",
    rowIndex: 320, // Row 321 in dataset
    station: "BLR_AWS_02",
    tag: "Lapse Rate Decoupled",
    icon: Layers,
    badgeClass: "border-indigo-500/30 text-indigo-300 bg-indigo-500/10 hover:bg-indigo-500/20",
  },
  {
    key: "WEATHER",
    label: "Synoptic Squall Front (Genuine Event)",
    shortLabel: "Genuine Weather Event",
    rowIndex: 451, // Row 452 in dataset
    station: "DEL_AWS_03",
    tag: "Spatial Corroboration",
    icon: Wind,
    badgeClass: "border-emerald-500/30 text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20",
  },
];

export function DemoControlBar() {
  const {
    dataSource,
    demoPlaying,
    demoCompleted,
    demoSpeed,
    demoIndex,
    demoTotal,
    demoRunId,
    demoScenario,
    demoScenariosList,
    pauseDemo,
    resumeDemo,
    resetDemo,
    returnToLive,
    setDemoSpeed,
    setDemoScenario,
    stepDemoIndex,
    currentPipelineResult,
  } = useSkyGuard();

  if (dataSource !== "DEMO") {
    return null;
  }

  const currentRow = Math.min(demoIndex + 1, demoTotal);
  const percentComplete = Math.min(100, Math.round((currentRow / demoTotal) * 100));
  const currentTimestamp = currentPipelineResult?.timestamp || "2026-09-01T00:00:00Z";
  const currentStation = currentPipelineResult?.station_id || "HYD_AWS_01";
  const isAnomaly = Boolean(currentPipelineResult?.is_anomaly);

  const handleQuickJump = async (rowIndex: number) => {
    pauseDemo();
    await stepDemoIndex(rowIndex);
  };

  return (
    <div className="w-full mb-5 space-y-3">
      {/* Main Simulation Control Dock */}
      <div className="bg-card border border-border rounded-xl p-4 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          {/* Left: Mode Title & Observation Position */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-amber-500/10 border border-amber-500/25 text-amber-300 text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              <span>DEMO MODE ACTIVE</span>
            </div>

            {demoRunId && (
              <span className="text-xs font-mono text-muted-foreground">
                Session: <strong className="text-foreground">{demoRunId}</strong>
              </span>
            )}

            <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
              <span>
                Row <strong className="text-foreground">{currentRow}</strong> / {demoTotal}
              </span>
              <span>•</span>
              <span className="text-sky-400 font-semibold">{currentStation}</span>
              <span className="hidden sm:inline">•</span>
              <span className="hidden sm:inline">{currentTimestamp}</span>
            </div>
          </div>

          {/* Right: Controls (Scenario, Speed, Play/Pause, Reset, Return Live) */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {/* Scenario Selector */}
            <div className="flex items-center gap-1.5 bg-muted/40 px-2.5 py-1 rounded-md border border-border">
              <span className="text-muted-foreground text-[11px]">Scenario:</span>
              <select
                aria-label="Select Demo Scenario"
                value={demoScenario}
                onChange={(e) => setDemoScenario(e.target.value)}
                className="bg-transparent text-xs font-medium text-foreground outline-hidden cursor-pointer"
              >
                {demoScenariosList.map((sc) => (
                  <option
                    key={sc.scenario}
                    value={sc.scenario}
                    className="bg-popover text-foreground"
                  >
                    {sc.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Speed Selector */}
            <div className="flex items-center bg-muted/40 p-0.5 rounded-md border border-border">
              <span className="px-1.5 text-[10px] text-muted-foreground font-mono">Speed</span>
              {[1, 5, 10].map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setDemoSpeed(s)}
                  className={cn(
                    "px-2 py-0.5 rounded text-xs font-mono transition-colors cursor-pointer",
                    demoSpeed === s
                      ? "bg-secondary text-foreground font-bold shadow-xs"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {s}x
                </button>
              ))}
            </div>

            {/* Play/Pause */}
            {demoPlaying ? (
              <button
                type="button"
                onClick={pauseDemo}
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs font-medium hover:bg-amber-500/30 transition-colors cursor-pointer"
              >
                <Pause className="h-3 w-3" />
                <span>Pause</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={resumeDemo}
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-primary text-primary-foreground text-xs font-medium hover:opacity-90 transition-opacity cursor-pointer"
              >
                <Play className="h-3 w-3 fill-current" />
                <span>{currentRow === 1 ? "Start" : "Resume"}</span>
              </button>
            )}

            {/* Reset */}
            <button
              type="button"
              onClick={resetDemo}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-muted text-muted-foreground hover:text-foreground border border-border text-xs transition-colors cursor-pointer"
              title="Reset stream to Row 1"
            >
              <RotateCcw className="h-3 w-3" />
              <span>Reset</span>
            </button>

            {/* Return to Live */}
            <button
              type="button"
              onClick={returnToLive}
              className="inline-flex items-center gap-1 px-3 py-1 rounded-md bg-rose-500/15 text-rose-300 hover:bg-rose-500/25 border border-rose-500/30 text-xs font-medium transition-colors cursor-pointer"
              title="Return to real-time live telemetry feed"
            >
              <span>Return Live</span>
            </button>
          </div>
        </div>

        {/* Linear Progress Bar */}
        <div className="mt-3 pt-2.5 border-t border-border/60">
          <div className="w-full bg-muted/50 rounded-full h-1.5 overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-200",
                isAnomaly ? "bg-rose-500" : "bg-sky-500",
              )}
              style={{ width: `${percentComplete}%` }}
            />
          </div>
          <div className="flex justify-between items-center text-[11px] font-mono text-muted-foreground mt-1.5">
            <span>
              Progress: {percentComplete}% ({currentRow} / {demoTotal} records)
            </span>
            <span>
              {isAnomaly ? (
                <span className="text-rose-400 font-semibold">
                  Detected: {currentPipelineResult?.anomaly_type.replace(/_/g, " ")} (
                  {currentPipelineResult?.severity})
                </span>
              ) : (
                <span className="text-emerald-400 font-medium">
                  Pipeline: Evaluating In-Situ Signal
                </span>
              )}
            </span>
          </div>
        </div>

        {/* 1-Click Anomaly Fault Injection Buttons */}
        <div className="mt-3 pt-2.5 border-t border-border/60">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-2">
            <span className="text-[11px] font-mono font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="h-3 w-3 text-amber-400" />
              <span>Inject Observation Fault (1-Click Evaluation):</span>
            </span>
            <span className="text-[10px] text-muted-foreground font-mono">
              Injected observations pass through the real backend detection pipeline
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {QUICK_ANOMALIES.map((qa) => {
              const isSelected = currentRow === qa.rowIndex + 1;
              const IconComp = qa.icon;
              return (
                <button
                  key={qa.key}
                  type="button"
                  onClick={() => handleQuickJump(qa.rowIndex)}
                  className={cn(
                    "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-medium border transition-colors cursor-pointer",
                    isSelected
                      ? "bg-secondary text-foreground border-border ring-1 ring-border shadow-xs"
                      : qa.badgeClass,
                  )}
                  title={`Jump to Row ${qa.rowIndex + 1} (${qa.station}) - ${qa.tag}`}
                >
                  <IconComp className="h-3 w-3 shrink-0" />
                  <span>{qa.shortLabel}</span>
                  <span className="text-[10px] text-muted-foreground opacity-80">
                    r{qa.rowIndex + 1}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* 5-Step Pipeline Progression Indicator */}
        <div className="mt-3 pt-2.5 border-t border-border/60">
          <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              1. Observation
            </span>
            <ArrowRight className="h-2.5 w-2.5 text-muted-foreground/50" />
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              2. Physical Gating
            </span>
            <ArrowRight className="h-2.5 w-2.5 text-muted-foreground/50" />
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              3. Temporal / LOF
            </span>
            <ArrowRight className="h-2.5 w-2.5 text-muted-foreground/50" />
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              4. Spatial Consensus
            </span>
            <ArrowRight className="h-2.5 w-2.5 text-muted-foreground/50" />
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              5. Explanation
            </span>
          </div>
        </div>
      </div>

      {/* Completion Banner */}
      {demoCompleted && (
        <div className="bg-card border border-border rounded-xl p-4 flex flex-col sm:flex-row items-center justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0" />
            <div>
              <div className="font-semibold text-sm text-foreground">
                Synthetic Stream Evaluation Complete
              </div>
              <div className="text-xs text-muted-foreground">
                All 504 chronological observations evaluated through the pipeline.
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <button
              type="button"
              onClick={resetDemo}
              className="px-3 py-1.5 rounded-md bg-muted hover:bg-secondary text-foreground border border-border transition-colors cursor-pointer"
            >
              Restart Replay
            </button>
            <button
              type="button"
              onClick={returnToLive}
              className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground font-medium hover:opacity-90 transition-opacity cursor-pointer"
            >
              Return Live
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
