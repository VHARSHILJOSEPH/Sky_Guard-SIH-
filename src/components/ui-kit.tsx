import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { SensorHealth, Severity, StationStatus } from "@/data/skyguard";
import { severityTone, statusTone } from "@/data/skyguard";

export type Tone = "normal" | "warning" | "critical" | "offline" | "info";

export const toneText: Record<Tone, string> = {
  normal: "text-emerald-400",
  warning: "text-amber-400",
  critical: "text-rose-400",
  offline: "text-slate-400",
  info: "text-sky-400",
};

export const toneDot: Record<Tone, string> = {
  normal: "bg-emerald-400",
  warning: "bg-amber-400",
  critical: "bg-rose-400",
  offline: "bg-slate-400",
  info: "bg-sky-400",
};

export const toneChip: Record<Tone, string> = {
  normal: "bg-emerald-500/10 text-emerald-300 border-emerald-500/25",
  warning: "bg-amber-500/10 text-amber-300 border-amber-500/25",
  critical: "bg-rose-500/10 text-rose-300 border-rose-500/25",
  offline: "bg-slate-500/10 text-slate-300 border-slate-500/25",
  info: "bg-sky-500/10 text-sky-300 border-sky-500/25",
};

export const TONE_HEX: Record<Tone, string> = {
  normal: "#10b981",
  warning: "#f59e0b",
  critical: "#ef4444",
  offline: "#64748b",
  info: "#0284c7",
};

export function Chip({
  tone = "info",
  children,
  className,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium tracking-normal",
        toneChip[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function StatusChip({ status }: { status: StationStatus | SensorHealth }) {
  const tone = statusTone(status) as Tone;
  return (
    <Chip tone={tone}>
      <span className={cn("h-1.5 w-1.5 rounded-full", toneDot[tone])} />
      {status}
    </Chip>
  );
}

export function SeverityChip({ severity }: { severity: Severity }) {
  return <Chip tone={severityTone(severity) as Tone}>{severity}</Chip>;
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={cn("bg-card border border-border rounded-xl overflow-hidden shadow-xs", className)}
    >
      {title && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border bg-muted/40 px-4 py-3">
          <div>
            <h2 className="text-sm font-semibold text-foreground tracking-tight">{title}</h2>
            {subtitle && <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cn("flex-1", bodyClassName ?? "p-4")}>{children}</div>
    </section>
  );
}

export function KpiCard({
  label,
  value,
  unit,
  hint,
  tone = "info",
  icon,
}: {
  label: string;
  value: string | number;
  unit?: string;
  hint?: string;
  tone?: Tone;
  icon?: ReactNode;
}) {
  return (
    <div className="bg-card border border-border rounded-xl p-4 shadow-xs relative overflow-hidden">
      <span className={cn("absolute inset-y-0 left-0 w-[3px]", toneDot[tone])} />
      <div className="flex items-start justify-between gap-2 pl-2">
        <div>
          <p className="text-xs font-medium text-muted-foreground">{label}</p>
          <p className="mt-1 text-2xl font-bold tracking-tight text-foreground font-sans">
            {value}
            {unit && <span className="ml-1 text-xs font-normal text-muted-foreground">{unit}</span>}
          </p>
          {hint && <p className="mt-1.5 text-xs text-muted-foreground">{hint}</p>}
        </div>
        {icon && <div className={cn("opacity-70", toneText[tone])}>{icon}</div>}
      </div>
    </div>
  );
}

export function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <p className="text-[11px] font-medium text-muted-foreground">{label}</p>
      <p className="truncate text-sm font-semibold text-foreground mt-0.5">{value}</p>
    </div>
  );
}

export function DemoTag({ children = "DEMO" }: { children?: ReactNode }) {
  return (
    <span className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase text-amber-300">
      {children}
    </span>
  );
}
