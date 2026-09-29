import { lazy, Suspense, useEffect, useState } from "react";
import type { StationMapProps } from "./StationMapInner";

const Inner = lazy(() => import("./StationMapInner"));

export function StationMap(props: StationMapProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const fallback = (
    <div
      className="flex flex-col items-center justify-center rounded-xl border border-border bg-card text-xs text-muted-foreground"
      style={{ height: props.height ?? 360 }}
    >
      <div className="w-5 h-5 border-2 border-sky-400 border-t-transparent rounded-full animate-spin mb-2" />
      <span>Loading meteorological spatial network…</span>
    </div>
  );

  if (!mounted) return fallback;

  return (
    <Suspense fallback={fallback}>
      <Inner {...props} />
    </Suspense>
  );
}
