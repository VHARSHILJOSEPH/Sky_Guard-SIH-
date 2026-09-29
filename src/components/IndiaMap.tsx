import { lazy, Suspense, useEffect, useState } from "react";
import type { IndiaMapProps } from "./IndiaMapInner";

const Inner = lazy(() => import("./IndiaMapInner"));

export function IndiaMap(props: IndiaMapProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const fallback = (
    <div
      className="flex items-center justify-center bg-secondary text-xs text-muted-foreground"
      style={{ height: props.height ?? 520 }}
    >
      Loading geographic view…
    </div>
  );

  if (!mounted) return fallback;

  return (
    <Suspense fallback={fallback}>
      <Inner {...props} />
    </Suspense>
  );
}
