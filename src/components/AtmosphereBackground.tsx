/**
 * SkyGuard AI — Calm ambient background
 * Replaces high-CPU animated particle loops and glowing spotlights with a restrained,
 * static meteorological atmospheric backdrop.
 */
export function AtmosphereBackground() {
  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden" aria-hidden="true">
      {/* Subtle top synoptic gradient */}
      <div className="absolute top-0 inset-x-0 h-64 bg-gradient-to-b from-sky-950/20 via-slate-900/5 to-transparent pointer-events-none" />
    </div>
  );
}
