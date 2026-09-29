import fs from "node:fs";

const files = [
  "src/routes/index.tsx",
  "src/routes/station.tsx",
  "src/routes/diagnostics.tsx",
  "src/routes/sensors.tsx",
  "src/routes/admin.tsx",
];

const pairs = [
  ["bg-emerald-500/120/[0.06]", "bg-emerald-400/10"],
  ["bg-emerald-500/120/15", "bg-emerald-400/15"],
  ["bg-emerald-500/120/12", "bg-emerald-400/12"],
  ["bg-emerald-500/120", "bg-emerald-400"],
  ["bg-red-500/120/10", "bg-red-500/10"],
  ["bg-red-500/18/60", "bg-red-500/15"],
  ["bg-amber-500/120/12", "bg-amber-500/12"],
  ["bg-amber-500/120/10", "bg-amber-500/10"],
  ["bg-amber-500/120", "bg-amber-400"],
  ["bg-cyan-400/120/12", "bg-cyan-400/12"],
  ["bg-cyan-400/120/10", "bg-cyan-400/10"],
  ["border-amber-400/50/25", "border-amber-400/25"],
  ["hover:border-slate-300", "hover:border-white/20"],
  ["bg-amber-100", "bg-amber-400/18"],
  ["bg-accent text-white", "bg-accent text-slate-950"],
  ['stroke="#cbd5e1"', 'stroke="#23404c"'],
  ['stroke="#94a3b8"', 'stroke="#3d5c68"'],
  ['fill="#64748b"', 'fill="#8aa4ae"'],
  ['fill="#006194"', 'fill="#5eead4"'],
  ['fill="#059669"', 'fill="#5eead4"'],
  ['stroke="#e2e8f0"', 'stroke="#1e3a44"'],
  ['fill: "#64748b"', 'fill: "#8aa4ae"'],
  ['backgroundColor: "#0b1c30"', 'backgroundColor: "#07141c"'],
  ['stroke: "#006194"', 'stroke: "#5eead4"'],
  ['fill="#fee2e2"', 'fill="#3f1216"'],
  ['stroke="#fca5a5"', 'stroke="#f87171"'],
  ['fill="#991b1b"', 'fill="#fecaca"'],
];

for (const f of files) {
  let s = fs.readFileSync(f, "utf8");
  for (const [a, b] of pairs) s = s.split(a).join(b);
  fs.writeFileSync(f, s);
  console.log("fixed", f);
}
