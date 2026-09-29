import { CircleMarker, MapContainer, Popup, TileLayer, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { useNavigate } from "@tanstack/react-router";
import type { Station } from "@/data/skyguard";
import { formatDateTime, statusTone } from "@/data/skyguard";
import { TONE_HEX, type Tone } from "./ui-kit";

export interface IndiaMapProps {
  stations: Station[];
  center?: [number, number];
  zoom?: number;
  height?: number | string;
  onSelect?: (id: string) => void;
  compact?: boolean;
  highlightId?: string;
  selectedId?: string;
}

export default function IndiaMapInner({
  stations,
  center = [22.6, 79.5],
  zoom = 4.6,
  height = 520,
  onSelect,
  compact = false,
  highlightId,
  selectedId,
}: IndiaMapProps) {
  const navigate = useNavigate();

  return (
    <MapContainer
      center={center}
      zoom={zoom}
      minZoom={3}
      scrollWheelZoom
      style={{ height, width: "100%", background: "#05131b" }}
      worldCopyJump={false}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />
      {stations.map((s) => {
        const tone = statusTone(s.status) as Tone;
        const color = TONE_HEX[tone];
        const isHero = s.station_id === (highlightId ?? selectedId);
        const isSimulated =
          s.is_simulated ||
          s.station_type === "SIMULATED_INDICATIVE" ||
          !["43189", "43150", "43245"].includes(s.station_id);
        const sourceLabel = isSimulated ? "SIMULATED INDICATIVE" : "IMD AWS REFERENCE";

        return (
          <CircleMarker
            key={s.station_id}
            center={[s.latitude, s.longitude]}
            radius={isHero ? 10 : compact ? 6 : 7}
            pathOptions={{
              color: isHero ? "#2dd4bf" : isSimulated ? "#a78bfa" : "#ffffff",
              weight: isHero ? 2.5 : isSimulated ? 1.8 : 1.5,
              dashArray: isSimulated ? "3, 3" : undefined,
              fillColor: color,
              fillOpacity: 0.92,
            }}
            eventHandlers={{ click: () => onSelect?.(s.station_id) }}
          >
            {compact && (
              <Tooltip direction="top" offset={[0, -6]}>
                <span style={{ fontSize: 11, fontFamily: "IBM Plex Mono" }}>
                  {isSimulated ? "[SIM] " : "[IMD] "}
                  {s.station_id} · {s.station_name} ·{" "}
                  {s.temperature !== null ? `${s.temperature.toFixed(1)} °C` : "No data"}
                </span>
              </Tooltip>
            )}
            {!compact && (
              <Popup>
                <div className="text-[12px] text-slate-100 font-sans" style={{ minWidth: 220 }}>
                  <div
                    className="flex items-center justify-between gap-2 px-3 py-2 text-white rounded-t-md"
                    style={{
                      background: "#0a1f2e",
                      borderBottom: "1px solid rgba(45,212,191,0.2)",
                    }}
                  >
                    <div className="flex items-center gap-1.5 font-mono text-xs font-semibold">
                      <span>{s.station_id}</span>
                      <span
                        className="text-[9px] px-1 py-0.2 rounded font-sans font-bold uppercase"
                        style={{
                          background: isSimulated
                            ? "rgba(167,139,250,0.2)"
                            : "rgba(16,185,129,0.2)",
                          color: isSimulated ? "#c084fc" : "#34d399",
                          border: isSimulated
                            ? "1px solid rgba(167,139,250,0.4)"
                            : "1px solid rgba(16,185,129,0.4)",
                        }}
                      >
                        {isSimulated ? "SIMULATED" : "IMD AWS"}
                      </span>
                    </div>
                    <span
                      className="rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider font-mono"
                      style={{ background: color, color: "#fff" }}
                    >
                      {s.status}
                    </span>
                  </div>
                  <div className="space-y-1.5 px-3 py-2.5" style={{ background: "#08141c" }}>
                    <div>
                      <p className="text-sm font-semibold leading-tight text-white">
                        {s.station_name}
                      </p>
                      <p className="text-[11px] text-slate-400">
                        {s.district}, {s.state}
                      </p>
                    </div>

                    <div className="text-[10px] font-mono text-slate-400 pt-0.5">
                      Provenance:{" "}
                      <span
                        className={
                          isSimulated
                            ? "text-violet-300 font-semibold"
                            : "text-emerald-300 font-semibold"
                        }
                      >
                        {sourceLabel}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-1 border-y border-white/10 py-1.5 font-mono text-center">
                      <div>
                        <p className="text-[9px] uppercase tracking-wider text-slate-400 font-sans">
                          Temp
                        </p>
                        <p className="text-xs font-semibold text-teal-300">
                          {s.temperature !== null ? `${s.temperature.toFixed(1)}°C` : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-[9px] uppercase tracking-wider text-slate-400 font-sans">
                          Pressure
                        </p>
                        <p className="text-xs font-semibold text-slate-200">
                          {s.pressure !== null ? `${s.pressure.toFixed(1)}` : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-[9px] uppercase tracking-wider text-slate-400 font-sans">
                          Humidity
                        </p>
                        <p className="text-xs font-semibold text-slate-200">
                          {s.humidity !== null ? `${s.humidity}%` : "—"}
                        </p>
                      </div>
                    </div>
                    <p className="text-[11px] font-mono">
                      <span className="text-slate-400 font-sans">Observed: </span>
                      <span className="text-slate-300">{formatDateTime(s.timestamp)}</span>
                    </p>
                    <p className="text-[11px]">
                      <span className="text-slate-400">Anomaly State: </span>
                      <span
                        className={
                          s.anomaly_type
                            ? "text-rose-300 font-semibold"
                            : "text-emerald-300 font-medium"
                        }
                      >
                        {s.anomaly_type ?? "Nominal (None)"}
                        {s.severity ? ` · ${s.severity}` : ""}
                      </span>
                    </p>
                    <div className="flex gap-1.5 pt-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          onSelect?.(s.station_id);
                          navigate({ to: "/station" });
                        }}
                        className="flex-1 rounded-md bg-teal-500/15 border border-teal-400/35 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-teal-200 hover:bg-teal-500/25 cursor-pointer transition-colors"
                      >
                        Station Detail
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          onSelect?.(s.station_id);
                          navigate({ to: "/diagnostics" });
                        }}
                        className="flex-1 rounded-md border border-white/20 bg-white/5 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-slate-300 hover:bg-white/10 cursor-pointer transition-colors"
                      >
                        Diagnostics
                      </button>
                    </div>
                  </div>
                </div>
              </Popup>
            )}
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
