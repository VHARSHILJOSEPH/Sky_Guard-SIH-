import { useEffect, useMemo } from "react";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { Station } from "@/data/skyguard";
import { statusTone } from "@/data/skyguard";
import { TONE_HEX, type Tone } from "./ui-kit";

export interface StationMapProps {
  stations: Station[];
  selectedStationId: string;
  onSelectStation?: (id: string) => void;
  height?: number | string;
  spatialRadius?: number;
}

function calculateDistanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return Math.round(R * c);
}

// Helper to auto-recenter map when selected station changes
function MapRecenter({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, map.getZoom(), { animate: true });
  }, [center, map]);
  return null;
}

export default function StationMapInner({
  stations,
  selectedStationId,
  onSelectStation,
  height = 360,
  spatialRadius = 150,
}: StationMapProps) {
  const selectedStation = useMemo(
    () => stations.find((s) => s.station_id === selectedStationId) || stations[0],
    [stations, selectedStationId],
  );

  // Calculate nearby stations within spatial cluster
  const nearbyStations = useMemo(() => {
    if (!selectedStation) return [];
    return stations
      .filter((s) => s.station_id !== selectedStation.station_id)
      .map((s) => ({
        station: s,
        distance: calculateDistanceKm(
          selectedStation.latitude,
          selectedStation.longitude,
          s.latitude,
          s.longitude,
        ),
      }))
      .sort((a, b) => a.distance - b.distance);
  }, [stations, selectedStation]);

  const clusterStations = useMemo(
    () => nearbyStations.filter((n) => n.distance <= spatialRadius),
    [nearbyStations, spatialRadius],
  );

  if (!selectedStation) {
    return null;
  }

  const center: [number, number] = [selectedStation.latitude, selectedStation.longitude];

  return (
    <div className="relative w-full rounded-xl overflow-hidden border border-border bg-card">
      {/* Top Overlay Badge */}
      <div className="absolute top-3 left-3 z-[1000] bg-card/90 backdrop-blur-sm border border-border px-3 py-1.5 rounded-lg shadow-xs flex items-center gap-2 text-xs">
        <span className="w-2 h-2 rounded-full bg-sky-400" />
        <span className="font-medium text-foreground">Spatial Network Context</span>
        <span className="text-muted-foreground font-mono text-[11px]">
          ({clusterStations.length} within {spatialRadius}km)
        </span>
      </div>

      <MapContainer
        center={center}
        zoom={6.5}
        minZoom={4}
        maxZoom={12}
        scrollWheelZoom={false}
        style={{ height, width: "100%", background: "#090d16" }}
        attributionControl={false}
      >
        <MapRecenter center={center} />

        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          subdomains="abcd"
          maxZoom={19}
        />

        {/* Spatial Corroboration Connection Lines */}
        {clusterStations.map(({ station: s, distance }) => (
          <Polyline
            key={`line-${s.station_id}`}
            positions={[
              [selectedStation.latitude, selectedStation.longitude],
              [s.latitude, s.longitude],
            ]}
            pathOptions={{
              color: s.status === "NORMAL" ? "rgba(16, 185, 129, 0.35)" : "rgba(239, 68, 68, 0.4)",
              weight: 1.5,
              dashArray: "4, 6",
            }}
          />
        ))}

        {/* All Stations Markers */}
        {stations.map((s) => {
          const isSelected = s.station_id === selectedStation.station_id;
          const tone = statusTone(s.status) as Tone;
          const color = TONE_HEX[tone] || "#10b981";

          return (
            <div key={s.station_id}>
              {/* Outer emphasis ring for selected station */}
              {isSelected && (
                <CircleMarker
                  center={[s.latitude, s.longitude]}
                  radius={14}
                  pathOptions={{
                    color: "#38bdf8",
                    weight: 2,
                    fillOpacity: 0.15,
                    fillColor: "#38bdf8",
                  }}
                />
              )}

              <CircleMarker
                center={[s.latitude, s.longitude]}
                radius={isSelected ? 8 : 5}
                pathOptions={{
                  color: isSelected ? "#ffffff" : "rgba(255, 255, 255, 0.4)",
                  weight: isSelected ? 2 : 1,
                  fillColor: color,
                  fillOpacity: 0.95,
                }}
                eventHandlers={{
                  click: () => onSelectStation?.(s.station_id),
                }}
              >
                <Tooltip direction="top" offset={[0, -6]}>
                  <div className="text-[11px] font-sans">
                    <p className="font-semibold text-white">
                      {s.station_name} {isSelected && "(Selected AWS)"}
                    </p>
                    <p className="font-mono text-slate-300">
                      {s.temperature !== null ? `${s.temperature.toFixed(1)} °C` : "No data"} ·{" "}
                      {s.status}
                    </p>
                  </div>
                </Tooltip>

                <Popup>
                  <div className="p-2 space-y-1.5 text-xs">
                    <div className="flex items-center justify-between gap-2 border-b border-border pb-1">
                      <strong className="text-foreground">{s.station_name}</strong>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {s.station_id}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-1 font-mono text-center pt-1">
                      <div className="bg-muted/40 p-1 rounded">
                        <div className="text-[10px] text-muted-foreground">TEMP</div>
                        <div className="font-semibold">
                          {s.temperature !== null ? `${s.temperature.toFixed(1)}°C` : "—"}
                        </div>
                      </div>
                      <div className="bg-muted/40 p-1 rounded">
                        <div className="text-[10px] text-muted-foreground">PRESS</div>
                        <div className="font-semibold">
                          {s.pressure !== null ? `${s.pressure.toFixed(1)}` : "—"}
                        </div>
                      </div>
                      <div className="bg-muted/40 p-1 rounded">
                        <div className="text-[10px] text-muted-foreground">HUM</div>
                        <div className="font-semibold">
                          {s.humidity !== null ? `${s.humidity}%` : "—"}
                        </div>
                      </div>
                    </div>
                    <div className="text-[11px] pt-1 text-muted-foreground flex justify-between">
                      <span>Status:</span>
                      <span className="font-medium text-foreground">{s.status}</span>
                    </div>
                    {!isSelected && (
                      <button
                        type="button"
                        onClick={() => onSelectStation?.(s.station_id)}
                        className="w-full mt-1.5 py-1 px-2 text-center rounded bg-sky-500/15 text-sky-300 hover:bg-sky-500/25 text-[11px] font-medium transition-colors cursor-pointer"
                      >
                        Set as Active Station
                      </button>
                    )}
                  </div>
                </Popup>
              </CircleMarker>
            </div>
          );
        })}
      </MapContainer>

      {/* Bottom Operational Map Legend */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2 border-t border-border bg-muted/30 text-[11px] font-mono text-muted-foreground">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            Normal
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            Warning
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            Anomaly
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-slate-400" />
            Offline
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded-full border-2 border-sky-400 bg-sky-400/20 inline-block" />
            Selected Station
          </span>
        </div>
      </div>
    </div>
  );
}
