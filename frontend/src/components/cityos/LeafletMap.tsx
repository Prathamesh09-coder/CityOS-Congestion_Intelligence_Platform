import { useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Circle, Polyline, Tooltip, useMap } from "react-leaflet";
import type { CityEvent } from "@/lib/cityos-data";
import { JUNCTIONS } from "@/lib/cityos-data";
import { useTheme } from "./theme";
import "leaflet/dist/leaflet.css";

interface Props {
  events: CityEvent[];
  selectedId?: string;
  onSelect?: (e: CityEvent) => void;
  showHeatmap?: boolean;
  showDiversion?: boolean;
  height?: number | string;
  diversionRoute?: number[][];
}

const CORRIDOR_PATHS = [
  // Mysore Rd
  [[12.9716, 77.5946], [12.9600, 77.5600], [12.9556, 77.5385]],
  // Hosur Rd
  [[12.9716, 77.5946], [12.9500, 77.6100], [12.9176, 77.6244]],
  // Bellary Rd
  [[12.9716, 77.5946], [13.0084, 77.5906], [13.1007, 77.5963]],
  // Nagavara ORR
  [[13.0416, 77.6248], [13.0084, 77.5906], [13.0371, 77.5255]]
];



function ChangeView({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

export default function LeafletMap({ events, selectedId, onSelect, showHeatmap = true, showDiversion = false, height = "100%", diversionRoute }: Props) {
  const { theme } = useTheme();
  
  // Find active selected event to center on
  const selectedEvent = events.find(e => e.id === selectedId) || events[0];
  const center: [number, number] = selectedEvent ? [selectedEvent.lat, selectedEvent.lng] : [12.9716, 77.5946];
  const zoom = selectedId ? 14 : 12;

  const tileUrl = theme === "dark" 
    ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    : "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png";

  const getRouteForEvent = (e: CityEvent): [number, number][] => {
    const { lat, lng, id } = e;
    // Generate a pseudo-random offset based on the event ID length or character codes
    // so the diversion route shapes look slightly different but are stable per-event.
    let seed = 0;
    if (id) {
      for (let i = 0; i < id.length; i++) {
        seed += id.charCodeAt(i);
      }
    }
    const offset1 = (seed % 10) * 0.0003 + 0.002;
    const offset2 = (seed % 7) * 0.0004 + 0.003;
    const offset3 = (seed % 5) * 0.0005 + 0.002;
    const signX = seed % 2 === 0 ? 1 : -1;
    const signY = seed % 3 === 0 ? 1 : -1;

    return [
      [lat, lng],
      [lat + offset1 * signY, lng + offset2 * signX],
      [lat + (offset1 + offset3) * signY, lng + (offset2 - offset1) * signX],
      [lat + offset3 * signY, lng - offset2 * signX],
      [lat, lng]
    ];
  };

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height,
        borderRadius: 12,
        border: "1px solid var(--color-border)",
        overflow: "hidden",
      }}
    >
      <style>{`
        .leaflet-container {
          z-index: 1;
        }
        .leaflet-tooltip {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          color: var(--color-text-primary);
          border-radius: 6px;
          box-shadow: var(--shadow-card);
          padding: 4px 8px;
        }
      `}</style>
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ width: "100%", height: "100%", background: theme === "dark" ? "#0B0E15" : "#EAEEF4" }}
        zoomControl={true}
        scrollWheelZoom={true}
      >
        <ChangeView center={center} zoom={zoom} />
        <TileLayer
          url={tileUrl}
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />

        {/* AI Managed Corridors */}
        {CORRIDOR_PATHS.map((path, idx) => (
          <Polyline 
            key={`corr-${idx}`} 
            positions={path as [number, number][]} 
            pathOptions={{ color: theme === "dark" ? "#4D8EF8" : "#2874F0", weight: 4, opacity: 0.35 }} 
          />
        ))}

        {/* Heatmap blobs (AI estimated density) */}
        {showHeatmap && (
          <>
            <Circle center={[12.9716, 77.5946]} radius={1500} pathOptions={{ fillColor: 'var(--color-primary)', fillOpacity: 0.1, stroke: false }} />
            <Circle center={[12.9176, 77.6244]} radius={1800} pathOptions={{ fillColor: '#F97316', fillOpacity: 0.12, stroke: false }} />
            <Circle center={[13.0371, 77.5255]} radius={1200} pathOptions={{ fillColor: 'var(--color-primary)', fillOpacity: 0.1, stroke: false }} />
            <Circle center={[13.0084, 77.5906]} radius={1400} pathOptions={{ fillColor: '#F59E0B', fillOpacity: 0.1, stroke: false }} />
          </>
        )}

        {/* Diversion routes */}
        {showDiversion && (
          diversionRoute && diversionRoute.length > 0 ? (
            <Polyline
              positions={diversionRoute as [number, number][]}
              pathOptions={{
                color: 'var(--color-success)',
                weight: 3.5,
                dashArray: '5, 8',
                opacity: 0.95
              }}
            />
          ) : (
            events
              .filter((e) => e.closure)
              .map((e) => (
                <Polyline
                  key={`dv-${e.id}`}
                  positions={getRouteForEvent(e)}
                  pathOptions={{
                    color: 'var(--color-success)',
                    weight: 3.5,
                    dashArray: '5, 8',
                    opacity: 0.95
                  }}
                />
              ))
          )
        )}

        {/* Junctions */}
        {JUNCTIONS.map((j) => (
          <CircleMarker 
            key={j.name}
            center={[j.lat, j.lng]} 
            radius={4} 
            pathOptions={{ fillColor: 'var(--map-label)', fillOpacity: 0.5, color: 'transparent' }}
          >
            <Tooltip direction="top" offset={[0, -2]} opacity={0.9}>
              <span style={{ fontSize: 10, fontWeight: 500 }}>{j.name}</span>
            </Tooltip>
          </CircleMarker>
        ))}

        {/* Event pins */}
        {events.map((e) => {
          const color = e.priority === "High" ? "var(--color-critical)" : "var(--color-primary)";
          const selected = selectedId === e.id;
          return (
            <g key={e.id}>
              {/* Outer pulse ring for active events */}
              {e.status === "active" && (
                <CircleMarker
                  center={[e.lat, e.lng]}
                  radius={selected ? 15 : 11}
                  pathOptions={{
                    fillColor: color,
                    fillOpacity: 0.15,
                    color: color,
                    weight: 1,
                    dashArray: '3, 4'
                  }}
                />
              )}
              <CircleMarker
                center={[e.lat, e.lng]}
                radius={selected ? 7 : 5}
                pathOptions={{
                  fillColor: color,
                  fillOpacity: 0.9,
                  color: "#ffffff",
                  weight: 1.5
                }}
                eventHandlers={{
                  click: () => onSelect?.(e)
                }}
              >
                <Tooltip direction="top" offset={[0, -5]} opacity={0.95}>
                  <div style={{ fontSize: 11, fontWeight: 600 }}>
                    {(e.cause || 'Unknown').replace(/_/g, " ").toUpperCase()}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                    {e.corridor} · Snap: {e.junction}
                  </div>
                </Tooltip>
              </CircleMarker>
            </g>
          );
        })}
      </MapContainer>

      {/* Legend */}
      <div
        style={{
          position: "absolute",
          bottom: 12,
          left: 12,
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: 8,
          padding: "8px 10px",
          fontSize: 11,
          color: "var(--color-text-secondary)",
          display: "flex",
          gap: 12,
          alignItems: "center",
          boxShadow: "var(--shadow-card)",
          zIndex: 1000,
          pointerEvents: "none"
        }}
      >
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: 99, background: "var(--color-critical)" }} /> High Risk
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: 99, background: "var(--color-primary)" }} /> Low Risk
        </span>
        <span style={{ color: "var(--color-text-muted)" }}>AI-Estimated Heatmap</span>
      </div>
    </div>
  );
}
