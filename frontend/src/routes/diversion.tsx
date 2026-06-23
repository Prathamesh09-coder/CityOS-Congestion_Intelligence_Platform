import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Route as RouteIcon, Clock, Sparkles, Activity } from "lucide-react";
import { AppShell } from "@/components/cityos/AppShell";
import { Card, PanelHeader, Badge, ProgressBar, PageHeader } from "@/components/cityos/primitives";
import { CityMap } from "@/components/cityos/CityMap";
import { ACTIVE_EVENTS, type CityEvent } from "@/lib/cityos-data";
import { useTrafficPrediction } from "../hooks/use-predict";
import { API_BASE_URL } from "@/lib/api";

export const Route = createFileRoute("/diversion")({
  head: () => ({
    meta: [
      { title: "Smart Diversion Engine · CityOS" },
      { name: "description", content: "Recommend optimal diversion routes for affected junctions, with before/after corridor flow visualization." },
    ],
  }),
  component: Diversion,
});

const CONFIDENCE_TABLE = [
  { cause: "VIP Movement", pct: 80, kind: "success" as const, label: "High" },
  { cause: "Public Event", pct: 46, kind: "warning" as const, label: "Medium" },
  { cause: "Construction", pct: 27, kind: "warning" as const, label: "Medium-Low" },
  { cause: "Vehicle Breakdown", pct: 4, kind: "neutral" as const, label: "Low" },
];

function Diversion() {
  const [selected, setSelected] = useState<CityEvent>(ACTIVE_EVENTS[0]);
  const queryClient = useQueryClient();
  const [telemetryStatus, setTelemetryStatus] = useState<string | null>(null);
  const [kafkaStatus, setKafkaStatus] = useState<string | null>(null);

  // Query M4 Graph WaveNet Traffic Forecasting Backbone
  const now = new Date();
  const trafficQuery = useTrafficPrediction({
    lat: [selected.lat],
    lng: [selected.lng],
    reported_datetime: now.toISOString(),
  });

  const trafficData = trafficQuery.data;
  const isPending = trafficQuery.isLoading;

  const simulateTomTom = async () => {
    setTelemetryStatus("Injecting...");
    try {
      const res = await fetch(`${API_BASE_URL}/stream/traffic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: selected.lat,
          lng: selected.lng,
          speed_kmh: 8.5,
          flow_veh_hr: 1850,
          congestion_index: 0.95,
        }),
      });
      if (res.ok) {
        setTelemetryStatus("Success! Blended");
        queryClient.invalidateQueries({ queryKey: ["trafficPrediction"] });
        setTimeout(() => setTelemetryStatus(null), 3000);
      } else {
        setTelemetryStatus("Error");
        setTimeout(() => setTelemetryStatus(null), 3000);
      }
    } catch (err) {
      setTelemetryStatus("Offline");
      setTimeout(() => setTelemetryStatus(null), 3000);
    }
  };

  const simulateKafka = async () => {
    setKafkaStatus("Ingesting...");
    try {
      const res = await fetch(`${API_BASE_URL}/stream/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_id: `evt_${Math.floor(Math.random() * 9000 + 1000)}`,
          event_cause: selected.cause,
          corridor: selected.corridor,
          priority: "High",
          reported_datetime: new Date().toISOString(),
          description: "Ingested via Live Stream Simulator",
          comment: "Kafka active telemetry trigger",
        }),
      });
      if (res.ok) {
        setKafkaStatus("Success!");
        queryClient.invalidateQueries({ queryKey: ["trafficPrediction"] });
        setTimeout(() => setKafkaStatus(null), 3000);
      } else {
        setKafkaStatus("Error");
        setTimeout(() => setKafkaStatus(null), 3000);
      }
    } catch (err) {
      setKafkaStatus("Offline");
      setTimeout(() => setKafkaStatus(null), 3000);
    }
  };

  // Dynamically update event closure and priority based on traffic prediction congestion index
  const dynamicClosure = trafficData ? (trafficData.metrics.congestion_index > 0.5) : selected.closure;
  const dynamicPriority = trafficData && trafficData.metrics.congestion_index > 0.5 ? "High" : selected.priority;
  
  const dynamicEvent: CityEvent = {
    ...selected,
    closure: dynamicClosure,
    priority: dynamicPriority,
  };

  const isFallback = trafficData?.road_network?.graph_validation_status?.includes("fallback");
  const delayText = trafficData ? `+${trafficData.metrics.average_delay_minutes} min` : "+18 min";
  const modelMode = trafficData 
    ? (isFallback ? "Local Simulation (Offline)" : "M4 WaveNet Backbone Active") 
    : "Offline Fallback Simulation";

  return (
    <AppShell>
      <style>{`
        @keyframes shimmer-eff {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        .shimmer-active {
          background: linear-gradient(90deg, var(--color-surface-elevated) 25%, var(--color-border) 50%, var(--color-surface-elevated) 75%);
          background-size: 200% 100%;
          animation: shimmer-eff 1.5s infinite linear;
        }
      `}</style>

      <PageHeader title="Smart Diversion Engine" subtitle="Recommend optimal diversion plans for affected junctions and corridors." />

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 360px", gap: 16, height: "calc(100vh - 240px)", minHeight: 600 }}>
        {/* LEFT */}
        <Card padded={false} style={{ display: "flex", flexDirection: "column" }}>
          <PanelHeader title="Affected Junctions" />
          <div style={{ overflow: "auto", padding: 12, display: "flex", flexDirection: "column", gap: 8, flex: 1 }}>
            {ACTIVE_EVENTS.map((e) => {
              const isSelected = selected.id === e.id;
              return (
                <button key={e.id} onClick={() => setSelected(e)} style={{
                  textAlign: "left", padding: "10px 12px", borderRadius: 8, cursor: "pointer",
                  background: isSelected ? "var(--color-primary-light)" : "var(--color-surface)",
                  border: isSelected ? "2px solid var(--color-primary)" : "1px solid var(--color-border)",
                }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--color-text-primary)" }}>{e.junction}</div>
                  <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 2 }}>{e.corridor}</div>
                  {e.closure && <div style={{ marginTop: 4 }}><Badge kind="closure">Closure</Badge></div>}
                </button>
              );
            })}
          </div>

          {/* Stream Ingestion Simulator Controls */}
          <div style={{ padding: 12, borderTop: "1px solid var(--color-border)", background: "var(--color-surface-elevated)" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8, display: "flex", alignItems: "center", gap: 4 }}>
              <Sparkles size={11} style={{ color: "var(--color-ai-accent)" }} />
              <span>Stream Ingestion Simulator</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <button 
                onClick={simulateTomTom}
                style={{
                  width: "100%", padding: "6px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600, cursor: "pointer",
                  background: "var(--color-primary)", color: "#fff", border: "none", display: "flex", justifyContent: "space-between", alignItems: "center"
                }}
              >
                <span>Trigger TomTom Speed (8.5 km/h)</span>
                {telemetryStatus && <span style={{ fontSize: 9, opacity: 0.8 }}>{telemetryStatus}</span>}
              </button>
              <button 
                onClick={simulateKafka}
                style={{
                  width: "100%", padding: "6px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600, cursor: "pointer",
                  background: "var(--color-surface)", color: "var(--color-text-secondary)", border: "1px solid var(--color-border)", display: "flex", justifyContent: "space-between", alignItems: "center"
                }}
              >
                <span>Trigger Kafka Event Stream</span>
                {kafkaStatus && <span style={{ fontSize: 9, opacity: 0.8 }}>{kafkaStatus}</span>}
              </button>
            </div>
          </div>
        </Card>

        {/* CENTER — Before / After comparison */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, flex: 1, minHeight: 0 }}>
            <Card padded={false} style={{ display: "flex", flexDirection: "column" }}>
              <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--color-border)", fontSize: 12, fontWeight: 600, color: "var(--color-text-primary)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>Current Flow</span>
                <Badge kind={dynamicClosure ? "closure" : "neutral"}>
                  {dynamicClosure ? "Affected Corridor (Closed)" : "Affected Corridor"}
                </Badge>
              </div>
              <div style={{ flex: 1, padding: 8 }}>
                <CityMap events={[dynamicEvent]} />
              </div>
            </Card>
            <Card padded={false} style={{ display: "flex", flexDirection: "column" }}>
              <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--color-border)", fontSize: 12, fontWeight: 600, color: "var(--color-text-primary)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>Diverted Flow</span>
                <Badge kind="success">AI route active</Badge>
              </div>
              <div style={{ flex: 1, padding: 8 }}>
                <CityMap events={[dynamicEvent]} showDiversion={true} diversionRoute={trafficData?.diversion_route} />
              </div>
            </Card>
          </div>
          <div style={{ fontSize: 11, color: "var(--color-text-muted)", fontStyle: "italic", textAlign: "center" }}>
            Real-time Spatio-Temporal Graph WaveNet forecasts based on junction network topology.
          </div>
        </div>

        {/* RIGHT */}
        <Card padded={false} style={{ borderLeft: "3px solid var(--color-ai-accent)", display: "flex", flexDirection: "column" }}>
          <PanelHeader title="Diversion Plan" accent right={<RouteIcon size={14} style={{ color: "var(--color-ai-accent)" }} />} />
          <div style={{ padding: 16, overflow: "auto", display: "flex", flexDirection: "column", gap: 12, fontSize: 12 }}>
            <KV label="Affected Corridor" value={<b style={{ color: "var(--color-text-primary)" }}>{selected.corridor}</b>} />
            <KV label="Disruption Cause" value={(selected.cause || "Unknown").replace(/_/g, " ")} />
            <KV label="Road Closure Prediction" value={<Badge kind={dynamicClosure ? "closure" : "neutral"}>{dynamicClosure ? "Required" : "Not required"}</Badge>} />
            
            {/* Live Model Telemetry metrics */}
            <div style={{ marginTop: 4 }}>
              <div style={{ color: "var(--color-text-secondary)", marginBottom: 6, display: "flex", alignItems: "center", gap: 4 }}>
                <Activity size={12} style={{ color: "var(--color-ai-accent)" }} />
                <span>Live Traffic Telemetry</span>
              </div>
              {isPending ? (
                <div className="shimmer-active" style={{ height: 96, borderRadius: 8 }} />
              ) : trafficData ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, background: "var(--color-bg)", padding: 10, borderRadius: 8 }}>
                  <div>
                    <div style={{ fontSize: 9, color: "var(--color-text-muted)" }}>PREDICTED SPEED</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "var(--color-primary)" }}>
                      {trafficData.metrics.predicted_speed_kmh} km/h
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: "var(--color-text-muted)" }}>PREDICTED FLOW</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "var(--color-primary)" }}>
                      {trafficData.metrics.predicted_flow_veh_hr} v/h
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: "var(--color-text-muted)" }}>CONGESTION INDEX</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "var(--color-critical)" }}>
                      {trafficData.metrics.congestion_index}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: "var(--color-text-muted)" }}>CONGESTED LINKS</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "var(--color-warning)" }}>
                      {trafficData.road_network.adjacent_segments_congested} / {trafficData.road_network.active_nodes_evaluated}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ color: "var(--color-text-muted)", fontStyle: "italic" }}>Failed to load traffic metrics.</div>
              )}
            </div>

            <div>
              <div style={{ color: "var(--color-text-secondary)", marginBottom: 6 }}>Alternative Corridor Options</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                <Badge kind="ai">Bellary Rd 2</Badge>
                <Badge kind="ai">ORR North 1</Badge>
                <Badge kind="ai">Old Madras Rd</Badge>
              </div>
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ color: "var(--color-text-secondary)" }}>Travel Time Delay</span>
                <span style={{ color: "var(--color-warning)", fontWeight: 600 }}>{isPending ? "Calculating..." : delayText}</span>
              </div>
              <span style={{ fontSize: 10, color: "var(--color-text-muted)" }}>
                <Clock size={9} style={{ display: "inline", marginRight: 4 }} /> {modelMode}
              </span>
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ color: "var(--color-text-secondary)" }}>Diversion Confidence</span>
                <span style={{ color: "var(--color-ai-accent)", fontWeight: 600 }}>{selected.confidence}%</span>
              </div>
              <ProgressBar value={selected.confidence} color="var(--color-ai-accent)" />
            </div>
            <div>
              <div style={{ color: "var(--color-text-secondary)", marginBottom: 6 }}>Routes</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                <Badge kind="success">Emergency: {selected.junction} → parallel</Badge>
                <Badge kind="ai">Backup: parallel segment</Badge>
              </div>
            </div>

            <div style={{ marginTop: 4, padding: 10, background: "var(--color-bg)", borderRadius: 8, border: "1px solid var(--color-border)" }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
                Closure Confidence by Cause
              </div>
              {CONFIDENCE_TABLE.map((c) => (
                <div key={c.cause} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "4px 0", fontSize: 11 }}>
                  <span style={{ color: "var(--color-text-primary)" }}>{c.cause}</span>
                  <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>{c.pct}%</span>
                    <Badge kind={c.kind}>{c.label}</Badge>
                  </span>
                </div>
              ))}
            </div>

            <div style={{ padding: 10, background: "var(--color-ai-accent-light)", borderRadius: 8, fontSize: 11, color: "var(--color-text-primary)", lineHeight: 1.4 }}>
              <Sparkles size={11} style={{ color: "var(--color-ai-accent)", display: "inline", marginRight: 4 }} />
              Recommended diversion redistributes ~62% of corridor traffic to parallel routes within zone. Estimated 18-min delay reduction vs. no-diversion baseline.
            </div>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

function KV({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
      <span style={{ color: "var(--color-text-secondary)" }}>{label}</span>
      <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{value}</span>
    </div>
  );
}
