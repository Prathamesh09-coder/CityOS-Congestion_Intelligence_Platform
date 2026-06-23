import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Search, Sparkles, Activity } from "lucide-react";
import { AppShell } from "@/components/cityos/AppShell";
import { Card, PanelHeader, Badge, Button, Field, Select, ProgressBar, PageHeader } from "@/components/cityos/primitives";
import { CORRIDORS, EVENT_CAUSES, ZONES, type CityEvent } from "@/lib/cityos-data";
import { useQuery } from "@tanstack/react-query";
import { predictSimilarity, getDashboardStream } from "@/lib/api";

export const Route = createFileRoute("/similarity")({
  head: () => ({
    meta: [
      { title: "Event Similarity Engine · CityOS" },
      { name: "description", content: "Find the most similar historical events to a new incident and surface an evidence-cited recommendation." },
    ],
  }),
  component: Similarity,
});

function Similarity() {
  const [cause, setCause] = useState("vehicle_breakdown");
  const [corridor, setCorridor] = useState("Mysore Road");
  const [zone, setZone] = useState(ZONES[0]);
  const [priority, setPriority] = useState<"High" | "Low">("High");
  const [type, setType] = useState<"planned" | "unplanned">("unplanned");
  const [liveEventId, setLiveEventId] = useState<string>("");

  const handleManualChange = <T,>(setter: React.Dispatch<React.SetStateAction<T>>) => (val: T) => {
    setter(val);
    if (liveEventId) setLiveEventId("");
  };

  const { data: streamData } = useQuery({
    queryKey: ["dashboardStream"],
    queryFn: getDashboardStream,
    refetchInterval: 5000,
  });

  const activeEvents = streamData?.events || [];

  // Initialize with the first live event if none is selected yet
  useEffect(() => {
    if (activeEvents.length > 0 && !liveEventId) {
      handleSelectLiveEvent(activeEvents[0].id);
    }
  }, [activeEvents, liveEventId]);

  const handleSelectLiveEvent = (id: string) => {
    setLiveEventId(id);
    const evt = activeEvents.find((e: CityEvent) => e.id === id);
    if (evt) {
      setCause(evt.cause);
      setCorridor(evt.corridor);
      setZone(evt.zone || ZONES[0]);
      setPriority(evt.priority as "High" | "Low");
      setType(evt.type as "planned" | "unplanned");
    }
  };

  const { data, isLoading } = useQuery({
    queryKey: ["similarity", { type, cause, corridor, zone, priority }],
    queryFn: () => predictSimilarity({ event_type: type, event_cause: cause, corridor, zone, priority }),
  });

  const rows = data?.results || [];
  const meta = data?.meta || { count: 0, avg_hrs: 0, closure_pct: 0, rec_off: "", rec_bar: "", label: cause };

  return (
    <AppShell>
      <PageHeader
        title="Event Similarity Engine"
        subtitle="Given a new event's fingerprint, find the most similar historical events using cosine/Jaccard similarity on encoded features."
        right={<Badge kind="ai" icon={<Activity size={11} />}>Live Stream Connected</Badge>}
      />

      <Card padded={false}>
        <PanelHeader title="Find Similar Historical Events" right={<Search size={14} style={{ color: "var(--color-text-muted)" }} />} />
        
        {/* Live Event Auto-fill Section */}
        {activeEvents.length > 0 && (
          <div style={{ padding: "12px 18px", borderBottom: "1px dashed var(--color-border)", background: "var(--color-bg)", display: "flex", gap: 16, alignItems: "center" }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--color-text-secondary)" }}>Auto-fill from Live Event:</span>
            <div style={{ width: 300 }}>
              <Select 
                value={liveEventId} 
                onChange={handleSelectLiveEvent} 
                options={[
                  { value: "", label: "Select an active TomTom incident..." },
                  ...activeEvents.map((e: CityEvent) => ({ value: e.id, label: `${(e.cause || "Unknown").replace(/_/g, " ")} on ${e.corridor} (${e.id})` }))
                ]} 
              />
            </div>
            <span style={{ fontSize: 11, color: "var(--color-text-muted)" }}>Select an active TomTom incident to populate the search fingerprint.</span>
          </div>
        )}

        <div style={{ padding: 18, display: "grid", gridTemplateColumns: "repeat(5, 1fr) auto", gap: 12, alignItems: "end" }}>
          <Field label="Event Type">
            <div style={{ display: "flex", gap: 6 }}>
              {(["planned", "unplanned"] as const).map((t) => (
                <button key={t} onClick={() => handleManualChange(setType)(t)} style={{
                  flex: 1, padding: "8px", borderRadius: 99, fontSize: 11, fontWeight: 600, cursor: "pointer",
                  border: `1px solid ${type === t ? "var(--color-primary)" : "var(--color-border)"}`,
                  background: type === t ? "var(--color-primary)" : "var(--color-surface)",
                  color: type === t ? "#fff" : "var(--color-text-secondary)",
                }}>{t}</button>
              ))}
            </div>
          </Field>
          <Field label="Event Cause">
            <Select value={cause} onChange={handleManualChange(setCause)} options={EVENT_CAUSES.map((c) => ({ value: c.value, label: c.label }))} />
          </Field>
          <Field label="Corridor">
            <Select 
              value={corridor} 
              onChange={handleManualChange(setCorridor)} 
              options={[
                ...CORRIDORS.map((c) => ({ value: c.name, label: c.name })),
                ...(!CORRIDORS.some(c => c.name === corridor) ? [{ value: corridor, label: corridor }] : [])
              ]} 
            />
          </Field>
          <Field label="Zone">
            <Select value={zone} onChange={handleManualChange(setZone)} options={ZONES.map((z) => ({ value: z, label: z }))} />
          </Field>
          <Field label="Priority">
            <div style={{ display: "flex", gap: 6 }}>
              {(["High", "Low"] as const).map((p) => (
                <button key={p} onClick={() => handleManualChange(setPriority)(p)} style={{
                  flex: 1, padding: "8px", borderRadius: 99, fontSize: 11, fontWeight: 600, cursor: "pointer",
                  border: `1px solid ${priority === p ? "var(--color-primary)" : "var(--color-border)"}`,
                  background: priority === p ? "var(--color-primary)" : "var(--color-surface)",
                  color: priority === p ? "#fff" : "var(--color-text-secondary)",
                }}>{p}</button>
              ))}
            </div>
          </Field>
          <Button variant="primary" style={{ padding: "9px 16px" }}>
            <Search size={14} /> Search
          </Button>
        </div>
      </Card>

      {/* AI Recommendation Banner */}
      <div
        className="slide-up"
        style={{
          marginTop: 16,
          background: "var(--color-ai-accent-light)",
          border: "1px solid var(--color-ai-accent)",
          borderRadius: 10,
          padding: "14px 18px",
          display: "flex", gap: 12, alignItems: "flex-start",
        }}
      >
        <Sparkles size={20} style={{ color: "var(--color-ai-accent)", flexShrink: 0, marginTop: 2 }} />
        <div style={{ fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.5 }}>
          Based on <b>{meta.count} similar {meta.label.toLowerCase()} events</b> on {corridor} with <b>{priority}</b> priority,
          predicted duration is <b style={{ color: "var(--color-ai-accent)" }}>{meta.avg_hrs} hrs</b>.
          Road closure required in <b>{meta.closure_pct}%</b> of cases.
          Recommend <b>{meta.rec_off} officers</b> and <b>{meta.rec_bar} barricades</b>.
        </div>
      </div>

      {/* Results Table */}
      <Card padded={false} style={{ marginTop: 16 }}>
        <PanelHeader title="Most Similar Historical Events" right={<Badge kind="ai">{rows.length} matches</Badge>} />
        <div style={{ overflow: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ background: "var(--color-surface-elevated)" }}>
                {["Event ID", "Similarity", "Cause", "Corridor", "Priority", "Closure", "Duration", "Status", "Time", "AI Officers", "AI Barricades"].map((h) => (
                  <th key={h} style={{ padding: "10px 12px", textAlign: "left", fontWeight: 600, fontSize: 11, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={r.id} style={{ background: i % 2 === 0 ? "var(--color-surface)" : "var(--color-bg)", borderTop: "1px solid var(--color-border)" }}>
                  <td style={{ padding: "10px 12px", fontFamily: "ui-monospace, monospace", color: "var(--color-text-secondary)" }}>{r.id}</td>
                  <td style={{ padding: "10px 12px", minWidth: 140 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ flex: 1, minWidth: 60 }}><ProgressBar value={r.sim} color="var(--color-primary)" /></div>
                      <span style={{ color: "var(--color-primary)", fontWeight: 600 }}>{r.sim}%</span>
                    </div>
                  </td>
                  <td style={{ padding: "10px 12px", color: "var(--color-text-primary)" }}>{r.cause}</td>
                  <td style={{ padding: "10px 12px", color: "var(--color-text-primary)" }}>{r.corridor}</td>
                  <td style={{ padding: "10px 12px" }}><Badge kind={r.priority === "High" ? "high" : "low"}>{r.priority}</Badge></td>
                  <td style={{ padding: "10px 12px" }}>{r.closure ? <Badge kind="closure">Yes</Badge> : <span style={{ color: "var(--color-text-muted)" }}>No</span>}</td>
                  <td style={{ padding: "10px 12px", color: "var(--color-text-primary)" }}>{r.duration} hrs</td>
                  <td style={{ padding: "10px 12px" }}>
                    {r.status === "closed" ? <Badge kind="resolved">closed</Badge> : <span style={{ color: "var(--color-text-muted)" }}>—</span>}
                  </td>
                  <td style={{ padding: "10px 12px" }}>{r.peak ? <Badge kind="warning">Peak</Badge> : <span style={{ color: "var(--color-text-muted)" }}>off-peak</span>}</td>
                  <td style={{ padding: "10px 12px" }}><Badge kind="ai">{r.officers}</Badge></td>
                  <td style={{ padding: "10px 12px" }}><Badge kind="ai">{r.barricades}</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </AppShell>
  );
}
