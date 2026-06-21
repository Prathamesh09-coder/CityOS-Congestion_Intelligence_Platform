import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AlertTriangle, Sparkles, MapPin, Clock } from "lucide-react";
import { AppShell } from "@/components/cityos/AppShell";
import { CityMap } from "@/components/cityos/CityMap";
import { Card, PanelHeader, Badge, Kpi, Gauge, ProgressBar, Divider } from "@/components/cityos/primitives";
import { ACTIVE_EVENTS as DEFAULT_EVENTS, KPI as DEFAULT_KPI, causeMeta, type CityEvent } from "@/lib/cityos-data";
import { useQuery } from "@tanstack/react-query";
import { getDashboardStream } from "@/lib/api";

export const Route = createFileRoute("/command")({
  head: () => ({
    meta: [
      { title: "City Command Center · CityOS" },
      { name: "description", content: "Live event feed, interactive city map and AI action brief for Bengaluru traffic authorities." },
    ],
  }),
  component: CommandCenter,
});

function CommandCenter() {
  const { data } = useQuery({
    queryKey: ["dashboardStream"],
    queryFn: getDashboardStream,
    refetchInterval: 5000,
  });

  const activeEvents = data?.events?.length > 0 ? data.events : DEFAULT_EVENTS;
  const kpi = data?.kpis || DEFAULT_KPI;

  const [selected, setSelected] = useState<CityEvent>(activeEvents[0]);
  const [timeline, setTimeline] = useState(50);

  return (
    <AppShell fullBleed>
      {/* Top KPI strip */}
      <div style={{ padding: "16px 24px", borderBottom: "1px solid var(--color-border)", background: "var(--color-surface)" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 12 }}>
          <Kpi
            label="City Readiness"
            value={kpi.readiness}
            color="var(--color-primary)"
            accent={<Gauge value={kpi.readiness} size={56} color="var(--color-primary)" />}
            sub="Composite score"
          />
          <Kpi
            label="Active Events"
            value={kpi.activeEvents.toLocaleString()}
            sub="Live"
            accent={<span style={{ width: 10, height: 10, borderRadius: 99, background: "var(--color-critical)" }} className="live-dot" />}
          />
          <Kpi label="Predicted High-Risk" value={kpi.predictedHighRisk} color="var(--color-warning)" sub="Next 2 hrs · ML forecast" />
          <Kpi label="Active Road Closures" value={kpi.activeRoadClosures} color="var(--color-critical)" sub="requires_road_closure = true" />
          <Kpi label="Avg Resolution" value={`${kpi.avgResolutionHrs} hrs`} sub="Across 2,588 events" />
          <div
            className="cityos-card count-up"
            style={{ padding: "12px 16px", background: "var(--color-warning-light)", borderColor: "var(--color-warning)", display: "flex", gap: 10, alignItems: "center" }}
          >
            <AlertTriangle size={20} style={{ color: "var(--color-warning)" }} />
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--color-warning)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                Peak Hour Alert
              </div>
              <div style={{ fontSize: 12, color: "var(--color-text-primary)", marginTop: 2 }}>
                Bengaluru peak: 5–6 AM · 7–9 PM
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main 3-column layout */}
      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr 340px", height: "calc(100vh - 72px - 110px)" }}>
        {/* LEFT: Active Events Feed */}
        <aside style={{ background: "var(--color-surface)", borderRight: "1px solid var(--color-border)", display: "flex", flexDirection: "column" }}>
          <div
            className="cityos-panel-header"
            style={{ padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}
          >
            <span style={{ fontWeight: 600, fontSize: 13 }}>Active Events</span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11, color: "var(--color-text-secondary)" }}>
              <span style={{ width: 7, height: 7, borderRadius: 99, background: "var(--color-critical)" }} className="live-dot" />
              {activeEvents.length} live
            </span>
          </div>
          <div style={{ overflow: "auto", padding: 12, display: "flex", flexDirection: "column", gap: 10 }}>
            {activeEvents.map((e) => (
              <EventCard key={e.id} event={e} active={selected.id === e.id} onClick={() => setSelected(e)} />
            ))}
          </div>
        </aside>

        {/* CENTER: Map */}
        <section style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
          <div style={{ flex: 1, minHeight: 0 }}>
            <CityMap events={activeEvents} selectedId={selected.id} onSelect={setSelected} showHeatmap showDiversion />
          </div>

          {/* Timeline slider */}
          <div className="cityos-card" style={{ padding: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ display: "flex", gap: 16, fontSize: 11, fontWeight: 600 }}>
                <span style={{ color: "var(--color-text-muted)" }}>← Past</span>
                <span style={{ color: "var(--color-primary)", display: "inline-flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 7, height: 7, borderRadius: 99, background: "var(--color-primary)" }} /> Present
                </span>
                <span style={{ color: "var(--color-ai-accent)" }}>Future →</span>
              </div>
              <span style={{ fontSize: 10, color: "var(--color-text-muted)", fontStyle: "italic" }}>
                <Sparkles size={10} style={{ display: "inline", marginRight: 4 }} />
                AI Forecast — not real-time sensor data
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={timeline}
              onChange={(e) => setTimeline(Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--color-primary)" }}
            />
          </div>
        </section>

        {/* RIGHT: AI Action Brief */}
        <aside style={{ background: "var(--color-surface)", borderLeft: "1px solid var(--color-border)", display: "flex", flexDirection: "column" }}>
          <ActionBrief event={selected} />
        </aside>
      </div>
    </AppShell>
  );
}

function EventCard({ event, active, onClick }: { event: CityEvent; active: boolean; onClick: () => void }) {
  const meta = causeMeta(event.cause);
  return (
    <button
      onClick={onClick}
      style={{
        textAlign: "left",
        background: active ? "var(--color-primary-light)" : "var(--color-surface)",
        border: active ? "2px solid var(--color-primary)" : "1px solid var(--color-border)",
        borderLeft: `3px solid ${event.priority === "High" ? "var(--color-critical)" : "var(--color-border)"}`,
        borderRadius: 10,
        padding: "12px 14px",
        cursor: "pointer",
        transition: "all 0.2s",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--color-text-primary)", lineHeight: 1.3 }}>
          {meta.label}
        </div>
        <Badge kind={event.priority === "High" ? "high" : "low"}>{event.priority}</Badge>
      </div>
      <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 4, display: "flex", gap: 4, alignItems: "center" }}>
        <MapPin size={10} /> {event.corridor} · {event.junction}
      </div>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {event.closure && <Badge kind="closure">Closure</Badge>}
        <Badge kind="ai" icon={<Sparkles size={9} />}>
          Est. {event.durationEstHrs} hrs left
        </Badge>
        <Badge kind="ai">{event.confidence}% conf</Badge>
      </div>
      <div style={{ fontSize: 10, color: "var(--color-text-muted)", marginTop: 6, display: "flex", alignItems: "center", gap: 4 }}>
        <Clock size={9} /> Started {event.startedMinAgo} min ago · {event.id}
      </div>
    </button>
  );
}

function ActionBrief({ event }: { event: CityEvent }) {
  const meta = causeMeta(event.cause);
  const deadlineMin = Math.max(0, 30 - event.startedMinAgo);
  return (
    <>
      <PanelHeader title="AI Action Brief" accent right={<Badge kind="ai">{event.id}</Badge>} />
      <div style={{ padding: 16, overflow: "auto", display: "flex", flexDirection: "column", gap: 14 }}>
        <Section title="Observed Data">
          <Row label="Event Type & Cause" value={`${event.type} · ${meta.label}`} />
          <Row label="Corridor / Junction" value={`${event.corridor} · ${event.junction}`} />
          <Row label="Zone" value={event.zone} muted prefix="~" />
          <Row label="Priority" value={<Badge kind={event.priority === "High" ? "high" : "low"}>{event.priority}</Badge>} />
          <Row label="Road Closure Required" value={<Badge kind={event.closure ? "closure" : "neutral"}>{event.closure ? "Yes" : "No"}</Badge>} />
          <Row label="Police Station" value="HSR Layout PS" />
        </Section>

        <Divider label="AI Generated Below" />

        <Section title="AI Model Output" accent>
          <Row label="Predicted Duration" value={`${event.durationEstHrs} hrs`} aiAccent />
          <Row label="Predicted Impact Score" value={`${Math.round(event.durationEstHrs * 6 + (event.priority === "High" ? 30 : 10))} / 100`} aiAccent />
          <Row label="Recommended Officers" value={event.recOfficers} aiAccent />
          <Row label="Recommended Barricades" value={event.recBarricades} aiAccent />
          <Row label="Recommended Diversion" value={`${event.corridor} → parallel route`} aiAccent />
          <Row label="Expected Resolution" value={`≈ ${event.durationEstHrs} hrs`} aiAccent />
          <div style={{ marginTop: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
              <span style={{ color: "var(--color-text-secondary)" }}>Confidence Score</span>
              <span style={{ color: "var(--color-ai-accent)", fontWeight: 600 }}>{event.confidence}%</span>
            </div>
            <ProgressBar value={event.confidence} color="var(--color-ai-accent)" />
          </div>
          <Row
            label="Deployment Deadline"
            value={
              <Badge kind={deadlineMin < 30 ? "warning" : "neutral"}>
                {deadlineMin} min remaining
              </Badge>
            }
          />
        </Section>
      </div>
    </>
  );
}

function Section({ title, children, accent }: { title: string; children: React.ReactNode; accent?: boolean }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
        {accent && <Sparkles size={12} style={{ color: "var(--color-ai-accent)" }} />}
        <span style={{ fontSize: 11, fontWeight: 700, color: accent ? "var(--color-ai-accent)" : "var(--color-text-primary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {title}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>{children}</div>
    </div>
  );
}

function Row({ label, value, aiAccent, muted, prefix }: { label: string; value: React.ReactNode; aiAccent?: boolean; muted?: boolean; prefix?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, fontSize: 12 }}>
      <span style={{ color: "var(--color-text-secondary)" }}>{label}</span>
      <span
        style={{
          color: aiAccent ? "var(--color-ai-accent)" : muted ? "var(--color-text-muted)" : "var(--color-text-primary)",
          fontWeight: 600,
          textAlign: "right",
        }}
      >
        {prefix}{value}
      </span>
    </div>
  );
}
