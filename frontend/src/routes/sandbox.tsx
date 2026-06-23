import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { FlaskConical, Play, Sparkles, Cloud, Construction, AlertTriangle, Download } from "lucide-react";
import { AppShell } from "@/components/cityos/AppShell";
import { Card, PanelHeader, Badge, Button, Field, Select, ProgressBar, Gauge, PageHeader } from "@/components/cityos/primitives";
import { CityMap } from "@/components/cityos/CityMap";
import { CORRIDORS, ZONES, causeMeta, ACTIVE_EVENTS, JUNCTIONS, type CityEvent } from "@/lib/cityos-data";
import { useClosurePrediction, useDurationPrediction } from "../hooks/use-predict";
import { useQuery } from "@tanstack/react-query";
import { getDashboardStream } from "@/lib/api";

export const Route = createFileRoute("/sandbox")({
  head: () => ({
    meta: [
      { title: "Scenario Sandbox · CityOS" },
      { name: "description", content: "What-if scenario builder — explore congestion outcomes by tweaking crowd, weather, closure and time-of-day." },
    ],
  }),
  component: Sandbox,
});

function Sandbox() {
  // Pre-loaded: Public Event on Mysore Road, 6 PM, closure required
  const [crowd, setCrowd] = useState(15000);
  const [hour, setHour] = useState(18);
  const [closure, setClosure] = useState(true);
  const [rain, setRain] = useState(false);
  const [construction, setConstruction] = useState(false);
  const [accident, setAccident] = useState(false);
  const [extra, setExtra] = useState(0);
  const [corridor, setCorridor] = useState("Mysore Road");
  const [zone, setZone] = useState(ZONES[0]);
  const [liveEventId, setLiveEventId] = useState<string>("");

  const { data: streamData } = useQuery({
    queryKey: ["dashboardStream"],
    queryFn: getDashboardStream,
    refetchInterval: 5000,
  });

  const activeEvents = streamData?.events || [];

  const handleLoadLiveEvent = (id: string) => {
    setLiveEventId(id);
    const evt = activeEvents.find((e: CityEvent) => e.id === id);
    if (evt) {
      setCorridor(evt.corridor);
      setZone(evt.zone || ZONES[0]);
      setClosure(evt.closure || false);
      
      // Reset special conditions
      setRain(false);
      setConstruction(false);
      setAccident(false);
      
      // Toggle condition based on cause
      if (evt.cause === "water_logging") setRain(true);
      if (evt.cause === "construction") setConstruction(true);
      if (evt.cause === "accident") setAccident(true);
      if (evt.cause === "public_event" || evt.cause === "protest" || evt.cause === "procession") {
        setCrowd(25000);
      } else {
        setCrowd(1000);
      }
      
      setHour(new Date().getHours());
    }
  };

  const predictClosure = useClosurePrediction();
  const predictDuration = useDurationPrediction();

  const isPending = predictClosure.isPending || predictDuration.isPending;

  const handleRunScenario = () => {
    const cause = rain ? "water_logging" : construction ? "construction" : accident ? "accident" : "public_event";
    const now = new Date();
    now.setHours(hour, 0, 0, 0);
    const reported_datetime = now.toISOString();

    const payload = {
      event_cause: cause,
      corridor,
      priority: crowd > 20000 ? "High" : "Low",
      reported_datetime,
      description: `${cause.replace("_", " ")} on ${corridor} with crowd size ${crowd}.`,
      comment: `Simulation run at hour ${hour}. rain=${rain}, construction=${construction}, accident=${accident}.`,
      vehicle_type: accident ? "heavy_vehicle" : null,
      junction: null,
      zone,
    };

    predictClosure.mutate(payload);
    predictDuration.mutate(payload);
  };

  const peak = (hour >= 5 && hour <= 6) || (hour >= 19 && hour <= 21);
  const baseCause = rain ? "water_logging" : construction ? "construction" : accident ? "accident" : "public_event";
  const meta = causeMeta(baseCause);
  const duration = Math.round((meta.avgHrs + extra) * 10) / 10;
  const risk = Math.min(100, Math.round(35 + (crowd / 1000) + (peak ? 18 : 0) + (closure ? 18 : 0) + (rain ? 14 : 0) + (construction ? 10 : 0)));

  const hasResult = predictClosure.data !== undefined && predictDuration.data !== undefined;

  // Determine actual prediction-derived metrics
  const displayClosureRequired = hasResult ? predictClosure.data!.closure_required : closure;
  const displayProbability = hasResult ? predictClosure.data!.probability : (meta.closurePct / 100);
  const displayDuration = hasResult ? predictDuration.data!.estimated_duration_hrs : duration;
  const displayRegime = hasResult ? predictDuration.data!.regime : (accident || rain || construction ? "chronic" : "acute");
  
  const displayRisk = hasResult
    ? Math.round(displayProbability * 100)
    : risk;

  const officers = Math.round(8 + crowd / 1500 + (displayClosureRequired ? 4 : 0));
  const barricades = Math.round(4 + (displayClosureRequired ? 4 : 0) + (rain ? 2 : 0));
  const priority = displayRisk > 60 ? "High" : "Low";

  const displayModelMode = hasResult
    ? (predictClosure.data!.model_mode.includes("fallback") ? "Fallback Simulation" : `Live Models (M1 LGBM + M2 ${displayRegime === "acute" ? "CatBoost" : "GBST"})`)
    : null;

  const shimmer = <div className="shimmer-active" style={{ width: 60, height: 20, borderRadius: 4 }} />;

  const junctionData = JUNCTIONS.find(j => j.name === ACTIVE_EVENTS.find(e => e.corridor === corridor)?.junction) || JUNCTIONS[0];
  const dynamicEvent: CityEvent = {
    id: "sandbox_preview",
    cause: baseCause,
    type: "unplanned",
    corridor: corridor,
    junction: junctionData.name,
    zone: zone,
    priority: priority,
    closure: displayClosureRequired,
    status: "active",
    startedMinAgo: 0,
    durationEstHrs: displayDuration,
    confidence: displayRisk,
    recOfficers: officers,
    recBarricades: barricades,
    lat: junctionData.lat,
    lng: junctionData.lng,
  };

  return (
    <AppShell>
      <style>{`
        @keyframes shimmer-eff {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        @keyframes spin-eff {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .shimmer-active {
          background: linear-gradient(90deg, var(--color-surface-elevated) 25%, var(--color-border) 50%, var(--color-surface-elevated) 75%);
          background-size: 200% 100%;
          animation: shimmer-eff 1.5s infinite linear;
        }
        .spinner-active {
          width: 110px;
          height: 110px;
          border: 6px solid var(--color-surface-elevated);
          border-top: 6px solid var(--color-ai-accent);
          border-radius: 50%;
          animation: spin-eff 1s infinite linear;
          margin: 11px;
        }
      `}</style>

      <PageHeader
        title="Scenario Sandbox"
        subtitle="What-if scenario builder. Try: Public Event on Mysore Road at 6 PM with closure — then toggle Add Rain to watch duration jump to 14.1 hrs."
        right={<Badge kind="ai" icon={<FlaskConical size={11} />}>Pre-loaded scenario</Badge>}
      />

      <div style={{ display: "grid", gridTemplateColumns: "340px 1fr 360px", gap: 16 }}>
        {/* LEFT: Controls */}
        <Card padded={false}>
          <PanelHeader title="What-If Scenario Builder" right={<FlaskConical size={14} style={{ color: "var(--color-ai-accent)" }} />} />
          
          {/* Live Event Load Section */}
          {activeEvents.length > 0 && (
            <div style={{ padding: "12px 18px", borderBottom: "1px solid var(--color-border)", background: "var(--color-bg)", display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: "var(--color-text-secondary)" }}>Load Live Event into Sandbox:</span>
                <Badge kind="ai" icon={<Download size={11} />}>Auto-fill</Badge>
              </div>
              <Select 
                value={liveEventId} 
                onChange={handleLoadLiveEvent} 
                options={[{ value: "", label: "Select an active incident..." }, ...activeEvents.map((e: CityEvent) => ({ value: e.id, label: `${e.cause.replace(/_/g, " ")} on ${e.corridor}` }))]} 
              />
            </div>
          )}

          <div style={{ padding: 18 }}>
            <Field label={`Crowd Size: ${crowd.toLocaleString()}`}>
              <input type="range" min={100} max={100000} step={100} value={crowd} onChange={(e) => setCrowd(Number(e.target.value))} style={{ width: "100%", accentColor: "var(--color-primary)" }} />
            </Field>
            <Field label={`Time of Day: ${String(hour).padStart(2, "0")}:00 ${peak ? "(peak)" : ""}`}>
              <input type="range" min={0} max={23} value={hour} onChange={(e) => setHour(Number(e.target.value))} style={{ width: "100%", accentColor: peak ? "var(--color-warning)" : "var(--color-primary)" }} />
              {peak && <div style={{ marginTop: 6 }}><Badge kind="warning">Peak: 5–6 AM · 7–9 PM</Badge></div>}
            </Field>

            <Field label="Road Closure">
              <button onClick={() => setClosure((v) => !v)} style={{
                width: 52, height: 28, borderRadius: 99,
                background: closure ? "var(--color-critical)" : "var(--color-border)",
                position: "relative", border: "none", cursor: "pointer",
              }}>
                <span style={{ position: "absolute", top: 3, left: closure ? 27 : 3, width: 22, height: 22, borderRadius: 99, background: "#fff", transition: "left 0.2s" }} />
              </button>
            </Field>

            <Field label="Add Conditions">
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <Checkbox checked={rain} onChange={() => setRain((v) => !v)} icon={<Cloud size={14} />} label="Add Rain / Flooding (water_logging — 14.1 hrs avg)" />
                <Checkbox checked={construction} onChange={() => setConstruction((v) => !v)} icon={<Construction size={14} />} label="Add Construction (13.3 hrs avg, 27% closure)" />
                <Checkbox checked={accident} onChange={() => setAccident((v) => !v)} icon={<AlertTriangle size={14} />} label="Add Accident (0.8 hrs, high urgency)" />
              </div>
            </Field>

            <Field label={`Extend Duration: ${extra >= 0 ? "+" : ""}${extra} hrs`}>
              <div style={{ display: "flex", gap: 6 }}>
                <Button variant="secondary" onClick={() => setExtra((v) => v - 1)}>−</Button>
                <Button variant="secondary" onClick={() => setExtra(0)}>Reset</Button>
                <Button variant="secondary" onClick={() => setExtra((v) => v + 1)}>+</Button>
              </div>
            </Field>

            <Field label="Corridor">
              <Select value={corridor} onChange={setCorridor} options={CORRIDORS.map((c) => ({ value: c.name, label: c.name }))} />
            </Field>
            <Field label="Zone">
              <Select value={zone} onChange={setZone} options={ZONES.map((z) => ({ value: z, label: z }))} />
            </Field>

            <Button full onClick={handleRunScenario} disabled={isPending}>
              {isPending ? (
                <>⏳ Running Predictions...</>
              ) : (
                <>
                  <Play size={14} /> Run Scenario
                </>
              )}
            </Button>
          </div>
        </Card>

        {/* CENTER: Preview map */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card padded={false} style={{ flex: 1, minHeight: 400, display: "flex", flexDirection: "column" }}>
            <PanelHeader title="Scenario Preview" right={<Badge kind={priority === "High" ? "high" : "low"}>{priority} Priority</Badge>} />
            <div style={{ flex: 1, padding: 10 }}>
              <CityMap events={[dynamicEvent]} showDiversion={displayClosureRequired} showHeatmap />
            </div>
          </Card>
        </div>

        {/* RIGHT: Output */}
        <Card padded={false} style={{ borderLeft: "3px solid var(--color-ai-accent)" }}>
          <PanelHeader title="Real-time Model Output" accent />
          <div style={{ padding: 18, display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              {isPending ? (
                <div className="spinner-active" />
              ) : (
                <Gauge
                  value={displayRisk}
                  size={132}
                  color={displayRisk > 70 ? "var(--color-critical)" : displayRisk > 45 ? "var(--color-warning)" : "var(--color-primary)"}
                  label="Risk Score"
                />
              )}
              <div style={{ fontSize: 11, color: "var(--color-text-muted)" }}>Congestion risk · 0–100</div>
            </div>

            <OutRow label="Officers Needed" value={isPending ? shimmer : officers} />
            <OutRow label="Barricades Needed" value={isPending ? shimmer : barricades} />
            <OutRow label="Predicted Duration" value={isPending ? shimmer : `${Math.round((displayDuration + extra) * 10) / 10} hrs`} />
            <OutRow label="Recommended Diversion" value={isPending ? shimmer : (displayClosureRequired ? <Badge kind="success">{corridor} → parallel</Badge> : <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>Not Needed</span>)} />

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                <span style={{ color: "var(--color-text-secondary)" }}>Road Closure Probability</span>
                <span style={{ color: "var(--color-critical)", fontWeight: 600 }}>
                  {isPending ? "Calculating..." : `${Math.round(displayProbability * 100)}%`}
                </span>
              </div>
              {isPending ? (
                <div className="shimmer-active" style={{ width: "100%", height: 6, borderRadius: 99 }} />
              ) : (
                <ProgressBar value={displayProbability * 100} color="var(--color-critical)" />
              )}
              <div style={{ fontSize: 10, color: "var(--color-text-muted)", marginTop: 4 }}>
                {displayModelMode ? displayModelMode : `Historical reference for ${meta.label.toLowerCase()}`}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

function Checkbox({ checked, onChange, icon, label }: { checked: boolean; onChange: () => void; icon: React.ReactNode; label: string }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", borderRadius: 8, background: checked ? "var(--color-primary-light)" : "var(--color-surface)", border: `1px solid ${checked ? "var(--color-primary)" : "var(--color-border)"}`, cursor: "pointer", fontSize: 12, color: "var(--color-text-primary)" }}>
      <input type="checkbox" checked={checked} onChange={onChange} style={{ accentColor: "var(--color-primary)" }} />
      <span style={{ color: "var(--color-primary)" }}>{icon}</span>
      {label}
    </label>
  );
}

function OutRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 12px", background: "var(--color-bg)", borderRadius: 8 }}>
      <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>{label}</span>
      <span style={{ fontSize: 18, fontWeight: 700, color: "var(--color-ai-accent)" }}>{value}</span>
    </div>
  );
}
