import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Users, Sparkles, Truck, Construction, Siren, ShieldAlert, MapPin, CheckCircle2 } from "lucide-react";
import { AppShell } from "@/components/cityos/AppShell";
import { Card, PanelHeader, Badge, Gauge, ProgressBar, PageHeader, Button } from "@/components/cityos/primitives";
import { CityMap } from "@/components/cityos/CityMap";
import { ACTIVE_EVENTS as DEFAULT_EVENTS, type CityEvent } from "@/lib/cityos-data";
import { useQuery, useMutation } from "@tanstack/react-query";
import { predictResources, deployResources, getDashboardStream } from "@/lib/api";

export const Route = createFileRoute("/resources")({
  head: () => ({
    meta: [
      { title: "AI Resource Command Center · CityOS" },
      { name: "description", content: "AI-recommended deployment plan: officers, barricades, tow vehicles, emergency teams, with live coverage scoring." },
    ],
  }),
  component: ResourceCmd,
});

const RESOURCE_TYPES = [
  { key: "officers",    label: "Officers",         icon: Users,        color: "var(--color-primary)" },
  { key: "barricades",  label: "Barricades",       icon: ShieldAlert,  color: "var(--color-critical)" },
  { key: "tow",         label: "Tow Vehicles",     icon: Truck,        color: "var(--color-warning)" },
  { key: "marshals",    label: "Traffic Marshals", icon: Users,        color: "var(--color-success)" },
  { key: "emergency",   label: "Emergency Teams",  icon: Siren,        color: "var(--color-critical)" },
  { key: "closureCrew", label: "Road Closure Crew",icon: Construction, color: "var(--color-critical)" },
];

function ResourceCmd() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [allocated, setAllocated] = useState({ officers: 0, barricades: 0, tow: 0, marshals: 0, emergency: 0, closureCrew: 0 });

  const { data: streamData } = useQuery({
    queryKey: ["dashboardStream"],
    queryFn: getDashboardStream,
    refetchInterval: 5000,
  });

  const activeEvents = streamData?.events?.length > 0 ? streamData.events : DEFAULT_EVENTS;

  useEffect(() => {
    if (!selectedId && activeEvents.length > 0) {
      setSelectedId(activeEvents[0].id);
    } else if (selectedId && !activeEvents.find((e: CityEvent) => e.id === selectedId) && activeEvents.length > 0) {
      setSelectedId(activeEvents[0].id);
    }
  }, [activeEvents, selectedId]);

  const selected = activeEvents.find((e: CityEvent) => e.id === selectedId) || activeEvents[0] || DEFAULT_EVENTS[0];

  const { data: recs, isLoading } = useQuery({
    queryKey: ["resources", selected.id],
    queryFn: () => predictResources({
      event_id: selected.id,
      event_cause: selected.cause,
      corridor: selected.corridor,
      priority: selected.priority,
      type: selected.type,
      closure: selected.closure || false
    }),
  });

  const targetOfficers = recs?.recOfficers ?? selected.recOfficers;
  const targetBarricades = recs?.recBarricades ?? (selected.recBarricades || 1);
  const targetTow = recs?.tow ?? (selected.cause === "vehicle_breakdown" ? 1 : 0);
  const targetMarshals = recs?.marshals ?? (selected.type === "planned" ? 4 : 0);
  const targetEmergency = recs?.emergency ?? (selected.cause === "accident" ? 1 : 0);
  const targetClosureCrew = recs?.closureCrew ?? (selected.closure ? 1 : 0);

  const coverage = Math.min(100, Math.round(((allocated.officers / Math.max(1, targetOfficers)) * 60) + ((allocated.barricades / Math.max(1, targetBarricades)) * 40)));
  const efficiency = Math.max(0, 100 - Math.abs(coverage - 100));
  const gap = targetOfficers - allocated.officers;

  const deployMut = useMutation({
    mutationFn: deployResources,
    onSuccess: () => {
      setAllocated({ officers: 0, barricades: 0, tow: 0, marshals: 0, emergency: 0, closureCrew: 0 });
    }
  });

  const handleDeploy = () => {
    deployMut.mutate({
      event_id: selected.id,
      ...allocated
    });
  };

  return (
    <AppShell>
      <PageHeader
        title="AI Resource Command Center"
        subtitle="All resource figures are AI-generated. Drag-and-drop to plan deployment for active events."
        right={<Badge kind="ai" icon={<Sparkles size={11} />}>AI Recommendations</Badge>}
      />

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 360px", gap: 16, height: "calc(100vh - 240px)", minHeight: 600 }}>
        {/* LEFT: Active events list */}
        <Card padded={false} style={{ display: "flex", flexDirection: "column" }}>
          <PanelHeader title="Active Events" />
          <div style={{ overflow: "auto", padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
            {activeEvents.map((e: CityEvent) => (
              <button key={e.id} onClick={() => { setSelectedId(e.id); setAllocated({ officers: 0, barricades: 0, tow: 0, marshals: 0, emergency: 0, closureCrew: 0 }); deployMut.reset(); }} style={{
                textAlign: "left", padding: "10px 12px", borderRadius: 8, cursor: "pointer",
                background: selected.id === e.id ? "var(--color-primary-light)" : "var(--color-surface)",
                border: selected.id === e.id ? "2px solid var(--color-primary)" : "1px solid var(--color-border)",
                borderLeft: `3px solid ${e.priority === "High" ? "var(--color-critical)" : "var(--color-border)"}`,
              }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--color-text-primary)" }}>{(e.cause || "Unknown").replace(/_/g, " ")}</div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 2 }}>{e.corridor}</div>
              </button>
            ))}
          </div>
        </Card>

        {/* CENTER: Map */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
          <div style={{ flex: 1, minHeight: 0 }}>
            <CityMap events={activeEvents} selectedId={selected.id} onSelect={(e) => setSelectedId(e.id)} showHeatmap />
          </div>
          {/* Draggable resource chips */}
          <Card padded>
            <div style={{ fontSize: 11, fontWeight: 700, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>
              Deploy Resources to Selected Event
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {RESOURCE_TYPES.map(({ key, label, icon: Icon, color }) => (
                <button
                  key={key}
                  onClick={() => setAllocated((a) => ({ ...a, [key]: (a as any)[key] + 1 }))}
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 8,
                    padding: "8px 12px", borderRadius: 99,
                    background: "var(--color-surface)", border: `1px solid var(--color-border)`,
                    borderLeft: `3px solid ${color}`, cursor: "pointer",
                    fontSize: 12, color: "var(--color-text-primary)", fontWeight: 500,
                  }}
                >
                  <Icon size={14} style={{ color }} /> +1 {label}
                  <span style={{ color: "var(--color-text-muted)", fontSize: 11 }}>({(allocated as any)[key]})</span>
                </button>
              ))}
            </div>
          </Card>
        </div>

        {/* RIGHT: AI plan + scoring */}
        <Card padded={false} style={{ borderLeft: "3px solid var(--color-ai-accent)", display: "flex", flexDirection: "column" }}>
          <PanelHeader title="AI Deployment Plan" accent />
          <div style={{ padding: 16, overflow: "auto", display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              Event: <b style={{ color: "var(--color-text-primary)" }}>{(selected.cause || "Unknown").replace(/_/g, " ")}</b> on {selected.corridor}
            </div>

            <RecRow icon={Users} label="Officers Required" value={targetOfficers} color="var(--color-primary)" />
            <RecRow icon={ShieldAlert} label="Barricades" value={targetBarricades} color="var(--color-critical)" />
            <RecRow icon={Truck} label="Tow Vehicles" value={targetTow} color="var(--color-warning)" />
            <RecRow icon={Users} label="Traffic Marshals" value={targetMarshals} color="var(--color-success)" />
            <RecRow icon={Siren} label="Emergency Teams" value={targetEmergency} color="var(--color-critical)" />
            <RecRow icon={Construction} label="Closure Crew" value={targetClosureCrew} color="var(--color-critical)" />

            <div style={{ height: 1, background: "var(--color-border)", margin: "4px 0" }} />

            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>
                Live Deployment Score
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, alignItems: "center" }}>
                <Gauge value={coverage} color="var(--color-primary)" size={90} label="Coverage" />
                <div style={{ fontSize: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>Impact reduction</span>
                    <span style={{ color: "var(--color-success)", fontWeight: 600 }}>↑ {Math.round(coverage * 0.6)}%</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>Cost index</span>
                    <span style={{ color: "var(--color-text-primary)", fontWeight: 600 }}>{allocated.officers * 8 + allocated.barricades * 3}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>Efficiency</span>
                    <span style={{ color: "var(--color-primary)", fontWeight: 600 }}>{efficiency}%</span>
                  </div>
                  <ProgressBar value={efficiency} color="var(--color-primary)" />
                </div>
              </div>
              <div style={{ marginTop: 12, padding: "8px 12px", borderRadius: 8, background: gap > 0 ? "var(--color-critical-light)" : "var(--color-success-light)", color: gap > 0 ? "var(--color-critical)" : "var(--color-success)", fontSize: 12, fontWeight: 600 }}>
                {gap > 0 ? `Gap from AI: under-allocated by ${gap} officers` : `Aligned with AI recommendation ✓`}
              </div>
            </div>

            <div style={{ borderTop: "1px solid var(--color-border)", paddingTop: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
                Nearest Police Station
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
                <MapPin size={14} style={{ color: "var(--color-primary)" }} />
                <span style={{ color: "var(--color-text-primary)", fontWeight: 600 }}>HSR Layout PS</span>
                <span style={{ color: "var(--color-text-muted)", fontSize: 11 }}>· 2.4 km</span>
              </div>
            </div>
            
            <div style={{ marginTop: "auto", paddingTop: 16 }}>
              <Button style={{ width: "100%" }} onClick={handleDeploy} disabled={deployMut.isPending || deployMut.isSuccess || coverage === 0}>
                {deployMut.isPending ? "Deploying..." : deployMut.isSuccess ? <><CheckCircle2 size={16} /> Deployed & Resolved</> : "Deploy & Resolve Event"}
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

function RecRow({ icon: Icon, label, value, color }: { icon: any; label: string; value: number; color: string }) {
  return (
    <div className="cityos-card" style={{ padding: "10px 14px", borderLeft: `3px solid ${color}`, display: "flex", alignItems: "center", gap: 12 }}>
      <Icon size={18} style={{ color }} />
      <div style={{ flex: 1, fontSize: 12, color: "var(--color-text-secondary)" }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: "var(--color-ai-accent)" }}>{value}</div>
      <Badge kind="ai" icon={<Sparkles size={9} />}>AI</Badge>
    </div>
  );
}
