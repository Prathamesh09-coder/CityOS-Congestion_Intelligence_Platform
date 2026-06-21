import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { ArrowRight, CheckCircle2, AlertTriangle, Sparkles, Users, Route as RouteIcon, ShieldCheck, RotateCw } from "lucide-react";
import { AppShell } from "@/components/cityos/AppShell";
import { Card, Badge, ProgressBar, Gauge, Button } from "@/components/cityos/primitives";
import { CityMap } from "@/components/cityos/CityMap";
import { ACTIVE_EVENTS } from "@/lib/cityos-data";
import { useClosurePrediction, useDurationPrediction, useMultimodalPrediction, useTrafficPrediction } from "../hooks/use-predict";

export const Route = createFileRoute("/demo")({
  head: () => ({
    meta: [
      { title: "Executive Demo · CityOS" },
      { name: "description", content: "Single-page animated storytelling demo of CityOS responding to a real-world VIP Movement event on Mysore Road." },
    ],
  }),
  component: Demo,
});

function Demo() {
  const [step, setStep] = useState(0);

  const predictClosure = useClosurePrediction();
  const predictDuration = useDurationPrediction();
  const predictMultimodal = useMultimodalPrediction();

  // Run live models for the Storyteller VIP Movement event on mount
  useEffect(() => {
    const payload = {
      event_cause: "vip_movement",
      corridor: "Mysore Road",
      priority: "High",
      reported_datetime: new Date().toISOString(),
      description: "VIP Convoy passing on Mysore Road corridor.",
      comment: "Executive story simulation run.",
      vehicle_type: null,
      junction: "K R Circle",
      zone: "Central Zone 1",
    };
    predictClosure.mutate(payload);
    predictDuration.mutate(payload);
    predictMultimodal.mutate({
      description: "VIP Convoy passing on Mysore Road corridor.",
      comment: "Executive story simulation run.",
      event_cause: "vip_movement",
      corridor: "Mysore Road",
    });
  }, []);

  // Query M4 Graph WaveNet traffic forecasting on K R Circle when step index is 3
  const trafficQuery = useTrafficPrediction(
    {
      lat: [12.9716],
      lng: [77.5946],
      reported_datetime: new Date().toISOString(),
    },
    step === 3
  );

  const isPending = predictClosure.isPending || predictDuration.isPending || predictMultimodal.isPending;
  const hasResult = predictClosure.data !== undefined && predictDuration.data !== undefined && predictMultimodal.data !== undefined;

  const displayProbability = hasResult ? predictClosure.data!.probability : 0.80;
  const displayDuration = hasResult ? predictDuration.data!.estimated_duration_hrs : 1.2;
  const isClosureRequired = hasResult ? predictClosure.data!.closure_required : true;
  const confidence = hasResult ? predictMultimodal.data!.prediction_confidence : 88;
  const actualDuration = Number((displayDuration * 0.75).toFixed(1));
  
  const officers = Math.round(8 + (isClosureRequired ? 4 : 0));
  const barricades = Math.round(4 + (isClosureRequired ? 4 : 0));

  const STEPS = [
    { title: "Event Appears", body: "VIP Movement pin drops on Mysore Road.", icon: AlertTriangle },
    { title: "AI Predicts Impact", body: `Road closure probability ${Math.round(displayProbability * 100)}%, predicted duration ${displayDuration} hrs.`, icon: Sparkles },
    { title: "AI Generates Resource Plan", body: `${officers} Officers · ${barricades} Barricades · 1 Tow Vehicle.`, icon: Users },
    { title: "AI Creates Diversion Plan", body: "Traffic redistributed to alternate corridor.", icon: RouteIcon },
    { title: "Authorities Deploy", body: "Coverage score: 0% → 84%.", icon: ShieldCheck },
    { title: "Impact Contained", body: `Closed in ${actualDuration} hrs — under predicted ${displayDuration} hrs.`, icon: CheckCircle2 },
    { title: "System Learns Outcome", body: "Model accuracy updated. Lessons stored.", icon: RotateCw },
  ];

  const current = STEPS[step];
  
  let modelMode = "Calculating...";
  if (hasResult) {
    const isM1Fallback = predictClosure.data!.model_mode.includes("fallback");
    const isM3Fallback = predictMultimodal.data!.model_mode.includes("fallback");
    if (isM1Fallback || isM3Fallback) {
      modelMode = "Simulation mode";
    } else {
      modelMode = "Live inference active";
    }
  }

  return (
    <AppShell>
      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "20px 0" }}>
        {/* Progress bar */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              Executive Demo · VIP Movement on Mysore Road ({modelMode})
            </span>
            <span style={{ fontSize: 12, color: "var(--color-text-muted)" }}>Step {step + 1} of {STEPS.length}</span>
          </div>
          <ProgressBar value={((step + 1) / STEPS.length) * 100} color="var(--color-primary)" height={4} />
        </div>

        {/* Step pills */}
        <div style={{ display: "flex", gap: 6, marginBottom: 24, flexWrap: "wrap" }}>
          {STEPS.map((s, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              style={{
                display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 12px", borderRadius: 99, cursor: "pointer",
                fontSize: 11, fontWeight: 600,
                background: i === step ? "var(--color-primary)" : i < step ? "var(--color-success-light)" : "var(--color-surface)",
                color: i === step ? "#fff" : i < step ? "var(--color-success)" : "var(--color-text-secondary)",
                border: `1px solid ${i === step ? "var(--color-primary)" : "var(--color-border)"}`,
              }}
            >
              <span style={{ width: 18, height: 18, borderRadius: 99, background: i === step ? "rgba(255,255,255,0.25)" : i < step ? "var(--color-success)" : "var(--color-surface-elevated)", color: i === step ? "#fff" : i < step ? "#fff" : "var(--color-text-secondary)", display: "grid", placeItems: "center", fontSize: 10 }}>
                {i + 1}
              </span>
              {s.title}
            </button>
          ))}
        </div>

        {/* Active step card */}
        <Card key={step} className="slide-up" style={{ borderColor: "var(--color-primary)", borderWidth: 2, boxShadow: "0 4px 24px rgba(40,116,240,0.15)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
            <div style={{ width: 48, height: 48, borderRadius: 12, background: "var(--color-primary-light)", color: "var(--color-primary)", display: "grid", placeItems: "center" }}>
              <current.icon size={24} />
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--color-primary)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                Step {step + 1}
              </div>
              <h2 style={{ fontSize: 22, fontWeight: 700, color: "var(--color-text-primary)", margin: 0 }}>{current.title}</h2>
              <div style={{ fontSize: 14, color: "var(--color-text-secondary)", marginTop: 4 }}>{current.body}</div>
            </div>
          </div>

          <StepContent
            step={step}
            probability={displayProbability}
            duration={displayDuration}
            actualDuration={actualDuration}
            officers={officers}
            barricades={barricades}
            confidence={confidence}
            isClosureRequired={isClosureRequired}
            isPending={isPending}
            multimodalData={predictMultimodal.data}
            trafficData={trafficQuery.data}
            trafficPending={trafficQuery.isLoading}
          />

          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 20 }}>
            <Button variant="secondary" onClick={() => setStep((s) => Math.max(0, s - 1))}>Previous</Button>
            <Button onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}>
              {step === STEPS.length - 1 ? "Restart" : "Next"} <ArrowRight size={14} />
            </Button>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}

interface ContentProps {
  step: number;
  probability: number;
  duration: number;
  actualDuration: number;
  officers: number;
  barricades: number;
  confidence: number;
  isClosureRequired: boolean;
  isPending: boolean;
  multimodalData?: any;
  trafficData?: any;
  trafficPending?: boolean;
}

function StepContent({
  step,
  probability,
  duration,
  actualDuration,
  officers,
  barricades,
  confidence,
  isClosureRequired,
  isPending,
  multimodalData,
  trafficData,
  trafficPending,
}: ContentProps) {
  const vip = ACTIVE_EVENTS[0]; // VIP Movement event

  const shimmer = <div className="shimmer-active" style={{ height: 64, borderRadius: 8 }} />;

  if (step === 0) {
    return (
      <>
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
        <div style={{ background: "var(--color-critical-light)", color: "var(--color-critical)", padding: "12px 16px", borderRadius: 10, fontWeight: 600, fontSize: 13, marginBottom: 12 }}>
          ⚠ New High-Priority Event Detected · {vip.corridor} · {vip.junction}
        </div>
        <div style={{ height: 380 }}>
          <CityMap events={[vip]} />
        </div>
      </>
    );
  }
  if (step === 1) {
    return (
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
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
        {isPending ? (
          <>
            {shimmer}
            {shimmer}
            {shimmer}
            {shimmer}
            {shimmer}
            {shimmer}
          </>
        ) : (
          <>
            <Stat label="Road Closure Probability" value={`${Math.round(probability * 100)}%`} color="var(--color-critical)">
              <ProgressBar value={probability * 100} color="var(--color-critical)" />
            </Stat>
            <Stat label="Predicted Duration" value={`${duration} hrs`} color="var(--color-ai-accent)" />
            <Stat label="Priority Prediction" value={<Badge kind={probability > 0.5 ? "high" : "low"}>{probability > 0.5 ? "High" : "Low"}</Badge>} />
            <Stat label="AI Confidence" value={`${confidence}%`} color="var(--color-ai-accent)">
              <ProgressBar value={confidence} color="var(--color-ai-accent)" />
            </Stat>
            <Stat label="M3 Zero-Shot Risk Score" value={`${multimodalData?.zero_shot_risk_score ?? 82}%`} color="var(--color-warning)">
              <ProgressBar value={multimodalData?.zero_shot_risk_score ?? 82} color="var(--color-warning)" />
            </Stat>
            <Stat label="M3 Inferred Cause" value={<Badge kind="ai">{multimodalData?.cause_inferred ?? "vip_movement"}</Badge>} />
          </>
        )}
      </div>
    );
  }
  if (step === 2) {
    return (
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
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
        {isPending ? (
          <>
            {shimmer}
            {shimmer}
            {shimmer}
          </>
        ) : (
          <>
            {[{ l: "Officers", v: officers }, { l: "Barricades", v: barricades }, { l: "Tow Vehicles", v: 1 }].map((r) => (
              <div key={r.l} className="cityos-card" style={{ padding: 16, borderLeft: "3px solid var(--color-ai-accent)" }}>
                <div style={{ fontSize: 11, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{r.l}</div>
                <div style={{ fontSize: 30, fontWeight: 700, color: "var(--color-ai-accent)" }}>{r.v}</div>
                <Badge kind="ai" icon={<Sparkles size={9} />}>AI Recommended</Badge>
              </div>
            ))}
            <div style={{ gridColumn: "span 3", padding: "10px 14px", background: "var(--color-critical-light)", color: "var(--color-critical)", borderRadius: 8, fontSize: 13, fontWeight: 600 }}>
              Current allocation: 0 — Deploy now.
            </div>
          </>
        )}
      </div>
    );
  }
  if (step === 3) {
    const dynamicEvent = { ...vip, closure: isClosureRequired };
    const delayVal = trafficData ? `${trafficData.metrics.average_delay_minutes} min` : "18 min";
    const speedVal = trafficData ? `${trafficData.metrics.predicted_speed_kmh} km/h` : "24.5 km/h";
    const flowVal = trafficData ? `${trafficData.metrics.predicted_flow_veh_hr} v/h` : "1050 v/h";
    const congestionIndex = trafficData ? trafficData.metrics.congestion_index : 0.65;

    return (
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <Card padded={false}>
          <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--color-border)", fontSize: 12, fontWeight: 600 }}>Before — Mysore Road blocked</div>
          <div style={{ height: 260, padding: 8 }}><CityMap events={[dynamicEvent]} /></div>
        </Card>
        <Card padded={false}>
          <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--color-border)", fontSize: 12, fontWeight: 600 }}>After — AI diversion</div>
          <div style={{ height: 260, padding: 8 }}><CityMap events={[dynamicEvent]} showDiversion /></div>
        </Card>
        
        {/* M4 Spatio-Temporal Graph WaveNet Telemetry Card */}
        <div style={{ gridColumn: "span 2", padding: 12, background: "var(--color-surface-elevated)", borderRadius: 8, border: "1px solid var(--color-border)" }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--color-ai-accent)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8, display: "flex", alignItems: "center", gap: 4 }}>
            <Sparkles size={12} style={{ color: "var(--color-ai-accent)" }} />
            <span>M4 Graph WaveNet Live Traffic Telemetry</span>
          </div>
          {trafficPending ? (
            <div className="shimmer-active" style={{ height: 48, borderRadius: 6 }} />
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
              <div>
                <div style={{ fontSize: 9, color: "var(--color-text-muted)" }}>PREDICTED SPEED</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "var(--color-primary)" }}>{speedVal}</div>
              </div>
              <div>
                <div style={{ fontSize: 9, color: "var(--color-text-muted)" }}>PREDICTED FLOW</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "var(--color-primary)" }}>{flowVal}</div>
              </div>
              <div>
                <div style={{ fontSize: 9, color: "var(--color-text-muted)" }}>CONGESTION INDEX</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "var(--color-critical)" }}>{congestionIndex}</div>
              </div>
              <div>
                <div style={{ fontSize: 9, color: "var(--color-text-muted)" }}>AVERAGE DELAY</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: "var(--color-warning)" }}>{delayVal}</div>
              </div>
            </div>
          )}
        </div>

        <div style={{ gridColumn: "span 2", textAlign: "center", color: "var(--color-success)", fontWeight: 600, padding: 12, background: "var(--color-success-light)", borderRadius: 8, fontSize: 13 }}>
          Traffic redistributed — estimated {delayVal} delay reduction.
        </div>
      </div>
    );
  }
  if (step === 4) {
    return (
      <div style={{ display: "flex", justifyContent: "center", padding: 30 }}>
        <Gauge value={84} color="var(--color-primary)" size={180} label="Coverage" />
      </div>
    );
  }
  if (step === 5) {
    return (
      <div style={{ background: "var(--color-success-light)", color: "var(--color-success)", padding: 20, borderRadius: 12, fontSize: 16, fontWeight: 600, textAlign: "center" }}>
        ✓ Event closed in <b>{actualDuration} hrs</b> · under predicted {duration} hrs<br />
        <span style={{ fontSize: 13, fontWeight: 400 }}>Prediction accurate. Response efficient.</span>
      </div>
    );
  }
  return (
    <div className="cityos-card" style={{ padding: 18, borderLeft: "3px solid var(--color-ai-accent)" }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: "var(--color-ai-accent)", textTransform: "uppercase", marginBottom: 8 }}>Post-Event Card</div>
      <div style={{ display: "flex", gap: 24, fontSize: 14 }}>
        <div><div style={{ color: "var(--color-text-muted)", fontSize: 11 }}>Predicted</div><div style={{ fontWeight: 700, color: "var(--color-ai-accent)" }}>{duration} hrs</div></div>
        <div><div style={{ color: "var(--color-text-muted)", fontSize: 11 }}>Actual</div><div style={{ fontWeight: 700, color: "var(--color-text-primary)" }}>{actualDuration} hrs</div></div>
        <div><div style={{ color: "var(--color-text-muted)", fontSize: 11 }}>Accuracy</div><div style={{ fontWeight: 700, color: "var(--color-success)" }}>+1.4%</div></div>
      </div>
      <div style={{ marginTop: 12, color: "var(--color-text-secondary)", fontSize: 13 }}>Model improved. Lessons stored in Post-Event Hub.</div>
    </div>
  );
}

function Stat({ label, value, color, children }: { label: string; value: React.ReactNode; color?: string; children?: React.ReactNode }) {
  return (
    <div className="cityos-card" style={{ padding: 16 }}>
      <div style={{ fontSize: 11, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: color ?? "var(--color-text-primary)", margin: "4px 0 8px" }}>{value}</div>
      {children}
    </div>
  );
}

