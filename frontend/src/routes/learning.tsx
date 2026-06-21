import { createFileRoute } from "@tanstack/react-router";
import { TrendingUp, Sparkles, CheckCircle2, AlertTriangle } from "lucide-react";
import { AppShell } from "@/components/cityos/AppShell";
import { Card, PanelHeader, Badge, ProgressBar, PageHeader } from "@/components/cityos/primitives";

export const Route = createFileRoute("/learning")({
  head: () => ({
    meta: [
      { title: "Post-Event Learning Hub · CityOS" },
      { name: "description", content: "Predicted vs actual: model accuracy, per-cause breakdown, monthly trend, and AI-generated lessons learned." },
    ],
  }),
  component: Learning,
});

const PREDICTION_VS_REALITY = [
  { metric: "Priority", predicted: "High", actual: "High", accuracy: 86 },
  { metric: "Duration", predicted: "1.2 hrs", actual: "0.9 hrs", accuracy: 78 },
  { metric: "Road Closure", predicted: "Required", actual: "Required", accuracy: 92 },
  { metric: "Resolution Status", predicted: "closed", actual: "closed", accuracy: 94 },
];

const TREND = [62, 65, 68, 70, 71, 74, 77, 79, 82, 84, 85, 87];
const MONTHS = ["Nov", "", "Dec", "", "Jan", "", "Feb", "", "Mar", "", "Apr", ""];

const PER_CAUSE = [
  { cause: "Vehicle Breakdown", accuracy: 91, warn: false },
  { cause: "Accident", accuracy: 84, warn: false },
  { cause: "Procession", accuracy: 81, warn: false },
  { cause: "VIP Movement", accuracy: 88, warn: false },
  { cause: "Public Event", accuracy: 73, warn: false },
  { cause: "Construction", accuracy: 68, warn: false },
  { cause: "Tree Fall", accuracy: 64, warn: false },
  { cause: "Water Logging", accuracy: 52, warn: true },
  { cause: "Pot Holes", accuracy: 48, warn: true },
];

function Learning() {
  return (
    <AppShell>
      <PageHeader
        title="Post-Event Learning Hub"
        subtitle="The model improves with every closed event. Real fields used: status, start_datetime, closed_datetime, requires_road_closure, priority."
        right={<Badge kind="success" icon={<TrendingUp size={11} />}>Model improving</Badge>}
      />

      {/* Prediction vs Reality */}
      <Card padded={false}>
        <PanelHeader title="Prediction vs. Reality" />
        <div style={{ overflow: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ background: "var(--color-surface-elevated)" }}>
                {["Metric", "Predicted", "Actual", "Accuracy"].map((h) => (
                  <th key={h} style={{ padding: "12px 16px", textAlign: "left", fontWeight: 600, fontSize: 11, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {PREDICTION_VS_REALITY.map((r, i) => (
                <tr key={r.metric} style={{ background: i % 2 === 0 ? "var(--color-surface)" : "var(--color-bg)", borderTop: "1px solid var(--color-border)" }}>
                  <td style={{ padding: "12px 16px", color: "var(--color-text-primary)", fontWeight: 600 }}>{r.metric}</td>
                  <td style={{ padding: "12px 16px", color: "var(--color-ai-accent)" }}>{r.predicted}</td>
                  <td style={{ padding: "12px 16px", color: "var(--color-text-primary)" }}>{r.actual}</td>
                  <td style={{ padding: "12px 16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ flex: 1, maxWidth: 180 }}>
                        <ProgressBar value={r.accuracy} color={r.accuracy >= 80 ? "var(--color-success)" : r.accuracy >= 60 ? "var(--color-warning)" : "var(--color-critical)"} />
                      </div>
                      <Badge kind={r.accuracy >= 80 ? "success" : r.accuracy >= 60 ? "warning" : "high"}>{r.accuracy}%</Badge>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Two-column: trend + per-cause */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
        <Card padded={false}>
          <PanelHeader title="Model Accuracy Trend" right={<Badge kind="success">Nov 2023 – Apr 2024</Badge>} />
          <div style={{ padding: 20 }}>
            <svg viewBox="0 0 400 180" style={{ width: "100%", height: 220 }}>
              {/* grid */}
              {[0, 1, 2, 3, 4].map((i) => (
                <line key={i} x1={0} y1={20 + i * 32} x2={400} y2={20 + i * 32} stroke="var(--color-surface-elevated)" strokeWidth={1} />
              ))}
              {/* line */}
              <polyline
                fill="none"
                stroke="var(--color-primary)"
                strokeWidth={2.5}
                points={TREND.map((v, i) => `${(i / (TREND.length - 1)) * 380 + 10},${160 - ((v - 50) / 50) * 140}`).join(" ")}
              />
              {/* dots */}
              {TREND.map((v, i) => (
                <circle key={i} cx={(i / (TREND.length - 1)) * 380 + 10} cy={160 - ((v - 50) / 50) * 140} r={3} fill="var(--color-primary)" />
              ))}
              {/* x labels */}
              {MONTHS.map((m, i) => m && (
                <text key={i} x={(i / (TREND.length - 1)) * 380 + 10} y={175} fontSize={9} fill="var(--color-text-muted)" textAnchor="middle">{m}</text>
              ))}
              {/* y labels */}
              {[50, 65, 80, 95].map((y, i) => (
                <text key={y} x={0} y={160 - ((y - 50) / 50) * 140} fontSize={9} fill="var(--color-text-muted)">{y}%</text>
              ))}
            </svg>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 8 }}>
              Accuracy improved from <b style={{ color: "var(--color-text-primary)" }}>62% → 87%</b> as the model ingested more closed events.
            </div>
          </div>
        </Card>

        <Card padded={false}>
          <PanelHeader title="Per-Cause Accuracy" />
          <div style={{ padding: 18, display: "flex", flexDirection: "column", gap: 8 }}>
            {PER_CAUSE.map((p) => (
              <div key={p.cause}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                  <span style={{ color: "var(--color-text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                    {p.cause}
                    {p.warn && <Badge kind="warning">High Variance</Badge>}
                  </span>
                  <span style={{ color: "var(--color-text-secondary)", fontWeight: 600 }}>{p.accuracy}%</span>
                </div>
                <ProgressBar value={p.accuracy} color={p.warn ? "var(--color-warning)" : "var(--color-primary)"} />
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Lessons Learned */}
      <Card padded={false} style={{ marginTop: 16, borderLeft: "3px solid var(--color-ai-accent)" }}>
        <PanelHeader title="AI-Generated Lessons Learned" accent />
        <div style={{ padding: 18, display: "flex", flexDirection: "column", gap: 12 }}>
          {[
            { icon: CheckCircle2, kind: "success", text: "Model accurately predicted High priority for 86% of VIP and accident events." },
            { icon: AlertTriangle, kind: "warning", text: "Water logging events in South Zone 1 monsoon corridors ran ~40% longer than predicted. Recalibrating with monsoon-month features." },
            { icon: Sparkles, kind: "ai", text: "Closed events on Mysore Road resolved 18% faster when 4+ officers were pre-staged within 30 min of detection." },
            { icon: TrendingUp, kind: "success", text: "AI Recommendation Adoption Rate (predicted-vs-actual priority match): 84% across all 8,173 events." },
          ].map((l, i) => (
            <div key={i} style={{ display: "flex", gap: 12, padding: 12, background: "var(--color-bg)", borderRadius: 8, border: "1px solid var(--color-border)" }}>
              <l.icon size={18} style={{ color: l.kind === "success" ? "var(--color-success)" : l.kind === "warning" ? "var(--color-warning)" : "var(--color-ai-accent)", flexShrink: 0, marginTop: 2 }} />
              <div style={{ fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.5 }}>{l.text}</div>
            </div>
          ))}
        </div>
      </Card>
    </AppShell>
  );
}
