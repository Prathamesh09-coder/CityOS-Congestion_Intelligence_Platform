import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight, ShieldCheck } from "lucide-react";
import { ThemeToggle } from "@/components/cityos/ThemeToggle";
import { CityMap } from "@/components/cityos/CityMap";
import { ACTIVE_EVENTS, DATA_NOTE } from "@/lib/cityos-data";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "CityOS · Launch" },
      { name: "description", content: "AI-Powered Event Congestion Intelligence for Bengaluru. Forecast traffic disruption before it happens." },
      { property: "og:title", content: "CityOS · Predict. Decide. Deploy." },
      { property: "og:description", content: "AI command and control for Bengaluru traffic." },
    ],
  }),
  component: Login,
});

function Login() {
  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg)", position: "relative", overflow: "hidden" }}>
      {/* Full-bleed Bengaluru map backdrop */}
      <div style={{ position: "absolute", inset: 0, opacity: 0.85 }}>
        <CityMap events={ACTIVE_EVENTS} height="100%" showHeatmap />
      </div>
      <div style={{ position: "absolute", inset: 0, background: "linear-gradient(180deg, transparent 0%, var(--color-bg) 100%)", opacity: 0.55 }} />

      <div style={{ position: "relative", zIndex: 2, minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
        <div
          className="slide-up"
          style={{
            position: "relative",
            background: "var(--color-surface)",
            borderRadius: 16,
            border: "1px solid var(--color-border)",
            padding: "40px 44px",
            maxWidth: 520,
            width: "100%",
            boxShadow: "0 8px 48px rgba(0,0,0,0.18)",
          }}
        >
          <div style={{ position: "absolute", top: 16, right: 16 }}>
            <ThemeToggle size={32} />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
            <div style={{ width: 44, height: 44, background: "var(--color-primary)", borderRadius: 12, display: "grid", placeItems: "center", color: "#fff" }}>
              <ShieldCheck size={22} />
            </div>
            <div>
              <div style={{ fontSize: 32, fontWeight: 700, color: "var(--color-text-primary)", lineHeight: 1 }}>CityOS</div>
              <div style={{ fontSize: 12, color: "var(--color-text-muted)", marginTop: 4, letterSpacing: "0.1em", textTransform: "uppercase" }}>
                Predict · Decide · Deploy
              </div>
            </div>
          </div>

          <h1 style={{ fontSize: 24, fontWeight: 600, color: "var(--color-text-primary)", margin: "0 0 10px" }}>
            AI-Powered Event Congestion Intelligence
          </h1>
          <p style={{ fontSize: 15, color: "var(--color-text-secondary)", lineHeight: 1.5, margin: 0 }}>
            Forecast traffic disruption before it happens. Deploy smarter interventions across Bengaluru's corridors.
          </p>

          <Link
            to="/command"
            style={{
              marginTop: 28,
              display: "inline-flex", alignItems: "center", gap: 8,
              background: "var(--color-primary)", color: "#fff",
              padding: "14px 28px", borderRadius: 8,
              fontSize: 15, fontWeight: 600, textDecoration: "none",
            }}
          >
            Launch CityOS <ArrowRight size={16} />
          </Link>

          <div style={{ marginTop: 28, padding: "12px 14px", background: "var(--color-surface-elevated)", borderRadius: 8, fontSize: 11, color: "var(--color-text-muted)", lineHeight: 1.5 }}>
            {DATA_NOTE}
          </div>
        </div>
      </div>
    </div>
  );
}
