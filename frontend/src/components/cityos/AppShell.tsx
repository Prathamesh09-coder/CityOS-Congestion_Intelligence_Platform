import { type ReactNode } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard, Activity, Search, Users, Route as RouteIcon,
  FlaskConical, BookOpenCheck, Play, ShieldCheck,
} from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { DATA_NOTE } from "@/lib/cityos-data";
import { Copilot } from "./Copilot";

const NAV = [
  { to: "/command", label: "City Command", icon: LayoutDashboard },
  { to: "/forecast", label: "Impact Forecast", icon: Activity },
  { to: "/similarity", label: "Similarity Engine", icon: Search },
  { to: "/resources", label: "Resource Command", icon: Users },
  { to: "/diversion", label: "Smart Diversion", icon: RouteIcon },
  { to: "/sandbox", label: "Scenario Sandbox", icon: FlaskConical },
  { to: "/learning", label: "Post-Event Learning", icon: BookOpenCheck },
  { to: "/demo", label: "Executive Demo", icon: Play },
];

export function AppShell({ children, fullBleed = false }: { children: ReactNode; fullBleed?: boolean }) {
  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg)", display: "flex", flexDirection: "column" }}>
      <TopBar />
      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <SideNav />
        <main style={{ flex: 1, minWidth: 0, padding: fullBleed ? 0 : 24, overflow: "auto" }}>{children}</main>
      </div>
      <Copilot />
    </div>
  );
}

function TopBar() {
  const now = new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
  return (
    <header
      style={{
        height: 72,
        background: "var(--color-command-bar)",
        borderBottom: "1px solid var(--color-command-bar-border)",
        display: "flex",
        alignItems: "center",
        padding: "0 24px",
        gap: 16,
        position: "sticky",
        top: 0,
        zIndex: 30,
      }}
    >
      <Link to="/command" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none" }}>
        <div
          style={{
            width: 34,
            height: 34,
            background: "var(--color-primary)",
            borderRadius: 9,
            display: "grid",
            placeItems: "center",
            color: "#fff",
          }}
        >
          <ShieldCheck size={18} />
        </div>
        <div style={{ lineHeight: 1.1 }}>
          <div style={{ fontWeight: 700, color: "var(--color-text-primary)", fontSize: 16 }}>CityOS</div>
          <div style={{ fontSize: 11, color: "var(--color-text-muted)" }}>Predict · Decide · Deploy</div>
        </div>
      </Link>

      <div style={{ marginLeft: 12, display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ width: 7, height: 7, borderRadius: 99, background: "var(--color-success)" }} className="live-dot" />
        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--color-success)", letterSpacing: "0.08em" }}>LIVE</span>
      </div>

      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
        <div
          title={DATA_NOTE}
          style={{
            maxWidth: 460,
            fontSize: 11,
            color: "var(--color-text-secondary)",
            background: "var(--color-surface-elevated)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            padding: "6px 10px",
            lineHeight: 1.35,
          }}
        >
          <span style={{ fontWeight: 600, color: "var(--color-ai-accent)" }}>Model:</span> Trained on 8,173 events
          (Nov 2023 – Apr 2024) · resource recommendations are AI-generated.
        </div>
        <ThemeToggle />
        <div style={{ fontSize: 12, color: "var(--color-text-muted)" }}>{now} IST</div>
        <div
          style={{
            width: 34, height: 34, borderRadius: 99,
            background: "var(--color-primary-light)",
            color: "var(--color-primary)",
            display: "grid", placeItems: "center",
            fontWeight: 600, fontSize: 12,
            border: "1px solid var(--color-border)",
          }}
        >
          TP
        </div>
      </div>
    </header>
  );
}

function SideNav() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <nav
      style={{
        width: 232,
        background: "var(--color-surface)",
        borderRight: "1px solid var(--color-border)",
        padding: "16px 12px",
        flexShrink: 0,
        overflow: "auto",
      }}
    >
      <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-text-muted)", letterSpacing: "0.1em", padding: "0 8px 8px" }}>
        OPERATIONS
      </div>
      {NAV.map(({ to, label, icon: Icon }) => {
        const active = pathname === to;
        return (
          <Link
            key={to}
            to={to}
            style={{
              display: "flex", alignItems: "center", gap: 10,
              padding: "10px 12px",
              borderRadius: 8,
              marginBottom: 2,
              fontSize: 13,
              fontWeight: active ? 600 : 500,
              color: active ? "var(--color-primary)" : "var(--color-text-secondary)",
              background: active ? "var(--color-primary-light)" : "transparent",
              borderLeft: active ? "3px solid var(--color-primary)" : "3px solid transparent",
              textDecoration: "none",
            }}
          >
            <Icon size={16} />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
