import { type CSSProperties, type ReactNode } from "react";
import { Sparkles } from "lucide-react";

export function Card({ children, style, padded = true, className }: { children: ReactNode; style?: CSSProperties; padded?: boolean; className?: string }) {
  return (
    <div
      className={`cityos-card ${className ?? ""}`}
      style={{ padding: padded ? "20px 24px" : 0, ...style }}
    >
      {children}
    </div>
  );
}

export function PanelHeader({ title, right, accent }: { title: string; right?: ReactNode; accent?: boolean }) {
  return (
    <div
      className="cityos-panel-header"
      style={{
        padding: "12px 16px",
        borderRadius: "12px 12px 0 0",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 8,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600, fontSize: 14 }}>
        {accent && <Sparkles size={14} style={{ color: "var(--color-ai-accent)" }} />}
        {title}
      </div>
      {right}
    </div>
  );
}

type BadgeKind = "high" | "low" | "ai" | "closure" | "resolved" | "warning" | "neutral" | "success";
const BADGE: Record<BadgeKind, { bg: string; fg: string; bd?: string }> = {
  high:     { bg: "var(--color-critical-light)", fg: "var(--color-critical)" },
  low:      { bg: "var(--color-surface-elevated)", fg: "var(--color-text-secondary)" },
  ai:       { bg: "var(--color-ai-accent-light)", fg: "var(--color-ai-accent)" },
  closure:  { bg: "var(--color-critical-light)", fg: "var(--color-critical)" },
  resolved: { bg: "var(--color-success-light)", fg: "var(--color-success)" },
  warning:  { bg: "var(--color-warning-light)", fg: "var(--color-warning)" },
  success:  { bg: "var(--color-success-light)", fg: "var(--color-success)" },
  neutral:  { bg: "var(--color-surface-elevated)", fg: "var(--color-text-secondary)" },
};

export function Badge({ kind = "neutral", children, icon }: { kind?: BadgeKind; children: ReactNode; icon?: ReactNode }) {
  const c = BADGE[kind];
  return (
    <span className="cityos-badge" style={{ background: c.bg, color: c.fg }}>
      {icon}
      {children}
    </span>
  );
}

export function Kpi({
  label, value, sub, color = "var(--color-text-primary)", accent,
}: { label: string; value: ReactNode; sub?: ReactNode; color?: string; accent?: ReactNode }) {
  return (
    <div className="cityos-card count-up" style={{ padding: "18px 20px" }}>
      <div style={{ fontSize: 11, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginTop: 6, gap: 8 }}>
        <div style={{ fontSize: 30, fontWeight: 700, color, lineHeight: 1.1 }}>{value}</div>
        {accent}
      </div>
      {sub && <div style={{ marginTop: 4, fontSize: 12, color: "var(--color-text-secondary)" }}>{sub}</div>}
    </div>
  );
}

export function Button({
  children, variant = "primary", onClick, full, style, type = "button",
}: { children: ReactNode; variant?: "primary" | "secondary" | "ghost"; onClick?: () => void; full?: boolean; style?: CSSProperties; type?: "button" | "submit" }) {
  const base: CSSProperties = {
    borderRadius: 8,
    padding: "10px 16px",
    fontWeight: 600,
    fontSize: 13,
    cursor: "pointer",
    transition: "all 0.2s",
    border: "1px solid transparent",
    width: full ? "100%" : undefined,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
  };
  const styles: Record<string, CSSProperties> = {
    primary: { background: "var(--color-primary)", color: "#fff" },
    secondary: { background: "var(--color-surface)", color: "var(--color-primary)", borderColor: "var(--color-primary)" },
    ghost: { background: "transparent", color: "var(--color-text-secondary)" },
  };
  return (
    <button type={type} onClick={onClick} style={{ ...base, ...styles[variant], ...style }}>
      {children}
    </button>
  );
}

export function Divider({ label }: { label?: string }) {
  if (!label) return <div className="cityos-divider" />;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "16px 0" }}>
      <div style={{ flex: 1, height: 1, background: "var(--color-border)" }} />
      <span style={{ fontSize: 10, fontWeight: 600, color: "var(--color-text-muted)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
        {label}
      </span>
      <div style={{ flex: 1, height: 1, background: "var(--color-border)" }} />
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label style={{ display: "block", marginBottom: 14 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
        {label}
      </div>
      {children}
    </label>
  );
}

export function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        width: "100%",
        padding: "9px 12px",
        borderRadius: 8,
        border: "1px solid var(--color-border)",
        background: "var(--color-surface)",
        color: "var(--color-text-primary)",
        fontSize: 13,
        outline: "none",
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

export function ProgressBar({ value, color = "var(--color-primary)", height = 6 }: { value: number; color?: string; height?: number }) {
  return (
    <div style={{ width: "100%", height, background: "var(--color-surface-elevated)", borderRadius: 99, overflow: "hidden" }}>
      <div style={{ width: `${Math.max(0, Math.min(100, value))}%`, height: "100%", background: color, transition: "width 0.5s ease" }} />
    </div>
  );
}

export function Gauge({ value, color = "var(--color-primary)", size = 96, label }: { value: number; color?: string; size?: number; label?: string }) {
  const r = (size - 12) / 2;
  const c = 2 * Math.PI * r;
  const off = c - (value / 100) * c;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-surface-elevated)" strokeWidth={8} />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke={color} strokeWidth={8}
          strokeDasharray={c} strokeDashoffset={off}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset 0.7s ease" }}
        />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", textAlign: "center" }}>
        <div>
          <div style={{ fontSize: size / 4, fontWeight: 700, color: "var(--color-text-primary)" }}>{value}</div>
          {label && <div style={{ fontSize: 9, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>}
        </div>
      </div>
    </div>
  );
}

export function PageHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 20, gap: 16, flexWrap: "wrap" }}>
      <div>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: "var(--color-text-primary)", margin: 0 }}>{title}</h1>
        {subtitle && <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginTop: 4 }}>{subtitle}</div>}
      </div>
      {right}
    </div>
  );
}
