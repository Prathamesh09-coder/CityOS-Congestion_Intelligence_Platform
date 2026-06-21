import { Moon, Sun } from "lucide-react";
import { useTheme } from "./theme";

export function ThemeToggle({ size = 36 }: { size?: number }) {
  const { theme, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      title={theme === "light" ? "Switch to Dark Mode" : "Switch to Light Mode"}
      aria-label="Toggle theme"
      style={{
        width: size,
        height: size,
        background: "var(--color-surface-elevated)",
        border: "1px solid var(--color-border)",
        borderRadius: 8,
        color: "var(--color-text-secondary)",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
      }}
    >
      {theme === "light" ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
