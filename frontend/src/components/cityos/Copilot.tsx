import { useState } from "react";
import { Sparkles, X, Send } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { API_BASE_URL } from "@/lib/api";

const COPILOT_API_KEY = "81275d9eaf0d45d98872a7a8ec8371ad.pqnwR-ARA0ruWkJmSXNVyCj4";

const SUGGESTIONS = [
  "How many officers should be deployed on Mysore Road right now?",
  "Which diversion route is best for MekhriCircle?",
  "What is the road closure risk for VIP movement?",
];



export function Copilot() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<{ role: "user" | "ai"; text: string }[]>([
    { role: "ai", text: "Hi — I'm CityOS Copilot. Ask me about active events, resource deployment, or diversion routes." },
  ]);

  const [isTyping, setIsTyping] = useState(false);

  const send = async (text: string) => {
    if (!text.trim()) return;
    
    setMsgs((m) => [...m, { role: "user", text }]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE_URL}/predict/copilot`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${COPILOT_API_KEY}`
        },
        body: JSON.stringify({
          model: "gpt-oss:120b",
          stream: false,
          messages: [
            ...msgs.map(m => ({
              role: m.role === "ai" ? "assistant" : "user",
              content: m.text
            })),
            { role: "user", content: text }
          ]
        })
      });

      const data = await response.json();
      const reply = data?.message?.content || data?.choices?.[0]?.message?.content || "No response received.";
      setMsgs((m) => [...m, { role: "ai", text: reply }]);
    } catch (err) {
      console.error("API Error:", err);
      setMsgs((m) => [...m, { role: "ai", text: "Error: Unable to connect to AI service." }]);
    } finally {
      setIsTyping(false);
    }
  };

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        aria-label="Open Copilot"
        style={{
          position: "fixed",
          bottom: 24,
          right: 24,
          width: 56,
          height: 56,
          borderRadius: 99,
          background: "var(--color-primary)",
          color: "#fff",
          border: "none",
          cursor: "pointer",
          boxShadow: "0 8px 24px rgba(40,116,240,0.4)",
          display: "grid",
          placeItems: "center",
          zIndex: 50,
        }}
      >
        <Sparkles size={22} />
      </button>
    );
  }

  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        right: 24,
        width: 360,
        height: 520,
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: 16,
        boxShadow: "var(--shadow-copilot)",
        display: "flex",
        flexDirection: "column",
        zIndex: 50,
        overflow: "hidden",
      }}
    >
      <style>{`
        .copilot-markdown {
          font-size: 12.5px;
          line-height: 1.5;
          overflow-x: auto;
        }
        .copilot-markdown p { margin-bottom: 8px; margin-top: 0; }
        .copilot-markdown p:last-child { margin-bottom: 0; }
        .copilot-markdown table { width: 100%; border-collapse: collapse; margin-bottom: 8px; }
        .copilot-markdown th, .copilot-markdown td { border: 1px solid var(--color-border); padding: 4px 8px; text-align: left; }
        .copilot-markdown th { background: var(--color-surface-elevated); font-weight: 600; }
        .copilot-markdown hr { border: 0; border-top: 1px solid var(--color-border); margin: 8px 0; }
        .copilot-markdown a { color: var(--color-primary); text-decoration: none; }
        .copilot-markdown a:hover { text-decoration: underline; }
        .copilot-markdown strong { color: var(--color-text-primary); font-weight: 600; }
      `}</style>
      <div
        style={{
          padding: "12px 16px",
          background: "var(--color-surface-elevated)",
          borderBottom: "1px solid var(--color-border)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <Sparkles size={16} style={{ color: "var(--color-ai-accent)" }} />
        <div style={{ flex: 1, fontWeight: 600, fontSize: 14, color: "var(--color-text-primary)" }}>CityOS Copilot</div>
        <button onClick={() => setOpen(false)} style={{ background: "transparent", border: "none", cursor: "pointer", color: "var(--color-text-secondary)" }}>
          <X size={16} />
        </button>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
        {msgs.map((m, i) => (
          <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
            <div
              style={{
                maxWidth: "82%",
                fontSize: 12.5,
                lineHeight: 1.45,
                padding: "8px 12px",
                borderRadius: 10,
                background: m.role === "user" ? "var(--color-primary-light)" : "var(--color-surface)",
                color: "var(--color-text-primary)",
                border: m.role === "ai" ? "1px solid var(--color-border)" : "none",
              }}
            >
              {m.role === "ai" && (
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--color-ai-accent)", marginBottom: 4, display: "flex", alignItems: "center", gap: 4 }}>
                  <Sparkles size={10} /> AI · Evidence-cited
                </div>
              )}
              {m.role === "ai" ? (
                <div className="copilot-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                </div>
              ) : (
                m.text
              )}
            </div>
          </div>
        ))}
        {isTyping && (
          <div style={{ display: "flex", justifyContent: "flex-start" }}>
            <div style={{
              maxWidth: "82%", fontSize: 12.5, lineHeight: 1.45, padding: "8px 12px",
              borderRadius: 10, background: "var(--color-surface)", color: "var(--color-text-secondary)",
              border: "1px solid var(--color-border)", fontStyle: "italic"
            }}>
              AI is analyzing...
            </div>
          </div>
        )}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4 }}>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              style={{
                fontSize: 11,
                padding: "5px 9px",
                borderRadius: 99,
                border: "1px solid var(--color-border)",
                background: "var(--color-surface)",
                color: "var(--color-text-secondary)",
                cursor: "pointer",
              }}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        style={{ borderTop: "1px solid var(--color-border)", padding: 10, display: "flex", gap: 6 }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask CityOS Copilot…"
          style={{
            flex: 1,
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            padding: "8px 10px",
            fontSize: 13,
            background: "var(--color-surface)",
            color: "var(--color-text-primary)",
            outline: "none",
          }}
        />
        <button
          type="submit"
          style={{
            width: 36,
            borderRadius: 8,
            background: "var(--color-primary)",
            color: "#fff",
            border: "none",
            cursor: "pointer",
            display: "grid",
            placeItems: "center",
          }}
          disabled={isTyping}
        >
          <Send size={14} />
        </button>
      </form>
    </div>
  );
}
