import { useState, useEffect, lazy, Suspense } from "react";
import type { CityEvent } from "@/lib/cityos-data";

interface Props {
  events: CityEvent[];
  selectedId?: string;
  onSelect?: (e: CityEvent) => void;
  showHeatmap?: boolean;
  showDiversion?: boolean;
  height?: number | string;
  diversionRoute?: number[][];
}

const LeafletMap = lazy(() => import("./LeafletMap"));

export function CityMap(props: Props) {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return (
      <div
        style={{
          width: "100%",
          height: props.height || "100%",
          borderRadius: 12,
          border: "1px solid var(--color-border)",
          background: "#0B0E15",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--color-text-muted)",
          fontSize: 12,
        }}
      >
        Loading city map telemetry...
      </div>
    );
  }

  return (
    <Suspense
      fallback={
        <div
          style={{
            width: "100%",
            height: props.height || "100%",
            borderRadius: 12,
            border: "1px solid var(--color-border)",
            background: "#0B0E15",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--color-text-muted)",
            fontSize: 12,
          }}
        >
          Loading city map telemetry...
        </div>
      }
    >
      <LeafletMap {...props} />
    </Suspense>
  );
}

