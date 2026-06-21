// API Service Client for connecting the CityOS React Frontend to the FastAPI ML backend.
import { toast } from "sonner";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

export interface ClosurePayload {
  event_cause: string;
  corridor: string;
  priority: string;
  reported_datetime: string;
  description: string;
  comment: string;
  vehicle_type?: string | null;
  junction?: string | null;
  zone?: string | null;
}

export interface ClosureResponse {
  closure_required: boolean;
  probability: number;
  threshold: number;
  cause_group: string;
  model_mode: string;
}

export interface DurationPayload {
  event_cause: string;
  corridor: string;
  priority: string;
  reported_datetime: string;
  description: string;
  comment: string;
  vehicle_type?: string | null;
  junction?: string | null;
  zone?: string | null;
}

export interface DurationResponse {
  regime: "acute" | "chronic";
  estimated_duration_hrs: number;
  risk_score?: number;
  model_mode: string;
}

export interface MultimodalPayload {
  description: string;
  comment: string;
  event_cause?: string;
  corridor?: string;
}

export interface MultimodalResponse {
  text_length: number;
  zero_shot_risk_score: number;
  prediction_confidence: number;
  cause_inferred: string;
  model_mode: string;
}

export interface TrafficPayload {
  lat: number[];
  lng: number[];
  reported_datetime: string;
}

export interface TrafficResponse {
  junction: string;
  lat: number;
  lng: number;
  forecast_time: string;
  metrics: {
    predicted_speed_kmh: number;
    predicted_flow_veh_hr: number;
    average_delay_minutes: number;
    congestion_index: number;
  };
  road_network: {
    active_nodes_evaluated: number;
    adjacent_segments_congested: number;
    graph_validation_status: string;
  };
  diversion_route: number[][];
}

/**
 * Handle fetch call with automated fallback if API server is offline.
 */
async function callApi<TPayload, TResponse>(
  endpoint: string,
  payload: TPayload,
  fallbackGenerator: (payload: TPayload) => TResponse
): Promise<TResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(`API error: ${res.statusText}`);
    }
    return await res.json();
  } catch (error) {
    console.warn(`[FastAPI Offline] Falling back to simulated response for ${endpoint}:`, error);
    // Silent fallback to keep the app responsive but log warning
    return fallbackGenerator(payload);
  }
}

/**
 * M1 Road Closure Necessity Classifier Endpoint Caller
 */
export async function predictClosure(payload: ClosurePayload): Promise<ClosureResponse> {
  return callApi<ClosurePayload, ClosureResponse>("predict/closure", payload, (p) => {
    // Replicate model logic locally
    const cause = p.event_cause.toLowerCase();
    let prob = 0.12;
    if (["vip_movement", "protest"].includes(cause)) prob = 0.85;
    else if (["public_event", "procession"].includes(cause)) prob = 0.65;
    else if (["water_logging", "construction"].includes(cause)) prob = 0.35;
    
    // Add random factor
    prob = Math.max(0.01, Math.min(0.99, prob + (Math.random() * 0.1 - 0.05)));
    const thresh = 0.15;
    return {
      closure_required: prob >= thresh,
      probability: Number(prob.toFixed(4)),
      threshold: thresh,
      cause_group: cause,
      model_mode: "simulated_local_fallback",
    };
  });
}

/**
 * M2 Clearance Duration Estimator Endpoint Caller
 */
export async function predictDuration(payload: DurationPayload): Promise<DurationResponse> {
  return callApi<DurationPayload, DurationResponse>("predict/duration", payload, (p) => {
    const cause = p.event_cause.toLowerCase();
    const acuteCauses = ["vehicle_breakdown", "accident", "congestion", "procession", "protest"];
    const isAcute = acuteCauses.includes(cause);

    const baseDurations: Record<string, number> = {
      vehicle_breakdown: 0.8,
      accident: 0.8,
      congestion: 1.2,
      procession: 0.9,
      protest: 3.4,
      pot_holes: 18.7,
      water_logging: 14.1,
      construction: 13.3,
      road_conditions: 10.9,
      tree_fall: 10.6,
    };
    const baseHrs = baseDurations[cause] ?? 1.5;
    const est = Math.max(0.1, baseHrs + (Math.random() * 0.2 - 0.1) * baseHrs);

    return {
      regime: isAcute ? "acute" : "chronic",
      estimated_duration_hrs: Number(est.toFixed(2)),
      risk_score: isAcute ? undefined : Number((Math.random() * 0.5 + 0.1).toFixed(4)),
      model_mode: "simulated_local_fallback",
    };
  });
}

/**
 * M3 Zero-Shot Multimodal Sparse Event Predictor Endpoint Caller
 */
export async function predictMultimodal(payload: MultimodalPayload): Promise<MultimodalResponse> {
  return callApi<MultimodalPayload, MultimodalResponse>("predict/multimodal", payload, (p) => {
    const length = p.description.length;
    const rng = Math.random();
    return {
      text_length: length,
      zero_shot_risk_score: Number((rng * 100).toFixed(1)),
      prediction_confidence: Number((72 + rng * 20).toFixed(1)),
      cause_inferred: p.event_cause ?? "inferred",
      model_mode: "simulated_local_fallback",
    };
  });
}

/**
 * M4 Spatio-Temporal Graph WaveNet Traffic Predictor Endpoint Caller
 */
export async function predictTraffic(payload: TrafficPayload): Promise<TrafficResponse> {
  return callApi<TrafficPayload, TrafficResponse>("predict/traffic", payload, (p) => {
    const isPeak = Math.random() > 0.5;
    const baseSpeed = 24.0;
    const predSpeed = Math.max(5, baseSpeed * (isPeak ? 0.6 : 0.9) + (Math.random() * 6 - 3));
    const predFlow = Math.round((isPeak ? 1300 : 750) + (Math.random() * 200 - 100));
    const delay = Math.max(0.5, (baseSpeed / predSpeed) * 8.0 - 8.0);
    const lat = p.lat[0] ?? 12.9716;
    const lng = p.lng[0] ?? 77.5946;

    return {
      junction: "Fallback Snapped Junction",
      lat: lat,
      lng: lng,
      forecast_time: p.reported_datetime,
      metrics: {
        predicted_speed_kmh: Number(predSpeed.toFixed(1)),
        predicted_flow_veh_hr: predFlow,
        average_delay_minutes: Number(delay.toFixed(1)),
        congestion_index: Number(Math.min(1.0, delay / 12.0).toFixed(2)),
      },
      road_network: {
        active_nodes_evaluated: 124,
        adjacent_segments_congested: isPeak ? 9 : 3,
        graph_validation_status: "simulated_local_fallback",
      },
      diversion_route: [
        [lat, lng],
        [lat + 0.003, lng + 0.003],
        [lat + 0.006, lng],
        [lat + 0.003, lng - 0.003],
        [lat, lng]
      ]
    };
  });
}
