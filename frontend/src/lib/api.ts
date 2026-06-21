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

export interface DashboardResponse {
  events: any[]; // We will map these to CityEvent in the frontend
  kpis: {
    readiness: number;
    activeEvents: number;
    predictedHighRisk: number;
    activeRoadClosures: number;
    avgResolutionHrs: number;
  };
  telemetry: Record<string, any>;
}

/**
 * Handle fetch call with automated fallback if API server is offline.
 */
async function callApi<TPayload, TResponse>(
  endpoint: string,
  payload: TPayload
): Promise<TResponse> {
  const res = await fetch(`${API_BASE_URL}/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * M1 Road Closure Necessity Classifier Endpoint Caller
 */
export async function predictClosure(payload: ClosurePayload): Promise<ClosureResponse> {
  return callApi<ClosurePayload, ClosureResponse>("predict/closure", payload);
}

/**
 * M2 Clearance Duration Estimator Endpoint Caller
 */
export async function predictDuration(payload: DurationPayload): Promise<DurationResponse> {
  return callApi<DurationPayload, DurationResponse>("predict/duration", payload);
}

/**
 * M3 Zero-Shot Multimodal Sparse Event Predictor Endpoint Caller
 */
export async function predictMultimodal(payload: MultimodalPayload): Promise<MultimodalResponse> {
  return callApi<MultimodalPayload, MultimodalResponse>("predict/multimodal", payload);
}

/**
 * M4 Spatio-Temporal Graph WaveNet Traffic Predictor Endpoint Caller
 */
export async function predictTraffic(payload: TrafficPayload): Promise<TrafficResponse> {
  return callApi<TrafficPayload, TrafficResponse>("predict/traffic", payload);
}

/**
 * Stream Dashboard Live Feed Caller
 */
export async function getDashboardStream(): Promise<DashboardResponse> {
  const res = await fetch(`${API_BASE_URL}/stream/dashboard`, {
    method: "GET",
    headers: { "Content-Type": "application/json" }
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.statusText}`);
  }
  return await res.json();
}
