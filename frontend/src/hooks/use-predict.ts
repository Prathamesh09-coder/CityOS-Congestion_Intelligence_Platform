import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  predictClosure,
  predictDuration,
  predictMultimodal,
  predictTraffic,
  type ClosurePayload,
  type ClosureResponse,
  type DurationPayload,
  type DurationResponse,
  type MultimodalPayload,
  type MultimodalResponse,
  type TrafficPayload,
  type TrafficResponse,
} from "../lib/api";

/**
 * Mutation hook for M1 Road Closure prediction
 */
export function useClosurePrediction() {
  return useMutation<ClosureResponse, Error, ClosurePayload>({
    mutationFn: predictClosure,
    onSuccess: (data) => {
      const mode = data.model_mode.includes("fallback") ? "Simulated Backup" : "Live LightGBM Champion";
      toast.success(
        `M1 Prediction complete via ${mode}: ${data.closure_required ? "Closure Required" : "No Closure Needed"} (${(
          data.probability * 100
        ).toFixed(1)}% probability)`
      );
    },
    onError: (err) => {
      toast.error(`M1 Prediction failed: ${err.message}`);
    },
  });
}

/**
 * Mutation hook for M2 Clearance Duration prediction
 */
export function useDurationPrediction() {
  return useMutation<DurationResponse, Error, DurationPayload>({
    mutationFn: predictDuration,
    onSuccess: (data) => {
      const mode = data.model_mode.includes("fallback") ? "Simulated Backup" : `Live ${data.regime === "acute" ? "CatBoost" : "GBST Survival"}`;
      toast.success(
        `M2 Clearance prediction complete via ${mode}: Estimated ${data.estimated_duration_hrs} hours`
      );
    },
    onError: (err) => {
      toast.error(`M2 Prediction failed: ${err.message}`);
    },
  });
}

/**
 * Mutation hook for M3 Zero-Shot Multimodal Sparse Event Forecast
 */
export function useMultimodalPrediction() {
  return useMutation<MultimodalResponse, Error, MultimodalPayload>({
    mutationFn: predictMultimodal,
    onSuccess: (data) => {
      const mode = data.model_mode.includes("fallback") ? "Simulated Backup" : "Live MuRIL LoRA Transformer";
      toast.success(
        `M3 Zero-Shot forecast complete via ${mode}: Risk ${data.zero_shot_risk_score}%, Confidence ${data.prediction_confidence}%`
      );
    },
    onError: (err) => {
      toast.error(`M3 Forecast failed: ${err.message}`);
    },
  });
}

/**
 * Query hook for M4 Spatio-Temporal Graph WaveNet traffic forecasting
 */
export function useTrafficPrediction(payload: TrafficPayload, enabled = true) {
  return useQuery<TrafficResponse, Error>({
    // Omit reported_datetime from queryKey to avoid infinite cache busting on every render
    queryKey: ["trafficPrediction", payload.junction, payload.lat, payload.lng],
    queryFn: () => predictTraffic(payload),
    enabled,
    staleTime: 30000, // 30 seconds
    refetchInterval: 5000, // Make it real-time every 5 seconds
    meta: {
      onError: (err: any) => {
        toast.error(`M4 Traffic forecast failed: ${err.message}`);
      },
    },
  });
}
