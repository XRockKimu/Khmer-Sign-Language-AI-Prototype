"use client";

import { useCallback, useReducer } from "react";
import type { PredictionTopKEntry } from "@/lib/predictionApi";

export const TOTAL_CAPTURE_FRAMES = 30;

export type PredictionFlowState =
  | { status: "idle" }
  | { status: "hand_detected" }
  | { status: "capturing"; framesCaptured: number; totalFrames: number }
  | { status: "predicting" }
  | { status: "result"; label: string; confidence: number; topK: PredictionTopKEntry[] }
  | { status: "error"; message: string };

type PredictionFlowAction =
  | { type: "HAND_DETECTED" }
  | { type: "HAND_LOST" }
  | { type: "START_CAPTURE" }
  | { type: "FRAME_CAPTURED" }
  | { type: "PREDICTION_SUCCESS"; label: string; confidence: number; topK: PredictionTopKEntry[] }
  | { type: "PREDICTION_ERROR"; message: string }
  | { type: "RESET" };

const initialState: PredictionFlowState = { status: "idle" };

function reducer(
  state: PredictionFlowState,
  action: PredictionFlowAction,
): PredictionFlowState {
  switch (action.type) {
    case "HAND_DETECTED":
      return state.status === "idle" ? { status: "hand_detected" } : state;

    case "HAND_LOST":
      return state.status === "hand_detected" ? { status: "idle" } : state;

    case "START_CAPTURE":
      return state.status === "hand_detected"
        ? { status: "capturing", framesCaptured: 0, totalFrames: TOTAL_CAPTURE_FRAMES }
        : state;

    case "FRAME_CAPTURED": {
      if (state.status !== "capturing") return state;
      const framesCaptured = Math.min(state.framesCaptured + 1, state.totalFrames);
      if (framesCaptured >= state.totalFrames) {
        return { status: "predicting" };
      }
      return { ...state, framesCaptured };
    }

    case "PREDICTION_SUCCESS":
      return state.status === "predicting"
        ? {
            status: "result",
            label: action.label,
            confidence: action.confidence,
            topK: action.topK,
          }
        : state;

    case "PREDICTION_ERROR":
      return state.status === "predicting"
        ? { status: "error", message: action.message }
        : state;

    case "RESET":
      return initialState;

    default:
      return state;
  }
}

/**
 * State machine for the capture -> predict -> result pipeline. Callers drive
 * it purely through the returned event dispatchers, so the same dispatchers
 * can later be wired to real MediaPipe hand-tracking events and a real
 * backend prediction response without changing this file.
 */
export function usePredictionFlow() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const reportHandDetected = useCallback(() => dispatch({ type: "HAND_DETECTED" }), []);
  const reportHandLost = useCallback(() => dispatch({ type: "HAND_LOST" }), []);
  const startCapture = useCallback(() => dispatch({ type: "START_CAPTURE" }), []);
  const reportFrameCaptured = useCallback(() => dispatch({ type: "FRAME_CAPTURED" }), []);
  const reportPredictionSuccess = useCallback(
    (label: string, confidence: number, topK: PredictionTopKEntry[]) =>
      dispatch({ type: "PREDICTION_SUCCESS", label, confidence, topK }),
    [],
  );
  const reportPredictionError = useCallback(
    (message: string) => dispatch({ type: "PREDICTION_ERROR", message }),
    [],
  );
  const reset = useCallback(() => dispatch({ type: "RESET" }), []);

  return {
    state,
    reportHandDetected,
    reportHandLost,
    startCapture,
    reportFrameCaptured,
    reportPredictionSuccess,
    reportPredictionError,
    reset,
  };
}
