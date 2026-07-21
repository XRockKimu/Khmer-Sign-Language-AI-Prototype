"use client";

import { useEffect } from "react";
import type { PredictionFlowState } from "@/hooks/usePredictionFlow";

const HAND_DETECTED_TO_CAPTURE_DELAY_MS = 500;

interface DevFlowControlsProps {
  state: PredictionFlowState;
  isCameraReady: boolean;
  reportHandDetected: () => void;
  reportHandLost: () => void;
  startCapture: () => void;
  reset: () => void;
}

/**
 * Temporary stand-in for automatic hand-presence detection only. Frame
 * capture itself is real as of this integration (see useGestureCapture),
 * and the prediction request (capturing -> predicting -> result/error) has
 * been real since Day 18 (usePredictionSubmission) -- this component now
 * only simulates the one remaining manual step: deciding *when* a capture
 * window starts. A human deliberately clicking "Start capture" before
 * performing a sign is actually preferable for controlled manual testing,
 * so this is not necessarily replaced by automatic detection later, only
 * supplemented by it.
 */
export function DevFlowControls({
  state,
  isCameraReady,
  reportHandDetected,
  reportHandLost,
  startCapture,
  reset,
}: DevFlowControlsProps) {
  useEffect(() => {
    if (state.status !== "hand_detected") return;
    const timer = setTimeout(startCapture, HAND_DETECTED_TO_CAPTURE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [state.status, startCapture]);

  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-amber-500/40 bg-amber-50 p-4 text-xs dark:bg-amber-950/20">
      <p className="font-medium text-amber-700 dark:text-amber-400">
        Developer controls &mdash; click before performing a sign
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        <button
          type="button"
          onClick={reportHandDetected}
          disabled={state.status !== "idle" || !isCameraReady}
          className="rounded-full border border-black/10 px-3 py-1.5 font-medium disabled:opacity-40 dark:border-white/20"
        >
          Start capture
        </button>
        <button
          type="button"
          onClick={reportHandLost}
          disabled={state.status !== "hand_detected"}
          className="rounded-full border border-black/10 px-3 py-1.5 font-medium disabled:opacity-40 dark:border-white/20"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={reset}
          disabled={state.status === "idle"}
          className="rounded-full border border-black/10 px-3 py-1.5 font-medium disabled:opacity-40 dark:border-white/20"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
