"use client";

import type { PredictionFlowState } from "@/hooks/usePredictionFlow";

interface DevFlowControlsProps {
  state: PredictionFlowState;
  isCameraReady: boolean;
  reportHandDetected: () => void;
  reset: () => void;
}

/**
 * Minimal manual controls for the capture session. Hand-presence detection
 * and the capture/predict pipeline are both fully real (see
 * useGestureCapture and usePredictionSubmission) -- the only step still
 * manual is *starting* a watching session, which is the intended UX
 * (a deliberate "Start capture" click before performing a sign), not a
 * stand-in for a missing feature. "Reset" ends an active session at any
 * point, covering both cancellation and recovering from an error/result.
 */
export function DevFlowControls({
  state,
  isCameraReady,
  reportHandDetected,
  reset,
}: DevFlowControlsProps) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-amber-500/40 bg-amber-50 p-4 text-xs dark:bg-amber-950/20">
      <p className="font-medium text-amber-700 dark:text-amber-400">
        Developer controls
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
