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
    <details
      open
      className="group w-full rounded-2xl border border-black/[.06] bg-white/70 text-xs shadow-sm backdrop-blur-md dark:border-white/[.08] dark:bg-zinc-900/50"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-5 py-3 font-medium text-zinc-500 transition-colors hover:text-zinc-700 [&::-webkit-details-marker]:hidden dark:text-zinc-400 dark:hover:text-zinc-200">
        <span className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 dark:bg-zinc-600" />
          Developer controls
        </span>
        <span className="text-zinc-400 transition-transform group-open:rotate-180 dark:text-zinc-600">
          ▾
        </span>
      </summary>
      <div className="flex flex-wrap gap-2 px-5 pb-4">
        <button
          type="button"
          onClick={reportHandDetected}
          disabled={state.status !== "idle" || !isCameraReady}
          className="flex-1 rounded-full bg-black px-3.5 py-1.5 font-medium text-white transition-colors hover:bg-zinc-800 disabled:opacity-30 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
        >
          Start capture
        </button>
        <button
          type="button"
          onClick={reset}
          disabled={state.status === "idle"}
          className="flex-1 rounded-full border border-black/10 px-3.5 py-1.5 font-medium text-zinc-700 transition-colors hover:bg-black/5 disabled:opacity-30 dark:border-white/20 dark:text-zinc-300 dark:hover:bg-white/10"
        >
          Reset
        </button>
      </div>
    </details>
  );
}
