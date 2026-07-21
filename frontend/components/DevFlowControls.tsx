"use client";

import { useEffect } from "react";
import type { PredictionFlowState } from "@/hooks/usePredictionFlow";

const HAND_DETECTED_TO_CAPTURE_DELAY_MS = 500;
const CAPTURE_FRAME_INTERVAL_MS = 40;

interface DevFlowControlsProps {
  state: PredictionFlowState;
  reportHandDetected: () => void;
  reportHandLost: () => void;
  startCapture: () => void;
  reportFrameCaptured: () => void;
  reset: () => void;
}

/**
 * Temporary stand-in for MediaPipe's hand-tracking signal. Once a hand is
 * detected, real MediaPipe integration will drive capture start and each
 * frame tick directly instead of these timers -- the effects below fire
 * usePredictionFlow's event dispatchers exactly the way that integration
 * will, so swapping them out later needs no changes elsewhere in the flow.
 * The actual prediction request (capturing -> predicting -> result/error) is
 * real as of Day 18 and lives in usePredictionSubmission, not here.
 */
export function DevFlowControls({
  state,
  reportHandDetected,
  reportHandLost,
  startCapture,
  reportFrameCaptured,
  reset,
}: DevFlowControlsProps) {
  useEffect(() => {
    if (state.status !== "hand_detected") return;
    const timer = setTimeout(startCapture, HAND_DETECTED_TO_CAPTURE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [state.status, startCapture]);

  useEffect(() => {
    if (state.status !== "capturing") return;
    const timer = setInterval(reportFrameCaptured, CAPTURE_FRAME_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [state.status, reportFrameCaptured]);

  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-amber-500/40 bg-amber-50 p-4 text-xs dark:bg-amber-950/20">
      <p className="font-medium text-amber-700 dark:text-amber-400">
        Developer controls &mdash; temporary until MediaPipe is wired up
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        <button
          type="button"
          onClick={reportHandDetected}
          disabled={state.status !== "idle"}
          className="rounded-full border border-black/10 px-3 py-1.5 font-medium disabled:opacity-40 dark:border-white/20"
        >
          Simulate hand detected
        </button>
        <button
          type="button"
          onClick={reportHandLost}
          disabled={state.status !== "hand_detected"}
          className="rounded-full border border-black/10 px-3 py-1.5 font-medium disabled:opacity-40 dark:border-white/20"
        >
          Simulate hand lost
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
