"use client";

import { useEffect } from "react";
import { predictSequence, PredictionApiError, type PredictionTopKEntry } from "@/lib/predictionApi";
import { TOTAL_CAPTURE_FRAMES, type PredictionFlowState } from "@/hooks/usePredictionFlow";

const SEQUENCE_FEATURES = 126;

/**
 * Placeholder for the real (30, 126) keypoint sequence MediaPipe will
 * produce. The shape must match the backend's contract; values are
 * arbitrary since no hand is actually being tracked yet. Once MediaPipe
 * lands, this function is replaced by the real accumulated frame buffer --
 * nothing else in this hook changes.
 */
function createMockSequence(): number[][] {
  return Array.from({ length: TOTAL_CAPTURE_FRAMES }, () =>
    Array.from({ length: SEQUENCE_FEATURES }, () => Math.random() * 0.01),
  );
}

interface UsePredictionSubmissionArgs {
  state: PredictionFlowState;
  onSuccess: (label: string, confidence: number, topK: PredictionTopKEntry[]) => void;
  onError: (message: string) => void;
}

/**
 * Fires the real backend prediction request as soon as the flow enters
 * "predicting", and reports the outcome back through the flow's own
 * dispatchers. This is the permanent Day 18 integration -- unlike
 * DevFlowControls, it is not deleted when MediaPipe is wired up.
 */
export function usePredictionSubmission({
  state,
  onSuccess,
  onError,
}: UsePredictionSubmissionArgs) {
  useEffect(() => {
    if (state.status !== "predicting") return;

    const controller = new AbortController();
    let ignore = false;

    predictSequence(createMockSequence(), { signal: controller.signal })
      .then((result) => {
        if (ignore) return;
        onSuccess(result.predictedLabel, result.confidence, result.topK);
      })
      .catch((err: unknown) => {
        if (ignore) return;
        const message =
          err instanceof PredictionApiError
            ? err.message
            : "Something went wrong while contacting the prediction service.";
        onError(message);
      });

    return () => {
      ignore = true;
      controller.abort();
    };
  }, [state.status, onSuccess, onError]);
}
