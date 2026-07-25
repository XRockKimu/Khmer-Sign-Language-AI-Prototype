"use client";

import { useEffect } from "react";
import {
  predictSequence as defaultPredictSequence,
  PredictionApiError,
  type PredictionResult,
  type PredictionTopKEntry,
} from "@/lib/predictionApi";
import type { PredictionFlowState } from "@/hooks/usePredictionFlow";

interface UsePredictionSubmissionArgs {
  state: PredictionFlowState;
  getSequence: () => number[][];
  onSuccess: (label: string, confidence: number, topK: PredictionTopKEntry[]) => void;
  onError: (message: string) => void;
  /**
   * Defaults to the existing backend's predictSequence. Pages targeting
   * backend_keras3 pass lib/lstmPredictionApi's version instead -- the
   * response shape is identical either way (see lstmPredictionApi.ts),
   * only the endpoint differs.
   */
  predictSequence?: (
    sequence: number[][],
    options?: { signal?: AbortSignal },
  ) => Promise<PredictionResult>;
}

/**
 * Fires the real backend prediction request as soon as the flow enters
 * "predicting", submitting whatever sequence getSequence() returns at that
 * moment (the real buffer accumulated by useGestureCapture -- (30, 126) for
 * the existing backend, (30, 258) for backend_keras3), and reports the
 * outcome back through the flow's own dispatchers.
 */
export function usePredictionSubmission({
  state,
  getSequence,
  onSuccess,
  onError,
  predictSequence = defaultPredictSequence,
}: UsePredictionSubmissionArgs) {
  useEffect(() => {
    if (state.status !== "predicting") return;

    const controller = new AbortController();
    let ignore = false;

    predictSequence(getSequence(), { signal: controller.signal })
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
  }, [state.status, getSequence, onSuccess, onError, predictSequence]);
}
