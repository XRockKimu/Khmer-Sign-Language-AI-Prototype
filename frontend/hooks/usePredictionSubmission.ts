"use client";

import { useEffect } from "react";
import { predictSequence, PredictionApiError, type PredictionTopKEntry } from "@/lib/predictionApi";
import type { PredictionFlowState } from "@/hooks/usePredictionFlow";

interface UsePredictionSubmissionArgs {
  state: PredictionFlowState;
  getSequence: () => number[][];
  onSuccess: (label: string, confidence: number, topK: PredictionTopKEntry[]) => void;
  onError: (message: string) => void;
}

/**
 * Fires the real backend prediction request as soon as the flow enters
 * "predicting", submitting whatever sequence getSequence() returns at that
 * moment (the real (30, 126) buffer accumulated by useGestureCapture), and
 * reports the outcome back through the flow's own dispatchers.
 */
export function usePredictionSubmission({
  state,
  getSequence,
  onSuccess,
  onError,
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
  }, [state.status, getSequence, onSuccess, onError]);
}
