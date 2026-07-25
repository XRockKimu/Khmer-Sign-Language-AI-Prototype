/**
 * Isolated client for backend_keras3's generalized POST
 * /predict/{modelId} endpoint (added in Milestone 9).
 *
 * lib/lstmPredictionApi.ts (Milestone 7) is left untouched and keeps
 * calling the original unparameterized /predict, which the backend
 * still serves identically for the LSTM model -- verified by
 * comparing responses for identical input during this milestone. This
 * file is for every OTHER model (gru now, bgru/blstm later), reusing
 * PredictionApiError and parsePredictionResult from lib/predictionApi.ts
 * exactly like lstmPredictionApi.ts does, since the response schema is
 * identical for every model regardless of which one served it.
 *
 * Requests go through the same /api/backend-lstm/* rewrite configured
 * in next.config.ts -- the rewrite's name predates this backend
 * serving more than one model. Renaming it was judged unnecessary
 * churn for this milestone (it is an internal routing detail, not
 * something a user or this file's callers see) and is not required
 * for correctness.
 */

import {
  PredictionApiError,
  parsePredictionResult,
  type PredictionResult,
} from "@/lib/predictionApi";

const REQUEST_TIMEOUT_MS = 8000;

interface PredictSequenceOptions {
  signal?: AbortSignal;
}

export async function predictSequence(
  modelId: string,
  sequence: number[][],
  options?: PredictSequenceOptions,
): Promise<PredictionResult> {
  const endpoint = `/api/backend-lstm/predict/${modelId}`;

  const timeoutController = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    timeoutController.abort();
  }, REQUEST_TIMEOUT_MS);

  const externalSignal = options?.signal;
  const forwardAbort = () => timeoutController.abort();
  externalSignal?.addEventListener("abort", forwardAbort);

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
      signal: timeoutController.signal,
    });
  } catch {
    if (timedOut) {
      throw new PredictionApiError(
        "timeout",
        "The prediction request timed out. Please try again.",
      );
    }
    throw new PredictionApiError(
      "network",
      "Could not reach the prediction service. Check your connection and try again.",
    );
  } finally {
    clearTimeout(timeoutId);
    externalSignal?.removeEventListener("abort", forwardAbort);
  }

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const errorBody: unknown = await response.json();
      if (
        typeof errorBody === "object" &&
        errorBody !== null &&
        typeof (errorBody as Record<string, unknown>).detail === "string"
      ) {
        detail = (errorBody as Record<string, unknown>).detail as string;
      }
    } catch {
      // Response body wasn't valid JSON; fall back to the generic message below.
    }
    throw new PredictionApiError(
      "http_error",
      detail ?? `Prediction request failed with status ${response.status}.`,
    );
  }

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new PredictionApiError(
      "invalid_response",
      "The prediction service returned an invalid response.",
    );
  }

  return parsePredictionResult(data);
}
