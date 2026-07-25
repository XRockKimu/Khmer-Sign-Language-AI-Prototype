/**
 * Isolated client for backend_keras3's POST /predict endpoint.
 *
 * Reuses PredictionApiError, PredictionResult, and parsePredictionResult
 * from lib/predictionApi.ts unchanged -- backend_keras3's InferenceService
 * was deliberately built to return the exact same response shape
 * (predicted_label, confidence, top_k, inference_ms), verified against a
 * running server during backend_keras3's own Milestone 5, so there is
 * nothing backend-specific about how the response is parsed. Only the
 * endpoint differs.
 *
 * Requests go through the /api/backend-lstm/* rewrite configured in
 * next.config.ts, same reasoning as lib/predictionApi.ts's /api/backend/*
 * rewrite: same-origin request, no CORS configuration needed on
 * backend_keras3.
 */

import {
  PredictionApiError,
  parsePredictionResult,
  type PredictionResult,
} from "@/lib/predictionApi";

const PREDICT_ENDPOINT = "/api/backend-lstm/predict";
const REQUEST_TIMEOUT_MS = 8000;

interface PredictSequenceOptions {
  signal?: AbortSignal;
}

export async function predictSequence(
  sequence: number[][],
  options?: PredictSequenceOptions,
): Promise<PredictionResult> {
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
    response = await fetch(PREDICT_ENDPOINT, {
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
