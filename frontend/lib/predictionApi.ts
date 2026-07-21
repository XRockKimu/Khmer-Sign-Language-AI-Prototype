/**
 * Isolated client for the backend's POST /predict endpoint. UI code never
 * calls fetch() directly -- it goes through predictSequence() and handles
 * PredictionApiError, so the request/response contract only needs to be
 * known in this one file.
 *
 * Requests go through the /api/backend/* rewrite configured in
 * next.config.ts rather than an absolute backend URL, so the browser sees a
 * same-origin request and no CORS configuration is needed on the backend.
 */

const PREDICT_ENDPOINT = "/api/backend/predict";
const REQUEST_TIMEOUT_MS = 8000;

export interface PredictionTopKEntry {
  label: string;
  confidence: number;
}

export interface PredictionResult {
  predictedLabel: string;
  confidence: number;
  topK: PredictionTopKEntry[];
  inferenceMs: number;
}

export type PredictionApiErrorKind =
  | "network"
  | "timeout"
  | "http_error"
  | "invalid_response";

export class PredictionApiError extends Error {
  readonly kind: PredictionApiErrorKind;

  constructor(kind: PredictionApiErrorKind, message: string) {
    super(message);
    this.name = "PredictionApiError";
    this.kind = kind;
  }
}

function isTopKEntry(value: unknown): value is PredictionTopKEntry {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  return typeof record.label === "string" && typeof record.confidence === "number";
}

function parsePredictionResult(data: unknown): PredictionResult {
  if (typeof data !== "object" || data === null) {
    throw new PredictionApiError(
      "invalid_response",
      "The prediction service returned an unexpected response.",
    );
  }

  const record = data as Record<string, unknown>;
  const { predicted_label, confidence, top_k, inference_ms } = record;

  if (
    typeof predicted_label !== "string" ||
    typeof confidence !== "number" ||
    !Array.isArray(top_k) ||
    !top_k.every(isTopKEntry) ||
    typeof inference_ms !== "number"
  ) {
    throw new PredictionApiError(
      "invalid_response",
      "The prediction service returned an unexpected response.",
    );
  }

  return {
    predictedLabel: predicted_label,
    confidence,
    topK: top_k,
    inferenceMs: inference_ms,
  };
}

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
