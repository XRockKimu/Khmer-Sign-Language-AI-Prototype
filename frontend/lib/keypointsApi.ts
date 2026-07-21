/**
 * Isolated client for the backend's POST /keypoints/extract endpoint. UI
 * code never calls fetch() directly for this -- it goes through
 * extractFramePosition() and handles KeypointsApiError, mirroring
 * predictionApi.ts's structure.
 *
 * Requests go through the same /api/backend/* rewrite used by
 * predictionApi.ts, so no CORS configuration is needed on the backend.
 */

const EXTRACT_ENDPOINT = "/api/backend/keypoints/extract";
const REQUEST_TIMEOUT_MS = 5000;
const POSITION_FEATURES = 126;

export type KeypointsApiErrorKind = "network" | "timeout" | "http_error" | "invalid_response";

export class KeypointsApiError extends Error {
  readonly kind: KeypointsApiErrorKind;

  constructor(kind: KeypointsApiErrorKind, message: string) {
    super(message);
    this.name = "KeypointsApiError";
    this.kind = kind;
  }
}

function parsePosition(data: unknown): number[] {
  if (typeof data !== "object" || data === null) {
    throw new KeypointsApiError(
      "invalid_response",
      "The keypoint extraction service returned an unexpected response.",
    );
  }

  const { position } = data as Record<string, unknown>;

  if (
    !Array.isArray(position) ||
    position.length !== POSITION_FEATURES ||
    !position.every((v) => typeof v === "number")
  ) {
    throw new KeypointsApiError(
      "invalid_response",
      "The keypoint extraction service returned an unexpected response.",
    );
  }

  return position;
}

interface ExtractFramePositionOptions {
  signal?: AbortSignal;
}

export async function extractFramePosition(
  frame: Blob,
  options?: ExtractFramePositionOptions,
): Promise<number[]> {
  const timeoutController = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    timeoutController.abort();
  }, REQUEST_TIMEOUT_MS);

  const externalSignal = options?.signal;
  const forwardAbort = () => timeoutController.abort();
  externalSignal?.addEventListener("abort", forwardAbort);

  const formData = new FormData();
  formData.append("frame", frame, "frame.jpg");

  let response: Response;
  try {
    response = await fetch(EXTRACT_ENDPOINT, {
      method: "POST",
      body: formData,
      signal: timeoutController.signal,
    });
  } catch {
    if (timedOut) {
      throw new KeypointsApiError(
        "timeout",
        "The keypoint extraction request timed out.",
      );
    }
    throw new KeypointsApiError(
      "network",
      "Could not reach the keypoint extraction service.",
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
    throw new KeypointsApiError(
      "http_error",
      detail ?? `Keypoint extraction failed with status ${response.status}.`,
    );
  }

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new KeypointsApiError(
      "invalid_response",
      "The keypoint extraction service returned an invalid response.",
    );
  }

  return parsePosition(data);
}
