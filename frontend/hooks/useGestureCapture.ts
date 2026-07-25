"use client";

import { useCallback, useEffect, useRef } from "react";
import {
  extractFramePosition as defaultExtractFramePosition,
  KeypointsApiError,
  type FramePosition,
} from "@/lib/keypointsApi";
import { TOTAL_CAPTURE_FRAMES, type PredictionFlowState } from "@/hooks/usePredictionFlow";

const HAVE_CURRENT_DATA = 2;

function captureFrameBlob(video: HTMLVideoElement): Promise<Blob | null> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  if (canvas.width === 0 || canvas.height === 0) {
    return Promise.resolve(null);
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.resolve(null);

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
}

interface UseGestureCaptureArgs {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  state: PredictionFlowState;
  startCapture: () => void;
  reportFrameCaptured: () => void;
  onCaptureError: (message: string) => void;
  onHandLost: () => void;
  /**
   * Defaults to the existing backend's extractFramePosition (126-length
   * hands-only vectors). Pages targeting backend_keras3 pass
   * lib/lstmKeypointsApi's version instead (258-length pose+hands
   * vectors) -- this hook's own capture-loop logic is identical either
   * way, only the network call differs.
   */
  extractFramePosition?: (
    frame: Blob,
    options?: { signal?: AbortSignal },
  ) => Promise<FramePosition>;
}

/**
 * Drives real gesture capture for the whole "watching for a hand -> capture
 * a 30-frame sequence" session, using the real MediaPipe-based
 * /keypoints/extract endpoint for both parts:
 *
 * - While the flow is "hand_detected" (a session is active but no hand has
 *   been seen yet), it repeatedly grabs a frame and checks the returned
 *   handDetected flag WITHOUT accumulating a sequence or calling /predict.
 *   The moment a frame reports a hand, it calls startCapture() itself,
 *   which flips the flow to "capturing".
 * - While the flow is "capturing", it accumulates 30 real frames as before.
 *   If a frame reports no hand, the partial sequence is discarded and
 *   onHandLost() is called, which returns the flow to "hand_detected" --
 *   this same effect then naturally restarts the watch loop above, so
 *   capture resumes automatically once a hand reappears.
 *
 * Either loop ends the moment the flow leaves "hand_detected"/"capturing"
 * (successful prediction, error, or an explicit reset), since the effect
 * simply stops running for any other status.
 *
 * Frames are grabbed sequentially (await each extraction before grabbing
 * the next), not on a fixed timer, since each grab is a real network round
 * trip whose duration can vary.
 */
export function useGestureCapture({
  videoRef,
  state,
  startCapture,
  reportFrameCaptured,
  onCaptureError,
  onHandLost,
  extractFramePosition = defaultExtractFramePosition,
}: UseGestureCaptureArgs) {
  const framesRef = useRef<number[][]>([]);

  useEffect(() => {
    if (state.status !== "hand_detected" && state.status !== "capturing") return;

    let cancelled = false;
    const controller = new AbortController();

    async function nextFrame(): Promise<FramePosition | null> {
      const video = videoRef.current;
      if (!video || video.readyState < HAVE_CURRENT_DATA) {
        onCaptureError("Camera feed is not ready. Please try again.");
        return null;
      }

      const blob = await captureFrameBlob(video);
      if (cancelled) return null;
      if (!blob) {
        onCaptureError("Could not capture a frame from the camera.");
        return null;
      }

      return extractFramePosition(blob, { signal: controller.signal });
    }

    function reportExtractionError(err: unknown) {
      if (cancelled) return;
      const message =
        err instanceof KeypointsApiError
          ? err.message
          : "Something went wrong while extracting hand keypoints.";
      onCaptureError(message);
    }

    async function waitForHandLoop() {
      try {
        while (!cancelled) {
          const result = await nextFrame();
          if (cancelled || result === null) return;

          if (result.handDetected) {
            startCapture();
            return;
          }
        }
      } catch (err) {
        reportExtractionError(err);
      }
    }

    async function captureLoop() {
      framesRef.current = [];
      try {
        for (let i = 0; i < TOTAL_CAPTURE_FRAMES; i++) {
          if (cancelled) return;

          const result = await nextFrame();
          if (cancelled || result === null) return;

          if (!result.handDetected) {
            framesRef.current = [];
            onHandLost();
            return;
          }

          framesRef.current.push(result.position);
          reportFrameCaptured();
        }
      } catch (err) {
        reportExtractionError(err);
      }
    }

    if (state.status === "capturing") {
      captureLoop();
    } else {
      waitForHandLoop();
    }

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [
    state.status,
    videoRef,
    startCapture,
    reportFrameCaptured,
    onCaptureError,
    onHandLost,
    extractFramePosition,
  ]);

  const getCapturedFrames = useCallback(() => framesRef.current, []);

  return { getCapturedFrames };
}
