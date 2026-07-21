"use client";

import { useCallback, useEffect, useRef } from "react";
import { extractFramePosition, KeypointsApiError } from "@/lib/keypointsApi";
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
  reportFrameCaptured: () => void;
  onCaptureError: (message: string) => void;
}

/**
 * Drives real gesture capture: while the flow is "capturing", sequentially
 * grabs a frame from the live camera feed, sends it to the backend's
 * /keypoints/extract endpoint (the real MediaPipe-based extractor), and
 * accumulates the returned 126-feature position vectors. Once 30 real
 * frames have been captured, getCapturedFrames() returns the full
 * (30, 126) sequence for usePredictionSubmission to submit to /predict.
 *
 * Frames are captured sequentially (await each extraction before grabbing
 * the next frame), not on a fixed timer, since each capture is a real
 * network round trip whose duration can vary.
 *
 * This replaces DevFlowControls' previous mock frame-capture timer. The
 * hand-detection trigger itself remains a manual dev control -- automatic
 * hand-presence detection is a separate, larger feature not required for
 * deliberate, one-sign-at-a-time manual testing.
 */
export function useGestureCapture({
  videoRef,
  state,
  reportFrameCaptured,
  onCaptureError,
}: UseGestureCaptureArgs) {
  const framesRef = useRef<number[][]>([]);

  useEffect(() => {
    if (state.status !== "capturing") return;

    let cancelled = false;
    framesRef.current = [];
    const controller = new AbortController();

    async function captureLoop() {
      for (let i = 0; i < TOTAL_CAPTURE_FRAMES; i++) {
        if (cancelled) return;

        const video = videoRef.current;
        if (!video || video.readyState < HAVE_CURRENT_DATA) {
          onCaptureError("Camera feed is not ready. Please try again.");
          return;
        }

        const blob = await captureFrameBlob(video);
        if (cancelled) return;
        if (!blob) {
          onCaptureError("Could not capture a frame from the camera.");
          return;
        }

        try {
          const position = await extractFramePosition(blob, {
            signal: controller.signal,
          });
          if (cancelled) return;
          framesRef.current.push(position);
          reportFrameCaptured();
        } catch (err) {
          if (cancelled) return;
          const message =
            err instanceof KeypointsApiError
              ? err.message
              : "Something went wrong while extracting hand keypoints.";
          onCaptureError(message);
          return;
        }
      }
    }

    captureLoop();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [state.status, videoRef, reportFrameCaptured, onCaptureError]);

  const getCapturedFrames = useCallback(() => framesRef.current, []);

  return { getCapturedFrames };
}
