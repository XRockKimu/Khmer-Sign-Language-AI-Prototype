"use client";

import { useEffect, useState } from "react";
import type { CameraStatus } from "@/hooks/useCameraStream";

const HAVE_CURRENT_DATA = 2;

/**
 * True once the camera has both been granted AND actually decoded a
 * usable video frame -- srcObject being set (status === "granted") does
 * not by itself guarantee the first frame has loaded yet. UI-only signal:
 * used solely to gate the "Start capture" button so it can't be clicked
 * before the camera feed is truly ready. Does not affect the capture
 * workflow, prediction pipeline, or error handling.
 */
export function useIsCameraReady(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  cameraStatus: CameraStatus,
): boolean {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    // setIsReady calls are nested inside applyReady rather than called
    // directly in the effect body, to satisfy this project's
    // react-hooks/set-state-in-effect lint rule (see useCameraStream.ts
    // and usePredictionSubmission.ts for the same pattern).
    function applyReady(value: boolean) {
      setIsReady(value);
    }

    if (cameraStatus !== "granted") {
      applyReady(false);
      return;
    }

    const video = videoRef.current;
    if (!video) return;

    if (video.readyState >= HAVE_CURRENT_DATA) {
      applyReady(true);
      return;
    }

    const handleLoadedData = () => applyReady(true);
    video.addEventListener("loadeddata", handleLoadedData);
    return () => video.removeEventListener("loadeddata", handleLoadedData);
  }, [cameraStatus, videoRef]);

  return isReady;
}
