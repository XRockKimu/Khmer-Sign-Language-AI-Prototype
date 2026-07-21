"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus =
  | "requesting"
  | "granted"
  | "denied"
  | "unavailable"
  | "error";

interface UseCameraStreamResult {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  errorMessage: string | null;
  requestCamera: () => void;
}

function isCameraSupported() {
  return (
    typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia
  );
}

function describeError(err: unknown): { status: CameraStatus; message: string } {
  if (err instanceof DOMException) {
    switch (err.name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
        return {
          status: "denied",
          message:
            "Camera access was denied. Please allow camera permission in your browser's site settings, then try again.",
        };
      case "NotFoundError":
      case "DevicesNotFoundError":
        return {
          status: "unavailable",
          message: "No camera was found on this device.",
        };
      case "NotReadableError":
      case "TrackStartError":
        return {
          status: "unavailable",
          message:
            "The camera could not be started. It may already be in use by another application.",
        };
      default:
        break;
    }
  }
  return {
    status: "error",
    message: "Something went wrong while accessing the camera. Please try again.",
  };
}

export function useCameraStream(): UseCameraStreamResult {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [attempt, setAttempt] = useState(0);
  // Always starts as "requesting" so the client's first render matches the
  // server-rendered HTML; `navigator` is unavailable during SSR, so checking
  // camera support here would otherwise cause a hydration mismatch.
  const [status, setStatus] = useState<CameraStatus>("requesting");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const requestCamera = useCallback(() => {
    setStatus("requesting");
    setErrorMessage(null);
    setAttempt((n) => n + 1);
  }, []);

  useEffect(() => {
    let ignore = false;

    (async () => {
      if (!isCameraSupported()) {
        if (!ignore) {
          setStatus("unavailable");
          setErrorMessage("This browser does not support camera access.");
        }
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        if (ignore) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStatus("granted");
      } catch (err) {
        if (ignore) return;
        const { status: nextStatus, message } = describeError(err);
        setStatus(nextStatus);
        setErrorMessage(message);
      }
    })();

    return () => {
      ignore = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [attempt]);

  return { videoRef, status, errorMessage, requestCamera };
}
