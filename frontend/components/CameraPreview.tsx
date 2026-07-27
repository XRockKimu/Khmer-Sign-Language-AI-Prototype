"use client";

import type { CameraStatus } from "@/hooks/useCameraStream";

interface CameraPreviewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  errorMessage: string | null;
  requestCamera: () => void;
}

/**
 * Presentational only -- the camera stream itself (useCameraStream) is
 * owned by the page, not this component, since the same videoRef also
 * needs to be read by useGestureCapture to grab frames for real keypoint
 * extraction. Two independent useCameraStream() calls would each request
 * their own separate MediaStream (and a duplicate permission prompt), so
 * the hook is lifted up and shared via props instead.
 */
export function CameraPreview({
  videoRef,
  status,
  errorMessage,
  requestCamera,
}: CameraPreviewProps) {
  const showVideo = status === "granted";

  return (
    <div className="w-full rounded-2xl border border-black/[.06] bg-white/70 p-3 shadow-sm backdrop-blur-md dark:border-white/[.08] dark:bg-zinc-900/50">
      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`h-full w-full object-cover ${showVideo ? "block" : "hidden"}`}
        />
        {!showVideo && (
          <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center text-sm">
            {status === "requesting" ? (
              <>
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />
                <p className="text-zinc-300">Requesting camera access…</p>
              </>
            ) : (
              <>
                <p className="text-zinc-200">{errorMessage}</p>
                <button
                  type="button"
                  onClick={requestCamera}
                  className="rounded-full border border-white/20 bg-white/10 px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-white/20"
                >
                  Try again
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
