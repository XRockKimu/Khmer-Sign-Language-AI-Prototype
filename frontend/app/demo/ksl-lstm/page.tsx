"use client";

import { CameraPreview } from "@/components/CameraPreview";
import { PredictionStatus } from "@/components/PredictionStatus";
import { DevFlowControls } from "@/components/DevFlowControls";
import { DemoNav } from "@/components/DemoNav";
import { useCameraStream } from "@/hooks/useCameraStream";
import { useIsCameraReady } from "@/hooks/useIsCameraReady";
import { usePredictionFlow } from "@/hooks/usePredictionFlow";
import { useGestureCapture } from "@/hooks/useGestureCapture";
import { usePredictionSubmission } from "@/hooks/usePredictionSubmission";
import { extractFramePosition } from "@/lib/lstmKeypointsApi";
import { predictSequence } from "@/lib/lstmPredictionApi";

/**
 * Milestone 7: the same demo flow as /demo/ksl-prediction, wired to
 * backend_keras3's LSTM model instead of the existing backend's 25-class
 * model. Every hook and component here is reused unchanged from the
 * original page -- only the two lib functions passed into
 * useGestureCapture/usePredictionSubmission differ (258-feature pose+hands
 * extraction and backend_keras3's /predict, instead of 126-feature
 * hands-only extraction and the existing backend's /predict).
 */
export default function KslLstmDemo() {
  const camera = useCameraStream();
  const isCameraReady = useIsCameraReady(camera.videoRef, camera.status);

  const {
    state,
    reportHandDetected,
    reportHandLost,
    startCapture,
    reportFrameCaptured,
    reportPredictionSuccess,
    reportPredictionError,
    reset,
  } = usePredictionFlow();

  const { getCapturedFrames } = useGestureCapture({
    videoRef: camera.videoRef,
    state,
    startCapture,
    reportFrameCaptured,
    onCaptureError: reportPredictionError,
    onHandLost: reportHandLost,
    extractFramePosition,
  });

  usePredictionSubmission({
    state,
    getSequence: getCapturedFrames,
    onSuccess: reportPredictionSuccess,
    onError: reportPredictionError,
    predictSequence,
  });

  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-zinc-950">
      <DemoNav />
      <main className="flex flex-1 w-full max-w-4xl flex-col items-center justify-center gap-10 px-6 py-20 text-center sm:px-10 sm:py-28">
        <div className="flex flex-col items-center gap-3">
          <span className="rounded-full border border-black/[.06] bg-black/[.02] px-3 py-1 text-xs font-medium tracking-wide text-zinc-500 dark:border-white/[.08] dark:bg-white/[.04] dark:text-zinc-400">
            Live Model Demo
          </span>
          <h1 className="text-3xl font-semibold tracking-tight text-black sm:text-4xl dark:text-zinc-50">
            Khmer Sign Language Prediction Demo (LSTM)
          </h1>
          <p className="max-w-md text-base leading-relaxed text-zinc-600 sm:text-lg dark:text-zinc-400">
            This page uses backend_keras3&apos;s LSTM model (pose + hands,
            20 classes) instead of the original 25-class hands-only model.
          </p>
        </div>
        <CameraPreview
          videoRef={camera.videoRef}
          status={camera.status}
          errorMessage={camera.errorMessage}
          requestCamera={camera.requestCamera}
        />
        <PredictionStatus state={state} />
        <DevFlowControls
          state={state}
          isCameraReady={isCameraReady}
          reportHandDetected={reportHandDetected}
          reset={reset}
        />
      </main>
    </div>
  );
}
