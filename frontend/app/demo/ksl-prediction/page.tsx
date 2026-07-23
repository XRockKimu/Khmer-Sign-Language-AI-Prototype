"use client";

import { CameraPreview } from "@/components/CameraPreview";
import { PredictionStatus } from "@/components/PredictionStatus";
import { DevFlowControls } from "@/components/DevFlowControls";
import { useCameraStream } from "@/hooks/useCameraStream";
import { useIsCameraReady } from "@/hooks/useIsCameraReady";
import { usePredictionFlow } from "@/hooks/usePredictionFlow";
import { useGestureCapture } from "@/hooks/useGestureCapture";
import { usePredictionSubmission } from "@/hooks/usePredictionSubmission";

export default function KslPredictionDemo() {
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
  });

  usePredictionSubmission({
    state,
    getSequence: getCapturedFrames,
    onSuccess: reportPredictionSuccess,
    onError: reportPredictionError,
  });

  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex flex-1 w-full max-w-3xl flex-col items-center justify-center gap-6 py-32 px-16 text-center">
        <h1 className="text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
          Khmer Sign Language Prediction Demo
        </h1>
        <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
          This page will let you sign a gesture in front of your camera and
          see the predicted Khmer label in real time.
        </p>
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
