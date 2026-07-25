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
import { extractFramePosition } from "@/lib/keras3KeypointsApi";
import { predictSequence as predictSequenceForModel } from "@/lib/keras3PredictionApi";

const MODEL_ID = "blstm";

/**
 * Same demo flow as /demo/ksl-gru and /demo/ksl-bgru, wired to
 * backend_keras3's BiLSTM model instead. Every hook and component here
 * is reused unchanged -- keypoint extraction is imported from
 * lib/keras3KeypointsApi (shared, model-agnostic) and predictSequence
 * is lib/keras3PredictionApi's generalized function, bound to "blstm"
 * via the small wrapper below rather than duplicating
 * usePredictionSubmission's call site logic.
 */
export default function KslBlstmDemo() {
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
    predictSequence: (sequence, options) =>
      predictSequenceForModel(MODEL_ID, sequence, options),
  });

  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-black">
      <DemoNav />
      <main className="flex flex-1 w-full max-w-3xl flex-col items-center justify-center gap-6 py-32 px-16 text-center">
        <h1 className="text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
          Khmer Sign Language Prediction Demo (BiLSTM)
        </h1>
        <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
          This page uses backend_keras3&apos;s BiLSTM model (pose + hands,
          20 classes).
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
