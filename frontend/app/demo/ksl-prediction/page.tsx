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
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans dark:bg-zinc-950">
      <DemoNav />
      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-8 px-6 py-10 sm:px-10 sm:py-14">
        <div className="flex flex-col gap-2">
          <span className="w-fit rounded-full border border-black/[.06] bg-black/[.02] px-3 py-1 text-xs font-medium tracking-wide text-zinc-500 dark:border-white/[.08] dark:bg-white/[.04] dark:text-zinc-400">
            Live Model Demo
          </span>
          <h1 className="text-2xl font-semibold tracking-tight text-black sm:text-3xl dark:text-zinc-50">
            Khmer Sign Language Prediction Demo
          </h1>
          <p className="max-w-xl text-sm leading-relaxed text-zinc-600 sm:text-base dark:text-zinc-400">
            Sign a gesture in front of your camera and see the predicted
            Khmer label in real time.
          </p>
        </div>

        <div className="flex flex-1 flex-col gap-6 lg:flex-row lg:items-start">
          <div className="w-full lg:w-[72%]">
            <CameraPreview
              videoRef={camera.videoRef}
              status={camera.status}
              errorMessage={camera.errorMessage}
              requestCamera={camera.requestCamera}
            />
          </div>

          <aside className="flex w-full flex-col gap-4 lg:w-[28%]">
            <PredictionStatus state={state} />
            <DevFlowControls
              state={state}
              isCameraReady={isCameraReady}
              reportHandDetected={reportHandDetected}
              reset={reset}
            />
          </aside>
        </div>
      </main>
    </div>
  );
}
