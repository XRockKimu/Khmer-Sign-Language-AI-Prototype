"use client";

import { CameraPreview } from "@/components/CameraPreview";
import { PredictionStatus } from "@/components/PredictionStatus";
import { DevFlowControls } from "@/components/DevFlowControls";
import { usePredictionFlow } from "@/hooks/usePredictionFlow";
import { usePredictionSubmission } from "@/hooks/usePredictionSubmission";

export default function KslPredictionDemo() {
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

  usePredictionSubmission({
    state,
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
        <CameraPreview />
        <PredictionStatus state={state} />
        <DevFlowControls
          state={state}
          reportHandDetected={reportHandDetected}
          reportHandLost={reportHandLost}
          startCapture={startCapture}
          reportFrameCaptured={reportFrameCaptured}
          reset={reset}
        />
      </main>
    </div>
  );
}
