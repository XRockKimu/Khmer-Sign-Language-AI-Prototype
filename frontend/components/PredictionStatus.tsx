import type { PredictionFlowState } from "@/hooks/usePredictionFlow";

type Tone = "neutral" | "info" | "active" | "success" | "warning" | "error";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300",
  info: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  active: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  success: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  warning: "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
  error: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
};

function StatusBadge({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  return (
    <span className={`rounded-full px-3 py-1 text-xs font-medium ${TONE_CLASSES[tone]}`}>
      {children}
    </span>
  );
}

export function PredictionStatus({ state }: { state: PredictionFlowState }) {
  switch (state.status) {
    case "idle":
      return (
        <div className="flex flex-col items-center gap-2">
          <StatusBadge tone="neutral">Waiting for a hand</StatusBadge>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Show a hand sign to the camera to begin.
          </p>
        </div>
      );

    case "hand_detected":
      return (
        <div className="flex flex-col items-center gap-2">
          <StatusBadge tone="info">Watching for a hand</StatusBadge>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Show your sign to the camera &mdash; capture starts automatically.
          </p>
        </div>
      );

    case "capturing": {
      const percent = Math.round((state.framesCaptured / state.totalFrames) * 100);
      return (
        <div className="flex w-full max-w-xs flex-col items-center gap-2">
          <StatusBadge tone="active">Capturing frames</StatusBadge>
          <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
            <div
              className="h-full rounded-full bg-amber-500 transition-[width]"
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            {state.framesCaptured} / {state.totalFrames} frames
          </p>
        </div>
      );
    }

    case "predicting":
      return (
        <div className="flex flex-col items-center gap-2">
          <StatusBadge tone="active">
            <span className="mr-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500 align-middle" />
            Predicting&hellip;
          </StatusBadge>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Running the sign through the model.
          </p>
        </div>
      );

    case "result":
      return (
        <div className="flex flex-col items-center gap-3">
          <StatusBadge tone="success">Result ready</StatusBadge>
          <p className="text-3xl font-semibold text-black dark:text-zinc-50">
            {state.label}
          </p>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Confidence: {Math.round(state.confidence * 100)}%
          </p>
          {state.topK.length > 0 && (
            <ol className="flex w-full max-w-xs flex-col gap-1 text-left text-xs text-zinc-600 dark:text-zinc-400">
              {state.topK.map((entry, index) => (
                <li
                  key={`${entry.label}-${index}`}
                  className="flex items-center justify-between rounded-md bg-zinc-100 px-2.5 py-1.5 dark:bg-zinc-800"
                >
                  <span>
                    {index + 1}. {entry.label}
                  </span>
                  <span className="tabular-nums">
                    {Math.round(entry.confidence * 100)}%
                  </span>
                </li>
              ))}
            </ol>
          )}
        </div>
      );

    case "uncertain":
      return (
        <div className="flex flex-col items-center gap-2">
          <StatusBadge tone="warning">Not confident enough</StatusBadge>
          <p className="max-w-xs text-sm text-zinc-500 dark:text-zinc-400">
            The model couldn&apos;t recognize that sign clearly. Please try
            again.
          </p>
          <p className="text-xs text-zinc-400 dark:text-zinc-500">
            Best guess: {state.label} ({Math.round(state.confidence * 100)}%)
          </p>
        </div>
      );

    case "error":
      return (
        <div className="flex flex-col items-center gap-2">
          <StatusBadge tone="error">Something went wrong</StatusBadge>
          <p className="max-w-xs text-sm text-zinc-500 dark:text-zinc-400">
            {state.message}
          </p>
          <p className="text-xs text-zinc-400 dark:text-zinc-500">
            Press Reset to try again.
          </p>
        </div>
      );
  }
}
