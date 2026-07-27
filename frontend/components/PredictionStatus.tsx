import type { PredictionFlowState } from "@/hooks/usePredictionFlow";

type Tone = "neutral" | "info" | "active" | "success" | "warning" | "error";

const TONE_CLASSES: Record<Tone, string> = {
  neutral:
    "border-black/[.06] bg-black/[.03] text-zinc-600 dark:border-white/[.08] dark:bg-white/[.04] dark:text-zinc-300",
  info: "border-blue-500/20 bg-blue-500/10 text-blue-700 dark:border-blue-500/20 dark:text-blue-300",
  active:
    "border-amber-500/20 bg-amber-500/10 text-amber-700 dark:border-amber-500/20 dark:text-amber-300",
  success:
    "border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:border-emerald-500/20 dark:text-emerald-300",
  warning:
    "border-orange-500/20 bg-orange-500/10 text-orange-700 dark:border-orange-500/20 dark:text-orange-300",
  error:
    "border-red-500/20 bg-red-500/10 text-red-700 dark:border-red-500/20 dark:text-red-300",
};

const DOT_CLASSES: Record<Tone, string> = {
  neutral: "bg-zinc-400 dark:bg-zinc-500",
  info: "bg-blue-500",
  active: "bg-amber-500",
  success: "bg-emerald-500",
  warning: "bg-orange-500",
  error: "bg-red-500",
};

function StatusBadge({
  tone,
  pulse,
  children,
}: {
  tone: Tone;
  pulse?: boolean;
  children: React.ReactNode;
}) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-[11px] font-semibold tracking-wide uppercase shadow-sm ${TONE_CLASSES[tone]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${DOT_CLASSES[tone]} ${pulse ? "animate-pulse" : ""}`}
      />
      {children}
    </span>
  );
}

export function PredictionStatus({ state }: { state: PredictionFlowState }) {
  switch (state.status) {
    case "idle":
      return (
        <div className="flex flex-col items-center gap-3">
          <StatusBadge tone="neutral">Waiting for a hand</StatusBadge>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Show a hand sign to the camera to begin.
          </p>
        </div>
      );

    case "hand_detected":
      return (
        <div className="flex flex-col items-center gap-3">
          <StatusBadge tone="info" pulse>
            Watching for a hand
          </StatusBadge>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Show your sign to the camera &mdash; capture starts automatically.
          </p>
        </div>
      );

    case "capturing": {
      const percent = Math.round((state.framesCaptured / state.totalFrames) * 100);
      return (
        <div className="flex w-full max-w-xs flex-col items-center gap-3">
          <StatusBadge tone="active" pulse>
            Capturing frames
          </StatusBadge>
          <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
            <div
              className="h-full rounded-full bg-amber-500 transition-[width] duration-300 ease-out"
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
        <div className="flex flex-col items-center gap-3">
          <StatusBadge tone="active" pulse>
            Predicting&hellip;
          </StatusBadge>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Running the sign through the model.
          </p>
        </div>
      );

    case "result":
      return (
        <div className="flex flex-col items-center gap-4">
          <StatusBadge tone="success">Result ready</StatusBadge>
          <p className="text-4xl font-semibold tracking-tight text-black dark:text-zinc-50">
            {state.label}
          </p>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Confidence: {Math.round(state.confidence * 100)}%
          </p>
          {state.topK.length > 0 && (
            <ol className="flex w-full max-w-xs flex-col gap-1.5 text-left text-xs text-zinc-600 dark:text-zinc-400">
              {state.topK.map((entry, index) => (
                <li
                  key={`${entry.label}-${index}`}
                  className={
                    index === 0
                      ? "flex items-center justify-between rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 font-medium text-emerald-700 dark:text-emerald-300"
                      : "flex items-center justify-between rounded-lg bg-zinc-100 px-3 py-2 dark:bg-zinc-800/70"
                  }
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
        <div className="flex flex-col items-center gap-3">
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
        <div className="flex flex-col items-center gap-3">
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
