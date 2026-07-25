/**
 * Single source of truth for every model demo page: the Demo Hub
 * (app/demo/page.tsx) and the nav bar (components/DemoNav.tsx) both
 * render themselves entirely from this array. Adding a new model demo
 * (GRU, BiGRU, BiLSTM, ...) should only ever require editing this file
 * -- add an entry with `href: null` before the page exists (it shows
 * up disabled everywhere automatically), then flip `href` to the real
 * route once the page is built. No other file should need to change.
 */
export interface DemoModelConfig {
  /** Stable identifier, used as the React key -- not displayed. */
  id: string;
  /** Short label for the nav bar, e.g. "LSTM". */
  name: string;
  /** Full title for the Demo Hub card, e.g. "LSTM Model (20 Classes)". */
  fullName: string;
  /** One or two sentences shown on the Demo Hub card. */
  description: string;
  /**
   * Route to the demo page, or null if it hasn't been built yet.
   * Both the hub and the nav bar render a disabled placeholder instead
   * of a link whenever this is null.
   */
  href: string | null;
}

export const DEMO_MODELS: DemoModelConfig[] = [
  {
    id: "original",
    name: "Original",
    fullName: "Original Model (25 Classes)",
    description:
      "The original hands-only model: TensorFlow 2.15 / Keras 2, 25 gesture classes including No_action.",
    href: "/demo/ksl-prediction",
  },
  {
    id: "lstm",
    name: "LSTM",
    fullName: "LSTM Model (20 Classes)",
    description:
      "Pose + hands model served by backend_keras3: TensorFlow 2.16 / Keras 3, 20 gesture classes.",
    href: "/demo/ksl-lstm",
  },
  {
    id: "gru",
    name: "GRU",
    fullName: "GRU Model (20 Classes)",
    description:
      "Pose + hands GRU model served by backend_keras3: TensorFlow 2.16 / Keras 3, 20 gesture classes.",
    href: "/demo/ksl-gru",
  },
  {
    id: "bigru",
    name: "BiGRU",
    fullName: "BiGRU Model (20 Classes)",
    description:
      "Pose + hands BiGRU model served by backend_keras3: TensorFlow 2.16 / Keras 3, 20 gesture classes.",
    href: "/demo/ksl-bgru",
  },
  {
    id: "bilstm",
    name: "BiLSTM",
    fullName: "BiLSTM Model (20 Classes)",
    description:
      "Pose + hands BiLSTM model served by backend_keras3: TensorFlow 2.16 / Keras 3, 20 gesture classes.",
    href: "/demo/ksl-blstm",
  },
];
