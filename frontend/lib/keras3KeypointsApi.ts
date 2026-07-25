/**
 * Shared keypoint-extraction client for every backend_keras3 model.
 *
 * /keypoints/extract has no model-specific behavior: all four Keras-3
 * models (lstm/gru/bgru/blstm) consume the identical 258-feature
 * pose+hands vector, so there is nothing to parameterize or duplicate
 * here. lib/lstmKeypointsApi.ts already implements this correctly
 * (Milestone 7) and is left untouched -- this file just re-exports it
 * under a name that doesn't imply "LSTM-only," for new pages (GRU now,
 * bgru/blstm later) to import from without that mismatch.
 */
export { extractFramePosition } from "@/lib/lstmKeypointsApi";
