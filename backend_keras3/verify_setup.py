"""
Milestone 3 verification script.

Extends Milestone 2: exercises the full offline 686-feature pipeline
(sequence_normalizer + feature_engineering, orchestrated by
feature_formatter) on a synthetic (30, 258) position sequence, prints
the shape at every stage, and confirms the final (30, 686) tensor runs
through the loaded model without a shape error.

Still not a FastAPI app, still no camera, still no frame buffering --
the synthetic input stands in for "already-extracted pose+hands
positions," which a later milestone will produce from a real camera
feed. The predicted label below is meaningless (random input), only
the shape and wiring are being verified.

Run directly:

    python verify_setup.py
"""

import numpy as np

from ai_inference import model_loader, label_loader, feature_formatter
from ai_inference.sequence_normalizer import SignLanguageNormalizer
from ai_inference.feature_engineering import FeatureEngineeringEngine


def main():
    # This script verifies the LSTM model specifically, unchanged in
    # purpose since Milestone 3 -- model_loader.load_model() now takes
    # a model_id (Milestone 9, multi-model), so this is passed
    # explicitly rather than relying on a since-removed default.
    model = model_loader.load_model("lstm")
    labels = label_loader.load_labels()

    print()
    print("=" * 50)
    print("MODEL SHAPES")
    print("=" * 50)
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    print()
    print("=" * 50)
    print("LABEL MAP")
    print("=" * 50)
    print(f"Total labels: {len(labels)}")
    for index in (0, 5, 19):
        print(f"  index {index}: {labels[index]}")

    # ------------------------------------------------------------
    # Stage-by-stage pipeline verification, using the same synthetic
    # (30, 258) sequence for both the manual walk-through below and
    # the real feature_formatter.format_sequence() entry point, so
    # the two can be cross-checked against each other.
    # ------------------------------------------------------------
    print()
    print("=" * 50)
    print("PIPELINE STAGE SHAPES (synthetic input)")
    print("=" * 50)

    positions_258 = np.random.rand(30, 258).astype(np.float32)
    print(f"1. Position features        : {positions_258.shape} "
          f"(expected (30, 258))")

    velocity = np.zeros_like(positions_258)
    velocity[1:] = positions_258[1:] - positions_258[:-1]
    positions_and_velocity_516 = np.concatenate([positions_258, velocity], axis=1)
    print(f"2. Position + velocity      : {positions_and_velocity_516.shape} "
          f"(expected (30, 516))")

    normalizer = SignLanguageNormalizer(
        use_root_center=True,
        use_shoulder_width=True,
        use_hand_normalization=True,
        use_scale_normalization=True,
        use_clipping=True,
        target_hand_size=0.3,
        clip_bounds=(-2.0, 2.0),
    )
    normalized_516 = normalizer.normalize_sequence(positions_and_velocity_516)
    print(f"3. Normalized (still 516)   : {normalized_516.shape} "
          f"(expected (30, 516))")

    feature_engine = FeatureEngineeringEngine()
    engineered_686 = feature_engine.process_sequence(normalized_516)
    print(f"4. Final engineered vector  : {engineered_686.shape} "
          f"(expected (30, 686))")

    assert engineered_686.shape == (30, 686), (
        f"Manual pipeline produced {engineered_686.shape}, expected (30, 686)"
    )

    # Cross-check: the real production entry point, given the exact
    # same (30, 258) input, must produce the identical (30, 686) tensor.
    via_format_sequence = feature_formatter.format_sequence(positions_258)
    print(f"5. feature_formatter output : {via_format_sequence.shape} "
          f"(expected (30, 686))")

    assert via_format_sequence.shape == (30, 686), (
        f"feature_formatter.format_sequence produced "
        f"{via_format_sequence.shape}, expected (30, 686)"
    )
    assert np.allclose(engineered_686, via_format_sequence), (
        "Manual stage-by-stage pipeline and feature_formatter.format_sequence "
        "produced different values for the same input -- wiring mismatch."
    )
    print("Manual pipeline and feature_formatter.format_sequence agree exactly.")

    # ------------------------------------------------------------
    # Feed the (30, 686) tensor into the loaded model.
    # ------------------------------------------------------------
    print()
    print("=" * 50)
    print("RUNNING PREDICTION ON PIPELINE OUTPUT")
    print("=" * 50)

    model_input = np.expand_dims(via_format_sequence, axis=0)  # (1, 30, 686)
    print(f"Model input shape: {model_input.shape}, dtype: {model_input.dtype}")

    prediction = model.predict(model_input, verbose=0)
    print(f"Prediction shape    : {prediction.shape}")
    print(f"Sum of probabilities: {prediction.sum():.6f}")

    predicted_index = int(np.argmax(prediction[0]))
    predicted_label = labels[predicted_index]
    confidence = float(prediction[0][predicted_index])

    print()
    print("=" * 50)
    print("PREDICTION -> LABEL")
    print("=" * 50)
    print(f"Predicted class index: {predicted_index}")
    print(f"Predicted label      : {predicted_label}")
    print(f"Confidence           : {confidence:.6f}")
    print()
    print("(Input was random synthetic data -- this label is not meaningful,")
    print(" only the shape compatibility and pipeline wiring are being verified.)")


if __name__ == "__main__":
    main()
