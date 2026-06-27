import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppresses the oneDNN warning

import numpy as np
from tensorflow.keras.models import load_model
import json

# ── 1. Load the model ─────────────────────────────────────────
print("=" * 50)
print("Loading model...")
model = load_model("models/best_model_25class_fix.h5")
print("Model loaded successfully!")

# ── 2. Print model details ────────────────────────────────────
print("=" * 50)
print("MODEL SUMMARY")
print("=" * 50)
model.summary()

# ── 3. Confirm input/output shape ─────────────────────────────
print("=" * 50)
print("SHAPE VERIFICATION")
print("=" * 50)
print(f"Input shape  : {model.input_shape}")   # Expect (None, 30, 252)
print(f"Output shape : {model.output_shape}")  # Expect (None, 25)

# ── 4. Load label map ─────────────────────────────────────────
print("=" * 50)
print("Loading label map...")
with open("models/label_map_25class.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Flip it from {label: index} to {index: label}
index_to_label = {v: k for k, v in label_map.items()}
print(f"Total classes : {len(index_to_label)}")
print(f"Sample labels : {index_to_label}")

# ── 5. Run dummy prediction ───────────────────────────────────
print("=" * 50)
print("DUMMY PREDICTION TEST")
print("=" * 50)
dummy_input = np.zeros((1, 30, 252), dtype=np.float32)
prediction = model.predict(dummy_input)

predicted_index = int(np.argmax(prediction))
predicted_label = index_to_label[predicted_index]
confidence = float(np.max(prediction))

print(f"Output shape      : {prediction.shape}")     # Expect (1, 25)
print(f"Predicted index   : {predicted_index}")
print(f"Predicted label   : {predicted_label}")
print(f"Confidence        : {confidence:.4f}")

# ── 6. Show top 3 predictions ─────────────────────────────────
print("=" * 50)
print("TOP 3 PREDICTIONS")
print("=" * 50)
top3_indices = np.argsort(prediction[0])[::-1][:3]
for i, idx in enumerate(top3_indices):
    label = index_to_label[idx]
    score = float(prediction[0][idx])
    print(f"  {i+1}. {label:<30s} {score:.4f}")

print("=" * 50)
print("Day 1 Complete!" if prediction.shape == (1, 25) else "Something is wrong!")
print("=" * 50)