import numpy as np

from ai_inference.model_loader import load_model
from ai_inference.label_loader import load_labels
from ai_inference.model_info import print_model_info

def dummy_prediction(model, labels):

    print()
    print("=" * 50)
    print("Dummy Prediction Test")
    print("=" * 50)

    dummy = np.zeros(
        (1, 30, 252),
        dtype=np.float32
    )

    prediction = model.predict(dummy, verbose=0)

    predicted_index = prediction.argmax()

    confidence = prediction.max()

    label_names = list(labels.keys())

    print(f"Predicted Index : {predicted_index}")
    print(f"Predicted Label : {label_names[predicted_index]}")
    print(f"Confidence      : {confidence:.4f}")


def main():

    model = load_model()

    labels = load_labels()

    print_model_info(model)

    dummy_prediction(model, labels)


if __name__ == "__main__":
    main()