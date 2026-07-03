import tensorflow as tf


def print_model_info(model):

    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)

    print(f"TensorFlow Version : {tf.__version__}")
    print(f"Input Shape        : {model.input_shape}")
    print(f"Output Shape       : {model.output_shape}")

    print()

    model.summary()