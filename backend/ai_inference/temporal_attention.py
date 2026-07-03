import tensorflow as tf
from tensorflow.keras.layers import Layer


class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        self.att_W = self.add_weight(
            name="att_W",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.att_b = self.add_weight(
            name="att_b",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )

        self.att_u = self.add_weight(
            name="att_u",
            shape=(feature_dim,),
            initializer="glorot_uniform",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time_steps, features)

        score = tf.tanh(
            tf.tensordot(inputs, self.att_W, axes=1) + self.att_b
        )

        score = tf.tensordot(
            score,
            self.att_u,
            axes=1,
        )

        attention_weights = tf.nn.softmax(score, axis=1)

        attention_weights = tf.expand_dims(
            attention_weights,
            axis=-1,
        )

        context = inputs * attention_weights

        return tf.reduce_sum(context, axis=1)

    def get_config(self):
        config = super().get_config()
        return config