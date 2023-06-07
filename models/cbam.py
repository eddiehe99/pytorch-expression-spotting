import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8):
        super().__init__()
        self.ratio = ratio
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.add = layers.Add()
        self.activation = layers.Activation("sigmoid")
        self.multiply = layers.Multiply()

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer = tf.keras.Sequential(
            [
                layers.Dense(
                    channel // self.ratio,
                    activation="relu",
                    kernel_initializer="he_normal",
                    use_bias=True,
                    bias_initializer="zeros",
                    name="shared_layer_one",
                ),
                layers.Dense(
                    channel,
                    kernel_initializer="he_normal",
                    use_bias=True,
                    bias_initializer="zeros",
                    name="shared_layer_two",
                ),
            ]
        )
        self.reshape = layers.Reshape((1, 1, channel))

    def call(self, input_feature):
        avg_pool = self.avg_pool(input_feature)
        avg_pool = self.reshape(avg_pool)
        avg_pool = self.shared_layer(avg_pool)

        max_pool = self.max_pool(input_feature)
        max_pool = self.reshape(max_pool)
        max_pool = self.shared_layer(max_pool)

        add = self.add([avg_pool, max_pool])
        activation = self.activation(add)
        return self.multiply([input_feature, activation])


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = layers.Lambda(
            lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True)
        )
        self.max_pool = layers.Lambda(
            lambda x: tf.keras.backend.max(x, axis=3, keepdims=True)
        )
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )
        self.concat = layers.Concatenate(axis=3)
        self.multiply = layers.Multiply()

    def call(self, input_feature):
        avg_pool = self.avg_pool(input_feature)
        max_pool = self.max_pool(input_feature)
        concat = self.concat([avg_pool, max_pool])
        conv = self.conv(concat)
        return self.multiply([input_feature, conv])


class CBAM(layers.Layer):
    def __init__(self, ratio=8, kernel_size=7):
        super().__init__()
        self.module = tf.keras.Sequential(
            [ChannelAttention(ratio=ratio), SpatialAttention(kernel_size=kernel_size)]
        )

    def call(self, input_feature):
        cbam_feature = self.module(input_feature)
        return cbam_feature
