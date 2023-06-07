import tensorflow as tf
from tensorflow.keras import layers
from .cbam import CBAM as CBAM


class SOFTNetCBAM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # channel 1
        self.channel_1 = tf.keras.Sequential(
            [
                CBAM(),
                layers.Conv2D(3, (5, 5), padding="same", activation="relu"),
                CBAM(),
                # layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)),
            ]
        )
        # channel 2
        self.channel_2 = tf.keras.Sequential(
            [
                CBAM(),
                layers.Conv2D(5, (5, 5), padding="same", activation="relu"),
                CBAM(),
                # layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)),
            ]
        )
        # channel 3
        self.channel_3 = tf.keras.Sequential(
            [
                CBAM(),
                layers.Conv2D(8, (5, 5), padding="same", activation="relu"),
                CBAM(),
                # layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)),
            ]
        )
        # merge
        self.merged = layers.Concatenate()
        # interpretation
        self.interpretation = tf.keras.Sequential(
            [
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                layers.Flatten(),
                layers.Dense(400, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )

    def call(self, inputs):
        inputs_1 = inputs[:, :, :, 0]
        inputs_1 = tf.expand_dims(inputs_1, -1)
        inputs_2 = inputs[:, :, :, 1]
        inputs_2 = tf.expand_dims(inputs_2, -1)
        inputs_3 = inputs[:, :, :, 2]
        inputs_3 = tf.expand_dims(inputs_3, -1)
        # channel 1
        channel_1 = self.channel_1(inputs_1)
        # channel 2
        channel_2 = self.channel_2(inputs_2)
        # channel 3
        channel_3 = self.channel_3(inputs_3)
        # merge
        merged = self.merged([channel_1, channel_2, channel_3])
        # interpretation
        outputs = self.interpretation(merged)
        return outputs


# def load_soft_net():
#     model = SOFTNet()
#     # compile
#     sgd = tf.keras.optimizers.SGD(learning_rate=0.0005)
#     model.compile(
#         loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()]
#     )
#     return model


# def SOFTNet():
#     inputs1 = layers.Input(shape=(42, 42, 1))
#     conv1 = layers.Conv2D(3, (5, 5), padding="same", activation="relu")(inputs1)
#     pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv1)
#     # channel 2
#     inputs2 = layers.Input(shape=(42, 42, 1))
#     conv2 = layers.Conv2D(5, (5, 5), padding="same", activation="relu")(inputs2)
#     pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv2)
#     # channel 3
#     inputs3 = layers.Input(shape=(42, 42, 1))
#     conv3 = layers.Conv2D(8, (5, 5), padding="same", activation="relu")(inputs3)
#     pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv3)
#     # merge
#     merged = layers.Concatenate()([pool1, pool2, pool3])
#     # interpretation
#     merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(merged)
#     flat = layers.Flatten()(merged_pool)
#     dense = layers.Dense(400, activation="relu")(flat)
#     outputs = layers.Dense(1, activation="linear")(dense)
#     # Takes input u,v,s
#     model = tf.keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
#     # compile
#     sgd = tf.keras.optimizers.SGD(learning_rate=0.0005)
#     model.compile(
#         loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()]
#     )
#     return model
