import tensorflow as tf
from IO_function import *


class Embedding(tf.keras.models.Model):
    def __init__(self, reduction_dimension, data_dimension):
        super(Embedding, self).__init__()
        self.reduction_dimension = reduction_dimension
        self.data_dimension = data_dimension
        self.embedding = tf.keras.layers.Embedding(1000000, reduction_dimension)
        self.flatten = tf.keras.layers.Flatten()
        self.decoder = tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu)

    def build(self, input_shape):
        super(Embedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.embedding(inputs)
        outputs = self.flatten(outputs)
        outputs = self.decoder(outputs)
        return outputs

    def encoder(self, inputs):
        outputs = self.embedding(inputs)
        outputs = self.flatten(outputs)
        return outputs

    def get_config(self):
        config = super(Embedding, self).get_config()
        config.update("reduction_dimension", self.reduction_dimension)
        config.update("data_dimension", self.data_dimension)
        return config


def embedding_train(x_data, dimension):
    model = Embedding(dimension, 25)

    model.build((None, 25))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])
    model.summary()

    model.fit(x_data, x_data, batch_size=32, epochs=200, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    return model

