from IO_function import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import manifold
import tensorflow as tf
from RBM_function import Restricted_Boltzmann_Machine


def scala_function(index) -> object:
    normal_data, _ = read_data()
    normal_data = normal_data.values

    y_data = normal_data[:, index]
    x_data = np.delete(normal_data, index, axis=1)

    x_scala = StandardScaler().fit_transform(x_data)

    return x_scala, y_data


def PCA_function(index, dimension) -> object:
    x_scala, y_data = scala_function(index)

    pca = PCA(n_components=dimension)
    x_train = pca.fit_transform(x_scala)

    return x_train, y_data


def Isomap_function(index, dimension) -> object:
    x_scala, y_data = scala_function(index)

    Isomap = manifold.Isomap(n_components=dimension)
    x_train = Isomap.fit_transform(x_scala)

    return x_train, y_data


def LLE_function(index, dimension) -> object:
    x_scala, y_data = scala_function(index)

    lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=dimension, method='standard')
    x_train = lle.fit_transform(x_scala)

    return x_train, y_data


class AutoEncoder(tf.keras.models.Model):
    def __init__(self, input_dimension, reduction_dimension):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(reduction_dimension, activation=tf.keras.activations.relu)
        ])

        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(input_dimension, activation=tf.keras.activations.relu)
        ])

    def build(self, input_shape):
        super(AutoEncoder, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        encoders = self.encoder(inputs)
        input_reconstruction = self.decoder(encoders)

        return input_reconstruction


def AE_function(index, dimension, batchSize, epochs) -> object:
    x_scala, y_data = scala_function(index)
    input_shape = np.shape(x_scala)

    model = AutoEncoder(input_shape[1], dimension)
    model.build(input_shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])
    model.fit(x_scala, x_scala, batch_size=batchSize, epochs=epochs, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20, verbose=2)])

    x_train = model.encoder(x_scala)

    return x_train, y_data


class Variational_AutoEncoder(tf.keras.models.Model):
    def __init__(self, input_dimension, reduction_dimension):
        super(Variational_AutoEncoder, self).__init__()
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
        ])

        self.mean_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(reduction_dimension, activation=tf.keras.activations.relu)
        ])

        self.variance_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(reduction_dimension, activation=tf.keras.activations.relu)
        ])

        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(input_dimension)
        ])

    @staticmethod
    def parameterization(mean, variance) -> object:
        standard = tf.exp(0.5 * variance)
        z = tf.keras.backend.random_normal(tf.shape(standard)) * standard + mean
        return z

    @staticmethod
    def KL_divergence_function(reconstruct, x, mean, variance):
        mse = tf.sqrt(tf.reduce_mean(tf.square(reconstruct - x)))
        KL_divergence = -0.5 * tf.reduce_sum(1 + variance - tf.exp(variance) - mean ** 2)
        loss = mse + KL_divergence

        return loss

    def build(self, input_shape):
        super(Variational_AutoEncoder, self).build(input_shape)

    def call(self, inputs, training=None, mask=None) -> object:
        x = self.encoder(inputs)
        mean = self.mean_layer(x)
        variance = self.variance_layer(x)
        x = self.parameterization(mean, variance)
        x = self.decoder(x)

        return x, mean, variance


def VAE_function(index, dimension, epochs) -> object:
    x_scala, y_data = scala_function(index)
    input_shape = np.shape(x_scala)

    vae = Variational_AutoEncoder(input_shape[1], dimension)
    vae.build(input_shape)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            x_reconstruct, mean, variance = vae(x_scala)
            loss = vae.KL_divergence_function(x_reconstruct, x_scala, mean, variance)
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

    hidden_values = vae.encoder(x_scala)
    mean = vae.mean_layer(hidden_values)
    variance = vae.variance_layer(hidden_values)

    sess = tf.compat.v1.Session()
    with sess.as_default():
        mean = mean.numpy()
        variance = variance.numpy()

    x_data = np.hstack((mean, variance))

    return x_data, y_data


def RBM_function(index, dimension, epochs) -> object:
    x_scala, y_data = scala_function(index)
    input_shape = np.shape(x_scala)

    rbm = Restricted_Boltzmann_Machine(input_shape[1], dimension)
    rbm.build()

    for epoch in range(epochs):
        rbm.fit(x_scala, 2, 20)

    x_data = rbm.data_reduction_function(x_scala)
    return x_data, y_data





