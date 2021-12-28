import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from IO_function import *


class VariationAutoEncoder(tf.keras.models.Model):
    def __init__(self, reduction_dimension, data_dimension):
        super(VariationAutoEncoder, self).__init__()
        self.data_dimension = data_dimension
        self.reduction_dimension = reduction_dimension
        self.encoder_layer = tf.keras.layers.Dense(self.reduction_dimension, activation=tf.keras.activations.relu,
                                                   name="encoder")
        self.decoder_layer = tf.keras.layers.Dense(self.data_dimension, activation=tf.keras.activations.relu,
                                                   name="decoder")

    def encoder(self, inputs):
        scale = StandardScaler()
        outputs = scale.fit_transform(inputs)
        mean = self.encoder_layer(outputs)
        log_sigma = self.encoder_layer(outputs)
        mean = mean.numpy()
        log_sigma = log_sigma.numpy()
        outputs = np.hstack((mean, log_sigma))
        return outputs

    def build(self, input_shape):
        super(VariationAutoEncoder, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean = self.encoder_layer(inputs)
        log_sigma = self.encoder_layer(inputs)

        log_sigma_shape = tf.shape(log_sigma)
        eps = tf.random.normal(log_sigma_shape, 0, 1, dtype=tf.float32)
        outputs = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma)), eps))
        outputs = self.decoder_layer(outputs)

        return outputs

    def get_config(self):
        config = super(VariationAutoEncoder, self).get_config()
        config.update("data_dimension", self.data_dimension)
        config.update("reduction_dimension", self.reduction_dimension)
        return config


def log_function(x1, x2):
    return tf.multiply(x1, tf.math.log(tf.math.divide(x1, x2)))


def kl_div(rho, rho_hat):
    term2_num = tf.constant(1.) - rho
    term2_den = tf.constant(1.) - rho_hat
    kl = log_function(rho, rho_hat) + log_function(term2_num, term2_den)
    return kl


def vae_train(x_data, dimension):
    scale = StandardScaler()
    x_data = scale.fit_transform(x_data)

    optimizer = tf.keras.optimizers.Adam(0.01)
    loss, mse, val_loss, val_mse = [], [], [], []
    train_loss = tf.keras.metrics.MeanAbsoluteError(name='loss')
    train_metric = tf.keras.metrics.MeanSquaredError(name='mean_squared_error')
    validation_loss = tf.keras.metrics.MeanAbsoluteError(name='val_loss')
    validation_metric = tf.keras.metrics.MeanSquaredError(name='val_mean_squared_error')
    logs = "Epoch={}, loss:{}, mse:{}, val_loss:{}, val_mse:{}"

    model = VariationAutoEncoder(dimension, 25)
    model.build((None, 25))
    model.summary()

    for epoch in tf.range(1, 2000):
        with tf.GradientTape() as tape:
            y_predict = model(x_data, training=True)
            kl_div_loss = tf.reduce_mean(kl_div(0.02, tf.reduce_mean(y_predict, 0)))
            loss_values = tf.keras.losses.MSE(x_data, y_predict) + kl_div_loss
        gradients = tape.gradient(loss_values, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(x_data, y_predict)
        train_metric.update_state(x_data, y_predict)

        y_prediction_validation = model(x_data[:1204, ])
        validation_loss.update_state(x_data[:1204, ], y_prediction_validation)
        validation_metric.update_state(x_data[:1204, ], y_prediction_validation)

        tf.print(tf.strings.format(logs, (epoch, train_loss.result(), train_metric.result(), validation_loss.result(),
                                          validation_metric.result())))

        loss.append(train_loss.result().numpy())
        mse.append(train_metric.result().numpy())
        val_loss.append(validation_loss.result().numpy())
        val_mse.append(validation_metric.result().numpy())

        train_loss.reset_states()
        validation_loss.reset_states()
        train_metric.reset_states()
        validation_metric.reset_states()

    return model
