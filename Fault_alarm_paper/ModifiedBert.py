import tensorflow as tf
from IO_function import *
import time
import os
import csv
import shutil


class ModifiedBert(tf.keras.models.Model):
    def __init__(self, dimension):
        super(ModifiedBert, self).__init__()
        self.dimension = dimension
        self.bert_nodes = 85
        self.dnn_nodes = 74
        self.weights_q = tf.keras.layers.Dense(self.dimension, activation=tf.keras.activations.relu, name="weights_q")
        self.weights_k = tf.keras.layers.Dense(self.dimension, activation=tf.keras.activations.relu, name="weights_k")
        self.weights_v = tf.keras.layers.Dense(self.dimension, activation=tf.keras.activations.relu, name="weights_v")
        self.attention_layer = tf.keras.layers.Dense(self.dimension, name="attention_layer")
        self.bert_hidden_1 = tf.keras.layers.Dense(self.bert_nodes, activation=tf.keras.activations.relu,
                                                   name="bert_hidden_1")
        self.bert_hidden_2 = tf.keras.layers.Dense(self.dimension, activation=tf.keras.activations.relu,
                                                   name="bert_hidden_2")
        self.dnn_hidden = tf.keras.layers.Dense(self.dnn_nodes, activation=tf.keras.activations.relu, name="dnn_hidden")
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.relu, name="output")

    def build(self, input_shape):
        super(ModifiedBert, self).build(input_shape)

    def call(self, inputs, **kwargs):
        weights_q = self.weights_q(inputs)
        weights_k = self.weights_k(inputs)
        weights_v = self.weights_v(inputs)

        matmul_qk = tf.matmul(weights_q, weights_k, transpose_b=True)
        dk = tf.cast(tf.shape(weights_k)[-1], tf.float32)
        scaled_attention_log = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_log, axis=-1)
        outputs = tf.matmul(attention_weights, weights_v)
        outputs = self.attention_layer(outputs)

        outputs = inputs + outputs
        mean = tf.math.reduce_mean(outputs, axis=0, keepdims=True)
        std = tf.math.reduce_std(outputs, axis=0, keepdims=True)
        outputs = (outputs - mean) / (std + 1e-6)

        outputs = self.bert_hidden_1(outputs)
        outputs = self.bert_hidden_2(outputs)
        outputs = self.dnn_hidden(outputs)
        outputs = self.output_layer(outputs)

        return outputs, attention_weights

    def get_config(self):
        config = super(ModifiedBert, self).get_config()
        config.update({'dimension': self.dimension})
        config.update({'bert_nodes': self.bert_nodes})
        config.update({'dnn_nodes': self.dnn_nodes})

        return config


def train_step(x_data, y_data, dimension, epochs) -> object:
    x_validation = x_data[:1204, ]
    y_validation = y_data[:1204]

    x_test = x_data[:301, ]
    y_test = y_data[:301]

    loss, mse, val_loss, val_mse = [], [], [], []

    optimizer = tf.keras.optimizers.Adam(0.01)

    train_loss = tf.keras.metrics.MeanAbsoluteError(name='loss')
    train_metric = tf.keras.metrics.MeanSquaredError(name='mean_squared_error')
    validation_loss = tf.keras.metrics.MeanAbsoluteError(name='val_loss')
    validation_metric = tf.keras.metrics.MeanSquaredError(name='val_mean_squared_error')

    model = ModifiedBert(dimension)
    model.build(input_shape=(None, dimension))
    model.summary()

    logs = "Epoch={}, loss:{}, mse:{}, val_loss:{}, val_mse:{}"

    start_time = time.time()
    for epoch in tf.range(1, epochs + 1):
        with tf.GradientTape() as tape:
            y_prediction, attention_weights = model(x_data, training=True)
            loss_values = tf.keras.losses.mape(y_data, y_prediction)
        gradients = tape.gradient(loss_values, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(y_data, y_prediction)
        train_metric.update_state(y_data, y_prediction)

        y_prediction_validation, _ = model(x_validation)
        validation_loss.update_state(y_validation, y_prediction_validation)
        validation_metric.update_state(y_validation, y_prediction_validation)

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

    train_time = time.time() - start_time

    start_time = time.time()
    y_prediction, _ = model(x_test)
    test_time = (time.time() - start_time) / 301.0

    model.save_weights('./checkpoint/ModifiedBert')
    model_size = os.path.getsize('./checkpoint')

    y_test = list(np.squeeze(np.transpose(y_test)))
    y_prediction = list(np.squeeze(np.transpose(y_prediction.numpy())))

    return [y_test, y_prediction, loss, mse, val_loss, val_mse, [train_time], [test_time], [model_size]], \
           attention_weights.numpy()


def train_modified_bert(file_name, index, dimension, train_times, epochs):
    attention_weights = 0
    file_name = file_name
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name)

    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

    x, y, _ = read_data_from_database(index, 0, [200, 220, 250])
    x_data = x.values
    y_data = y.values

    for i_index in range(train_times):
        result, attention_weights = train_step(x_data, y_data, dimension, epochs)

        if i_index == 0:
            file_path_y = os.path.join(path, file_name + "_y.csv")
            with open(file_path_y, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result[0])

        modified_bert_record_csv(path, file_name, result[1:])

    attention_mat = np.dot(attention_weights, x_data)
    attention = np.mean(attention_mat, axis=0)
    file_path_attention_weights = os.path.join(path, file_name + "_attention_weights.csv")
    numpy.savetxt(file_path_attention_weights, attention, delimiter=',')


# train_modified_bert("thermal_power", 0, 26, 50, 200)
# train_modified_bert("electric_power", 1, 26, 50, 200)
# train_modified_bert("hot_leg_temp", 4, 26, 50, 200)
# train_modified_bert("cold_leg_temp", 6, 26, 50, 200)
# train_modified_bert("SG_steam_flow", 16, 26, 50, 200)
train_modified_bert("PR_water_temp", 24, 26, 50, 200)
