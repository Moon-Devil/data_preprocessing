import shutil
import tensorflow as tf
import time
from IO_function import *
import csv

TRAIN_TIMES = 50
EPOCHS = 100
DIMENSION = 26
DATA_SET_LENGTH = 301


def dnn(x_data, y_data) -> object:
    x_data = x_data.values
    y_data = y_data.values

    x_test = x_data[:DATA_SET_LENGTH, ]
    y_test = y_data[:DATA_SET_LENGTH]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(74, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    train_start_time = time.time()
    history = model.fit(x_data, y_data, epochs=EPOCHS, batch_size=32, validation_split=0.2)
    train_time = time.time() - train_start_time

    train_loss = history.history['loss']
    train_mse = history.history['mean_squared_error']
    train_val_loss = history.history['val_loss']
    train_val_mse = history.history['val_mean_squared_error']

    eva_loss, eva_accuracy = model.evaluate(x_data, y_data)

    test_start_time = time.time()
    y_predict = model.predict(x_test)
    test_time = time.time() - test_start_time

    y_test = list(np.squeeze(np.transpose(y_test)))
    y_predict = list(np.squeeze(np.transpose(y_predict)))

    model.save("train_model.h5")
    model_size = os.path.getsize("train_model.h5")

    return [y_test, y_predict, train_loss, train_mse, train_val_loss, train_val_mse, [eva_loss], [eva_accuracy],
            [train_time], [test_time], [model_size]]


def train_dnn(index, file_name):
    file_name = file_name
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name)

    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

    x_train, y_train, _ = read_data_from_database(index, 0, [200, 220, 250])
    for i_index in range(TRAIN_TIMES):
        result = dnn(x_train, y_train)

        if i_index == 0:
            file_path_y = os.path.join(path, file_name + "_y.csv")
            with open(file_path_y, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result[0])

        best_record_csv(path, file_name, result[1:])


def rnn(x_data, y_data) -> object:
    x_data = x_data.values
    y_data = y_data.values
    x_data = x_data[:, np.newaxis]

    x_test = x_data[:DATA_SET_LENGTH, ]
    y_test = y_data[:DATA_SET_LENGTH]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(None, DIMENSION), dtype=tf.float32))
    model.add(tf.keras.layers.SimpleRNN(100, return_sequences=True))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    train_start_time = time.time()
    history = model.fit(x_data, y_data, epochs=EPOCHS, batch_size=32, validation_split=0.2)
    train_time = time.time() - train_start_time

    train_loss = history.history['loss']
    train_mse = history.history['mean_squared_error']
    train_val_loss = history.history['val_loss']
    train_val_mse = history.history['val_mean_squared_error']

    eva_loss, eva_accuracy = model.evaluate(x_data, y_data)

    test_start_time = time.time()
    y_predict = model.predict(x_test)
    test_time = time.time() - test_start_time

    y_test = list(np.squeeze(np.transpose(y_test)))
    y_predict = list(np.squeeze(np.transpose(y_predict)))

    model.save("train_model.h5")
    model_size = os.path.getsize("train_model.h5")

    return [y_test, y_predict, train_loss, train_mse, train_val_loss, train_val_mse, [eva_loss], [eva_accuracy],
            [train_time], [test_time], [model_size]]


def train_rnn(index, file_name):
    file_name = file_name
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name)

    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

    x_train, y_train, _ = read_data_from_database(index, 0, [200, 220, 250])
    for i_index in range(TRAIN_TIMES):
        result = rnn(x_train, y_train)

        if i_index == 0:
            file_path_y = os.path.join(path, file_name + "_y.csv")
            with open(file_path_y, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result[0])

        best_record_csv(path, file_name, result[1:])


def lstm(x_data, y_data) -> object:
    x_data = x_data.values
    y_data = y_data.values
    x_data = x_data[:, np.newaxis]

    x_test = x_data[:DATA_SET_LENGTH, ]
    y_test = y_data[:DATA_SET_LENGTH]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(None, DIMENSION), dtype=tf.float32))
    model.add(tf.keras.layers.LSTM(100, return_sequences=True))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    train_start_time = time.time()
    history = model.fit(x_data, y_data, epochs=EPOCHS, batch_size=32, validation_split=0.2)
    train_time = time.time() - train_start_time

    train_loss = history.history['loss']
    train_mse = history.history['mean_squared_error']
    train_val_loss = history.history['val_loss']
    train_val_mse = history.history['val_mean_squared_error']

    eva_loss, eva_accuracy = model.evaluate(x_data, y_data)

    test_start_time = time.time()
    y_predict = model.predict(x_test)
    test_time = time.time() - test_start_time

    y_test = list(np.squeeze(np.transpose(y_test)))
    y_predict = list(np.squeeze(np.transpose(y_predict)))

    model.save("train_model.h5")
    model_size = os.path.getsize("train_model.h5")

    return [y_test, y_predict, train_loss, train_mse, train_val_loss, train_val_mse, [eva_loss], [eva_accuracy],
            [train_time], [test_time], [model_size]]


def train_lstm(index, file_name):
    file_name = file_name
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name)

    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

    x_train, y_train, _ = read_data_from_database(index, 0, [200, 220, 250])
    for i_index in range(TRAIN_TIMES):
        result = lstm(x_train, y_train)

        if i_index == 0:
            file_path_y = os.path.join(path, file_name + "_y.csv")
            with open(file_path_y, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result[0])

        best_record_csv(path, file_name, result[1:])


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, ):
        super(SelfAttention, self).__init__()
        self.dimension = DIMENSION

        self.wq = tf.keras.layers.Dense(DIMENSION, activation=tf.keras.activations.relu)
        self.wk = tf.keras.layers.Dense(DIMENSION, activation=tf.keras.activations.relu)
        self.wv = tf.keras.layers.Dense(DIMENSION, activation=tf.keras.activations.relu)

        self.dense = tf.keras.layers.Dense(DIMENSION)

    @staticmethod
    def scaled_dot_product_attention(q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_log = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_log, axis=-1)
        out = tf.matmul(attention_weights, v)

        return out, attention_weights

    def call(self, x, **kwargs):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        out = self.dense(scaled_attention)

        return out


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.eps = 1e-6

    def call(self, x, **kwargs):
        mean = tf.math.reduce_mean(x, axis=0, keepdims=True)
        std = tf.math.reduce_std(x, axis=0, keepdims=True)
        out = (x - mean) / (std + self.eps)

        return out


def point_wise_feed_forward_network(n_nodes):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(n_nodes, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(DIMENSION, activation=tf.keras.activations.relu)
    ])


class BERT(tf.keras.layers.Layer):
    def __init__(self, n_nodes):
        super(BERT, self).__init__()
        self.attention = SelfAttention()
        self.norm = LayerNorm()
        self.nn = point_wise_feed_forward_network(n_nodes)

    def call(self, x, **kwargs):
        out = self.attention(x)
        out = self.norm(x + out)
        out = self.nn(out)

        return out


def bert(x_data, y_data) -> object:
    x_data = x_data.values
    y_data = y_data.values

    x_test = x_data[:DATA_SET_LENGTH, ]
    y_test = y_data[:DATA_SET_LENGTH]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(DIMENSION, dtype=tf.float32))
    model.add(BERT(85))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    train_start_time = time.time()
    history = model.fit(x_data, y_data, epochs=EPOCHS, batch_size=32, validation_split=0.2)
    train_time = time.time() - train_start_time

    train_loss = history.history['loss']
    train_mse = history.history['mean_squared_error']
    train_val_loss = history.history['val_loss']
    train_val_mse = history.history['val_mean_squared_error']

    eva_loss, eva_accuracy = model.evaluate(x_data, y_data)

    test_start_time = time.time()
    y_predict = model.predict(x_test)
    test_time = time.time() - test_start_time

    y_test = list(np.squeeze(np.transpose(y_test)))
    y_predict = list(np.squeeze(np.transpose(y_predict)))

    # model.save("train_model.h5")
    # model_size = os.path.getsize("train_model.h5")

    return [y_test, y_predict, train_loss, train_mse, train_val_loss, train_val_mse, [eva_loss], [eva_accuracy],
            [train_time], [test_time], [0]]


def train_bert(index, file_name):
    file_name = file_name
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name)

    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

    x_train, y_train, _ = read_data_from_database(index, 0, [200, 220, 250])
    for i_index in range(TRAIN_TIMES):
        result = bert(x_train, y_train)

        if i_index == 0:
            file_path_y = os.path.join(path, file_name + "_y.csv")
            with open(file_path_y, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result[0])

        best_record_csv(path, file_name, result[1:])
