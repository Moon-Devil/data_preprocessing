import tensorflow as tf
from keras_tuner import BayesianOptimization
from IO_function import *
import csv

MAX_TRIALS = 50
EPOCHS = 100
DIMENSION = 26
MAX_N_LAYERS = 100
MIN_N_LAYERS = 1
MAX_N_NODES = 100
MIN_N_NODES = 1


def build_dnn(hp) -> object:
    model = tf.keras.models.Sequential()
    n_layers = hp.Int('n_layers', MIN_N_LAYERS, MAX_N_LAYERS)
    n_nodes = hp.Int('n_nodes', MIN_N_NODES, MAX_N_NODES)
    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(units=n_nodes, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    return model


def best_dnn(x_data, y_data):
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper")
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, "dnn_result.csv")
    if os.path.exists(file_name):
        os.remove(file_name)

    x_data = x_data.values
    y_data = y_data.values

    tuner = BayesianOptimization(build_dnn, objective='loss', max_trials=MAX_TRIALS, directory='dnn_result',
                                 project_name='dnn_search')
    tuner.search(x_data, y_data, epochs=EPOCHS, validation_split=0.2)
    best_hyper = tuner.get_best_hyperparameters(num_trials=1)

    search_dict = tuner.oracle.trials
    length = len(search_dict)
    key_names = list(search_dict.keys())
    f = open(file_name, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)

    for i_index in range(length):
        hyper_parameters_dict = search_dict[key_names[i_index]].hyperparameters.values
        n_layers = hyper_parameters_dict['n_layers']
        n_nodes = hyper_parameters_dict['n_nodes']
        score = search_dict[key_names[i_index]].score

        evaluations_dict = search_dict[key_names[i_index]].metrics.metrics
        eva_names = list(evaluations_dict.keys())
        in_length = len(eva_names)
        evaluations = []
        for j_index in range(in_length):
            evaluations.append((evaluations_dict[eva_names[j_index]]._observations)[0].value[0])

        result = [n_layers, n_nodes, score] + evaluations
        if i_index == 0:
            result_name = ['n_layer', 'n_nodes', 'score', 'loss', 'mse', 'val_loss', 'val_mse']
            csv_writer.writerow(result_name)

        csv_writer.writerow(result)

    f.close()

    file_name = os.path.join(path, "dnn_result_best.csv")
    if os.path.exists(file_name):
        os.remove(file_name)

    best_names = ['n_layers', 'n_nodes']
    n_layers = (best_hyper[0])['n_layers']
    n_nodes = (best_hyper[0])['n_nodes']

    f = open(file_name, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(best_names)
    csv_writer.writerow([n_layers, n_nodes])

    f.close()


def build_rnn(hp) -> object:
    model = tf.keras.models.Sequential()
    n_layers = hp.Int('n_layers', MIN_N_LAYERS, MAX_N_LAYERS)
    n_nodes = hp.Int('n_nodes', MIN_N_NODES, MAX_N_NODES)

    model.add(tf.keras.layers.Input(shape=(None, DIMENSION), dtype=tf.float32))
    for i in range(n_layers):
        model.add(tf.keras.layers.SimpleRNN(units=n_nodes, return_sequences=True))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    return model


def best_rnn(x_data, y_data):
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper")
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, "rnn_result.csv")
    if os.path.exists(file_name):
        os.remove(file_name)

    x_data = x_data.values
    y_data = y_data.values
    x_data = x_data[:, np.newaxis]

    tuner = BayesianOptimization(build_rnn, objective='loss', max_trials=MAX_TRIALS, directory='rnn_result',
                                 project_name='rnn_search')
    tuner.search(x_data, y_data, epochs=EPOCHS, validation_split=0.2)
    best_hyper = tuner.get_best_hyperparameters(num_trials=1)

    search_dict = tuner.oracle.trials
    length = len(search_dict)
    key_names = list(search_dict.keys())
    f = open(file_name, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)

    for i_index in range(length):
        hyper_parameters_dict = search_dict[key_names[i_index]].hyperparameters.values
        n_layers = hyper_parameters_dict['n_layers']
        n_nodes = hyper_parameters_dict['n_nodes']
        score = search_dict[key_names[i_index]].score

        evaluations_dict = search_dict[key_names[i_index]].metrics.metrics
        eva_names = list(evaluations_dict.keys())
        in_length = len(eva_names)
        evaluations = []
        for j_index in range(in_length):
            evaluations.append((evaluations_dict[eva_names[j_index]]._observations)[0].value[0])

        result = [n_layers, n_nodes, score] + evaluations
        if i_index == 0:
            result_name = ['n_layer', 'n_nodes', 'score', 'loss', 'mse', 'val_loss', 'val_mse']
            csv_writer.writerow(result_name)

        csv_writer.writerow(result)

    f.close()

    file_name = os.path.join(path, "rnn_result_best.csv")
    if os.path.exists(file_name):
        os.remove(file_name)

    best_names = ['n_layers', 'n_nodes']
    n_layers = (best_hyper[0])['n_layers']
    n_nodes = (best_hyper[0])['n_nodes']

    f = open(file_name, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(best_names)
    csv_writer.writerow([n_layers, n_nodes])

    f.close()


def build_lstm(hp) -> object:
    model = tf.keras.models.Sequential()
    n_layers = hp.Int('n_layers', MIN_N_LAYERS, MAX_N_LAYERS)
    n_nodes = hp.Int('n_nodes', MIN_N_NODES, MAX_N_NODES)

    model.add(tf.keras.layers.Input(shape=(None, DIMENSION), dtype=tf.float32))
    for i in range(n_layers):
        model.add(tf.keras.layers.LSTM(units=n_nodes, return_sequences=True))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    return model


def best_lstm(x_data, y_data):
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper")
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, "lstm_result.csv")
    if os.path.exists(file_name):
        os.remove(file_name)

    x_data = x_data.values
    y_data = y_data.values
    x_data = x_data[:, np.newaxis]

    tuner = BayesianOptimization(build_lstm, objective='loss', max_trials=MAX_TRIALS, directory='lstm_result',
                                 project_name='lstm_search')
    tuner.search(x_data, y_data, epochs=EPOCHS, validation_split=0.2)
    best_hyper = tuner.get_best_hyperparameters(num_trials=1)

    search_dict = tuner.oracle.trials
    length = len(search_dict)
    key_names = list(search_dict.keys())
    f = open(file_name, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)

    for i_index in range(length):
        hyper_parameters_dict = search_dict[key_names[i_index]].hyperparameters.values
        n_layers = hyper_parameters_dict['n_layers']
        n_nodes = hyper_parameters_dict['n_nodes']
        score = search_dict[key_names[i_index]].score

        evaluations_dict = search_dict[key_names[i_index]].metrics.metrics
        eva_names = list(evaluations_dict.keys())
        in_length = len(eva_names)
        evaluations = []
        for j_index in range(in_length):
            evaluations.append((evaluations_dict[eva_names[j_index]]._observations)[0].value[0])

        result = [n_layers, n_nodes, score] + evaluations
        if i_index == 0:
            result_name = ['n_layer', 'n_nodes', 'score', 'loss', 'mse', 'val_loss', 'val_mse']
            csv_writer.writerow(result_name)

        csv_writer.writerow(result)

    f.close()

    file_name = os.path.join(path, "lstm_result_best.csv")
    if os.path.exists(file_name):
        os.remove(file_name)

    best_names = ['n_layers', 'n_nodes']
    n_layers = (best_hyper[0])['n_layers']
    n_nodes = (best_hyper[0])['n_nodes']

    f = open(file_name, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(best_names)
    csv_writer.writerow([n_layers, n_nodes])

    f.close()


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


def build_bert(hp) -> object:
    model = tf.keras.models.Sequential()
    n_layers = hp.Int('n_layers', MIN_N_LAYERS, MAX_N_LAYERS)
    n_nodes = hp.Int('n_nodes', MIN_N_NODES, MAX_N_NODES)

    model.add(tf.keras.layers.Input(DIMENSION, dtype=tf.float32))
    for i in range(n_layers):
        model.add(BERT(n_nodes))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])
    model.summary()

    return model


def best_bert(x_data, y_data):
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))
    path = os.path.join(grandfather_path, "Calculations\\Fault alarm paper")
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, "bert_result.csv")
    if os.path.exists(file_name):
        os.remove(file_name)

    x_data = x_data.values
    y_data = y_data.values

    tuner = BayesianOptimization(build_bert, objective='loss', max_trials=MAX_TRIALS, directory='bert_result',
                                 project_name='bert_search')
    tuner.search(x_data, y_data, epochs=EPOCHS, validation_split=0.2)
    best_hyper = tuner.get_best_hyperparameters(num_trials=1)

    search_dict = tuner.oracle.trials
    length = len(search_dict)
    key_names = list(search_dict.keys())
    f = open(file_name, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)

    for i_index in range(length):
        hyper_parameters_dict = search_dict[key_names[i_index]].hyperparameters.values
        n_layers = hyper_parameters_dict['n_layers']
        n_nodes = hyper_parameters_dict['n_nodes']
        score = search_dict[key_names[i_index]].score

        evaluations_dict = search_dict[key_names[i_index]].metrics.metrics
        eva_names = list(evaluations_dict.keys())
        in_length = len(eva_names)
        evaluations = []
        for j_index in range(in_length):
            evaluations.append((evaluations_dict[eva_names[j_index]]._observations)[0].value[0])

        result = [n_layers, n_nodes, score] + evaluations
        if i_index == 0:
            result_name = ['n_layer', 'n_nodes', 'score', 'loss', 'mse', 'val_loss', 'val_mse']
            csv_writer.writerow(result_name)

        csv_writer.writerow(result)

    f.close()

    file_name = os.path.join(path, "bert_result_best.csv")
    if os.path.exists(file_name):
        os.remove(file_name)

    best_names = ['n_layers', 'n_nodes']
    n_layers = (best_hyper[0])['n_layers']
    n_nodes = (best_hyper[0])['n_nodes']

    f = open(file_name, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(best_names)
    csv_writer.writerow([n_layers, n_nodes])

    f.close()
