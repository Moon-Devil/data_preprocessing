import tensorflow as tf
import time
from IO_function import *


def NN_function(x_dataframe, y_dataframe, layers, nodes, epochs, stop_flag, stop_patience):
    x_train = x_dataframe.values
    y_train = y_dataframe.values
    dimensions = np.shape(x_train)[1]

    x_test = x_train[: data_set_length, ]
    y_test = y_train[: data_set_length, ]

    start_train = time.time()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(dimensions, name="input_layer"))
    for index in np.arange(layers):
        model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, name="hidden_layer_" +
                                                                                          str(index + 1)))
    model.add(tf.keras.layers.Dense(1, name="output_layer"))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    if stop_flag == 1:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience,
                                                                        verbose=2)])
    else:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2)
    end_train = time.time()

    start_predict = time.time()
    y_predict_temp = model.predict(x_test)
    end_predict = time.time()
    y_true = []
    y_predict = []

    length = len(y_test)
    for index in np.arange(length):
        y_true.append(y_test[index, 0])
        y_predict.append(y_predict_temp[index, 0])

    train_time = end_train - start_train
    predict_time = end_predict - start_predict

    return history, y_true, y_predict, train_time, predict_time


def LSTM_function(x_dataframe, y_dataframe, layers=2, nodes=20, epochs=10, stop_flag=1, stop_patience=2) -> object:
    x_data = x_dataframe.values
    y_data = y_dataframe.values
    dimension = np.shape(x_data)[1]

    x_train = x_data[:, np.newaxis]
    x_test = x_data[0: data_set_length, ]
    x_test = x_test[:, np.newaxis]

    y_train = y_data
    y_test = y_data[0: data_set_length]

    start_train = time.time()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(None, dimension), dtype=tf.float32, name="Input_layer"))

    for _ in np.arange(layers):
        model.add(tf.keras.layers.LSTM(nodes, return_sequences=True))

    model.add(tf.keras.layers.Dense(1, name="Output_layer"))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])
    model.summary()

    if stop_flag == 1:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=8, validation_split=0.2,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience,
                                                                        verbose=2)])
    else:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2)
    end_train = time.time()

    start_predict = time.time()
    y_predict = model.predict(x_test)
    end_predict = time.time()
    y_true = y_test

    train_time = end_train - start_train
    predict_time = end_predict - start_predict

    y_true = np.squeeze(y_true)
    y_predict = np.squeeze(y_predict)

    return history, y_true, y_predict, train_time, predict_time


