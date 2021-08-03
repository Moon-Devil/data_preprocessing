from Data_Reduction_paper.IO_function import *
import tensorflow as tf
import time


def divide_data_by_window(x, y, time_windows) -> object:
    i_index = 0
    i_length = 151
    start_index = 20 - time_windows
    end_index = 20

    while i_index <= i_length:
        array_start_index = start_index + i_index
        array_end_index = end_index + i_index
        x_output = x[array_start_index: array_end_index, ]
        y_output = y[array_start_index: array_end_index]
        i_index = i_index + 1
        yield x_output, y_output


def NeuralNetwork_train(x, y, time_windows, power, parameter):
    file_name = "NeuralNetwork_" + "TimeWindows" + str(time_windows) + "_train_0_mse_" + parameter + "_" + power
    clear_file(file_name)
    x_origin = x[20 - time_windows: 20, ]
    y_origin = y[20 - time_windows: 20]
    x_dimension = (np.shape(x))[1]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(x_dimension, activation=tf.keras.activations.relu, input_shape=(x_dimension, )))
    model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.relu))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    for x_data, y_data in divide_data_by_window(x, y, time_windows):
        model.fit(x_data, y_data, epochs=1)
        y_predict = model.predict(x_origin)
        mse = np.sqrt((y_predict[2] - y_origin[2]) ** 2)
        record_list = mse
        write_to_text(file_name, record_list, "a+")

    file_name = "NeuralNetwork_predict_" + "TimeWindows" + str(time_windows) + "_" + parameter + "_" + power
    clear_file(file_name)
    index = 20
    for x_data, _ in divide_data_by_window(x, y, time_windows):
        y_true = y[index]
        start_predict_time = time.time()
        y_predict = model.predict(x_data)
        end_predict_time = time.time()
        predict_time = end_predict_time - start_predict_time
        error = np.abs(y_true - y_predict[2]) / y_true
        record_list = [y_true, y_predict[2][0], error[0], predict_time]
        write_to_text(file_name, record_list, "a+")
        index = index + 1
