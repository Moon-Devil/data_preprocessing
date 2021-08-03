import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import mean_squared_error
import os
from tqdm import tqdm


grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\online learning'
if not os.path.exists(path):
    os.mkdir(path)


def divide_data_by_window(x, y, index, time_widow) -> object:
    x_list = []
    y_list = []
    start_index = index + 1 - time_widow
    for i_index in np.arange(start_index, index + 1):
        x_list.append(x[i_index])
        y_list.append(y[i_index])
    x_array = np.array(x_list)
    y_window = np.array(y_list)
    x_window = x_array.reshape(time_widow, -1)

    return x_window, y_window


def recode_result(x, x_name, file):
    length = len(x)
    with open(file, 'a+') as f:
        f.write(x_name + '\t')
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(x[i_value]) + ',')
            else:
                f.write(str(x[i_value]) + '\n')


def train_model(x, y, time_window, data_name):
    train_time_list = []
    predict_time_list = []
    y_0_mse_list = []
    history_epoch = 0

    x_0_test, y_0_test = divide_data_by_window(x, y, 0, time_window)
    input_shape = np.shape(x_0_test)
    data_number = input_shape[0]
    data_dimension = input_shape[1]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu,
                                    input_shape=(data_dimension, )))
    model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    length = len(x)
    for i_value in tqdm(np.arange(time_window - 1, length)):
        x_train_step, y_train_step = divide_data_by_window(x, y, i_value, time_window)

        train_start_time = time.time_ns()
        history_epoch = model.fit(x_train_step, y_train_step, epochs=200, batch_size=time_window, validation_split=0.2,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
        train_end_time = time.time_ns()
        train_time = train_end_time - train_start_time
        train_time_list.append(train_time)

        predict_start_time = time.time_ns()
        y_0_predict = model.predict(x_0_test)
        predict_end_time = time.time_ns()
        predict_time = predict_end_time - predict_start_time
        predict_time_list.append(predict_time)

        y_0_mse = mean_squared_error(y_0_test, y_0_predict)
        y_0_mse_list.append(y_0_mse)

    history = history_epoch
    y_predict_start_time = time.time_ns()
    y_predict_temp = model.predict(x)

    y_predict = []
    length = len(y_predict_temp)
    for i_value in np.arange(length):
        y_predict.append(y_predict_temp[i_value][0])
    y_predict = np.array(y_predict)

    y_predict_end_time = time.time_ns()
    y_predict_time = y_predict_end_time - y_predict_start_time

    file = path + "\\NN power decreasing " + data_name + "_time window " + str(time_window) + ".txt"
    if os.path.exists(file):
        os.remove(file)

    recode_result(y, 'y_true', file)
    recode_result(y_predict, 'y_predict', file)
    recode_result(y_0_mse_list, 'mse', file)
    recode_result(train_time_list, 'train_time', file)
    recode_result(predict_time_list, 'predict_time', file)

    recode_result(history.history['loss'], 'loss', file)
    recode_result(history.history['mean_squared_error'], 'mse', file)
    recode_result(history.history['val_loss'], 'val_loss', file)
    recode_result(history.history['val_mean_squared_error'], 'val_mse', file)

    with open(file, 'a+') as f:
        f.write("y_predict_time" + '\t' + str(y_predict_time))
