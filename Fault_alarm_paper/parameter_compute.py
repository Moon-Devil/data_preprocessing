import shutil
import tensorflow as tf
import time
from IO_function import *
import csv

TRAIN_TIMES = 50
EPOCHS = 201
DIMENSION = 26
DATA_SET_LENGTH = 301


def dnn_sg(x_data, y_data) -> object:
    x_data = x_data.values[:, 0: 24]
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

    model.save("sg_train_model.h5")
    model_size = os.path.getsize("sg_train_model.h5")

    return [y_test, y_predict, train_loss, train_mse, train_val_loss, train_val_mse, [eva_loss], [eva_accuracy],
            [train_time], [test_time], [model_size]]


def train_dnn_sg(index, file_name):
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
        result = dnn_sg(x_train, y_train)

        if i_index == 0:
            file_path_y = os.path.join(path, file_name + "_y.csv")
            with open(file_path_y, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result[0])

        best_record_csv(path, file_name, result[1:])


def dnn_pr(x_data, y_data) -> object:
    x_data = x_data.values[:, 0: 24]
    y_data = y_data.values

    x_test = x_data[:DATA_SET_LENGTH, ]
    y_test = y_data[:DATA_SET_LENGTH]

    model = tf.keras.models.Sequential()
    for i_index in range(13):
        model.add(tf.keras.layers.Dense(91, activation=tf.keras.activations.relu))
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

    model.save("pr_train_model.h5")
    model_size = os.path.getsize("pr_train_model.h5")

    return [y_test, y_predict, train_loss, train_mse, train_val_loss, train_val_mse, [eva_loss], [eva_accuracy],
            [train_time], [test_time], [model_size]]


def train_dnn_pr(index, file_name):
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
        result = dnn_pr(x_train, y_train)

        if i_index == 0:
            file_path_y = os.path.join(path, file_name + "_y.csv")
            with open(file_path_y, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result[0])

        best_record_csv(path, file_name, result[1:])


# train_dnn_sg(16, "sg_best_model")
train_dnn_pr(24, "pr_best_model")
