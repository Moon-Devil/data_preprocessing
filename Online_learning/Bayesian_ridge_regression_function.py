import numpy as np
from sklearn.linear_model import BayesianRidge
import time
from sklearn.metrics import mean_squared_error
import os


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

    x_0_test, y_0_test = divide_data_by_window(x, y, 0, time_window)

    br = BayesianRidge()
    length = len(x)
    for i_value in np.arange(time_window - 1, length):
        x_train_step, y_train_step = divide_data_by_window(x, y, i_value, time_window)

        train_start_time = time.time_ns()
        br.fit(x_train_step, y_train_step)
        train_end_time = time.time_ns()
        train_time = train_end_time - train_start_time
        train_time_list.append(train_time)

        predict_start_time = time.time_ns()
        y_0_predict = br.predict(x_0_test)
        predict_end_time = time.time_ns()
        predict_time = predict_end_time - predict_start_time
        predict_time_list.append(predict_time)

        y_0_mse = mean_squared_error(y_0_test, y_0_predict)
        y_0_mse_list.append(y_0_mse)

    y_predict_start_time = time.time_ns()
    y_predict = br.predict(x)
    y_predict_end_time = time.time_ns()
    y_predict_time = y_predict_end_time - y_predict_start_time

    file = path + "\\power decreasing " + data_name + "_time window " + str(time_window) + ".txt"
    if os.path.exists(file):
        os.remove(file)

    recode_result(y, 'y_true', file)
    recode_result(y_predict, 'y_predict', file)
    recode_result(y_0_mse_list, 'mse', file)
    recode_result(train_time_list, 'train_time', file)
    recode_result(predict_time_list, 'predict_time', file)

    with open(file, 'a+') as f:
        f.write("y_predict_time" + '\t' + str(y_predict_time))
