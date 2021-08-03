from IO_function import *
from xgboost import XGBClassifier
import time


def data_generator() -> object:
    normal_data, accident_data = read_data()
    normal_data = normal_data.values
    accident_data = accident_data.values

    x_train = normal_data
    x_test = x_train[:1000, ]
    length = len(x_train)
    y_train = np.zeros(length)
    y_test = np.zeros(1000)

    for i_index in range(6):
        start_index = i_index * 25
        end_index = (i_index + 1) * 25

        x_train = np.vstack((x_train, accident_data[:, start_index: end_index]))
        x_test = np.vstack((x_test, accident_data[:1000, start_index: end_index]))
        y_train = np.hstack((y_train, np.full(length, (i_index + 1))))
        y_test = np.hstack((y_test, np.full(1000, (i_index + 1))))

    return x_train, y_train, x_test, y_test


def Xgboost_function(learning_rate, number_estimator):
    x_train, y_train, _, _ = data_generator()
    xgboost = XGBClassifier(learning_rate=learning_rate, n_estimators=number_estimator, objective='multi:softmax',
                            random_state=0, silent=True)

    xgboost.fit(x_train, y_train)

    start_predict = time.time()
    y_predict = xgboost.predict(x_train)
    predict_time = time.time() - start_predict

    clear_file("a_index")
    clear_file("time_accuracy")
    for i_index in range(7):
        base_index = i_index * 6000
        for j_index in range(5):
            start_index = base_index + j_index * 301
            end_index = base_index + (j_index + 1) * 301
            positive = 0
            negative = 0
            accuracy = np.zeros(301)
            index_0 = 0
            for k_index in range(301):
                if y_predict[end_index - k_index - 1] == y_train[end_index - k_index - 1]:
                    positive += 1
                else:
                    negative += 1

                amount = float(index_0 + 1)
                accuracy[301 - index_0 - 1] = positive * 100 / amount
                index_0 += 1
                write_to_text("a_index", [start_index, end_index, k_index, end_index - k_index - 1], "a+")
            write_to_text("time_accuracy", accuracy.tolist(), "a+")


Xgboost_function(0.05, 1)
