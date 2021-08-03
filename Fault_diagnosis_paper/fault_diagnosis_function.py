from IO_function import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc


def data_generator(normal_data, accident_data, index) -> object:
    if index == 0:
        x_data = normal_data
        x_test = normal_data[: 1000, ]
        y_data = np.full(len(normal_data), 1)
        y_test = np.full(1000, 1)

        for i_index in range(6):
            start_index = i_index * 25
            end_index = (i_index + 1) * 25
            x_data_temp = accident_data[:, start_index: end_index]
            x_test_temp = accident_data[: 1000, start_index: end_index]
            y_data_temp = np.zeros(len(x_data_temp))
            y_test_temp = np.zeros(1000)
            x_data = np.vstack((x_data, x_data_temp))
            x_test = np.vstack((x_test, x_test_temp))
            y_data = np.hstack((y_data, y_data_temp))
            y_test = np.hstack((y_test, y_test_temp))
    else:
        start_index = (index - 1) * 25
        end_index = index * 25
        x_data = accident_data[:, start_index: end_index]
        x_test = accident_data[: 1000, start_index: end_index]
        y_data = np.full(len(x_data), 1)
        y_test = np.full(1000, 1)

        x_data_temp = normal_data
        x_test_temp = normal_data[: 1000, ]
        y_data_temp = np.zeros(len(normal_data))
        y_test_temp = np.zeros(1000)

        x_data = np.vstack((x_data, x_data_temp))
        x_test = np.vstack((x_test, x_test_temp))
        y_data = np.hstack((y_data, y_data_temp))
        y_test = np.hstack((y_test, y_test_temp))

        index_list = [1, 2, 3, 4, 5, 6]
        index_list.remove(index)

        for i_index in index_list:
            start_index = (i_index - 1) * 25
            end_index = i_index * 25
            x_data_temp = accident_data[:, start_index: end_index]
            x_test_temp = accident_data[: 1000, start_index: end_index]
            y_data_temp = np.zeros(len(x_data_temp))
            y_test_temp = np.zeros(1000)
            x_data = np.vstack((x_data, x_data_temp))
            x_test = np.vstack((x_test, x_test_temp))
            y_data = np.hstack((y_data, y_data_temp))
            y_test = np.hstack((y_test, y_test_temp))

    return x_data, y_data, x_test, y_test


def dataSet(accident_name) -> object:
    normal_data, accident_data = read_data()
    normal_data = normal_data.values
    accident_data = accident_data.values

    name = accident_name.split('@')
    accident_name = name[1]

    if accident_name == "PowerR":
        x_data, y_data, x_test, y_test = data_generator(normal_data, accident_data, 0)
    elif accident_name == "PRLL":
        x_data, y_data, x_test, y_test = data_generator(normal_data, accident_data, 1)
    elif accident_name == "PRSL":
        x_data, y_data, x_test, y_test = data_generator(normal_data, accident_data, 2)
    elif accident_name == "CL_LOCA":
        x_data, y_data, x_test, y_test = data_generator(normal_data, accident_data, 3)
    elif accident_name == "HL_LOCA":
        x_data, y_data, x_test, y_test = data_generator(normal_data, accident_data, 4)
    elif accident_name == "SG2L":
        x_data, y_data, x_test, y_test = data_generator(normal_data, accident_data, 5)
    else:
        x_data, y_data, x_test, y_test = data_generator(normal_data, accident_data, 6)

    return x_data, y_data, x_test, y_test


def ROC_function(y_true, y_predict, y_probability) -> object:
    TP, FN, FP, TN = 0, 0, 0, 0
    length = len(y_predict)
    for i_index in range(length):
        if y_predict[i_index] == 1 and y_true[i_index] == 1:
            TP = TP + 1
        elif y_predict[i_index] == 0 and y_true[i_index] == 1:
            FN = FN + 1
        elif y_predict[i_index] == 1 and y_true[i_index] == 0:
            FP = FP + 1
        elif y_predict[i_index] == 0 and y_true[i_index] == 0:
            TN = TN + 1
        else:
            exit()

    fpr, tpr, threshold = roc_curve(y_true, y_probability, pos_label=1)
    roc = list(zip(fpr, tpr))
    auc_value = auc(fpr, tpr)

    accident_classification = []
    right, wrong = 0, 0
    for i_index in range(length):
        if i_index % 1000 == 0 and i_index is not 0:
            accident_classification.append([right, wrong])
            right, wrong = 0, 0
        if y_predict[i_index] == 1 and y_true[i_index] == 1:
            right += 1
        elif y_predict[i_index] == 0 and y_true[i_index] == 1:
            wrong += 1
        elif y_predict[i_index] == 0 and y_true[i_index] == 0:
            right += 1
        elif y_predict[i_index] == 1 and y_true[i_index] == 0:
            wrong += 1
        else:
            exit()
    accident_classification.append([right, wrong])

    return [[TP], [FN], [FP], [TN]], roc, auc_value, accident_classification


def GBDT_LR_function(accident_name) -> object:
    x_data, y_data, x_test, y_test = dataSet(accident_name)

    model = GradientBoostingClassifier(n_estimators=30)
    model.fit(x_data, y_data)
    feature = model.apply(x_data).reshape(-1, 30)

    enc = OneHotEncoder()
    enc.fit(feature)

    feature = np.array(enc.transform(feature).toarray())
    lr = LogisticRegression()
    lr.fit(feature, y_data)

    start_predict = time.time()
    predict_feature = model.apply(x_test).reshape(-1, 30)
    predict_feature = np.array(enc.transform(predict_feature).toarray())
    y_predict = lr.predict(predict_feature)
    end_predict = time.time()
    predict_time = end_predict - start_predict

    y_probability = lr.predict_proba(predict_feature)
    result, roc, auc_value, accuracy = ROC_function(y_test, y_predict, y_probability[:, 1])

    x_acc = x_test[: 301, ]
    y_acc_predict = y_predict[:301, ]
    y_acc_true = y_test[:301, ]

    x_no_acc = x_test[1000: 1301, ]
    y_no_acc_predict = y_predict[1000: 1301, ]
    y_no_acc_true = y_test[1000: 1301, ]

    return result, roc, auc_value, accuracy, x_acc, y_acc_predict, y_acc_true, x_no_acc, y_no_acc_predict, \
           y_no_acc_true, predict_time


def is_accident_GBDT_LR(accident_name):
    result, roc, auc_value, accuracy, x_acc, y_acc_predict, y_acc_true, x_no_acc, y_no_acc_predict, y_no_acc_true, \
    predict_time = GBDT_LR_function(accident_name)

    filename_accuracy = accident_name + "_accuracy"
    filename_result = accident_name + "_result"
    filename_roc = accident_name + "_roc"
    filename_time = accident_name + "_time"
    filename_auc = accident_name + "_auc"
    clear_file(filename_accuracy)
    clear_file(filename_result)
    clear_file(filename_roc)
    clear_file(filename_time)
    clear_file(filename_auc)

    record_lists(filename_accuracy, accuracy, "a+")
    record_lists(filename_result, result, "a+")
    record_lists(filename_roc, roc, "a+")
    record_lists(filename_time, [[predict_time], [0]], "a+")
    record_lists(filename_auc, [[auc_value], [0]], "a+")

    filename_accident = accident_name + "_accident"
    clear_file(filename_accident)
    length = len(x_acc)
    for i_index in range(length):
        inputs = [x_acc[i_index][j_index] for j_index in range(len(x_acc[i_index]))]
        inputs.append(y_acc_true[i_index])
        inputs.append(y_acc_predict[i_index])
        write_to_text(filename_accident, inputs, "a+")

    filename_no_accident = accident_name + "_no_accident"
    clear_file(filename_no_accident)
    length = len(x_no_acc)
    for i_index in range(length):
        inputs = [x_no_acc[i_index][j_index] for j_index in range(len(x_no_acc[i_index]))]
        inputs.append(y_no_acc_true[i_index])
        inputs.append(y_no_acc_predict[i_index])
        write_to_text(filename_no_accident, inputs, "a+")


def SVM_function(accident_name) -> object:
    x_data, y_data, x_test, y_test = dataSet(accident_name)
    model = SVC(C=10, probability=True)
    model.fit(x_data, y_data)

    start_predict = time.time()
    y_predict = model.predict(x_test)
    end_predict = time.time()
    predict_time = end_predict - start_predict

    y_probability = model.decision_function(x_test)
    result, roc, auc_value, accuracy = ROC_function(y_test, y_predict, y_probability)

    x_acc = x_test[: 301, ]
    y_acc_predict = y_predict[:301, ]
    y_acc_true = y_test[:301, ]

    x_no_acc = x_test[1000: 1301, ]
    y_no_acc_predict = y_predict[1000: 1301, ]
    y_no_acc_true = y_test[1000: 1301, ]

    return result, roc, auc_value, accuracy, x_acc, y_acc_predict, y_acc_true, x_no_acc, y_no_acc_predict, \
           y_no_acc_true, predict_time


def is_accident_SVM(accident_name):
    result, roc, auc_value, accuracy, x_acc, y_acc_predict, y_acc_true, x_no_acc, y_no_acc_predict, y_no_acc_true, \
    predict_time = SVM_function(accident_name)

    filename_accuracy = accident_name + "_accuracy"
    filename_result = accident_name + "_result"
    filename_roc = accident_name + "_roc"
    filename_time = accident_name + "_time"
    filename_auc = accident_name + "_auc"
    clear_file(filename_accuracy)
    clear_file(filename_result)
    clear_file(filename_roc)
    clear_file(filename_time)
    clear_file(filename_auc)

    record_lists(filename_accuracy, accuracy, "a+")
    record_lists(filename_result, result, "a+")
    record_lists(filename_roc, roc, "a+")
    record_lists(filename_time, [[predict_time], [0]], "a+")
    record_lists(filename_auc, [[auc_value], [0]], "a+")

    filename_accident = accident_name + "_accident"
    clear_file(filename_accident)
    length = len(x_acc)
    for i_index in range(length):
        inputs = [x_acc[i_index][j_index] for j_index in range(len(x_acc[i_index]))]
        inputs.append(y_acc_true[i_index])
        inputs.append(y_acc_predict[i_index])
        write_to_text(filename_accident, inputs, "a+")

    filename_no_accident = accident_name + "_no_accident"
    clear_file(filename_no_accident)
    length = len(x_no_acc)
    for i_index in range(length):
        inputs = [x_no_acc[i_index][j_index] for j_index in range(len(x_no_acc[i_index]))]
        inputs.append(y_no_acc_true[i_index])
        inputs.append(y_no_acc_predict[i_index])
        write_to_text(filename_no_accident, inputs, "a+")


def Adaboost_function(accident_name) -> object:
    x_data, y_data, x_test, y_test = dataSet(accident_name)
    model = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME", n_estimators=10)
    model.fit(x_data, y_data)

    start_predict = time.time()
    y_predict = model.predict(x_test)
    end_predict = time.time()
    predict_time = end_predict - start_predict

    y_probability = model.predict_proba(x_test)
    result, roc, auc_value, accuracy = ROC_function(y_test, y_predict, y_probability[:, 1])

    x_acc = x_test[: 301, ]
    y_acc_predict = y_predict[:301, ]
    y_acc_true = y_test[:301, ]

    x_no_acc = x_test[1000: 1301, ]
    y_no_acc_predict = y_predict[1000: 1301, ]
    y_no_acc_true = y_test[1000: 1301, ]

    return result, roc, auc_value, accuracy, x_acc, y_acc_predict, y_acc_true, x_no_acc, y_no_acc_predict, \
           y_no_acc_true, predict_time


def is_accident_Adaboost(accident_name):
    result, roc, auc_value, accuracy, x_acc, y_acc_predict, y_acc_true, x_no_acc, y_no_acc_predict, y_no_acc_true, \
    predict_time = Adaboost_function(accident_name)

    filename_accuracy = accident_name + "_accuracy"
    filename_result = accident_name + "_result"
    filename_roc = accident_name + "_roc"
    filename_time = accident_name + "_time"
    filename_auc = accident_name + "_auc"
    clear_file(filename_accuracy)
    clear_file(filename_result)
    clear_file(filename_roc)
    clear_file(filename_time)
    clear_file(filename_auc)

    record_lists(filename_accuracy, accuracy, "a+")
    record_lists(filename_result, result, "a+")
    record_lists(filename_roc, roc, "a+")
    record_lists(filename_time, [[predict_time], [0]], "a+")
    record_lists(filename_auc, [[auc_value], [0]], "a+")

    filename_accident = accident_name + "_accident"
    clear_file(filename_accident)
    length = len(x_acc)
    for i_index in range(length):
        inputs = [x_acc[i_index][j_index] for j_index in range(len(x_acc[i_index]))]
        inputs.append(y_acc_true[i_index])
        inputs.append(y_acc_predict[i_index])
        write_to_text(filename_accident, inputs, "a+")

    filename_no_accident = accident_name + "_no_accident"
    clear_file(filename_no_accident)
    length = len(x_no_acc)
    for i_index in range(length):
        inputs = [x_no_acc[i_index][j_index] for j_index in range(len(x_no_acc[i_index]))]
        inputs.append(y_no_acc_true[i_index])
        inputs.append(y_no_acc_predict[i_index])
        write_to_text(filename_no_accident, inputs, "a+")


def kNN_function(accident_name) -> object:
    x_data, y_data, x_test, y_test = dataSet(accident_name)
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(x_data, y_data)

    start_predict = time.time()
    y_predict = model.predict(x_test)
    end_predict = time.time()
    predict_time = end_predict - start_predict

    y_probability = model.predict_proba(x_test)
    result, roc, auc_value, accuracy = ROC_function(y_test, y_predict, y_probability[:, 1])

    x_acc = x_test[: 301, ]
    y_acc_predict = y_predict[:301, ]
    y_acc_true = y_test[:301, ]

    x_no_acc = x_test[1000: 1301, ]
    y_no_acc_predict = y_predict[1000: 1301, ]
    y_no_acc_true = y_test[1000: 1301, ]

    return result, roc, auc_value, accuracy, x_acc, y_acc_predict, y_acc_true, x_no_acc, y_no_acc_predict, \
           y_no_acc_true, predict_time


def is_accident_kNN(accident_name):
    result, roc, auc_value, accuracy, x_acc, y_acc_predict, y_acc_true, x_no_acc, y_no_acc_predict, y_no_acc_true, \
    predict_time = kNN_function(accident_name)

    filename_accuracy = accident_name + "_accuracy"
    filename_result = accident_name + "_result"
    filename_roc = accident_name + "_roc"
    filename_time = accident_name + "_time"
    filename_auc = accident_name + "_auc"
    clear_file(filename_accuracy)
    clear_file(filename_result)
    clear_file(filename_roc)
    clear_file(filename_time)
    clear_file(filename_auc)

    record_lists(filename_accuracy, accuracy, "a+")
    record_lists(filename_result, result, "a+")
    record_lists(filename_roc, roc, "a+")
    record_lists(filename_time, [[predict_time], [0]], "a+")
    record_lists(filename_auc, [[auc_value], [0]], "a+")

    filename_accident = accident_name + "_accident"
    clear_file(filename_accident)
    length = len(x_acc)
    for i_index in range(length):
        inputs = [x_acc[i_index][j_index] for j_index in range(len(x_acc[i_index]))]
        inputs.append(y_acc_true[i_index])
        inputs.append(y_acc_predict[i_index])
        write_to_text(filename_accident, inputs, "a+")

    filename_no_accident = accident_name + "_no_accident"
    clear_file(filename_no_accident)
    length = len(x_no_acc)
    for i_index in range(length):
        inputs = [x_no_acc[i_index][j_index] for j_index in range(len(x_no_acc[i_index]))]
        inputs.append(y_no_acc_true[i_index])
        inputs.append(y_no_acc_predict[i_index])
        write_to_text(filename_no_accident, inputs, "a+")
