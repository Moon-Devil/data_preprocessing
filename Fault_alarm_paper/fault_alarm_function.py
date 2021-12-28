import time
from SparseAutoEncoder import *
from VariationalAutoEncoder import *
import pickle


def sg_sae_fault_alarm(detection_values, measured_values, condition_data, actual_flags, file_name):
    start_load_time = time.time()
    sg_model = tf.keras.models.load_model('sg_train_model.h5')
    sae_model = SparseAutoEncoder(24, 25)
    sae_model.build((None, 25))
    sae_model.load_weights('sae_train_model.h5')

    with open('sae_iForest.pk', 'rb') as f:
        iForest_model = pickle.load(f)
    load_time = time.time() - start_load_time

    encoder_values = sae_model.encoder(condition_data)
    isNormal = iForest_model.predict(encoder_values)

    isRight = []
    start_predict_time = time.time()
    y_predict = sg_model.predict(measured_values)
    predict_time = time.time() - start_predict_time

    length = len(y_predict)
    for i_index in range(length):
        temp = np.abs(detection_values[i_index, ] - y_predict[i_index, ]) * 100 / (detection_values[i_index, ] + 1e-6)
        if temp < 5:
            isRight.append(1)
        else:
            isRight.append(-1)
    isRight = np.array(isRight)

    result = []
    for i_index in range(length):
        if isNormal[i_index] == 1 and isRight[i_index] == 1:
            result.append(0)
        elif isNormal[i_index] == 1 and isRight[i_index] == -1:
            result.append(1)
        else:
            result.append(-1)
    result = np.array(result)

    normal, accident, fault = 0, 0, 0
    for i_index in range(length):
        if result[i_index] == 0:
            normal += 1
        elif result[i_index] == -1:
            accident += 1
        else:
            fault += 1

    if actual_flags == 0:
        accuracy = normal / float(length)
    elif actual_flags == 1:
        accuracy = fault / float(length)
    else:
        accuracy = accident / float(length)

    result_list = [normal, fault, accident, accuracy, load_time, predict_time]

    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name + "_sae_result.csv")
    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['normal', 'fault', 'accident', 'accuracy', 'load_time', 'predict_time'])
        writer.writerow(result_list)

    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name + "_sae_data.csv")
    if os.path.exists(document):
        os.remove(document)

    true_values = np.transpose(detection_values)
    predict_values = np.transpose(y_predict)
    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(true_values[0])
        writer.writerow(predict_values[0])
        writer.writerow(result)


def sg_vae_fault_alarm(detection_values, measured_values, condition_data, actual_flags, file_name):
    start_load_time = time.time()
    sg_model = tf.keras.models.load_model('sg_train_model.h5')
    vae_model = VariationAutoEncoder(17, 25)
    vae_model.build((None, 25))
    vae_model.load_weights('vae_train_model.h5')

    with open('vae_iForest.pk', 'rb') as f:
        iForest_model = pickle.load(f)
    load_time = time.time() - start_load_time

    encoder_values = vae_model.encoder(condition_data)
    isNormal = iForest_model.predict(encoder_values)

    isRight = []
    start_predict_time = time.time()
    y_predict = sg_model.predict(measured_values)
    predict_time = time.time() - start_predict_time

    length = len(y_predict)
    for i_index in range(length):
        temp = np.abs(detection_values[i_index, ] - y_predict[i_index, ]) * 100 / (detection_values[i_index, ] + 1e-6)
        if temp < 5:
            isRight.append(1)
        else:
            isRight.append(-1)
    isRight = np.array(isRight)

    result = []
    for i_index in range(length):
        if isNormal[i_index] == 1 and isRight[i_index] == 1:
            result.append(0)
        elif isNormal[i_index] == 1 and isRight[i_index] == -1:
            result.append(1)
        else:
            result.append(-1)
    result = np.array(result)

    normal, accident, fault = 0, 0, 0
    for i_index in range(length):
        if result[i_index] == 0:
            normal += 1
        elif result[i_index] == -1:
            accident += 1
        else:
            fault += 1

    if actual_flags == 0:
        accuracy = normal / float(length)
    elif actual_flags == 1:
        accuracy = fault / float(length)
    else:
        accuracy = accident / float(length)

    result_list = [normal, fault, accident, accuracy, load_time, predict_time]

    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name + "_vae_result.csv")
    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['normal', 'fault', 'accident', 'accuracy', 'load_time', 'predict_time'])
        writer.writerow(result_list)

    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\" + file_name + "_vae_data.csv")
    if os.path.exists(document):
        os.remove(document)

    true_values = np.transpose(detection_values)
    predict_values = np.transpose(y_predict)
    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(true_values[0])
        writer.writerow(predict_values[0])
        writer.writerow(result)


def sae_normal_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 1000, 0: 24]
    y_data = y_data.values[: 1000]

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_sae_fault_alarm(y_data, x_data, normal_data, 0, "sg_normal_test")


def sae_accident_condition():
    _, anomaly_data = read_data()
    anomaly_data = anomaly_data.values[: 1000, 0: 25]

    y_data = np.transpose(anomaly_data[:, 16])[:, np.newaxis]
    x_data = np.delete(anomaly_data, 16, axis=1)

    sg_sae_fault_alarm(y_data, x_data, anomaly_data, -1, "sg_accident_test")


def sae_fault_upper_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 1000, 0: 24]
    y_data = np.full((1000, 1), 300)

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_sae_fault_alarm(y_data, x_data, normal_data, 1, "sg_fault_upper_test")


def sae_fault_lower_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 1000, 0: 24]
    y_data = np.zeros((1000, 1))

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_sae_fault_alarm(y_data, x_data, normal_data, 1, "sg_fault_lower_test")


def vae_normal_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 1000, 0: 24]
    y_data = y_data.values[: 1000]

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_vae_fault_alarm(y_data, x_data, normal_data, 0, "sg_normal_test")


def vae_accident_condition():
    _, anomaly_data = read_data()
    anomaly_data = anomaly_data.values[: 1000, 0: 25]

    y_data = np.transpose(anomaly_data[:, 16])[:, np.newaxis]
    x_data = np.delete(anomaly_data, 16, axis=1)

    sg_vae_fault_alarm(y_data, x_data, anomaly_data, -1, "sg_accident_test")


def vae_fault_upper_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 1000, 0: 24]
    y_data = np.full((1000, 1), 300)

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_vae_fault_alarm(y_data, x_data, normal_data, 1, "sg_fault_upper_test")


def vae_fault_lower_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 1000, 0: 24]
    y_data = np.zeros((1000, 1))

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_vae_fault_alarm(y_data, x_data, normal_data, 1, "sg_fault_lower_test")


def show_sae_fault_upper_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 301, 0: 24]
    y_data = y_data.values[: 151]
    fault_data = np.full((150, 1), 300)
    y_data = np.vstack((y_data, fault_data))

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_sae_fault_alarm(y_data, x_data, normal_data, 1, "show_sg_fault_upper_test")


def show_sae_fault_lower_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 301, 0: 24]
    y_data = y_data.values[: 151]
    fault_data = np.zeros((150, 1))
    y_data = np.vstack((y_data, fault_data))

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_sae_fault_alarm(y_data, x_data, normal_data, 1, "show_sg_fault_lower_test")


def show_vae_fault_upper_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 301, 0: 24]
    y_data = y_data.values[: 151]
    fault_data = np.full((150, 1), 300)
    y_data = np.vstack((y_data, fault_data))

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_vae_fault_alarm(y_data, x_data, normal_data, 1, "show_sg_fault_upper_test")


def show_vae_fault_lower_condition():
    x_data, y_data, _ = read_data_from_database(16, 0, [200, 220, 250])
    x_data = x_data.values[: 301, 0: 24]
    y_data = y_data.values[: 151]
    fault_data = np.zeros((150, 1))
    y_data = np.vstack((y_data, fault_data))

    normal_data, _ = read_data()
    normal_data = normal_data.values[: 1000, 0: 25]

    sg_vae_fault_alarm(y_data, x_data, normal_data, 1, "show_sg_fault_lower_test")


# sae_normal_condition()
# sae_accident_condition()
# sae_fault_upper_condition()
# sae_fault_lower_condition()

# vae_normal_condition()
# vae_accident_condition()
# vae_fault_upper_condition()
# vae_fault_lower_condition()

show_sae_fault_upper_condition()
show_sae_fault_lower_condition()
show_vae_fault_upper_condition()
show_vae_fault_lower_condition()
