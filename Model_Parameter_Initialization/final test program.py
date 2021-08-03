from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
import tensorflow as tf
import numpy as np
from sklearn import manifold
import os
import time
from sklearn.metrics import mean_squared_error
from scipy import stats


# 设置计算结果存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing finally result'

if not os.path.exists(path):
    os.mkdir(path)

data_dimension = 16
hidden_layers_nodes = 69
learning_rate = 0.0001491072657856008
batch_size = 32
epochs_size = 200
all_epochs = 100

pressurizer_water_y = power_decreasing[..., -3]
pressurizer_water_x = np.delete(power_decreasing, -3, axis=1)

# LLE降维
lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=data_dimension, method='standard')
pressurizer_water_x = lle.fit_transform(pressurizer_water_x)

pressurizer_water_x_test_1 = pressurizer_water_x[:301, ...]
pressurizer_water_y_test_1 = pressurizer_water_y[:301]

pressurizer_water_x_test_2 = pressurizer_water_x[1511:1812, ...]
pressurizer_water_y_test_2 = pressurizer_water_y[1511:1812]

pressurizer_water_x_test_3 = pressurizer_water_x[2659:2960, ...]
pressurizer_water_y_test_3 = pressurizer_water_y[2659:2960]

train_time = []
predict_time_1 = []
predict_time_2 = []
predict_time_3 = []

predict_1_list = []
predict_2_list = []
predict_3_list = []

for epoch in np.arange(all_epochs):
    start_train_time = time.time()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu,
                                    input_shape=(data_dimension, )))
    model.add(tf.keras.layers.Dense(hidden_layers_nodes, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=tf.keras.losses.MAPE,
                  metrics=[tf.keras.losses.MSE])
    history = model.fit(pressurizer_water_x, pressurizer_water_y, batch_size=batch_size, epochs=epochs_size,
                        validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping
                                                         (monitor='val_loss', patience=10)])
    end_train_time = time.time()
    train_time.append(end_train_time - start_train_time)

    start_predict_time_1 = time.time()
    y_predict_1 = model.predict(pressurizer_water_x_test_1)
    end_predict_time_1 = time.time()
    predict_time_1.append(end_predict_time_1 - start_predict_time_1)
    predict_1_list.append(y_predict_1)

    start_predict_time_2 = time.time()
    y_predict_2 = model.predict(pressurizer_water_x_test_2)
    end_predict_time_2 = time.time()
    predict_time_2.append(end_predict_time_2 - start_predict_time_2)
    predict_2_list.append(y_predict_2)

    start_predict_time_3 = time.time()
    y_predict_3 = model.predict(pressurizer_water_x_test_3)
    end_predict_time_3 = time.time()
    predict_time_3.append(end_predict_time_3 - start_predict_time_3)
    predict_3_list.append(y_predict_3)

train_time = np.array(train_time)
predict_time_1 = np.array(predict_time_1)
predict_time_2 = np.array(predict_time_2)
predict_time_3 = np.array(predict_time_3)

predict_1 = np.zeros((all_epochs, 301))
predict_2 = np.zeros((all_epochs, 301))
predict_3 = np.zeros((all_epochs, 301))

for epoch in range(all_epochs):
    for i_value in range(301):
        predict_1[epoch][i_value] = predict_1_list[epoch][i_value][0]
        predict_2[epoch][i_value] = predict_2_list[epoch][i_value][0]
        predict_3[epoch][i_value] = predict_3_list[epoch][i_value][0]

predict_1_mean = np.sum(predict_1, axis=0)/all_epochs
predict_2_mean = np.sum(predict_2, axis=0)/all_epochs
predict_3_mean = np.sum(predict_3, axis=0)/all_epochs

predict_1_mse = []
predict_2_mse = []
predict_3_mse = []

for epoch in np.arange(all_epochs):
    predict_1_mse.append(mean_squared_error(pressurizer_water_y_test_1, predict_1[epoch, ...]))
    predict_2_mse.append(mean_squared_error(pressurizer_water_y_test_2, predict_2[epoch, ...]))
    predict_3_mse.append(mean_squared_error(pressurizer_water_y_test_3, predict_3[epoch, ...]))

predict_1_max_index = predict_1_mse.index(max(predict_1_mse))
predict_2_max_index = predict_2_mse.index(max(predict_2_mse))
predict_3_max_index = predict_3_mse.index(max(predict_3_mse))

predict_1_min_index = predict_1_mse.index(min(predict_1_mse))
predict_2_min_index = predict_2_mse.index(min(predict_2_mse))
predict_3_min_index = predict_3_mse.index(min(predict_3_mse))

predict_1_max = predict_1[predict_1_max_index, ...]
predict_2_max = predict_2[predict_2_max_index, ...]
predict_3_max = predict_3[predict_3_max_index, ...]

predict_1_min = predict_1[predict_1_min_index, ...]
predict_2_min = predict_2[predict_2_min_index, ...]
predict_3_min = predict_3[predict_3_min_index, ...]

ci_1_upper = []
ci_2_upper = []
ci_3_upper = []

ci_1_lower = []
ci_2_lower = []
ci_3_lower = []

for i_value in np.arange(301):
    temp = np.squeeze(np.transpose(predict_1[:, i_value]))
    ci_1 = stats.t.interval(0.95, all_epochs - 1, np.mean(temp), stats.tsem(temp, ddof=1))
    ci_1_upper.append(ci_1[0])
    ci_1_lower.append(ci_1[1])

    temp = np.squeeze(np.transpose(predict_2[:, i_value]))
    ci_2 = stats.t.interval(0.95, all_epochs - 1, np.mean(temp), stats.tsem(temp, ddof=1))
    ci_2_upper.append(ci_2[0])
    ci_2_lower.append(ci_2[1])

    temp = np.squeeze(np.transpose(predict_3[:, i_value]))
    ci_3 = stats.t.interval(0.95, all_epochs - 1, np.mean(temp), stats.tsem(temp, ddof=1))
    ci_3_upper.append(ci_3[0])
    ci_3_lower.append(ci_3[1])

file = path + "\\finally_calculations_predict_100.txt"
if os.path.exists(file):
    os.remove(file)


def record_result(data, data_name):
    data = np.array(data)
    with open(file, 'a+') as f:
        f.write(data_name + '\n')
        columns = len(data)

        for column in np.arange(columns):
            if column != columns - 1:
                f.write(str(data[column]) + ',')
            else:
                f.write(str(data[column]) + '\n')


record_result(pressurizer_water_y_test_1, 'y_true_1')
record_result(predict_1_mean, 'predict_1_mean')
record_result(predict_1_max, 'predict_1_max')
record_result(predict_1_min, 'predict_1_min')

record_result(pressurizer_water_y_test_2, 'y_true_2')
record_result(predict_2_mean, 'predict_2_mean')
record_result(predict_2_max, 'predict_2_max')
record_result(predict_2_min, 'predict_2_min')

record_result(pressurizer_water_y_test_3, 'y_true_3')
record_result(predict_3_mean, 'predict_3_mean')
record_result(predict_3_max, 'predict_3_max')
record_result(predict_3_min, 'predict_3_min')

record_result(ci_1_upper, 'ci_1_upper')
record_result(ci_1_lower, 'ci_1_lower')

record_result(ci_2_upper, 'ci_2_upper')
record_result(ci_2_lower, 'ci_2_lower')

record_result(ci_3_upper, 'ci_3_upper')
record_result(ci_3_lower, 'ci_3_lower')

record_result(train_time, 'train_time')
record_result(predict_time_1, 'predict_time_1')
record_result(predict_time_2, 'predict_time_2')
record_result(predict_time_3, 'predict_time_3')
