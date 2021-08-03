from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
import tensorflow as tf
import numpy as np
from sklearn import manifold
import os


# 设置计算结果存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\test_optimization'

if not os.path.exists(path):
    os.mkdir(path)

data_dimension = 16
batch_size = 32
epochs_size = 401

pressurizer_water_y = power_decreasing[..., -3]
pressurizer_water_x = np.delete(power_decreasing, -3, axis=1)

# LLE降维
lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=data_dimension, method='standard')
pressurizer_water_x = lle.fit_transform(pressurizer_water_x)

pressurizer_water_x_test_1 = pressurizer_water_x[:301, ...]
pressurizer_water_y_test_1 = pressurizer_water_y[:301]

pressurizer_water_x_test_2 = pressurizer_water_x[1511:1812, ...]
pressurizer_water_y_test_2 = pressurizer_water_y[1511:1812]

file = path + "\\none_Bayesian_optimization.txt"

model_none = tf.keras.models.Sequential()
model_none.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu,
                                     input_shape=(data_dimension, )))
model_none.add(tf.keras.layers.Dense(20, activation=tf.keras.activations.relu))
model_none.add(tf.keras.layers.Dense(1))
model_none.summary()

model_none.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.MAPE,
                   metrics=[tf.keras.losses.MSE])
history_none = model_none.fit(pressurizer_water_x, pressurizer_water_y, batch_size=batch_size, epochs=epochs_size,
                              validation_split=0.2)

predict_none_200 = model_none.predict(pressurizer_water_x_test_1)
predict_none_220 = model_none.predict(pressurizer_water_x_test_2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu,  input_shape=(data_dimension, )))
model.add(tf.keras.layers.Dense(69, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001491072657856008), loss=tf.keras.losses.MAPE,
              metrics=[tf.keras.losses.MSE])
history = model.fit(pressurizer_water_x, pressurizer_water_y, batch_size=batch_size, epochs=epochs_size,
                    validation_split=0.2)

predict_200 = model.predict(pressurizer_water_x_test_1)
predict_220 = model.predict(pressurizer_water_x_test_2)

with open(file, 'w+') as f:
    f.write('None Bayesian optimization\n')
    f.write('loss\t')
    temp = history_none.history['loss']
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('mean_squared_error\t')
    temp = history_none.history['mean_squared_error']
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('val_loss\t')
    temp = history_none.history['val_loss']
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('val_mean_squared_error\t')
    temp = history_none.history['val_mean_squared_error']
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('predict_none_200\t')
    temp = predict_none_200
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value][0]) + ',')
        else:
            f.write(str(temp[i_value][0]) + '\n')

    f.write('predict_none_220\t')
    temp = predict_none_220
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value][0]) + ',')
        else:
            f.write(str(temp[i_value][0]) + '\n')

    f.write('Bayesian optimization\n')
    f.write('loss\t')
    temp = history.history['loss']
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('mean_squared_error\t')
    temp = history.history['mean_squared_error']
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('val_loss\t')
    temp = history.history['val_loss']
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('val_mean_squared_error\t')
    temp = history.history['val_mean_squared_error']
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('predict_200\t')
    temp = predict_200
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value][0]) + ',')
        else:
            f.write(str(temp[i_value][0]) + '\n')

    f.write('predict_220\t')
    temp = predict_220
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value][0]) + ',')
        else:
            f.write(str(temp[i_value][0]) + '\n')

    f.write('true_200\t')
    temp = pressurizer_water_y_test_1
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

    f.write('true_220\t')
    temp = pressurizer_water_y_test_2
    length = len(temp)
    for i_value in np.arange(length):
        if i_value != length - 1:
            f.write(str(temp[i_value]) + ',')
        else:
            f.write(str(temp[i_value]) + '\n')

