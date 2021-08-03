from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
import tensorflow as tf
import numpy as np
from sklearn import manifold
# from sklearn.model_selection import train_test_split
import os


# 设置计算结果存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing Initialization'

if not os.path.exists(path):
    os.mkdir(path)

data_dimension = 16
hidden_layers_nodes = 69
learning_rate = 0.0001491072657856008
batch_size = 1
epochs_size = 200
all_epochs = [1, 2, 3, 5]

pressurizer_water_y = power_decreasing[..., -3]
pressurizer_water_x = np.delete(power_decreasing, -3, axis=1)

# LLE降维
lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=data_dimension, method='standard')
pressurizer_water_x = lle.fit_transform(pressurizer_water_x)

pressurizer_water_x_test_1 = pressurizer_water_x[:301, ...]
pressurizer_water_y_test_1 = pressurizer_water_y[:301]

pressurizer_water_x_test_2 = pressurizer_water_x[1511:1812, ...]
pressurizer_water_y_test_2 = pressurizer_water_y[1511:1812]

model_MAML = tf.keras.models.Sequential()
model_MAML.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu, input_shape=(data_dimension,)))
model_MAML.add(tf.keras.layers.Dense(hidden_layers_nodes, activation=tf.keras.activations.relu))
model_MAML.add(tf.keras.layers.Dense(1))
model_MAML.summary()

model_MAML.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=tf.keras.losses.MAPE,
                   metrics=[tf.keras.losses.MSE])
model_MAML.fit(pressurizer_water_x, pressurizer_water_y, batch_size=batch_size, epochs=3, validation_split=0.2)
parameters = model_MAML.get_weights()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu, input_shape=(data_dimension,)))
model.add(tf.keras.layers.Dense(hidden_layers_nodes, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.set_weights(parameters)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=tf.keras.losses.MAPE,
              metrics=[tf.keras.losses.MSE])
history = model.fit(pressurizer_water_x, pressurizer_water_y, batch_size=batch_size, epochs=301, validation_split=0.2)

y_predict_1 = model.predict(pressurizer_water_x_test_1)
y_predict_2 = model.predict(pressurizer_water_x_test_2)

file = path + "\\MAML_finally_calculations.txt"
with open(file, "w+") as f:
    f.write("loss" + "\t")
    temp = history.history['loss']
    for i_value in np.arange(len(temp)):
        if i_value != len(temp) - 1:
            f.write(str(temp[i_value]) + ",")
        else:
            f.write(str(temp[i_value]) + "\n")

    f.write("MSE" + "\t")
    temp = history.history['mean_squared_error']
    for i_value in np.arange(len(temp)):
        if i_value != len(temp) - 1:
            f.write(str(temp[i_value]) + ",")
        else:
            f.write(str(temp[i_value]) + "\n")

    f.write("val_loss" + "\t")
    temp = history.history['val_loss']
    for i_value in np.arange(len(temp)):
        if i_value != len(temp) - 1:
            f.write(str(temp[i_value]) + ",")
        else:
            f.write(str(temp[i_value]) + "\n")

    f.write("val_MSE" + "\t")
    temp = history.history['val_mean_squared_error']
    for i_value in np.arange(len(temp)):
        if i_value != len(temp) - 1:
            f.write(str(temp[i_value]) + ",")
        else:
            f.write(str(temp[i_value]) + "\n")

    f.write("true_1" + "\t")
    temp = pressurizer_water_y_test_1
    for i_value in np.arange(len(temp)):
        if i_value != len(temp) - 1:
            f.write(str(temp[i_value]) + ",")
        else:
            f.write(str(temp[i_value]) + "\n")

    f.write("predict_1" + "\t")
    temp = y_predict_1
    for i_value in np.arange(len(temp)):
        if i_value != len(temp) - 1:
            f.write(str(temp[i_value][0]) + ",")
        else:
            f.write(str(temp[i_value][0]) + "\n")

    f.write("true_2" + "\t")
    temp = pressurizer_water_y_test_2
    for i_value in np.arange(len(temp)):
        if i_value != len(temp) - 1:
            f.write(str(temp[i_value]) + ",")
        else:
            f.write(str(temp[i_value]) + "\n")

    f.write("predict_2" + "\t")
    temp = y_predict_2
    for i_value in np.arange(len(temp)):
        if i_value != len(temp) - 1:
            f.write(str(temp[i_value][0]) + ",")
        else:
            f.write(str(temp[i_value][0]) + "\n")

print('done')
