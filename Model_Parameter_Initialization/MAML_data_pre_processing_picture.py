from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
import tensorflow as tf
import numpy as np
from sklearn import manifold
from sklearn.model_selection import train_test_split
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
all_epochs = [1, 2, 3, 4, 5, 10, 50, 100]

pressurizer_water_y = power_decreasing[..., -3]
pressurizer_water_x = np.delete(power_decreasing, -3, axis=1)

# LLE降维
lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=data_dimension, method='standard')
pressurizer_water_x = lle.fit_transform(pressurizer_water_x)

pressurizer_water_x_test_1 = pressurizer_water_x[:300, ...]
pressurizer_water_y_test_1 = pressurizer_water_y[:300]

pressurizer_water_x_test_2 = pressurizer_water_x[1511:1811, ...]
pressurizer_water_y_test_2 = pressurizer_water_y[1511:1811]

x_train_pressurizer_water, x_test_pressurizer_water, y_train_pressurizer_water, y_test_pressurizer_water = \
    train_test_split(pressurizer_water_x, pressurizer_water_y, test_size=0.3, random_state=0)

for epoch in all_epochs:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu,
                                    input_shape=(data_dimension,)))
    model.add(tf.keras.layers.Dense(hidden_layers_nodes, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1))

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=tf.keras.losses.MAPE,
                  metrics=[tf.keras.losses.MSE])

    history = model.fit(x_train_pressurizer_water, y_train_pressurizer_water, batch_size=batch_size, epochs=epoch,
                        validation_split=0.2)

    # , callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                                                           patience=20, verbose=0)]

    y_predict = model.predict(x_test_pressurizer_water[:300, ...])
    y_predict_1 = model.predict(pressurizer_water_x_test_1)
    y_predict_2 = model.predict(pressurizer_water_x_test_2)

    file = path + "\\finally_calculations_loss_" + str(epoch) + ".txt"
    if epoch == 1:
        if os.path.exists(file):
            os.remove(file)
        with open(file, "w+") as f:
            f.write("=========================\t" + "loss" + "\t=========================\n")

    with open(file, "a") as f:
        temp_list = history.history['loss']
        length = len(temp_list)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value])+",")
            else:
                f.write(str(temp_list[i_value]) + "\n")

    file = path + "\\finally_calculations_MSE_" + str(epoch) + ".txt"
    if epoch == 1:
        if os.path.exists(file):
            os.remove(file)
        with open(file, "w+") as f:
            f.write("=========================\t" + "MSE" + "\t=========================\n")

    with open(file, "a") as f:
        temp_list = history.history['mean_squared_error']
        length = len(temp_list)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value]) + ",")
            else:
                f.write(str(temp_list[i_value]) + "\n")

    file = path + "\\finally_calculations_val_loss_" + str(epoch) + ".txt"
    if epoch == 1:
        if os.path.exists(file):
            os.remove(file)
        with open(file, "w+") as f:
            f.write("=========================\t" + "val_loss" + "\t=========================\n")

    with open(file, "a") as f:
        temp_list = history.history['val_loss']
        length = len(temp_list)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value]) + ",")
            else:
                f.write(str(temp_list[i_value]) + "\n")

    file = path + "\\finally_calculations_val_MSE_" + str(epoch) + ".txt"
    if epoch == 1:
        if os.path.exists(file):
            os.remove(file)
        with open(file, "w+") as f:
            f.write("=========================\t" + "val_MSE" + "\t=========================\n")

    with open(file, "a") as f:
        temp_list = history.history['val_mean_squared_error']
        length = len(temp_list)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value]) + ",")
            else:
                f.write(str(temp_list[i_value]) + "\n")

    file = path + "\\finally_calculations_y_predict_" + str(epoch) + ".txt"
    if epoch == 1:
        if os.path.exists(file):
            os.remove(file)
        with open(file, "w+") as f:
            f.write("=========================\t" + "y_true" + "\t=========================\n")
            temp_list = y_test_pressurizer_water
            length = len(temp_list)
            for i_value in np.arange(length):
                if i_value != length - 1:
                    f.write(str(temp_list[i_value]) + ",")
                else:
                    f.write(str(temp_list[i_value]) + "\n")
            f.write("=========================\t" + "y_predict" + "\t=========================\n")

    with open(file, "a") as f:
        temp_list = y_predict
        length = len(temp_list)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value][0]) + ",")
            else:
                f.write(str(temp_list[i_value][0]) + "\n")

    file = path + "\\finally_calculations_y_predict_1_" + str(epoch) + ".txt"
    if epoch == 1:
        if os.path.exists(file):
            os.remove(file)
        with open(file, "w+") as f:
            f.write("=========================\t" + "y_true_1" + "\t=========================\n")
            temp_list = pressurizer_water_y_test_1
            length = len(temp_list)
            for i_value in np.arange(length):
                if i_value != length - 1:
                    f.write(str(temp_list[i_value]) + ",")
                else:
                    f.write(str(temp_list[i_value]) + "\n")
            f.write("=========================\t" + "y_predict_1" + "\t=========================\n")

    with open(file, "a") as f:
        temp_list = y_predict_1
        length = len(temp_list)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value][0]) + ",")
            else:
                f.write(str(temp_list[i_value][0]) + "\n")

    file = path + "\\finally_calculations_y_predict_2_" + str(epoch) + ".txt"
    if epoch == 1:
        if os.path.exists(file):
            os.remove(file)
        with open(file, "w+") as f:
            f.write("=========================\t" + "y_true_2" + "\t=========================\n")
            temp_list = pressurizer_water_y_test_2
            length = len(temp_list)
            for i_value in np.arange(length):
                if i_value != length - 1:
                    f.write(str(temp_list[i_value]) + ",")
                else:
                    f.write(str(temp_list[i_value]) + "\n")
            f.write("=========================\t" + "y_predict_2" + "\t=========================\n")

    with open(file, "a") as f:
        temp_list = y_predict_2
        length = len(temp_list)
        for i_value in np.arange(length):
            if i_value != length - 1:
                f.write(str(temp_list[i_value][0]) + ",")
            else:
                f.write(str(temp_list[i_value][0]) + "\n")
