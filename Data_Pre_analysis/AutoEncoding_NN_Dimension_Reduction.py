from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import gc


power_decreasing_array = power_decreasing

# AE-神经网络降维
AE_NN_history = dict()
AE_NN_accuracy = dict()
best_accuracy = 10000000

low_dimension = 5   # 降维-最低维度
high_dimension = 27  # 降维-最高维度
batch_size = 32     # NN批处理长度
epochs_size = 2000    # NN循环代数

AE_loss = []
AE_MSE = []
AE_valid_loss = []
AE_valid_MSE = []

AE_NN_loss = []
AE_NN_MSE = []
AE_NN_valid_loss = []
AE_NN_valid_MSE = []
best_dimension = 0

for i_value in range(low_dimension, high_dimension):    # low_dimension, high_dimension
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # AE降维
    inputs = tf.keras.layers.InputLayer(input_shape=(x_train_set.shape[1],))
    hidden1 = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    hidden2 = tf.keras.layers.Dense(i_value, activation=tf.keras.activations.relu)
    hidden3 = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    outputs = tf.keras.layers.Dense(x_train_set.shape[1])

    AE_model = tf.keras.models.Sequential([inputs, hidden1, hidden2, hidden3, outputs])
    AE_model.summary()

    encoder_model = tf.keras.models.Sequential([inputs, hidden1, hidden2])
    encoder_model.summary()

    AE_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
                     metrics=[tf.keras.losses.MSE])
    AE_history = AE_model.fit(x_train_set, x_train_set, batch_size=batch_size, epochs=epochs_size,
                              validation_split=0.2,
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])
    # ,
    #                                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)]

    power_decreasing_dict = encoder_model.predict(x_train_set)

    x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict, y_train_set, test_size=0.3,
                                                        random_state=0)

    # 建模 DNN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(i_value, activation=tf.nn.relu, input_shape=(i_value,)))
    model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
                  metrics=[tf.keras.losses.MSE])

    AE_NN_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_size, validation_split=0.2,
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])
    y_predict = model.predict(x_test)
    MSE_accuracy = mean_squared_error(y_test, y_predict)

    AE_NN_accuracy[i_value] = MSE_accuracy

    if AE_NN_accuracy[i_value] < best_accuracy:
        best_accuracy = AE_NN_accuracy[i_value]
        AE_NN_loss = AE_NN_history.history['loss']
        AE_NN_MSE = AE_NN_history.history['mean_squared_error']
        AE_NN_valid_loss = AE_NN_history.history['val_loss']
        AE_NN_valid_MSE = AE_NN_history.history['val_mean_squared_error']

        AE_loss = AE_history.history['loss']
        AE_MSE = AE_history.history['mean_squared_error']
        AE_valid_loss = AE_history.history['val_loss']
        AE_valid_MSE = AE_history.history['val_mean_squared_error']

        best_dimension = i_value

del model, AE_history, AE_NN_history, hidden1, hidden2, hidden3, encoder_model, inputs, outputs
gc.collect()

AE_NN_accuracy_length = len(AE_NN_accuracy)

# 设置计算结果存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'

if not os.path.exists(path):
    os.mkdir(path)
file = path + "\\AE_Dimension_Reduction_AE_NN.txt"
if os.path.exists(file):
    os.remove(file)

with open(file, "w+") as f:
    f.write(str("====================================AE_NN=================================\n"))
    for i_value in np.arange(AE_NN_accuracy_length):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t" + "accuracy=" + str(AE_NN_accuracy[index]) + "\n")

    f.write("Best dimension=" + str(best_dimension) + "\n")
    f.write("AE_NN_loss_length= " + str(len(AE_NN_loss)) + "\n")
    f.write("AE_NN_MSE_length= " + str(len(AE_NN_MSE)) + "\n")
    f.write("AE_NN_val_loss_length= " + str(len(AE_NN_valid_loss)) + "\n")
    f.write("AE_NN_val_MSE_length= " + str(len(AE_NN_valid_MSE)) + "\n")

    f.write("AE_loss_length= " + str(len(AE_loss)) + "\n")
    f.write("AE_MSE_length= " + str(len(AE_MSE)) + "\n")
    f.write("AE_val_loss_length= " + str(len(AE_valid_loss)) + "\n")
    f.write("AE_val_MSE_length= " + str(len(AE_valid_MSE)) + "\n")

    f.write("AE_NN_loss=" + "\t")
    for i_value in np.arange(len(AE_NN_loss)):
        if i_value != (len(AE_NN_loss) - 1):
            f.write(str(AE_NN_loss[i_value]) + ",")
        else:
            f.write(str(AE_NN_loss[i_value]) + "\n")

    f.write("AE_NN_MSE=" + "\t")
    for i_value in np.arange(len(AE_NN_MSE)):
        if i_value != (len(AE_NN_MSE) - 1):
            f.write(str(AE_NN_MSE[i_value]) + ",")
        else:
            f.write(str(AE_NN_MSE[i_value]) + "\n")

    f.write("AE_NN_val_loss=" + "\t")
    for i_value in np.arange(len(AE_NN_valid_loss)):
        if i_value != (len(AE_NN_valid_loss) - 1):
            f.write(str(AE_NN_valid_loss[i_value]) + ",")
        else:
            f.write(str(AE_NN_valid_loss[i_value]) + "\n")

    f.write("AE_NN_val_MSE=" + "\t")
    for i_value in np.arange(len(AE_NN_valid_MSE)):
        if i_value != (len(AE_NN_valid_MSE) - 1):
            f.write(str(AE_NN_valid_MSE[i_value]) + ",")
        else:
            f.write(str(AE_NN_valid_MSE[i_value]) + "\n")

    f.write("AE_loss=" + "\t")
    for i_value in np.arange(len(AE_loss)):
        if i_value != (len(AE_loss) - 1):
            f.write(str(AE_loss[i_value]) + ",")
        else:
            f.write(str(AE_loss[i_value]) + "\n")

    f.write("AE_MSE=" + "\t")
    for i_value in np.arange(len(AE_MSE)):
        if i_value != (len(AE_MSE) - 1):
            f.write(str(AE_MSE[i_value]) + ",")
        else:
            f.write(str(AE_MSE[i_value]) + "\n")

    f.write("AE_val_loss=" + "\t")
    for i_value in np.arange(len(AE_valid_loss)):
        if i_value != (len(AE_valid_loss) - 1):
            f.write(str(AE_valid_loss[i_value]) + ",")
        else:
            f.write(str(AE_valid_loss[i_value]) + "\n")

    f.write("AE_val_MSE=" + "\t")
    for i_value in np.arange(len(AE_valid_MSE)):
        if i_value != (len(AE_valid_MSE) - 1):
            f.write(str(AE_valid_MSE[i_value]) + ",")
        else:
            f.write(str(AE_valid_MSE[i_value]) + "\n")
