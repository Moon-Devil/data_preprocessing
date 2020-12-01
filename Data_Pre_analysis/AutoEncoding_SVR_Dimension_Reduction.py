from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
from sklearn import svm
from sklearn.model_selection import cross_val_score
import os

grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径

power_decreasing_dict = dict()
power_decreasing_array = power_decreasing

low_dimension = 5   # 降维-最低维度
high_dimension = 27  # 降维-最高维度
batch_size = 32     # NN批处理长度
epochs_size = 400     # NN循环代数

# AE-支持向量回归降维
AE_SVR_history = dict()
AE_SVR_accuracy = dict()
AE_SVR_y_predict = dict()
AE_SVR_y_test = dict()
history = dict()

for i_value in range(low_dimension, high_dimension):    # low_dimension, high_dimension
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # AE降维
    inputs = tf.keras.layers.InputLayer(input_shape=(x_train_set.shape[1], ))
    hidden1 = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    hidden2 = tf.keras.layers.Dense(i_value, activation=tf.keras.activations.relu)
    hidden3 = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)
    outputs = tf.keras.layers.Dense(x_train_set.shape[1])

    AE_model = tf.keras.models.Sequential([inputs, hidden1, hidden2, hidden3, outputs])
    AE_model.summary()

    encoder_model = tf.keras.models.Sequential([inputs, hidden1, hidden2])
    encoder_model.summary()

    decoder_input = tf.keras.layers.InputLayer((i_value,))
    decoder_output = AE_model.layers[-1]
    decoder_model = tf.keras.models.Sequential([decoder_input, hidden3, decoder_output])
    decoder_model.summary()

    AE_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
                     metrics=[tf.keras.losses.MSE])
    history[i_value] = AE_model.fit(x_train_set, x_train_set, batch_size=batch_size, epochs=epochs_size,
                                    validation_split=0.2)
    # ,
    #                                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)]

    power_decreasing_dict[i_value] = encoder_model.predict(x_train_set)
    AE_SVR_accuracy[i_value] = []

    x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
                                                        random_state=0)
    model = svm.SVR(kernel='rbf')
    AE_SVR_history[i_value] = model.fit(x_train, y_train)
    scores = cross_val_score(model, x_test, y_test, cv=k_folds_number, scoring='neg_mean_squared_error')
    AE_SVR_accuracy[i_value] = scores

AE_SVR_accuracy_i_value = len(AE_SVR_accuracy)
AE_SVR_accuracy_j_value = len(AE_SVR_accuracy[low_dimension])

# 设置文件存储路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'
if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\AE_Dimension_Reduction_AE_SVR.txt"
if os.path.exists(file):
    os.remove(file)

best_dimension = 0
best_mean = 1000000
best_AE_model = dict()

with open(file, "w+") as f:
    f.write(str("====================================AE_SVR=================================\n"))
    for i_value in np.arange(AE_SVR_accuracy_i_value):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t")
        temp_mean = np.mean(AE_SVR_accuracy[index]) * (-1)
        if temp_mean < best_mean:
            best_mean = temp_mean
            best_dimension = index
            best_AE_model = history[index]
        f.write("mean=" + str(temp_mean) + "\t")

        f.write("accuracy=" + "\t")
        for j_value in np.arange(AE_SVR_accuracy_j_value):
            if j_value != (AE_SVR_accuracy_j_value - 1):
                f.write(str(AE_SVR_accuracy[index][j_value] * -1) + ",")
            else:
                f.write(str(AE_SVR_accuracy[index][j_value] * -1) + "\n")

    f.write("best_dimension=" + str(best_dimension) + "\n")
    f.write("best_mean=" + str(best_mean) + "\n")

    real_epoch = len(best_AE_model.history['loss'])
    f.write("real_epoch=" + str(real_epoch) + "\n")

    f.write("best_AE_model_loss=" + "\t")
    for i_value in np.arange(len(best_AE_model.history['loss'])):
        if i_value != (len(best_AE_model.history['loss']) - 1):
            f.write(str(best_AE_model.history['loss'][i_value]) + ",")
        else:
            f.write(str(best_AE_model.history['loss'][i_value]) + "\n")

    f.write("best_AE_model_mean_squared_error" + "\t")
    for i_value in np.arange(len(best_AE_model.history['mean_squared_error'])):
        if i_value != (len(best_AE_model.history['mean_squared_error']) - 1):
            f.write(str(best_AE_model.history['mean_squared_error'][i_value]) + ",")
        else:
            f.write(str(best_AE_model.history['mean_squared_error'][i_value]) + "\n")

    f.write("best_AE_model_val_loss" + "\t")
    for i_value in np.arange(len(best_AE_model.history['val_loss'])):
        if i_value != (len(best_AE_model.history['val_loss']) - 1):
            f.write(str(best_AE_model.history['val_loss'][i_value]) + ",")
        else:
            f.write(str(best_AE_model.history['val_loss'][i_value]) + "\n")

    f.write("best_AE_model_val_mean_squared_error" + "\t")
    for i_value in np.arange(len(best_AE_model.history['val_mean_squared_error'])):
        if i_value != (len(best_AE_model.history['val_mean_squared_error']) - 1):
            f.write(str(best_AE_model.history['val_mean_squared_error'][i_value]) + ",")
        else:
            f.write(str(best_AE_model.history['val_mean_squared_error'][i_value]) + "\n")
