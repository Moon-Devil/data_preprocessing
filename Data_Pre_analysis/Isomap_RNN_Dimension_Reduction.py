from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn import manifold
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
from sklearn.metrics import mean_squared_error
import os


power_decreasing_dict = dict()
power_decreasing_array = power_decreasing

low_dimension = 5   # 降维-最低维度
high_dimension = 27  # 降维-最高维度
batch_size = 32     # NN与RNN批处理长度
epochs_size = 400     # NN与RNN循环代数

Isomap_RNN_loss = []
Isomap_RNN_MSE = []
Isomap_RNN_valid_loss = []
Isomap_RNN_valid_MSE = []
best_dimension = 0
best_accuracy = 1000000

# Isomap-RNN降维
Isomap_RNN_history = dict()
Isomap_RNN_accuracy = dict()

for i_value in range(low_dimension, high_dimension):    # low_dimension, high_dimension
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # Isomap降维
    Isomap = manifold.Isomap(n_components=i_value)
    power_decreasing_dict[i_value] = Isomap.fit_transform(x_train_set)
    Isomap_RNN_accuracy[i_value] = []

    x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
                                                        random_state=0)
    x_train = x_train[:, :, np.newaxis]
    y_train = y_train[:, np.newaxis]
    x_test = x_test[:, :, np.newaxis]
    y_test = y_test[:, np.newaxis]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(i_value, return_sequences=True, input_shape=[None, 1]))
    model.add(tf.keras.layers.SimpleRNN(40, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(40))
    model.add(tf.keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
                  metrics=[tf.keras.losses.MSE])
    Isomap_RNN_history = model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs_size,
                                   validation_split=0.2 , callbacks=[tf.keras.callbacks.EarlyStopping
                                                                     (monitor='val_loss', patience=20)])
    # , callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)]
    y_predict = model.predict(x_test)
    y_predict = np.squeeze(y_predict)
    y_test = np.squeeze(y_test)

    accuracy = mean_squared_error(y_test, y_predict)
    Isomap_RNN_accuracy[i_value] = accuracy

    if Isomap_RNN_accuracy[i_value] < best_accuracy:
        best_accuracy = Isomap_RNN_accuracy[i_value]
        Isomap_RNN_loss = Isomap_RNN_history.history['loss']
        Isomap_RNN_MSE = Isomap_RNN_history.history['mean_squared_error']
        Isomap_RNN_valid_loss = Isomap_RNN_history.history['val_loss']
        Isomap_RNN_valid_MSE = Isomap_RNN_history.history['val_mean_squared_error']
        best_dimension = i_value

Isomap_RNN_accuracy_length = len(Isomap_RNN_accuracy)

# 设置计算结果存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'

if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\Isomap_Dimension_Reduction_Isomap_RNN.txt"
if os.path.exists(file):
    os.remove(file)

with open(file, "w+") as f:
    f.write(str("====================================Isomap_RNN=================================\n"))
    for i_value in np.arange(Isomap_RNN_accuracy_length):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t" + "accuracy=" + str(Isomap_RNN_accuracy[index]) + "\n")

    f.write("Best dimension=" + str(best_dimension) + "\n")
    f.write("Isomap_RNN_loss_length= " + str(len(Isomap_RNN_loss)) + "\n")
    f.write("Isomap_RNN_MSE_length= " + str(len(Isomap_RNN_MSE)) + "\n")
    f.write("Isomap_RNN_val_loss_length= " + str(len(Isomap_RNN_valid_loss)) + "\n")
    f.write("Isomap_RNN_val_MSE_length= " + str(len(Isomap_RNN_valid_MSE)) + "\n")

    f.write("Isomap_RNN_loss=" + "\t")
    for i_value in np.arange(len(Isomap_RNN_loss)):
        if i_value != (len(Isomap_RNN_loss) - 1):
            f.write(str(Isomap_RNN_loss[i_value]) + ",")
        else:
            f.write(str(Isomap_RNN_loss[i_value]) + "\n")

    f.write("Isomap_RNN_MSE=" + "\t")
    for i_value in np.arange(len(Isomap_RNN_MSE)):
        if i_value != (len(Isomap_RNN_MSE) - 1):
            f.write(str(Isomap_RNN_MSE[i_value]) + ",")
        else:
            f.write(str(Isomap_RNN_MSE[i_value]) + "\n")

    f.write("LLE_RNN_val_loss=" + "\t")
    for i_value in np.arange(len(Isomap_RNN_valid_loss)):
        if i_value != (len(Isomap_RNN_valid_loss) - 1):
            f.write(str(Isomap_RNN_valid_loss[i_value]) + ",")
        else:
            f.write(str(Isomap_RNN_valid_loss[i_value]) + "\n")

    f.write("LLE_RNN_val_MSE=" + "\t")
    for i_value in np.arange(len(Isomap_RNN_valid_MSE)):
        if i_value != (len(Isomap_RNN_valid_MSE) - 1):
            f.write(str(Isomap_RNN_valid_MSE[i_value]) + ",")
        else:
            f.write(str(Isomap_RNN_valid_MSE[i_value]) + "\n")
