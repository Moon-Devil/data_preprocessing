from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn import manifold
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
from sklearn.metrics import mean_squared_error
import os


power_decreasing_dict = dict()
power_decreasing_array = power_decreasing

# LLE-神经网络降维
LLE_NN_history = dict()
LLE_NN_accuracy = dict()
best_accuracy = 10000

low_dimension = 5   # 降维-最低维度
high_dimension = 27  # 降维-最高维度
batch_size = 32     # NN批处理长度
epochs_size = 400     # NN循环代数

LLE_NN_loss = []
LLE_NN_MSE = []
LLE_NN_valid_loss = []
LLE_NN_valid_MSE = []
best_dimension = 0

for i_value in range(low_dimension, high_dimension):
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # LLE降维
    lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=i_value, method='standard')
    power_decreasing_dict[i_value] = lle.fit_transform(x_train_set)
    LLE_NN_accuracy[i_value] = []

    # 建模 DNN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(i_value, activation=tf.nn.relu, input_shape=(i_value,)))
    model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
                  metrics=[tf.keras.losses.MSE])

    x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set,
                                                        test_size=0.3, random_state=0)
    # 此部分为论文结果统一，随机种子设为0，若实际程序，请设为随机值

    LLE_NN_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_size, validation_split=0.2,
                               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0)])
    # , callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0)]

    y_predict = model.predict(x_test)
    MSE_accuracy = mean_squared_error(y_test, y_predict)

    LLE_NN_accuracy[i_value] = MSE_accuracy

    if LLE_NN_accuracy[i_value] < best_accuracy:
        best_accuracy = LLE_NN_accuracy[i_value]
        LLE_NN_loss = LLE_NN_history.history['loss']
        LLE_NN_MSE = LLE_NN_history.history['mean_squared_error']
        LLE_NN_valid_loss = LLE_NN_history.history['val_loss']
        LLE_NN_valid_MSE = LLE_NN_history.history['val_mean_squared_error']
        best_dimension = i_value

LLE_NN_accuracy_length = len(LLE_NN_accuracy)

# 设置计算结果存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'

if not os.path.exists(path):
    os.mkdir(path)
file = path + "\\LLE_Dimension_Reduction_LLE_NN.txt"
if os.path.exists(file):
    os.remove(file)

with open(file, "w+") as f:
    f.write(str("====================================LLE_NN=================================\n"))
    for i_value in np.arange(LLE_NN_accuracy_length):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t" + "accuracy=" + str(LLE_NN_accuracy[index]) + "\n")

    f.write("Best dimension=" + str(best_dimension) + "\n")
    f.write("LLE_NN_loss_length= " + str(len(LLE_NN_loss)) + "\n")
    f.write("LLE_NN_MSE_length= " + str(len(LLE_NN_MSE)) + "\n")
    f.write("LLE_NN_val_loss_length= " + str(len(LLE_NN_valid_loss)) + "\n")
    f.write("LLE_NN_val_MSE_length= " + str(len(LLE_NN_valid_MSE)) + "\n")

    f.write("LLE_NN_loss=" + "\t")
    for i_value in np.arange(len(LLE_NN_loss)):
        if i_value != (len(LLE_NN_loss) - 1):
            f.write(str(LLE_NN_loss[i_value]) + ",")
        else:
            f.write(str(LLE_NN_loss[i_value]) + "\n")

    f.write("LLE_NN_MSE=" + "\t")
    for i_value in np.arange(len(LLE_NN_MSE)):
        if i_value != (len(LLE_NN_MSE) - 1):
            f.write(str(LLE_NN_MSE[i_value]) + ",")
        else:
            f.write(str(LLE_NN_MSE[i_value]) + "\n")

    f.write("LLE_NN_val_loss=" + "\t")
    for i_value in np.arange(len(LLE_NN_valid_loss)):
        if i_value != (len(LLE_NN_valid_loss) - 1):
            f.write(str(LLE_NN_valid_loss[i_value]) + ",")
        else:
            f.write(str(LLE_NN_valid_loss[i_value]) + "\n")

    f.write("LLE_NN_val_MSE=" + "\t")
    for i_value in np.arange(len(LLE_NN_valid_MSE)):
        if i_value != (len(LLE_NN_valid_MSE) - 1):
            f.write(str(LLE_NN_valid_MSE[i_value]) + ",")
        else:
            f.write(str(LLE_NN_valid_MSE[i_value]) + "\n")
