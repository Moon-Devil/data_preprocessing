import tensorflow as tf
import numpy as np
from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn import manifold
from sklearn.model_selection import train_test_split
import os


data_dimension = 16
nodes = 11
learning_rate = 0.00008891149105107228
epochs_size = 400
batch_size = 32

power_decreasing_array = power_decreasing

# LLE-DNN降维
LLE_DNN_history = dict()
LLE_DNN_accuracy = dict()

# 预测稳压器水空间温度
y_train_set = power_decreasing_array[..., -3]
x_train_set = np.delete(power_decreasing_array, -3, axis=1)
temp_accuracy = []

# LLE降维
lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=data_dimension, method='standard')
power_decreasing_LLE = lle.fit_transform(x_train_set)

# 划分训练集、验证集和测试集
x_train, x_test, y_train, y_test = train_test_split(power_decreasing_LLE, y_train_set, test_size=0.3,
                                                    random_state=0)

# 建模
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu, input_shape=(data_dimension, )))
model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, ))
model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, ))
model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, ))
model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, ))
model.add(tf.keras.layers.Dense(1))
model.summary()

optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss=tf.keras.losses.MAPE, metrics=[tf.keras.losses.MSE])
history = model.fit(x=x_train, y=y_train, epochs=epochs_size, batch_size=batch_size, validation_split=0.2)
# ,
#                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)]

DNN_loss = history.history['loss']
DNN_MSE = history.history['mean_squared_error']
DNN_val_loss = history.history['val_loss']
DNN_val_MSE = history.history['val_mean_squared_error']

y_predict_1 = []
y_true_1 = []
for i_value in np.arange(403):
    y_predict_1.append(model.predict(power_decreasing_LLE[i_value].reshape(1, data_dimension)))
    y_true_1.append(y_train_set[i_value])

y_predict_2 = []
y_true_2 = []
for i_value in np.arange(1511, 1947):
    y_predict_2.append(model.predict(power_decreasing_LLE[i_value].reshape(1, data_dimension)))
    y_true_2.append(y_train_set[i_value])

# 设置计算结果存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing build model'

if not os.path.exists(path):
    os.mkdir(path)
file = path + "\\Model_after_optimization.txt"
if os.path.exists(file):
    os.remove(file)

with open(file, "w+") as f:
    f.write("DNN_loss" + "\t")
    for i_value in np.arange(len(DNN_loss)):
        if i_value != len(DNN_loss) - 1:
            f.write(str(DNN_loss[i_value]) + ",")
        else:
            f.write(str(DNN_loss[i_value]) + "\n")

    f.write("DNN_MSE" + "\t")
    for i_value in np.arange(len(DNN_MSE)):
        if i_value != len(DNN_MSE) - 1:
            f.write(str(DNN_MSE[i_value]) + ",")
        else:
            f.write(str(DNN_MSE[i_value]) + "\n")

    f.write("DNN_val_loss" + "\t")
    for i_value in np.arange(len(DNN_val_loss)):
        if i_value != len(DNN_val_loss) - 1:
            f.write(str(DNN_val_loss[i_value]) + ",")
        else:
            f.write(str(DNN_val_loss[i_value]) + "\n")

    f.write("DNN_val_MSE" + "\t")
    for i_value in np.arange(len(DNN_val_MSE)):
        if i_value != len(DNN_val_MSE) - 1:
            f.write(str(DNN_val_MSE[i_value]) + ",")
        else:
            f.write(str(DNN_val_MSE[i_value]) + "\n")

    f.write("y_predict_1" + "\t")
    for i_value in np.arange(len(y_predict_1)):
        if i_value != len(y_predict_1) - 1:
            f.write(str((y_predict_1[i_value][0][0])) + ",")
        else:
            f.write(str(y_predict_1[i_value][0][0]) + "\n")

    f.write("y_true_1" + "\t")
    for i_value in np.arange(len(y_true_1)):
        if i_value != len(y_true_1) - 1:
            f.write(str(y_true_1[i_value]) + ",")
        else:
            f.write(str(y_true_1[i_value]) + "\n")

    f.write("y_predict_2" + "\t")
    for i_value in np.arange(len(y_predict_2)):
        if i_value != len(y_predict_2) - 1:
            f.write(str(y_predict_2[i_value][0][0]) + ",")
        else:
            f.write(str(y_predict_2[i_value][0][0]) + "\n")

    f.write("y_true_2" + "\t")
    for i_value in np.arange(len(y_true_2)):
        if i_value != len(y_true_2) - 1:
            f.write(str(y_true_2[i_value]) + ",")
        else:
            f.write(str(y_true_2[i_value]) + "\n")
