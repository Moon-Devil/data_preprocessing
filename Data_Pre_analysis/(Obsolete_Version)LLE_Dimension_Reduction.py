from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import gaussian_process
import os
from sklearn import manifold
np.set_printoptions(threshold=np.inf)


power_decreasing_dict = dict()
power_decreasing_array = power_decreasing

# 设置数据表存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'

if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\LLE_Dimension_Reduction.txt"
if os.path.exists(file):
    os.remove(file)

# LLE-神经网络降维
LLE_NN_history = dict()
LLE_NN_accuracy = dict()

for i_value in range(low_dimension, high_dimension):
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # LLE降维
    lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=i_value, method='standard')
    power_decreasing_dict[i_value] = lle.fit_transform(x_train_set)
    LLE_NN_accuracy[i_value] = []

    # k折交叉验证
    for j_value in range(k_folds_number):
        random_seed = np.random.randint(0, 1000)
        x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set,
                                                            test_size=0.3, random_state=random_seed)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(i_value, activation=tf.nn.relu, input_shape=(i_value, )))
        model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu,
                  kernel_regularizer=tf.keras.regularizers.l1_l2()))
        model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu,
                                        kernel_regularizer=tf.keras.regularizers.l1_l2()))
        model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
                      metrics=[tf.keras.losses.MSE])

        LLE_NN_history[i_value] = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_size,
                                            validation_split=0.2)
        y_predict = model.predict(x_test)
        temp_accuracy.append(mean_squared_error(y_test, y_predict))
        accuracy_array = np.array(temp_accuracy)
        LLE_NN_accuracy[i_value] = accuracy_array

LLE_NN_accuracy_i_value = len(LLE_NN_accuracy)
LLE_NN_accuracy_j_value = len(LLE_NN_accuracy[low_dimension])

with open(file, "w+") as f:
    f.write(str("====================================LLE_NN=================================\n"))
    for i_value in np.arange(LLE_NN_accuracy_i_value):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t")
        temp_mean = np.mean(LLE_NN_accuracy[index])
        f.write("mean=" + str(temp_mean) + "\t")

        f.write("accuracy=" + "\t")
        for j_value in np.arange(LLE_NN_accuracy_j_value):
            if j_value != (LLE_NN_accuracy_j_value - 1):
                f.write(str(LLE_NN_accuracy[index][j_value]) + ",")
            else:
                f.write(str(LLE_NN_accuracy[index][j_value]) + "\n")


# LLE-支持向量回归降维
LLE_SVR_history = dict()
LLE_SVR_accuracy = dict()
LLE_SVR_y_predict = dict()
LLE_SVR_y_test = dict()

for i_value in range(low_dimension, high_dimension):
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # LLE降维
    lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=i_value, method='standard')
    power_decreasing_dict[i_value] = lle.fit_transform(x_train_set)
    LLE_NN_accuracy[i_value] = []

    x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
                                                        random_state=0)
    model = svm.SVR(kernel='rbf')
    LLE_SVR_history[i_value] = model.fit(x_train, y_train)
    scores = cross_val_score(model, x_test, y_test, cv=k_folds_number, scoring='neg_mean_squared_error')
    LLE_SVR_accuracy[i_value] = scores

    y_predict = model.predict(x_test)
    LLE_SVR_y_predict[i_value] = y_predict
    LLE_SVR_y_predict[i_value] = y_test

LLE_SVR_accuracy_i_value = len(LLE_SVR_accuracy)
LLE_SVR_accuracy_j_value = len(LLE_SVR_accuracy[low_dimension])

with open(file, "a") as f:
    f.write(str("====================================LLE_SVR=================================\n"))
    for i_value in np.arange(LLE_SVR_accuracy_i_value):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t")
        temp_mean = np.mean(LLE_SVR_accuracy[index])
        f.write("mean=" + str(temp_mean * (-1)) + "\t")

        f.write("accuracy=" + "\t")
        for j_value in np.arange(LLE_SVR_accuracy_j_value):
            if j_value != (LLE_SVR_accuracy_j_value - 1):
                f.write(str(LLE_SVR_accuracy[index][j_value] * -1) + ",")
            else:
                f.write(str(LLE_SVR_accuracy[index][j_value] * -1) + "\n")


# LLE-高斯过程回归降维
LLE_GPR_history = dict()
LLE_GPR_accuracy = dict()

for i_value in range(low_dimension, high_dimension):
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # LLE降维
    lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=i_value, method='standard')
    power_decreasing_dict[i_value] = lle.fit_transform(x_train_set)
    LLE_NN_accuracy[i_value] = []

    x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
                                                        random_state=0)
    model = gaussian_process.GaussianProcessRegressor()

    LLE_GPR_history[i_value] = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    x_array = np.arange(len(y_predict))

    scores = cross_val_score(model, x_test, y_test, cv=k_folds_number, scoring='neg_mean_squared_error')
    LLE_GPR_accuracy[i_value] = scores

LLE_GPR_accuracy_i_value = len(LLE_GPR_accuracy)
LLE_GPR_accuracy_j_value = len(LLE_GPR_accuracy[low_dimension])

with open(file, "a") as f:
    f.write(str("====================================LLE_GPR=================================\n"))
    for i_value in np.arange(LLE_GPR_accuracy_i_value):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t")
        temp_mean = np.mean(LLE_GPR_accuracy[index])
        f.write("mean=" + str(temp_mean * (-1)) + "\t")

        f.write("accuracy=" + "\t")
        for j_value in np.arange(LLE_GPR_accuracy_j_value):
            if j_value != (LLE_GPR_accuracy_j_value - 1):
                f.write(str(LLE_GPR_accuracy[index][j_value] * -1) + ",")
            else:
                f.write(str(LLE_GPR_accuracy[index][j_value] * -1) + "\n")


# LLE-RNN降维
LLE_RNN_history = dict()
LLE_RNN_accuracy = dict()

for i_value in range(low_dimension, high_dimension):
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # LDA降维
    lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=i_value, method='standard')
    power_decreasing_dict[i_value] = lle.fit_transform(x_train_set)
    LLE_NN_accuracy[i_value] = []

    # k折交叉验证
    for j_value in range(k_folds_number):
        random_seed = np.random.randint(0, 1000)
        x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
                                                            random_state=random_seed)
        x_train = x_train[:, :, np.newaxis]
        y_train = y_train[:, np.newaxis]
        x_test = x_test[:, :, np.newaxis]
        y_test = y_test[:, np.newaxis]

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.SimpleRNN(i_value, return_sequences=True, input_shape=[None, 1]))
        model.add(tf.keras.layers.SimpleRNN(40))
        model.add(tf.keras.layers.SimpleRNN(40))
        model.add(tf.keras.layers.Dense(1))
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
                      metrics=[tf.keras.losses.MSE])
        LLE_RNN_history[i_value] = model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs_size)
        y_predict = model.predict(x_test)
        y_predict = np.squeeze(y_predict)
        y_test = np.squeeze(y_test)

        temp_accuracy.append(mean_squared_error(y_test, y_predict))
        accuracy_array = np.array(temp_accuracy)
        LLE_RNN_accuracy[i_value] = accuracy_array

LLE_RNN_accuracy_i_value = len(LLE_RNN_accuracy)
LLE_RNN_accuracy_j_value = len(LLE_RNN_accuracy[low_dimension])

with open(file, "a") as f:
    f.write(str("====================================LLE_RNN=================================\n"))
    for i_value in np.arange(LLE_RNN_accuracy_i_value):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t")
        temp_mean = np.mean(LLE_RNN_accuracy[index])
        f.write("mean=" + str(temp_mean) + "\t")

        f.write("accuracy=" + "\t")
        for j_value in np.arange(LLE_RNN_accuracy_j_value):
            if j_value != (LLE_RNN_accuracy_j_value - 1):
                f.write(str(LLE_RNN_accuracy[index][j_value]) + ",")
            else:
                f.write(str(LLE_RNN_accuracy[index][j_value]) + "\n")
