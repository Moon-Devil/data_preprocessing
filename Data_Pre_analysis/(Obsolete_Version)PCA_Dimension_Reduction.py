from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import gaussian_process
import os
# np.set_printoptions(threshold=np.inf)


# 设置数据表存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径

# # power_decreasing的协方差矩阵
# power_decreasing_covariance = np.cov(np.transpose(power_decreasing))
# power_decreasing_covariance_column = len(power_decreasing_covariance)
# power_decreasing_covariance_row = len(power_decreasing_covariance[0])
#
# # 存储路径
# grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
# path = grandfather_path + 'Calculations\\power decreasing pre_analysis'
# if not os.path.exists(path):
#     os.mkdir(path)
#
# covariance_file = path + '\\Database_Statistics.txt'
# if os.path.exists(covariance_file):
#     os.remove(covariance_file)
#
# with open(covariance_file, "w+") as f:
#     f.write(str("====================================power_decreasing_covariance_matrix============================\n"))
#     for i_value in np.arange(power_decreasing_covariance_column):
#         for j_value in np.arange(power_decreasing_covariance_row):
#             if j_value != (power_decreasing_covariance_row - 1):
#                 f.write(str(power_decreasing_covariance[i_value][j_value]) + ",")
#             else:
#                 f.write(str(power_decreasing_covariance[i_value][j_value]) + "\n")
#
# print("数据统计已完成...")

power_decreasing_dict = dict()
power_decreasing_array = power_decreasing
#
# # PCA-神经网络降维
# PCA_NN_history = dict()
# PCA_NN_accuracy = dict()
# best_mean = 10000
#
# for i_value in range(low_dimension, high_dimension):
#     # 预测稳压器水空间温度
#     y_train_set = power_decreasing_array[..., -3]
#     x_train_set = np.delete(power_decreasing_array, -3, axis=1)
#     temp_accuracy = []
#
#     # PCA降维
#     pca = PCA(n_components=i_value)
#     power_decreasing_dict[i_value] = pca.fit_transform(x_train_set)
#     PCA_NN_accuracy[i_value] = []
#
#     # 建模 DNN
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Dense(i_value, activation=tf.nn.relu, input_shape=(i_value,)))
#     model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu,
#                                     kernel_regularizer=tf.keras.regularizers.l1_l2()))
#     model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu,
#                                     kernel_regularizer=tf.keras.regularizers.l1_l2()))
#     model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
#     model.summary()
#
#     model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
#                   metrics=[tf.keras.losses.MSE])
#
#     # k折交叉验证
#     for _ in range(k_folds_number):
#         x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set,
#                                                             test_size=0.3, random_state=0)
#         # 此部分为论文结果统一，随机种子设为0，若实际程序，请设为随机值
#         temp_model = model
#         PCA_NN_history = temp_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_size,
#                                         validation_split=0.2)
#         y_predict = temp_model.predict(x_test)
#         temp_accuracy.append(mean_squared_error(y_test, y_predict))
#         k_folds_accuracy_array = np.array(temp_accuracy)
#
#     PCA_NN_accuracy[i_value] = k_folds_accuracy_array
#     temp_mean = np.mean(PCA_NN_accuracy[i_value])
#     if temp_mean < best_mean:
#         if temp_mean < 0:
#             print("均方差小于0，程序退出")
#             exit()
#
#         PCA_NN_loss = PCA_NN_history.history['loss']
#         PCA_NN_MSE = PCA_NN_history.history['mean_squared_error']
#         PCA_NN_valid_loss = PCA_NN_history.history['val_loss']
#         PCA_NN_valid_MSE = PCA_NN_history.history['val_mean_squared_error']
#         best_dimension = i_value
#
# PCA_NN_accuracy_i_value = len(PCA_NN_accuracy)
# PCA_NN_accuracy_j_value = len(PCA_NN_accuracy[low_dimension])
#
# file = path + "\\PCA_Dimension_Reduction_PCA_NN.txt"
# if os.path.exists(file):
#     os.remove(file)
#
# with open(file, "w+") as f:
#     f.write(str("====================================PCA_NN=================================\n"))
#     for i_value in np.arange(PCA_NN_accuracy_i_value):
#         index = i_value + low_dimension
#         f.write("Dimension=" + str(index) + "\t")
#         temp_mean = np.mean(PCA_NN_accuracy[index])
#         f.write("mean=" + str(temp_mean) + "\t")
#
#         f.write("accuracy=" + "\t")
#         for j_value in np.arange(PCA_NN_accuracy_j_value):
#             if j_value != (PCA_NN_accuracy_j_value - 1):
#                 f.write(str(PCA_NN_accuracy[index][j_value]) + ",")
#             else:
#                 f.write(str(PCA_NN_accuracy[index][j_value]) + "\n")
#
#     f.write("Best dimension=" + "\t" + str(best_dimension) + "\n")
#
#     f.write("PCA_NN_loss=" + "\t")
#     for i_value in np.arange(len(PCA_NN_loss)):
#         if i_value != (len(PCA_NN_loss) - 1):
#             f.write(str(PCA_NN_loss[i_value]) + ",")
#         else:
#             f.write(str(PCA_NN_loss[i_value]) + "\n")
#
#     f.write("PCA_NN_MSE=" + "\t")
#     for i_value in np.arange(len(PCA_NN_MSE)):
#         if i_value != (len(PCA_NN_MSE) - 1):
#             f.write(str(PCA_NN_MSE[i_value]) + ",")
#         else:
#             f.write(str(PCA_NN_MSE[i_value]) + "\n")
#
#     f.write("PCA_NN_valid_loss=" + "\t")
#     for i_value in np.arange(len(PCA_NN_valid_loss)):
#         if i_value != (len(PCA_NN_valid_loss) - 1):
#             f.write(str(PCA_NN_valid_loss[i_value]) + ",")
#         else:
#             f.write(str(PCA_NN_valid_loss[i_value]) + "\n")
#
#     f.write("PCA_NN_valid_MSE=" + "\t")
#     for i_value in np.arange(len(PCA_NN_valid_MSE)):
#         if i_value != (len(PCA_NN_valid_MSE) - 1):
#             f.write(str(PCA_NN_valid_MSE[i_value]) + ",")
#         else:
#             f.write(str(PCA_NN_valid_MSE[i_value]) + "\n")


# # PCA-支持向量回归降维
# PCA_SVR_history = dict()
# PCA_SVR_accuracy = dict()
# PCA_SVR_y_predict = dict()
# PCA_SVR_y_test = dict()
#
# for i_value in range(low_dimension, high_dimension):
#     # 预测稳压器水空间温度
#     y_train_set = power_decreasing_array[..., -3]
#     x_train_set = np.delete(power_decreasing_array, -3, axis=1)
#     temp_accuracy = []
#
#     # PCA降维
#     pca = PCA(n_components=i_value)
#     power_decreasing_dict[i_value] = pca.fit_transform(x_train_set)
#     PCA_SVR_accuracy[i_value] = []
#
#     x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
#                                                         random_state=0)
#     model = svm.SVR(kernel='rbf')
#     PCA_SVR_history[i_value] = model.fit(x_train, y_train)
#     scores = cross_val_score(model, x_test, y_test, cv=k_folds_number, scoring='neg_mean_squared_error')
#     PCA_SVR_accuracy[i_value] = scores
#
# PCA_SVR_accuracy_i_value = len(PCA_SVR_accuracy)
# PCA_SVR_accuracy_j_value = len(PCA_SVR_accuracy[low_dimension])
#
#
# path = grandfather_path + 'Calculations\\power decreasing pre_analysis'
# if not os.path.exists(path):
#     os.mkdir(path)
#
# file = path + "\\PCA_Dimension_Reduction_PCA_SVR.txt"
# if os.path.exists(file):
#     os.remove(file)
#
# with open(file, "w+") as f:
#     f.write(str("====================================PCA_SVR=================================\n"))
#     for i_value in np.arange(PCA_SVR_accuracy_i_value):
#         index = i_value + low_dimension
#         f.write("Dimension=" + str(index) + "\t")
#         temp_mean = np.mean(PCA_SVR_accuracy[index])
#         f.write("mean=" + str(temp_mean * (-1)) + "\t")
#
#         f.write("accuracy=" + "\t")
#         for j_value in np.arange(PCA_SVR_accuracy_j_value):
#             if j_value != (PCA_SVR_accuracy_j_value - 1):
#                 f.write(str(PCA_SVR_accuracy[index][j_value] * -1) + ",")
#             else:
#                 f.write(str(PCA_SVR_accuracy[index][j_value] * -1) + "\n")
#
#
# # PCA-高斯过程回归降维
# PCA_GPR_history = dict()
# PCA_GPR_accuracy = dict()
#
# for i_value in range(low_dimension, high_dimension):
#     # 预测稳压器水空间温度
#     y_train_set = power_decreasing_array[..., -3]
#     x_train_set = np.delete(power_decreasing_array, -3, axis=1)
#     temp_accuracy = []
#
#     # PCA降维
#     pca = PCA(n_components=i_value)
#     power_decreasing_dict[i_value] = pca.fit_transform(x_train_set)
#     PCA_GPR_accuracy[i_value] = []
#
#     x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
#                                                         random_state=0)
#     model = gaussian_process.GaussianProcessRegressor()
#
#     PCA_GPR_history[i_value] = model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     x_array = np.arange(len(y_predict))
#
#     scores = cross_val_score(model, x_test, y_test, cv=k_folds_number, scoring='neg_mean_squared_error')
#     PCA_GPR_accuracy[i_value] = scores
#
# PCA_GPR_accuracy_i_value = len(PCA_GPR_accuracy)
# PCA_GPR_accuracy_j_value = len(PCA_GPR_accuracy[low_dimension])
#
# # 设置文件存储路径
# path = grandfather_path + 'Calculations\\power decreasing pre_analysis'
# if not os.path.exists(path):
#     os.mkdir(path)
#
# file = path + "\\PCA_Dimension_Reduction_PCA_GPR.txt"
# if os.path.exists(file):
#     os.remove(file)
#
# with open(file, "w+") as f:
#     f.write(str("====================================PCA_GPR=================================\n"))
#     for i_value in np.arange(PCA_GPR_accuracy_i_value):
#         index = i_value + low_dimension
#         f.write("Dimension=" + str(index) + "\t")
#         temp_mean = np.mean(PCA_GPR_accuracy[index])
#         f.write("mean=" + str(temp_mean * (-1)) + "\t")
#
#         f.write("accuracy=" + "\t")
#         for j_value in np.arange(PCA_GPR_accuracy_j_value):
#             if j_value != (PCA_GPR_accuracy_j_value - 1):
#                 f.write(str(PCA_GPR_accuracy[index][j_value] * -1) + ",")
#             else:
#                 f.write(str(PCA_GPR_accuracy[index][j_value] * -1) + "\n")
#
#
# PCA-RNN降维
PCA_RNN_history = dict()
PCA_RNN_accuracy = dict()

for i_value in range(low_dimension, high_dimension):
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # PCA降维
    pca = PCA(n_components=i_value)
    power_decreasing_dict[i_value] = pca.fit_transform(x_train_set)
    PCA_RNN_accuracy[i_value] = []

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
        PCA_RNN_history[i_value] = model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs_size)
        y_predict = model.predict(x_test)
        y_predict = np.squeeze(y_predict)
        y_test = np.squeeze(y_test)

        temp_accuracy.append(mean_squared_error(y_test, y_predict))
        accuracy_array = np.array(temp_accuracy)
        PCA_RNN_accuracy[i_value] = accuracy_array

PCA_RNN_accuracy_i_value = len(PCA_RNN_accuracy)
PCA_RNN_accuracy_j_value = len(PCA_RNN_accuracy[low_dimension])

with open(file, "a") as f:
    f.write(str("====================================PCA_RNN=================================\n"))
    for i_value in np.arange(PCA_RNN_accuracy_i_value):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t")
        temp_mean = np.mean(PCA_RNN_accuracy[index])
        f.write("mean=" + str(temp_mean) + "\t")

        f.write("accuracy=" + "\t")
        for j_value in np.arange(PCA_RNN_accuracy_j_value):
            if j_value != (PCA_RNN_accuracy_j_value - 1):
                f.write(str(PCA_RNN_accuracy[index][j_value]) + ",")
            else:
                f.write(str(PCA_RNN_accuracy[index][j_value]) + "\n")

