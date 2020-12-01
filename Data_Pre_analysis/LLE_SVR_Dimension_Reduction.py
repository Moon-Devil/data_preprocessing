from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn import manifold
from sklearn.model_selection import train_test_split
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
from sklearn import svm
from sklearn.model_selection import cross_val_score
import os


grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径

power_decreasing_dict = dict()
power_decreasing_array = power_decreasing


k_folds_number = 10  # k折交叉验证
low_dimension = 5   # 降维-最低维度
high_dimension = 27  # 降维-最高维度

# PCA-支持向量回归降维
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
    LLE_SVR_accuracy[i_value] = []

    x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
                                                        random_state=0)
    model = svm.SVR(kernel='rbf')
    LLE_SVR_history[i_value] = model.fit(x_train, y_train)
    scores = cross_val_score(model, x_test, y_test, cv=k_folds_number, scoring='neg_mean_squared_error')
    LLE_SVR_accuracy[i_value] = scores

LLE_SVR_accuracy_i_value = len(LLE_SVR_accuracy)
LLE_SVR_accuracy_j_value = len(LLE_SVR_accuracy[low_dimension])

# 设置文件存储路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'
if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\LLE_Dimension_Reduction_LLE_SVR.txt"
if os.path.exists(file):
    os.remove(file)

best_dimension = 0
best_mean = 1000000

with open(file, "w+") as f:
    f.write(str("====================================LLE_SVR=================================\n"))
    for i_value in np.arange(LLE_SVR_accuracy_i_value):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t")
        temp_mean = np.mean(LLE_SVR_accuracy[index]) * (-1)
        if temp_mean < best_mean:
            best_mean = temp_mean
            best_dimension = index
        f.write("mean=" + str(temp_mean) + "\t")

        f.write("accuracy=" + "\t")
        for j_value in np.arange(LLE_SVR_accuracy_j_value):
            if j_value != (LLE_SVR_accuracy_j_value - 1):
                f.write(str(LLE_SVR_accuracy[index][j_value] * -1) + ",")
            else:
                f.write(str(LLE_SVR_accuracy[index][j_value] * -1) + "\n")

    f.write("best_dimension=" + str(best_dimension) + "\n")
    f.write("best_mean=" + str(best_mean) + "\n")


