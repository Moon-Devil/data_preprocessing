from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn import manifold
from sklearn.model_selection import train_test_split
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
from sklearn.model_selection import cross_val_score
from sklearn import gaussian_process
import os

power_decreasing_dict = dict()
power_decreasing_array = power_decreasing

k_folds_number = 10  # k折交叉验证
low_dimension = 5   # 降维-最低维度
high_dimension = 27  # 降维-最高维度

# Isomap-高斯过程回归降维
Isomap_GPR_history = dict()
Isomap_GPR_accuracy = dict()

for i_value in range(low_dimension, high_dimension):
    # 预测稳压器水空间温度
    y_train_set = power_decreasing_array[..., -3]
    x_train_set = np.delete(power_decreasing_array, -3, axis=1)
    temp_accuracy = []

    # Isomap降维
    Isomap = manifold.Isomap(n_components=i_value)
    power_decreasing_dict[i_value] = Isomap.fit_transform(x_train_set)
    Isomap_GPR_accuracy[i_value] = []

    x_train, x_test, y_train, y_test = train_test_split(power_decreasing_dict[i_value], y_train_set, test_size=0.3,
                                                        random_state=0)
    model = gaussian_process.GaussianProcessRegressor()

    Isomap_GPR_history[i_value] = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    x_array = np.arange(len(y_predict))

    scores = cross_val_score(model, x_test, y_test, cv=k_folds_number, scoring='neg_mean_squared_error')
    Isomap_GPR_accuracy[i_value] = scores

Isomap_GPR_accuracy_i_value = len(Isomap_GPR_accuracy)
Isomap_GPR_accuracy_j_value = len(Isomap_GPR_accuracy[low_dimension])

# 设置文件存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'
if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\Isomap_Dimension_Reduction_Isomap_GPR.txt"
if os.path.exists(file):
    os.remove(file)

best_dimension = 0
best_mean = 1000000

with open(file, "w+") as f:
    f.write(str("====================================Isomap_GPR=================================\n"))
    for i_value in np.arange(Isomap_GPR_accuracy_i_value):
        index = i_value + low_dimension
        f.write("Dimension=" + str(index) + "\t")
        temp_mean = np.mean(Isomap_GPR_accuracy[index]) * (-1)
        if temp_mean < best_mean:
            best_mean = temp_mean
            best_dimension = index
        f.write("mean=" + str(temp_mean) + "\t")

        f.write("accuracy=" + "\t")
        for j_value in np.arange(Isomap_GPR_accuracy_j_value):
            if j_value != (Isomap_GPR_accuracy_j_value - 1):
                f.write(str(Isomap_GPR_accuracy[index][j_value] * -1) + ",")
            else:
                f.write(str(Isomap_GPR_accuracy[index][j_value] * -1) + "\n")

    f.write("best_dimension=" + str(best_dimension) + "\n")
    f.write("best_mean=" + str(best_mean) + "\n")
