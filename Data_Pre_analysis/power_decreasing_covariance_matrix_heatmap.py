from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# power_decreasing的协方差矩阵
power_decreasing_covariance = np.cov(np.transpose(power_decreasing))
power_decreasing_covariance_column = len(power_decreasing_covariance)
power_decreasing_covariance_row = len(power_decreasing_covariance[0])

for i_value in np.arange(power_decreasing_covariance_column):
    for j_value in np.arange(power_decreasing_covariance_row):
        power_decreasing_covariance[i_value][j_value] = abs(power_decreasing_covariance[i_value][j_value])

# 存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'
if not os.path.exists(path):
    os.mkdir(path)

covariance_file = path + '\\Power_decreasing_covariance_matrix.txt'
if os.path.exists(covariance_file):
    os.remove(covariance_file)

with open(covariance_file, "w+") as f:
    f.write(str("====================================power_decreasing_covariance_matrix============================\n"))
    for i_value in np.arange(power_decreasing_covariance_column):
        for j_value in np.arange(power_decreasing_covariance_row):
            if j_value != (power_decreasing_covariance_row - 1):
                f.write(str(power_decreasing_covariance[i_value][j_value]) + ",")
            else:
                f.write(str(power_decreasing_covariance[i_value][j_value]) + "\n")

for i_value in np.arange(power_decreasing_covariance_column):
    for j_value in np.arange(power_decreasing_covariance_row):
        if i_value == j_value:
            power_decreasing_covariance[i_value][j_value] = 0
        power_decreasing_covariance[i_value][j_value] = 500 if power_decreasing_covariance[i_value][j_value] > 500 \
            else power_decreasing_covariance[i_value][j_value]

sns.set()
power_decreasing_covariance = pd.DataFrame(power_decreasing_covariance)

plt.figure(figsize=[5, 4])
heatmap = sns.heatmap(power_decreasing_covariance, cmap='binary', cbar=False)
cb = heatmap.figure.colorbar(heatmap.collections[0])
cb.ax.tick_params(labelsize=9)
plt.xlabel('Data set labels', fontdict={'family': 'Times New Roman', 'size': 10.5})
plt.ylabel('Data set labels', fontdict={'family': 'Times New Roman', 'size': 10.5})
plt.xticks(fontproperties='Times New Roman', size=10.5)
plt.yticks(fontproperties='Times New Roman', size=10.5)
plt.savefig(grandfather_path + 'Pictures\\power decreasing pre_analysis\\' + 'power decreasing covariance.svg')

plt.figure(figsize=[5, 4])
heatmap = sns.heatmap(power_decreasing_covariance, cmap='binary', cbar=False)
cb = heatmap.figure.colorbar(heatmap.collections[0])
cb.ax.tick_params(labelsize=9)
plt.xlabel('数据集标签', fontdict={'family': '宋体', 'size': 10.5})
plt.ylabel('数据集标签', fontdict={'family': '宋体', 'size': 10.5})
plt.xticks(fontproperties='Times New Roman', size=10.5)
plt.yticks(fontproperties='Times New Roman', size=10.5)
plt.savefig(grandfather_path + 'Pictures\\power decreasing pre_analysis\\' + 'power decreasing covariance_zn.svg')

print("数据统计已完成...")
