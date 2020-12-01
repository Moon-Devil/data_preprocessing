from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
import os


# 设置数据表存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径

# power_decreasing的协方差矩阵
power_decreasing_covariance = np.cov(np.transpose(power_decreasing))
power_decreasing_covariance_column = len(power_decreasing_covariance)
power_decreasing_covariance_row = len(power_decreasing_covariance[0])

# 存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'
if not os.path.exists(path):
    os.mkdir(path)

covariance_file = path + '\\Covariance_Matrix.txt'
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

print("数据统计已完成...")
