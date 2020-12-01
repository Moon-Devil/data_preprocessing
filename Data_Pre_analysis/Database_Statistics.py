import MySQLdb
import numpy as np
import os
import seaborn as sns
import pandas as pd

host = "localhost"  # 数据库地址
username = "root"  # 数据库用户名
passwd = "1111"  # 数据库密码
database = "mysql"  # 数据库类型

table_header = ['id',
                'thermal_power', 'electric_power',
                'coolant_flow_primary_circuit', 'coolant_flow_secondary_circuit',
                'hot_leg_temperature_primary', 'hot_leg_temperature_secondary',
                'cold_leg_temperature_primary', 'cold_leg_temperature_secondary',
                'pressure_steam_generator_primary', 'pressure_steam_generator_secondary',
                'water_level_primary', 'water_level_secondary',
                'feed_water_flow_steam_generator_1', 'feed_water_flow_steam_generator_2',
                'feed_water_flow_steam_generator_3', 'feed_water_flow_steam_generator_4',
                'feed_water_temp_steam_generator_1', 'feed_water_temp_steam_generator_2',
                'feed_water_temp_steam_generator_3', 'feed_water_temp_steam_generator_4',
                'steam_outlet_flow_primary', 'steam_outlet_flow_secondary',
                'steam_outlet_temperature_primary', 'steam_outlet_temperature_secondary',
                'pressurizer_pressure', 'pressurizer_water_level',
                'pressurizer_heat_power', 'pressurizer_steam_space_temperature', 'pressurizer_water_space_temperature']

# 连接数据库
db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
cursor = db.cursor()

sql = "USE power_decreasing"
cursor.execute(sql)

# 读取power_decreasing_target200数据
sql = "SELECT * from power_decreasing_target200"
cursor.execute(sql)
results = cursor.fetchall()

# 从target200中读取数据
power_decreasing_target200 = dict()
for i_value in range(len(table_header)):
    power_decreasing_target200[table_header[i_value]] = []

for row in results:
    for i_value in range(len(table_header)):
        power_decreasing_target200[table_header[i_value]].append(row[i_value])

# target200数组
array_column = len(power_decreasing_target200[table_header[0]])
array_row = len(table_header)

temp_data_array = np.zeros((array_column, array_row))
for i_value in range(array_column):
    for j_value in range(array_row):
        temp_data_array[i_value][j_value] = power_decreasing_target200[table_header[j_value]][i_value]

# 去掉重复值和恒定值
power_decreasing_array_target200 = np.delete(temp_data_array, [0, 15, 16, 19, 20], axis=1)

sql = "SELECT * from power_decreasing_target220"
cursor.execute(sql)
results = cursor.fetchall()

# 从target220中读取数据
power_decreasing_target220 = dict()
for i_value in range(len(table_header)):
    power_decreasing_target220[table_header[i_value]] = []

for row in results:
    for i_value in range(len(table_header)):
        power_decreasing_target220[table_header[i_value]].append(row[i_value])

# target220数组
array_column = len(power_decreasing_target220[table_header[0]])
array_row = len(table_header)

temp_data_array = np.zeros((array_column, array_row))
for i_value in range(array_column):
    for j_value in range(array_row):
        temp_data_array[i_value][j_value] = power_decreasing_target220[table_header[j_value]][i_value]

# 去掉重复值和恒定值
power_decreasing_array_target220 = np.delete(temp_data_array, [0, 15, 16, 19, 20], axis=1)

sql = "SELECT * from power_decreasing_target250"
cursor.execute(sql)
results = cursor.fetchall()

# 从target250中读取数据
power_decreasing_target250 = dict()
for i_value in range(len(table_header)):
    power_decreasing_target250[table_header[i_value]] = []

for row in results:
    for i_value in range(len(table_header)):
        power_decreasing_target250[table_header[i_value]].append(row[i_value])

# target250数组
array_column = len(power_decreasing_target250[table_header[0]])
array_row = len(table_header)

temp_data_array = np.zeros((array_column, array_row))
for i_value in range(array_column):
    for j_value in range(array_row):
        temp_data_array[i_value][j_value] = power_decreasing_target250[table_header[j_value]][i_value]

# 去掉重复值和恒定值
power_decreasing_array_target250 = np.delete(temp_data_array, [0, 15, 16, 19, 20], axis=1)

# 读取all data数据
sql = "SELECT * from all_data"
cursor.execute(sql)
results = cursor.fetchall()

# 从数据库中读取power decreasing数据
power_decreasing = dict()
for i_value in range(len(table_header)):
    power_decreasing[table_header[i_value]] = []

for row in results:
    for i_value in range(len(table_header)):
        power_decreasing[table_header[i_value]].append(row[i_value])

# power decreasing数组
array_column = len(power_decreasing[table_header[0]])
array_row = len(table_header)

temp_data_array = np.zeros((array_column, array_row))
for i_value in range(array_column):
    for j_value in range(array_row):
        temp_data_array[i_value][j_value] = power_decreasing[table_header[j_value]][i_value]

# 去掉重复值和恒定值
power_decreasing_array = np.delete(temp_data_array, [0, 15, 16, 19, 20], axis=1)

target200_length = len(power_decreasing_array_target200)
target200_mean = power_decreasing_array_target200.mean()
target200_var = power_decreasing_array_target200.var()
target200_std = power_decreasing_array_target200.std()
target200_max = power_decreasing_array_target200.max()
target200_min = power_decreasing_array_target200.min()

target220_length = len(power_decreasing_array_target220)
target220_mean = power_decreasing_array_target220.mean()
target220_var = power_decreasing_array_target220.var()
target220_std = power_decreasing_array_target220.std()
target220_max = power_decreasing_array_target220.max()
target220_min = power_decreasing_array_target220.min()

target250_length = len(power_decreasing_array_target250)
target250_mean = power_decreasing_array_target250.mean()
target250_var = power_decreasing_array_target250.var()
target250_std = power_decreasing_array_target250.std()
target250_max = power_decreasing_array_target250.max()
target250_min = power_decreasing_array_target250.min()

power_decreasing_length = len(power_decreasing_array)
power_decreasing_mean = power_decreasing_array.mean()
power_decreasing_var = power_decreasing_array.var()
power_decreasing_std = power_decreasing_array.std()
power_decreasing_max = power_decreasing_array.max()
power_decreasing_min = power_decreasing_array.min()

print("===========================================================================================")
print("label\t\ttarget200\t\t\ttarget220\t\t\ttarget250\t\t\tpower_decreasing")
print("length\t\t" + str(target200_length) + "\t\t\t\t" + str(target220_length) + "\t\t\t\t" + str(target250_length)
      + "\t\t\t\t" + str(power_decreasing_length))
print("mean\t\t" + str(target200_mean) + "\t" + str(target220_mean) + "\t" + str(target250_mean) + "\t"
      + str(power_decreasing_mean))
print("var\t\t\t" + str(target200_var) + "\t" + str(target220_var) + "\t" + str(target250_var) + "\t"
      + str(power_decreasing_var))
print("std\t\t\t" + str(target200_std) + "\t" + str(target220_std) + "\t" + str(target250_std) + "\t"
      + str(power_decreasing_std))
print("max\t\t\t" + str(target200_max) + "\t\t\t" + str(target220_max) + "\t\t\t" + str(target250_max) + "\t\t\t"
      + str(power_decreasing_max))
print("min\t\t\t" + str(target200_min) + "\t\t\t\t" + str(target220_min) + "\t\t\t\t" + str(target250_min) + "\t\t\t\t\t"
      + str(power_decreasing_min))
print("===========================================================================================")

# 设置数据表存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing pre_analysis'

if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\Database_Statistics.txt"
if os.path.exists(file):
    os.remove(file)

with open(path + "\\Database_Statistics.txt", "w") as f:
    f.write("===========================================================================================\n")
    f.write("label\t\ttarget200\t\t\ttarget220\t\t\ttarget250\t\t\tpower_decreasing\n")
    f.write("---------------------------------------------------------------------------------------------------"
            "---------------------------------------------------------\n")
    f.write("length\t\t" + str(target200_length) + "\t\t\t" + str(target220_length) + "\t\t\t"
            + str(target250_length) + "\t\t\t" + str(power_decreasing_length) + "\n")
    f.write("mean\t\t" + str(target200_mean) + "\t\t" + str(target220_mean) + "\t\t" + str(target250_mean) + "\t\t"
            + str(power_decreasing_mean) + "\n")
    f.write("var\t\t" + str(target200_var) + "\t\t" + str(target220_var) + "\t\t" + str(target250_var) + "\t\t"
            + str(power_decreasing_var) + "\n")
    f.write("std\t\t" + str(target200_std) + "\t\t" + str(target220_std) + "\t\t" + str(target250_std) + "\t\t"
            + str(power_decreasing_std) + "\n")
    f.write("max\t\t" + str(target200_max) + "\t\t" + str(target220_max) + "\t\t" + str(target250_max)
            + "\t\t" + str(power_decreasing_max) + "\n")
    f.write("min\t\t" + str(target200_min) + "\t\t\t" + str(target220_min) + "\t\t\t" + str(target250_min)
            + "\t\t\t" + str(power_decreasing_min) + "\n")
    f.write("===========================================================================================\n\n\n")

db.close()
