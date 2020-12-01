import numpy as np
import math
import matplotlib.pyplot as plt
import MySQLdb

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

k_folds_number = 10  # k折交叉验证
low_dimension = 5   # 降维-最低维度
high_dimension = 6  # 降维-最高维度(包括原维度)
batch_size = 32     # NN与RNN批处理长度
epochs_size = 2000     # NN与RNN循环代数


def normal_distribution(x, mean_value, sigma_value):
    return np.exp(-1 * ((x - mean_value) ** 2) / (2 * (sigma_value ** 2))) / (math.sqrt(2 * np.pi) * sigma_value)


def set_picture_EN(x_axis_label, y_axis_label):
    plt.xlabel(x_axis_label, fontdict={'family': 'Times New Roman', 'size': 10.5})
    plt.xticks(fontproperties='Times New Roman', size=9)
    plt.ylabel(y_axis_label, fontdict={'family': 'Times New Roman', 'size': 10.5})
    plt.yticks(fontproperties='Times New Roman', size=9)


# 将target和rate作为数据，形成新的数组
def data_from_database(database_name, table_name, value1, value2) -> object:
    # 连接数据库
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    sql = "USE " + database_name
    cursor.execute(sql)

    sql = "SELECT * from " + table_name
    cursor.execute(sql)
    results = cursor.fetchall()

    # 从数据库中读取table_name数据
    temp_dict = dict()
    for i_value in range(len(table_header)):
        temp_dict[table_header[i_value]] = []

    for row in results:
        for i_value in range(len(table_header)):
            temp_dict[table_header[i_value]].append(row[i_value])

    # 数组
    array_column = len(temp_dict[table_header[0]])
    array_row = len(table_header)

    temp_data_array = np.zeros((array_column, array_row))
    for i_value in range(array_column):
        for j_value in range(array_row):
            temp_data_array[i_value][j_value] = temp_dict[table_header[j_value]][i_value]

    # 去掉重复值和恒定值
    temp_array = np.delete(temp_data_array, [0, 15, 16, 19, 20], axis=1)

    # 添加target和rate值
    new_row_1 = np.full((1, len(temp_array)), value1)
    new_row_2 = np.full((1, len(temp_array)), value2)
    new_rows = np.append(new_row_1, new_row_2, axis=0)
    new_rows = new_rows.T

    array = np.append(temp_array, new_rows, axis=1)

    db.close()
    return array

