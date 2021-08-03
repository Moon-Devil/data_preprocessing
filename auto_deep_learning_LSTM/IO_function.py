import MySQLdb
import os
import numpy as np
import re
import pandas as pd

data_set_length = 301


def add_file(file_name) -> object:
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = os.path.join(grandfather_path, "Calculations\\Auto_deep_learning")
    if not os.path.exists(path):
        os.mkdir(path)

    file = os.path.join(path, file_name)
    return file


def read_data_from_database(y_index, time_diff, target_list) -> object:
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
                    'pressurizer_heat_power', 'pressurizer_steam_space_temperature',
                    'pressurizer_water_space_temperature', 'target', 'rate']

    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()
    cursor.execute("use power_decreasing")

    table_names = []
    cursor.execute("show tables")
    results = cursor.fetchall()
    for row in results:
        if row[0][0: 6] == "target":
            table_names.append(row[0])

    power_decreasing_data_set = []
    for table_name in table_names:
        string = str.split(table_name, '_')
        if len(string) == 2:
            target = int(re.sub("\D", "", string[0]))
            rate = int(re.sub("\D", "", string[1]))
        else:
            target = int(re.sub("\D", "", string[0]))
            rate = int(re.sub("\D", "", string[1])) + int(string[2]) * 0.1

        temp_list = []
        sql = "select * from " + table_name
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            temp_row = []
            for i_value in np.arange(len(table_header) - 2):
                temp_row.append(row[i_value])
            temp_row.extend([target, rate])
            temp_list.append(temp_row)
        temp_array = np.array(temp_list)
        temp_power_decreasing_array = np.delete(temp_array, [0, 15, 16, 19, 20], axis=1)
        temp_y_data_0 = temp_power_decreasing_array[: data_set_length, y_index]
        temp_y_data = temp_power_decreasing_array[time_diff: time_diff + data_set_length, y_index]
        temp_x_data = np.delete(temp_power_decreasing_array, y_index, axis=1)
        temp_x_data = temp_x_data[0: data_set_length]

        power_decreasing_data_set.append([target, temp_x_data, temp_y_data, temp_y_data_0])

    power_decreasing_array_table_header = table_header
    del power_decreasing_array_table_header[20]
    del power_decreasing_array_table_header[19]
    del power_decreasing_array_table_header[16]
    del power_decreasing_array_table_header[15]
    del power_decreasing_array_table_header[0]

    y_table_header = power_decreasing_array_table_header[y_index]
    del power_decreasing_array_table_header[y_index]
    print("=" * 20 + "y_label" + "=" * 20)
    print(y_table_header)
    print("=" * 20 + "x_label" + "=" * 20)
    for i_value in np.arange(len(power_decreasing_array_table_header)):
        print(str(i_value) + "\t" + power_decreasing_array_table_header[i_value])

    x_data = pd.DataFrame(columns=power_decreasing_array_table_header)
    y_data = pd.DataFrame(columns=[y_table_header])
    y_data_0 = pd.DataFrame(columns=[y_table_header])
    for i_value in np.arange(len(power_decreasing_data_set)):
        if power_decreasing_data_set[i_value][0] in target_list:
            temp_x_df = pd.DataFrame(power_decreasing_data_set[i_value][1], columns=power_decreasing_array_table_header)
            x_data = pd.concat([x_data, temp_x_df])
            temp_y_df = pd.DataFrame(power_decreasing_data_set[i_value][2], columns=[y_table_header])
            y_data = pd.concat([y_data, temp_y_df])
            temp_y_0_df = pd.DataFrame(power_decreasing_data_set[i_value][3], columns=[y_table_header])
            y_data_0 = pd.concat([y_data_0, temp_y_0_df])

    return x_data, y_data, y_data_0


def history_record(file_name, history, y_true, y_predict, y_data_origin, train_time, predict_time, time_diff):
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = os.path.join(grandfather_path, "Calculations\\Auto_deep_learning")
    file_path_history = os.path.join(path, file_name + "_history.txt")
    file_path_y = os.path.join(path, file_name + "_y.txt")
    file_path_y_0 = os.path.join(path, file_name + "_y_origin.txt")
    file_path_time = os.path.join(path, file_name + "_time.txt")
    y_data_origin = y_data_origin.values
    y_data_origin = y_data_origin[:data_set_length, ]

    with open(file_path_history, "w+") as f:
        length = len(history.history['loss'])
        for index in np.arange(length):
            f.write(str(history.history['loss'][index]) + "," +
                    str(history.history['mean_squared_error'][index]) + "," +
                    str(history.history['val_loss'][index]) + "," +
                    str(history.history['val_mean_squared_error'][index]) + "\n")

    with open(file_path_y, "w+") as f:
        length = len(y_true)
        for index in np.arange(length):
            f.write(str(y_true[index]) + "," + str(y_predict[index]) + "\n")

    with open(file_path_y_0, "w+") as f:
        length = len(y_true)
        length_0 = len(y_data_origin) + time_diff
        all_y = []
        for index in np.arange(time_diff):
            all_y.append(y_data_origin[index, 0])

        for index in np.arange(length):
            all_y.append(y_true[index])

        for index in np.arange(length_0):
            f.write(str(all_y[index]) + "\n")

    with open(file_path_time, "w+") as f:
        f.write(str(train_time) + "," + str(predict_time) + "\n")





