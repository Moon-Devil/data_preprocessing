import MySQLdb
import os
import numpy as np
import re


data_set_length = 301


def read_data_from_database(y_index, target_list) -> object:
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
        temp_y_data = temp_power_decreasing_array[: data_set_length, y_index]

        power_decreasing_data_set.append([target, temp_y_data])

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
    y_data_list = []

    for i_value in np.arange(len(power_decreasing_data_set)):
        if power_decreasing_data_set[i_value][0] in target_list:
            y_data_list.append(power_decreasing_data_set[i_value][1])

    y_data_array = np.array(y_data_list).T
    return y_data_array


def history_record(file_name, y_data_record):
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = os.path.join(grandfather_path, "Calculations\\Auto_deep_learning")
    file_path = os.path.join(path, file_name + "_y_record.txt")

    with open(file_path, "w+") as f:
        shape = np.shape(y_data_record)
        columns = shape[0]
        rows = shape[1]

        for i_index in np.arange(columns):
            for j_index in np.arange(rows):
                if j_index != rows - 1:
                    f.write(str(y_data_record[i_index][j_index]) + ",")
                else:
                    f.write(str(y_data_record[i_index][j_index]) + "\n")


y_data = read_data_from_database(4, [250])
history_record("y_250_CT", y_data)

print("done...")
