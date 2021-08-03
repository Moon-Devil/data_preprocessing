import numpy as np
import MySQLdb
import pandas as pd
import os


def read_data_from_database(database_name, database_number, database_length, flag) -> object:
    host = "localhost"  # 数据库地址
    username = "root"  # 数据库用户名
    passwd = "1111"  # 数据库密码
    database = "mysql"  # 数据库类型

    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    sql = "USE " + database_name
    cursor.execute(sql)

    sql = "show tables;"
    cursor.execute(sql)
    table_names = []
    results = cursor.fetchall()
    for result in results:
        table_names.append(result[0])

    data = 0
    if flag == "normal":
        start = 4
        end = len(table_names) - 2
    else:
        start = 1
        end = database_number + 1

    for index in np.arange(start, end):
        sql = "select * from " + table_names[index]
        cursor.execute(sql)
        results = cursor.fetchall()

        data_list = []
        column_index = 0
        for row in results:
            temp = []
            row_length = len(row)
            for j_value in np.arange(row_length):
                temp.append(row[j_value])
            data_list.append(temp)
            column_index = column_index + 1

        column_length = len(data_list)
        if column_length != 0 and column_length > database_length:
            row_length = len(data_list[0])
            data_array = np.zeros(shape=(database_length, row_length))
            for column in np.arange(database_length):
                for row in np.arange(row_length):
                    data_array[column][row] = data_list[column][row]

            data_array = np.delete(data_array, [0, 15, 16, 19, 20], axis=1)

            if index == start:
                data = data_array
            else:
                data = np.vstack((data, data_array))
        else:
            print("The length of database is smaller than " + str(database_length) + ".")

    table_header = ['thermal_power', 'electric_power',
                    'coolant_flow_primary_circuit', 'coolant_flow_secondary_circuit',
                    'hot_leg_temperature_primary', 'hot_leg_temperature_secondary',
                    'cold_leg_temperature_primary', 'cold_leg_temperature_secondary',
                    'pressure_steam_generator_primary', 'pressure_steam_generator_secondary',
                    'water_level_primary', 'water_level_secondary',
                    'feed_water_flow_steam_generator_1', 'feed_water_flow_steam_generator_2',
                    'feed_water_temp_steam_generator_1', 'feed_water_temp_steam_generator_2',
                    'steam_outlet_flow_primary', 'steam_outlet_flow_secondary',
                    'steam_outlet_temperature_primary', 'steam_outlet_temperature_secondary',
                    'pressurizer_pressure', 'pressurizer_water_level',
                    'pressurizer_heat_power', 'pressurizer_steam_space_temperature',
                    'pressurizer_water_space_temperature']

    data_dataframe = pd.DataFrame(data, columns=table_header)
    return data_dataframe


def read_data() -> object:
    normal_data = read_data_from_database("power_decreasing", 0, 301, "normal")
    normal_data = normal_data.drop(normal_data.tail(20).index)

    prz_liquid = read_data_from_database("prz_liquid_space_leak", 20, 301, 0)

    prz_vapour = read_data_from_database("prz_vapour_space_leak", 20, 301, 0)

    rcs_cl_1 = read_data_from_database("rcs_cl_loca_1", 15, 201, 0)
    rcs_cl_2 = read_data_from_database("rcs_cl_loca_2", 15, 201, 0)
    rcs_cl = pd.concat([rcs_cl_1, rcs_cl_2], ignore_index=True)

    rcs_hl_1 = read_data_from_database("rcs_hl_loca_1", 15, 201, 0)
    rcs_hl_2 = read_data_from_database("rcs_hl_loca_2", 15, 201, 0)
    rcs_hl = pd.concat([rcs_hl_1, rcs_hl_2], ignore_index=True)

    sg_2nd = read_data_from_database("sg_2nd_side_leak", 20, 301, 0)

    sgtr60 = read_data_from_database("sgtr60_power", 15, 201, 0)
    sgtr100 = read_data_from_database("sgtr_power", 15, 201, 0)
    sgtr = pd.concat([sgtr60, sgtr100], ignore_index=True)

    anomaly_data_pandas = pd.concat([prz_liquid, prz_vapour, rcs_cl, rcs_hl, sg_2nd, sgtr], axis=1)

    table_header = ['thermal_power', 'electric_power',
                    'coolant_flow_primary_circuit', 'coolant_flow_secondary_circuit',
                    'hot_leg_temperature_primary', 'hot_leg_temperature_secondary',
                    'cold_leg_temperature_primary', 'cold_leg_temperature_secondary',
                    'pressure_steam_generator_primary', 'pressure_steam_generator_secondary',
                    'water_level_primary', 'water_level_secondary',
                    'feed_water_flow_steam_generator_1', 'feed_water_flow_steam_generator_2',
                    'feed_water_temp_steam_generator_1', 'feed_water_temp_steam_generator_2',
                    'steam_outlet_flow_primary', 'steam_outlet_flow_secondary',
                    'steam_outlet_temperature_primary', 'steam_outlet_temperature_secondary',
                    'pressurizer_pressure', 'pressurizer_water_level',
                    'pressurizer_heat_power', 'pressurizer_steam_space_temperature',
                    'pressurizer_water_space_temperature']

    first_index = [index for index in ["prz_liquid", "prz_vapour", "rcs_cl", "rcs_hl", "sg_2nd", "sgtr"]
                   for _ in np.arange(25)]
    second_index = [table_header[index] for _ in np.arange(6) for index in np.arange(len(table_header))]
    index = [first_index, second_index]

    anomaly_data = pd.DataFrame(anomaly_data_pandas.values, columns=index)
    anomaly_data = anomaly_data.drop(anomaly_data.tail(30).index)

    return normal_data, anomaly_data


def write_to_text(filename, data_lists, flag):
    father_path = os.path.abspath('../../..') + 'Calculations\\'
    result_directory = father_path + 'data_reduction_paper\\'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    file = os.path.join(result_directory, filename + ".txt")
    with open(file, flag) as f:
        length = len(data_lists)
        for i_index in np.arange(length):
            if i_index != length - 1:
                f.write(str(data_lists[i_index]) + ",")
            else:
                f.write(str(data_lists[i_index]) + "\n")


def clear_file(filename):
    if filename is not None:
        father_path = os.path.abspath('../../..') + 'Calculations\\'
        result_directory = father_path + 'data_reduction_paper\\'
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        file = os.path.join(result_directory, filename + ".txt")
        if os.path.exists(file):
            os.remove(file)


def write_history(filename, history, epochs):
    father_path = os.path.abspath('../../..') + 'Calculations\\'
    result_directory = father_path + 'data_reduction_paper\\'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    file = os.path.join(result_directory, filename + ".txt")
    with open(file, "a+") as f:
        for i_index in range(epochs):
            f.write(str(history.history['loss'][i_index]) + "," +
                    str(history.history['mean_squared_error'][i_index]) + "," +
                    str(history.history['val_loss'][i_index]) + "," +
                    str(history.history['val_mean_squared_error'][i_index]) + "\n")
