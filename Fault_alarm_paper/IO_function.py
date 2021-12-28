import csv

import MySQLdb
import os

import numpy
import numpy as np
import re
import pandas as pd

data_set_length = 301


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


def classified_read_data_from_database(database_name, database_number, database_length, flag) -> object:
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
    normal_data = classified_read_data_from_database("power_decreasing", 0, 301, "normal")
    normal_data = normal_data.drop(normal_data.tail(20).index)

    prz_liquid = classified_read_data_from_database("prz_liquid_space_leak", 20, 301, 0)

    prz_vapour = classified_read_data_from_database("prz_vapour_space_leak", 20, 301, 0)

    rcs_cl_1 = classified_read_data_from_database("rcs_cl_loca_1", 15, 201, 0)
    rcs_cl_2 = classified_read_data_from_database("rcs_cl_loca_2", 15, 201, 0)
    rcs_cl = pd.concat([rcs_cl_1, rcs_cl_2], ignore_index=True)

    rcs_hl_1 = classified_read_data_from_database("rcs_hl_loca_1", 15, 201, 0)
    rcs_hl_2 = classified_read_data_from_database("rcs_hl_loca_2", 15, 201, 0)
    rcs_hl = pd.concat([rcs_hl_1, rcs_hl_2], ignore_index=True)

    sg_2nd = classified_read_data_from_database("sg_2nd_side_leak", 20, 301, 0)

    sgtr60 = classified_read_data_from_database("sgtr60_power", 15, 201, 0)
    sgtr100 = classified_read_data_from_database("sgtr_power", 15, 201, 0)
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


def best_record_csv(path, file_name, result_list):
    file_path_y_predict = os.path.join(path, file_name + "_y_predict.csv")

    file_path_train_loss = os.path.join(path, file_name + "_train_loss.csv")
    file_path_train_mse = os.path.join(path, file_name + "_train_mse.csv")
    file_path_train_val_loss = os.path.join(path, file_name + "_train_val_loss.csv")
    file_path_train_val_mse = os.path.join(path, file_name + "_train_val_mse.csv")

    file_path_eva_loss = os.path.join(path, file_name + "_eva_loss.csv")
    file_path_eva_accuracy = os.path.join(path, file_name + "_eva_accuracy.csv")

    file_path_train_time = os.path.join(path, file_name + "_train_time.csv")
    file_path_test_time = os.path.join(path, file_name + "_test_time.csv")

    file_path_model_size = os.path.join(path, file_name + "_model_size.csv")

    file_path = [file_path_y_predict, file_path_train_loss, file_path_train_mse, file_path_train_val_loss,
                 file_path_train_val_mse, file_path_eva_loss, file_path_eva_accuracy, file_path_train_time,
                 file_path_test_time, file_path_model_size]

    file_length = len(file_path)
    for index in range(file_length):
        with open(file_path[index], "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result_list[index])


def modified_bert_record_csv(path, file_name, result_list):
    file_path_y_predict = os.path.join(path, file_name + "_y_predict.csv")

    file_path_train_loss = os.path.join(path, file_name + "_train_loss.csv")
    file_path_train_mse = os.path.join(path, file_name + "_train_mse.csv")
    file_path_train_val_loss = os.path.join(path, file_name + "_train_val_loss.csv")
    file_path_train_val_mse = os.path.join(path, file_name + "_train_val_mse.csv")

    file_path_train_time = os.path.join(path, file_name + "_train_time.csv")
    file_path_test_time = os.path.join(path, file_name + "_test_time.csv")

    file_path_model_size = os.path.join(path, file_name + "_model_size.csv")

    file_path = [file_path_y_predict, file_path_train_loss, file_path_train_mse, file_path_train_val_loss,
                 file_path_train_val_mse, file_path_train_time, file_path_test_time, file_path_model_size]

    file_length = len(file_path)
    for index in range(file_length):
        with open(file_path[index], "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result_list[index])
