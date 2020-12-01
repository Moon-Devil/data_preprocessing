import os
import numpy as np
import sys


# 从文件中读取数据
def read_data(path, keywords, keywords_index, start_index, end_index, index_margin) -> object:
    data_list = []  # 数据列表缓存
    data_dict = dict()  # 数据字典缓存
    for i_value in np.arange(start_index, end_index, index_margin):
        i_value = i_value / 1000
        i_value = int(i_value) if i_value == int(i_value) else i_value
        file = path + '\\' + str(keywords[keywords_index]) + str(i_value) + '.txt'  # 设置读取文件路径
        temp_array = np.array(keywords[keywords_index].split(' '))

        # 若该文件存在，则读取数据
        if os.path.exists(file):
            array_name = temp_array[0] + '_' + temp_array[1] + '_' + temp_array[2] + '_' + temp_array[3] + str(i_value)
            temp_data = []
            with open(file, "r") as f:
                for line in f.readlines():
                    temp_value = line.strip('\n').strip(' ').split(',')
                    temp_list = list(map(float, temp_value))
                    temp_data.append(temp_list)
                data_dict[array_name] = temp_data
                data_list.append(temp_data)
        else:
            pass
    return data_list, data_dict


# 同类数据汇总成一个数组
def data_summary(data_list) -> object:
    temp_list = []
    list_length_1 = len(data_list)
    list_row = 0
    list_number = len(data_list[0][0][0])

    for i_value in range(list_length_1):
        temp_value_1 = len(data_list[i_value])
        for j_value in range(temp_value_1):
            temp_value_2 = len(data_list[i_value][j_value])
            list_row += temp_value_2
            for k_value in range(temp_value_2):
                if list_number != len(data_list[i_value][j_value][k_value]):
                    print("Waring!!! 各组数据长度不相等，请检查数据......")
                    sys.exit(0)
                temp_list.append(data_list[i_value][j_value][k_value])

    data_array = np.array(temp_list).reshape(list_row, list_number)
    return data_array


# 若同名数据表存在，则卸载该同名数据表
def drop_table(path, keywords, keywords_index, start_index, end_index, margin) -> object:
    temp_string: str = ""
    for i_value in np.arange(start_index, end_index, margin):
        temp_value = i_value / 1000
        temp_value = int(temp_value) if temp_value == int(temp_value) else temp_value
        file = path + '\\' + keywords[keywords_index] + str(temp_value) + '.txt'
        temp_array = np.array(keywords[keywords_index].split(' '))

        if os.path.exists(file):
            if temp_value == int(temp_value):
                table_name = temp_array[0] + '_' + temp_array[1] + '_' + temp_array[2] + '_' \
                             + temp_array[3] + str(temp_value)
            else:
                temp_string_1 = str(temp_value).split('.')
                table_name = temp_array[0] + '_' + temp_array[1] + '_' + temp_array[2] \
                             + '_' + temp_array[3] + temp_string_1[0] + '_' + temp_string_1[1]

            if i_value != end_index - 1:
                temp_string += table_name + ', '
            else:
                temp_string += table_name

    return temp_string


# 创建target单数据表
temp_string_head = "CREATE TABLE "
temp_string_tail = "(" \
                   + "id int NOT NULL AUTO_INCREMENT, " \
                   + "thermal_power double NULL , " \
                   + "electric_power double NULL , " \
                   + "coolant_flow_primary_circuit double NULL , " \
                   + "coolant_flow_secondary_circuit double NULL , " \
                   + "hot_leg_temperature_primary double NULL , " \
                   + "hot_leg_temperature_secondary double NULL , " \
                   + "cold_leg_temperature_primary double NULL , " \
                   + "cold_leg_temperature_secondary double NULL , " \
                   + "pressure_steam_generator_primary double NULL , " \
                   + "pressure_steam_generator_secondary double NULL , " \
                   + "water_level_primary double NULL , " \
                   + "water_level_secondary double NULL , " \
                   + "feed_water_flow_steam_generator_1 double NULL , " \
                   + "feed_water_flow_steam_generator_2 double NULL , " \
                   + "feed_water_flow_steam_generator_3 double NULL , " \
                   + "feed_water_flow_steam_generator_4 double NULL , " \
                   + "feed_water_temp_steam_generator_1 double NULL , " \
                   + "feed_water_temp_steam_generator_2 double NULL , " \
                   + "feed_water_temp_steam_generator_3 double NULL , " \
                   + "feed_water_temp_steam_generator_4 double NULL , " \
                   + "steam_outlet_flow_primary double NULL , " \
                   + "steam_outlet_flow_secondary double NULL , " \
                   + "steam_outlet_temperature_primary double NULL , " \
                   + "steam_outlet_temperature_secondary double NULL , " \
                   + "pressurizer_pressure double NULL , " \
                   + "pressurizer_water_level double NULL , " \
                   + "pressurizer_heat_power double NULL , " \
                   + "pressurizer_steam_space_temperature double NULL , " \
                   + "pressurizer_water_space_temperature double NULL , " \
                   + "PRIMARY KEY (id)" \
                   + ")ENGINE=InnoDB;"

temp_insert_head = "INSERT INTO "
temp_insert_tail = " (thermal_power, electric_power, coolant_flow_primary_circuit,\
                    coolant_flow_secondary_circuit, hot_leg_temperature_primary,\
                    hot_leg_temperature_secondary, cold_leg_temperature_primary,\
                    cold_leg_temperature_secondary, pressure_steam_generator_primary,\
                    pressure_steam_generator_secondary, water_level_primary,\
                    water_level_secondary, feed_water_flow_steam_generator_1,\
                    feed_water_flow_steam_generator_2, feed_water_flow_steam_generator_3,\
                    feed_water_flow_steam_generator_4, feed_water_temp_steam_generator_1,\
                    feed_water_temp_steam_generator_2, feed_water_temp_steam_generator_3,\
                    feed_water_temp_steam_generator_4, steam_outlet_flow_primary,\
                    steam_outlet_flow_secondary, steam_outlet_temperature_primary,\
                    steam_outlet_temperature_secondary, pressurizer_pressure,\
                    pressurizer_water_level, pressurizer_heat_power,\
                    pressurizer_steam_space_temperature, pressurizer_water_space_temperature\
                    ) VALUES "


def create_single_table(path, keywords, keywords_index, start_index, end_index, margin, db,
                        data_dict, insert_head=temp_insert_head,
                        insert_tail=temp_insert_tail):
    for i_value in np.arange(start_index, end_index, margin):
        temp_value = i_value / 1000
        temp_value = int(temp_value) if temp_value == int(temp_value) else temp_value
        file = path + '\\' + keywords[keywords_index] + str(temp_value) + '.txt'
        temp_array = np.array(keywords[keywords_index].split(' '))
        array_name = temp_array[0] + '_' + temp_array[1] + '_' + temp_array[2] + '_' + temp_array[3] + str(temp_value)

        if os.path.exists(file):
            if temp_value == int(temp_value):
                table_name = temp_array[0] + '_' + temp_array[1] + '_' + temp_array[2] + '_' \
                             + temp_array[3] + str(temp_value)
            else:
                temp_string_1 = str(temp_value).split('.')
                table_name = temp_array[0] + '_' + temp_array[1] + '_' + temp_array[2] + '_' + temp_array[3] \
                             + temp_string_1[0] + '_' + temp_string_1[1]

            sql = temp_string_head + table_name + temp_string_tail
            cursor = db.cursor()
            cursor.execute(sql)

            temp_insert = insert_head + table_name + insert_tail + "("
            temp_list = data_dict[array_name]
            for j_value in range(len(temp_list)):
                temp_string_value = ""
                temp_length = len(temp_list[j_value])
                for k_value in range(temp_length):
                    if k_value != temp_length - 1:
                        temp_string_value += str(temp_list[j_value][k_value]) + ", "
                    else:
                        temp_string_value += str(temp_list[j_value][k_value])
                sql = temp_insert + temp_string_value + ")"
                cursor = db.cursor()
                try:
                    cursor.execute(sql)
                    db.commit()
                except Warning:
                    db.rollback()


# 创建数据总表
def create_summary_table(data_array, db, insert_head=temp_insert_head, insert_tail=temp_insert_tail,
                         string_head=temp_string_head, string_tail=temp_string_tail):
    sql = "DROP TABLE IF EXISTS all_data"
    cursor = db.cursor()
    cursor.execute(sql)
    sql = string_head + 'all_data' + string_tail
    cursor = db.cursor()
    cursor.execute(sql)

    row = len(data_array)
    for i_value in range(row):
        data_string = ""
        length = len(data_array[i_value])
        for j_value in range(length):
            if j_value != length - 1:
                data_string += str(data_array[i_value][j_value]) + ", "
            else:
                data_string += str(data_array[i_value][j_value])
        sql = insert_head + "all_data" + insert_tail + "(" + data_string + ")"
        cursor = db.cursor()
        try:
            cursor.execute(sql)
            db.commit()
        except Warning:
            db.rollback()
