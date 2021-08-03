import MySQLdb
import numpy as np
import os


grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\Data_equivalence'
if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\Data_equivalence_improved_SG_stream_outlet_flow_rate.txt"
if os.path.exists(file):
    os.remove(file)

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


def read_data_from_database(database_name):
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    sql = "USE power_decreasing"
    cursor.execute(sql)

    sql = "SELECT * from " + database_name
    cursor.execute(sql)
    results = cursor.fetchall()

    data_dict = dict()
    for i_value in range(len(table_header)):
        data_dict[table_header[i_value]] = []

    for row in results:
        for i_value in range(len(table_header)):
            data_dict[table_header[i_value]].append(row[i_value])

    array_column = len(data_dict[table_header[0]])
    array_row = len(table_header)

    temp_data_array = np.zeros((array_column, array_row))
    for i_value in range(array_column):
        for j_value in range(array_row):
            temp_data_array[i_value][j_value] = data_dict[table_header[j_value]][i_value]

    power_decreasing_array = np.delete(temp_data_array, [0, 15, 16, 19, 20], axis=1)
    return power_decreasing_array
