import MySQLdb
from matplotlib import pyplot as plt
from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *
import os
import shutil

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

if __name__ == "__main__":
    # 连接数据库
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    sql = "USE power_decreasing"
    cursor = db.cursor()
    cursor.execute(sql)

    sql = "DROP VIEW IF EXISTS power_decreasing_target200"
    cursor = db.cursor()
    cursor.execute(sql)

    # 创建power_decreasing_target200的数据库视窗
    sql = "CREATE VIEW power_decreasing_target200 AS " + "SELECT * FROM target200_rate30 " + "UNION ALL " \
          + "SELECT * FROM target200_rate32 " + "UNION ALL " \
          + "SELECT * FROM target200_rate34 " + "UNION ALL " \
          + "SELECT * FROM target200_rate35"
    cursor = db.cursor()
    cursor.execute(sql)

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

    sql = "DROP VIEW IF EXISTS power_decreasing_target220"
    cursor = db.cursor()
    cursor.execute(sql)

    # 创建power_decreasing_target220的数据库视窗
    sql = "CREATE VIEW power_decreasing_target220 AS " + "SELECT * FROM target220_rate22 " + "UNION ALL " \
          + "SELECT * FROM target220_rate23 " + "UNION ALL " \
          + "SELECT * FROM target220_rate24 " + "UNION ALL " \
          + "SELECT * FROM target220_rate25 " + "UNION ALL " \
          + "SELECT * FROM target220_rate26 " + "UNION ALL " \
          + "SELECT * FROM target220_rate27 " + "UNION ALL " \
          + "SELECT * FROM target220_rate28 "
    cursor = db.cursor()
    cursor.execute(sql)

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

    sql = "DROP VIEW IF EXISTS power_decreasing_target250"
    cursor = db.cursor()
    cursor.execute(sql)

    # 创建power_decreasing_target250的数据库视窗
    sql = "CREATE VIEW power_decreasing_target250 AS " + "SELECT * FROM target250_rate5 " + "UNION ALL " \
          + "SELECT * FROM target250_rate7_5 " + "UNION ALL " \
          + "SELECT * FROM target250_rate10 " + "UNION ALL " \
          + "SELECT * FROM target250_rate12 " + "UNION ALL " \
          + "SELECT * FROM target250_rate13 " + "UNION ALL " \
          + "SELECT * FROM target250_rate14 " + "UNION ALL " \
          + "SELECT * FROM target250_rate15 " + "UNION ALL " \
          + "SELECT * FROM target250_rate16 " + "UNION ALL " \
          + "SELECT * FROM target250_rate17 " + "UNION ALL " \
          + "SELECT * FROM target250_rate18 " + "UNION ALL " \
          + "SELECT * FROM target250_rate20 "
    cursor = db.cursor()
    cursor.execute(sql)

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
    cursor = db.cursor()
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

    # 设置图片存储路径
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    path = grandfather_path + 'Pictures\\power decreasing pre_analysis'

    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)

    power_decreasing_header_en = ['Thermal Power(MW)', 'Electric Power(MW)', '1st Coolant Flow(kg/s)',
                                  '2nd Coolant Flow(kg/s)', '1st HL Temperature' + '(' + str(u'°C') + ')',
                                  '2nd HL Temperature' + '(' + str(u'°C') + ')',
                                  '1st CL Temperature' + '(' + str(u'°C') + ')',
                                  '2nd CL Temperature' + '(' + str(u'°C') + ')', '1st Stream_Pressure(MPa)',
                                  '2nd Stream Pressure(MPa)', '1st SG Level(m)', '2nd SG Level(m)',
                                  '1st SG Flow(kg/s)', '2nd SG Flow(kg/s)',
                                  '1st SG Feed Temper' + '(' + str(u'°C') + ')',
                                  '2nd SG Feed Temper' + '(' + str(u'°C') + ')', '1st Outlet Flow(kg/s)',
                                  '2nd Outlet Flow(kg/s)',
                                  '1st Outlet Temper' + '(' + str(u'°C') + ')',
                                  '2nd Outlet Temper' + '(' + str(u'°C') + ')',
                                  'PZ Pressure(MPa)', 'PZ Level(m)', 'PZ Heat Power(MW)',
                                  'PZ Steam Temper' + '(' + str(u'°C') + ')',
                                  'PZ Water Temper' + '(' + str(u'°C') + ')']

    power_decreasing_header_zn = ['热功率', '电功率', '主回路1冷却剂流量', '主回路2冷却剂流量', '主回路1热管段冷却剂温度',
                                  '主回路2热管段冷却剂温度', '主回路1冷管段冷却剂温度', '主回路2冷管段冷却剂温度',
                                  '主回路1蒸发器二次侧压力', '主回路2蒸发器二次侧压力', '主回路蒸发器水位', '主回路2蒸发器水位',
                                  '主回路1蒸发器给水流量', '主回路2蒸发器给水流量', '主回路1蒸发器给水温度', '主回路2蒸发器给水温度',
                                  '主回路1蒸发器蒸汽出口流量', '主回路2蒸发器蒸汽出口流量', '主回路1蒸发器蒸汽出口温度',
                                  '主回路2蒸发器蒸汽出口温度', '稳压器压力', '稳压器水位', '稳压器加热功率', '稳压器蒸汽空间温度',
                                  '稳压器水空间温度']

    # 绘制初始数据关系曲线
    array_length = len(power_decreasing_array[0])
    for i_value in range(array_length):
        if i_value == 0:
            mean = np.mean(power_decreasing_array[..., 0])
            std = np.std(power_decreasing_array[..., 0])
            x_distribution = np.linspace(0, 1500, 1500)
            y_distribution = normal_distribution(x_distribution, mean, std)
            plt.figure(figsize=(2.2, 1.65))
            plt.plot(x_distribution, y_distribution, linewidth='1', color='black')
            set_picture_EN(power_decreasing_header_en[0], power_decreasing_header_en[0])
            plt.savefig(path + '\\' + '加热功率正态分布曲线.pdf', dip=300, bbox_inches='tight')
            plt.close()
        else:
            x_distribution_1 = power_decreasing_array_target200[..., 0]
            y_distribution_1 = power_decreasing_array_target200[..., i_value]
            x_distribution_2 = power_decreasing_array_target220[..., 0]
            y_distribution_2 = power_decreasing_array_target220[..., i_value]
            x_distribution_3 = power_decreasing_array_target250[..., 0]
            y_distribution_3 = power_decreasing_array_target250[..., i_value]
            plt.figure(figsize=(2.2, 1.65))
            p1 = plt.scatter(x_distribution_1, y_distribution_1, s=0.1)
            p2 = plt.scatter(x_distribution_2, y_distribution_2, s=0.1)
            p3 = plt.scatter(x_distribution_3, y_distribution_3, s=0.1)
            plt.legend([p1, p2, p3], ['target200', 'target220', 'target250'], frameon=False,
                       prop={'family': 'Times New Roman', 'size': 8})
            set_picture_EN(power_decreasing_header_en[0], power_decreasing_header_en[i_value])
            plt.savefig(path + '\\' + '加热功率与' + power_decreasing_header_zn[i_value] + '关系曲线图' + '.pdf', dip=300,
                        bbox_inches='tight')
            plt.close()

    for i_value in range(array_length):
        if i_value == 1:
            mean = np.mean(power_decreasing_array[..., 1])
            std = np.std(power_decreasing_array[..., 1])
            x_distribution = np.linspace(0, 1500, 1500)
            y_distribution = normal_distribution(x_distribution, mean, std)
            plt.figure(figsize=(2.2, 1.65))
            plt.plot(x_distribution, y_distribution, linewidth='1', color='black')
            set_picture_EN(power_decreasing_header_en[1], power_decreasing_header_en[1])
            plt.savefig(path + '\\' + '电功率正态分布曲线.pdf', dip=300, bbox_inches='tight')
            plt.close()
        else:

            x_distribution_1 = power_decreasing_array_target200[..., 1]
            y_distribution_1 = power_decreasing_array_target200[..., i_value]
            x_distribution_2 = power_decreasing_array_target220[..., 1]
            y_distribution_2 = power_decreasing_array_target220[..., i_value]
            x_distribution_3 = power_decreasing_array_target250[..., 1]
            y_distribution_3 = power_decreasing_array_target250[..., i_value]
            plt.figure(figsize=(2.2, 1.65))
            p1 = plt.scatter(x_distribution_1, y_distribution_1, s=0.1)
            p2 = plt.scatter(x_distribution_2, y_distribution_2, s=0.1)
            p3 = plt.scatter(x_distribution_3, y_distribution_3, s=0.1)
            plt.legend([p1, p2, p3], ['target200', 'target220', 'target250'], frameon=False,
                       prop={'family': 'Times New Roman', 'size': 8})
            set_picture_EN(power_decreasing_header_en[1], power_decreasing_header_en[i_value])
            plt.savefig(path + '\\' + '电功率与' + power_decreasing_header_zn[i_value] + '关系曲线图' + '.pdf', dip=300,
                        bbox_inches='tight')
            plt.close()

    db.close()
