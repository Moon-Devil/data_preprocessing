import numpy as np
import os
from datetime import datetime
import MySQLdb


def read_dat(file_name):
    time_label = []
    test_time = []
    data = []

    with open(file_name, "r") as file:
        file.readline()
        line = file.readline()

        while line:
            data_line = line.split()

            time_label.append(data_line[0])

            test_time.append(datetime.strptime(data_line[1], '%H:%M:%S'))

            data_temp = data_line[2:]
            data_list = [np.float32(temp) for temp in data_temp]
            data.append(data_list)

            line = file.readline()

    time_label = np.array(time_label)
    test_time = np.array(test_time)
    data = np.array(data)

    return time_label, test_time, data


def read_file_name(file_dir):
    file_name = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.dat':
                file_name.append(os.path.join(root, file))

    return file_name


def write_sql(file_names, database_name, data_names):
    db = MySQLdb.connect(host="localhost", user="root", passwd="1111", charset="utf8")

    for file_name in file_names:
        time_label, time, data = read_dat(file_name)

        cursor = db.cursor()
        sql = "CREATE DATABASE IF NOT EXISTS " + database_name + " DEFAULT CHARSET utf8 COLLATE utf8_general_ci"
        cursor.execute(sql)

        sql = "USE " + database_name
        cursor.execute(sql)

        table_name = ((file_name.split('\\'))[-1].split('.'))[0]
        sql = "DROP TABLE IF EXISTS " + table_name
        cursor.execute(sql)

        sql = "CREATE TABLE " + table_name + " (" + data_names[0] + " CHAR(10), " + data_names[1] + " DATETIME(6), "
        data_names_length = len(data_names)
        for index in range(2, data_names_length - 1):
            sql = sql + data_names[index] + " FLOAT, "
        sql = sql + data_names[data_names_length - 1] + " FLOAT)"
        cursor.execute(sql)

        sql_header = "INSERT INTO " + table_name + " ("
        for index in range(data_names_length - 1):
            sql_header = sql_header + data_names[index] + ", "
        sql_header = sql_header + data_names[data_names_length - 1] + ") VALUES ("

        data_length = len(time_label)
        data_dimension = np.shape(data)[1]
        for i_index in range(data_length):
            sql = sql_header + "'" + time_label[i_index] + "'" + ", " + "'" + str(time[i_index]) + "'" + ", "
            for j_index in range(data_dimension - 1):
                sql = sql + str(data[i_index, j_index]) + ", "

            sql = sql + str(data[i_index, data_dimension - 1]) + ")"
            try:
                cursor.execute(sql)
                db.commit()
            except IOError:
                db.rollback()
    db.close()
    print("已完成数据输入......")


if __name__ == '__main__':
    folder_name = '2014-09-02'
    data_names_ = ['Label', 'Date', 'Flow', 'PressureDifference14', 'PressureDifference45', 'Temperature3',
                   'Temperature5', 'Temperature9', 'Temperature11', 'Temperature13', 'Temperature15', 'Temperature17',
                   'TemperatureInlet', 'TemperatureOutlet', 'TemperaturePreHeater', 'SystemPressure', 'InletPressure',
                   'OutletPressure']
    pwd = os.getcwd()
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
    grand_father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + "..")

    path = os.path.join(grand_father_path, 'Source Data\\Swing\\') + folder_name
    file_names_ = read_file_name(path)
    write_sql(file_names_, 'swing', data_names_)
