import MySQLdb
import numpy as np


def read_data():
    data_dictionary = {}

    db = MySQLdb.connect(host="localhost", user="root", passwd="1111", charset="utf8")
    cursor = db.cursor()

    sql = "USE swing"
    cursor.execute(sql)

    cursor.execute("show tables")
    table_list = [tables[0] for tables in cursor.fetchall()]

    for table in table_list:
        sql = "SELECT * FROM " + table
        cursor.execute(sql)

        results = cursor.fetchall()

        shape = np.shape(results)
        dataset_length = shape[0]
        data_dimension = shape[1]

        data_array = np.zeros((dataset_length, data_dimension - 1), dtype=np.float32)

        time_diff = 0
        for i_index in range(dataset_length):
            data_array[i_index, 0] = time_diff
            time_diff = time_diff + 0.0625
            for j_index in range(2, data_dimension):
                data_array[i_index, j_index - 1] = results[i_index][j_index]

        data_dictionary[table] = data_array

    print("数据读取已完成......")
    return data_dictionary


if __name__ == '__main__':
    data = read_data()
