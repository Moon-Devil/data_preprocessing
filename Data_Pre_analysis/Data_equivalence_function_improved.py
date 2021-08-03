import MySQLdb
import numpy as np
import tensorflow as tf
import time
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


def read_data_from_database(database_name) -> object:
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


def build_model(dataset, index, dataset_name, penalty, flag) -> object:
    y_data = dataset[:, index]
    x_data = np.delete(dataset, index, axis=1)

    x_data_shape = np.shape(x_data)
    dimension = x_data_shape[1]

    start_train = time.time()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(dimension, activation=tf.keras.activations.relu, input_shape=(dimension,),
                                    kernel_regularizer=tf.keras.regularizers.l2(penalty)))
    model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu,
                                    kernel_regularizer=tf.keras.regularizers.l2(penalty)))
    model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu,
                                    kernel_regularizer=tf.keras.regularizers.l2(penalty)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse, metrics=[tf.keras.metrics.mse])
    model.summary()

    history = model.fit(x_data, y_data, epochs=400, batch_size=16, validation_split=0.2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                                                    verbose=2)])
    end_train = time.time()

    start_predict = time.time()
    y_predict = model.predict(x_data[:301, ])
    end_predict = time.time()
    y_true = y_data[:301, ]

    train_time = end_train - start_train
    predict_time = end_predict - start_predict

    # with open(file, flag) as f:
    #     f.write(dataset_name + '\n')
    #     f.write('train_time\t' + str(train_time) + '\n')
    #     f.write('predict_time\t' + str(predict_time) + '\n')
    #
    #     temp = history.history['loss']
    #     length = len(temp)
    #     f.write('loss\t')
    #     for i_value in np.arange(length):
    #         if i_value != length - 1:
    #             f.write(str(temp[i_value]) + ',')
    #         else:
    #             f.write(str(temp[i_value]) + '\n')
    #
    #     temp = history.history['mean_squared_error']
    #     length = len(temp)
    #     f.write('mean_squared_error\t')
    #     for i_value in np.arange(length):
    #         if i_value != length - 1:
    #             f.write(str(temp[i_value]) + ',')
    #         else:
    #             f.write(str(temp[i_value]) + '\n')
    #
    #     temp = history.history['val_loss']
    #     length = len(temp)
    #     f.write('val_loss\t')
    #     for i_value in np.arange(length):
    #         if i_value != length - 1:
    #             f.write(str(temp[i_value]) + ',')
    #         else:
    #             f.write(str(temp[i_value]) + '\n')
    #
    #     temp = history.history['val_mean_squared_error']
    #     length = len(temp)
    #     f.write('val_mean_squared_error\t')
    #     for i_value in np.arange(length):
    #         if i_value != length - 1:
    #             f.write(str(temp[i_value]) + ',')
    #         else:
    #             f.write(str(temp[i_value]) + '\n')
    #
    #     length = len(y_predict)
    #     f.write('y_predict\t')
    #     for i_value in np.arange(length):
    #         if i_value != length - 1:
    #             f.write(str(y_predict[i_value][0]) + ',')
    #         else:
    #             f.write(str(y_predict[i_value][0]) + '\n')
    #
    #     length = len(y_true)
    #     f.write('y_true\t')
    #     for i_value in np.arange(length):
    #         if i_value != length - 1:
    #             f.write(str(y_true[i_value]) + ',')
    #         else:
    #             f.write(str(y_true[i_value]) + '\n')

    return y_predict, y_true


def add_noise(dataset, noise_ratio) -> object:
    dataset_shape = np.shape(dataset)
    dataset_length = dataset_shape[0]
    dataset_dimension = dataset_shape[1]
    random_matrix = np.random.random((dataset_length, dataset_dimension)) * 2 - 1
    dataset_noise = np.zeros((dataset_length, dataset_dimension))

    for j_value in np.arange(dataset_dimension):
        lists = dataset[:, j_value]
        mean = np.mean(lists)
        for i_value in np.arange(dataset_length):
            dataset_noise[i_value][j_value] = dataset[i_value][j_value] + mean * noise_ratio * \
                                              random_matrix[i_value][j_value]

    return dataset
