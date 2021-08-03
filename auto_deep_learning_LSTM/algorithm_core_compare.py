from algorithm_pool import *

x_dataframe, y_dataframe, y_data_origin = read_data_from_database(16, 10, [200, 220, 250])

history, y_true, y_predict, train_time, predict_time = NN_function(x_dataframe, y_dataframe, 3, 100, 201, 0, 20)
history_record("steam_outlet_10_NN", history, y_true, y_predict, y_data_origin, train_time, predict_time, 10)


# history, y_true, y_predict, train_time, predict_time = LSTM_function(x_dataframe, y_dataframe, 50, 100, 201, 0, 2)
# history_record("pressurizer_water_LSTM", history, y_true, y_predict, y_data_origin, train_time, predict_time, 1)

print("done...")
