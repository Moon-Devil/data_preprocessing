from MAML_function import *

x_dataframe, y_dataframe, y_data_origin = read_data_from_database(24, 3, [200, 220, 250])
history, y_true, y_predict, train_time, predict_time = MAML_function(x_dataframe, y_dataframe, 38, 24, 3, 201, 0, 20)
history_record("pressurizer_water_3_3_MAML_", history, y_true, y_predict, y_data_origin, train_time, predict_time, 3)

# history, y_true, y_predict, train_time, predict_time = MAML_origin_function(x_dataframe, y_dataframe, 68, 43, 201, 0,
#                                                                             20)
# history_record("pressurizer_water_0_MAML_", history, y_true, y_predict, y_data_origin, train_time, predict_time, 1)
