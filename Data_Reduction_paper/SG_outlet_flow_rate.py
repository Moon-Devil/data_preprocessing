from data_reduction_function import PCA_function
from sklearn import gaussian_process
from IO_function import *
import time


index = 24
dimension = 15
filename = "PR_water_temperature_30"
clear_file(filename)

x_data, y_data = PCA_function(index, dimension)
model = gaussian_process.GaussianProcessRegressor()
model.fit(x_data, y_data)

x_test = x_data[:301, ]
y_test = y_data[:301]

start_predict_time = time.time()
y_predict = model.predict(x_test)
end_predict_time = time.time()

predict_time = end_predict_time - start_predict_time

for i_index in range(301):
    error = abs(y_test[i_index] - y_predict[i_index]) / y_test[i_index]
    data_lists = [y_test[i_index], y_predict[i_index], error]
    write_to_text(filename, data_lists, "a+")

data_lists = [predict_time]
write_to_text(filename, data_lists, "a+")
