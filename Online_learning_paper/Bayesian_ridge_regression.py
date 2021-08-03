from Bayesian_ridge_regression_function import *

time_windows = 20
y_index = 24

normal_data, _ = read_data()
normal_data = normal_data.values

offset = 0
y_data = normal_data[offset: 172 + offset, y_index]
x_data = (np.delete(normal_data, y_index, axis=1))[offset: 172 + offset, ]
Bayesian_train(x_data, y_data, time_windows, "200", "PR")

offset = 322
y_data = normal_data[offset: 172 + offset, y_index]
x_data = (np.delete(normal_data, y_index, axis=1))[offset: 172 + offset, ]
Bayesian_train(x_data, y_data, time_windows, "220", "PR")

print("done...")
