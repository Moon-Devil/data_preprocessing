from Neural_network_function import *

time_windows = 20
y_index = 16

normal_data, _ = read_data()
normal_data = normal_data.values

offset = 0
y_data = normal_data[offset: 172 + offset, y_index]
x_data = (np.delete(normal_data, y_index, axis=1))[offset: 172 + offset, ]
NeuralNetwork_train(x_data, y_data, time_windows, "200", "SG")

offset = 322
y_data = normal_data[offset: 172 + offset, y_index]
x_data = (np.delete(normal_data, y_index, axis=1))[offset: 172 + offset, ]
NeuralNetwork_train(x_data, y_data, time_windows, "220", "SG")

print("done...")

