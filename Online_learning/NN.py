from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from NN_function import *


pressurizer_water_y = power_decreasing[..., -3]
pressurizer_water_x = np.delete(power_decreasing, -3, axis=1)

x_train_200_30 = pressurizer_water_x[:241, ...]
y_train_200_30 = pressurizer_water_y[:241]

x_train_220_22 = pressurizer_water_x[1511:1752, ...]
y_train_220_22 = pressurizer_water_y[1511:1752]

x_train_250_5 = pressurizer_water_x[2659:2900, ...]
y_train_250_5 = pressurizer_water_y[2659:2900]

for time_window_width in [5, 10, 16, 32]:
    train_model(x_train_200_30, y_train_200_30, time_window_width, '200 30')
    train_model(x_train_220_22, y_train_220_22, time_window_width, '220 22')
    train_model(x_train_250_5, y_train_250_5, time_window_width, '250 5')

print('done...')

