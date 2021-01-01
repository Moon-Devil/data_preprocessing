from Data_Pre_analysis.Data_equivalence_function import *
from sklearn.metrics import mean_squared_error


target250 = read_data_from_database("power_decreasing_target250")
target250 = target250[:4000, ]

y_predict, y_true = build_model(target250, 16, 'target250', 'w+')

target250_noise = add_noise(target250, 0.01)
y_predict_1, _ = build_model(target250_noise, 16, 'target250_0.01', 'a')

target250_noise = add_noise(target250, 0.02)
y_predict_2, _ = build_model(target250_noise, 16, 'target250_0.02', 'a')

target250_noise = add_noise(target250, 0.05)
y_predict_5, _ = build_model(target250_noise, 16, 'target250_0.05', 'a')

target250_noise = add_noise(target250, 0.1)
y_predict_10, _ = build_model(target250_noise, 16, 'target250_0.1', 'a')

mse_1 = mean_squared_error(y_true, y_predict_1)
mse_2 = mean_squared_error(y_true, y_predict_2)
mse_5 = mean_squared_error(y_true, y_predict_5)
mse_10 = mean_squared_error(y_true, y_predict_10)

print(str(mse_1) + '\t' + str(mse_2) + '\t' + str(mse_5) + '\t' + str(mse_10))
print("done...")
