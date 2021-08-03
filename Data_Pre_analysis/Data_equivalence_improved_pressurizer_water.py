from Data_Pre_analysis.Data_equivalence_function_improved_pressurize_water import *
from sklearn.metrics import mean_squared_error


target250 = read_data_from_database("power_decreasing_target250")
target250 = target250[:4000, ]

# penalties = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [86, 87, 88, 89, 91, 92, 93, 94]
# mse = []
# print(penalties)
# for penalty in penalties:
#     y_predict, y_true = build_model(target250, 24, 'target250', penalty, 'w+')
#     mse_temp = mean_squared_error(y_true, y_predict)
#     mse.append(mse_temp)
#     print("==========" * 10 + '\n')
#     print("==========" * 10 + '\n')
#     print(penalty)
#     print("==========" * 10 + '\n')
#     print("==========" * 10 + '\n')
#
# penalty_zip = zip(penalties, mse)
# penalty_sorted = sorted(penalty_zip, key=lambda x: (x[1], x[0]))
#
# file = path + "\\Data_equivalence_improved_pressurizer_water_space_temperature_lambda.txt"
# if os.path.exists(file):
#     os.remove(file)
#
# with open(file, 'w+') as f:
#     for temp in penalty_sorted:
#         f.write(str(temp[0]) + ',' + str(temp[1]) + '\n')
#
# print(penalty_sorted)

penalty = 0.8

y_predict, y_true = build_model(target250, 24, 'target250', penalty, 'w+')

target250_noise = add_noise(target250, 0.01)
y_predict_1, _ = build_model(target250_noise, 24, 'target250_0.01', penalty, 'a')

target250_noise = add_noise(target250, 0.05)
y_predict_5, _ = build_model(target250_noise, 24, 'target250_0.05', penalty, 'a')

target250_noise = add_noise(target250, 0.1)
y_predict_10, _ = build_model(target250_noise, 24, 'target250_0.1', penalty, 'a')

mse_1 = mean_squared_error(y_true, y_predict_1)
mse_5 = mean_squared_error(y_true, y_predict_5)
mse_10 = mean_squared_error(y_true, y_predict_10)

print(str(mse_1) + '\t' + '\t' + str(mse_5) + '\t' + str(mse_10))
print("done...")
