import tensorflow as tf
from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import manifold
from sklearn.model_selection import GridSearchCV
import os
from Model_Parameter_Optimization.model_optimization_setup_function import *


power_decreasing_array = power_decreasing

# LLE-DNN降维
LLE_DNN_history = dict()
LLE_DNN_accuracy = dict()

# 预测稳压器水空间温度
y_train_set = power_decreasing_array[..., -3]
x_train_set = np.delete(power_decreasing_array, -3, axis=1)
temp_accuracy = []

# LLE降维
lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=data_dimension, method='standard')
power_decreasing_LLE = lle.fit_transform(x_train_set)

# 划分训练集、验证集和测试集
x_train, x_test, y_train, y_test = train_test_split(power_decreasing_LLE, y_train_set, test_size=0.3, random_state=0)


def build_model(layers_size=default_layers_size, nodes=default_nodes, learning_rate=default_learning_rate) -> object:
    function_model = tf.keras.models.Sequential()
    function_model.add(tf.keras.layers.Dense(data_dimension, activation='relu', input_shape=x_train.shape[1:],))

    for _ in np.arange(layers_size):
        function_model.add(tf.keras.layers.Dense(nodes, activation='relu'))

    function_model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    function_model.compile(optimizer=optimizer, loss=tf.keras.losses.MAPE, metrics=[tf.keras.losses.MSE])
    return function_model


sklearn_model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)]

sklearn_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs_size, callbacks=callbacks)

param_distribution = {
    'layers_size': [layers_size_low_value, layers_size_high_value],
    'nodes': np.arange(nodes_low_value, nodes_high_value),
    'learning_rate': np.arange(learning_rate_low_value, learning_rate_high_value, learning_rate_margin),
}

# 网格搜索
grid_search = GridSearchCV(sklearn_model, param_distribution, cv=2, n_jobs=1)
history_grid_search = grid_search.fit(x_train, y_train)

grid_search_model = grid_search.best_estimator_.model
grid_search_test_loss = grid_search_model.evaluate(x_test, y_test)

grid_search_loss = grid_search_model.history.history['loss']
grid_search_mean_squared_error = grid_search_model.history.history['mean_squared_error']
grid_search_grid_scores = grid_search.cv_results_

print("===================================================================================================")
print("grid search best parameter = " + str(grid_search.best_params_))
print("grid search best score = " + str(grid_search.best_score_))
print("grid search loss = " + str(grid_search_loss))
print("grid search mean squared error = " + str(grid_search_mean_squared_error))
print("grid search grid scores = " + str(grid_search_grid_scores['params']))
print("grid search test loss = " + str(grid_search_test_loss))

# 设置数据表存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing build model'

if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\LLE_DNN_Grid_Search.txt"
if os.path.exists(file):
    os.remove(file)

loss_length = len(grid_search_loss)
mean_squared_error_length = len(grid_search_mean_squared_error)
grid_scores_length = len(grid_search_grid_scores['params'])

with open(file, "w+") as f:
    f.write("grid_search_best_parameter\t" + str(grid_search.best_params_) + "\n")
    f.write("grid_search_best_score\t" + str(grid_search.best_score_) + "\n")

    f.write("grid_search_loss\t")
    for i_value in np.arange(loss_length):
        if i_value != (loss_length - 1):
            f.write(str(grid_search_loss[i_value]) + ",")
        else:
            f.write(str(grid_search_loss[i_value]) + "\n")

    f.write("grid_search_mean_squared_error\t")
    for i_value in np.arange(mean_squared_error_length):
        if i_value != (mean_squared_error_length - 1):
            f.write(str(grid_search_mean_squared_error[i_value]) + ",")
        else:
            f.write(str(grid_search_mean_squared_error[i_value]) + "\n")

    f.write("grid_search_grid_scores\n")
    for i_value in np.arange(grid_scores_length):
        f.write(str(grid_search_grid_scores['params'][i_value]['layers_size']) + "," +
                str(grid_search_grid_scores['params'][i_value]['nodes']) + "," +
                str(grid_search_grid_scores['params'][i_value]['learning_rate']) + "\n")

    f.write("grid_search_test_loss\t" + str(grid_search_test_loss) + "\n")
