import tensorflow as tf
from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import manifold
from skopt import gp_minimize
from skopt.space import Real, Integer
import skopt.utils
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
x_train, x_test, y_train, y_test = train_test_split(power_decreasing_LLE, y_train_set, test_size=0.3,
                                                    random_state=0)

# 超参数声明
dim_layers_size = Integer(low=layers_size_low_value, high=layers_size_high_value, name='layers_size')
dim_nodes = Integer(low=nodes_low_value, high=nodes_high_value, name='nodes')
dim_learning_rate = Real(low=learning_rate_low_value, high=learning_rate_high_value, prior='log-uniform',
                         name='learning_rate')

dimensions = [
    dim_layers_size,
    dim_nodes,
    dim_learning_rate
]

# 超参数测试数值
default_parameters = [default_layers_size, default_nodes, default_learning_rate]


@skopt.utils.use_named_args(dimensions=dimensions)
def fitness(layers_size, nodes, learning_rate) -> object:
    function_model = tf.keras.models.Sequential()
    function_model.add(tf.keras.layers.Dense(data_dimension, activation=tf.keras.activations.relu,
                                             input_shape=(data_dimension,)))

    for _ in np.arange(layers_size):
        function_model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, ))

    function_model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    function_model.compile(optimizer=optimizer, loss=tf.keras.losses.MAPE, metrics=[tf.keras.losses.MSE])
    history = function_model.fit(x=x_train, y=y_train, epochs=epochs_size, batch_size=batch_size,
                                 validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                                                   patience=20)])
    function_accuracy = history.history['val_mean_squared_error'][-1]
    return function_accuracy


fitness(x=default_parameters)
history_Bayesian_optimization = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=200,
                                            x0=default_parameters)

Bayesian_optimization_best_loss = history_Bayesian_optimization.fun
Bayesian_optimization_iteration_process = history_Bayesian_optimization.x_iters
Bayesian_optimization_best_value = history_Bayesian_optimization.x
Bayesian_optimization_loss = history_Bayesian_optimization.func_vals

print("===================================================================================================")
print("Bayesian optimization best loss = " + str(Bayesian_optimization_best_loss))
print("Bayesian optimization iteration process = " + str(Bayesian_optimization_iteration_process))
print("Bayesian optimization best value = " + str(Bayesian_optimization_best_value))
print("Bayesian optimization loss = " + str(Bayesian_optimization_loss))

# 设置数据表存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing build model'

if not os.path.exists(path):
    os.mkdir(path)

file = path + "\\LLE_DNN_Bayesian_Optimization.txt"
if os.path.exists(file):
    os.remove(file)

iteration_length_i_value = len(Bayesian_optimization_iteration_process)
iteration_length_j_value = len(Bayesian_optimization_iteration_process[0])

loss_length = len(Bayesian_optimization_loss)
with open(file, "w+") as f:
    f.write("Bayesian_optimization_best_loss\t" + str(Bayesian_optimization_best_loss) + "\n")

    f.write("Bayesian_optimization_iteration_process\n")
    for i_value in np.arange(iteration_length_i_value):
        for j_value in np.arange(iteration_length_j_value):
            if j_value != (iteration_length_j_value - 1):
                f.write(str(Bayesian_optimization_iteration_process[i_value][j_value]) + ",")
            else:
                f.write(str(Bayesian_optimization_iteration_process[i_value][j_value]) + "\n")

    f.write("Bayesian_optimization_best_value\t" + str(Bayesian_optimization_best_value) + "\n")

    f.write("Bayesian_optimization_loss\t")
    for i_value in np.arange(loss_length):
        if i_value != (loss_length - 1):
            f.write(str(Bayesian_optimization_loss[i_value]) + ",")
        else:
            f.write(str(Bayesian_optimization_loss[i_value]) + "\n")
