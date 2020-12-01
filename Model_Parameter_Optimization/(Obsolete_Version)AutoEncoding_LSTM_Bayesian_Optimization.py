import tensorflow as tf
from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing
from sklearn.model_selection import train_test_split
# from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn import manifold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


power_decreasing_dict = dict()
power_decreasing_array = power_decreasing

batch_size = 32     # 批处理大小
epochs_size = 5     # 循环代数
k_folds = 5         # k折交叉验证


# AE-RNN降维
AE_RNN_history = dict()
AE_RNN_accuracy = dict()

data_dimension = 14

# 预测稳压器水空间温度
y_train_set = power_decreasing_array[..., -3]
x_train_set = np.delete(power_decreasing_array, -3, axis=1)
temp_accuracy = []

# AE降维
inputs = tf.keras.layers.InputLayer(input_shape=(x_train_set.shape[1], ))
code = tf.keras.layers.Dense(data_dimension, activation='relu')
outputs = tf.keras.layers.Dense(x_train_set.shape[1], activation='softmax')

AE_model = tf.keras.models.Sequential([inputs, code, outputs])
AE_model.summary()

AE_model.compile(optimizer='adam', loss='mse')
AE_model.fit(x_train_set, x_train_set, batch_size=batch_size, epochs=epochs_size, validation_split=0.1)

encoder_model = tf.keras.models.Sequential([inputs, code])

decoder_input = tf.keras.layers.InputLayer((data_dimension,))
decoder_output = AE_model.layers[-1]
decoder_model = tf.keras.models.Sequential([decoder_input, decoder_output])

power_decreasing_dict = encoder_model.predict(x_train_set)

# 划分训练集、验证集和测试集
random_seed = np.random.randint(0, 1000)
x_train_all, x_test_normal, y_train_all, y_test_normal = train_test_split(power_decreasing_dict, y_train_set,
                                                                          test_size=0.3, random_state=random_seed)

random_seed = np.random.randint(0, 1000)
x_train_normal, x_valid_normal, y_train_normal, y_valid_normal = train_test_split(x_train_all, y_train_all,
                                                                                  test_size=0.3,
                                                                                  random_state=random_seed)

# 将输入数据集归一化
stander = StandardScaler()
x_train_stander = stander.fit_transform(x_train_normal)
x_valid_stander = stander.transform(x_valid_normal)
x_test_stander = stander.transform(x_test_normal)

x_train = x_train_stander[:, :, np.newaxis]
y_train = y_train_normal[:, np.newaxis]
x_test = x_test_stander[:, :, np.newaxis]
y_test = y_test_normal[:, np.newaxis]
x_valid = x_valid_stander[:, :, np.newaxis]
y_valid = y_valid_normal[:, np.newaxis]


def build_model(layers_size=1, nodes=20) -> object:
    function_model = tf.keras.models.Sequential()
    function_model.add(tf.keras.layers.LSTM(data_dimension, return_sequences=True, input_shape=[None, 1]))

    for _ in np.arange(layers_size):
        function_model.add(tf.keras.layers.LSTM(nodes, activation='relu', return_sequences=True))

    function_model.add(tf.keras.layers.Dense(1))
    function_model.compile(optimizer='adam', loss='mape', metrics=['mse'])
    return function_model


sklearn_model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]

sklearn_model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs_size, callbacks=callbacks)

param_distribution = {
    'layers_size': [1, 2],
    'nodes': np.arange(1, 3)
}

# 网格搜索
grid_search = RandomizedSearchCV(sklearn_model, param_distribution, n_iter=2, cv=3, scoring='neg_log_loss', n_jobs=1)
grid_search.fit(x_train, y_train)

model = grid_search.best_estimator_.model
history = model.evaluate(x_test, y_test)
print(history)


# AE_RNN_history[i_value] = model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs_size)
# y_predict = model.predict(x_test)
# y_predict = np.squeeze(y_predict)
# y_test = np.squeeze(y_test)
#
# temp_accuracy.append(mean_squared_log_error(y_test, y_predict))
# accuracy_array = np.array(temp_accuracy)
# AE_RNN_accuracy[i_value] = accuracy_array