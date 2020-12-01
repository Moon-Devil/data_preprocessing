from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing, power_decreasing_nodes
import numpy as np
from sklearn.model_selection import train_test_split
import gc
from sklearn import manifold

data_dimension = 16
batch_sizes = 32
data_set_length = 320 + 1
nodes = 69
learning_rate = 0.001
inner_learning_rate = 0.001
epoch_sizes = 20

nodes_length = len(power_decreasing_nodes)
x_train_power_decreasing_list = []
y_train_power_decreasing_list = []
index_start = 0

y_train_power_decreasing = power_decreasing[..., -3]
x_train_power_decreasing = np.delete(power_decreasing, -3, axis=1)


# LLE降维
lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=data_dimension, method='standard')
x_train_power_decreasing_LLE = lle.fit_transform(x_train_power_decreasing)

for i_value in np.arange(nodes_length):
    index_end = index_start + power_decreasing_nodes[i_value]
    x_train_power_decreasing_list.append(x_train_power_decreasing_LLE[index_start: index_end, :])
    y_train_power_decreasing_list.append(y_train_power_decreasing[index_start: index_end])
    index_start = index_end

del index_start, index_end, x_train_power_decreasing, y_train_power_decreasing, \
    power_decreasing, lle, power_decreasing_nodes

# 划分数据集
x_train_set = []
y_train_set = []

for i_value in np.arange(nodes_length):
    temp_list = np.array((y_train_power_decreasing_list[i_value])[:data_set_length, ...])
    y_train_set.append(temp_list)

for i_value in np.arange(nodes_length):
    temp_list = np.array((x_train_power_decreasing_list[i_value])[:data_set_length, ...])
    x_train_set.append(temp_list)

x_train_set = np.array(x_train_set)
y_train_set = np.array(y_train_set)

del temp_list, x_train_power_decreasing_list, y_train_power_decreasing_list

x_train = []
y_train = []

for i_value in np.arange(nodes_length):
    temp_x_train, _, temp_y_train, _ = train_test_split(x_train_set[i_value], y_train_set[i_value], test_size=1,
                                                        random_state=0)
    x_train.append(temp_x_train)
    y_train.append(temp_y_train)

del temp_x_train, temp_y_train, x_train_set, y_train_set

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train.reshape((nodes_length, int(data_set_length/batch_sizes), batch_sizes, data_dimension))
y_train = y_train.reshape((nodes_length, int(data_set_length/batch_sizes), batch_sizes))

gc.collect()
print("数据预处理已完成")
