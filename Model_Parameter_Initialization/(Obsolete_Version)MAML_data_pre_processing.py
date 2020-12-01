from Data_Pre_analysis.Power_Decreasing_Finally_Data import power_decreasing, power_decreasing_nodes
import numpy as np
from sklearn.model_selection import train_test_split
from Model_Parameter_Initialization.model_agnostic_meta_learning_function import batch_generator
import gc
from sklearn import manifold


epochs_size = 2
batch_size = 5
data_dimension = 16

nodes_length = len(power_decreasing_nodes)
x_train_power_decreasing_list = []
y_train_power_decreasing_list = []
index_start = 0

y_train_power_decreasing = power_decreasing[..., 3]
x_train_power_decreasing = np.delete(power_decreasing, -3, axis=1)

# LLE降维
lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=data_dimension, method='standard')
x_train_power_decreasing_LLE = lle.fit_transform(x_train_power_decreasing)

for i_value in np.arange(nodes_length):
    index_end = index_start + power_decreasing_nodes[i_value]
    x_train_power_decreasing_list.append(x_train_power_decreasing_LLE[index_start: index_end, :])
    y_train_power_decreasing_list.append(y_train_power_decreasing[index_start: index_end])
    index_start = index_end

del index_start, index_end, x_train_power_decreasing_LLE, y_train_power_decreasing, power_decreasing

# 划分数据集
x_train_set = []
y_train_set = []

for i_value in np.arange(nodes_length):
    temp_list = (y_train_power_decreasing_list[i_value])[:306, ...]
    y_train_set.append(temp_list)

for i_value in np.arange(nodes_length):
    temp_list = (x_train_power_decreasing_list[i_value])[:306, ...]
    x_train_set.append(temp_list)

del temp_list, x_train_power_decreasing_list, y_train_power_decreasing_list

x_train_all = []
y_train_all = []
x_test_normal = []
y_test = []

for i_value in np.arange(nodes_length):
    temp_x_train, temp_x_test, temp_y_train, temp_y_test = train_test_split(x_train_set[i_value], y_train_set[i_value],
                                                                            test_size=52, random_state=0)
    x_train_all.append(temp_x_train)
    x_test_normal.append(temp_x_test)
    y_train_all.append(temp_y_train)
    y_test.append(temp_y_test)

x_train_normal = []
x_valid_normal = []
y_train = []
y_valid = []

for i_value in np.arange(nodes_length):
    temp_x_train, temp_x_valid, temp_y_train, temp_y_valid = train_test_split(x_train_all[i_value],
                                                                              y_train_all[i_value],
                                                                              test_size=52, random_state=0)
    x_train_normal.append(temp_x_train)
    x_valid_normal.append(temp_x_valid)
    y_train.append(temp_y_train)
    y_valid.append(temp_y_valid)

del temp_x_train, temp_y_train, temp_x_valid, temp_y_valid, x_train_all, y_train_all, temp_x_test
del temp_y_test, x_train_set, y_train_set, power_decreasing_nodes
gc.collect()

# 形成批数据集
temp_epochs_size = epochs_size
x_train_batch = []
y_train_batch = []
x_valid_batch = []
y_valid_batch = []
x_test_batch = []
y_test_batch = []

while temp_epochs_size > 0:
    for i_value in np.arange(nodes_length):
        temp_x_train_batch, temp_y_train_batch = batch_generator(x_train_normal[i_value], y_train[i_value], batch_size)
        x_train_batch.append(temp_x_train_batch)
        y_train_batch.append(temp_y_train_batch)

        temp_x_valid_batch, temp_y_valid_batch = batch_generator(x_valid_normal[i_value], y_valid[i_value], batch_size)
        x_valid_batch.append(temp_x_valid_batch)
        y_valid_batch.append(temp_y_valid_batch)

        temp_x_test_batch, temp_y_test_batch = batch_generator(x_test_normal[i_value], y_test[i_value], batch_size)
        x_test_batch.append(temp_x_test_batch)
        y_test_batch.append(temp_y_test_batch)

    temp_epochs_size -= 1

del x_train_normal, x_test_normal, x_valid_normal, y_train, y_test, y_valid, temp_x_train_batch, temp_y_train_batch
del temp_x_valid_batch, temp_y_valid_batch, temp_x_test_batch, temp_y_test_batch, temp_epochs_size
gc.collect()

x_train_list = np.array(x_train_batch[0][0])
x_valid_list = np.array(x_valid_batch[0][0])
x_test_list = np.array(x_test_batch[0][0])

y_train_list = np.array(y_train_batch[0][0])
y_valid_list = np.array(y_valid_batch[0][0])
y_test_list = np.array(y_test_batch[0][0])

for i_value in np.arange(nodes_length):
    for j_value in np.arange(batch_size):
        if i_value == 0 and j_value == 0:
            pass
        else:
            x_train_list = np.vstack((x_train_list, np.array(x_train_batch[i_value][j_value])))
            x_valid_list = np.vstack((x_valid_list, np.array(x_valid_batch[i_value][j_value])))
            x_test_list = np.vstack((x_test_list, np.array(x_test_batch[i_value][j_value])))

            y_train_list = np.vstack((y_train_list, np.array(y_train_batch[i_value][j_value])))
            y_valid_list = np.vstack((y_valid_list, np.array(y_valid_batch[i_value][j_value])))
            y_test_list = np.vstack((y_test_list, np.array(y_test_batch[i_value][j_value])))

(x_train_column, x_train_row) = x_train_list.shape
x_train = np.zeros((x_train_column, x_train_row))
for i_value in np.arange(x_train_column):
    for j_value in np.arange(x_train_row):
        x_train[i_value][j_value] = x_train_list[i_value][j_value]

(x_valid_column, x_valid_row) = x_valid_list.shape
x_valid = np.zeros((x_valid_column, x_valid_row))
for i_value in np.arange(x_valid_column):
    for j_value in np.arange(x_valid_row):
        x_valid[i_value][j_value] = x_valid_list[i_value][j_value]

(x_test_column, x_test_row) = x_test_list.shape
x_test = np.zeros((x_test_column, x_test_row))
for i_value in np.arange(x_test_column):
    for j_value in np.arange(x_test_row):
        x_test[i_value][j_value] = x_test_list[i_value][j_value]

y_train_column = y_train_list.shape[0]
y_train = np.zeros((y_train_column, 1))
for i_value in np.arange(y_train_column):
    y_train[i_value][0] = y_train_list[i_value][0]

y_valid_column = y_valid_list.shape[0]
y_valid = np.zeros((y_valid_column, 1))
for i_value in np.arange(y_valid_column):
    y_valid[i_value][0] = y_valid_list[i_value][0]

y_test_column = y_test_list.shape[0]
y_test = np.zeros((y_test_column, 1))
for i_value in np.arange(y_test_column):
    y_test[i_value][0] = y_test_list[i_value][0]

del x_train_batch, x_valid_batch, x_test_batch, y_train_batch, y_valid_batch, y_test_batch
del x_train_list, x_valid_list, x_test_list, y_train_list, y_valid_list, y_test_list
del x_train_column, x_valid_column, x_test_column, y_train_column, y_valid_column, y_test_column
del x_train_row, x_valid_row, x_test_row
gc.collect()
