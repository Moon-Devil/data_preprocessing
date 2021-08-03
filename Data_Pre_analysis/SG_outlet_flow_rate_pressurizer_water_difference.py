from Data_equivalence_function_improved import *

target250 = read_data_from_database("power_decreasing_target250")
target250 = target250[:4000, ]


def A_matrix_function(data_set, index_value, penalty_value, name):
    x_data = np.delete(data_set, index_value, axis=1)

    x_shape = x_data.shape
    new_dimension = np.ones(x_shape[0])
    x_expand = np.column_stack((x_data, new_dimension))
    x_expand_shape = x_expand.shape
    I_matrix = np.eye(x_expand_shape[1])

    temp_matrix = np.dot(np.transpose(x_expand), x_expand) - penalty_value * I_matrix
    temp_matrix_inv = np.linalg.inv(temp_matrix)
    temp_A_matrix = np.dot(x_expand, temp_matrix_inv)
    A_matrix = np.dot(temp_A_matrix, np.transpose(x_expand))

    A_matrix_shape = A_matrix.shape
    all_1_matrix = np.full(A_matrix_shape[0], 1)
    A_1_matrix = np.dot(A_matrix, all_1_matrix)

    A_1_matrix_shape = A_1_matrix.shape

    file_name = path + "\\" + name + "_A_1_matrix.txt"
    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, 'w+') as f:
        for i_index in np.arange(A_1_matrix_shape[0]):
            if i_index != A_1_matrix_shape[0] - 1:
                f.write(str(A_1_matrix[i_index]) + ',')
            else:
                f.write(str(A_1_matrix[i_index]) + '\n')


A_matrix_function(target250, 16, 1, "SG_outlet_flow_rate")
A_matrix_function(target250, 24, 0.8, "pressurizer_water_space_temperature")
print("done...")
