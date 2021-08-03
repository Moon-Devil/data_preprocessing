from data_reduction_function import *
from regression_function import *


def PCA_DNN_function(index, batchSize, epochs):
    filename = "DNN_PCA_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = PCA_function(index, i_index)
        mse = DNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("DNN_PCA_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def Isomap_DNN_function(index, batchSize, epochs):
    filename = "DNN_Isomap_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = Isomap_function(index, i_index)
        mse = DNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("DNN_Isomap_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def LLE_DNN_function(index, batchSize, epochs):
    filename = "DNN_LLE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = LLE_function(index, i_index)
        mse = DNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("DNN_LLE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def AE_DNN_function(index, batchSize, epochs):
    filename = "DNN_AE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = AE_function(index, i_index, batchSize, epochs)
        mse = DNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("DNN_AE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def VAE_DNN_function(index, batchSize, epochs):
    filename = "DNN_VAE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = VAE_function(index, i_index, epochs)
        mse = DNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("DNN_VAE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def RBM_DNN_function(index, batchSize, epochs):
    filename = "DNN_RBM_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = RBM_function(index, i_index, epochs)
        mse = DNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("DNN_RBM_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def PCA_SVR_function(index):
    filename = "SVR_PCA_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = PCA_function(index, i_index)
        mse = SVR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("SVR_PCA_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def Isomap_SVR_function(index):
    filename = "SVR_Isomap_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = Isomap_function(index, i_index)
        mse = SVR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("SVR_Isomap_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def LLE_SVR_function(index):
    filename = "SVR_LLE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = LLE_function(index, i_index)
        mse = SVR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("SVR_LLE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def AE_SVR_function(index, batchSize, epochs):
    filename = "SVR_AE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = AE_function(index, i_index, batchSize, epochs)
        mse = SVR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("SVR_AE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def VAE_SVR_function(index, epochs):
    filename = "SVR_VAE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = VAE_function(index, i_index, epochs)
        mse = SVR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("SVR_VAE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def RBM_SVR_function(index, epochs):
    filename = "SVR_RBM_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = RBM_function(index, i_index, epochs)
        mse = SVR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("SVR_RBM_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def PCA_GPR_function(index):
    filename = "GPR_PCA_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = PCA_function(index, i_index)
        mse = GPR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("GPR_PCA_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def Isomap_GPR_function(index):
    filename = "GPR_Isomap_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = Isomap_function(index, i_index)
        mse = GPR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("GPR_Isomap_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def LLE_GPR_function(index):
    filename = "GPR_LLE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = LLE_function(index, i_index)
        mse = GPR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("GPR_LLE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def AE_GPR_function(index, batchSize, epochs):
    filename = "GPR_AE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = AE_function(index, i_index, batchSize, epochs)
        mse = GPR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("GPR_AE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def VAE_GPR_function(index, epochs):
    filename = "GPR_VAE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = VAE_function(index, i_index, epochs)
        mse = GPR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("GPR_VAE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def RBM_GPR_function(index, epochs):
    filename = "GPR_RBM_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = RBM_function(index, i_index, epochs)
        mse = GPR_function(x_data, y_data)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("GPR_RBM_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def PCA_RNN_function(index, batchSize, epochs):
    filename = "RNN_PCA_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = PCA_function(index, i_index)
        mse = RNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("RNN_PCA_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def Isomap_RNN_function(index, batchSize, epochs):
    filename = "RNN_Isomap_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = Isomap_function(index, i_index)
        mse = RNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("RNN_Isomap_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def LLE_RNN_function(index, batchSize, epochs):
    filename = "RNN_LLE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = LLE_function(index, i_index)
        mse = RNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("RNN_LLE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def AE_RNN_function(index, batchSize, epochs):
    filename = "RNN_AE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = AE_function(index, i_index, batchSize, epochs)
        mse = RNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("RNN_AE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def VAE_RNN_function(index, batchSize, epochs):
    filename = "RNN_VAE_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = VAE_function(index, i_index, epochs)
        mse = RNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("RNN_VAE_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))


def RBM_RNN_function(index, batchSize, epochs):
    filename = "RNN_RBM_PR_mse"
    clear_file(filename)
    for i_index in range(1, 25):
        x_data, y_data = RBM_function(index, i_index, epochs)
        mse = RNN_function(x_data, y_data, batchSize, epochs)
        data_lists = [mse]
        write_to_text(filename, data_lists, "a+")
        print("RNN_RBM_PR \t dimension = " + str(i_index) + "\t" + "mse = " + str(mse))

