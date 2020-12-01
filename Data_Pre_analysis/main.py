import os

# 各文件名称即为程序的功能
# 每个文件可在main程序中集中运行也可单独运行
# 程序整体循环时部分NN和RNN精度有时会出现超大值，估计是内存溢出引起的，由于程序编写时间短促，这个bug就不解决了。出现超大值可单独计算该精度。
if __name__ == "__main__":
    os.system("python Database_Statistics.py")
    os.system("python power_decreasing_covariance_matrix.py")
    os.system("python power_decreasing_covariance_matrix_heatmap.py")
    os.system("python PCA_NN_Dimension_Reduction.py")
    os.system("python PCA_SVR_Dimension_Reduction.py")
    os.system("python PCA_GPR_Dimension_Reduction.py")
    os.system("python PCA_RNN_Dimension_Reduction.py")
    os.system("python LLE_NN_Dimension_Reduction.py")
    os.system("python LLE_SVR_Dimension_Reduction.py")
    os.system("python LLE_GPR_Dimension_Reduction.py")
    os.system("python LLE_RNN_Dimension_Reduction.py")
    os.system("python Isomap_NN_Dimension_Reduction.py")
    os.system("python Isomap_SVR_Dimension_Reduction.py")
    os.system("python Isomap_GPR_Dimension_Reduction.py")
    os.system("python Isomap_RNN_Dimension_Reduction.py")
    os.system("python AutoEncoding_NN_Dimension_Reduction.py")
    os.system("python AutoEncoding_SVR_Dimension_Reduction.py")
    os.system("python AutoEncoding_GPR_Dimension_Reduction.py")
    os.system("python AutoEncoding_RNN_Dimension_Reduction.py")
    print("==============================================================================================")
    print("数据预处理已完成！！！")
