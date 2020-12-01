import os


if __name__ == "__main__":
    os.system("python LLE_DNN_Grid_Search.py")
    os.system("python LLE_DNN_Random_Search.py")
    os.system("python LLE_DNN_Bayesian_Optimization")
    print("==============================================================================================")
    print("模型参数优化已完成！！！")
