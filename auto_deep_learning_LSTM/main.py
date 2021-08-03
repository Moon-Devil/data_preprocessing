from QGA_algorithm import *

x_dataframe, y_dataframe, y_data_origin = read_data_from_database(16, 1, [200, 220, 250])
qga = QGA(x_dataframe, y_dataframe, 5, 2, 10, 100, 1, 200, 0.01*np.pi)
qga.train_step()

print("done...")
