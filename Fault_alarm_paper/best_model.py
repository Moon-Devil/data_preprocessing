from algorithms_pool import *


# x_train, y_train, _ = read_data_from_database(16, 0, [200, 220, 250])
# best_dnn(x_train, y_train)
# best_rnn(x_train, y_train)
# best_lstm(x_train, y_train)
# best_bert(x_train, y_train)

x_train, y_train, _ = read_data_from_database(16, 0, [200, 220, 250])
best_dnn(x_train, y_train)

print("done...")
