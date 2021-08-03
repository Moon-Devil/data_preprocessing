from IO_function import *
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def data_generator() -> object:
    normal_data, accident_data = read_data()
    normal_data = normal_data.values
    accident_data = accident_data.values

    x_train = normal_data
    x_test = x_train[:1000, ]
    length = len(x_train)
    y_train = np.zeros(length)
    y_test = np.zeros(1000)

    for i_index in range(6):
        start_index = i_index * 25
        end_index = (i_index + 1) * 25

        x_train = np.vstack((x_train, accident_data[:, start_index: end_index]))
        x_test = np.vstack((x_test, accident_data[:1000, start_index: end_index]))
        y_train = np.hstack((y_train, np.full(length, (i_index + 1))))
        y_test = np.hstack((y_test, np.full(1000, (i_index + 1))))

    return x_train, y_train, x_test, y_test


def Xgboost_best_parameter(cv, interval) -> object:
    x_train, y_train, _, _ = data_generator()
    learning_rate = np.linspace(0.01, 0.1, interval)
    number_estimator = np.linspace(1, 10, interval)
    other_params = {
        'silent': 1, 'objective': 'multi:softmax', 'random_state': 0
    }

    parameter_distribution = {'learning_rate': learning_rate, 'n_estimatores': number_estimator}
    xgboost = XGBClassifier(**other_params)
    grid_search = GridSearchCV(xgboost, parameter_distribution, cv=cv, verbose=2, return_train_score=True)
    history = grid_search.fit(x_train, y_train)
    recode_history(history)

    learning_rate = history.best_params_['learning_rate']
    number_estimator = int(history.best_params_['n_estimatores'])

    return learning_rate, number_estimator


def Xgboost_function(learning_rate, number_estimator):
    x_train, y_train, x_test, y_test = data_generator()
    xgboost = XGBClassifier(learning_rate=learning_rate, n_estimators=number_estimator, objective='multi:softmax',
                            random_state=0, silent=True)
    xgboost.fit(x_train, y_train)
    y_predict = xgboost.predict(x_test)

    clear_file('Positive')
    positive, negative = 0, 0
    length = len(y_predict)
    for i_index in range(length):
        if i_index % 1000 == 0 and i_index is not 0:
            record_lists = [positive, negative]
            write_to_text('Positive', record_lists, 'a+')
            positive, negative = 0, 0

        if y_predict[i_index] == y_test[i_index]:
            positive = positive + 1
        else:
            negative = negative + 1

    record_lists = [positive, negative]
    write_to_text('Positive', record_lists, 'a+')

    clear_file('result')
    for i_index in range(7):
        start_index = i_index * 1000
        end_index = i_index * 1000 + 301

        x_temp = x_test[start_index: end_index, ]
        y_temp = y_test[start_index: end_index]

        y_predict = xgboost.predict(x_temp)
        write_to_text('result', y_predict.tolist(), 'a+')
        write_to_text('result', y_temp.tolist(), 'a+')


learning, number = Xgboost_best_parameter(5, 10)
Xgboost_function(learning, number)
print("done...")
