from fault_diagnosis_function import *


def test_Adaboost_function(accident_name):
    x_data, y_data, _, _ = dataSet(accident_name)
    model = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME", n_estimators=10)
    model.fit(x_data, y_data)

    y_predict = model.predict(x_data)

    clear_file("index")
    for i_index in range(5):
        start_index = i_index * 301
        end_index = (i_index + 1) * 301
        positive, negative = 0, 0
        accuracy = np.zeros(301)
        for j_index in range(301):
            index = end_index - j_index - 1
            if y_predict[index] == y_data[index]:
                positive += 1
            else:
                negative += 1

            amount = float(j_index + 1)
            array_index = 301 - j_index - 1
            accuracy[array_index] = positive * 100 / amount
            write_to_text("index", [start_index, end_index, index, array_index], "a+")
        write_to_text("time_accuracy", accuracy.tolist(), "a+")

    write_to_text("result", y_predict[:301].tolist(), "a+")
    write_to_text("result", y_data[:301].tolist(), "a+")


def test():
    clear_file("time_accuracy")
    clear_file("result")
    test_Adaboost_function("Ada@PowerR")
    test_Adaboost_function("Ada@PRLL")
    test_Adaboost_function("Ada@PRSL")
    test_Adaboost_function("Ada@CL_LOCA")
    test_Adaboost_function("Ada@HL_LOCA")
    test_Adaboost_function("Ada@SG2L")
    test_Adaboost_function("Ada@SGTR")


test()
