from algorithm_pool import *
from IO_function import *


def MAML_NN_function(x_dataframe, y_dataframe, layers, nodes, epochs):
    x_train = x_dataframe.values
    y_train = y_dataframe.values
    dimensions = np.shape(x_train)[1]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(dimensions, name="input_layer"))
    for index in np.arange(layers):
        model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, name="hidden_layer_" +
                                                                                          str(index + 1)))
    model.add(tf.keras.layers.Dense(1, name="output_layer"))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])
    model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2)

    model.save_weights('MAML_' + str(epochs) + '.h5')


def MAML_function(x_dataframe, y_dataframe, layers, nodes, epochs_MAML, epochs, stop_flag, stop_patience):
    MAML_NN_function(x_dataframe, y_dataframe, layers, nodes, epochs_MAML)

    x_train = x_dataframe.values
    y_train = y_dataframe.values
    dimensions = np.shape(x_train)[1]

    x_test = x_train[: data_set_length, ]
    y_test = y_train[: data_set_length, ]

    start_train = time.time()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(dimensions, name="input_layer"))
    for index in np.arange(layers):
        model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, name="hidden_layer_" +
                                                                                          str(index + 1)))
    model.add(tf.keras.layers.Dense(1, name="output_layer"))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])
    model.load_weights('MAML_' + str(epochs_MAML) + '.h5')

    if stop_flag == 1:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience,
                                                                        verbose=2)])
    else:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2)
    end_train = time.time()

    start_predict = time.time()
    y_predict_temp = model.predict(x_test)
    end_predict = time.time()
    y_true = []
    y_predict = []

    length = len(y_test)
    for index in np.arange(length):
        y_true.append(y_test[index, 0])
        y_predict.append(y_predict_temp[index, 0])

    train_time = end_train - start_train
    predict_time = end_predict - start_predict

    return history, y_true, y_predict, train_time, predict_time


def MAML_origin_function(x_dataframe, y_dataframe, layers, nodes, epochs, stop_flag, stop_patience):
    x_train = x_dataframe.values
    y_train = y_dataframe.values
    dimensions = np.shape(x_train)[1]

    x_test = x_train[: data_set_length, ]
    y_test = y_train[: data_set_length, ]

    start_train = time.time()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(dimensions, name="input_layer"))
    for index in np.arange(layers):
        model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.activations.relu, name="hidden_layer_" +
                                                                                          str(index + 1)))
    model.add(tf.keras.layers.Dense(1, name="output_layer"))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])

    if stop_flag == 1:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_patience,
                                                                        verbose=2)])
    else:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2)
    end_train = time.time()

    start_predict = time.time()
    y_predict_temp = model.predict(x_test)
    end_predict = time.time()
    y_true = []
    y_predict = []

    length = len(y_test)
    for index in np.arange(length):
        y_true.append(y_test[index, 0])
        y_predict.append(y_predict_temp[index, 0])

    train_time = end_train - start_train
    predict_time = end_predict - start_predict

    return history, y_true, y_predict, train_time, predict_time
