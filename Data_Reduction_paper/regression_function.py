import tensorflow as tf
from sklearn import svm
from sklearn.metrics import mean_squared_error
from IO_function import *
from sklearn import gaussian_process


def DNN_function(x_data, y_data, batchSize, epochs) -> float:
    input_dimension = np.shape(x_data)[1]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu, input_shape=(input_dimension,)))
    model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mape, metrics=[tf.keras.metrics.mse])
    model.fit(x_data, y_data, batch_size=batchSize, epochs=epochs, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20, verbose=2)])

    y_predict = model.predict(x_data)
    y_predict = np.squeeze(y_predict)
    mse = mean_squared_error(y_data, y_predict)

    return mse


def SVR_function(x_data, y_data) -> object:
    model = svm.SVR(kernel='rbf')
    model.fit(x_data, y_data)

    y_predict = model.predict(x_data)
    mse = mean_squared_error(y_data, y_predict)

    return mse


def GPR_function(x_data, y_data) -> object:
    model = gaussian_process.GaussianProcessRegressor()
    model.fit(x_data, y_data)

    y_predict = model.predict(x_data)
    mse = mean_squared_error(y_data, y_predict)

    return mse


def RNN_function(x_data, y_data, batchSize, epochs) -> float:
    input_dimension = np.shape(x_data)[1]
    x_train = x_data[:, :, np.newaxis]
    y_train = y_data[:, np.newaxis]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(input_dimension, return_sequences=True, input_shape=[None, 1]))
    model.add(tf.keras.layers.SimpleRNN(100, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(100))
    model.add(tf.keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.MAPE,
                  metrics=[tf.keras.losses.MSE])
    model.fit(x_train, y_train, batch_size=batchSize, verbose=0, epochs=epochs, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])

    y_predict = model.predict(x_train)
    y_predict = np.squeeze(y_predict)

    mse = mean_squared_error(y_data, y_predict)

    return mse
