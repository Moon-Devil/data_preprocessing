from Data_Pre_analysis.Data_equivalence_function_improved_tensorboard import *
import tensorflow as tf
import os
import shutil


target250 = read_data_from_database("power_decreasing_target250")
target250 = target250[:4000, ]


if os.path.exists('logs'):
    shutil.rmtree('logs')

log_dir = os.path.join('logs')
os.mkdir(log_dir)
tensorboard_show = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

penalty = 1
dataset = target250
index = 16
dataset_name = 'target250'

y_data = dataset[:, index]
x_data = np.delete(dataset, index, axis=1)

x_data_shape = np.shape(x_data)
dimension = x_data_shape[1]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(dimension, activation=tf.keras.activations.relu, input_shape=(dimension,),
                                kernel_regularizer=tf.keras.regularizers.l2(penalty)))
model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu,
                                kernel_regularizer=tf.keras.regularizers.l2(penalty)))
model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu,
                                kernel_regularizer=tf.keras.regularizers.l2(penalty)))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse, metrics=[tf.keras.metrics.mse])
model.summary()

history = model.fit(x_data, y_data, epochs=400, batch_size=16, validation_split=0.2, callbacks=[tensorboard_show])
print("done...")
