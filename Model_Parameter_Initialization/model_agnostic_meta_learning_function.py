from Model_Parameter_Initialization.MAML_data_pre_processing import *
import tensorflow as tf
import numpy as np
import os


# 设置数据表存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing Initialization'
file_losses = path + "\\model_agnostic_meta_learning_losses.txt"
file_predict = path + "\\model_agnostic_meta_learning_predict.txt"

if not os.path.exists(path):
    os.mkdir(path)


class Model_Agnostic_Meta_Learning(tf.keras.models.Model):
    def __init__(self, input_nodes, hidden_layer_nodes):
        super(Model_Agnostic_Meta_Learning, self).__init__()
        self.hidden_nodes = hidden_layer_nodes
        self.input_nodes = input_nodes
        self.hidden_layer_1 = tf.keras.layers.Dense(self.input_nodes, activation=tf.keras.activations.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(self.hidden_nodes, activation=tf.keras.activations.relu)
        self.output_layer = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        super(Model_Agnostic_Meta_Learning, self).build(input_shape)

    def call(self, x):
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x


model = Model_Agnostic_Meta_Learning(data_dimension, nodes)
model.build(input_shape=(None, data_dimension))
model.summary()

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
for epoch in np.arange(epoch_sizes):
    if epoch == 0 and os.path.exists(file_losses):
        os.remove(file_losses)
        with open(file_losses, "w+") as f:
            f.write("==========================================\tstart" +
                    "\t==========================================\n")

    if epoch == 0 and os.path.exists(file_predict):
        os.remove(file_predict)
        with open(file_predict, "w+") as f:
            f.write("==========================================\tstart" +
                    "\t==========================================\n")

    for task in np.arange(nodes_length):  # nodes_length
        for i_value in np.arange(x_train.shape[1]):  # x_train.shape[1]
            temp_x_train = x_train[task][i_value]
            temp_y_train = y_train[task][i_value]
            temp_x_train = tf.convert_to_tensor(temp_x_train)
            temp_y_train = tf.convert_to_tensor(temp_y_train)

            temp_x_train = tf.cast(temp_x_train, dtype=tf.float32)
            temp_y_train = tf.cast(temp_y_train, dtype=tf.float32)

            with tf.GradientTape() as test_tape:
                with tf.GradientTape() as train_tape:
                    temp_y_predict = model(temp_x_train)
                    train_loss = tf.keras.losses.mean_squared_error(y_true=temp_y_train, y_pred=temp_y_predict)
                gradients_train = train_tape.gradient(train_loss, model.trainable_variables)

                k = 0

                copied_model = Model_Agnostic_Meta_Learning(data_dimension, nodes)
                copied_model(temp_x_train)
                copied_model.set_weights(model.get_weights())

                inner_learning_rate = tf.convert_to_tensor(inner_learning_rate)
                inner_learning_rate = tf.cast(inner_learning_rate, dtype=tf.float32)

                for j_value in np.arange(len(model.layers)):
                    copied_model.layers[j_value].kernel = tf.subtract(model.layers[j_value].kernel,
                                                                      tf.multiply(inner_learning_rate,
                                                                                  gradients_train[k]))
                    copied_model.layers[j_value].bias = tf.subtract(model.layers[j_value].bias,
                                                                    tf.multiply(inner_learning_rate,
                                                                                gradients_train[k + 1]))
                    k += 2

                y_predict_test = copied_model(temp_x_train)
                test_loss = tf.keras.losses.mean_squared_error(y_true=temp_y_train, y_pred=y_predict_test)

            gradients_test = test_tape.gradient(test_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients_test, model.trainable_variables))

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                temp_loss = sess.run(test_loss)
            temp_loss = np.mean(temp_loss)

            print("epoch=" + str(epoch + 1) + "\ttask=" + str(task + 1) + "\tvalue=" + str(i_value + 1) +
                  "\ttemp_loss=" + str(temp_loss) + "\n")

            with open(file_losses, "a") as f:
                f.write(str(temp_loss) + "\n")

            MAML_model = Model_Agnostic_Meta_Learning(data_dimension, nodes)
            MAML_model.build(input_shape=(None, data_dimension))
            MAML_model.summary()

            MAML_model.set_weights(model.get_weights())
            y_predict = MAML_model(temp_x_train)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                y_predict = sess.run(y_predict)

            with open(file_predict, "a") as f:
                f.write("===============\tepoch\t" + str(epoch) + "\t" + str(task) + "\t" + str(i_value) + "\n")
                length = len(y_predict)
                f.write("y_predict\n")
                for k in np.arange(length):
                    if k != len(y_predict) - 1:
                        f.write(str(y_predict[i_value][k]) + ",")
                    else:
                        f.write(str(y_predict[i_value][k]) + "\n")

                length = y_train[task][i_value].shape[0]
                f.write("y_predict\n")
                for k in np.arange(length):
                    if k != length - 1:
                        f.write(str(y_train[task][i_value][k]) + ",")
                    else:
                        f.write(str(y_train[task][i_value][k]) + "\n")


print("done")
