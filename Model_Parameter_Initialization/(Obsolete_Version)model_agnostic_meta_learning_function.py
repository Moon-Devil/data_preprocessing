import tensorflow as tf
import numpy as np
import os


# 设置数据表存储路径
grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
path = grandfather_path + 'Calculations\\power decreasing Initialization'

if not os.path.exists(path):
    os.mkdir(path)


def batch_generator(x_array, y_array, batch_size, shuffle=True):
    x_array = [np.array(i_value) for i_value in x_array]
    y_array = [np.array(i_value) for i_value in y_array]
    x_temp = x_array
    y_temp = y_array
    x_size = len(x_array)

    if shuffle:
        new_index = np.random.permutation(x_size)
        x_temp = [x_array[new_index[i_value]] for i_value in np.arange(x_size)]
        y_temp = [y_array[new_index[i_value]] for i_value in np.arange(x_size)]

    batch_count = 0
    x = []
    y = []
    while batch_count * batch_size + batch_size < x_size:
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1

        for i_value in np.arange(start, end):
            x.append(x_temp[i_value])
            y.append(y_temp[i_value])

    return x, y


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


def recode_data(file_name, flag, epoch, x_train, y_train, weights, hidden_nodes, input_nodes):
    MAML_model = Model_Agnostic_Meta_Learning(input_nodes, hidden_nodes)
    MAML_model.build(input_shape=(None, input_nodes))
    MAML_model.summary()

    MAML_model.set_weights(weights)

    x_train_array = x_train
    x_train_tensor = tf.convert_to_tensor(x_train_array)
    x_train_tensor = tf.cast(x_train_tensor, dtype=tf.float32)
    y_predict = MAML_model(x_train_tensor)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_predict = sess.run(y_predict)

    if epoch == 0 and os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, flag) as f:
        f.write("==========================================\tepoch\t" + str(epoch) + "\t==========================================\n")
        length = len(y_train)
        f.write("y_train\n")
        for i_value in np.arange(length):
            if i_value != len(y_train) - 1:
                f.write(str(y_train[i_value][0]) + ",")
            else:
                f.write(str(y_train[i_value][0]) + "\n")

        f.write("y_predict\n")
        for i_value in np.arange(len(y_predict)):
            if i_value != len(y_predict) - 1:
                f.write(str(y_predict[i_value][0]) + ",")
            else:
                f.write(str(y_predict[i_value][0]) + "\n")


def MAML_Generator_function(model, x_train, y_train, hidden_nodes, input_nodes, data_batch_size, epochs_size,
                            nodes_length, lr_inner):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00015)
    losses = []

    for epoch in np.arange(epochs_size):
        total_loss = 0

        for task in np.arange(nodes_length):
            start = data_batch_size * task
            end = data_batch_size * (task + 1)
            x_train_array = x_train[start: end, ]
            x_train_tensor = tf.convert_to_tensor(x_train_array)
            x_train_tensor = tf.cast(x_train_tensor, dtype=tf.float32)
            y_train_array = y_train[start: end, ]
            y_train_tensor = tf.convert_to_tensor(y_train_array)
            y_train_tensor = tf.cast(y_train_tensor, dtype=tf.float32)

            with tf.GradientTape() as test_tape:
                with tf.GradientTape() as train_tape:
                    y_predict_train = model(x_train_tensor)
                    train_loss = tf.keras.backend.mean(tf.keras.losses.MSE(y_true=y_train_tensor,
                                                                           y_pred=y_predict_train))
                gradients_train = train_tape.gradient(train_loss, model.trainable_variables)

                k = 0

                copied_model = Model_Agnostic_Meta_Learning(input_nodes, hidden_nodes)
                copied_model(x_train_tensor)
                copied_model.set_weights(model.get_weights())

                lr_inner = tf.convert_to_tensor(lr_inner)
                lr_inner = tf.cast(lr_inner, dtype=tf.float32)

                for i_value in np.arange(len(model.layers)):
                    copied_model.layers[i_value].kernel = tf.subtract(model.layers[i_value].kernel,
                                                                      tf.multiply(lr_inner, gradients_train[k]))
                    copied_model.layers[i_value].bias = tf.subtract(model.layers[i_value].bias,
                                                                    tf.multiply(lr_inner, gradients_train[k + 1]))
                    k += 2

                y_predict_train = copied_model(x_train_tensor)
                test_loss = tf.keras.backend.mean(tf.keras.losses.mean_squared_error(y_true=y_train_tensor,
                                                                                     y_pred=y_predict_train))
            gradients_test = test_tape.gradient(test_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients_test, model.trainable_variables))

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                temp_loss = sess.run(test_loss)
            total_loss += temp_loss
        loss = total_loss / (task + 1.0)
        losses.append(loss)

        file = path + "\\model_agnostic_meta_learning_predict.txt"
        weights = model.get_weights()
        if epoch == 0:
            recode_data(file, "w+", epoch, x_train, y_train, weights, hidden_nodes, input_nodes)
        else:
            recode_data(file, "a", epoch, x_train, y_train, weights, hidden_nodes, input_nodes)

        print("epoch = " + str(epoch))

    file = path + "\\model_agnostic_meta_learning_losses.txt"
    if os.path.exists(file):
        os.remove(file)

    length = len(losses)
    with open(file, "w+") as f:
        for i_value in np.arange(length):
            f.write(str(losses[i_value]) + "\n")
