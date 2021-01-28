import threading
from datetime import datetime

import tensorflow as tf
import pandas as pd
import csv
from tensorflow.keras.regularizers import l2
import os


class SingleFoldThread(threading.Thread):
    def __init__(self, train_set, val_set, test_set, activation_fun_layer_1, activation_fun_layer_2, neurons_layer_1,
                 neurons_layer_2):
        threading.Thread.__init__(self)
        self.train = train_set
        self.valid = val_set
        self.test = test_set
        self.activation_fun_layer_1 = activation_fun_layer_1
        self.activation_fun_layer_2 = activation_fun_layer_2
        self.neurons_layer_1 = neurons_layer_1
        self.neurons_layer_2 = neurons_layer_2
        self.val_loss = None
        self.test_loss = None

    def run(self):
        no_epochs = 500
        verbosity = 0

        model, opt = get_compiled_model(self.activation_fun_layer_1, self.neurons_layer_1,
                                        self.activation_fun_layer_2, self.neurons_layer_2)
        # Stop criterion, patience - number of worse loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, min_delta=0.001)
        # Fit data to model
        history = model.fit(self.train,
                            epochs=no_epochs, callbacks=[callback],
                            verbose=verbosity, validation_data=self.valid)

        scores = model.evaluate(self.valid, verbose=0)
        test_scores = make_evaluation(model, self.test)

        self.val_loss = scores[0]
        self.test_loss = test_scores[0]

    def get_return(self):
        if self.is_alive():
            return 0, 0
        else:
            return self.val_loss, self.test_loss


def training_with_test(training_set: str, test_set: str, activation_fun_name_layer_1, no_neurons_in_layer_1,
                                   activation_fun_name_layer_2, no_neurons_in_layer_2):
    print('************************************************************************')
    print("Training starts for:")
    print("training set:" + training_set + " test set:" + test_set)
    print(f'Layers name: {activation_fun_name_layer_1}, layers no {no_neurons_in_layer_1} | '
          f'Layers name: {activation_fun_name_layer_2}, layers no {no_neurons_in_layer_2}')
    print("Time:", datetime.now().strftime("%H:%M:%S"))

    no_batch_size = 500

    x_train, x_val, y_train, y_val = read_training_dataset(training_set)
    test_dataset = read_test_dataset(test_set)

    # preparation of training set
    input_train = tf.data.Dataset.from_tensor_slices(
            (x_train.values.reshape([len(x_train), 2]), y_train.values))
    train_dataset = input_train.shuffle(len(y_train.values)).batch(no_batch_size)

    input_val = tf.data.Dataset.from_tensor_slices(
        (x_val.values.reshape([len(x_val), 2]), y_val.values))
    val_dataset = input_val.shuffle(len(y_val.values)).batch(no_batch_size)

    thread = SingleFoldThread(train_set=train_dataset, val_set=val_dataset, test_set=test_dataset,
                              activation_fun_layer_1=activation_fun_name_layer_1,
                              activation_fun_layer_2=activation_fun_name_layer_2,
                              neurons_layer_1=no_neurons_in_layer_1, neurons_layer_2=no_neurons_in_layer_2,
                              )
    thread.start()
    thread.join()
    single_val_loss, single_test_loss = thread.get_return()

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print("Training ends for:")
    print("training set:" + training_set + " test set:" + test_set)
    print(f'Layers name: {activation_fun_name_layer_1}, layers no {no_neurons_in_layer_1} | '
          f'Layers name: {activation_fun_name_layer_2}, layers no {no_neurons_in_layer_2}')
    print("validation accuracy:" + str(single_val_loss) + "test accuracy:" + str(single_test_loss))
    print("Time:", datetime.now().strftime("%H:%M:%S"))

    return single_val_loss, single_test_loss


def get_compiled_model(activation_fun_names_layer_1, no_neurons_in_layer_1, activation_fun_names_layer_2,
                       no_neurons_in_layer_2):
    if no_neurons_in_layer_2 == 0:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(no_neurons_in_layer_1, activation=activation_fun_names_layer_1,
                                  kernel_regularizer=l2(1e-5), kernel_initializer=tf.keras.initializers.ones),
            tf.keras.layers.Dense(1)
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(no_neurons_in_layer_1, activation=activation_fun_names_layer_1,
                                  kernel_regularizer=l2(1e-5), kernel_initializer=tf.keras.initializers.ones),
            tf.keras.layers.Dense(no_neurons_in_layer_2, activation=activation_fun_names_layer_2,
                                  kernel_regularizer=l2(1e-5), kernel_initializer=tf.keras.initializers.ones),
            tf.keras.layers.Dense(1)
        ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=['MeanSquaredError'],
                  metrics=['MeanSquaredError'])
    # print(f'Layers name: {activation_fun_names_layer_1}, layers no {no_neurons_in_layer_1}')
    # print(f'Layers name: {activation_fun_names_layer_2}, layers no {no_neurons_in_layer_2}')

    return model, opt


def write_to_csv(file_name, val_loss, no_of_layers):
    if no_of_layers == 1:
        path_no_layer = "one/"
    else:
        path_no_layer = "two/"

    if not os.path.exists("test/"):
        os.makedirs("test/")

    if not os.path.exists("test/" + path_no_layer):
        os.makedirs("test/" + path_no_layer)

    with open("test/" + path_no_layer + file_name + ".csv", 'a', newline='',
              encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(val_loss)


def read_training_dataset(training_dataset):
    val_dataset_numb = 20000
    dataset = pd.read_csv(training_dataset, names=["x1", "x2", "y"])
    y_dataset = dataset.pop("y")
    x_train, x_val = dataset[:-val_dataset_numb], dataset[-val_dataset_numb:]
    y_train, y_val = y_dataset[:-val_dataset_numb], y_dataset[-val_dataset_numb:]
    return x_train, x_val, y_train, y_val


def read_test_dataset(test_dataset):
    dataset = pd.read_csv(test_dataset, names=["x1", "x2", "y"])
    return dataset


def make_evaluation(model, test_set):
    y_test_set = test_set.pop("y")

    loss_test = model.evaluate(
        x=test_set, y=y_test_set, batch_size=None, verbose=0, sample_weight=None, steps=None,
        callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
        return_dict=False
    )

    print(f'Loss for test dataset: {loss_test}')

    return loss_test


def test_everything(test_set):
    params = [
        ("sigmoid", "sigmoid", 65, 0),
        ("tanh", "tanh", 45, 0),
        ("elu", "elu", 140, 0),
        ("swish", "swish", 80, 0),
        ("sigmoid", "sigmoid", 45, 18),
        ("tanh", "tanh", 50, 9),
        ("elu", "elu", 12, 7),
        ("swish", "swish", 3, 25),
        ]

    train_sets = [
        ("dataset/dataset_noise_100000_005.csv", 0.005),
        ("dataset/dataset_noise_100000_01.csv", 0.1),
        ("dataset/dataset_noise_100000_1.csv", 1),
        ("dataset/dataset_noise_100000_5.csv", 5),
        ("dataset/dataset_noise_100000_10.csv", 10),
        ("dataset/dataset_noise_100000_20.csv", 20),
    ]

    for activation_1, activation_2, neurons_1, neurons_2 in params:
        best_ones = []
        if neurons_2 == 0:
            no_of_layers = 1
        else:
            no_of_layers = 2
        for training_set, noise in train_sets:
            val_loss = []
            test_loss = []
            for x in range(5):
                single_val_loss, single_test_loss = training_with_test(
                    training_set=training_set,
                    test_set=test_set,
                    activation_fun_name_layer_1=activation_1,
                    no_neurons_in_layer_1=neurons_1,
                    activation_fun_name_layer_2=activation_2,
                    no_neurons_in_layer_2=neurons_2
                )
                val_loss.append(single_val_loss)
                test_loss.append(single_test_loss)

            minimal_val_loss_idx = val_loss.index(min(test_loss))
            val_loss_of_best = val_loss[minimal_val_loss_idx]
            test_loss_of_best = test_loss[minimal_val_loss_idx]
            best_ones.append((noise, test_loss_of_best, val_loss_of_best))

            print('------------------------------------------------------------------------')
            print("Best result for:")
            print("training set:" + training_set + " test set:" + test_set)
            print(f'Layers name: {activation_1}, layers no {neurons_1} | '
                  f'Layers name: {activation_2}, layers no {neurons_2}')
            print("validation accuracy:" + str(val_loss_of_best))
            print("test accuracy:" + str(test_loss_of_best))

            file_name = str(no_of_layers) + "_" + \
                        activation_1 + "_" + str(neurons_1) + "_" + \
                        activation_2 + "_" + str(neurons_2) + "_noise_" + \
                        str(noise)
            write_to_csv(file_name, zip(val_loss, test_loss), no_of_layers)
        file_name_for_best = str(no_of_layers) + "_" + \
                        activation_1 + "_" + str(neurons_1) + "_" + \
                        activation_2 + "_" + str(neurons_2) + "_best"
        write_to_csv(file_name_for_best, best_ones, no_of_layers)


if __name__ == '__main__':
    # turn off CUDA support
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset_without_noise = "dataset/dataset_100000.csv"
    test_everything(test_set=dataset_without_noise)
