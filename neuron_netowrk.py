import threading
from datetime import datetime
from itertools import chain

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import json
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
import os


class ThreadSafePrinter:
    def __init__(self):
        self._lock = threading.Lock()

    def print(self, fold_no, metrics_name, score, val_loss_epochs):
        with self._lock:
            print('------------------------------------------------------------------------')
            print(
                f'Score for fold {fold_no}: {metrics_name} of {score}')
            print(val_loss_epochs)


class SingleFoldThread(threading.Thread):
    def __init__(self, train_set, valid_set, fold_no, activation_fun_layer_1, activation_fun_layer_2, neurons_layer_1,
                 neurons_layer_2, printer: ThreadSafePrinter):
        threading.Thread.__init__(self)
        self.train = train_set
        self.valid = valid_set
        self.fold_no = fold_no
        self.activation_fun_layer_1 = activation_fun_layer_1
        self.activation_fun_layer_2 = activation_fun_layer_2
        self.neurons_layer_1 = neurons_layer_1
        self.neurons_layer_2 = neurons_layer_2
        self.printer = printer
        self.val_loss_epochs = None
        self.val_loss = None
        self.no_epochs_from_val_loss = None

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

        self.val_loss_epochs = [scores[0], len(history.history['loss'])]
        self.val_loss = scores[0]
        self.no_epochs_from_val_loss = len(history.history['loss'])

        self.printer.print(self.fold_no, model.metrics_names[0], scores[0], self.val_loss_epochs)

    def get_return(self):
        if self.is_alive():
            return [0, 0], 0, 0
        else:
            return self.val_loss_epochs, self.val_loss, self.no_epochs_from_val_loss


def training_with_cross_validation(dataset_without_noise, activation_fun_names_layer_1, no_neurons_in_layer_1,
                                   activation_fun_names_layer_2, no_neurons_in_layer_2,
                                   val_loss, no_epochs_from_val_loss):
    print('************************************************************************')
    print("Training starts for:")
    print(f'Layers name: {activation_fun_names_layer_1}, layers no {no_neurons_in_layer_1} | '
          f'Layers name: {activation_fun_names_layer_2}, layers no {no_neurons_in_layer_2}')
    print("Time:", datetime.now().strftime("%H:%M:%S"))

    num_folds = 8
    no_batch_size = 500
    fold_no = 1
    val_loss_epochs = []
    printer = ThreadSafePrinter()
    threads: list = []

    x_train, x_test, y_train, y_test = read_dataset(dataset_without_noise)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    for train, valid in kfold.split(x_train.values):
        # preparation of training set
        input_train = tf.data.Dataset.from_tensor_slices(
            (x_train.values.reshape([len(x_train), 2])[train, :], y_train.values[train]))
        train_dataset = input_train.shuffle(len(y_train.values[train])).batch(no_batch_size)
        # preparation of validation set
        input_valid = tf.data.Dataset.from_tensor_slices((x_train.values.reshape([len(x_train), 2])[valid, :],
                                                          y_train.values[valid]))
        valid_dataset = input_valid.shuffle(len(y_train.values[valid])).batch(no_batch_size)

        thread = SingleFoldThread(train_set=train_dataset, valid_set=valid_dataset, fold_no=fold_no,
                                  activation_fun_layer_1=activation_fun_names_layer_1,
                                  activation_fun_layer_2=activation_fun_names_layer_2,
                                  neurons_layer_1=no_neurons_in_layer_1, neurons_layer_2=no_neurons_in_layer_2,
                                  printer=printer)
        threads.append(thread)
        thread.start()

        # Increase fold number
        fold_no = fold_no + 1

    for th in threads:
        th.join()
        single_val_loss_epoch, single_val_loss, single_no_epochs_from_vl = th.get_return()
        val_loss_epochs.append(single_val_loss_epoch)
        val_loss.append(single_val_loss)
        no_epochs_from_val_loss.append(single_no_epochs_from_vl)

    return val_loss_epochs, val_loss, no_epochs_from_val_loss


def make_training(dataset_without_noise, activation_fun_names_layer_1, no_neurons_in_layer_1,
                  activation_fun_names_layer_2, no_neurons_in_layer_2,
                  val_loss, no_epochs_from_val_loss):
    if no_neurons_in_layer_2 == 0:
        no_of_layers = 1
    else:
        no_of_layers = 2

    file_name = str(no_of_layers) + "_" \
                + activation_fun_names_layer_1 + "_" + str(no_neurons_in_layer_1) + "_" + \
                activation_fun_names_layer_2 + "_" + str(no_neurons_in_layer_2)

    val_loss_epochs, val_loss, no_epochs_from_val_loss = training_with_cross_validation(dataset_without_noise,
                                                                                        activation_fun_names_layer_1,
                                                                                        no_neurons_in_layer_1,
                                                                                        activation_fun_names_layer_2,
                                                                                        no_neurons_in_layer_2,
                                                                                        val_loss,
                                                                                        no_epochs_from_val_loss)

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print("Training ends for:")
    print(f'Layers name: {activation_fun_names_layer_1}, layers no {no_neurons_in_layer_1} | '
          f'Layers name: {activation_fun_names_layer_2}, layers no {no_neurons_in_layer_2}')
    print(val_loss_epochs)
    print("Time:", datetime.now().strftime("%H:%M:%S"))
    # Write to csv files and compute average for loss and number of epochs
    write_to_csv(file_name, val_loss_epochs, no_of_layers)
    # average_loss = get_avarge(val_loss)
    # average_epochs = get_avarge(no_epochs_from_val_loss)
    # average = [[average_loss, int(average_epochs), file_name]]
    # average_file_name = str(no_of_layers) + "_average"
    # write_to_csv(average_file_name, average, no_of_layers)

    return file_name


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

    if not os.path.exists("out/" + path_no_layer):
        os.makedirs("out/" + path_no_layer)

    with open("out/" + path_no_layer + file_name + ".csv", 'a', newline='',
              encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(val_loss)


def get_avarge(array):
    y = 0
    for x in array:
        y = y + x
    average = y / len(array)
    return average


def draw_plot(model):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-10, 10, 0.1)
    Y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(X, Y)
    grid = np.stack((X, Y))
    grid = grid.T.reshape(-1, 2)

    outs = model.predict(grid)
    Z = outs.T[0].reshape(X.shape[0], X.shape[0])
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.autumn,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-4, 4)
    ax.elev = 10
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('3d_plot_out_of_nn_sigmoid_proba.pdf')
    plt.show()


def read_dataset(dataset_without_noise):
    test_dataset_numb = 20000
    dataset = pd.read_csv(dataset_without_noise, names=["x1", "x2", "y"])
    y_dataset = dataset.pop("y")
    x_train, x_test = dataset[:-test_dataset_numb], dataset[-test_dataset_numb:]
    y_train, y_test = y_dataset[:-test_dataset_numb], y_dataset[-test_dataset_numb:]
    return x_train, x_test, y_train, y_test


def make_model(dataset_without_noise):
    # activation_fun_names_1 = ["sigmoid", "tanh", "elu", "swish"]
    activation_fun_names_1 = ["sigmoid"]
    no_neurons_in_layer_1 = list(chain(
        range(2, 10, 1),
        range(10, 21, 2)))

    activation_fun_names_2 = ["sigmoid"]
    no_neurons_in_layer_2 = list(chain(
        range(3, 10, 1),
        range(10, 20, 2),
        range(20, 51, 5)))

    # Loops for nn with one layer
    # for activation_fun_names_layer_1 in activation_fun_names_1:
    #     for no_neurons_in_layer_1 in no_neurons_in_layer_1:
    #         val_loss = []
    #         no_epochs_from_val_loss = []
    #         no_of_layers = 1
    #         for x in range(5):
    #         no_neurons_in_layer_2 = 0
    #         activation_fun_names_layer_2 = "0"
    #         file_name = make_training(dataset_without_noise, activation_fun_names_layer_1, no_neurons_in_layer_1,
    #                                   activation_fun_names_layer_2, no_neurons_in_layer_2, val_loss,
    #                                   no_epochs_from_val_loss)
    #         average_loss = get_avarge(val_loss)
    #         std_dev_of_los = np.std(val_loss)
    #         average_epochs = get_avarge(no_epochs_from_val_loss)
    #         average = [[average_loss, std_dev_of_los, int(average_epochs), file_name]]
    #         average_file_name = str(no_of_layers) + "_" + str(activation_fun_names_layer_1) + "_average"
    #         write_to_csv(average_file_name, average, no_of_layers)

    # Loops for nn with two layers
    for activation_fun_names_layer_1 in activation_fun_names_1:
        for activation_fun_names_layer_2 in activation_fun_names_2:
            for no_neurons_in_layer_1 in no_neurons_in_layer_1:
                for no_neurons_in_layer_2 in no_neurons_in_layer_2:
                    val_loss = []
                    no_epochs_from_val_loss = []
                    no_of_layers = 2
                    # for x in range(5):
                    file_name = make_training(dataset_without_noise, activation_fun_names_layer_1,
                                              no_neurons_in_layer_1,
                                              activation_fun_names_layer_2, no_neurons_in_layer_2, val_loss,
                                              no_epochs_from_val_loss)
                    average_loss = get_avarge(val_loss)
                    std_dev_of_los = np.std(val_loss)
                    average_epochs = get_avarge(no_epochs_from_val_loss)
                    average = [[average_loss, std_dev_of_los, int(average_epochs), file_name]]
                    average_file_name = str(no_of_layers) + "_" + str(activation_fun_names_layer_1) + "_" + \
                                        str(activation_fun_names_layer_2) + "_average"
                    write_to_csv(average_file_name, average, no_of_layers)
