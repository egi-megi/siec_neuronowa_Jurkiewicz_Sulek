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

def training_with_cross_validation(dataset_without_noise, activation_fun_names_layer_1, no_neurons_in_layer_1,
                  activation_fun_names_layer_2, no_neurons_in_layer_2,
                  val_loss, no_epochs_from_val_loss):
    num_folds = 5
    no_batch_size = 10000
    no_epochs = 500
    verbosity = 1
    fold_no = 1
    val_loss_epochs = []

    x_train, x_test, y_train, y_test = read_dataset(dataset_without_noise)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    for train, valid in kfold.split(x_train.values):
        model, opt = get_compiled_model(activation_fun_names_layer_1, no_neurons_in_layer_1,
                                        activation_fun_names_layer_2, no_neurons_in_layer_2)
        input_train = tf.data.Dataset.from_tensor_slices(
            (x_train.values.reshape([len(x_train), 2])[train, :], y_train.values[train]))
        train_dataset = input_train.shuffle(len(y_train.values[train])).batch(no_batch_size)
        # Stop criterion, patience - number of worse loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001)
        # Generate generalization metrics
        input_valid = tf.data.Dataset.from_tensor_slices((x_train.values.reshape([len(x_train), 2])[valid, :],
                                                          y_train.values[valid]))
        valid_dataset = input_valid.shuffle(len(y_train.values[valid])).batch(no_batch_size)
        # Fit data to model
        history = model.fit(train_dataset,
                            epochs=no_epochs, callbacks=[callback],
                            verbose=verbosity, validation_data=valid_dataset)


        scores = model.evaluate(valid_dataset, verbose=0)

        val_loss_epochs.append([scores[0], len(history.history['loss'])])
        val_loss.append(scores[0])
        no_epochs_from_val_loss.append(len(history.history['loss']))

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}')
        print(val_loss_epochs)

        # Increase fold number
        fold_no = fold_no + 1

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
                                                                                        activation_fun_names_layer_1, no_neurons_in_layer_1,
                                                                                        activation_fun_names_layer_2, no_neurons_in_layer_2,
                                                                                        val_loss, no_epochs_from_val_loss)

    # Write to csv files and compute average for loss and number of epochs
    write_to_csv(file_name, val_loss_epochs, no_of_layers)
    #average_loss = get_avarge(val_loss)
    #average_epochs = get_avarge(no_epochs_from_val_loss)
    #average = [[average_loss, int(average_epochs), file_name]]
    #average_file_name = str(no_of_layers) + "_average"
    #write_to_csv(average_file_name, average, no_of_layers)

    return file_name


def get_compiled_model(activation_fun_names_layer_1, no_neurons_in_layer_1, activation_fun_names_layer_2, no_neurons_in_layer_2):

    if no_neurons_in_layer_2 == 0:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(no_neurons_in_layer_1, activation=activation_fun_names_layer_1, kernel_regularizer=l2(1e-5), kernel_initializer=tf.keras.initializers.ones),
            tf.keras.layers.Dense(1)
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(no_neurons_in_layer_1, activation=activation_fun_names_layer_1, kernel_regularizer=l2(1e-5), kernel_initializer=tf.keras.initializers.ones),
            tf.keras.layers.Dense(no_neurons_in_layer_2, activation=activation_fun_names_layer_2, kernel_regularizer=l2(1e-5), kernel_initializer=tf.keras.initializers.ones),
            tf.keras.layers.Dense(1)
        ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=['MeanSquaredError'],
                  metrics=['MeanSquaredError'])
    print(f'Layers name: {activation_fun_names_layer_1}, layers no {no_neurons_in_layer_1}')
    print(f'Layers name: {activation_fun_names_layer_2}, layers no {no_neurons_in_layer_2}')

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
        y=y+x
    average = y/len(array)
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

    #activation_fun_names = ["sigmoid", "tanh", "elu", "swish"]
    activation_fun_names = ["tanh", "elu", "swish"]
    no_neurons_in_layer = range(4, 20, 3)

    # Loops for nn with one layer
    for activation_fun_names_layer_1 in activation_fun_names:
        for no_neurons_in_layer_1 in no_neurons_in_layer:
            val_loss = []
            no_epochs_from_val_loss = []
            no_of_layers = 1
            for x in range(5):
                no_neurons_in_layer_2 = 0
                activation_fun_names_layer_2 = "0"
                file_name = make_training(dataset_without_noise, activation_fun_names_layer_1, no_neurons_in_layer_1,
                              activation_fun_names_layer_2, no_neurons_in_layer_2, val_loss, no_epochs_from_val_loss)
            average_loss = get_avarge(val_loss)
            std_dev_of_los = np.std(val_loss)
            average_epochs = get_avarge(no_epochs_from_val_loss)
            average = [[average_loss, std_dev_of_los, int(average_epochs), file_name]]
            average_file_name = str(no_of_layers) + "_average"
            write_to_csv(average_file_name, average, no_of_layers)

    # Loops for nn with two layers
    for activation_fun_names_layer_1 in activation_fun_names:
        for activation_fun_names_layer_2 in activation_fun_names:
            for no_neurons_in_layer_1 in no_neurons_in_layer:
                for no_neurons_in_layer_2 in no_neurons_in_layer:
                    val_loss = []
                    no_epochs_from_val_loss = []
                    no_of_layers = 2
                    for x in range(10):
                        file_name = make_training(dataset_without_noise, activation_fun_names_layer_1, no_neurons_in_layer_1,
                                      activation_fun_names_layer_2, no_neurons_in_layer_2, val_loss, no_epochs_from_val_loss)
                    average_loss = get_avarge(val_loss)
                    average_epochs = get_avarge(no_epochs_from_val_loss)
                    average = [[average_loss, int(average_epochs), file_name]]
                    average_file_name = str(no_of_layers) + "_average"
                    write_to_csv(average_file_name, average, no_of_layers)



