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

def make_training(no_of_layers, activation_fun_names_layer_1, no_neurons_in_layer_1, activation_fun_names_layer_2, no_neurons_in_layer_2):

    x_train, x_test, y_train, y_test = read_dataset()

    num_folds = 3
    no_batch_size = 100000
    no_epochs = 200
    verbosity = 1
    fold_no = 1
    val_loss = []
    no_epochs_for_each_kfold = []

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    file_name = str(no_of_layers) + "_" \
                + activation_fun_names_layer_1 + "_" + str(no_neurons_in_layer_1) + "_" + \
                activation_fun_names_layer_2 + "_" + str(no_neurons_in_layer_2) + ".csv"

    # K-fold Cross Validation model evaluation
    for train, valid in kfold.split(x_train.values):

        model, opt = get_compiled_model(no_of_layers, activation_fun_names_layer_1, no_neurons_in_layer_1,
                                        activation_fun_names_layer_2, no_neurons_in_layer_2)

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        input_train = tf.data.Dataset.from_tensor_slices(
            (x_train.values.reshape([len(x_train), 2])[train, :], y_train.values[train]))

        train_dataset = input_train.shuffle(len(y_train.values[train])).batch(no_batch_size)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        # Fit data to model
        history = model.fit(train_dataset,
                            epochs=no_epochs, callbacks=[callback],
                            verbose=verbosity)

        #val_loss.append([len(history.history['loss'])])

        # Generate generalization metrics
        input_valid = tf.data.Dataset.from_tensor_slices((x_train.values.reshape([len(x_train), 2])[valid, :],
                                                    y_train.values[valid]))
        test_dataset = input_valid.shuffle(len(y_train.values[valid])).batch(no_batch_size)
        scores = model.evaluate(test_dataset, verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}')
        val_loss.append([scores[0], len(history.history['loss'])])

        # Increase fold number
        fold_no = fold_no + 1

        print(val_loss)
    write_to_csv(file_name, val_loss, no_of_layers)

def get_compiled_model(no_of_layers, activation_fun_names_layer_1, no_neurons_in_layer_1, activation_fun_names_layer_2, no_neurons_in_layer_2):

    if no_of_layers == 1:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(no_neurons_in_layer_1, activation=activation_fun_names_layer_1, kernel_regularizer=l2(1e-5)),
            tf.keras.layers.Dense(1)
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(no_neurons_in_layer_1, activation=activation_fun_names_layer_1, kernel_regularizer=l2(1e-5)),
            tf.keras.layers.Dense(no_neurons_in_layer_2, activation=activation_fun_names_layer_2, kernel_regularizer=l2(1e-5)),
            tf.keras.layers.Dense(1)
        ])

    print(f'Layers name: {activation_fun_names_layer_1}, layers no {no_neurons_in_layer_1}')
    print(f'Layers name: {activation_fun_names_layer_2}, layers no {no_neurons_in_layer_2}')

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=['MeanSquaredError'],
                  metrics=['MeanSquaredError'])
    return model, opt

def write_to_csv(file_name, val_loss, no_of_layers):

    path_no_layer = ""

    if no_of_layers == 1:
        path_no_layer = "one/"
    else:
        path_no_layer = "two/"

    if not os.path.exists("out/" + path_no_layer):
        os.makedirs("out/" + path_no_layer)

    with open("out/" + path_no_layer + file_name, 'a', newline='',
                  encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(val_loss)


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

def read_dataset():
    test_dataset_numb = 20000
    dataset = pd.read_csv('dataset/dataset_100000.csv', names=["x1", "x2", "y"])
    y_dataset = dataset.pop("y")
    x_train, x_test = dataset[:-test_dataset_numb], dataset[-test_dataset_numb:]
    y_train, y_test = y_dataset[:-test_dataset_numb], y_dataset[-test_dataset_numb:]
    return x_train, x_test, y_train, y_test

def make_model():

    #activation_fun_names = ["sigmoid", "tanh", "elu", "swish"]
    activation_fun_names = ["sigmoid"]
    no_neurons_in_layer = [100]

    for no_of_layers in range(1, 2):
        for activation_fun_names_layer_1 in activation_fun_names:
            for activation_fun_names_layer_2 in activation_fun_names:
                for no_neurons_in_layer_1 in no_neurons_in_layer:
                    for no_neurons_in_layer_2 in no_neurons_in_layer:
                        make_training(no_of_layers, activation_fun_names_layer_1, no_neurons_in_layer_1, activation_fun_names_layer_2, no_neurons_in_layer_2)




