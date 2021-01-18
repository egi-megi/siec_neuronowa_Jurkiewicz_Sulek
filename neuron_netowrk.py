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

def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='sigmoid', kernel_regularizer=l2(1e-5)),
        tf.keras.layers.Dense(4, activation='sigmoid', kernel_regularizer=l2(1e-5)),
        tf.keras.layers.Dense(1)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=['MeanSquaredError'],
                  metrics=['MeanSquaredError'])
    return model, opt

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

    x_train, x_test, y_train, y_test = read_dataset()


    ## Printing dimensions
    print(x_train.dtypes)
    print(x_train.shape, y_train.shape)

    ## Make corss validtion
    n_split = 10
    for train_index, val_index in KFold(n_split).split(x_train, y_train):
        ##x_train_cv, x_val_cv = x_train[train_index], x_train[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
        ##print(x_train_cv.shape, y_train_cv.shape)


    dataset = tf.data.Dataset.from_tensor_slices((x_train.values.reshape([len(x_train), 2]), y_train.values))
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))
    train_dataset = dataset.shuffle(len(x_train)).batch(10)

    model, opt = get_compiled_model()
    hist = model.fit(train_dataset, epochs=5, validation_data=(x_test.values.reshape([len(x_test), 2]), y_test.values))
    print(hist.history)
    with open("out/losses_2layers_100neurons_100000dataset_sigmoid_proba.json", "w") as f:
        json.dump(hist.history, f)
    draw_plot(model)


