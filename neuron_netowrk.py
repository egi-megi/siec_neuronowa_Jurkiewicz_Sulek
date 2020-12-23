import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import json
from tensorflow.keras.regularizers import l2

def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='sigmoid', kernel_regularizer=l2(1e-5), kernel_initializer=tf.keras.initializers.ones),
        tf.keras.layers.Dense(100, activation='sigmoid', kernel_regularizer=l2(1e-5), kernel_initializer=tf.keras.initializers.ones),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear, kernel_initializer=tf.keras.initializers.ones)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=['MeanSquaredError'],
                  metrics=['MeanSquaredError'])
    return model, opt


def load_dataset(flatten=False):
    val_dataset_numb = 10000
    train = pd.read_csv('dataset/dataset_100000.csv', names=["x1", "x2", "y"])
    y_train = train.pop("y")
    test = pd.read_csv('dataset/dataset_100000.csv', names=["x1", "x2", "y"])
    train, val = train[:-val_dataset_numb], train[-val_dataset_numb:]
    y_train, y_val = y_train[:-val_dataset_numb], y_train[-val_dataset_numb:]
    if flatten:
        x_val = val.reshape([val.shape[0], -1])
    ## Printing dimensions
    print(train.dtypes)
    print(train.shape, y_train.shape)
    dataset = tf.data.Dataset.from_tensor_slices((train.values.reshape([len(train), 2]), y_train.values))
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))
    train_dataset = dataset.shuffle(len(train)).batch(10)

    model, opt = get_compiled_model()
    hist = model.fit(train_dataset, epochs=50, validation_data=(val.values.reshape([len(val), 2]), y_val.values))
    print(hist.history)
    with open("out/losses_2layers_100neurons_100000dataset_sigmoid.json", "w") as f:
        json.dump(hist.history, f)
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
    plt.savefig('3d_plot_out_of_nn_sigmoid.pdf')
    plt.show()
