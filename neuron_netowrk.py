import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def get_compiled_model(activation_method):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation=activation_method, kernel_initializer=tf.keras.initializers.ones, input_shape=(2,)),
        # tf.keras.layers.Dense(150, activation=activation_method, kernel_initializer=tf.keras.initializers.ones),
        # tf.keras.layers.Dense(25, activation=activation_method),
        # tf.keras.layers.Dense(25, activation=activation_method),
        # tf.keras.layers.Dense(25, activation=activation_method),
        # tf.keras.layers.Dense(25, activation=activation_method),
        # tf.keras.layers.Dense(25, activation=activation_method),
        # tf.keras.layers.Dense(25, activation=activation_method),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
    ])

    model.compile(optimizer='adam',
                loss=['MeanSquaredError'],
                metrics=['MeanSquaredError'])
    return model


def load_dataset(flatten=False):
    ACTIVATION_METHOD = tf.keras.activations.sigmoid
    EPOCHS = 300

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train = pd.read_csv('dataset_100000.csv',  names=["x1", "x2", "y"])
    y_train = train.pop("y")
    test = pd.read_csv('dataset_100000.csv', names=["x1", "x2", "y"])
    # we reserve the last 1000 training examples for validation
    train, val = train[:-10000], train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    if flatten:
        #x_train = x_train.reshape([x_train.shape[0], -1])
        x_val = val.reshape([val.shape[0], -1])
        #x_test = x_test.reshape([x_test.shape[0], -1])
        #return x_train, y_train, x_val, y_val, x_test, y_train, x_val, y_val, x_test, y_test
    ## Printing dimensions
    print(train.dtypes)
    print(train.shape, y_train.shape)
    dataset = tf.data.Dataset.from_tensor_slices((train.values.reshape([len(train), 2]), y_train.values))
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))
    train_dataset = dataset.shuffle(len(train)).batch(1)

    model = get_compiled_model(ACTIVATION_METHOD)
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=(val.values.reshape([len(val), 2]), y_val.values))

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
    #    np.sin(0.5 * X + Y) * np.exp(-(2 * X * X + 2 * Y * Y) / 100) * 2 + np.cos((X + Y) * 0.5)
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
    plt.savefig('3d_plot_out_of_nn.pdf')
    plt.show()

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()



