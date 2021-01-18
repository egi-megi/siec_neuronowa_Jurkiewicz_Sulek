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

    num_folds = 10
    no_batch_size = 100
    no_epochs = 1
    verbosity = 1


    ## Printing dimensions
    print(x_train.dtypes)
    print(x_train.shape, y_train.shape)

    #input_x = tf.data.Dataset.from_tensor_slices(x_train.values.reshape([len(x_train), 2]))
    #input_y = tf.data.Dataset.from_tensor_slices(y_train.values)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x_train.values):
        model, opt = get_compiled_model()

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        input= tf.data.Dataset.from_tensor_slices((x_train.values.reshape([len(x_train), 2])[train,:],y_train.values[train]))
        print((x_train.values.reshape([len(x_train), 2])[train,:].shape))
        print(y_train.values[train].shape)
        train_dataset = input.shuffle(len(y_train.values[train])).batch(no_batch_size)
        for feat, targ in input.take(5):
            print('Features: {}, Target: {}'.format(feat, targ))
        # Fit data to model
        history = model.fit(train_dataset,
                            epochs=no_epochs,
                            verbose=verbosity)

        # Generate generalization metrics
        input = tf.data.Dataset.from_tensor_slices((x_train.values.reshape([len(x_train), 2])[test,:],
                                                   y_train.values[test]))
        test_dataset=input.shuffle(len(y_train.values[test])).batch(no_batch_size)
        scores = model.evaluate(test_dataset, verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

        # Increase fold number
        fold_no = fold_no + 1

    # dataset = tf.data.Dataset.from_tensor_slices((x_train.values.reshape([len(x_train), 2]), y_train.values))
    #for feat, targ in dataset.take(5):
    #    print('Features: {}, Target: {}'.format(feat, targ))
    #train_dataset = dataset.shuffle(len(x_train)).batch(10)

        #model, opt = get_compiled_model()
        #hist = model.fit(train_dataset, epochs=5, validation_data=(x_test.values.reshape([len(x_test), 2]), y_test.values))
        #print(hist.history)
        #with open("out/losses_2layers_100neurons_100000dataset_sigmoid_proba.json", "w") as f:
        #    json.dump(hist.history, f)
        #draw_plot(model)


