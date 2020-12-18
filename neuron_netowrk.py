import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='elu'),
    tf.keras.layers.Dense(25, activation='elu'),
    tf.keras.layers.Dense(25, activation='elu'),
    tf.keras.layers.Dense(25, activation='elu'),
    tf.keras.layers.Dense(25, activation='elu'),
    tf.keras.layers.Dense(25, activation='elu'),
    tf.keras.layers.Dense(25, activation='elu'),
    tf.keras.layers.Dense(25, activation='elu'),
    tf.keras.layers.Dense(25, activation='elu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=['MeanSquaredError'],
                metrics=['MeanSquaredError'])
  return model


def load_dataset(flatten=False):
    train = pd.read_csv('dataset_10000.csv',  names=["x1", "x2", "y"])
    y_train = train.pop("y")
    test = pd.read_csv('dataset_10000.csv', names=["x1", "x2", "y"])
    #x_test = test.pop("x")
    #y_test = test.pop("y")
    #x_train = x_train.astype(float)
    #x_test = x_test.astype(float)
    # we reserve the last 100 training examples for validation
    train, val = train[:-1000], train[-1000:]
    y_train, y_val = y_train[:-1000], y_train[-1000:]
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

    model = get_compiled_model()
    model.fit(train_dataset, epochs=32)



