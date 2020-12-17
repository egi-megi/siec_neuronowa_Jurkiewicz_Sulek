import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=['MeanSquaredError'],
                metrics=['accuracy'])
  return model


def load_dataset(flatten=False):
    train = pd.read_csv('dataset_10000.csv',  names=["X","y"])
    X_train = train.pop("X")
    y_train = train.pop("y")
    test = pd.read_csv('dataset_10000.csv', names=["X","y"])
    X_test = test.pop("X")
    y_test = test.pop("y")
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)  # we reserve the last 100 training examples for validation
    X_train, X_val = X_train[:-1000], X_train[-1000:]
    y_train, y_val = y_train[:-1000], y_train[-1000:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
        return X_train, y_train, X_val, y_val, X_test, y_train, X_val, y_val, X_test, y_test
    ## Printing dimensions
    print(train.dtypes)
    print(X_train.shape, y_train.shape)
    dataset = tf.data.Dataset.from_tensor_slices((X_train.values.reshape([9000, 1]), y_train.values))
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))
    train_dataset = dataset.shuffle(len(X_train)).batch(1)

    model = get_compiled_model()
    model.fit(train_dataset, epochs=15)



