import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv




def load_dataset(flatten=False):
    train = pd.read_csv('dataset_1000.csv',  names=["X","y"])
    X_train = train.pop("X")
    y_train = train.pop("y")
    test = pd.read_csv('dataset_1000.csv', names=["X","y"])
    X_test = test.pop("X")
    y_test = test.pop("y")
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)  # we reserve the last 100 training examples for validation
    X_train, X_val = X_train[:-100], X_train[-100:]
    y_train, y_val = y_train[:-100], y_train[-100:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
        return X_train, y_train, X_val, y_val, X_test, y_train, X_val, y_val, X_test, y_test
    ## Printing dimensions
    print(X_train.shape, y_train.shape)
