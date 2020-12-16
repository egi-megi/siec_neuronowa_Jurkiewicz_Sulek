import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv




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
    print(X_train.shape, y_train.shape)


    ######################## set learning variables ##################
    learning_rate = 0.0005
    epochs = 2000
    batch_size = 3

    ######################## set some variables #######################
    x = tf.placeholder(tf.float32, [None, 1], name='x')  # 1 features
    y = tf.placeholder(tf.float32, [None, 1], name='y')  # 1 outputs

    # hidden layer 1
    W1 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.truncated_normal([10]), name='b1')

    # hidden layer 2
    W2 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.truncated_normal([1]), name='b2')

    ######################## Activations, outputs ######################
    # output hidden layer 1
    hidden_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

    # total output
    y_ = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))

    ####################### Loss Function  #########################
    mse = tf.losses.mean_squared_error(y, y_)

    ####################### Optimizer      #########################
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

    ###################### Initialize, Accuracy and Run #################
    # initialize variables
    init_op = tf.global_variables_initializer()

    # accuracy for the test set
    accuracy = tf.reduce_mean(tf.square(tf.subtract(y, y_)))  # or could use tf.losses.mean_squared_error

    # run
    with tf.Session() as sess:
      sess.run(init_op)
      total_batch = int(len(y_train) / batch_size)
      for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
          batch_x, batch_y = X_train[i * batch_size:min(i * batch_size + batch_size, len(X_train)), :], \
                             y_train[i * batch_size:min(i * batch_size + batch_size, len(y_train)), :]
          _, c = sess.run([optimizer, mse], feed_dict={x: batch_x, y: batch_y})
          avg_cost += c / total_batch
        if epoch % 10 == 0:
          print('Epoch:', (epoch + 1), 'cost =', '{:.3f}'.format(avg_cost))
      print(sess.run(mse, feed_dict={x: X_test, y: y_test}))