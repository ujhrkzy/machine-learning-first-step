# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras import optimizers
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.models import Sequential

__author__ = "ujihirokazuya"


class LinerRegressionTrainer(object):

    def __init__(self, load_data_path, learning_rate, epoch_size):
        self._load_data_path = load_data_path
        self._learning_rate = learning_rate
        self._epoch_size = epoch_size

    def train_with_keras(self):
        # build model
        model = Sequential()
        # see https://keras.io/ja/regularizers if kernel_regularizer is used.
        model.add(Dense(1, input_dim=1))
        optimizer = optimizers.SGD(lr=self._learning_rate)
        model.compile(loss='mse', optimizer=optimizer)

        # load data
        train_X_data, train_y_data = self._load_data()

        # train
        model.fit(train_X_data, train_y_data, epochs=self._epoch_size)
        model.save("result/liner_regression_keras" + ".h5")

        # check model
        def predict(x):
            return model.predict(x)
        self._visualize(predict, train_X_data, train_y_data, np.linspace(0, 25), "result/liner_regression_keras.png")

    def train_with_tf(self):
        # create calculation graph(build model)
        X_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        W = tf.Variable(tf.zeros([1]))
        b = tf.Variable(tf.zeros([1]))
        # Hypothesis
        y = W * X_data + b
        # Loss function(Cost function)
        loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        model = optimizer.minimize(loss)

        # load data
        train_X_data, train_y_data = self._load_data()

        # train
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for e in range(self._epoch_size):
                sess.run(model, feed_dict={X_data: train_X_data, y_data: train_y_data})
                if e % 100 == 0:
                    print(e, sess.run(loss, feed_dict={X_data: train_X_data, y_data: train_y_data}))
            loss_value = sess.run(loss, feed_dict={X_data: train_X_data, y_data: train_y_data})
            W_value = sess.run(W)
            b_value = sess.run(b)
            print("loss:{}, W:{}, b:{}".format(loss_value, W_value, b_value))
            saver = tf.train.Saver()
            saver.save(sess, "result/liner_regression_tf.ckpt", global_step=self._epoch_size)

        # check model
        def predict(x):
            return W_value * x + b_value
        self._visualize(predict, train_X_data, train_y_data, np.linspace(0, 25), "result/liner_regression_tf.png")

    def train_with_tf_dense(self):
        # create calculation graph(build model)
        X_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        y = tf.layers.dense(X_data, units=1)
        # Loss function(Cost function)
        loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        model = optimizer.minimize(loss)

        # load data
        train_X_data, train_y_data = self._load_data()

        # train
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for e in range(self._epoch_size):
                sess.run(model, feed_dict={X_data: train_X_data, y_data: train_y_data})
                if e % 100 == 0:
                    print(e, sess.run(loss, feed_dict={X_data: train_X_data, y_data: train_y_data}))
            print(sess.run(loss, feed_dict={X_data: train_X_data, y_data: train_y_data}))
            saver = tf.train.Saver()
            saver.save(sess, "result/liner_regression_tf_dense.ckpt", global_step=self._epoch_size)

            # check model
            def predict(x):
                tmp_values = list()
                for tmp in x:
                    tmp_values.append([tmp])
                return sess.run(y, feed_dict={X_data: tmp_values})
            self._visualize(predict, train_X_data, train_y_data, np.linspace(0, 25),
                            "result/liner_regression_tf_dense.png")

    def _load_data(self):
        data = np.genfromtxt(self._load_data_path, delimiter=',')
        X = data[:, (0,)]
        y = data[:, (1,)]
        return X, y

    @staticmethod
    def _visualize(predictor, scatter_X, scatter_y, X, save_path=None):
        plt.scatter(scatter_X, scatter_y)
        line_y_values = predictor(X)
        plt.plot(X, line_y_values)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    data_path = os.path.join("data", "population_profit.txt")
    tmp_learning_rate = 0.01
    iterations = 1500
    trainer = LinerRegressionTrainer(data_path, tmp_learning_rate, iterations)
    trainer.train_with_tf()
    trainer.train_with_tf_dense()
    trainer.train_with_keras()
