# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.contrib.keras.api.keras import optimizers
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.models import Sequential

__author__ = "ujihirokazuya"


class LogisticRegressionTrainer(object):

    def __init__(self, load_data_path, learning_rate, epoch_size):
        self._load_data_path = load_data_path
        self._learning_rate = learning_rate
        self._epoch_size = epoch_size

    def train_with_keras(self):
        # build model
        model = Sequential()
        # see https://keras.io/ja/regularizers if kernel_regularizer is used.
        model.add(Dense(1, input_dim=2, activation="sigmoid"))
        optimizer = optimizers.Adam(lr=self._learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        # load data
        train_X_data, train_y_data = self._load_data()
        train_X_data = preprocessing.scale(train_X_data)

        # train
        model.fit(train_X_data, train_y_data, epochs=self._epoch_size, batch_size=100)
        model.save("result/liner_regression_keras" + ".h5")

        # check model
        def predict(x):
            predictions = (model.predict(x) >= 0.5).astype(np.int32)
            y_data = train_y_data.astype(np.int32).tolist()
            result = predictions == y_data
            return result
        self._visualize(predict, train_X_data, "result/logistic_regression_keras.png")

    def train_with_tf(self):
        # create calculation graph(build model)
        X_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        W = tf.Variable(tf.zeros([2]))
        b = tf.Variable(tf.zeros([1]))
        # Hypothesis
        y = W * X_data + b
        y = tf.sigmoid(y)
        # Loss function(Cost function)
        loss = tf.reduce_mean((-1 * y_data * tf.log(y)) - ((1 - y_data) * tf.log(1 - y)))
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        model = optimizer.minimize(loss)

        # define accuracy
        prediction = tf.round(y)
        correct = tf.cast(tf.equal(prediction, y_data), dtype=tf.float32)
        accuracy = tf.reduce_mean(correct)

        # load data
        train_X_data, train_y_data = self._load_data()
        train_X_data = preprocessing.scale(train_X_data)

        # train
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for e in range(self._epoch_size):
                sess.run(model, feed_dict={X_data: train_X_data, y_data: train_y_data})
                if e % 100 == 0:
                    loss_value = sess.run(loss, feed_dict={X_data: train_X_data, y_data: train_y_data})
                    acc_value = sess.run(accuracy, feed_dict={X_data: train_X_data, y_data: train_y_data})
                    print("epoch:{}, loss:{}, acc:{}".format(e, loss_value, acc_value))
            loss_value = sess.run(loss, feed_dict={X_data: train_X_data, y_data: train_y_data})
            W_value = sess.run(W)
            b_value = sess.run(b)
            print("loss:{}, W:{}, b:{}".format(loss_value, W_value, b_value))
            saver = tf.train.Saver()
            saver.save(sess, "result/logistic_regression_tf.ckpt", global_step=self._epoch_size)

            # check model
            def predict(x):
                return sess.run(correct, feed_dict={X_data: x, y_data: train_y_data})
            self._visualize(predict, train_X_data, "result/logistic_regression_tf.png")

    def train_with_tf_dense(self):
        # create calculation graph(build model)
        X_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # Hypothesis
        y = tf.layers.dense(X_data, units=1)
        # Loss function(Cost function)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_data))
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        model = optimizer.minimize(loss)

        # define accuracy
        prediction = tf.round(tf.sigmoid(y))
        correct = tf.cast(tf.equal(prediction, y_data), dtype=tf.float32)
        accuracy = tf.reduce_mean(correct)

        # load data
        train_X_data, train_y_data = self._load_data()
        train_X_data = preprocessing.scale(train_X_data)

        # train
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for e in range(self._epoch_size):
                sess.run(model, feed_dict={X_data: train_X_data, y_data: train_y_data})
                if e % 100 == 0:
                    loss_value = sess.run(loss, feed_dict={X_data: train_X_data, y_data: train_y_data})
                    acc_value = sess.run(accuracy, feed_dict={X_data: train_X_data, y_data: train_y_data})
                    print("epoch:{}, loss:{}, acc:{}".format(e, loss_value, acc_value))
            print("end")
            loss_value = sess.run(loss, feed_dict={X_data: train_X_data, y_data: train_y_data})
            acc_value = sess.run(accuracy, feed_dict={X_data: train_X_data, y_data: train_y_data})
            print("loss:{}, acc:{}".format(loss_value, acc_value))
            saver = tf.train.Saver()
            saver.save(sess, "result/logistic_regression_tf_dense.ckpt", global_step=self._epoch_size)

            # check model
            def predict(x):
                return sess.run(correct, feed_dict={X_data: x, y_data: train_y_data})
            self._visualize(predict, train_X_data, "result/logistic_regression_tf_dense.png")

    def _load_data(self):
        data = np.genfromtxt(self._load_data_path, delimiter=',')
        X = data[:, (0, 1)]
        y = data[:, (2,)]
        return X, y

    @staticmethod
    def _visualize(predictor, scatter_X, save_path=None):

        def _build_plot(accuracy_values, X_values, ok_color, ng_color):
            ok_index = np.where(accuracy_values == 1)[0]
            ng_index = np.where(accuracy_values == 0)[0]
            x1 = X_values[:, 0]
            x2 = X_values[:, 1]
            ok_x = x1[ok_index]
            ok_y = x2[ok_index]
            ng_x = x1[ng_index]
            ng_y = x2[ng_index]
            plt.scatter(ok_x, ok_y, c=ok_color)
            plt.scatter(ng_x, ng_y, c=ng_color)
        accuracy = predictor(scatter_X)
        _build_plot(accuracy_values=accuracy, X_values=scatter_X, ok_color="blue", ng_color="yellow")
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    data_path = os.path.join("data", "exam_scores.txt")
    tmp_learning_rate = 0.01
    iterations = 1500
    trainer = LogisticRegressionTrainer(data_path, tmp_learning_rate, iterations)
    trainer.train_with_tf()
    trainer.train_with_tf_dense()
    trainer.train_with_keras()
