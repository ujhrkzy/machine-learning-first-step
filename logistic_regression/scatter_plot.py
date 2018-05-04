# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

__author__ = "ujihirokazuya"


class ScatterPlot(object):

    def __init__(self, load_data_path, x_label, y_label):
        self._load_data_path = load_data_path
        self._x_label = x_label
        self._y_label = y_label

    def visualize(self):
        data = np.genfromtxt(self._load_data_path, delimiter=',')
        labels = data[:, 2]
        ok_index = np.where(labels == 1)
        ng_index = np.where(labels == 0)
        X = data[:, 0]
        y = data[:, 1]
        ok_X = X[ok_index]
        ok_y = y[ok_index]
        ng_X = X[ng_index]
        ng_y = y[ng_index]

        plt.xlabel(self._x_label)
        plt.ylabel(self._y_label)
        plt.scatter(ok_X, ok_y, c="green")
        plt.scatter(ng_X, ng_y, c="red")
        plt.show()


if __name__ == '__main__':
    data_path = os.path.join("data", "exam_scores.txt")
    plot_model = ScatterPlot(load_data_path=data_path,
                             x_label="Exam1 Score",
                             y_label="Exam2 Score")
    plot_model.visualize()
