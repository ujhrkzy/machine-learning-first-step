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
        X = data[:, 0]
        y = data[:, 1]

        plt.xlabel(self._x_label)
        plt.ylabel(self._y_label)
        plt.scatter(X, y)
        plt.show()


if __name__ == '__main__':
    data_path = os.path.join("data", "population_profit.txt")
    plot_model = ScatterPlot(load_data_path=data_path,
                             x_label="Population of City in 10,000s",
                             y_label="Profit in $10,000s")
    plot_model.visualize()
