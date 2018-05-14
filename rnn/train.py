# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.keras.api.keras.callbacks import EarlyStopping
from tensorflow.contrib.keras.api.keras.layers import Activation
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.optimizers import Adam

__author__ = "ujihirokazuya"


class DataGenerator(object):

    def __init__(self):
        pass

    @staticmethod
    def sin(x, T=100):
        return np.sin(2.0 * np.pi * x / T)

    def toy_problem(self, T=100, amplitude=0.05):
        x = np.arange(0, 2 * T + 1)
        noise = amplitude * np.random.uniform(low=-1.0, high=1.0, size=len(x))
        return self.sin(x) + noise

    @staticmethod
    def make_dataset(low_data):
        data, target = [], []
        max_length = 25

        for i in range(len(low_data) - max_length):
            data.append(low_data[i:i + max_length])
            target.append(low_data[i + max_length])

        re_data = np.array(data).reshape(len(data), max_length, 1)
        re_target = np.array(target).reshape(len(data), 1)

        return re_data, re_target


class RnnTrainer(object):

    def __init__(self):
        self._generator = DataGenerator()

    def train(self):
        # f = [0 - 200] 、(201,)の1次元ベクトル
        f = self._generator.toy_problem()
        # g = (176, 25, 1)の3次元ベクトル、, h = (176, 1)の2次元ベクトル
        g, h = self._generator.make_dataset(f)
        future_test = g[175].T

        # 1つの学習データの時間の長さ -> 25
        time_length = future_test.shape[1]
        # 未来の予測データを保存していく変数
        future_result = np.empty(0)

        length_of_sequence = g.shape[1]
        in_out_neurons = 1
        n_hidden = 300

        # モデル構築
        model = Sequential()
        model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
        model.add(Dense(in_out_neurons))
        model.add(Activation("linear"))
        optimizer = Adam(lr=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)

        # 学習
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        model.fit(g, h,
                  batch_size=300,
                  epochs=100,
                  validation_split=0.1,
                  callbacks=[early_stopping])

        # 予測
        predicted = model.predict(g)

        # 未来予想
        for step2 in range(400):
            test_data = np.reshape(future_test, (1, time_length, 1))
            batch_predict = model.predict(test_data)

            future_test = np.delete(future_test, 0)
            future_test = np.append(future_test, batch_predict)

            future_result = np.append(future_result, batch_predict)

        # sin波をプロット
        plt.figure()
        plt.plot(range(25, len(predicted)+25), predicted, color="r", label="predict")
        plt.plot(range(0, len(f)), f, color="b", label="row")
        plt.plot(range(0+len(f), len(future_result)+len(f)), future_result, color="g", label="future")
        plt.legend()
        plt.savefig("prediction.png")
        plt.show()


if __name__ == '__main__':
    RnnTrainer().train()
