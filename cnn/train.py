# -*- coding: utf-8 -*-

import os
from datetime import datetime

import numpy as np
from tensorflow.contrib.keras.api.keras import optimizers
from tensorflow.contrib.keras.api.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.contrib.keras.api.keras.datasets import mnist
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.utils import to_categorical

__author__ = "ujihirokazuya"

_current_path = os.path.abspath(os.path.dirname(__file__))
_log_root_path = os.path.join(_current_path, "log")


class CnnTrainer(object):

    _date_format = "%Y%m%d_%H%M%S"
    _model_file_name_suffix = "_{epoch:03d}.h5"
    _parameter_format = "lr-{}_dr-{}"

    def __init__(self):
        self.learning_rate = 0.001
        self.epoch_size = 50
        self.dropout_rate = 0.2
        self._file_name_format = None

    @property
    def _model_file_name_prefix(self):
        if self._file_name_format is None:
            line = "{}_{}".format(datetime.now().strftime(self._date_format), self._param_str)
            self._file_name_format = line
        return self._file_name_format

    @property
    def _model_file_name(self):
        return self._model_file_name_prefix + self._model_file_name_suffix

    @property
    def _param_str(self):
        line = self._parameter_format.format(self.learning_rate, self.dropout_rate)
        return line

    @property
    def _log_dir(self):
        return os.path.join(_log_root_path, self._model_file_name_prefix)

    def train_with_keras(self):
        # load data
        (train_X_data, train_y_data), (test_X_data, test_y_data) = mnist.load_data()
        # 画素を0.0-1.0の範囲に変換
        train_X_data = train_X_data.astype(np.float32)
        row_dimension = 28
        train_X_data = train_X_data.reshape(train_X_data.shape[0], row_dimension, row_dimension, 1)
        test_X_data = test_X_data.astype(np.float32)
        test_X_data = test_X_data.reshape(test_X_data.shape[0], row_dimension, row_dimension, 1)
        train_X_data /= 255
        test_X_data /= 255
        # one-hot-encoding
        nb_classes = 10
        train_y_data = to_categorical(train_y_data, nb_classes)
        test_y_data = to_categorical(test_y_data, nb_classes)

        # build model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(row_dimension, row_dimension, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout_rate))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(nb_classes, activation='softmax'))
        optimizer = optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["acc"])

        # train
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = ModelCheckpoint(self._model_file_name, save_best_only=True, save_weights_only=False,
                                           period=10)
        model.fit(train_X_data, train_y_data, validation_data=(test_X_data, test_y_data),
                  epochs=self.epoch_size, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, TensorBoard(log_dir=self._log_dir)])
        model.save(os.path.join("result", self._model_file_name_prefix + ".h5"))


if __name__ == '__main__':
    trainer = CnnTrainer()
    trainer.train_with_keras()
