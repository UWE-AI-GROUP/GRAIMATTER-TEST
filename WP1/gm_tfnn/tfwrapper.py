import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from typing import List


def config_generator(config_grid: dict) -> List[dict]:
    """
    Generates list of config dictionaries from a grid.

    :param config_grid: example:
    config_grid = {
        'conv': [True],
        'conv_sizes': [(64,), (64, 64)],
        'conv_kernel_size': [(3, 3), (5, 5)],
        'conv_activation': ['tanh', 'relu'],
        'conv_regularizer': [None, 'l2'],
        'dense_sizes': [(128,), (128, 128)],
        'dense_activation': ['relu'],
        'output_activation': ['softmax'],
        'dropout_rate': [0.0, 0.2],
        'regularizer': [None, 'l2'],
        'optimizer': ['adam'],
        'loss': ['categorical_crossentropy'],
        'batch_size': [32, 64],
        'epochs': [1]
    }
    :return:
    """

    config_list = []
    keys = config_grid.keys()
    all_combinations = itertools.product(*config_grid.values())
    for combination in all_combinations:
        config_list.append({n: v for n, v in zip(keys, combination)})
    return config_list


class TFClassifier:

    model = None

    def __init__(self,
                 conv=False,
                 conv_sizes=[64,],
                 conv_kernel_size=[3,3],
                 conv_activation='relu',
                 regularizer=None,
                 dropout_rate=0.0,
                 dense_sizes=[64,],
                 dense_activation='relu',
                 output_activation='softmax',
                 optimizer='adam',
                 loss='categorical_crossentropy',
                 batch_size='32',
                 epochs=1):

        self.conv = conv
        self.conv_sizes = conv_sizes
        self.conv_kernel_size = conv_kernel_size
        self.conv_activation = conv_activation
        self.regularizer = regularizer
        self.dropout_rate = dropout_rate
        self.dense_sizes = dense_sizes
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs

    def _build_network(self, feature_shape, n_classes):
        input_layer = Input(shape=feature_shape)
        x = input_layer

        if self.conv:
            for units in self.conv_sizes:
                x = Conv2D(units,
                           kernel_size=self.conv_kernel_size,
                           activation=self.conv_activation,
                           kernel_regularizer=self.regularizer)(x)
                x = MaxPooling2D()(x)
            x = Flatten()(x)
            x = Dropout(self.dropout_rate)(x)

        for units in self.dense_sizes:
            x = Dense(units,
                      activation=self.dense_activation,
                      kernel_regularizer=self.regularizer)(x)
            x = Dropout(self.dropout_rate)(x)

        output_layer = Dense(n_classes, activation=self.output_activation)(x)

        self.model = Model(input_layer, output_layer)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

    def fit(self, x: np.ndarray, y: np.ndarray):

        feature_shape = x[0].shape
        n_classes = np.max(y) + 1

        self._build_network(feature_shape, n_classes)

        _y = np.eye(n_classes)[y]
        self.model.fit(x, _y,
                       batch_size=self.batch_size,
                       epochs=self.epochs
                       )

    def predict(self, x: np.ndarray):
        return np.argmax(self.model.predict(x), axis=1)

    def predict_proba(self, x: np.ndarray):
        return self.model.predict(x)

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print('Model not initialized.')
