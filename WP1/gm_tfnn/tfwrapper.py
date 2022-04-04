import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

from scipy.special import softmax

from typing import List, Optional


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
                 conv: bool = False,
                 conv_sizes: list = [64,],
                 conv_kernel_size: list =[3,3],
                 conv_activation: str = 'relu',
                 regularizer: Optional[str] = None,
                 dropout_rate: float = 0.0,
                 dense_sizes: list = [64,],
                 dense_activation: str = 'relu',
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 1,
                 # DP-Patch
                 use_dp: bool = False,
                 l2_norm_clip: float = 1.0,
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 microbatches: int = 32
                ):

        self.conv = conv
        self.conv_sizes = conv_sizes
        self.conv_kernel_size = conv_kernel_size
        self.conv_activation = conv_activation
        self.regularizer = regularizer
        self.dropout_rate = dropout_rate
        self.dense_sizes = dense_sizes
        self.dense_activation = dense_activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        # DP-Patch
        self.use_dp = use_dp
        self.l2_norm_clip = l2_norm_clip
        self.epsilon = epsilon
        self.delta = delta
        self.microbatches = microbatches

    def _build_network(self, dataset_size, feature_shape, n_classes):        
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

        output_layer = Dense(n_classes)(x)
        
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
        if self.use_dp:
            noise_multiplier = compute_noise(dataset_size, 
                                             self.batch_size, 
                                             self.epsilon, 
                                             self.epochs, 
                                             self.delta, 
                                             1e-12)
            optimizer = DPKerasAdamOptimizer(
                            l2_norm_clip=self.l2_norm_clip,
                            noise_multiplier=noise_multiplier,
                            num_microbatches=self.microbatches,
                            learning_rate=self.learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model = Model(input_layer, output_layer)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def fit(self, x: np.ndarray, y: np.ndarray):

        feature_shape = x[0].shape
        n_classes = np.max(y) + 1

        self._build_network(len(x), feature_shape, n_classes)

        _y = np.eye(n_classes)[y]
        self.model.fit(x, _y,
                       batch_size=self.batch_size,
                       epochs=self.epochs
                       )

    def predict(self, x: np.ndarray):
        return np.argmax(self.model.predict(x), axis=1)

    def predict_proba(self, x: np.ndarray):
        logits = self.model.predict(x)
        return softmax(logits, axis=1)

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print('Model not initialized.')
