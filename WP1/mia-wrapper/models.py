import numpy as np
import itertools

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import keras_tuner as kt

# scikit-learn classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from typing import List, Union

Sklearn_classifier = Union[MLPClassifier,
                           KNeighborsClassifier,
                           SVC,
                           GaussianProcessClassifier,
                           DecisionTreeClassifier,
                           RandomForestClassifier,
                           AdaBoostClassifier,
                           XGBClassifier]
TensorFlow_classifier = Union[Model, Sequential]
Classifier = Union[Sklearn_classifier, TensorFlow_classifier]

SUPPORTED_ALGORITHMS = ['MLPClassifier',
                        'KNeighborsClassifier',
                        'SVC',
                        'GaussianProcessClassifier',
                        'DecisionTreeClassifier',
                        'RandomForestClassifier',
                        'AdaBoostClassifier',
                        'XGBClassifier',
                        'NeuralNetwork']


def train_model(algorithm: str, x: np.ndarray, y: np.ndarray) -> Classifier:
    """

    :param algorithm: Name of a supported classifier algorithm. Supported algorithms include 'MLPClassifier',
    'KNeighborsClassifier', 'SVC', 'GaussianProcessClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier',
    'AdaBoostClassifier', 'XGBClassifier', and 'NeuralNetwork'.
    :param x: Training features
    :param y: Training labels
    :return: A trained classifier
    """

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise Exception()

    if algorithm == 'MLPClassifier':
        return _train_MLPClassifier(x, y)
    elif algorithm == 'KNeighborsClassifier':
        return _train_KNeighborsClassifier(x, y)
    elif algorithm == 'SVC':
        return _train_SVC(x, y)
    elif algorithm == 'GaussianProcessClassifier':
        return _train_GaussianProcessClassifier(x, y)
    elif algorithm == 'DecisionTreeClassifier':
        return _train_DecisionTreeClassifier(x, y)
    elif algorithm == 'RandomForestClassifier':
        return _train_RandomForestClassifier(x, y)
    elif algorithm == 'AdaBoostClassifier':
        return _train_AdaBoostClassifier(x, y)
    elif algorithm == 'XGBClassifier':
        return _train_XGBClassifier(x, y)
    elif algorithm == 'NeuralNetwork':
        return _train_TFNeuralNetwork(x, y)
    else:
        raise Exception()


def _train_MLPClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn multilayer perceptron classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained MLPClassifier
    """
    base_clf = MLPClassifier(max_iter=500)

    param_grid = {
        'hidden_layer_sizes': [(64,), (32, 32), (32, 32, 32)],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_


def _train_KNeighborsClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn k-Nearest Neighbours classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained KNeighborsClassifier
    """
    base_clf = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_


def _train_SVC(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Support Vector Machine classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained SVC
    """
    base_clf = SVC(probability=True)

    param_grid = {
        'C': [1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.001, 0.0001],
        'kernel': ['linear', 'rbf']
    }

    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_


def _train_GaussianProcessClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Gaussian Process classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained GaussianProcessClassifier
    """

    base_clf = GaussianProcessClassifier()

    param_grid = {
        'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_


def _train_DecisionTreeClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Decision Tree classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained DecisionTreeClassifier
    """

    base_clf = DecisionTreeClassifier()

    param_grid = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': [0.1, .01, .001],
        'max_depth': [5, 6, 7, 8, 9],
        'criterion': ['gini', 'entropy']
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_


def _train_RandomForestClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Random Forest classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained RandomForestClassifier
    """

    base_clf = RandomForestClassifier()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_


def _train_AdaBoostClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn Ada Boost classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained AdaBoostClassifier
    """

    base_clf = AdaBoostClassifier()

    param_grid = {
     'n_estimators': np.arange(10, 300, 10),
     'learning_rate': [0.01, 0.05, 0.1, 1]
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_


def _train_XGBClassifier(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a scikit-learn XGBoost classifier using GridSearch.

    :param x: Features
    :param y: Labels
    :return: A trained XGBClassifier
    """

    base_clf = XGBClassifier()

    param_grid = {
        'max_depth': range(2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]
    }
    n_jobs = -1

    clf = GridSearchCV(base_clf, param_grid=param_grid, cv=3, n_jobs=n_jobs, scoring='roc_auc', refit=True, verbose=0)
    clf.fit(x, y)

    return clf.best_estimator_


def _train_TFNeuralNetwork(x: np.ndarray, y: np.ndarray) -> Classifier:
    """
    Trains a TensorFlow classifier using Keras Tuner.
    :param x: Features
    :param y: Labels
    :return: A Tensorflow classifier
    """

    # One-hot encoding of labels
    n_classes = np.max(y) + 1
    y_oh = np.eye(n_classes)[y]

    # Multi-layer perceptron builder function with different hyperparameter options
    def build_mlp(hp):
        model = Sequential()
        for i in range(hp.Int('dense_layers', 1, 4)):
            model.add(Dense(
                units=hp.Int(f'dense_layer_{i}_units', 32, 128, step=32),
                activation='relu'
            ))
            model.add(Dropout(
                rate=hp.Float(f'dense_layer_{i}_dropout', 0.0, 0.5)
            ))
        model.add(Dense(n_classes))

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-4])
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    # Convolutional model builder function with different hyperparameter options
    def build_convnet(hp):
        model = Sequential()
        for i in range(hp.Int('conv_layers', 1, 3)):
            model.add(Conv2D(
                filters=hp.Int(f'conv_layer_{i}_filters', 32, 128, step=32),
                kernel_size=(3, 3),
                activation=hp.Choice(f'conv_layer_{i}_act', ['relu', 'tanh'])
            ))
            model.add(MaxPooling2D())
        model.add(Flatten())
        for i in range(hp.Int('dense_layers', 1, 2)):
            model.add(Dense(
                units=hp.Int(f'dense_layer_{i}_units', 32, 128, step=32),
                activation='relu'
            ))
            model.add(Dropout(
                rate=hp.Float(f'dense_layer_{i}_dropout', 0.0, 0.5)
            ))
        model.add(Dense(n_classes))

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-4])
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    # Decide whether an MLP or a ConvNet is needed: in general, inputs for MLPs
    # are of shape (n_instances, n_features) while inputs for ConvNets are of
    # shape (n_instances, height, width, channels).
    if len(x.shape) == 2:
        # Hyperparameter tuner for a MLP
        tuner = kt.Hyperband(build_mlp,
                             objective='val_accuracy',
                             max_epochs=10,
                             factor=3,
                             project_name='tf_model_hp')
    elif len(x.shape) == 4:
        # Hyperparameter tuner for a ConvNet
        tuner = kt.Hyperband(build_convnet,
                             objective='val_accuracy',
                             max_epochs=10,
                             factor=3,
                             project_name='tf_model_hp')
    else:
        raise Exception()

    # Callback to stop training when validation loss peaks
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Search for optimal hyperparameters and return the best combination
    tuner.search(x, y_oh, epochs=50, validation_split=0.2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    clf = tuner.hypermodel.build(best_hps)

    # Train network
    clf.fit(x, y_oh, epochs=50, validation_split=0.2)

    return clf


def config_generator(config_grid: dict) -> List[dict]:
    """
    Generates list of config dictionaries from a grid.

    :param config_grid: example:
    config_grid = {
        'feature_shape': [x_train[0].shape],
        'n_classes': [10],
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


def _TFModel_from_config(x_train: np.ndarray,
                         x_test: np.ndarray,
                         y_train: np.ndarray,
                         y_test: np.ndarray,
                         config: dict) -> TensorFlow_classifier:
    """
    Instantiate relatively simple TF models from a dict of parameters

    :param x_train: Train features
    :param x_test: Test features
    :param y_train: Train labels
    :param y_test: Test labels
    :param config: Example config file:
    config = {
        'feature_shape': (32, 32, 3),
        'conv': True,
        'conv_sizes': (64, 64),
        'conv_kernel_size': (3,3),
        'conv_activation': 'relu',
        'conv_regularizer': None,
        'dense_sizes': (128, 128),
        'dense_activation': 'relu',
        'output_activation': 'softmax',
        'dropout_rate': 0.0,
        'regularizer': None,
        'n_classes': 10,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy'
    }
    :return:
    """

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

    input_layer = Input(shape=config['feature_shape'])
    x = input_layer

    if config['conv']:
        for units in config['conv_sizes']:
            x = Conv2D(units,
                       kernel_size=config['conv_kernel_size'],
                       activation=config['conv_activation'],
                       kernel_regularizer=config['regularizer'])(x)
            x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dropout(config['dropout_rate'])(x)

    for units in config['dense_sizes']:
        x = Dense(units,
                  activation=config['dense_activation'],
                  kernel_regularizer=config['regularizer'])(x)
        x = Dropout(config['dropout_rate'])(x)

    output_layer = Dense(config['n_classes'], activation=config['output_activation'])(x)

    model = Model(input_layer, output_layer)
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train,
              batch_size=config['batch_size'],
              epochs=config['epochs'],
              validation_data=(x_test, y_test),
              callbacks=[early_stopping]
              )

    return model
