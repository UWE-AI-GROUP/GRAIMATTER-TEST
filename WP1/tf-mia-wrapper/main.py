import graimatter_mia_tf as mia
import numpy as np
import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import mnist

def test2():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    config_grid = {
        'feature_shape': [x_train[0].shape],
        'n_classes': [10],
        'conv': [True],
        'conv_sizes': [(64,), (64, 64)],
        'conv_kernel_size': [(3, 3), (5, 5)],
        'conv_activation': ['relu'],
        'conv_regularizer': [None],
        'dense_sizes': [(128,), (128, 128)],
        'dense_activation': ['relu'],
        'output_activation': ['softmax'],
        'dropout_rate': [0.0, 0.2],
        'regularizer': [None],
        'optimizer': ['adam'],
        'loss': ['categorical_crossentropy'],
        'batch_size': [32, 64],
        'epochs': [1]
    }

    configs = models.config_generator(config_grid)
    for config in configs:
        models._TFModel_from_config(x_train, x_test, y_train, y_test, config)

if __name__ == '__main__':
    test2()
    # (Xt, yt), (Xs, ys), (Xd, yd) = mia.generate_synthetic_data(10000, 2, 300)
    #
    # yt = np.eye(2)[yt]
    # # Split target dataset into train and test data
    # Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, test_size=0.2)
    #
    # # Define target model
    # input_data = Input(shape=Xt_train[0].shape)
    # x = Dense(128, activation='relu')(input_data)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # output = Dense(2, activation='softmax')(x)
    #
    # target_model = Model(input_data, output)
    # target_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # # Train target model
    # r = target_model.fit(Xt_train,
    #                      yt_train,
    #                      validation_data=(Xt_test, yt_test),
    #                      epochs=10,
    #                      batch_size=32
    #                      )
    #
    # # Get attack data from target model (could also use a shadow model)
    # proba, membership = mia.get_attack_data(target_model, 2, Xt_train, Xt_test)
    #
    # # Train attack model
    # attack_model = mia.train_attack_model(proba, membership)
    #
    # # Evaluate attack
    # mia.evaluate_attack(target_model, attack_model, Xt_train, Xt_test, 2)