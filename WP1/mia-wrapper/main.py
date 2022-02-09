import graimatter_mia as mia
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    (Xt, yt), (Xs, ys), (Xd, yd) = mia.generate_synthetic_data(10000, 2, 300)

    yt = np.eye(2)[yt]
    # Split target dataset into train and test data
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, test_size=0.2)

    # Define target model
    input_data = Input(shape=Xt_train[0].shape)
    x = Dense(128, activation='relu')(input_data)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)

    target_model = Model(input_data, output)
    target_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train target model
    r = target_model.fit(Xt_train,
                         yt_train,
                         validation_data=(Xt_test, yt_test),
                         epochs=10,
                         batch_size=32
                         )

    # Get attack data from target model (could also use a shadow model)
    proba, membership = mia.get_attack_data(target_model, 2, Xt_train, Xt_test)

    # Train attack model
    attack_model = mia.train_attack_model(proba, membership)

    # Evaluate attack
    mia.evaluate_attack(target_model, attack_model, Xt_train, Xt_test, 2)
