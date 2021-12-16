from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import tensorflow.keras as keras

import tensorflow as tf
import numpy as np
import datetime
import os


def create_mlp_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation=keras.activations.softmax))
    return model

def cnn_classique():
    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":

    run_id = "MLP_140epochs_lr001" + str(datetime.datetime.now())
    run_id = run_id.replace(" ", "_").replace(":", "_")

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    print("shape of x_train",x_train.shape)
    print("shape of y_train", y_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = create_mlp_model()

    logdir = f"./logs/{run_id}"
    tensorboard_callback = keras.callbacks.TensorBoard(logdir)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.categorical_accuracy])


    model.predict(x_test)

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=140, batch_size=1024,
              callbacks=[tensorboard_callback,
                         keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                           patience=20)])


