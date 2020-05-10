import pandas as pd
import numpy as np

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

import time


def train_ddn(feature_path, target_path, model_id=str(int(time.time()))):

    model = Sequential()
    model.add(Conv2D(128, (4, 4), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))

    opt = keras.optimizers.Adam()

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    data_feature = np.genfromtxt(feature_path, delimiter=',')
    data_target = np.genfromtxt(target_path, delimiter=',')

    data_feature = data_feature.reshape(data_feature.shape[0], 6, 7, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target)

    model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test))

    model.save("model_" + model_id + ".h5")


train_ddn('features_100000_1589092983.csv', 'targets_100000_1589092983.csv')
