import pandas as pd
import numpy as np

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

import time


def train_nn_1(feature_path, target_path, model_id=str(int(time.time()))):

    model = Sequential()
    model.add(Conv2D(64, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))

    opt = keras.optimizers.Adam()

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    data_feature = np.genfromtxt(feature_path, delimiter=',')
    data_target = np.genfromtxt(target_path, delimiter=',')

    data_feature = data_feature.reshape(data_feature.shape[0], 6, 7, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target)

    model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

    model.save("model_1_" + model_id + ".h5")


def train_nn_2(feature_path, target_path, model_id=str(int(time.time()))):

    model = Sequential()
    model.add(Conv2D(64, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))

    # opt = keras.optimizers.RMSprop(learning_rate=0.0001)
    # opt = keras.optimizers.SGD(lr=0.001, decay=1e-4)
    # opt = keras.optimizers.Adadelta()
    opt = keras.optimizers.Adam()

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])

    data_feature = np.genfromtxt(feature_path, delimiter=',')
    data_target = np.genfromtxt(target_path, delimiter=',')

    data_feature = data_feature.reshape(data_feature.shape[0], 6, 7, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target)

    model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

    model.save("model_2_" + model_id + ".h5")


def train_nn_3(feature_path, target_path, model_id=str(int(time.time()))):

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(1))

    opt = keras.optimizers.Adam()

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])

    data_feature = np.genfromtxt(feature_path, delimiter=',')
    data_target = np.genfromtxt(target_path, delimiter=',')

    data_feature = data_feature.reshape(data_feature.shape[0], 6, 7, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target)

    model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

    model.save("model_3_" + model_id + ".h5")


def train_nn_4(feature_path, target_path, model_id=str(int(time.time()))):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(1))

    opt = keras.optimizers.Adam()

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])

    data_feature = np.genfromtxt(feature_path, delimiter=',')
    data_target = np.genfromtxt(target_path, delimiter=',')

    data_feature = data_feature.reshape(data_feature.shape[0], 6, 7, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target)

    model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

    model.save("model_4_" + model_id + ".h5")


def train_nn_5(feature_path, target_path, model_id=str(int(time.time()))):
    model = Sequential()
    model.add(Conv2D(128, (4, 4), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(1))

    opt = keras.optimizers.Adam()

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])

    data_feature = np.genfromtxt(feature_path, delimiter=',')
    data_target = np.genfromtxt(target_path, delimiter=',')

    data_feature = data_feature.reshape(data_feature.shape[0], 6, 7, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target)

    model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

    model.save("model_5_" + model_id + ".h5")




def train_nn_6(feature_path, target_path, model_id=str(int(time.time()))):

    model = Sequential()
    model.add(Conv2D(256, (4, 4), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2), input_shape=(6, 7, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(1))

    opt = keras.optimizers.Adam()

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])

    data_feature = np.genfromtxt(feature_path, delimiter=',')
    data_target = np.genfromtxt(target_path, delimiter=',')

    data_feature = data_feature.reshape(data_feature.shape[0], 6, 7, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target)

    model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

    model.save("model_6_" + model_id + ".h5")



# train_nn_1('TrainingData/features_100000_Random_Random_1589134303.csv', 'TrainingData/targets_100000_Random_Random_1589134303.csv')
# train_nn_2('TrainingData/features_100000_Random_Random_1589134303.csv', 'TrainingData/targets_100000_Random_Random_1589134303.csv')
# train_nn_3('TrainingData/features_100000_Random_Random_1589134303.csv', 'TrainingData/targets_100000_Random_Random_1589134303.csv')
# train_nn_4('TrainingData/features_100000_Random_Random_1589134303.csv', 'TrainingData/targets_100000_Random_Random_1589134303.csv')
# train_nn_5('TrainingData/features_100000_Random_Random_1589134303.csv', 'TrainingData/targets_100000_Random_Random_1589134303.csv')
# train_nn_6('TrainingData/features_100000_Random_Random_1589134303.csv', 'TrainingData/targets_100000_Random_Random_1589134303.csv')


