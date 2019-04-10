#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using a mask based appoach.
"""

from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Reshape, Input, ReLU, Activation
from keras.models import Sequential, Model
from keras.activations import sigmoid
from unmix.source.configuration import Configuration


name = 'mask'

def generate(alpha1, alpha2, rate, channels=2):
    # input_shape = (769, 108, 1)
    #
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
    #                  input_shape=input_shape))  # conv2d_1
    # model.add(LeakyReLU(alpha=alpha1))  # leaky_re_lu_1
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))  # conv2d_2
    # model.add(LeakyReLU(alpha=alpha1))  # leaky_re_lu_2
    # model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))  # max_pooling2d_1
    # model.add(Dropout(0.5))  # dropout_1
    #
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # conv2d_3
    # model.add(LeakyReLU(alpha=alpha1))  # leaky_re_lu_3
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))  # conv2d_4
    # model.add(LeakyReLU(alpha=alpha1))  # leaky_re_lu_4
    # model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))  # max_pooling2d_2
    # model.add(Dropout(0.5))  # dropout_2
    #
    # model.add(Flatten())  # flatten_1
    # model.add(Dense(128))  # dense_1
    # model.add(LeakyReLU(alpha=alpha1))  # leaky_re_lu_5
    # model.add(Dropout(0.5))  # dropout_3
    #
    # model.add(Dense(769))  # dense_2
    # model.add(Reshape((769, 1, 1)))
    #
    # return model


    batch_size = Configuration.get("training.batch_size")
    window_size = Configuration.get('transformation.options.size')
    input = Input(batch_shape=(None, 769, window_size, 1), name='input')
    model = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    # model = ReLU(alpha=alpha1)(model)
    model = Conv2D(16, (3, 3), activation='relu', padding='same')(model)
    # model = LeakyReLU(alpha=alpha1)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
    # model = LeakyReLU(alpha=alpha1)(model)
    model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
    # model = LeakyReLU(alpha=alpha1)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.5)(model)
    model = Flatten()(model)
    # model = Dense(384, activation='relu', )(model)
    # model = LeakyReLU(alpha=alpha1)(model)
    # model = Dropout(0.5)(model)
    model = Dense(769, activation='relu', )(model)
    model = Reshape((769, 1, 1))(model)
    print(model.shape)
    return Model(input=input, outputs=model)
