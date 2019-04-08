#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using a mask based appoach.
"""

from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Reshape
from keras.models import Sequential

from unmix.source.configuration import Configuration


name = 'mask'

def generate(alpha1, alpha2, rate, channels=2):
    input_shape = (769, 25, 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                input_shape=input_shape))                               # conv2d_1
    model.add(LeakyReLU(alpha=alpha1))                                  # leaky_re_lu_1
    model.add(Conv2D(16, (3, 3), padding='same'))    # conv2d_2
    model.add(LeakyReLU(alpha=alpha1))                                  # leaky_re_lu_2
    model.add(MaxPooling2D(pool_size=(3, 3)))           # max_pooling2d_1
    model.add(Dropout(0.1))                                            # dropout_1

    model.add(Conv2D(64, (3, 3), padding='same'))    # conv2d_3
    model.add(LeakyReLU(alpha=alpha1))                                  # leaky_re_lu_3
    model.add(Conv2D(16, (3, 3), padding='same'))    # conv2d_4
    model.add(LeakyReLU(alpha=alpha1))                                  # leaky_re_lu_4
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))           # max_pooling2d_2
    model.add(Dropout(0.1))                                            # dropout_2

    model.add(Flatten())                                                # flatten_1
    model.add(Dense(128))                                               # dense_1
    model.add(LeakyReLU(alpha=alpha1))                                  # leaky_re_lu_5
    model.add(Dropout(0.1))                                            # dropout_3

    model.add(Dense(769))                                               # dense_2
    model.add(Reshape((769, 1, 1)))
    return model