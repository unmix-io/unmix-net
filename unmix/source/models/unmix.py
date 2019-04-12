#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using the unmix.io architecture.
"""

from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, ZeroPadding2D, Cropping2D, Reshape
from keras.models import Sequential

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel

class UnmixModel(BaseModel):
    name = 'unmix'

    def build(self):
        
        input_shape = (769, 64, 1)

        model = Sequential()
        model.add(ZeroPadding2D(((1, 0), (0, 0)), input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        #model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        #model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        #model.add(Dropout(0.25))

        #model.add(UpSampling2D((2, 2)))
        model.add(UpSampling2D((2, 2)))
        model.add(Cropping2D(((1,0), (0,0))))
        model.add(Dense(1, activation='relu'))
        model.add(Reshape((input_shape[0], 1, input_shape[1])))
        model.add(Dense(1))

        return model