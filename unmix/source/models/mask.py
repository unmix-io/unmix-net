#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using a mask based appoach.
"""


from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Reshape, Input, ReLU, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.activations import sigmoid
from functools import reduce 

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel


class MaskModel(BaseModel):
    name = 'mask'

    def build(self, config):
        transformation = Configuration.get('transformation.options', False)

        input_shape = (769, transformation.size, 1)

        dropout_rate = config.options.dropout_rate
        alpha = config.options.alpha
        filter_factor = config.options.filter_factor
        
        model = Sequential()
        model.add(Conv2D(32 * filter_factor, (3, 3), padding='same',
                         input_shape=input_shape, kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Conv2D(16 * filter_factor, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=alpha))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(dropout_rate))
        
        model.add(Conv2D(64 * filter_factor, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Conv2D(16 * filter_factor, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=alpha))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(dropout_rate)) 
        
        model.add(Flatten())
        model.add(Dense(256, kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))
        #model.add(BatchNormalization())

        model.add(Dense(769 * transformation.step, kernel_initializer='he_normal'))
        model.add(Reshape((769, transformation.step, 1)))
        #model.add(ReLU(max_value=0.999, negative_slope=0.01))
        #model.add(Activation('sigmoid'))
        return model