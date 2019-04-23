#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using a mask based appoach.
"""


from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Reshape, Input, ReLU, Activation, BatchNormalization, LSTM
from keras.models import Sequential, Model
from keras.activations import sigmoid
from functools import reduce 

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel


class RnnModel(BaseModel):
    name = 'rnn'

    def build(self, config):
        transformation = Configuration.get('transformation.options', True)

        input_shape = (769, transformation.size, 1)

        model = Sequential()
        model.add(Reshape((769,1), input_shape=input_shape))
        model.add(LSTM(32))
        #model.add(LSTM(64))
        
        #model.add(Flatten())
        model.add(LeakyReLU(alpha=0.3))

        model.add(Dense(769 * transformation.step, kernel_initializer='he_normal'))
        model.add(Reshape((769, transformation.step, 1)))
        #model.add(ReLU(max_value=0.999, negative_slope=0.01))
        model.add(Activation('sigmoid'))

        return model