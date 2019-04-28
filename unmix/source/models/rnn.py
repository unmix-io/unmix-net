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
        model.add(Reshape((transformation.size,769), input_shape=input_shape))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(512, return_sequences=True))
        model.add(LSTM(1024))
        
        model.add(LeakyReLU(alpha=0.3))

        model.add(Dense(769 * transformation.step))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Reshape((769, transformation.step, 1)))

        return model