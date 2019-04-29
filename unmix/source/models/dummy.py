#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using a dummy architecture.
"""

from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel

class DummyModel(BaseModel):
    name = 'dummy'

    def build(self, config):
        model = Sequential()
        model.add(Dense((2), input_shape=(769, 64, 2)))
        return model