#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using a mask based appoach.
"""


from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Reshape, Input, ReLU, Activation
from keras.models import Sequential, Model
from keras.activations import sigmoid
from functools import reduce 

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel


class MaskModel(BaseModel):
    name = 'mask'

    def build(self, config):

        transformation = Configuration.get('transformation.options', True)


        input_shape = (769, transformation.size, 1)

        alpha1 = 0.3
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=input_shape))
        model.add(LeakyReLU(alpha=alpha1))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(LeakyReLU(alpha=config.alpha1))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
        model.add(Dropout(0.5))
        
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(LeakyReLU(alpha=alpha1))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(LeakyReLU(alpha=alpha1))
        model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
        model.add(Dropout(0.5)) 
        
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=alpha1))
        model.add(Dropout(0.5))
        
        model.add(Dense(769 * transformation.step))
        model.add(Reshape((769, transformation.step, 1)))
        
        return model