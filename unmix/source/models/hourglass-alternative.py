#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using LeakyReLU activation layers.
"""

from keras.models import *
from keras.layers import *

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel

# Base implementation from: https://github.com/sungheonpark/music_source_sepearation_SH_net


class HourglassModel(BaseModel):
    name = 'Hourglass-Alternative'

    def build(self, config):
        transformation = Configuration.get('transformation.options', True)

        input_shape = (769, transformation.size, 1)
        input = Input(input_shape)
        conv = Conv2D(64, kernel_size=(7, 7), strides=(1, 1),
                      padding='same', activation='relu')(input)
        conv = Conv2D(128, kernel_size=(3, 3),
                      padding='same', activation='relu')(conv)
        conv = Conv2D(128, kernel_size=(3, 3),
                      padding='same', activation='relu')(conv)
        conv = Conv2D(128, kernel_size=(3, 3),
                      padding='same', activation='relu')(conv)
        conv = Conv2D(128, kernel_size=(3, 3),
                      padding='same', activation='relu')(conv)

        inter = conv

        outputs = []
        for i in range(config.options.stacks):
            hourglass = self.__build_hourglass(inter, config.options.stacks)

            conv = Conv2D(128, kernel_size=(3, 3), padding='same',
                          activation='relu')(hourglass)
            conv = Conv2D(128, kernel_size=(1, 1),
                          padding='same', activation='relu')(conv)

            conv = Conv2D(128, kernel_size=(1, 1), padding='same')(conv)
            # Reshape((769, transformation.step, 1))(conv)
            outputs.append(conv)

            if i < 3:
                # Residual link across hourglasses
                inter = Add()([inter, conv, Conv2D(
                    128, kernel_size=(1, 1), padding='same')(outputs[-1])])
        return Model(input=input, outputs=outputs[-1])

    def __build_hourglass(self, input, stacks, size=128):
        upper_1 = Conv2D(size, kernel_size=(3, 3),
                         padding='same', activation='relu')(input)

        # Lower branch
        pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(input)
        lower_1 = Conv2D(size, kernel_size=(3, 3),
                         padding='same', activation='relu')(pool)
        # Recursive hourglass
        if stacks > 1:
            lower_2 = self.__build_hourglass(lower_1, stacks - 1, size)
        else:
            lower_2 = Conv2D(size, kernel_size=(3, 3),
                             padding='same', activation='relu')(pool)
        lower_3 = Conv2D(size, kernel_size=(3, 3),
                         padding='same', activation='relu')(lower_2)

        upper_2 = UpSampling2D(size=tuple(map(lambda x, y: 
            x.value / y.value, upper_1.shape.dims[1:3], lower_3.shape.dims[1:3])))(lower_3)
        return Add()([upper_1, upper_2])
