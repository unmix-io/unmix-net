#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using LeakyReLU activation layers.
"""

from keras.models import *
from keras.layers import *

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel

"""
Modified implementation based on:
Title: Music Source Separation Using Stacked Hourglass Networks
Author: Sungheon Park, Taehoon Kim, Kyogu Lee, Nojun Kwak
Date: 2018-06-22
Paper: https://arxiv.org/abs/1805.08559
Availability: https://github.com/sungheonpark/music_source_sepearation_SH_net
"""


class HourglassModel(BaseModel):
    name = 'Hourglass-Alternative'

    def build(self, config):
        transformation = Configuration.get('transformation.options', False)

        input_shape = (769, transformation.size, 1)
        input_initial = Input(input_shape)
        input = Cropping2D(cropping=((1, 0), (0, 0)))(input_initial)
        conv = Conv2D(64, kernel_size=(7, 7), strides=(1, 1),
                      padding='same', activation='relu')(input)
        conv = Conv2D(128, kernel_size=(3, 3),
                      padding='same', activation='relu')(conv)
        conv = Conv2D(128, kernel_size=(3, 3),
                      padding='same', activation='relu')(conv)
        conv = Conv2D(128, kernel_size=(3, 3),
                      padding='same', activation='relu')(conv)
        conv = Conv2D(256, kernel_size=(3, 3),
                      padding='same', activation='relu')(conv)

        inter = conv

        outputs = []
        for i in range(1):
            hourglass = self.__build_hourglass(inter, config.options.stacks)
            
            conv = Conv2D(256, kernel_size=(3, 3), padding='same',
                          activation='relu')(hourglass)
            conv = conv_x = Conv2D(256, kernel_size=(1, 1),
                          padding='same', activation='relu')(conv)

            conv = Conv2D(1, kernel_size=(1, 1), padding='same')(conv)
            padded = ZeroPadding2D(padding=((1, 0), (0, 0)))(conv)

            outputs.append(padded)

            if i < 3:
                # Residual link across hourglasses
                inter = Add()([inter, conv_x, Conv2D(
                    256, kernel_size=(1, 1), padding='same')(conv)])

        return Model(input=input_initial, outputs=outputs[-1])

    def __build_hourglass(self, input, stacks, size=256):
        upper_1 = Conv2D(size, kernel_size=(3, 3),
                         padding='same', activation='relu')(input)
        pool = MaxPooling2D(pool_size=(2, 2))(input)
        lower_1 = Conv2D(size, kernel_size=(3, 3),
                         padding='same', activation='relu')(pool)
        # Recursive hourglass
        if stacks > 1:
            lower_2 = self.__build_hourglass(lower_1, stacks - 1, size)
        else:
            lower_2 = Conv2D(size, kernel_size=(3, 3),
                             padding='same', activation='relu')(lower_1)
        lower_3 = Conv2D(size, kernel_size=(3, 3),
                         padding='same', activation='relu')(lower_2)
        upper_2 = UpSampling2D(size=(2, 2))(lower_3)
        return Add()([upper_1, upper_2])
