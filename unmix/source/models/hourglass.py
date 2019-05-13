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
    name = 'Hourglass'

    def build(self, config):
        transformation = Configuration.get('transformation.options', optional=False)
        channels = 2 if Configuration.get('collection.stereo', default=False) else 1

        input_shape = (769, transformation.size, channels)
        input_initial = Input(input_shape)
        input = Cropping2D(cropping=((1, 0), (0, 0)))(input_initial)

        conv = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input) #conv2d_1
        if config.options.initial_convolutions.batchnormalization: conv = BatchNormalization()(conv)
        conv = Dropout(config.options.initial_convolutions.dropoutfactor)(conv)

        conv = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv) #conv2d_2
        if config.options.initial_convolutions.batchnormalization: conv = BatchNormalization()(conv)
        conv = Dropout(config.options.initial_convolutions.dropoutfactor)(conv)

        conv = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv) #conv2d_3
        if config.options.initial_convolutions.batchnormalization: conv = BatchNormalization()(conv)
        conv = Dropout(config.options.initial_convolutions.dropoutfactor)(conv)

        conv = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv) #conv2d_4
        if config.options.initial_convolutions.batchnormalization: conv = BatchNormalization()(conv)
        conv = Dropout(config.options.initial_convolutions.dropoutfactor)(conv)

        conv = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv) #conv2d_5
        if config.options.initial_convolutions.batchnormalization: conv = BatchNormalization()(conv)
        conv = Dropout(config.options.initial_convolutions.dropoutfactor)(conv)

        inter = conv

        outputs = []
        for i in range(1):
            hourglass = self.__build_hourglass(inter, config.options.stacks, config.options.hg_module.filters, config.options.hg_module.batchnormalization, config.options.hg_module.dropoutfactor)
            
            conv = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(hourglass) #conv2d_19
            if config.options.hg_module_end.batchnormalization: conv = BatchNormalization()(conv)
            conv = Dropout(config.options.hg_module_end.dropoutfactor)(conv)

            conv = conv_x = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu')(conv) #conv2d_20
            conv = Dropout(config.options.hg_module_end.dropoutfactor)(conv)

            conv = Conv2D(1, kernel_size=(1, 1), padding='same')(conv) #conv2d_21

            padded = ZeroPadding2D(padding=((1, 0), (0, 0)))(conv) #zero_padding2d_1

            outputs.append(padded)

            if i < 3:
                # Residual link across hourglasses
                inter = Add()([inter, conv_x, Conv2D(256, kernel_size=(1, 1), padding='same')(conv)]) #conv2d_22

        return Model(input=input_initial, outputs=outputs[-1])

    def __build_hourglass(self, input, stacks, filters, batchnormalization, dropoutfactor):
        upper_1 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(input) #conv2d_6, conv2d_8, conv2d_10, conv2d_12
        if batchnormalization: upper_1 = BatchNormalization()(upper_1)
        upper_1 = Dropout(dropoutfactor)(upper_1)
        pool = MaxPooling2D(pool_size=(2, 2))(input) #max_pooling2d_1, #max_pooling2d_2, max_pooling2d_3, max_pooling2d_4

        lower_1 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(pool) #conv2d_7, conv2d_9, conv2d_11, conv2d_13
        if batchnormalization: lower_1 = BatchNormalization()(lower_1)
        lower_1 = Dropout(dropoutfactor)(lower_1)

        # Recursive hourglass
        if stacks > 1:
            lower_2 = self.__build_hourglass(lower_1, stacks - 1, filters, batchnormalization, dropoutfactor)
        else:
            lower_2 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(lower_1) #conv2d_14
            if batchnormalization: lower_2 = BatchNormalization()(lower_2)
            lower_2 = Dropout(dropoutfactor)(lower_2)

        lower_3 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(lower_2) #conv2d15, conv2d16, conv2d_17, conv2d18
        if batchnormalization: lower_3 = BatchNormalization()(lower_3)
        lower_3 = Dropout(dropoutfactor)(lower_3)
        upper_2 = UpSampling2D(size=(2, 2))(lower_3) #upsampling2d_1, upsampling2d_2, upsampling2d_3, upsampling2d_4

        return Add()([upper_1, upper_2])
