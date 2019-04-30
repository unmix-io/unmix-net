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
    batch_normalization_init_layers = False
    batch_normalization_hg_layers = False
    batch_normalization_hg_layers_end = False
    dropoutfact_init_layers = 0
    dropoutfact_hg_layers = 0
    dropoutfact_hg_layers_end = 0

    def build(self, config):
        transformation = Configuration.get('transformation.options', optional=False)


        input_shape = (769, transformation.size, 1)
        input_initial = Input(input_shape)
        input = Cropping2D(cropping=((1, 0), (0, 0)))(input_initial)
        conv = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input) #conv2d_1
        if HourglassModel.batch_normalization_init_layers: conv = BatchNormalization()(conv)

        conv = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv) #conv2d_2
        if HourglassModel.batch_normalization_init_layers: conv = BatchNormalization()(conv)
        conv = Dropout(HourglassModel.dropoutfact_init_layers)(conv)

        conv = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv) #conv2d_3
        if HourglassModel.batch_normalization_init_layers: conv = BatchNormalization()(conv)
        conv = Dropout(HourglassModel.dropoutfact_init_layers)(conv)

        conv = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv) #conv2d_4
        if HourglassModel.batch_normalization_init_layers: conv = BatchNormalization()(conv)
        conv = Dropout(HourglassModel.dropoutfact_init_layers)(conv)

        conv = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv) #conv2d_5
        if HourglassModel.batch_normalization_init_layers: conv = BatchNormalization()(conv)
        conv = Dropout(HourglassModel.dropoutfact_init_layers)(conv)

        inter = conv

        outputs = []
        for i in range(1):
            hourglass = self.__build_hourglass(inter, config.options.stacks)
            
            conv = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(hourglass) #conv2d_19
            if HourglassModel.batch_normalization_hg_layers_end: conv = BatchNormalization()(conv)
            conv = Dropout(HourglassModel.dropoutfact_hg_layers_end)(conv)

            conv = conv_x = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu')(conv) #conv2d_20
            conv = Dropout(HourglassModel.dropoutfact_hg_layers_end)(conv)

            conv = Conv2D(1, kernel_size=(1, 1), padding='same')(conv) #conv2d_21

            padded = ZeroPadding2D(padding=((1, 0), (0, 0)))(conv) #zero_padding2d_1

            outputs.append(padded)

            if i < 3:
                # Residual link across hourglasses
                inter = Add()([inter, conv_x, Conv2D(256, kernel_size=(1, 1), padding='same')(conv)]) #conv2d_22

        return Model(input=input_initial, outputs=outputs[-1])

    def __build_hourglass(self, input, stacks, size=256):
        upper_1 = Conv2D(size, kernel_size=(3, 3), padding='same', activation='relu')(input) #conv2d_6, conv2d_8, conv2d_10, conv2d_12
        if HourglassModel.batch_normalization_hg_layers: upper_1 = BatchNormalization()(upper_1)
        upper_1 = Dropout(HourglassModel.dropoutfact_hg_layers)(upper_1)
        pool = MaxPooling2D(pool_size=(2, 2))(input) #max_pooling2d_1, #max_pooling2d_2, max_pooling2d_3, max_pooling2d_4

        lower_1 = Conv2D(size, kernel_size=(3, 3), padding='same', activation='relu')(pool) #conv2d_7, conv2d_9, conv2d_11, conv2d_13
        if HourglassModel.batch_normalization_hg_layers: lower_1 = BatchNormalization()(lower_1)
        lower_1 = Dropout(HourglassModel.dropoutfact_hg_layers)(lower_1)

        # Recursive hourglass
        if stacks > 1:
            lower_2 = self.__build_hourglass(lower_1, stacks - 1, size)
        else:
            lower_2 = Conv2D(size, kernel_size=(3, 3), padding='same', activation='relu')(lower_1) #conv2d_14
            if HourglassModel.batch_normalization_hg_layers: lower_2 = BatchNormalization(lower_2)
            lower_2 = Dropout(HourglassModel.dropoutfact_hg_layers)(lower_2)

        lower_3 = Conv2D(size, kernel_size=(3, 3), padding='same', activation='relu')(lower_2) #conv2d15, conv2d16, conv2d_17, conv2d18
        if HourglassModel.batch_normalization_hg_layers: lower_3 = BatchNormalization()(lower_3)
        lower_3 = Dropout(HourglassModel.dropoutfact_hg_layers)(lower_3)
        upper_2 = UpSampling2D(size=(2, 2))(lower_3) #upsampling2d_1, upsampling2d_2, upsampling2d_3, upsampling2d_4

        return Add()([upper_1, upper_2])
