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
            hourglass, padding_added_hor, cropping_added_ver = self.__build_hourglass(inter, config.options.stacks)
            cropping = ((0, 0), (0, 0))
            if padding_added_hor and cropping_added_ver:
                cropping = ((1, 0), (0, 1))
            elif padding_added_hor and not cropping_added_ver:
                cropping = ((1, 0), (0, 0))
            elif not padding_added_hor and cropping_added_ver:
                cropping = ((0, 0), (0, 1))

            conv = Cropping2D(cropping=cropping)(hourglass)
            conv = Conv2D(128, kernel_size=(3, 3), padding='same',
                          activation='relu')(conv)
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
        padding_added_horizontal = False
        padding_added_horizontal_upper = False
        padding_added_vertical = False
        padding_added_vertical_upper = False
        if not upper_1.shape[1] % 2 == 0:
            upper_1 = ZeroPadding2D(padding=((1, 0), (0, 0)))(upper_1)
            input = ZeroPadding2D(padding=((1, 0), (0, 0)))(input)
            padding_added_horizontal = True
        if not upper_1.shape[2] % 2 == 0:
            upper_1 = ZeroPadding2D(padding=((0, 0), (0, 1)))(upper_1)
            input = ZeroPadding2D(padding=((0, 0), (0, 1)))(input)
            padding_added_vertical = True
        # Lower branch
        pool = MaxPooling2D(pool_size=(2, 2))(input)
        lower_1 = Conv2D(size, kernel_size=(3, 3),
                         padding='same', activation='relu')(pool)
        # Recursive hourglass
        if stacks > 1:
            lower_2, padding_added_horizontal_upper, padding_added_vertical_upper = self.__build_hourglass(lower_1, stacks - 1, size)
        else:
            lower_2 = Conv2D(size, kernel_size=(3, 3),
                             padding='same', activation='relu')(pool)
        lower_3 = Conv2D(size, kernel_size=(3, 3),
                         padding='same', activation='relu')(lower_2)

        #upper_2 = UpSampling2D(size=tuple(map(lambda x, y:
        #    x.value // y.value, upper_1.shape.dims[1:3], lower_3.shape.dims[1:3])))(lower_3)
        cropping = ((0, 0), (0, 0))
        if padding_added_horizontal_upper and padding_added_vertical_upper:
            cropping = ((1, 0), (0, 1))
        elif padding_added_horizontal_upper and not padding_added_vertical_upper:
            cropping = ((1, 0), (0, 0))
        elif not padding_added_horizontal_upper and padding_added_vertical_upper:
            cropping = ((0, 0), (0, 1))

        upper_2 = Cropping2D(cropping=cropping)(lower_3)
        upper_2 = UpSampling2D(size=(2, 2))(upper_2)
        return Add()([upper_1, upper_2]), padding_added_horizontal, padding_added_vertical
