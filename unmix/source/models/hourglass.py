#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using LeakyReLU activation layers.
"""

from keras.models import *
from keras.layers import *

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel

# Base implementation from: https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras


class HourglassModel(BaseModel):
    name = 'Hourglass'

    def build(self, config):
        transformation = Configuration.get('transformation.options', optional=False)

        input_shape = (769, transformation.size, 1)
        input = Input(input_shape)

        front_features = self.__front_module(
            input, config.options.channels)

        head_next_stage = front_features

        outputs = []
        for i in range(config.options.stacks):
            head_next_stage, head_to_loss = self.__module(
                head_next_stage, config.options.classes, config.options.channels, i)
            outputs.append(head_to_loss)

        return Model(input=input, outputs=outputs[-1])

    def __front_module(self, input, channels):
        # front module, input to 1/4 resolution
        # 1 7x7 conv + maxpooling
        # 3 residual block

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1x1')(
            input)
        x = BatchNormalization()(x)

        x = self.__bottleneck(x, channels // 2, 'front_residualx1')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = self.__bottleneck(x, channels // 2, 'front_residualx2')
        x = self.__bottleneck(x, channels, 'front_residualx3')

        return x

    def __module(self, bottom, classes, channels, hgid):
        # create left features , f1, f2, f4, and f8
        left_features = self.__left_half_blocks(
            bottom, hgid, channels)

        # create right features, connect with left features
        rf1 = self.__right_half_blocks(
            left_features, hgid, channels)

        # add 1x1 conv with two heads, head_next_stage is sent to next stage
        # head_parts is used for intermediate supervision
        head_next_stage, head_parts = self.__heads(
            bottom, rf1, classes, hgid, channels)

        return head_next_stage, head_parts

    def __bottleneck(self, bottom, out_channels, block_name):
        # skip layer
        if K.int_shape(bottom)[-1] == out_channels:
            _skip = bottom
        else:
            _skip = Conv2D(out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                           name=block_name + 'skip')(bottom)

        # residual: 3 conv blocks,  [out_channels/2  -> out_channels/2 -> out_channels]
        x = Conv2D(out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                   name=block_name + '_conv_1x1x1')(bottom)
        x = BatchNormalization()(x)
        x = Conv2D(out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                   name=block_name + '_conv_3x3x2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                   name=block_name + '_conv_1x1x3')(x)
        x = BatchNormalization()(x)
        x = Add(name=block_name + '_residual')([_skip, x])

        return x

    def __left_half_blocks(self, bottom, hglayer, channels):
        """
        Creates left half blocks for hourglass module.
        f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution
        """

        hgname = 'hg' + str(hglayer)

        f1 = self.__bottleneck(bottom, channels, hgname + '_l1')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

        f2 = self.__bottleneck(x, channels, hgname + '_l2')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

        f4 = self.__bottleneck(x, channels, hgname + '_l4')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

        f8 = self.__bottleneck(x, channels, hgname + '_l8')

        return (f1, f2, f4, f8)

    def __connect(self, left, right, name, channels):
        """
        :param left: connect left feature to right feature
        :param name: layer name
        """

        xleft = self.__bottleneck(left, channels, name + '_connect')
        xright = UpSampling2D()(right)
        add = Add()([xleft, xright])
        out = self.__bottleneck(add, channels, name + '_connect_conv')
        return out

    def __bottom_layer(self, lf8, hgid, channels):
        """
        Builds blocks of the lowest resolution.
        """

        lf8_connect = self.__bottleneck(lf8, channels, str(hgid) + "_lf8")

        x = self.__bottleneck(lf8, channels, str(hgid) + "_lf8x1")
        x = self.__bottleneck(x, channels, str(hgid) + "_lf8x2")
        x = self.__bottleneck(x, channels, str(hgid) + "_lf8x3")

        rf8 = Add()([x, lf8_connect])

        return rf8

    def __right_half_blocks(self, left_features, hglayer, channels):
        lf1, lf2, lf4, lf8 = left_features

        rf8 = self.__bottom_layer(lf8, hglayer, channels)

        rf4 = self.__connect(lf4, rf8, 'hg' +
                             str(hglayer) + '_rf4', channels)

        rf2 = self.__connect(lf2, rf4, 'hg' +
                             str(hglayer) + '_rf2', channels)

        rf1 = self.__connect(lf1, rf2, 'hg' +
                             str(hglayer) + '_rf1', channels)

        return rf1

    def __heads(self, prelayer_features, rf1, classes, hgid, channels):
        # two head, one head to next stage, one head to intermediate features
        head = Conv2D(channels, kernel_size=(1, 1), activation='relu', padding='same', name=str(hgid) + '_conv_1x1x1')(
            rf1)
        head = BatchNormalization()(head)

        # for head as intermediate supervision, use 'linear' as activation.
        head_parts = Conv2D(classes, kernel_size=(1, 1), activation='linear', padding='same',
                            name=str(hgid) + '_conv_1x1_parts')(head)

        # use linear activation
        head = Conv2D(channels, kernel_size=(1, 1), activation='linear', padding='same',
                      name=str(hgid) + '_conv_1x1x2')(head)
        head_m = Conv2D(channels, kernel_size=(1, 1), activation='linear', padding='same',
                        name=str(hgid) + '_conv_1x1x3')(head_parts)

        head_next_stage = Add()([head, head_m, prelayer_features])
        return head_next_stage, head_parts
