#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using a U-Net architecture.
"""

from keras.models import *
from keras.layers import *

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel

# Base implementation from: https://www.kaggle.com/cjansen/u-net-in-keras


class UNetModel(BaseModel):
    name = 'U-Net'

    def build(self, config):
        transformation = Configuration.get('transformation.options', False)
        concat_axis = 3

        base_filter_count = 32
        input_shape = (769, transformation.size, 1)
        input = Input(input_shape)

        conv1 = Conv2D(base_filter_count, (3, 3), padding="same", name="conv1_1",
                       activation="relu", data_format="channels_last")(input)
        conv1 = Conv2D(base_filter_count, (3, 3), padding="same", activation="relu",
                       data_format="channels_last")(conv1)
        pool1 = MaxPooling2D(pool_size=(
            2, 2), data_format="channels_last")(conv1)
        conv2 = Conv2D(base_filter_count * 2, (3, 3), padding="same", activation="relu",
                       data_format="channels_last")(pool1)
        conv2 = Conv2D(base_filter_count * 2, (3, 3), padding="same", activation="relu",
                       data_format="channels_last")(conv2)
        pool2 = MaxPooling2D(pool_size=(
            2, 2), data_format="channels_last")(conv2)

        conv3 = Conv2D(base_filter_count * 4, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(pool2)
        conv3 = Conv2D(base_filter_count * 4, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(conv3)
        pool3 = MaxPooling2D(pool_size=(
            2, 2), data_format="channels_last")(conv3)

        conv4 = Conv2D(base_filter_count * 8, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(pool3)
        conv4 = Conv2D(base_filter_count * 8, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(conv4)
        pool4 = MaxPooling2D(pool_size=(
            2, 2), data_format="channels_last")(conv4)

        conv5 = Conv2D(base_filter_count * 16, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(pool4)
        conv5 = Conv2D(base_filter_count * 16, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(conv5)

        up_conv5 = UpSampling2D(
            size=(2, 2), data_format="channels_last")(conv5)
        ch, cw = self.__crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(
            cropping=(ch, cw), data_format="channels_last")(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(base_filter_count * 8, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(up6)
        conv6 = Conv2D(base_filter_count * 8, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(conv6)

        up_conv6 = UpSampling2D(
            size=(2, 2), data_format="channels_last")(conv6)
        ch, cw = self.__crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(
            cropping=(ch, cw), data_format="channels_last")(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(base_filter_count * 4, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(up7)
        conv7 = Conv2D(base_filter_count * 4, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(conv7)

        up_conv7 = UpSampling2D(
            size=(2, 2), data_format="channels_last")(conv7)
        ch, cw = self.__crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(
            cropping=(ch, cw), data_format="channels_last")(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(base_filter_count * 2, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(up8)
        conv8 = Conv2D(base_filter_count * 2, (3, 3), padding="same", activation="relu",
                       data_format="channels_last")(conv8)

        up_conv8 = UpSampling2D(
            size=(2, 2), data_format="channels_last")(conv8)
        ch, cw = self.__crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(
            cropping=(ch, cw), data_format="channels_last")(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(base_filter_count, (3, 3), padding="same",
                       activation="relu", data_format="channels_last")(up9)
        conv9 = Conv2D(base_filter_count, (3, 3), padding="same", activation="relu",
                       data_format="channels_last")(conv9)


        dense1 = Dense(1, activation='relu')(conv9)
        padding = ZeroPadding2D(((1, 0), (0, 0)))(dense1)

        #flatten = Flatten()(conv9)
        #dense1 = Dense(64, activation='relu')(flatten)
        #bn = BatchNormalization()(dense1)
        #dense2 = Dense(769 * transformation.step, activation='sigmoid')(bn)
        #output = Reshape((769, transformation.step, 1))(dense2)

        return Model(input=input, output=padding)

    def __crop_shape(self, target, refer):
        # 3rd dimension (width)
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = cw // 2, cw // 2 + 1
        else:
            cw1, cw2 = cw // 2, cw // 2
        # 2nd dimension (height)
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = ch// 2, ch // 2 + 1
        else:
            ch1, ch2 = ch // 2, ch // 2

        return (ch1, ch2), (cw1, cw2)
