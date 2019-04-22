#!/usr/bin/env python3
# coding: utf8

"""
Keras model for training using LeakyReLU activation layers.
"""

from keras.models import *
from keras.layers import *

from unmix.source.configuration import Configuration
from unmix.source.models.basemodel import BaseModel


class LeakyReluModel(BaseModel):
    name = 'LargeLeakyReLU'

    def build(self, config):
        alpha1 = config.alpha1
        alpha2 = config.alpha2
        dropout_rate = config.dropout_rate
        channels = 1
        batch_size = Configuration.get("training.batch_size")

        mashup = Input(batch_shape=(batch_size, 769, 64, 1), name='input')
        padding = ZeroPadding2D(((3, 0), (0, 0)))(mashup)
        dropout = Dropout(rate=dropout_rate)(padding)

        conv_a = Conv2D(filters=512, kernel_size=3, padding='same')(dropout)
        conv_a = LeakyReLU(alpha=alpha1)(conv_a)
        conv_a = Dropout(rate=dropout_rate)(conv_a)

        conv = Conv2D(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False)(conv_a)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = BatchNormalization()(conv)

        conv_b = Conv2D(filters=128, kernel_size=3, padding='same')(conv)
        conv_b = LeakyReLU(alpha=alpha1)(conv_b)
        conv_b = Dropout(rate=dropout_rate)(conv_b)

        conv = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', use_bias=False)(conv_b)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = BatchNormalization()(conv)

        conv = Conv2D(filters=64, kernel_size=3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = BatchNormalization()(conv)
        conv = UpSampling2D(size=(2, 2))(conv)

        conv = Concatenate()([conv, conv_b])

        conv = Conv2D(filters=32, kernel_size=3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = BatchNormalization()(conv)
        conv = UpSampling2D(size=(2, 2))(conv)

        conv = Concatenate()([conv, conv_a])

        conv = Conv2D(filters=64, kernel_size=3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = Conv2D(filters=64, kernel_size=3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = Conv2D(filters=32, kernel_size=3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate=dropout_rate)(conv)

        conv = Conv2D(filters=channels, kernel_size=3, padding='same')(conv)
        conv = Cropping2D(((3, 0), (0, 0)))(conv)

        return Model(inputs=mashup, outputs=conv)
