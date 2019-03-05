"""
Keras model for training using LeakyReLU activation layers.
"""

from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, BatchNormalization, UpSampling2D, Concatenate, LeakyReLU

name = 'LeakyReLU'

def generate(alpha1, alpha2, rate, channels=2):
    mashup = Input(shape=(None, None, channels), name='input')
    dropout = Dropout(rate)(mashup)

    conv_a = Conv2D(64, 3, padding='same')(dropout)
    conv_a = LeakyReLU(alpha=alpha1)(conv_a)
    conv_a = Dropout(rate)(conv_a)

    conv = Conv2D(64, 4, strides=2, padding='same', use_bias=False)(conv_a)
    conv = LeakyReLU(alpha=alpha1)(conv)
    conv = Dropout(rate)(conv)

    conv = BatchNormalization()(conv)

    conv_b = Conv2D(64, 3, padding='same')(conv)
    conv_b = LeakyReLU(alpha=alpha1)(conv_b)
    conv_b = Dropout(rate)(conv_b)

    conv = Conv2D(64, 4, strides=2, padding='same', use_bias=False)(conv_b)
    conv = LeakyReLU(alpha=alpha1)(conv)
    conv = Dropout(rate)(conv)

    conv = BatchNormalization()(conv)

    conv = Conv2D(128, 3, padding='same')(conv)
    conv = LeakyReLU(alpha=alpha1)(conv)
    conv = Dropout(rate)(conv)

    conv = Conv2D(128, 3, padding='same', use_bias=False)(conv)
    conv = LeakyReLU(alpha=alpha1)(conv)
    conv = Dropout(rate)(conv)

    conv = BatchNormalization()(conv)
    conv = UpSampling2D((2, 2))(conv)

    conv = Concatenate()([conv, conv_b])

    conv = Conv2D(64, 3, padding='same')(conv)
    conv = LeakyReLU(alpha=alpha1)(conv)
    conv = Dropout(rate)(conv)

    conv = Conv2D(64, 3, padding='same', use_bias=False)(conv)
    conv = LeakyReLU(alpha=alpha1)(conv)
    conv = Dropout(rate)(conv)

    conv = BatchNormalization()(conv)
    conv = UpSampling2D((2, 2))(conv)

    conv = Concatenate()([conv, conv_a])

    conv = Conv2D(64, 3, padding='same')(conv)
    conv = LeakyReLU(alpha=alpha2)(conv)
    conv = Dropout(rate)(conv)

    conv = Conv2D(64, 3, padding='same')(conv)
    conv = LeakyReLU(alpha=alpha2)(conv)
    conv = Dropout(rate)(conv)

    conv = Conv2D(32, 3, padding='same')(conv)
    conv = LeakyReLU(alpha=alpha2)(conv)
    conv = Dropout(rate)(conv)

    conv = Conv2D(channels, 3, padding='same')(conv)

    vocal = conv
    return Model(inputs=mashup, outputs=vocal)
