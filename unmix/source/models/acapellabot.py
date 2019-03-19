"""
Keras model for training using the acapellabot architecture.
"""

from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, BatchNormalization, UpSampling2D, Concatenate, LeakyReLU

name = 'AcapellaBot'

def generate(channels=2):
    mashup = Input(shape=(None, None, channels), name='input')
    conv_a = Conv2D(64, 3, activation='relu', padding='same')(mashup)
    conv = Conv2D(64, 4, strides=2, activation='relu',
                  padding='same', use_bias=False)(conv_a)
    conv = BatchNormalization()(conv)

    conv_b = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 4, strides=2, activation='relu',
                  padding='same', use_bias=False)(conv_b)
    conv = BatchNormalization()(conv)

    conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(128, 3, activation='relu',
                  padding='same', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D((2, 2))(conv)

    conv = Concatenate()([conv, conv_b])
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 3, activation='relu',
                  padding='same', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D((2, 2))(conv)

    conv = Concatenate()([conv, conv_a])
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(channels, 3, activation='relu', padding='same')(conv)
    return Model(inputs=mashup, outputs=conv)
