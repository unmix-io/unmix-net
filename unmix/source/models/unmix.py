"""
Keras model for training using the unmix.io architecture.
"""

from keras.models import Sequential
from keras.layers import Input, Dropout, Conv2D, BatchNormalization, UpSampling2D, Concatenate, LeakyReLU, MaxPooling2D, Flatten, Dense

name = 'unmix'

def generate(alpha1, alpha2, rate, channels=2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model