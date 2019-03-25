"""
Keras model for training using the unmix.io architecture.
"""

from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from unmix.source.configuration import Configuration


name = 'unmix'

def generate(alpha1, alpha2, rate, channels=2):
    batch_size = Configuration.get("training.batch_size")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(batch_size, None, None, None, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(batch_size, None, 20, 20, 2)))
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