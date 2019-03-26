"""
Keras model for training using a dummy architecture.
"""

from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from unmix.source.configuration import Configuration


name = 'dummy'

def generate(alpha1, alpha2, rate, channels=2):
    batch_size = Configuration.get("training.batch_size")

    model = Sequential()
    model.add(Dense((2), input_shape=(769, 64, 2)))

    return model