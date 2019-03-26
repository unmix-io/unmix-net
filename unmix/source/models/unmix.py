"""
Keras model for training using the unmix.io architecture.
"""

from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.models import Sequential

from unmix.source.configuration import Configuration


name = 'unmix'

def generate(alpha1, alpha2, rate, channels=2):
    batch_size = Configuration.get("training.batch_size")
    
    input_shape = (769, 64, 2)

    model = Sequential()
    model.add(ZeroPadding2D(((1, 0), (0, 0)), input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    #model.add(Dropout(0.25))

    #model.add(UpSampling2D((2, 2)))
    model.add(UpSampling2D((2, 2)))
    model.add(Dense(2))
    model.add(Cropping2D(((1,0), (0,0))))

    return model