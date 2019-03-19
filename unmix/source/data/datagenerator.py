"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import keras
import numpy as np

from unmix.source.configuration import Configuration

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, collection):
        'Initialization'
        self.batch_size = Configuration.get("training.batch_size")
        self.collection = collection
        self.shuffle = Configuration.get("training.epoch.shuffle")
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.collection) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        temp_collection = [self.collection[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(temp_collection)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.collection))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_collection):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(temp_collection):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y)