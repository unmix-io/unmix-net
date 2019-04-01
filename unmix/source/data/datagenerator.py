#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import gc
import keras
import numpy as np

from unmix.source.configuration import Configuration
from unmix.source.data.batchitem import BatchItem
from unmix.source.data.song import Song


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, collection, choppers, normalizer):
        'Initialization'
        self.collection = collection
        self.choppers = choppers
        self.normalizer = normalizer
        self.batch_size = Configuration.get("training.batch_size")
        self.shuffle = Configuration.get("training.epoch.shuffle")
        self.shuffle_in_song = Configuration.get("training.shuffle_chops_in_song")

        self.on_epoch_end()

    def generate_index(self):
        self.index = np.array([])
        for file in self.collection:
            song = Song(file)
            if self.choppers:
                for chopper in self.choppers:
                    batchitems = [BatchItem(song, i) 
                        for i in range(chopper.calculate_chops(song.width))]
                    if(self.shuffle_in_song):
                        np.random.shuffle(batchitems)
                    self.index = np.append(self.index, batchitems)
            else:
                self.index = np.append(self.index, BatchItem(song, 0))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.index) / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        subset = self.index[i*self.batch_size:(i+1)*self.batch_size]
        X, y = self.__data_generation(subset)
        self.index[i*self.batch_size:(i+1)*self.batch_size] = None
        return X, y

    def on_epoch_end(self):
        'Updates index after each epoch'
        self.generate_index()
        if self.shuffle:
            np.random.shuffle(self.index)

    def __data_generation(self, subset):
        'Generates data containing batch_size samples'

        X = []
        y = []

        for item in subset:
            mix, vocals = item.song.load(self.choppers, item.offset)
            X.append(self.normalizer.normalize(mix))
            y.append(self.normalizer.normalize(vocals))

        return np.array(X), np.array(y)
