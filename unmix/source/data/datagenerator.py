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
from unmix.source.logging.logger import Logger
from unmix.source.metrics.accuracy import Accuracy
from unmix.source.helpers import memorymonitor


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, name, engine, collection, transformer, run_tests=False):
        self.name = name
        self.collection = collection
        self.transformer = transformer
        self.batch_size = Configuration.get('training.batch_size')
        self.epoch_shuffle = Configuration.get('training.epoch.shuffle')
        self.engine = engine
        self.run_tests = run_tests
        self.accuracy = Accuracy(self.engine)
        self.count = 0
        self.on_epoch_end()


    def generate_index(self):
        self.index = np.array([])
        for file in self.collection:
            try:
                song = Song(file)
                items = [BatchItem(song, i) for i in range(self.transformer.calculate_items(song.width))]
                if self.transformer.shuffle:
                    np.random.shuffle(items)
                self.index = np.append(self.index, items)
            except Exception as e:
                if self.count == 0:
                    Logger.warn("Skip file while generating index: %s" % str(e.args))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.index) // self.batch_size

    def __getitem__(self, i):
        'Generate one batch of data'
        subset = self.index[i*self.batch_size:(i+1)*self.batch_size]
        X, y = self.__data_generation(subset)
        # clear already used items from index to free up memory after a song has been used
        self.index[i*self.batch_size:(i+1)*self.batch_size] = None
        return X, y

    def on_epoch_end(self):
        'Updates index after each epoch'
        Logger.debug("%s epoch %d ended." % (self.name, self.count))
        self.generate_index()
        if self.epoch_shuffle:
            np.random.shuffle(self.index)
        if self.engine.test_songs and self.run_tests:
            self.accuracy.evaluate(self.count)
        self.count += 1

    def __data_generation(self, subset):
        'Generates data containing batch_size samples'
        X = []
        Y = []

        for item in subset:
            mix, vocals = item.load()
            x, y = self.transformer.run(item.name, mix, vocals, item.index)
            X.append(x)
            Y.append(y)

        return np.array(X), np.array(Y)
