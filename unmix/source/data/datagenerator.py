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
from unmix.source.helpers import console
from unmix.source.helpers import audiohandler


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, collection, chopper, normalizer):
        'Initialization'
        self.collection = collection
        self.chopper = chopper
        self.normalizer = normalizer
        self.batch_size = Configuration.get('training.batch_size')
        self.shuffle = Configuration.get('training.epoch.shuffle')
        self.inside_shuffle = Configuration.get('training.inside_shuffle')
        self.on_epoch_end()

    def generate_index(self):
        self.index = np.array([])
        for file in self.collection:
            try:
                song = Song(file)
                if self.chopper:
                    batchitems = [BatchItem(song, i)
                                  for i in range(self.chopper.calculate_chops(song.width))]
                    if(self.inside_shuffle):
                        np.random.shuffle(batchitems)
                    self.index = np.append(self.index, batchitems)
                else:
                    self.index = np.append(self.index, BatchItem(song, 0))
            except Exception as e:
                console.warn(
                    "Skip file while generating index: %s", str(e.args))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.index) / self.batch_size)

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
            mix, vocals = item.load(self.chopper)
            if Configuration.get('training.save_chops'):
                audiohandler.spectrogram_to_audio(
                    '%s-%s_vocals.wav' % (item.song.name, item.offset), vocals)
                audiohandler.spectrogram_to_audio(
                    '%s-%s_mix.wav' % (item.song.name, item.offset), mix)

            if self.mask:
                mix_cut = self.cutter.cut(mix)
                vocal_cut = self.cutter.cut(vocals)

                mask = masker.mask(mix_cut, vocal_cut)

                X.append(self.normalizer.normalize(mix)[0])
                y.append(self.normalizer.normalize(vocals)[0])
            else:
                X.append(self.normalizer.normalize(mix)[0])
                y.append(self.normalizer.normalize(vocals)[0])

        return np.array(X), np.array(y)
