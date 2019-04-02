#!/usr/bin/env python3
# coding: utf8

"""
Model of a song to train with.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import gc
import glob
import h5py
import os

from unmix.source.data.track import Track
from unmix.source.exceptions.dataerror import DataError


class Song(object):

    PREFIX_INSTRUMENTAL = "instrumental_"
    PREFIX_VOCALS = "vocals_"

    def __init__(self, folder):
        vocals_file = ""
        for file in glob.iglob(os.path.join(folder, "%s*.h5" % Song.PREFIX_VOCALS)):
            vocals_file = file
            break
        instrumental_file = ""
        for file in glob.iglob(os.path.join(folder, "%s*.h5" % Song.PREFIX_INSTRUMENTAL)):
            instrumental_file = file
            break
        if not (instrumental_file and vocals_file):
            raise DataError(folder, 'missing vocal or instrumental track')
        data = h5py.File(vocals_file, 'r')
        self.folder = folder
        self.height = data['height'].value
        self.width = data['width'].value
        self.depth = data['depth'].value
        self.fft_window = data['fft_window'].value
        self.sample_rate = data['sample_rate'].value
        self.collection = data['collection'].value
        self.name = data['song'].value
        self.vocals = Track("vocals", self.height, self.width, self.depth, vocals_file)
        self.instrumental = Track("instrumental", self.height, self.width, self.depth, instrumental_file)
        self.mix = Track("mix", self.height, self.width, self.depth)

    def load(self, choppers=[], offset=0):
        if not self.mix.initialized and not self.mix.chopped:
            self.mix.mix(self.vocals, self.instrumental) # After this step all tracks are initialized
        self.mix.chop(choppers)
        self.vocals.chop(choppers)
        self.clean_up(False)
        return self.mix.chops[offset], self.vocals.chops[offset]

    def clean_up(self, clean_chops):
        self.vocals.clean_up(clean_chops)
        self.mix.clean_up(clean_chops)
        if hasattr(self, 'instrumental') and self.instrumental:
            self.instrumental.clean_up(clean_chops)
            del self.instrumental
        self.instrumental = []
        gc.collect()