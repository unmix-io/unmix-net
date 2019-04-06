#!/usr/bin/env python3
# coding: utf8

"""
Model of a song to train with.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


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
        data_vocals = h5py.File(vocals_file, 'r')
        data_instrumental = h5py.File(instrumental_file, 'r')
        self.folder = folder
        self.height = data_vocals['height'][()]
        self.width = min(int(data_vocals['width'][()]), int(data_instrumental['width'][()]))
        self.depth = data_vocals['depth'][()]
        self.fft_window = data_vocals['fft_window'][()]
        self.sample_rate = data_vocals['sample_rate'][()]
        self.collection = data_vocals['collection'][()]
        self.name = data_vocals['song'][()]
        self.vocals = Track("vocals", self.height, self.width,
                                self.depth, vocals_file)
        self.instrumental = Track("instrumental", self.height, self.width, 
                                self.depth, instrumental_file)
        self.mix = Track("mix", self.height, self.width, self.depth)

    def load(self):
        if not self.mix.initialized:
            try:
                # After this step all tracks are initialized
                self.mix.mix(self.vocals, self.instrumental)
            except Exception as e:
                raise DataError(self.folder, str(e))
            finally:
                self.clean_up()
        return self.mix.channels, self.vocals.channels

    def clean_up(self):
        self.instrumental = []
