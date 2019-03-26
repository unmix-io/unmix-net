"""
Model of a track of a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import gc
import h5py
import numpy as np
from functools import reduce

from unmix.source.exceptions.dataerror import DataError
import unmix.source.helpers.reducer as reducer


class Track(object):

    def __init__(self, track_type, height, width, depth, file=None, initilize=False):
        self.type = track_type
        self.initialized = False
        self.chopped = False
        self.file = file
        self.height = height
        self.width = width
        self.depth = depth
        if initilize:
            self.load

    def load(self, data=None, force=False):
        if self.initialized and not force:
            return self
        try:
            if not data:            
                data = h5py.File(self.file, 'r')
            if not data:
                raise DataError('?', 'missing data to load')
            self.stereo = data['stereo'].value
            if self.stereo:
                self.channels = np.array([data['spectrogram_left'].value, data['spectrogram_right'].value])
            else:
                self.channels = np.array([data['spectrogram'].value])
            self.initialized = True
            return self
        except Exception as e:            
            raise DataError(self.file, str(e))

    def mix(self, force=False, *tracks):
        if self.initialized and not force:
            return self
        first = tracks[0]
        if not first.initialized:
            first.load()
        self.channels = first.channels
        for track in tracks[1:]:
            track.load()
            for i in range(len(self.channels)):
                self.channels[i] += track.channels[i]
        self.initialized = True
        return self

    def chop(self, choppers, force=False):
        if self.chopped and not force:
            return
        if not self.initialized:
            self.load()
        self.chops = np.array([reduce((lambda input, chopper: chopper.chop(input)), [channel]  + choppers if choppers else [])
                            for channel in self.channels])
        self.chops = reducer.rflatter(self.chops.transpose(1,2,3,0,4))
        self.chopped = True

    def clean_up(self, clean_chops):
        if hasattr(self, 'channels'):
            del self.channels
        self.initialized = False
        gc.collect()