"""
Model of a track of a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import sys
import glob
import h5py
import numpy as np

from unmix.source.helpers import reducer
from unmix.source.configuration import Configuration
from unmix.source.exceptions.dataerror import DataError


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
            self.channels = []
            if self.stereo:
                self.channels.append(data['spectrogram_left'].value)
                self.channels.append(data['spectrogram_right'].value)
            else:
                self.channels.append(data['spectrogram'].value)
            self.initialized = True
            return self
        except Exception as e:            
            raise DataError(self.file, str(e))

    def mix(self, *tracks):
        first = tracks[0]
        if not first.initialized:
            first.load()
        self.channels = first.channels
        for track in tracks[1:]:
            if not track.initialized:
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
        self.chops = []
        for channel in self.channels:
            input = channel
            for chopper in choppers:
                input = chopper.chop(input)
            self.chops.append(input)
            #self.chops.append(reducer.lflatter(input, len(choppers)))
        self.chopped = True