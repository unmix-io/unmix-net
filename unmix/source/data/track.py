"""
Model of a track of a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import sys
import glob
import h5py

from unmix.source.configuration import Configuration
from unmix.source.exceptions.dataerror import DataError


class Track(object):

    def __init__(self, track_type, height, width, depth, file=None, initilize=False):
        self.type = track_type
        self.initialized = False
        self.file = file
        self.height = height
        self.width = width
        self.depth = depth
        if initilize:
            self.load

    def load(self, data=None):
        if not data:            
            data = h5py.File(self.file, 'r')
        if not data:
            raise DataError('?', 'missing data to load')
        self.real_stereo = data['real_stereo'].value
        self.channels = []
        if self.real_stereo:
            self.channels.append(data['spectrogram_left'].value)
            self.channels.append(data['spectrogram_right'].value)
        else:
            self.channels.append(data['spectrogram_mono'].value)
        self.initialized = True

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
        return self
