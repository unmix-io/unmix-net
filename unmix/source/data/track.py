#!/usr/bin/env python3
# coding: utf8

"""
Model of a track of a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import h5py
import numpy as np
from threading import Lock

from unmix.source.exceptions.dataerror import DataError

class Track(object):

    def __init__(self, track_type, height, width, depth, file=None):
        self.type = track_type
        self.initialized = False
        self.file = file
        self.height = height
        self.width = width
        self.depth = depth
        self.mutex = Lock()

    def load(self, data=None):
        self.mutex.acquire() # make sure only one thread loads the file
        try:
            if self.initialized:
                return self
            if not data:
                data = h5py.File(self.file, 'r')
            if not data:
                raise DataError('?', "missing data to load")
            self.stereo = data['stereo'].value
            if self.stereo:
                self.channels = np.array(
                    [self.to_complex(data['spectrogram_left'].value), self.to_complex(data['spectrogram_right'].value)])
            else:
                self.channels = np.array([self.to_complex(data['spectrogram'].value)])
            self.initialized = True
            return self
        except Exception as e:
            raise DataError(self.file, str(e))
        finally:
            self.mutex.release()

    def mix(self, *tracks):
        self.mutex.acquire()
        try:
            if self.initialized:
                return self
            first = tracks[0]
            first.load()
            self.channels = np.copy(first.channels)
            for track in tracks[1:]:
                track.load()
                for i in range(len(self.channels)):
                    self.channels[i] += track.channels[i]
            self.initialized = True
            return self
        finally:
            self.mutex.release()

    def to_complex(self, realimag):
        'Converts the real-imag array of the training values to complex values'
        real = realimag[:, :, 0]
        imag = realimag[:, :, 1]
        return real + imag * 1j