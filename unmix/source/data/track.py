#!/usr/bin/env python3
# coding: utf8

"""
Model of a track of a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


from functools import reduce
import h5py
import numpy as np
from threading import Lock

from unmix.source.exceptions.dataerror import DataError
from unmix.source.helpers import converter


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
        self.mutex.acquire()  # make sure only one thread loads the file
        try:
            if self.initialized:
                return self
            if not data:
                data = h5py.File(self.file, 'r')
            if not data:
                raise DataError('?', "missing data to load")
            self.stereo = data['stereo'][()]
            if self.stereo:
                self.channels = np.array(
                    [converter.to_complex(data['spectrogram_left'][()][:,:self.width]), converter.to_complex(data['spectrogram_right'][()][:,:self.width])])
            else:
                self.channels = np.array(
                    [converter.to_complex(data['spectrogram'][()][:,:self.width])])
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
            self.channels = reduce((lambda x, y: x + y),
                                 map(lambda track: track.load().channels, tracks))
            self.initialized = True
            return self
        finally:
            self.mutex.release()