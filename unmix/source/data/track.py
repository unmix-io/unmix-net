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
from unmix.source.exceptions.configurationerror import ConfigurationError


class Track(object):

    def __init__(self, track_type, data=None):
        self.type = track_type
        if data:
            self.real_stereo = data['real_stereo'].value
            self.channels = []
            if self.real_stereo:
                self.channels.append(data['spectrogram_left'].value)
                self.channels.append(data['spectrogram_right'].value)
            else:
                self.channels.append(data['spectrogram_mono'].value)

    def mix(self, *tracks):
        self.channels = tracks[0].channels
        for track in tracks[1:]:
            for i in range(len(self.channels)):
                self.channels[i] += track.channels[i]
        return self