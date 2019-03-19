"""
Model of a song to train with.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import sys
import glob
import h5py


from unmix.source.data.track import Track
from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError

class Song(object):

    PREFIX_INSTRUMENTAL = "instrumental_"
    PREFIX_VOCALS = "vocals_"

    def __init__(self, folder):
        vocals_data = ""
        for file in glob.iglob(os.path.join(folder, "%s*.h5" % Song.PREFIX_VOCALS)):
            vocals_data = h5py.File(file, 'r')
            break
        instrumental_data = ""
        for file in glob.iglob(os.path.join(folder, "%s*.h5" % Song.PREFIX_INSTRUMENTAL)):
            instrumental_data = h5py.File(file, 'r')
            break
        self.fft_window = vocals_data['fft_window'].value
        self.sample_rate = vocals_data['sample_rate'].value
        self.collection = vocals_data['collection'].value
        self.song_name = vocals_data['song'].value
        self.vocals = Track("vocals", vocals_data)
        self.instrumental = Track("instrumental", instrumental_data)
        self.mix = Track("mix").mix(self.vocals, self.instrumental)
