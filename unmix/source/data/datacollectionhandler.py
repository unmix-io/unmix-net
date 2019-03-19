"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import sys
import glob

from unmix.source.data.song import Song
from unmix.source.helpers import console
from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError


class DataCollectionHandler(object):

    def __init__(self):
        self.load()

    def load(self):
        base_path = Configuration.get_path("environment.data_folder")
        path = os.path.join(base_path, "**")
        song_names = []
        for file in glob.iglob(os.path.join(path, "%s*.h5" % Song.PREFIX_VOCALS), recursive=True):
            song_names.append(os.path.dirname(file))
        
        console.debug('Found %d songs' % len(song_names))
