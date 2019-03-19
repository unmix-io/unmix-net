"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os
import sys
import glob
import random

from unmix.source.data.song import Song
from unmix.source.helpers import console
from unmix.source.configuration import Configuration
from unmix.source.exceptions.configurationerror import ConfigurationError


class DataCollectionHandler(object):

    VALIDATION_MODE_SHUFFLE = "shuffle"

    def load(self):
        base_path = Configuration.get_path("environment.collection.folder")
        path = os.path.join(base_path, "**")
        songs = []
        for file in glob.iglob(os.path.join(path, "%s*.h5" % Song.PREFIX_VOCALS), recursive=True):
            songs.append(os.path.dirname(file))

        validation_ratio = Configuration.get_path(
            "environment.collection.validation.ratio")
        validation_mode = Configuration.get_path(
            "environment.collection.validation.mode")
        if validation_mode == DataCollectionHandler.VALIDATION_MODE_SHUFFLE:
            self.validation_songs = random.sample(
                songs, int(validation_ratio * len(songs)))
            self.training_songs = set(songs) - set(self.validation_songs)

        console.debug('Found %d songs for traing and %d songs for validation.'
                      % (len(self.training_songs), len(self.validation_songs)))

        return self.training_songs, self.validation_songs
