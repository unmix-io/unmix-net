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


class DataLoader(object):

    VALIDATION_MODE_SHUFFLE = "shuffle"

    def load(self):
        path = Configuration.get_path("collection.folder")
        songs = [Song(os.path.dirname(file)) for file in glob.iglob(os.path.join(path, '**', '%s*.h5' % Song.PREFIX_VOCALS), recursive=True)]

        validation_ratio = Configuration.get("collection.validation.ratio")
        validation_mode = Configuration.get("collection.validation.mode")
        if validation_mode == DataLoader.VALIDATION_MODE_SHUFFLE:
            self.validation_songs = random.sample(
                songs, int(validation_ratio * len(songs)))
            self.training_songs = set(songs) - set(self.validation_songs)

        console.debug("Found %d songs for traing and %d songs for validation."
                      % (len(self.training_songs), len(self.validation_songs)))

        return self.training_songs, self.validation_songs
