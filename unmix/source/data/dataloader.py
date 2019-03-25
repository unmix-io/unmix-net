"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import glob
import os
import random

from unmix.source.configuration import Configuration
from unmix.source.data.song import Song
from unmix.source.helpers import console


class DataLoader(object):

    VALIDATION_MODE_SHUFFLE = "shuffle"

    def load(self):
        path = Configuration.get_path("collection.folder")
        songs = [Song(os.path.dirname(file)) for file in glob.iglob(os.path.join(path, '**', '%s*.h5' % Song.PREFIX_VOCALS), recursive=True)]

        validation = Configuration.get("collection.validation")
        if validation.mode == DataLoader.VALIDATION_MODE_SHUFFLE:
            self.validation_songs = random.sample(
                songs, int(validation.ratio * len(songs)))
            self.training_songs = set(songs) - set(self.validation_songs)

        console.debug("Found %d songs for traing and %d songs for validation."
                      % (len(self.training_songs), len(self.validation_songs)))

        return self.training_songs, self.validation_songs
