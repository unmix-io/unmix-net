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
    VALIDATION_MODE_FIRST = "first"
    VALIDATION_MODE_LAST = "last"

    @staticmethod
    def load():
        path = Configuration.get_path("collection.folder")
        files = [os.path.dirname(file) for file in glob.iglob(os.path.join(path, '**', '%s*.h5' % Song.PREFIX_VOCALS), recursive=True)]


        validation = Configuration.get("collection.validation")
        validation_count = int(validation.ratio * len(files))

        validation_files = []
        if validation.mode == DataLoader.VALIDATION_MODE_SHUFFLE:
            validation_files = random.sample(files, validation_count)
        if validation.mode == DataLoader.VALIDATION_MODE_FIRST:
            files.sort(key=lambda x: x, reverse=False)
            validation_files = files[:validation_count]
        if validation.mode == DataLoader.VALIDATION_MODE_LAST:
            files.sort(key=lambda x: x, reverse=False)
            validation_files = files[-validation_count:]
        training_files = list(set(files) - set(validation_files))

        console.debug("Found %d songs for traing and %d songs for validation."
                      % (len(training_files), len(validation_files)))

        return training_files, validation_files
