#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import hashlib
import glob
import os
import random

from unmix.source.configuration import Configuration
from unmix.source.data.song import Song
from unmix.source.logging.logger import Logger


class DataLoader(object):

    TEST_DATA_COUNT = 50

    @staticmethod
    def load():
        path = Configuration.get_path('collection.folder', False)
        files = [os.path.dirname(file) for file in glob.iglob(
            os.path.join(path, '**', '%s*.h5' % Song.PREFIX_VOCALS), recursive=True)]

        # Sort files by hash value of folder to guarantee a consistent order
        files.sort(key=lambda x: hashlib.md5(
            os.path.basename(x).encode('utf-8', 'surrogatepass')).hexdigest())

        test_files = None
        test_frequency = Configuration.get('collection.test_frequency')
        if test_frequency and test_frequency > 0:
            test_files = files[-DataLoader.TEST_DATA_COUNT:]
            files = files[:len(files) - DataLoader.TEST_DATA_COUNT]

        song_limit = Configuration.get('collection.song_limit')

        if song_limit and song_limit > 0:
            if song_limit <= 1:  # Configuration as percentage share
                song_limit = song_limit * len(files)
            song_limit = min(song_limit, len(files))
            files = files[:song_limit]

        validation_ratio = Configuration.get('collection.validation_ratio')
        validation_files = files[:int(len(files) * validation_ratio)]
        training_files = files[len(validation_files):]

        Logger.debug("Found %d songs for traing and %d songs for validation."
                     % (len(training_files), len(validation_files)))
        if test_files:
            Logger.debug("Use %d songs for tests after every %d epoch."
                         % (len(test_files), test_frequency))

        if len(training_files) == 0:
            Logger.error("No training files assigned.")
        if len(validation_files) == 0:
            Logger.warn("No validation files assigned.")
        return training_files, validation_files, test_files
